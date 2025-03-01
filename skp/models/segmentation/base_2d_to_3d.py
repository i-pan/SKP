"""
Base class for 2D->3D segmentation models.
Uses a 2D encoder which processes 2D slices of an image independently.
Extracted feature maps are then fed into a 3D decoder.
"""

import torch
import torch.nn as nn

from einops import rearrange
from timm import create_model
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Union

from skp.configs import Config
from skp.models.segmentation import decoders
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.drop = nn.Dropout3d(p=dropout)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        if isinstance(size, (tuple, list)):
            self.up = nn.Upsample(size=size, mode="trilinear")
        else:
            self.up = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.conv(self.drop(x)))


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        encoder_args = {
            "model_name": self.cfg.backbone,
            "pretrained": self.cfg.pretrained,
            "features_only": True,
            "in_chans": self.cfg.num_input_channels,
        }
        if self.cfg.backbone_img_size:
            encoder_args["img_size"] = self.cfg.image_height, self.cfg.image_width
        self.encoder = create_model(**encoder_args)

        if self.cfg.enable_gradient_checkpointing and not self.cfg.freeze_encoder:
            print("Enabling gradient checkpointing ...")
            self.encoder.set_grad_checkpointing()

        with torch.no_grad():
            out = self.encoder(
                torch.randn(
                    (
                        2,
                        self.cfg.num_input_channels,
                        self.cfg.image_height,
                        self.cfg.image_width,
                    )
                )
            )
            self.cfg.encoder_channels = [o.shape[1] for o in out]
            del out

        self.decoder = getattr(decoders, self.cfg.decoder_type)(self.cfg)
        decoder_out_channels = (
            self.cfg.decoder_out_channels[-1]
            if isinstance(self.cfg.decoder_out_channels, list)
            else self.cfg.decoder_out_channels
        )

        output_size = (
            (self.cfg.num_slices, self.cfg.image_height, self.cfg.image_width)
            if self.cfg.output_size is None
            else self.cfg.output_size
        )
        self.segmentation_head = SegmentationHead(
            decoder_out_channels,
            self.cfg.num_classes,
            size=output_size,
            dropout=self.cfg.seg_dropout or 0,
        )

        self.criterion = None

        if self.cfg.deep_supervision:
            self.aux_segmentation_heads = nn.ModuleList()
            for idx in range(self.cfg.deep_supervision_num_levels or 1):
                aux_decoder_out_channels = (
                    self.cfg.decoder_out_channels[-(idx + 2)]
                    if isinstance(self.cfg.decoder_out_channels, list)
                    else self.cfg.decoder_out_channels
                )
                self.aux_segmentation_heads.append(
                    SegmentationHead(
                        aux_decoder_out_channels, self.cfg.num_classes, size=None
                    )
                )

        if self.cfg.enable_gradient_checkpointing:
            # encountered error with checkpointing when using timm ResNet encoder
            # due to inplace ReLU
            for module in self.encoder.modules():
                if hasattr(module, "inplace"):
                    module.inplace = False

        if self.cfg.load_pretrained_encoder:
            self.load_pretrained_encoder()

        if self.cfg.load_pretrained_decoder:
            self.load_pretrained_decoder()

        if self.cfg.load_pretrained_model:
            self.load_pretrained_model()

        if self.cfg.freeze_encoder:
            print("Freezing encoder ...")
            self.freeze_encoder()

        if self.cfg.freeze_decoder:
            print("Freezing decoder ...")
            self.freeze_decoder()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
        return_decoder_output: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        x = batch["x"]
        y = batch["y"] if "y" in batch else None

        if return_loss:
            assert y is not None

        # if not self.training and self.cfg.remove_batch_dim_and_reshape_during_val:
        #     assert x.size(0) == 1
        #     x = x[0]
        #     if y is not None:
        #         assert y.size(0) == 1
        #         y = y[0]
        #     x = rearrange(x, "(b n) c h w -> b n c h w", n=self.cfg.num_slices)
        #     if y is not None:
        #         if y.ndim == 3:
        #             y = rearrange(y, "(b n) h w -> b n h w", n=self.cfg.num_slices)
        #         elif y.ndim == 4:
        #             y = rearrange(y, "(b n) c h w -> b n c h w", n=self.cfg.num_slices)
        #     batch["y"] = y
        #     batch["training"] = self.training
            
        b, n = x.shape[:2]
        x = rearrange(x, "b n c h w -> (b n) c h w")
        x = self.normalize(x)
        feature_maps = self.encoder(x)
        feature_maps = [
            rearrange(fm, "(b n) c h w -> b c n h w", b=b, n=n) for fm in feature_maps
        ]
        if self.cfg.enable_gradient_checkpointing and self.training:
            decoder_output = checkpoint(self.decoder, feature_maps, use_reentrant=False)
        else:
            decoder_output = self.decoder(feature_maps)
        logits = self.segmentation_head(decoder_output[-1])

        out = {"logits": logits}

        if return_features:
            out["features"] = feature_maps

        if return_decoder_output:
            out["decoder_output"] = decoder_output

        if self.cfg.deep_supervision:
            aux_logits_list = []
            for idx, each_head in enumerate(self.aux_segmentation_heads):
                aux_logits_list.append(each_head(decoder_output[-(idx + 2)]))
            out["aux_logits"] = aux_logits_list

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def normalize(self, x):
        if self.cfg.normalization == "-1_1":
            mini, maxi = (
                self.cfg.normalization_params["min"],
                self.cfg.normalization_params["max"],
            )
            x = x - mini
            x = x / (maxi - mini)
            x = x - 0.5
            x = x * 2.0
        elif self.cfg.normalization == "0_1":
            mini, maxi = (
                self.cfg.normalization_params["min"],
                self.cfg.normalization_params["max"],
            )
            x = x - mini
            x = x / (maxi - mini)
        elif self.cfg.normalization == "mean_sd":
            mean, sd = (
                self.cfg.normalization_params["mean"],
                self.cfg.normalization_params["sd"],
            )
            x = (x - mean) / sd
        elif self.cfg.normalization == "per_channel_mean_sd":
            mean, sd = (
                self.cfg.normalization_params["mean"],
                self.cfg.normalization_params["sd"],
            )
            assert len(mean) == len(sd) == x.size(1)
            mean, sd = torch.tensor(mean).unsqueeze(0), torch.tensor(sd).unsqueeze(0)
            for i in range(x.ndim - 2):
                mean, sd = mean.unsqueeze(-1), sd.unsqueeze(-1)
            x = (x - mean) / sd
        return x

    def load_pretrained_encoder(self) -> None:
        print(f"Loading pretrained encoder from {self.cfg.load_pretrained_encoder} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_encoder)
        weights = filter_weights_by_prefix(weights, "model.")
        # Get backbone only
        filtered_weights = filter_weights_by_prefix(weights, "backbone.")
        if len(filtered_weights) == 0:
            # If no backbone weights, check to see if encoder weights exist
            filtered_weights = filter_weights_by_prefix(weights, "encoder.")
        if len(filtered_weights) == 0:
            filtered_weights = filter_weights_by_prefix(weights, "segmenter.encoder.")
        # seems weight names are not necessarily the same if features_only=True ...
        # for maxvit
        if self.cfg.backbone.startswith("maxvit_"):
            filtered_weights = {
                k.replace("stages.", "stages_"): v
                for k, v in filtered_weights.items()
                if not k.startswith("head.")
            }
        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            filtered_weights, strict=False
        )
        if len(missing_keys) > 0:
            raise Exception(
                f"Error in loading state_dict, missing keys: {missing_keys}"
            )

    def load_pretrained_decoder(self) -> None:
        print(f"Loading pretrained decoder from {self.cfg.load_pretrained_decoder} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_decoder)
        weights = filter_weights_by_prefix(weights, "model.")
        weights = filter_weights_by_prefix(weights, "decoder.")
        self.decoder.load_state_dict(weights, strict=True)

    def load_pretrained_segmentation_heads(
        self, weights: Dict[str, torch.Tensor]
    ) -> None:
        segmentation_head_weights = filter_weights_by_prefix(
            weights, "model.segmentation_head."
        )
        unequal_classes = (
            segmentation_head_weights["conv.weight"].shape[0] != self.cfg.num_classes
        )
        if self.cfg.load_segmentation_head_classes is not None:
            class_indices = self.cfg.load_segmentation_head_classes
            if isinstance(class_indices, int):
                class_indices = [class_indices]
            # load head weights only for specified classes
            # should just be conv.weight and conv.bias
            print(f"Loading segmentation head weights for classes: {class_indices} ...")
            segmentation_head_weights = {
                k: v[class_indices] for k, v in segmentation_head_weights.items()
            }
        elif unequal_classes:
            print(
                "Pretrained segmentation head weights contain",
                segmentation_head_weights["conv.weight"].shape[0],
                "classes, but the current model has",
                f"{self.cfg.num_classes} classes ...",
            )
            print(f"Reshaping head weights to {self.cfg.num_classes} classes ...")
            for k, v in segmentation_head_weights.items():
                v = torch.stack([v.mean(0)] * self.cfg.num_classes)
                segmentation_head_weights[k] = v
        self.segmentation_head.load_state_dict(segmentation_head_weights)

        if self.cfg.deep_supervision:
            # load auxiliary heads for deep supervision, if available
            aux_segmentation_heads_weights = filter_weights_by_prefix(
                weights, "model.aux_segmentation_heads."
            )
            if len(aux_segmentation_heads_weights) > 0:
                if self.cfg.load_segmentation_head_classes:
                    # load head weights only for specified classes
                    aux_segmentation_heads_weights = {
                        k: v[class_indices]
                        for k, v in aux_segmentation_heads_weights.items()
                    }
                elif unequal_classes:
                    for k, v in aux_segmentation_heads_weights.items():
                        v = torch.stack([v.mean(0)] * self.cfg.num_classes)
                        aux_segmentation_heads_weights[k] = v
                self.aux_segmentation_heads.load_state_dict(
                    aux_segmentation_heads_weights
                )

    def load_pretrained_model(self) -> None:
        print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_model)
        segmenter_keys = [k for k in [*weights] if "segmenter" in k]
        if len(segmenter_keys) > 0:
            # if loading segmenter part of seg_cls model
            for k in segmenter_keys:
                weights[k.replace("segmenter.", "")] = weights.pop(k)
        encoder_weights = filter_weights_by_prefix(weights, "model.encoder.")
        decoder_weights = filter_weights_by_prefix(weights, "model.decoder.")
        self.encoder.load_state_dict(encoder_weights)
        self.decoder.load_state_dict(decoder_weights)
        self.load_pretrained_segmentation_heads(weights)

    def freeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss


if __name__ == "__main__":
    import torch
    from einops import rearrange
    from skp.configs import Config
    from timm import create_model

    cfg = Config()
    cfg.backbone = "tf_efficientnetv2_b0"
    cfg.decoder_type = "DeepLabV3PlusDecoder3d"
    cfg.pretrained = True
    cfg.num_input_channels = 3
    cfg.num_slices = 32
    cfg.image_height = 256
    cfg.image_width = 256
    cfg.decoder_out_channels = 256
    cfg.atrous_rates = (2, 4, 6, 12)
    cfg.aspp_separable = False
    cfg.aspp_dropout = 0.1
    cfg.num_classes = 100

    x = torch.randn(
        (2, cfg.num_slices, cfg.num_input_channels, cfg.image_height, cfg.image_width)
    )
    model = Net(cfg)
    out = model({"x": x})
    print(out["logits"].shape)
