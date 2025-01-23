import torch
import torch.nn as nn

from timm import create_model
from typing import Dict, List, Union

from skp.configs import Config
from skp.models.segmentation.decoders.unet import DeepLabV3PlusDecoder
from skp.models.utils import _torch_load_weights, _filter_weights_by_prefix


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
        self.drop = nn.Dropout2d(p=dropout)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        if isinstance(size, (tuple, list)):
            self.up = nn.Upsample(size=size, mode="bilinear")
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

        with torch.no_grad():
            out = self.encoder(
                torch.randn(
                    (
                        1,
                        self.cfg.num_input_channels,
                        self.cfg.image_height,
                        self.cfg.image_width,
                    )
                )
            )
            self.cfg.encoder_channels = [self.cfg.num_input_channels] + [
                o.shape[1] for o in out
            ]
            del out

        self.decoder = DeepLabV3PlusDecoder(self.cfg)
        self.segmentation_head = SegmentationHead(
            self.cfg.decoder_channels[-1],
            self.cfg.num_classes,
            size=(self.cfg.image_height, self.cfg.image_width),
            dropout=self.cfg.seg_dropout or 0,
        )

        self.criterion = None

        if self.cfg.deep_supervision:
            self.aux_segmentation_heads = nn.ModuleList(
                [
                    SegmentationHead(
                        self.cfg.decoder_out_channels,
                        self.cfg.num_classes,
                        size=None,
                    )
                    for idx in range(self.cfg.deep_supervision_num_levels or 1)
                ]
            )

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

        x = self.normalize(x)
        feature_maps = self.encoder(x)
        decoder_output = self.decoder(feature_maps)  # top level features, aspp features
        logits = self.segmentation_head(decoder_output[0])

        out = {"logits": logits}

        if return_features:
            out["features"] = feature_maps

        if return_decoder_output:
            out["decoder_output"] = decoder_output

        if self.cfg.deep_supervision:
            aux_logits_list = []
            for idx, each_head in enumerate(self.aux_segmentation_heads):
                aux_logits_list.append(each_head(decoder_output[1]))
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
        weights = _torch_load_weights(self.cfg.load_pretrained_encoder)
        weights = _filter_weights_by_prefix(weights, "model.")
        # Get backbone only
        weights = _filter_weights_by_prefix(weights, "backbone.")
        if len(weights) == 0:
            # If no backbone weights, check to see if encoder weights exist
            weights = _filter_weights_by_prefix(weights, "encoder.")
        # seems weight names are not necessarily the same if features_only=True ...
        # for maxvit
        if self.cfg.backbone.startswith("maxvit_"):
            weights = {
                k.replace("stages.", "stages_"): v
                for k, v in weights.items()
                if not k.startswith("head.")
            }
        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            weights, strict=False
        )
        if len(missing_keys) > 0:
            raise Exception(
                f"Error in loading state_dict, missing keys: {missing_keys}"
            )

    def load_pretrained_decoder(self) -> None:
        print(f"Loading pretrained decoder from {self.cfg.load_pretrained_decoder} ...")
        weights = _torch_load_weights(self.cfg.load_pretrained_decoder)
        weights = _filter_weights_by_prefix(weights, "model.")
        weights = _filter_weights_by_prefix(weights, "decoder.")
        self.decoder.load_state_dict(weights, strict=True)

    def load_pretrained_model(self) -> None:
        print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
        weights = _torch_load_weights(self.cfg.load_pretrained_model)
        encoder_weights = _filter_weights_by_prefix(weights, "model.encoder.")
        decoder_weights = _filter_weights_by_prefix(weights, "model.decoder.")
        self.encoder.load_state_dict(encoder_weights)
        self.decoder.load_state_dict(decoder_weights)
        if self.cfg.load_pretrained_segmentation_heads:
            segmentation_head_weights = _filter_weights_by_prefix(
                weights, "model.segmentation.head."
            )
            self.segmentation_head.load_state_dict(segmentation_head_weights)
            if self.cfg.deep_supervision:
                aux_segmentation_heads_weights = _filter_weights_by_prefix(
                    weights, "model.aux_segmentation_heads."
                )
                if len(aux_segmentation_heads_weights) > 0:
                    self.aux_segmentation_heads.load_state_dict(
                        aux_segmentation_heads_weights
                    )

    def freeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
