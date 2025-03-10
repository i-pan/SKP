import re
import torch
import torch.nn as nn

from einops import rearrange
from transformers import AutoModel
from typing import Dict, List, Union

from skp.configs import Config
from skp.models.segmentation.decoders.upernet import UperNetDecoder


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
        self.encoder = AutoModel.from_pretrained("microsoft/rad-dino-maira-2")

        mean, sd = [0.5307] * 3, [0.2583] * 3
        mean, sd = torch.tensor(mean), torch.tensor(sd)
        self.mean = rearrange(mean, "b -> 1 b 1 1")
        self.sd = rearrange(sd, "b -> 1 b 1 1")

        self.cfg.encoder_channels = [self.cfg.num_input_channels] + [768] * 4
        self.hidden_state_indices = [2, 4, 10, 12]

        self.decoder = UperNetDecoder(self.cfg)
        self.segmentation_head = SegmentationHead(
            self.cfg.hidden_size,
            self.cfg.num_classes,
            size=(self.cfg.image_height, self.cfg.image_width),
            dropout=self.cfg.seg_dropout or 0,
        )

        self.criterion = None

        if self.cfg.deep_supervision:
            self.aux_segmentation_heads = nn.ModuleList(
                [
                    SegmentationHead(
                        self.cfg.hidden_size,
                        self.cfg.num_classes,
                        size=None,
                    )
                    for idx in range(self.cfg.deep_supervision_num_levels or 2)
                ]
            )

        if self.cfg.load_pretrained_encoder:
            print(
                f"Loading pretrained encoder from {self.cfg.load_pretrained_encoder} ..."
            )
            weights = torch.load(
                self.cfg.load_pretrained_encoder,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )["state_dict"]
            weights = {re.sub(r"^model\.", "", k): v for k, v in weights.items()}
            # Get backbone only
            weights = {
                re.sub(r"^backbone\.", "", k): v
                for k, v in weights.items()
                if k.startswith("backbone.")
            }
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

        if self.cfg.freeze_encoder:
            print("Freezing encoder ...")
            self.freeze_encoder()

        if self.cfg.load_pretrained_model:
            print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
            weights = torch.load(
                self.cfg.load_pretrained_model,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )["state_dict"]
            encoder_weights = {
                re.sub(r"^model\.encoder\.", "", k): v
                for k, v in weights.items()
                if k.startswith("model.encoder.")
            }
            decoder_weights = {
                re.sub(r"^model\.decoder\.", "", k): v
                for k, v in weights.items()
                if k.startswith("model.decoder.")
            }
            if self.cfg.load_pretrained_segmentation_heads:
                segmentation_head_weights = {
                    re.sub(r"^model\.segmentation.head\.", "", k): v
                    for k, v in weights.items()
                    if k.startswith("model.segmentation_head")
                }
                self.segmentation_head.load_state_dict(segmentation_head_weights)
                if self.cfg.deep_supervision:
                    aux_segmentation_heads_weights = {
                        re.sub(r"^model\.aux_segmentation_heads\.", "", k): v
                        for k, v in weights.items()
                        if k.startswith("model.aux_segmentation_heads")
                    }
                    if len(aux_segmentation_heads_weights) > 0:
                        self.aux_segmentation_heads.load_state_dict(
                            aux_segmentation_heads_weights
                        )
            self.encoder.load_state_dict(encoder_weights)
            self.decoder.load_state_dict(decoder_weights)

    def normalize(self, x):
        dev1, dev2 = x.device, self.mean.device
        if dev1 != dev2:
            self.mean = self.mean.to(dev1)
            self.sd = self.sd.to(dev1)

        x = x / 255.0
        x = (x - self.mean) / self.sd
        return x

    def reshape_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        image_size = self.cfg.image_height
        patch_size = self.encoder.config.patch_size
        embeddings_size = image_size // patch_size
        return rearrange(x, "b (h w) c -> b c h w", h=embeddings_size)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        x = batch["x"]
        y = batch["y"] if "y" in batch else None

        if return_loss:
            assert y is not None

        x = self.normalize(x)
        feature_maps = self.encoder(pixel_values=x, output_hidden_states=True)
        feature_maps = [
            feature_maps.hidden_states[i] for i in self.hidden_state_indices
        ]
        # remove class token and reshape
        feature_maps = [self.reshape_patch_embeddings(fm[:, 1:]) for fm in feature_maps]
        decoder_output = self.decoder(feature_maps)
        logits = self.segmentation_head(decoder_output[-1])

        out = {"logits": logits}

        if return_features:
            out["features"] = feature_maps

        if self.cfg.deep_supervision and self.training:
            aux_logits_list = []
            for idx, each_head in enumerate(self.aux_segmentation_heads):
                aux_logits_list.append(each_head(decoder_output[-(idx + 2)]))
            out["aux_logits"] = aux_logits_list

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss


if __name__ == "__main__":
    import numpy as np
    import torch

    from skp.configs import Config

    cfg = Config()
    cfg.image_height = 518
    cfg.image_width = 518
    cfg.pool_scales = (1, 2, 3, 6)
    cfg.hidden_size = 256
    cfg.num_classes = 5

    upernet = Net(cfg).eval()
    x = torch.from_numpy(
        np.random.randint(0, 255, (1, 3, cfg.image_height, cfg.image_width))
    )
    out = upernet({"x": x})
    print(out["logits"].shape)
