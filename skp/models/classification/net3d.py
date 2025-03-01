"""
Simple model for 2D classification (or regression)
Uses timm for backbones
"""

import torch
import torch.nn as nn

# from timm import create_model
from typing import Dict

from skp.configs.base import Config
from skp.models.encoders3d import get_encoder
from skp.models.pooling import get_pool_layer
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = get_encoder(self.cfg)
        self.change_num_input_channels()
        # get feature dim by passing sample through net
        self.feature_dim = self.get_feature_dim()

        if self.cfg.enable_gradient_checkpointing:
            print("Enabling gradient checkpointing ...")

        self.feature_dim = self.feature_dim * (2 if self.cfg.pool == "catavgmax" else 1)
        self.pooling = get_pool_layer(self.cfg, dim=3)

        # main purpose of this was to reduce feature dimensions if
        # passing features as input to a sequence model, thus
        # reducing the # of parameters in the sequence model
        #
        # however, it's probably better to put the feature reduction
        # module in the sequence model where it can be trained
        # with the rest of the sequence model
        #
        # see skp/models/embed_seq/net2d_seq.py
        #
        # if isinstance(self.cfg.reduce_feature_dim, int):
        #     self.backbone = nn.Sequential(
        #         self.backbone,
        #         FeatureReduction(self.feature_dim, self.cfg.reduce_feature_dim),
        #     )
        #     self.feature_dim = self.cfg.reduce_feature_dim

        self.dropout = nn.Dropout(p=self.cfg.dropout)
        self.linear = nn.Linear(self.feature_dim, self.cfg.num_classes)

        if self.cfg.load_pretrained_backbone:
            self.load_pretrained_backbone()

        if self.cfg.load_pretrained_model:
            self.load_pretrained_model()

        self.criterion = None

        self.backbone_frozen = False
        if self.cfg.freeze_backbone:
            self.freeze_backbone()

    def forward(
        self, batch: Dict, return_loss: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        y = batch.get("y", None)

        if return_loss:
            assert y is not None

        features = self.extract_features(x, normalize=True)
        if self.cfg.multisample_dropout:
            logits = torch.stack(
                [self.linear(self.dropout(features)) for _ in range(5)]
            ).mean(0)
        else:
            logits = self.linear(self.dropout(features))

        if self.cfg.model_activation_fn == "sigmoid":
            logits = logits.sigmoid()
        elif self.cfg.model_activation_fn == "softmax":
            logits = logits.softmax(dim=1)

        out = {"logits": logits}
        if return_features:
            out["features"] = features
        if return_loss:
            loss = self.criterion(out, batch)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss

        return out

    @torch.no_grad()
    def get_feature_dim(self):
        rand_input = torch.randn(
            (
                2,
                self.cfg.num_input_channels,
                self.cfg.num_slices,
                self.cfg.image_height,
                self.cfg.image_width,
            )
        )
        # TODO: fix this so it's not so hacky
        if "efficientnet" in self.cfg.backbone or "convnext" in self.cfg.backbone:
            feature_dim = self.backbone(rand_input).shape[1]
        else:
            feature_dim = self.backbone(rand_input)[-1].shape[1]
        del rand_input
        return feature_dim

    def change_num_input_channels(self):
        # Assumes original number of input channels in model is 3
        for i, m in enumerate(self.backbone.modules()):
            if isinstance(m, nn.Conv3d) and m.in_channels == 3:
                m.in_channels = self.cfg.num_input_channels
                # First, sum across channels
                W = m.weight.sum(1, keepdim=True)
                # Then, divide by number of channels
                W = W / self.cfg.num_input_channels
                # Then, repeat by number of channels
                size = [1] * W.ndim
                size[1] = self.cfg.num_input_channels
                W = W.repeat(size)
                m.weight = nn.Parameter(W)
                break

    def extract_features(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.normalize(x) if normalize else x
        # use last feature map
        # TODO: fix this so it's not so hacky
        if "efficientnet" in self.cfg.backbone or "convnext" in self.cfg.backbone:
            features = self.pooling(self.backbone(x))
        else:
            features = self.pooling(self.backbone(x)[-1])
        return features

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
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
        elif self.cfg.normalization == "none":
            x = x
        return x

    def load_pretrained_backbone(self) -> None:
        print(
            f"Loading pretrained backbone from {self.cfg.load_pretrained_backbone} ..."
        )
        weights = torch_load_weights(self.cfg.load_pretrained_backbone)
        backbone_weights = filter_weights_by_prefix(weights, "model.backbone.")
        # sometimes loading encoder from trained segmentation model as backbone
        if len(backbone_weights) == 0:
            backbone_weights = filter_weights_by_prefix(weights, "model.encoder.")
        if len(backbone_weights) == 0:
            # seg_cls model
            backbone_weights = filter_weights_by_prefix(
                weights, "model.segmenter.encoder."
            )
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_weights, strict=False
        )
        if len(missing_keys) > 0:
            print(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected keys: {unexpected_keys}")

    def load_pretrained_model(self) -> None:
        print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_model)
        weights = filter_weights_by_prefix(weights, "model.")
        self.load_state_dict(weights)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss


if __name__ == "__main__":
    from skp.configs import Config

    cfg = Config()
    cfg.backbone = "csn_r26"
    cfg.dim0_strides = [2, 2, 2, 1, 1]
    cfg.num_input_channels = 1
    cfg.num_slices = 40
    cfg.image_height = 256
    cfg.image_width = 256
    cfg.pool = "avg"
    cfg.num_classes = 10
    cfg.dropout = 0.1
    cfg.enable_gradient_checkpointing = True

    x = torch.randn(
        (2, cfg.num_input_channels, cfg.num_slices, cfg.image_height, cfg.image_width)
    )

    model = Net(cfg)

    with torch.no_grad():
        out = model({"x": x})

    print(out["logits"].shape)
