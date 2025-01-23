"""
Simple model for 2D classification (or regression)
Incorporates embedding of non-image features
Uses timm for backbones
"""

import re
import torch
import torch.nn as nn

from timm import create_model
from typing import Dict

from skp.configs.base import Config
from skp.models.modules import FeatureReduction
from skp.models.pooling import get_pool_layer


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        backbone_args = {
            "pretrained": self.cfg.pretrained,
            "num_classes": 0,
            "global_pool": "",
            "features_only": self.cfg.features_only,
            "in_chans": self.cfg.num_input_channels,
        }
        if self.cfg.backbone_img_size:
            # some models require specifying image size (e.g., coatnet)
            if "efficientvit" in self.cfg.backbone:
                backbone_args["img_size"] = self.cfg.image_height
            else:
                backbone_args["img_size"] = (
                    self.cfg.image_height,
                    self.cfg.image_width,
                )
        self.backbone = create_model(self.cfg.backbone, **backbone_args)
        # get feature dim by passing sample through net
        self.feature_dim = self.backbone(
            torch.randn(
                (
                    2,
                    self.cfg.num_input_channels,
                    self.cfg.image_height,
                    self.cfg.image_width,
                )
            )
        ).size(
            -1 if "xcit" in self.cfg.backbone else 1
        )  # xcit models are channels-last

        self.feature_dim = self.feature_dim * (2 if self.cfg.pool == "catavgmax" else 1)
        self.pooling = get_pool_layer(self.cfg, dim=2)

        if isinstance(self.cfg.reduce_feature_dim, int):
            self.backbone = nn.Sequential(
                self.backbone,
                FeatureReduction(self.feature_dim, self.cfg.reduce_feature_dim),
            )
            self.feature_dim = self.cfg.reduce_feature_dim

        self.embed = nn.Embedding(self.cfg.embed_num_classes, self.cfg.embed_dim)
        # allows for interaction between elements of image feature vector and embedding
        self.mlp = nn.Linear(self.feature_dim + self.cfg.embed_dim, self.feature_dim)
        self.dropout = nn.Dropout(p=self.cfg.dropout)
        self.linear = nn.Linear(self.feature_dim, self.cfg.num_classes)

        if self.cfg.load_pretrained_backbone:
            print(
                f"Loading pretrained backbone from {self.cfg.load_pretrained_backbone} ..."
            )
            weights = torch.load(
                self.cfg.load_pretrained_backbone,
                map_location=lambda storage, loc: storage,
            )["state_dict"]
            # Replace model prefix as this does not exist in Net
            weights = {re.sub(r"^model.", "", k): v for k, v in weights.items()}
            # Get backbone only
            weights = {
                re.sub(r"^backbone.", "", k): v
                for k, v in weights.items()
                if "backbone" in k
            }
            self.backbone.load_state_dict(weights)

        self.criterion = None

        self.backbone_frozen = False
        if self.cfg.freeze_backbone:
            self.freeze_backbone()

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

    def forward(
        self, batch: Dict, return_loss: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        y = batch.get("y", None)
        var = batch["var"]

        if return_loss:
            assert y is not None

        features = self.extract_features(x, var, normalize=True)

        if self.cfg.multisample_dropout:
            logits = torch.stack(
                [self.linear(self.dropout(features)) for _ in range(5)]
            ).mean(0)
        else:
            logits = self.linear(self.dropout(features))

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

    def extract_features(
        self, x: torch.Tensor, var: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        x = self.normalize(x) if normalize else x
        var = self.embed(var)
        feat = self.pooling(self.backbone(x))
        feat = torch.cat([feat, var], dim=1)
        feat = self.mlp(feat)
        return feat

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
