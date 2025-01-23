import torch
import torch.nn as nn

from einops import rearrange
from timm import create_model
from typing import Dict, Tuple

from skp.configs.base import Config
from skp.models.modules import FeatureReduction
from skp.models.pooling import get_pool_layer
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


class Attention(nn.Module):
    """
    Given a batch containing bags of features (B, N, D),
    generate attention scores over the features in a bag, N,
    and perform an attention-weighted element-wise mean of the features (B, D)
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0, version: str = "v1"):
        super().__init__()
        version = version.lower()
        if version == "v1":
            self.mlp = nn.Sequential(
                nn.Tanh(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim)
            )
        elif version == "v2":
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        a = self.mlp(x)
        a = a.softmax(dim=1)
        x = (x * a).sum(dim=1)
        return x, a


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

        if self.cfg.enable_gradient_checkpointing:
            print("Enabling gradient checkpointing ...")
            self.backbone.set_grad_checkpointing()

        if isinstance(self.cfg.reduce_feature_dim, int):
            self.backbone = nn.Sequential(
                self.backbone,
                FeatureReduction(self.feature_dim, self.cfg.reduce_feature_dim),
            )
            self.feature_dim = self.cfg.reduce_feature_dim

        self.attn = Attention(
            self.feature_dim,
            dropout=self.cfg.attn_dropout or 0.0,
            version=self.cfg.attn_version or "v1",
        )
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
        self,
        batch: Dict,
        return_loss: bool = False,
        return_features: bool = False,
        return_attn_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        y = batch.get("y", None)

        if return_loss:
            assert y is not None

        B, N = x.shape[:2]
        x = rearrange(x, "b n c h w -> (b n) c h w")
        features = self.extract_features(x, normalize=True)
        features = rearrange(features, "(b n) d -> b n d", b=B, n=N)
        features, attn_scores = self.attn(features)

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
        if return_attn_scores:
            out["attn_scores"] = attn_scores
        if return_loss:
            loss = self.criterion(out, batch)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss

        return out

    def extract_features(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.normalize(x) if normalize else x
        return self.pooling(self.backbone(x))

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
        if self.cfg.ignore_pretrained_linear:
            weights = {k: v for k, v in weights.items() if not k.startswith("linear.")}
        missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
        if len(missing_keys) > 0:
            print(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected keys: {unexpected_keys}")

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
