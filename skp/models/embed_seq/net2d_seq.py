import torch
import torch.nn as nn

from einops import rearrange
from timm import create_model
from typing import Dict

from skp.configs.base import Config
from skp.models.embed_seq import heads
from skp.models.modules import FeatureReduction
from skp.models.pooling import get_pool_layer
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


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
        if self.cfg.enable_gradient_checkpointing:
            print("Enabling gradient checkpointing ...")
            self.backbone.set_grad_checkpointing()

        self.feature_dim = self.feature_dim * (2 if self.cfg.pool == "catavgmax" else 1)
        self.pooling = get_pool_layer(self.cfg, dim=2)

        if self.cfg.reduce_feature_dim is not None:
            self.feature_reduction = FeatureReduction(
                self.feature_dim,
                self.cfg.reduce_feature_dim,
                dim=1,
                reduce_grouped_conv=self.cfg.reduce_grouped_conv or False,
                add_norm=self.cfg.add_norm or True,
                add_act=self.cfg.add_act or True,
            )
            self.feature_dim = self.cfg.reduce_feature_dim

        self.head = getattr(heads, self.cfg.head_type)(self.cfg, self.feature_dim)

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
        
        b, n, c, h, w = x.shape
        x = rearrange(x, "b n c h w -> (b n) c h w")
        features = self.extract_features(x, normalize=True)
        if hasattr(self, "feature_reduction"):
            features = rearrange(features, "(b n) d -> b d n", b=b, n=n)
            features = self.feature_reduction(features)
            features = rearrange(features, "b d n -> b n d")
        else:
            features = rearrange(features, "(b n) d -> b n d", b=b, n=n)
        out = self.head(features)

        if self.cfg.model_activation_fn == "sigmoid":
            out = {k: v.sigmoid() for k, v in out.items()}
        elif self.cfg.model_activation_fn == "softmax":
            out = {k: v.softmax(dim=-1) for k, v in out.items()}

        if return_features:
            out["features"] = features
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
    cfg.add_cls_token = True
    cfg.transformer_nhead = 8
    cfg.transformer_num_layers = 1
    cfg.transformer_dropout = 0.1
    cfg.transformer_activation = "gelu"
    cfg.transformer_dim_feedforward = 256
    cfg.transformer_norm_first = False
    cfg.dropout = 0.1
    cfg.seq_num_classes = 10
    cfg.cls_num_classes = 20
    cfg.head_type = "TransformerHead"

    # head = TransformerHead(cfg, 256)
    # x = torch.randn((2, 32, 256))
    # out = head(x)
    # print(out["logits"].shape, out["logits_cls"].shape)

    cfg.backbone = "tf_efficientnetv2_m"
    cfg.pretrained = False 
    cfg.features_only = False 
    cfg.num_input_channels = 3
    cfg.backbone_img_size = False 
    cfg.image_height = 256 
    cfg.image_width = 256
    cfg.pool = "avg"
    # cfg.lstm_hidden_size = 128
    # cfg.lstm_num_layers = 2
    # cfg.lstm_dropout = 0.1
    # cfg.seq_num_classes = 10
    # cfg.cls_num_classes = 20
    # cfg.head_type = "BiLSTMHead"
    # cfg.add_attention_cls = True
    # cfg.attention_type = "basic"
    # cfg.attention_dropout = 0.1
    # cfg.attention_version = "v1"
    x = torch.randn((2, 8, 3, 256, 256))
    model = Net(cfg)
    out = model({"x":x})
    print(out["logits"].shape, out["logits_cls"].shape)