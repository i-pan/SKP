"""
This model comprises the following parts:
    1- 2D CNN to extract image-level embeddings
    2- GRU or LSTM for sequence modeling over image embeddings
    3- [Optional] Auxiliary classifier which takes as input individual image embeddings
    4- [Optional] Attention to aggregate final sequence hidden states into a single
                  embedding for series-level classification
"""

import torch
import torch.nn as nn

from einops import rearrange
from timm import create_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict

from skp.configs.base import Config
from skp.models.modules import FeatureReduction
from skp.models.pooling import get_pool_layer
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


class Attention(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.0):
        super().__init__()
        self.att = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.att(x)
        a = a.softmax(dim=1)
        if mask is not None:
            # set attention weights for padding to 0, then re-normalize
            a = a * mask.unsqueeze(2).float()
            a /= a.sum(dim=1, keepdim=True)
        print(x.shape, a.shape)
        x = (x * a).sum(dim=1)
        return x, a


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Create backbone for extracting image-level embeddings
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
        self.feature_dim = self.get_feature_dim()

        if self.cfg.enable_gradient_checkpointing:
            print("Enabling gradient checkpointing ...")
            self.backbone.set_grad_checkpointing()

        self.feature_dim = self.feature_dim * (2 if self.cfg.pool == "catavgmax" else 1)
        self.pooling = get_pool_layer(self.cfg, dim=2)

        # Reduce feature dimensionality, if specified
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

        # Auxiliary for image embeddings (before RNN)
        if self.cfg.add_auxiliary_classifier:
            self.linear_aux = nn.Sequential(
                nn.Dropout(self.cfg.linear_aux_dropout or 0.0),
                nn.Linear(
                    self.feature_dim, self.cfg.seq_num_classes or self.cfg.num_classes
                ),
            )

        # Create RNN
        rnn = self.cfg.rnn_class
        assert rnn in {"GRU", "LSTM"}

        self.rnn = getattr(nn, rnn)(
            input_size=self.feature_dim,
            hidden_size=self.feature_dim // 2,
            num_layers=self.cfg.rnn_num_layers or 1,
            batch_first=True,
            bidirectional=True,
            dropout=self.cfg.rnn_dropout or 0.0,
        )

        # Classifier for RNN hidden states
        self.linear_seq = nn.Sequential(
            nn.Dropout(self.cfg.linear_seq_dropout or 0.0),
            nn.Linear(
                self.feature_dim, self.cfg.seq_num_classes or self.cfg.num_classes
            ),
        )

        if self.cfg.add_sequence_level_classifier:
            self.attention = Attention(
                self.feature_dim, self.cfg.attention_dropout or 0.0
            )
            self.linear_cls = nn.Sequential(
                nn.Dropout(self.cfg.linear_cls_dropout or 0.0),
                nn.Linear(
                    self.feature_dim, self.cfg.cls_num_classes or self.cfg.num_classes
                ),
            )
        if self.cfg.load_pretrained_backbone:
            self.load_pretrained_backbone()

        if self.cfg.load_pretrained_model:
            self.load_pretrained_model()

        self.criterion = None

        self.backbone_frozen = False
        if self.cfg.freeze_backbone:
            self.freeze_backbone()

        if self.cfg.freeze_layers is not None:
            for name, param in self.named_parameters():
                for layer in self.cfg.freeze_layers:
                    if name.startswith(layer):
                        print(f"Freezing {name} ...")
                        param.requires_grad = False

    def forward_chunks(
        self, batch: Dict, return_loss: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        y_seq = batch.get("y_seq", None)

        if return_loss:
            assert y_seq is not None

        b, n, c, h, w = x.shape
        assert b == 1, f"batch size must be 1 for forward_chunks, got {b}"
        assert not self.training, "training mode not supported for forward_chunks"

        x = rearrange(x, "b n c h w -> (b n) c h w")
        features = self.extract_features(x, normalize=True)
        if hasattr(self, "feature_reduction"):
            features = rearrange(features, "(b n) d -> b d n", b=b, n=n)
            features = self.feature_reduction(features)
            features = rearrange(features, "b d n -> b n d")
        else:
            features = rearrange(features, "(b n) d -> b n d", b=b, n=n)

        out = {}
        if return_features:
            out["features"] = features

        num_chunks = n // self.cfg.seq_len + 1
        start_indices = torch.linspace(0, n - self.cfg.seq_len, num_chunks).long()
        chunks = [
            features[:, start_indices[i] : start_indices[i] + self.cfg.seq_len]
            for i in range(num_chunks)
        ]
        assert len(chunks) == num_chunks

        chunks = torch.cat(chunks, dim=0)
        # chunks.shape = (num_chunks, seq_len, d)
        hidden_states, _ = self.rnn(chunks)
        # skip connection
        hidden_states = hidden_states + chunks

        logits = self.linear_seq(hidden_states)
        logits_seq = torch.zeros(
            (1, n, self.cfg.seq_num_classes), dtype=logits.dtype
        ).to(logits.device)
        counts_seq = torch.zeros((1, n, self.cfg.seq_num_classes)).to(logits.device)
        for idx, chunk_logits in enumerate(logits):
            logits_seq[
                0, start_indices[idx] : start_indices[idx] + self.cfg.seq_len
            ] += chunk_logits
            counts_seq[
                0, start_indices[idx] : start_indices[idx] + self.cfg.seq_len
            ] += 1

        logits_seq = logits_seq / counts_seq
        out["logits_seq"] = logits_seq

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def forward_train(
        self, batch: Dict, return_loss: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        y_seq = batch.get("y_seq", None)
        y_cls = batch.get("y_cls", None)
        mask = batch.get("mask", None)

        if return_loss:
            assert y_seq is not None
            if self.cfg.add_sequence_level_classifier:
                assert y_cls is not None

        b, n, c, h, w = x.shape
        x = rearrange(x, "b n c h w -> (b n) c h w")
        features = self.extract_features(x, normalize=True)
        if hasattr(self, "feature_reduction"):
            features = rearrange(features, "(b n) d -> b d n", b=b, n=n)
            features = self.feature_reduction(features)
            features = rearrange(features, "b d n -> b n d")
        else:
            features = rearrange(features, "(b n) d -> b n d", b=b, n=n)

        out = {}
        if hasattr(self, "linear_aux"):
            out["aux_logits_seq"] = self.linear_aux(features)

        skip = features
        convert_to_packed = mask is not None and mask.sum().item() < (
            mask.shape[0] * mask.shape[1]
        )

        if convert_to_packed:
            L = features.shape[1]
            features = self.convert_seq_to_packed_sequence(features, mask)

        hidden_states, _ = self.rnn(features)

        if convert_to_packed:
            hidden_states, _ = pad_packed_sequence(
                hidden_states, batch_first=True, total_length=L
            )

        # skip connection
        hidden_states = hidden_states + skip
        out["logits_seq"] = self.linear_seq(hidden_states)

        if hasattr(self, "linear_cls"):
            agg_feature, _ = self.attention(hidden_states, mask)
            out["logits_cls"] = self.linear_cls(agg_feature)

        if return_features:
            if convert_to_packed:
                features, _ = pad_packed_sequence(
                    features, batch_first=True, total_length=L
                )
            out["features"] = features

        if return_loss:
            loss = self.criterion(out, batch)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss

        return out

    def forward(
        self, batch: Dict, return_loss: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        if self.training:
            return self.forward_train(
                batch, return_loss=return_loss, return_features=return_features
            )
        else:
            if self.cfg.forward_chunks:
                return self.forward_chunks(
                    batch, return_loss=return_loss, return_features=return_features
                )
            else:
                return self.forward_train(
                    batch, return_loss=return_loss, return_features=return_features
                )

    @torch.no_grad()
    def get_feature_dim(self) -> int:
        feature_maps = self.backbone(
            torch.randn(
                2,
                self.cfg.num_input_channels,
                self.cfg.image_height,
                self.cfg.image_width,
            )
        )
        # xcit models are channels-last
        return feature_maps.size(1 if "xcit" not in self.cfg.backbone else -1)

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

    @staticmethod
    def convert_seq_to_packed_sequence(
        seq: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert seq.shape[0] == mask.shape[0]
        lengths = mask.sum(1).cpu().int()
        sorted_indices = torch.argsort(lengths, descending=True)
        seq, lengths = seq[sorted_indices], lengths[sorted_indices]
        seq = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=True)
        return seq

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
        print("Freezing backbone ...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss


if __name__ == "__main__":
    from skp.configs import Config

    cfg = Config()
    cfg.backbone = "tf_efficientnetv2_b0"
    cfg.pretrained = True
    cfg.num_input_channels = 3
    cfg.backbone_img_size = False
    cfg.pool = "avg"
    cfg.seq_num_classes = 6
    cfg.cls_num_classes = 6
    cfg.add_auxiliary_classifier = True
    cfg.add_sequence_level_classifier = True
    cfg.rnn_class = "GRU"
    cfg.rnn_num_layers = 2
    cfg.rnn_dropout = 0.1
    cfg.attention_dropout = 0.1
    cfg.linear_seq_dropout = 0.1
    cfg.linear_aux_dropout = 0.1
    cfg.linear_cls_dropout = 0.1

    cfg.seq_len = 32
    cfg.image_height = 256
    cfg.image_width = 256

    x = torch.randn(
        (2, cfg.seq_len, cfg.num_input_channels, cfg.image_height, cfg.image_width)
    )
    mask = torch.ones((2, cfg.seq_len), dtype=torch.bool)
    mask[0, 10:] = 0
    mask[1, 5:] = 0

    net = Net(cfg)
    print(net)
    out = net({"x": x, "mask": mask})
    for k, v in out.items():
        print(k, v.shape, v)
