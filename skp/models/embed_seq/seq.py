"""
Sequence model only
"""

import torch
import torch.nn as nn

from einops import rearrange

from skp.models.embed_seq import heads
from skp.models.modules import FeatureReduction


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = self.cfg.feature_dim
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
        self.head = getattr(heads, cfg.head_type)(cfg, feature_dim=self.feature_dim)
        self.criterion = None

    def forward(
        self, batch: dict[str, torch.Tensor], return_loss: bool = False
    ) -> dict[str, torch.Tensor]:
        x = batch["x"]
        if hasattr(self, "feature_reduction"):
            x = rearrange(x, "b n c -> b c n")
            x = self.feature_reduction(x)
            x = rearrange(x, "b c n -> b n c")
        out = self.head(x, mask=batch.get("mask", None))
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, criterion: nn.Module) -> None:
        self.criterion = criterion
