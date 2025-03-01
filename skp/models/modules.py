"""
Assortment of neural net modules.
"""

import math
import torch
import torch.nn as nn

from einops import rearrange
from typing import Sequence


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        # channels-first to channels-last and back
        if x.ndim == 3:
            pat1, pat2 = "n c d -> n d c", "n d c -> n c d"
        elif x.ndim == 4:
            pat1, pat2 = "n c h w -> n h w c", "n h w c -> n c h w"
        elif x.ndim == 5:
            pat1, pat2 = "n c t h w -> n t h w c", "n t h w c -> n c t h w"

        x = rearrange(x, pat1)
        x = super().forward(x)
        x = rearrange(x, pat2)
        return x


class FeatureReduction(nn.Module):
    """
    Reduce feature dimensionality
    Grouped convolution can be used to reduce # of extra parameters
    """

    def __init__(
        self,
        feature_dim: int,
        reduce_feature_dim: int,
        dim: int = 2,
        reduce_grouped_conv: bool = False,
        add_norm: bool = True,
        add_act: bool = True,
    ):
        super().__init__()
        groups = math.gcd(feature_dim, reduce_feature_dim)
        self.reduce = getattr(nn, f"Conv{dim}d")(
            feature_dim,
            reduce_feature_dim,
            groups=groups if reduce_grouped_conv else 1,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.norm = LayerNorm(reduce_feature_dim) if add_norm else nn.Identity()
        self.act = nn.GELU() if add_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.reduce(x)))


class ConvWSO(nn.Module):
    """
    Learnable windowing convolution for CT images
    Based on:
        https://github.com/MGH-LMIC/windows_optimization
        https://arxiv.org/pdf/1812.00572
    """

    def __init__(self, WL: Sequence[int], WW: Sequence[int], act_fn: str = "sigmoid"):
        super().__init__()
        assert len(WL) == len(WW)
        self.conv = nn.Conv2d(1, len(WL), kernel_size=1, stride=1, padding=0, bias=True)
        self.upper = torch.tensor(255)
        self.act_fn = act_fn
        self.init_weights(WL, WW)

    def init_weights(self, WL: Sequence[int], WW: Sequence[int]) -> None:
        state_dict = self.conv.state_dict().copy()
        for idx in range(len(WL)):
            width, level = WW[idx], WL[idx]
            if self.act_fn == "relu":
                state_dict["weight"][idx, 0, 0, 0] = self.upper / width
                state_dict["bias"][idx] = -self.upper * (level - width / 2.0) / width
            elif self.act_fn == "sigmoid":
                state_dict["weight"][idx, 0, 0, 0] = (2.0 / width) * torch.log(
                    self.upper - 1
                )
                state_dict["bias"][idx] = (-2.0 * level / width) * torch.log(
                    self.upper - 1
                )
        self.conv.load_state_dict(state_dict)

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        if self.act_fn == "relu":
            x = torch.minimum(torch.relu(x), torch.tensor(self.upper).to(x.device))
        elif self.act_fn == "sigmoid":
            x = self.upper * x.sigmoid()
        return x
