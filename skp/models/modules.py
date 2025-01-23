"""
Contains commonly used neural net modules.
"""

import math
import torch
import torch.nn as nn

from typing import Sequence


class FeatureReduction(nn.Module):
    """
    Reduce feature dimensionality
    Intended use is after the last layer of the neural net backbone, before pooling
    Grouped convolution is used to reduce # of extra parameters
    """

    def __init__(self, feature_dim: int, reduce_feature_dim: int):
        super().__init__()
        groups = math.gcd(feature_dim, reduce_feature_dim)
        self.reduce = nn.Conv2d(
            feature_dim,
            reduce_feature_dim,
            groups=groups,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(reduce_feature_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.reduce(x)))


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

    def forward(self, x):
        x = self.conv(x)
        if self.act_fn == "relu":
            x = torch.minimum(torch.relu(x), torch.tensor(self.upper).to(x.device))
        elif self.act_fn == "sigmoid":
            x = self.upper * x.sigmoid()
        return x
