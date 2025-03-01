import torch
import torch.nn as nn

from einops import rearrange
from typing import Callable


def num_groups(group_size: int | None, channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


def make_divisible(
    v: int, divisor: int = 8, min_value: int | None = None, round_limit: float = 0.9
):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SqueezeExcite3d(nn.Module):
    """Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
        self,
        in_chs: int,
        rd_ratio: float = 0.25,
        rd_channels: int | None = None,
        act_layer: Callable = nn.ReLU,
        gate_layer: Callable = nn.Sigmoid,
        force_act_layer: Callable | None = None,
        rd_round_fn: Callable | None = None,
    ):
        super().__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv3d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv3d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class LayerScale3d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class LayerNorm3d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "n c t h w -> n t h w c")
        x = super().forward(x)
        x = rearrange(x, "n t h w c -> n c t h w")
        return x


class SeparableConv3d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        depthwise_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(depthwise_conv, pointwise_conv)
