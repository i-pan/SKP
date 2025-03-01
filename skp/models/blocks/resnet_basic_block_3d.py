import torch
import torch.nn as nn

from typing import Callable

from .utils import LayerNorm3d, SeparableConv3d, SqueezeExcite3d


class ResNetBasicBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: tuple[int, int, int] = (2, 1, 1),
        se_layer: bool = False,
        norm_layer: Callable = LayerNorm3d,
        act_layer: Callable = nn.GELU,
        separable: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        conv_layer = SeparableConv3d if separable else nn.Conv3d

        self.conv1 = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm1 = norm_layer(out_channels)
        self.act1 = act_layer()

        self.conv2 = conv_layer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm2 = norm_layer(out_channels)
        self.act2 = act_layer()

        if se_layer:
            self.se = SqueezeExcite3d(out_channels, act_layer=act_layer)
        else:
            self.se = None
        self.downsample = (
            nn.Sequential(
                nn.AvgPool3d(stride, stride),
                nn.Conv3d(
                    out_channels, out_channels, 1, stride=1, padding=0, bias=False
                ),
                norm_layer(out_channels),
            )
            if stride[0] > 1
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.se is not None:
            x = self.se(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut

        x = self.act2(x)
        return x
