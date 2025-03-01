import torch.nn as nn

from typing import Callable

from .utils import make_divisible, num_groups, SqueezeExcite3d, LayerScale3d


class UniversalInvertedResidual3d(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size_start: int = 0,
        dw_kernel_size_mid: int = 3,
        dw_kernel_size_end: int = 0,
        stride: int | tuple[int, int, int] = (2, 1, 1),
        dilation: int = 1,
        padding: int = 1,
        group_size: int = 1,
        noskip: bool = False,
        exp_ratio: float = 1.0,
        act_layer: Callable = nn.ReLU,
        norm_layer: Callable = nn.BatchNorm3d,
        se_layer: bool = False,
        conv_kwargs: dict | None = None,
        layer_scale_init_value: float | None = 1e-5,
    ):
        super().__init__()
        conv_kwargs = conv_kwargs or {}
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.has_skip = (in_chs == out_chs and stride[0] in {1, 2}) and not noskip
        if stride[0] > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = nn.Sequential(
                nn.Conv3d(
                    in_chs,
                    in_chs,
                    kernel_size=dw_kernel_size_start,
                    stride=dw_start_stride,
                    dilation=dilation,
                    padding=padding,
                    groups=dw_start_groups,
                    bias=False,
                ),
                norm_layer(in_chs),
                # no activation
            )
        else:
            self.dw_start = nn.Identity()

        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = nn.Sequential(
            nn.Conv3d(in_chs, mid_chs, 1, bias=False),
            norm_layer(mid_chs),
            act_layer(),
        )

        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = nn.Sequential(
                nn.Conv3d(
                    mid_chs,
                    mid_chs,
                    dw_kernel_size_mid,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                norm_layer(mid_chs),
                act_layer(),
            )

        self.se = (
            SqueezeExcite3d(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()
        )

        self.pw_proj = nn.Sequential(
            nn.Conv3d(mid_chs, out_chs, 1, bias=False),
            norm_layer(out_chs),
            # no activation
        )

        if dw_kernel_size_end:
            dw_end_stride = (
                stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            )
            dw_end_groups = num_groups(group_size, out_chs)
            self.dw_end = nn.Sequential(
                nn.Conv3d(
                    out_chs,
                    out_chs,
                    dw_kernel_size_end,
                    stride=dw_end_stride,
                    dilation=dilation,
                    padding=padding,
                    groups=dw_end_groups,
                    bias=False,
                ),
                norm_layer(out_chs),
                act_layer(),
            )
        else:
            self.dw_end = nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale3d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        if stride[0] == 2 and self.has_skip:
            self.downsample = nn.Sequential(
                nn.AvgPool3d(stride, stride),
                nn.Conv3d(in_chs, out_chs, 1, padding=0, bias=False),
                norm_layer(out_chs),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = x + self.downsample(shortcut)
        return x
