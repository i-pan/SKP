import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.create_act import create_act_layer
from timm.layers.trace_utils import _assert
from timm.layers.padding import get_same_padding

from typing import List, Optional, Tuple


def _create_act(act_layer, act_kwargs=None, inplace=False, apply_act=True):
    act_kwargs = act_kwargs or {}
    act_kwargs.setdefault("inplace", inplace)
    act = None
    if apply_act:
        act = create_act_layer(act_layer, **act_kwargs)
    return nn.Identity() if act is None else act


def pad_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int] = (1, 1, 1),
    value: float = 0,
):
    it, ih, iw = x.size()[-3:]
    pad_t = get_same_padding(it, kernel_size[0], stride[0], dilation[0])
    pad_h = get_same_padding(ih, kernel_size[1], stride[1], dilation[1])
    pad_w = get_same_padding(iw, kernel_size[2], stride[2], dilation[2])
    padding = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_t // 2,
        pad_t - pad_t // 2,
    )
    x = F.pad(
        x,
        padding,
        value=value,
    )
    return x


def conv3d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv3dSame(nn.Conv3d):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode=None,
    ):
        super(Conv3dSame, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        return conv3d_same(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BatchNormAct3d(nn.BatchNorm3d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
        drop_layer=None,
        device=None,
        dtype=None,
    ):
        try:
            factory_kwargs = {"device": device, "dtype": dtype}
            super(BatchNormAct3d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                **factory_kwargs,
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAct3d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm3d.forward impl to avoid issues with torchscript and tracing
        _assert(x.ndim == 5, f"expected 5D input (got {x.ndim}D input)")

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x
