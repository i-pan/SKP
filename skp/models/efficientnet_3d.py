"""
Converts 2D EfficientNets from PyTorch Image Models (timm) to 3D.
This script was mostly written by Gemini 2.0 Flash Thinking Experimental 01-21

NOTE: most recently tested with timm-1.0.12 
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from timm.layers import BatchNormAct2d, Conv2dSame
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


def inflate_conv2d_to_conv3d(conv2d, conv3d, center_inflation=False):
    """Inflates a 2D Conv layer weights to 3D Conv layer weights by copying
    the 2D weights along the new dimension and averaging.

    Args:
        conv2d (nn.Conv2d): The original 2D convolutional layer.
        conv3d (nn.Conv3d): The target 3D convolutional layer.
    """
    # Get the weights from the 2D conv layer
    weight_2d = conv2d.weight.data

    # Inflate the weights to 3D by repeating along the depth dimension
    kernel_size_3d = conv3d.kernel_size[
        0
    ]  # Assume cubic kernel or use conv3d.kernel_size[2] if needed
    if center_inflation:
        # Center Inflation: Copy 2D weights to the central slice and zero out others
        weight_3d = torch.zeros_like(conv3d.weight.data)
        center_index = kernel_size_3d // 2
        weight_3d[:, :, center_index, :, :] = weight_2d
    else:
        # Averaging Inflation: Repeat 2D weights along depth and average
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, kernel_size_3d, 1, 1)# / kernel_size_3d

    # Copy the inflated weights to the 3D conv layer
    conv3d.weight.data.copy_(weight_3d)

    # Copy bias if it exists
    if conv2d.bias is not None and conv3d.bias is not None:
        conv3d.bias.data.copy_(conv2d.bias.data)


def inflate_linear_to_linear(linear2d, linear3d):
    """Inflates a 2D Linear layer weights to 3D Linear layer weights by simply copying.
    Linear layers are not spatially dependent and thus don't need inflation like Conv layers.

    Args:
        linear2d (nn.Linear): The original 2D linear layer.
        linear3d (nn.Linear): The target 3D linear layer.
    """
    linear3d.weight.data.copy_(linear2d.weight.data)
    if linear2d.bias is not None and linear3d.bias is not None:
        linear3d.bias.data.copy_(linear2d.bias.data)


def convert_efficientnet2d_to_3d(
    model_name, pretrained=True, num_input_channels=3, features_only=False
):
    """
    Converts a 2D EfficientNet model from timm to a 3D EfficientNet model.

    Args:
        model_name (str): The name of the EfficientNet model in timm (e.g., 'efficientnet_b0').
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        nn.Module: The converted 3D EfficientNet model.
    """

    # Load the 2D pretrained model
    kwargs = {
        "pretrained": pretrained,
        "in_chans": num_input_channels,
        "features_only": features_only,
        "global_pool": "",
        "num_classes": 0,
    }
    model_2d = timm.create_model(model_name, **kwargs)
    kwargs["pretrained"] = False
    model_3d = timm.create_model(model_name, **kwargs)  # Create a fresh model structure

    # Store original 2D weights for inflation
    pretrained_weights = OrderedDict()
    if pretrained:
        for name, module in model_2d.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                pretrained_weights[name + ".weight"] = module.weight.data
                if module.bias is not None:
                    pretrained_weights[name + ".bias"] = module.bias.data
            elif isinstance(module, (nn.BatchNorm2d)):
                pretrained_weights[name + ".weight"] = module.weight.data
                pretrained_weights[name + ".bias"] = module.bias.data
                pretrained_weights[name + ".running_mean"] = module.running_mean.data
                pretrained_weights[name + ".running_var"] = module.running_var.data
                if module.affine and module.track_running_stats:
                    pretrained_weights[name + ".num_batches_tracked"] = (
                        module.num_batches_tracked.data
                    )
            elif (
                hasattr(module, "__class__")
                and module.__class__.__name__ == "BatchNormAct2d"
            ):  # Handle custom BatchNormAct2d
                pretrained_weights[name + ".weight"] = module.weight.data
                pretrained_weights[name + ".bias"] = module.bias.data
                pretrained_weights[name + ".running_mean"] = module.running_mean.data
                pretrained_weights[name + ".running_var"] = module.running_var.data
                if module.affine and module.track_running_stats:
                    pretrained_weights[name + ".num_batches_tracked"] = (
                        module.num_batches_tracked.data
                    )

    modules_converted = 0
    modules_inflated = 0

    for name, module_2d, module_3d in zip(
        model_2d.named_modules(), model_2d.modules(), model_3d.modules()
    ):
        name = name[0]  # name in named_modules is tuple (name, module)

        if isinstance(module_2d, nn.Conv2d):
            # Get 2D layer attributes
            out_channels = module_2d.out_channels
            in_channels = module_2d.in_channels
            kernel_size = module_2d.kernel_size
            stride = module_2d.stride
            padding = module_2d.padding
            dilation = module_2d.dilation
            groups = module_2d.groups
            bias = module_2d.bias is not None
            padding_mode = module_2d.padding_mode

            # Create 3D Conv layer with inflated parameters
            kernel_size_3d = (
                kernel_size[0],
                kernel_size[1],
                kernel_size[0],
            )  # Expand to cubic kernel
            stride_3d = (stride[0], stride[1], stride[0])  # Expand stride
            padding_3d = (padding[0], padding[1], padding[0])  # Expand padding
            dilation_3d = (dilation[0], dilation[1], dilation[0])  # Expand dilation

            conv_layer = Conv3dSame if isinstance(module_2d, Conv2dSame) else nn.Conv3d
            conv3d = conv_layer(
                out_channels=out_channels,
                in_channels=in_channels,
                kernel_size=kernel_size_3d,
                stride=stride_3d,
                padding=padding_3d,
                dilation=dilation_3d,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )

            # Replace the 2D conv layer with the 3D conv layer in the 3D model
            if name:  # Handle top level module replacement
                parent_name_parts = name.split(".")
                parent_module = model_3d
                for part in parent_name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, parent_name_parts[-1], conv3d)
            else:
                model_3d = conv3d  # For the very first layer

            modules_converted += 1
            if pretrained:
                inflate_conv2d_to_conv3d(module_2d, conv3d)
                modules_inflated += 1

        elif isinstance(module_2d, nn.MaxPool2d):
            # Get 2D layer attributes
            kernel_size = module_2d.kernel_size
            stride = module_2d.stride
            padding = module_2d.padding
            dilation = module_2d.dilation
            return_indices = module_2d.return_indices
            ceil_mode = module_2d.ceil_mode

            # Create 3D MaxPool layer with inflated parameters
            kernel_size_3d = (
                (kernel_size, kernel_size, kernel_size)
                if isinstance(kernel_size, int)
                else (kernel_size[0], kernel_size[1], kernel_size[0])
            )
            stride_3d = (
                (stride, stride, stride)
                if isinstance(stride, int)
                else (stride[0], stride[1], stride[0])
                if stride is not None
                else stride
            )
            padding_3d = (
                (padding, padding, padding)
                if isinstance(padding, int)
                else (padding[0], padding[1], padding[0])
            )
            dilation_3d = (
                (dilation, dilation, dilation)
                if isinstance(dilation, int)
                else (dilation[0], dilation[1], dilation[0])
            )

            maxpool3d = nn.MaxPool3d(
                kernel_size=kernel_size_3d,
                stride=stride_3d,
                padding=padding_3d,
                dilation=dilation_3d,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            # Replace the 2D layer with the 3D layer in the 3D model
            if name:
                parent_name_parts = name.split(".")
                parent_module = model_3d
                for part in parent_name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, parent_name_parts[-1], maxpool3d)
            else:
                model_3d = maxpool3d
            modules_converted += 1

        elif isinstance(module_2d, nn.AvgPool2d):
            # Get 2D layer attributes
            kernel_size = module_2d.kernel_size
            stride = module_2d.stride
            padding = module_2d.padding
            ceil_mode = module_2d.ceil_mode
            count_include_pad = module_2d.count_include_pad
            divisor_override = module_2d.divisor_override

            # Create 3D AvgPool layer with inflated parameters
            kernel_size_3d = (
                (kernel_size, kernel_size, kernel_size)
                if isinstance(kernel_size, int)
                else (kernel_size[0], kernel_size[1], kernel_size[0])
            )
            stride_3d = (
                (stride, stride, stride)
                if isinstance(stride, int)
                else (stride[0], stride[1], stride[0])
                if stride is not None
                else stride
            )
            padding_3d = (
                (padding, padding, padding)
                if isinstance(padding, int)
                else (padding[0], padding[1], padding[0])
            )

            avgpool3d = nn.AvgPool3d(
                kernel_size=kernel_size_3d,
                stride=stride_3d,
                padding=padding_3d,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
            # Replace the 2D layer with the 3D layer in the 3D model
            if name:
                parent_name_parts = name.split(".")
                parent_module = model_3d
                for part in parent_name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, parent_name_parts[-1], avgpool3d)
            else:
                model_3d = avgpool3d
            modules_converted += 1

        elif isinstance(module_2d, nn.AdaptiveAvgPool2d):
            # Get 2D layer attributes
            output_size = module_2d.output_size

            # Create 3D AdaptiveAvgPool layer with inflated parameters
            output_size_3d = (
                (output_size, output_size, output_size)
                if isinstance(output_size, int)
                else (output_size[0], output_size[1], output_size[0])
            )

            adaptiveavgpool3d = nn.AdaptiveAvgPool3d(output_size=output_size_3d)
            # Replace the 2D layer with the 3D layer in the 3D model
            if name:
                parent_name_parts = name.split(".")
                parent_module = model_3d
                for part in parent_name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, parent_name_parts[-1], adaptiveavgpool3d)
            else:
                model_3d = adaptiveavgpool3d
            modules_converted += 1

        elif isinstance(module_2d, nn.BatchNorm2d):
            # Get 2D layer attributes
            num_features = module_2d.num_features
            eps = module_2d.eps
            momentum = module_2d.momentum
            affine = module_2d.affine
            track_running_stats = module_2d.track_running_stats

            if isinstance(module_2d, BatchNormAct2d):
                # Create 3D BatchNormAct layer
                batchnorm3d = BatchNormAct3d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                    act_layer=type(module_2d.act),
                    inplace=module_2d.act.inplace
                    if hasattr(module_2d.act, "inplace")
                    else False,
                    drop_layer=type(module_2d.drop),
                )

            else:
                # Create 3D BatchNorm layer
                batchnorm3d = nn.BatchNorm3d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                )

            # Replace the 2D layer with the 3D layer in the 3D model
            if name:
                parent_name_parts = name.split(".")
                parent_module = model_3d
                for part in parent_name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, parent_name_parts[-1], batchnorm3d)
            else:
                model_3d = batchnorm3d
            modules_converted += 1
            if pretrained:
                # Copy BatchNorm weights directly
                for param_name in ["weight", "bias", "running_mean", "running_var"]:
                    if name + "." + param_name in pretrained_weights:
                        getattr(batchnorm3d, param_name).data.copy_(
                            pretrained_weights[name + "." + param_name]
                        )
                if (
                    module_2d.affine
                    and module_2d.track_running_stats
                    and name + ".num_batches_tracked" in pretrained_weights
                ):
                    batchnorm3d.num_batches_tracked.data.copy_(
                        pretrained_weights[name + ".num_batches_tracked"]
                    )
                modules_inflated += 1

        elif isinstance(module_2d, nn.Linear):
            # Get 2D layer attributes
            in_features = module_2d.in_features
            out_features = module_2d.out_features
            bias = module_2d.bias is not None

            # Create 3D Linear layer (though linear layers are not inherently 2D/3D, we replace for consistency)
            linear3d = nn.Linear(
                in_features=in_features, out_features=out_features, bias=bias
            )

            # Replace the 2D layer with the 3D layer in the 3D model
            if name:
                parent_name_parts = name.split(".")
                parent_module = model_3d
                for part in parent_name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, parent_name_parts[-1], linear3d)
            else:
                model_3d = linear3d
            modules_converted += 1
            if pretrained:
                inflate_linear_to_linear(module_2d, linear3d)
                modules_inflated += 1

    print(f"Converted {modules_converted} modules from 2D to 3D.")
    if pretrained:
        print(f"Inflated pretrained weights for {modules_inflated} modules.")

    return model_3d


def change_dim0_strides(model, dim0_strides=[2, 2, 2, 2, 2]):
    stride2_idx = 0
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            if module.stride[0] == 2:
                module.stride = (
                    dim0_strides[stride2_idx],
                    module.stride[1],
                    module.stride[2],
                )
                stride2_idx += 1


if __name__ == "__main__":
    model_name = "tf_efficientnetv2_b0"  # Example EfficientNet model name
    model_3d = convert_efficientnet2d_to_3d(
        model_name, pretrained=True, features_only=True
    )

    # Example usage: dummy 5D input (B, C, D, H, W) for video/3D data
    batch_size = 2
    channels = 3
    depth = 32
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, channels, depth, height, width)

    dim0_strides = [1, 1, 1, 1, 1]
    change_dim0_strides(model_3d, dim0_strides)
    output_3d = model_3d(dummy_input)
    model_3d.set_grad_checkpointing()
    if isinstance(output_3d, torch.Tensor):
        print(
            "3D EfficientNet Output shape:", output_3d.shape
        )  # Should be [batch_size, num_classes]
    elif isinstance(output_3d, list):
        print("3D EfficientNet Output shape:", [o.shape for o in output_3d])
    # Verify that layers are actually 3D
    # for name, module in model_3d.named_modules():
    #     if isinstance(
    #         module,
    #         (
    #             nn.Conv3d,
    #             nn.MaxPool3d,
    #             nn.AvgPool3d,
    #             nn.AdaptiveAvgPool3d,
    #             nn.BatchNorm3d,
    #             BatchNormAct3d,
    #         ),
    #     ):
    #         print(f"{name}: {module.__class__.__name__}")

    # out = model_3d.conv_stem(dummy_input)
    # print(out.shape)
    # print(model_3d)

    dim0_strides = [2, 2, 2, 2, 2]
    stride2_idx = 0
    for name, module in model_3d.named_modules():
        if isinstance(module, nn.Conv3d):
            if module.stride[0] == 2:
                module.stride = tuple([dim0_strides[stride2_idx]] * 3)
                stride2_idx += 1
                print(module)
    print(output_3d)