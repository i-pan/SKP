"""
Adapted from:
https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/deeplabv3/decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Iterable, Callable


class LayerNorm3d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "n c t h w -> n t h w c")
        x = super().forward(x)
        x = rearrange(x, "n t h w c -> n c t h w")
        return x


class DeepLabV3PlusDecoder3d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.decoder_out_channels = self.cfg.decoder_out_channels or 256
        # following ASPP-L in https://arxiv.org/pdf/1606.00915
        self.atrous_rates = self.cfg.atrous_rates or (6, 12, 18, 24)
        self.aspp_separable = self.cfg.aspp_separable or False
        self.aspp_dropout = self.cfg.aspp_dropout or 0.0

        decoder_norm_layer = self.cfg.decoder_norm_layer or "batch_norm"
        if decoder_norm_layer == "batch_norm":
            norm_layer = nn.BatchNorm3d
        elif decoder_norm_layer == "layer_norm":
            norm_layer = LayerNorm3d

        decoder_act_layer = self.cfg.decoder_act_layer or "relu"
        if decoder_act_layer == "relu":
            act_layer = nn.ReLU()
        elif decoder_act_layer == "gelu":
            act_layer = nn.GELU()
        elif decoder_act_layer == "silu":
            act_layer = nn.SiLU()

        self.aspp = nn.Sequential(
            ASPP(
                self.cfg.encoder_channels[-1],
                self.decoder_out_channels,
                self.atrous_rates,
                separable=self.aspp_separable,
                dropout=self.aspp_dropout,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
            SeparableConv3d(
                self.decoder_out_channels,
                self.decoder_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm_layer(self.decoder_out_channels),
            act_layer,
        )

        highres_in_channels = self.cfg.encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv3d(
                highres_in_channels, highres_out_channels, kernel_size=1, bias=False
            ),
            norm_layer(highres_out_channels),
            act_layer,
        )
        self.block2 = nn.Sequential(
            SeparableConv3d(
                highres_out_channels + self.decoder_out_channels,
                self.decoder_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm_layer(self.decoder_out_channels),
            act_layer,
        )

    def forward(self, features):
        aspp_features = self.aspp(features[-1])
        high_res_features = self.block1(features[-4])
        aspp_features = F.interpolate(
            aspp_features, size=high_res_features.shape[2:], mode="trilinear"
        )
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return aspp_features, fused_features


class ASPPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        norm_layer: Callable = nn.BatchNorm3d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer,
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        norm_layer: Callable = nn.BatchNorm3d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__(
            SeparableConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer,
        )


class ASPPPooling(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable = nn.BatchNorm3d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            act_layer,
        )

    def forward(self, x):
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: Iterable[int],
        separable: bool,
        dropout: float,
        norm_layer: Callable = nn.BatchNorm3d,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                act_layer,
            )
        ]

        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv
        for rate in atrous_rates:
            modules.append(
                ASPPConvModule(
                    in_channels,
                    out_channels,
                    rate,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )

        modules.append(
            ASPPPooling(
                in_channels, out_channels, norm_layer=norm_layer, act_layer=act_layer
            )
        )

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(
                (len(atrous_rates) + 2) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


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


if __name__ == "__main__":
    import torch
    from einops import rearrange
    from skp.configs import Config
    from timm import create_model

    cfg = Config()
    cfg.decoder_out_channels = 256
    cfg.atrous_rates = (2, 4, 6, 12)
    cfg.aspp_separable = False
    cfg.aspp_dropout = 0.1
    cfg.decoder_norm_layer = "layer_norm"
    cfg.decoder_act_layer = "gelu"

    bb = create_model("tf_efficientnetv2_b0", features_only=True)
    x = torch.randn((2, 32, 3, 256, 256))
    x = rearrange(x, "b n c h w -> (b n) c h w")
    features = bb(x)
    features = [rearrange(f, "(b n) c h w -> b c n h w", b=2) for f in features]
    cfg.encoder_channels = [f.shape[1] for f in features]
    model = DeepLabV3PlusDecoder3d(cfg)
    print(model)
    out = model(features)
    print(out[0].shape)
