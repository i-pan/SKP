import torch
import torch.nn as nn

from einops import rearrange
from timm import create_model
from torch.utils.checkpoint import checkpoint


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


class Conv3dAdapterBasicBlock(nn.Module):
    """
    Follows structure of ResNet BasicBlock
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: tuple[int, int, int] = (2, 1, 1),
        act_fn: str = "gelu",
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
        self.norm1 = LayerNorm3d(out_channels)
        self.conv2 = conv_layer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm2 = LayerNorm3d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.AvgPool3d(stride, stride),
                nn.Conv3d(
                    out_channels, out_channels, 1, stride=1, padding=0, bias=False
                ),
                LayerNorm3d(out_channels),
            )
            if stride[0] > 1
            else None
        )
        if act_fn == "gelu":
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()
        elif act_fn == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif act_fn == "silu":
            self.act1 = nn.SiLU()
            self.act2 = nn.SiLU()
        else:
            raise Exception(
                f"`act_fn` must be one of [`gelu`, `relu`, `silu`], got `{act_fn}`"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut

        x = self.act2(x)
        return x


def unstack_3d(x: torch.Tensor, n: int) -> torch.Tensor:
    x = rearrange(x, "(n t) c h w -> n c t h w", n=n)
    return x


def stack_2d(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "n c t h w -> (n t) c h w")
    return x


class EffNetAdapt3d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert (
            "efficientnet" in self.cfg.backbone
        ), "Only EfficientNet backbones are supported"
        self.backbone = create_model(
            self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            global_pool="",
            num_classes=0,
            in_chans=self.cfg.num_input_channels,
            features_only=self.cfg.features_only or False,
        )
        adapter_channels, downsample_blocks, feature_dim = (
            self.get_adapter_channels_downsample_blocks_and_dim_feats()
        )
        self.adapter_channels = adapter_channels
        self.downsample_blocks = downsample_blocks
        self.feature_dim = feature_dim

        self.adapters = self.get_adapters()

        if self.cfg.freeze_backbone:
            self.freeze_backbone()

    def forward_blocks(
        self, x: torch.Tensor, n: int, return_feature_maps: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        feature_maps = []
        for idx, block in enumerate(self.backbone.blocks):
            x = block(x)
            if idx in self.downsample_blocks:
                x = unstack_3d(x, n)
                x = self.adapters[self.downsample_blocks.index(idx) + 1](x)
                if return_feature_maps:
                    feature_maps.append(x)
                x = stack_2d(x)
        if return_feature_maps:
            return x, feature_maps
        return x

    def forward(
        self, x: torch.Tensor, return_feature_maps: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        n, c, t, h, w = x.shape
        x = stack_2d(x)
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = unstack_3d(x, n)
        x = self.adapters[0](x)
        if return_feature_maps:
            feature_maps = [x]
        x = stack_2d(x)
        if self.cfg.enable_gradient_checkpointing:
            x = checkpoint(
                self.forward_blocks, *(x, n, return_feature_maps), use_reentrant=False
            )
        else:
            x = self.forward_blocks(x, n, return_feature_maps)
        if return_feature_maps:
            feature_maps.extend(x[1])
            x = x[0]
        if self.cfg.features_only:
            return feature_maps
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = unstack_3d(x, n)
        out = {"logits": x}
        if return_feature_maps:
            out["feature_maps"] = feature_maps
        return out

    @torch.no_grad()
    def get_adapter_channels_downsample_blocks_and_dim_feats(
        self,
    ) -> tuple[list[int], list[int], int]:
        model = self.backbone
        in_chans = self.cfg.num_input_channels
        x0 = torch.randn((2, in_chans, 64, 64))
        x = model.conv_stem(x0)
        adapter_channels = [x.shape[1]]
        x = model.bn1(x)
        downsample_blocks = []
        for idx, block in enumerate(model.blocks):
            h, w = x.shape[2:]
            x = block(x)
            if h > x.shape[2]:
                downsample_blocks.append(idx)
                adapter_channels.append(x.shape[1])

        if not self.cfg.features_only:
            x = model.conv_head(x)
            dim_feats = x.shape[1]
        else:
            dim_feats = None 

        return adapter_channels, downsample_blocks, dim_feats

    def get_adapters(self) -> nn.ModuleList:
        if self.cfg.adapter_num_blocks == 1:
            adapters = nn.ModuleList(
                [
                    Conv3dAdapterBasicBlock(
                        in_channels=self.adapter_channels[i],
                        out_channels=self.adapter_channels[i],
                        kernel_size=self.cfg.adapter_kernel_size,
                        stride=(self.cfg.adapter_strides[i], 1, 1),
                        separable=self.cfg.adapter_separable_conv,
                    )
                    for i in range(len(self.adapter_channels))
                ]
            )
        else:
            adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        *[
                            Conv3dAdapterBasicBlock(
                                in_channels=self.adapter_channels[i],
                                out_channels=self.adapter_channels[i],
                                kernel_size=self.cfg.adapter_kernel_size,
                                stride=(self.cfg.adapter_strides[i], 1, 1)
                                if _ == 0
                                else (1, 1, 1),
                                separable=self.cfg.adapter_separable_conv,
                            )
                            for _ in range(self.cfg.adapter_num_blocks)
                        ]
                    )
                    for i in range(len(self.adapter_channels))
                ]
            )
        return adapters

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    from skp.configs import Config
    from skp.toolbox.functions import count_parameters

    cfg = Config()
    cfg.backbone = "tf_efficientnetv2_b0"
    cfg.pretrained = True
    cfg.num_input_channels = 1
    cfg.adapter_num_blocks = 1
    cfg.adapter_kernel_size = (7, 7, 7)
    cfg.adapter_strides = [2, 2, 1, 1, 1]
    cfg.adapter_separable_conv = True
    cfg.enable_gradient_checkpointing = True
    cfg.freeze_backbone = False
    model = EffNetAdapt3d(cfg)
    x = torch.randn((2, cfg.num_input_channels, 256, 256, 256))
    y = model(x, return_feature_maps=True)
    for i in y["feature_maps"]:
        print(i.shape)
    count_parameters(model)
    count_parameters(model.backbone)
    count_parameters(model.adapters)
