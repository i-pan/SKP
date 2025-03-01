import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from einops import rearrange
from timm import create_model
from timm.layers import trunc_normal_, DropPath
from torch.utils.checkpoint import checkpoint
from urllib.parse import urlparse


def download_checkpoint(url: str, cache_dir: str = torch.hub.get_dir()) -> str:
    """
    Downloads a PyTorch model checkpoint from the web to a local cache directory,
    if it does not already exist.

    Args:
        url (str): The URL of the PyTorch model checkpoint.
        cache_dir (str): The local directory to cache the checkpoint in.

    Returns:
        str: The local path to the downloaded checkpoint file.
             Returns None if download fails.
    """
    os.makedirs(cache_dir, exist_ok=True)  # Create cache directory if it doesn't exist

    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if (
        not filename
    ):  # Handle cases where basename might be empty (e.g., URL ends with '/')
        filename = "downloaded_checkpoint"  # Default filename if extraction fails

    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        print(f"Checkpoint already exists at: {filepath}")
        return filepath

    print(f"Downloading checkpoint from: {url} to {filepath}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Checkpoint downloaded successfully to: {filepath}")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"Error downloading checkpoint from {url}: {e}")
        return None


_PRETRAINED_WEIGHTS = {
    "convnextv2_atto": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt",
    "convnextv2_femto": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt",
    "convnextv2_pico": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt",
    "convnextv2_nano": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt",
    "convnextv2_tiny": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
    "convnextv2_base": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt",
    "convnextv2_large": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt",
    "convnextv2_huge": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt",
}


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class GRN3d(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is channels-last
        Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        kernel_size: tuple[int, int, int] | int = 7,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        if dim_out is None:
            dim_out = dim
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN3d(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim_out)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = rearrange(x, "n c t h w -> n t h w c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = rearrange(x, "n t h w c -> n c t h w")

        x = input + self.drop_path(x)
        return x


class DecoderBlock(Block):
    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        if skip is not None:
            t, h, w = skip.shape[-3:]
            x = F.interpolate(x, size=(t, h, w), mode="nearest")
            x = torch.cat([x, skip], dim=1)
        x = super().forward(x)
        return x


class ConvNeXtV2_3d(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        block_kernel_size: tuple[int, int, int] | int = (7, 7, 7),
        spatial_strides: list[int] = [4, 2, 2, 2],
        temporal_strides: list[int] = [4, 2, 2, 2],
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        features_only: bool = False,
    ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem_stride = (temporal_strides[0], spatial_strides[0], spatial_strides[0])
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=stem_stride, stride=stem_stride),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            stride = (
                temporal_strides[i + 1],
                spatial_strides[i + 1],
                spatial_strides[i + 1],
            )
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=stride, stride=stride),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        kernel_size=block_kernel_size,
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.features_only = features_only
        if not self.features_only:
            #self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
            self.head = nn.Linear(dims[-1], num_classes)

            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.grad_checkpointing = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        feature_maps = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            # actually don't return stem feature map, like in timm
            # if return_feature_maps:
            #     if i == 0:
            #         # stem
            #         feature_maps.append(x)
            if self.grad_checkpointing:
                x = checkpoint(self.stages[i], *(x,), use_reentrant=False)
            else:
                x = self.stages[i](x)
            feature_maps.append(x)
        # if self.features_only:
        #     return feature_maps
        # gap_feature = self.norm(F.adaptive_avg_pool3d(x, 1).flatten(1))
        # return gap_feature, feature_maps
        return feature_maps

    def forward(self, x, return_feature_maps=False):
        feature_maps = self.forward_features(x)
        if self.features_only:
            return feature_maps
        x = self.head(feature_maps[-1])
        return x

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable


def _inflate_pretrained_weights(weights, state_dict_shapes):
    inflated_weights = OrderedDict()
    for k, v in weights.items():
        if k not in state_dict_shapes:
            continue
        if state_dict_shapes[k] == v.shape:
            inflated_weights[k] = v
            continue
        if k.endswith(".weight"):
            # add extra spatial dimension
            v = v.unsqueeze(2)
            # check shape in model
            v_shape = state_dict_shapes[k][2]
            if v_shape > 1:
                # repeat weight values along new spatial axis
                v = v.repeat(1, 1, v_shape, 1, 1)
                # scale values by v_shape
                # sum of weights along new spatial axis will equal
                # original weight
                v = v / v_shape
        elif "grn" in k:
            # add extra spatial dimension
            v = v.unsqueeze(2)
        inflated_weights[k] = v
    return inflated_weights


_REPLACE_NAMES = {
    "blocks.": "",
    "conv_dw": "dwconv",
    "mlp.fc1": "pwconv1",
    "mlp.fc2": "pwconv2",
    "stem": "downsample_layers.0",
    "stages.1.downsample": "downsample_layers.1",
    "stages.2.downsample": "downsample_layers.2",
    "stages.3.downsample": "downsample_layers.3",
    "mlp.grn.weight": "grn.gamma",
    "mlp.grn.bias": "grn.beta",
    "head.fc": "head",
    "head.norm": "norm",
}


def _convert_timm_pretrained_weights(weights):
    new_weights = OrderedDict()
    for k, v in weights.items():
        for src, dst in _REPLACE_NAMES.items():
            k = k.replace(src, dst)
        new_weights[k] = v
    for k, v in new_weights.items():
        if "pwconv" in k and "weight" in k:
            if v.ndim == 4:
                new_weights[k] = v[:, :, 0, 0]
        elif "grn" in k:
            new_weights[k] = v.expand(1, 1, 1, -1)
    return new_weights


def _create_convnextv2_3d(name, depths, dims, **kwargs):
    pretrained = kwargs.pop("pretrained", False)
    use_timm_weights = kwargs.pop("use_timm_weights", False)
    headless = kwargs.pop("headless", False)
    features_only = kwargs.get("features_only", False)
    spatial_strides = kwargs.get("spatial_strides", [4, 2, 2, 2])

    model = ConvNeXtV2_3d(depths=depths, dims=dims, **kwargs)
    if pretrained:
        if use_timm_weights:
            timm_model = create_model(name, pretrained=True)
            pretrained_weights = timm_model.state_dict()
            pretrained_weights = _convert_timm_pretrained_weights(pretrained_weights)
        else:
            weights_path = download_checkpoint(_PRETRAINED_WEIGHTS[name])
            pretrained_weights = torch.load(
                weights_path, map_location="cpu", weights_only=True
            )["model"]
        state_dict_shapes = {k: v.shape for k, v in model.state_dict().items()}
        if spatial_strides[0] == 2:
            pretrained_weights["downsample_layers.0.0.weight"] = F.avg_pool2d(
                pretrained_weights["downsample_layers.0.0.weight"], 2, 2
            )
        inflated_pretrained_weights = _inflate_pretrained_weights(
            pretrained_weights, state_dict_shapes
        )
        if spatial_strides[0] == 2:
            pass
        model.load_state_dict(inflated_pretrained_weights, strict=True)
    if headless and not features_only:
        model.head = nn.Identity()
    return model


def convnextv2_3d_atto(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_atto", depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs
    )
    return model


def convnextv2_3d_femto(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_femto", depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs
    )
    return model


def convnextv2_3d_pico(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_pico", depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs
    )
    return model


def convnextv2_3d_nano(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_nano", depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs
    )
    return model


def convnextv2_3d_tiny(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_tiny", depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs
    )
    return model


def convnextv2_3d_base(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_base",
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        **kwargs,
    )
    return model


def convnextv2_3d_large(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_large",
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        **kwargs,
    )
    return model


def convnextv2_3d_huge(**kwargs):
    model = _create_convnextv2_3d(
        name="convnextv2_huge",
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        **kwargs,
    )
    return model


def create_convnextv2_3d(name, **kwargs):
    return globals()[name](**kwargs)


if __name__ == "__main__":
    from skp.toolbox.functions import count_parameters

    x = torch.randn(1, 3, 3, 32, 32)
    model = convnextv2_3d_atto(
        pretrained=True,
        temporal_strides=[1, 1, 1, 1],
        block_kernel_size=(3, 7, 7),
        use_timm_weights=True,
        features_only=True,
    )
    model.set_grad_checkpointing()
    print(model)
    count_parameters(model)
    feature_maps = model(x)
    for fm in feature_maps:
        print(fm.shape)
    decoder_block = DecoderBlock(dim=256 + 64, kernel_size=(3, 7, 7))
    x = torch.randn((1, 256, 32, 32, 32))
    skip = torch.randn((1, 64, 64, 64, 64))
    y = decoder_block(x, skip)
    count_parameters(decoder_block)
    print(y.shape)
