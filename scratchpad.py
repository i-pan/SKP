import torch
import torch.nn as nn

from einops import rearrange
from timm import create_model

from transformer_encoder_layer import TransformerEncoderLayerWithAttnWeights


device = "cuda"

x0 = torch.randn((2, 32, 3, 224, 224))
b, n, c, h, w = x0.shape

backbone = "tf_efficientnetv2_s"

net = create_model(backbone, pretrained=True, global_pool="avg", num_classes=0).to(
    device
)

x = rearrange(x0, "b n c h w -> (b n) c h w")

x = x.to(device)
features = net(x)
features = rearrange(features, "(b n) d -> b n d", b=b)
print(features.shape)

cls_token = nn.Parameter(torch.randn(1, 1, features.shape[-1]))
cls_token = cls_token.expand(features.shape[0], -1, -1).to(device)
features = torch.cat([cls_token, features], dim=1)

transformer = TransformerEncoderLayerWithAttnWeights(
    d_model=features.shape[-1],
    nhead=16,
    dim_feedforward=features.shape[-1],
    dropout=0.1,
    activation="gelu",
    batch_first=True,
    norm_first=True,
    device=device,
)

out, wts = transformer(features, return_attn_weights=True)
print(wts[0, 0, 1:])


from skp.models.encoders3d import get_encoder
from skp.configs import Config
from skp.toolbox.functions import count_parameters
import torch 

cfg = Config()
cfg.backbone = "convnextv2_3d_atto"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.features_only = False
cfg.dim0_strides = [2, 2, 2, 2, 2]
model = get_encoder(cfg)
x = torch.randn((2, 3, 64, 64, 64))
out = model(x)
print(out.shape)
count_parameters(model)