import torch

from importlib import import_module
from skp.configs import Config


cfg = Config()
cfg.model = "segmentation.base"
cfg.backbone = "tf_efficientnetv2_s"
cfg.decoder_type = "UnetDecoder"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.seg_dropout = 0.1
cfg.decoder_out_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.deep_supervision = True
cfg.deep_supervision_num_levels = 2
cfg.num_classes = 4
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.image_height = 1024
cfg.image_width = 1024

net = import_module(f"skp.models.{cfg.model}").Net(cfg)
x = torch.randn((2, cfg.num_input_channels, cfg.image_height, cfg.image_width))
out = net({"x": x}, return_decoder_output=True)
print([o.shape for o in out["decoder_output"]])

cfg.decoder_out_channels = [256, 128, 64, 32]
cfg.decoder_n_blocks = 4

net = import_module(f"skp.models.{cfg.model}").Net(cfg)
x = torch.randn((2, cfg.num_input_channels, cfg.image_height, cfg.image_width))
out = net({"x": x}, return_decoder_output=True)
print([o.shape for o in out["decoder_output"]])

cfg.decoder_out_channels = [256, 128, 64]
cfg.decoder_n_blocks = 3

net = import_module(f"skp.models.{cfg.model}").Net(cfg)
x = torch.randn((2, cfg.num_input_channels, cfg.image_height, cfg.image_width))
out = net({"x": x}, return_decoder_output=True)
print([o.shape for o in out["decoder_output"]])
