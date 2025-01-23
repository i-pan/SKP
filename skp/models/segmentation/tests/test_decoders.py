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
cfg.image_height = 512
cfg.image_width = 512

net = import_module(f"skp.models.{cfg.model}").Net(cfg)
x = torch.randn((2, cfg.num_input_channels, cfg.image_height, cfg.image_width))
out = net({"x": x})
print(out["logits"].shape)
print([o.shape for o in out["aux_logits"]])

cfg.decoder_type = "DeepLabV3PlusDecoder"
cfg.decoder_out_channels = 256
cfg.atrous_rates = (6, 12, 18)
cfg.deep_supervision_num_levels = 1

net = import_module(f"skp.models.{cfg.model}").Net(cfg)
out = net({"x": x})
print(out["logits"].shape)
print([o.shape for o in out["aux_logits"]])

cfg.decoder_type = "UperNetDecoder"
cfg.deep_supervision_num_levels = 2

net = import_module(f"skp.models.{cfg.model}").Net(cfg)
out = net({"x": x})
print(out["logits"].shape)
print([o.shape for o in out["aux_logits"]])
