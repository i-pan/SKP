import torch
import torch.nn as nn

from typing import Dict, List, Union
from skp.configs import Config
from skp.models.pooling import get_pool_layer
from skp.models.segmentation.unet import Net as Segmenter
from skp.models.segmentation.decoders.unet import Conv2dAct


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.segmenter = Segmenter(cfg)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            Conv2dAct(
                self.cfg.num_classes,
                self.cfg.cls_head_hidden_size or 64,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer="bn",
                activation="ReLU",
            ),
            get_pool_layer(self.cfg, dim=2),
            nn.Dropout(p=self.cfg.cls_dropout or 0.0),
            # can specify different # of classes for segmentation and classification
            # otherwise, use same # of classes for both
            nn.Linear(
                self.cfg.cls_head_hidden_size or 64,
                self.cfg.cls_num_classes or self.cfg.num_classes,
            ),
        )

        self.criterion = None

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        seg_out = self.segmenter(batch["seg"], return_loss=False, return_features=return_features)
        out = {}
        out["seg"] = seg_out
        out["cls"] = {"logits": self.classifier(seg_out["logits"])}

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
