import torch
import torch.nn as nn

from typing import Dict, List, Union
from skp.configs import Config
from skp.models.pooling import get_pool_layer
from skp.models.segmentation.rad_dino_upernet import Net as Segmenter


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.segmenter = Segmenter(cfg)
        self.classifier = nn.Sequential(
            get_pool_layer(self.cfg, dim=2),
            nn.Dropout(p=self.cfg.cls_dropout or 0.0),
            # can specify different # of classes for segmentation and classification
            # otherwise, use same # of classes for both
            nn.Linear(
                self.segmenter.cfg.encoder_channels[-1],
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
        seg_out = self.segmenter(batch["seg"], return_loss=False, return_features=True)
        features = seg_out["features"] if return_features else seg_out.pop("features")
        out = {}
        out["seg"] = seg_out
        out["cls"] = {"logits": self.classifier(features[-1])}

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out
    
    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
