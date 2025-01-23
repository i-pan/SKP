import torch
import torch.nn as nn

from typing import Dict, List, Union
from skp.configs import Config
from skp.models.segmentation.unet import Net as Segmenter


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.segmenter = Segmenter(cfg)
        if self.cfg.cls_num_classes:
            assert (
                self.cfg.cls_num_classes == self.cfg.num_classes
            ), "If using max pixel for classification, number of segmentation and classification classes should be the same"

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
        out["cls"] = {"logits": seg_out["logits"].amax((2, 3))}

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
