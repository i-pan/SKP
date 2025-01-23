import torch
import torch.nn as nn

from typing import Dict, List, Union
from skp.configs import Config
from skp.models.pooling import get_pool_layer
from skp.models.segmentation.base import Net as Segmenter
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.segmenter = Segmenter(cfg)
        if not self.cfg.max_pixel_classifier:
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
        else:
            if self.cfg.cls_num_classes:
                assert (
                    self.cfg.cls_num_classes == self.cfg.num_classes
                ), "If using max pixel for classification, number of segmentation and classification classes should be the same"

        if self.cfg.load_pretrained_segmenter:
            print(
                f"Loading pretrained segmenter from {self.cfg.load_pretrained_segmenter} ..."
            )
            weights = torch_load_weights(self.cfg.load_pretrained_segmenter)
            weights = filter_weights_by_prefix(weights, "model.")
            self.segmenter.load_state_dict(weights)

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

        if self.cfg.max_pixel_classifier:
            cls_logits = seg_out["logits"].amax((2, 3))
        else:
            cls_logits = self.classifier(features[-1])
        out["cls"] = {"logits": cls_logits}

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
