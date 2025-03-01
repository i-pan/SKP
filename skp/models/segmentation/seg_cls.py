import torch
import torch.nn as nn

from timm.layers import BatchNormAct2d
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
            cls_features_dim = self.segmenter.cfg.encoder_channels[-1]
            classifier = []
            if self.cfg.add_conv_head_pre_classifier:
                # for some timm backbones (e.g., EfficientNet), features_only
                # does not have conv_head, so provide option to add it back
                if "efficientnet" in self.cfg.backbone:
                    conv_head_features_dim = 1280
                    conv_head = nn.Conv2d(
                        self.segmenter.cfg.encoder_channels[-1],
                        conv_head_features_dim,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    )
                    norm_act = BatchNormAct2d(
                        num_features=conv_head_features_dim,
                        eps=0.001,
                        momentum=0.1,
                        affine=True,
                        track_running_stats=True,
                        apply_act=True,
                        act_layer=nn.SiLU,
                        inplace=True,
                        drop_layer=None,
                    )
                    classifier.extend([conv_head, norm_act])
                else:
                    raise NotImplementedError(
                        "`add_conv_head_pre_classifier` only supported for EfficientNet"
                    )
                cls_features_dim = conv_head_features_dim
            classifier.extend(
                [
                    get_pool_layer(self.cfg, dim=2),
                    nn.Dropout(p=self.cfg.cls_dropout or 0.0),
                    # can specify different # of classes for segmentation and classification
                    # otherwise, use same # of classes for both
                    nn.Linear(
                        cls_features_dim,
                        self.cfg.cls_num_classes or self.cfg.num_classes,
                    ),
                ]
            )
            self.classifier = nn.Sequential(*classifier)
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
