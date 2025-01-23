"""
Custom losses for specific tasks (e.g., Kaggle competitions) that may or may not be
transferrable to other tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


class AgeViewFemaleLoss(nn.Module):
    """
    Part of CheXmask training.
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.weights = params["weights"]

    def forward(self, out: Dict, batch: Dict) -> Dict:
        p, t = out["logits"], batch["y"]
        # only calculate view loss for CheXpert dataset
        # since NIH does not have any laterals
        # to avoid unwanted bias in the model
        valid_view = batch["valid_view"]
        age_loss = F.l1_loss(p[:, 0], t[:, 0].float())
        if valid_view.sum() > 0:
            view_loss = F.cross_entropy(p[valid_view, 1:4], t[valid_view, 1].long())
        else:
            view_loss = torch.tensor(0.0, device=p.device)
        female_loss = F.binary_cross_entropy_with_logits(p[:, 4], t[:, 2].float())
        loss_dict = {
            "age_loss": age_loss * self.weights[0],
            "view_loss": view_loss * self.weights[1],
            "female_loss": female_loss * self.weights[2],
        }
        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict


class L1CELoss(nn.Module):
    """
    Originally intended for bone age regression + classification.
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.l1_weight = params["l1_weight"]
        self.ce_weight = params["ce_weight"]
        self.l1_func = F.smooth_l1_loss if params.get("smooth_l1", False) else F.l1_loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_reg, p_cls, t_reg, t_cls = (
            out["logits0"],
            out["logits1"],
            batch["y"]["reg"],
            batch["y"]["cls"],
        )
        loss_dict = {}
        loss_dict["l1_loss"] = self.l1_func(p_reg, t_reg)
        loss_dict["ce_loss"] = F.cross_entropy(p_cls, t_cls)
        loss_dict["loss"] = (
            loss_dict["l1_loss"] * self.l1_weight
            + loss_dict["ce_loss"] * self.ce_weight
        )
        return loss_dict


class DoubleL1Loss(nn.Module):
    """
    Originally intended for bone age regression + classification.

    Considers outputs from 2 heads:
        1) Regression head (1 class)
        2) Classification head (240 classes, 1 for each month)
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.reg_weight = params["reg_weight"]
        self.cls_weight = params["cls_weight"]
        self.l1_func = F.smooth_l1_loss if params.get("smooth_l1", False) else F.l1_loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_reg, p_cls, t = out["logits0"], out["logits1"], batch["y"]
        # logits -> probabilities
        p_cls = p_cls.softmax(dim=1)
        # probabilties over classes -> bone age scalar
        p_cls = p_cls * torch.arange(p_cls.size(1)).to(p_cls.device)
        p_cls = p_cls.sum(dim=1, keepdims=True)
        loss_dict = {}
        loss_dict["l1_loss_reg"] = self.l1_func(p_reg, t)
        loss_dict["l1_loss_cls"] = self.l1_func(p_cls, t)
        loss_dict["loss"] = (
            self.reg_weight * loss_dict["l1_loss_reg"]
            + self.cls_weight * loss_dict["l1_loss_cls"]
        ) / (self.reg_weight + self.cls_weight)
        return loss_dict


class MURA_BCE_CELoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.bce_weight = params.get("bce_weight", 1.0)
        self.ce_weight = params.get("ce_weight", 1.0)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        p0, t0 = p[:, 0], t[:, 0]
        p1, t1 = p[:, 1:], t[:, 1]
        loss_dict = {}
        loss_dict["bce_loss"] = F.binary_cross_entropy_with_logits(p0, t0.float())
        loss_dict["ce_loss"] = F.cross_entropy(p1, t1.long())
        loss_dict["loss"] = (
            loss_dict["bce_loss"] * self.bce_weight
            + loss_dict["ce_loss"] * self.ce_weight
        )
        return loss_dict


class MammoCancerWithAuxLosses(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.loss_weights = torch.tensor(params.get("loss_weights", [1.0] * 6))
        self.classes = [
            "cancer",
            "biopsy",
            "invasive",
            "difficult",
            "birads",
            "density",
        ]

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        if self.loss_weights.device != p.device:
            self.loss_weights = self.loss_weights.to(p.device)

        p0, t0 = p[:, :4], t[:, :4]  # cancer, biopsy, invasive, difficult_negative_case
        p1, t1 = p[:, [4, 5, 6]], t[:, 4]  # BIRADS - 3 classes
        p2, t2 = p[:, [7, 8, 9, 10]], t[:, 5]  # density - 4 classes
        loss_dict = {}

        bce_loss = F.binary_cross_entropy_with_logits(p0, t0.float(), reduction="none")
        bce_loss = self.loss_weights[:4] * bce_loss
        for i, c in enumerate(self.classes[:4]):
            loss_dict[f"{c}_loss"] = bce_loss[:, i].mean()

        birads_present = t1 != -1  # only present for one site
        if birads_present.sum() == 0:
            birads_loss = torch.tensor(0.0, device=p.device)
        else:
            birads_loss = F.cross_entropy(
                p1[birads_present], t1[birads_present].long(), reduction="none"
            )
        birads_loss = self.loss_weights[4] * birads_loss
        loss_dict["birads_loss"] = birads_loss.mean()

        density_present = t2 != -1  # only present for one site
        if density_present.sum() == 0:
            density_loss = torch.tensor(0.0, device=p.device)
        else:
            density_loss = F.cross_entropy(
                p2[density_present], t2[density_present].long(), reduction="none"
            )
        density_loss = self.loss_weights[5] * density_loss
        loss_dict["density_loss"] = density_loss.mean()

        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict
