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


class ICHSeqMaskedBCELossWithAux(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.aux_weight = params.get("aux_weight", 0.4)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_seq, p_aux = out["logits_seq"], out["aux_logits_seq"]
        t_seq = batch["y_seq"]

        seq_loss = F.binary_cross_entropy_with_logits(p_seq, t_seq, reduction="none")
        aux_loss = F.binary_cross_entropy_with_logits(p_aux, t_seq, reduction="none")

        mask = batch["mask"]
        seq_loss = seq_loss[mask]
        aux_loss = aux_loss[mask]

        # upweight any by 2
        seq_loss = (seq_loss[:, 0] * 2 + seq_loss[:, 1:].sum(1)) / 7.0
        seq_loss = seq_loss.mean()
        aux_loss = (aux_loss[:, 0] * 2 + aux_loss[:, 1:].sum(1)) / 7.0
        aux_loss = aux_loss.mean()

        loss_dict = {"seq_loss": seq_loss, "aux_loss": aux_loss}
        loss_dict["loss"] = seq_loss + self.aux_weight * aux_loss

        return loss_dict


class ICHSeqClsMaskedLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.cls_weight = params.get("cls_weight", 1.0)
        self.seq_weight = params.get("seq_weight", 1.0)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_seq, p_cls = out["logits_seq"], out["logits_cls"]
        t_seq, t_cls = batch["y_seq"], batch["y_cls"]
        seq_loss = F.binary_cross_entropy_with_logits(p_seq, t_seq, reduction="none")
        seq_mask = batch["mask"]
        seq_loss = seq_loss[~seq_mask]
        # upweight any by 2
        seq_loss = seq_loss[:, 0] * 2 + seq_loss[:, 1:].sum(1)
        seq_loss = seq_loss / 7.0
        seq_loss = seq_loss.mean()

        cls_loss = F.binary_cross_entropy_with_logits(p_cls, t_cls, reduction="none")
        # upweight any by 2
        cls_loss = cls_loss[:, 0] * 2 + cls_loss[:, 1:].sum(1)
        cls_loss = cls_loss / 7.0
        cls_loss = cls_loss.mean()

        loss_dict = {"seq_loss": seq_loss, "cls_loss": cls_loss}
        loss_dict["loss"] = self.cls_weight * cls_loss + self.seq_weight * seq_loss
        return loss_dict


class ICHLogLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p = out["logits"]
        t = batch["y"]
        assert p.size(1) == t.size(1) == 6
        loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        # loss.shape = (B, 6)
        sample_weights = torch.ones(loss.size(), device=p.device).float()
        # upweight any class by 2
        sample_weights[:, 0] = 2.0
        loss = loss.reshape(-1) * sample_weights.reshape(-1)
        loss = loss.sum() / sample_weights.sum()
        loss_dict = {"loss": loss}
        return loss_dict


class ICHLogLossSeqWithAux(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.aux_weight = params.get("aux_weight", 0.2)

    @staticmethod
    def calculate_loss(
        p: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        # loss.shape = (B, N, 6)
        B = loss.size(0)
        sample_weights = torch.ones(loss.size(), device=p.device).float()
        # upweight any class by 2
        sample_weights[:, :, 0] = 2.0
        sample_weights = sample_weights.reshape(B, -1)
        loss = loss.reshape(B, -1) * sample_weights
        mask = mask.reshape(B, -1)
        loss = loss[mask].sum() / sample_weights[mask].sum()
        return loss 
        
    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, p_aux = out["logits_seq"], out["aux_logits_seq"]
        t = batch["y_seq"]
        mask = batch["mask"].unsqueeze(2).expand(-1, -1, 6)
        assert p.size(2) == t.size(2) == 6
        seq_loss = self.calculate_loss(p, t, mask)
        aux_loss = self.calculate_loss(p_aux, t, mask)
        loss_dict = {}
        loss_dict["seq_loss"] = seq_loss
        loss_dict["aux_loss"] = aux_loss
        loss_dict["loss"] = seq_loss + self.aux_weight * aux_loss
        return loss_dict
