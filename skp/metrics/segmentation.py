import numpy as np
import torch
import torchmetrics as tm

from einops import rearrange, reduce
from torch.nn.functional import one_hot
from typing import Dict

from skp.configs import Config


class MulticlassDiceScore(tm.Metric):
    # Each class is mutually exclusive

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"], batch["y"]
        if "mask_present" in batch:
            # ignore samples without a valid mask
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]

        if "pseudolabel" in batch:
            # ignore mask pseudolabels
            p, t = p[~batch["pseudolabel"]], t[~batch["pseudolabel"]]

        num_classes = p.size(1)
        assert self.cfg.activation_fn == "softmax"
        p = p.argmax(1)

        p = one_hot(p, num_classes=num_classes)
        t = one_hot(t, num_classes=num_classes)

        if self.cfg.metric_invert_background:
            # assumes background class is index 0
            p[..., 0] = 1 - p[..., 0]
            t[..., 0] = 1 - t[..., 0]

        if self.cfg.metric_ignore_class0:
            p, t = p[..., 1:], t[..., 1:]

        if p.ndim == 5:
            s1 = "b x y z c -> b c x y z"
            s2 = "b c x y z -> b c"
        elif p.ndim == 4:
            s1 = "b h w c -> b c h w"
            s2 = "b c h w -> b c"
        else:
            raise Exception(f"p is not a valid shape {p.shape}")

        p = rearrange(p, s1).long()
        t = rearrange(t, s1).long()

        intersection = reduce(p * t, s2, "sum")
        denominator = reduce(p + t, s2, "sum")

        dice = (2 * intersection) / denominator

        self.dice_scores.append(dice)

    def compute(self) -> Dict[str, torch.Tensor]:
        dice_scores = torch.cat(self.dice_scores, dim=0)  # (N, C)
        metrics_dict = {}
        for c in range(dice_scores.shape[1]):
            # ignore NaN
            tmp_dice_scores = dice_scores[:, c].cpu().numpy()
            tmp_dice_scores = tmp_dice_scores[~np.isnan(tmp_dice_scores)]
            if self.cfg.metric_ignore_class0:
                c += 1
            metrics_dict[f"dice{c}"] = torch.tensor(np.mean(tmp_dice_scores))
        metrics_dict["dice_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        return metrics_dict


class MultilabelDiceScore(tm.Metric):
    # Pixel can be >1 class

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.thresholds = torch.tensor(self.cfg.metric_thresholds or [0.5])
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"], batch["y"]
        if "mask_present" in batch:
            # ignore samples without a valid mask
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]

        if "pseudolabel" in batch:
            # ignore mask pseudolabels
            p, t = p[~batch["pseudolabel"]], t[~batch["pseudolabel"]]
            
        if self.cfg.metric_labels_to_onehot:
            # if we are converting to onehot and using this metric
            # that means we have a multiclass classification problem
            # but we are using sigmoid act for specific reasons
            t = one_hot(t, num_classes=p.size(1))
            if p.ndim == 5:
                t = rearrange(t, "b x y z c -> b c x y z")
            elif p.ndim == 4:
                t = rearrange(t, "b h w c -> b c h w")

        # t should be one-hot encoded
        assert self.cfg.activation_fn == "sigmoid"
        p = p.sigmoid()

        # if only one class, add channel dimension
        if p.size(1) == 1 and p.ndim == t.ndim + 1:
            t = t.unsqueeze(1)

        assert (
            p.ndim == t.ndim
        ), f"prediction [{p.ndim}] and label [{t.ndim}] tensors should have same # of dimensions, ensure labels are one-hot encoded"

        if self.cfg.loss_params.get("invert_background", False):
            # assumes background class is index 0
            t[:, 0] = 1 - t[:, 0]
            # note, p already calibrated for inverted background
            # so no need to invert p background here

        p = torch.stack([p >= _th for _th in self.thresholds])
        t = torch.stack([t] * len(self.thresholds))
        if p.ndim == 6:
            s = "n b c x y z -> n b c"
        elif p.ndim == 5:
            s = "n b c h w -> n b c"
        else:
            raise Exception(f"p is not a valid shape {p.shape}")

        intersection = reduce(p * t, s, "sum")
        denominator = reduce(p + t, s, "sum")

        dice = (2 * intersection) / denominator

        self.dice_scores.append(dice)

    @staticmethod
    def _dice_ignore_nan(x: torch.Tensor) -> torch.Tensor:
        x = x.cpu().numpy()
        x = x[~np.isnan(x)]
        return torch.tensor(x.mean())

    def compute(self) -> Dict[str, torch.Tensor]:
        dice = torch.cat(
            self.dice_scores, dim=1
        )  # shape = (num_thresholds, num_samples, num_classes)
        dice_over_thresholds = torch.stack(
            [
                torch.stack(
                    [
                        self._dice_ignore_nan(dice[num_t, :, num_c])
                        for num_t in range(dice.shape[0])
                    ]
                )
                for num_c in range(dice.shape[2])
            ]
        )  # shape = (num_classes, num_thresholds)

        best_dice = dice_over_thresholds.amax(dim=1)
        best_thresholds = self.thresholds[
            dice_over_thresholds.argmax(dim=1).cpu().numpy()
        ]
        metrics_dict = {f"dice{idx}": _dice for idx, _dice in enumerate(best_dice)}
        metrics_dict["dice_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        metrics_dict.update(
            {f"th{idx}": _th for idx, _th in enumerate(best_thresholds)}
        )

        return metrics_dict


# class MultilabelDiceScore(tm.Metric):
#     # Pixel can be >1 class

#     def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.cfg = cfg
#         self.add_state("dice_scores", default=[], dist_reduce_fx=None)

#     def update(self, out: Dict, batch: Dict) -> None:
#         p, t = out["logits"], batch["y"]

#         if self.cfg.metric_labels_to_onehot:
#             # if we are converting to onehot and using this metric
#             # that means we have a multiclass classification problem
#             # but we are using sigmoid act for specific reasons
#             t = one_hot(t, num_classes=p.size(1))
#             if p.ndim == 5:
#                 t = rearrange(t, "b x y z c -> b c x y z")
#             elif p.ndim == 4:
#                 t = rearrange(t, "b h w c -> b c h w")

#         # t should be one-hot encoded
#         assert self.cfg.activation_fn == "sigmoid"
#         p = p.sigmoid()
#         p = (p >= 0.5).long()

#         # if only one class, add channel dimension
#         if p.size(1) == 1 and p.ndim == t.ndim + 1:
#             t = t.unsqueeze(1)

#         assert (
#             p.ndim == t.ndim
#         ), f"prediction [{p.ndim}] and label [{t.ndim}] tensors should have same # of dimensions, ensure labels are one-hot encoded"

#         if self.cfg.loss_params.get("invert_background", False):
#             # assumes background class is index 0
#             t[:, 0] = 1 - t[:, 0]
#             # note, p already calibrated for inverted background
#             # so no need to invert p background here

#         if p.ndim == 5:
#             intersection = reduce(p * t, "b c x y z -> b c", "sum")
#             denominator = reduce(p + t, "b c x y z -> b c", "sum")
#         elif p.ndim == 4:
#             intersection = reduce(p * t, "b c h w -> b c", "sum")
#             denominator = reduce(p + t, "b c h w -> b c", "sum")

#         dice = (2 * intersection) / denominator

#         self.dice_scores.append(dice)

#     def compute(self) -> Dict[str, torch.Tensor]:
#         dice_scores = torch.cat(self.dice_scores, dim=0)  # (N, C)
#         metrics_dict = {}
#         for c in range(dice_scores.shape[1]):
#             # ignore NaN
#             tmp_dice_scores = dice_scores[:, c].cpu().numpy()
#             tmp_dice_scores = tmp_dice_scores[~np.isnan(tmp_dice_scores)]
#             metrics_dict[f"ml_dice{c}"] = torch.tensor(np.mean(tmp_dice_scores))
#         metrics_dict["ml_dice_mean"] = torch.stack([v for v in metrics_dict.values()]).mean(0)
#         return metrics_dict
