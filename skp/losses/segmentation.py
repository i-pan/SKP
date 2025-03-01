"""
Commonly used losses for segmentation tasks.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from typing import Dict, List, Optional, Tuple, Union

from skp.losses import _check_shapes_equal


def _weight_func(ground_truth: torch.Tensor, weight_type: str = "square"):
    if not torch.is_floating_point(ground_truth):
        ground_truth = ground_truth.float()
    if weight_type == "square":
        return torch.reciprocal(ground_truth**2)
    elif weight_type == "simple":
        return torch.reciprocal(ground_truth)
    else:
        raise Exception(f"Invalid weight_type: {weight_type}")


def _dice_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    pred_power: float = 1.0,
    smooth: float = 1e-5,
    activation_fn: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    compute_method: str = "per_sample",
    generalized: bool = False,
    weight_type: str = "square",  # only if generalized=True
) -> torch.Tensor:
    assert compute_method in {
        "per_sample",
        "per_batch",
    }, f"Invalid compute_method: {compute_method}"

    assert x.shape == y.shape, f"x.shape [{x.shape}] does not equal y.shape [{y.shape}]"
    if not torch.is_floating_point(y):
        y = y.float()

    # y should be one-hot encoded
    if activation_fn == "sigmoid":
        x = x.sigmoid()
    elif activation_fn == "softmax":
        x = x.softmax(dim=1)
    elif activation_fn is None:
        pass
    else:
        raise Exception(f"Invalid activation_fn: {activation_fn}")

    if compute_method == "per_sample":
        # compute dice score per sample, then average
        if x.ndim == 5:  # 3D
            s = "b c x y z -> b c"
        elif x.ndim == 4:  # 2D
            s = "b c h w -> b c"
    elif compute_method == "per_batch":
        # aggregate all pixels in batch, compute dice
        if x.ndim == 5:
            s = "b c x y z -> 1 c"
        elif x.ndim == 4:
            s = "b c h w -> 1 c"

    intersection = reduce(x * y, s, "sum")

    x = reduce(x.pow(pred_power), s, "sum")
    # y is 0 or 1 so raising to pred_power does nothing
    y = reduce(y, s, "sum")

    denominator = x + y

    if generalized:
        # From: https://docs.monai.io/en/stable/losses.html#generalizeddiceloss
        w = _weight_func(y, weight_type)
        infs = torch.isinf(w)
        if compute_method == "per_batch":
            w[infs] = 0.0
            w = w + infs * torch.max(w)
        elif compute_method == "per_sample":
            w[infs] = 0.0
            max_values = torch.max(w, dim=1)[0].unsqueeze(dim=1)
            w = w + infs * max_values
        intersection *= w
        denominator *= w

    dice_loss = -(2.0 * intersection + smooth) / (denominator + smooth)

    if class_weights is not None:
        dice_loss = dice_loss * class_weights.to(dice_loss.device)
        dice_loss = reduce(dice_loss, "b c -> b", "sum")
        dice_loss = dice_loss / class_weights.sum()

    return dice_loss.mean()


def _tversky_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    smooth: float = 1e-5,
    activation_fn: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    compute_method: str = "per_sample",
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    assert compute_method in {
        "per_sample",
        "per_batch",
    }, f"Invalid compute_method: {compute_method}"
    if not torch.is_floating_point(y):
        y = y.float()

    # y should be one-hot encoded
    if activation_fn == "sigmoid":
        x = x.sigmoid()
    elif activation_fn == "softmax":
        x = x.softmax(dim=1)

    p0 = x
    p1 = 1 - p0
    g0 = y
    g1 = 1 - g0

    if compute_method == "per_sample":
        # compute dice score per sample, then average
        if x.ndim == 5:  # 3D
            s = "b c x y z -> b c"
        elif x.ndim == 4:  # 2D
            s = "b c h w -> b c"
    elif compute_method == "per_batch":
        # aggregate all pixels in batch, compute dice
        if x.ndim == 5:
            s = "b c x y z -> 1 c"
        elif x.ndim == 4:
            s = "b c h w -> 1 c"

    tp = reduce(p0 * g0, s, "sum")
    fp = alpha * reduce(p0 * g1, s, "sum")
    fn = beta * reduce(p1 * g0, s, "sum")

    numerator = tp + smooth
    denominator = tp + fp + fn + smooth

    score = 1.0 - numerator / denominator

    if class_weights is not None:
        score = score * class_weights.to(score.device)
        score = reduce(score, "b c -> b", "sum")
        score = score / class_weights.sum()

    return score.mean()


def _sigmoid_focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    From:
    https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/losses/focal_loss.py

    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    # computing binary cross entropy with logits
    # equivalent to F.binary_cross_entropy_with_logits(input, target, reduction='none')
    # see also https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp#L363
    loss: torch.Tensor = input - input * target - F.logsigmoid(input)

    # sigmoid(-i) if t==1; sigmoid(i) if t==0 <=>
    # 1-sigmoid(i) if t==1; sigmoid(i) if t==0 <=>
    # 1-p if t==1; p if t==0 <=>
    # pfac, that is, the term (1 - pt)
    invprobs = F.logsigmoid(-input * (target * 2 - 1))  # reduced chance of overflow
    # (pfac.log() * gamma).exp() <=>
    # pfac.log().exp() ^ gamma <=>
    # pfac ^ gamma
    loss = (invprobs * gamma).exp() * loss

    if alpha is not None:
        # alpha if t==1; (1-alpha) if t==0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss

    return loss


class _SegmentationLoss(nn.Module):
    @staticmethod
    def get_inputs(out: Dict, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        if "mask_present" in batch:
            # ignore samples without a valid mask
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]
        return p, t

    def format_labels(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.ignore_background and self.invert_background:
            raise Exception(
                "ignore_background and invert_background cannot both be True"
            )
        if self.invert_background:
            if hasattr(self, "activation_fn") and self.activation_fn == "softmax":
                raise Exception(
                    "invert_background and softmax activation should not be used together"
                )
        t = t.clone()
        if self.convert_labels_to_onehot:
            num_classes = p.size(1)
            if self.ignore_background:
                num_classes += 1
            t = F.one_hot(t.long(), num_classes=num_classes).float()
            if p.ndim == 5:
                t = rearrange(t, "b x y z c -> b c x y z")
            elif p.ndim == 4:
                t = rearrange(t, "b h w c -> b c h w")
        # if only one class
        if p.size(1) == 1 and p.ndim == t.ndim + 1:
            t = t.unsqueeze(1)  # add channel dim
        if self.invert_background:
            t[:, 0] = 1 - t[:, 0]
        if self.ignore_background:
            t = t[:, 1:]
        if self.resize_ground_truth is not None:
            t = F.interpolate(
                t.float(), size=self.resize_ground_truth, mode="nearest"
            ).long()
        return t


class DiceLoss(_SegmentationLoss):
    def __init__(self, params: Dict):
        super().__init__()
        params = copy.deepcopy(params)
        self.convert_labels_to_onehot = params.pop("convert_labels_to_onehot", False)
        self.resize_ground_truth = params.pop("resize_ground_truth", None)
        self.invert_background = params.pop("invert_background", False)
        self.ignore_background = params.pop("ignore_background", False)
        if "class_weights" in params:
            params["class_weights"] = torch.tensor(params["class_weights"])
        else:
            params["class_weights"] = None
        if params.get("activation_fn", None) is None:
            print(
                "WARN: No activation function provided for Dice loss.",
                "It is recommended to provide an activation function for improved training.",
            )
        self.loss_args = params

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        loss_dict = {"loss": _dice_loss(p, t, **self.loss_args)}
        return loss_dict


class DiceBCELoss(DiceLoss):
    def __init__(self, params: Dict):
        params = copy.deepcopy(params)
        self.dice_weight = params.pop("dice_weight", 1.0)
        self.bce_weight = params.pop("bce_weight", 1.0)
        self.alpha = params.pop("alpha", None)
        super().__init__(params)
        assert self.alpha is None or 0 < self.alpha < 1

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        dice_loss = _dice_loss(p, t, **self.loss_args)
        bce_loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        if self.alpha is not None:
            # alpha if t==1; (1-alpha) if t==0
            alpha_factor = t * self.alpha + (1 - t) * (1 - self.alpha)
            bce_loss = alpha_factor * bce_loss
        n, c = bce_loss.shape[:2]
        bce_loss = bce_loss.reshape(n, c, -1).mean(dim=2)
        if self.loss_args.get("class_weights", None) is not None:
            bce_loss = bce_loss * self.loss_args["class_weights"]
        bce_loss = bce_loss.mean()
        loss_dict = {"dice_loss": dice_loss, "bce_loss": bce_loss}
        loss_dict["loss"] = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return loss_dict


class DiceFocalLoss(DiceLoss):
    def __init__(self, params: Dict):
        params = copy.deepcopy(params)
        self.dice_weight = params.pop("dice_weight", 1.0)
        self.focal_weight = params.pop("focal_weight", 1.0)
        self.gamma = params.pop("gamma", 2.0)
        self.alpha = params.pop("alpha", None)
        self.scale = params.pop("scale", 1.0)
        super().__init__(params)
        assert self.alpha is None or 0 < self.alpha < 1

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        dice_loss = _dice_loss(p, t, **self.loss_args)
        focal_loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        n, c = focal_loss.shape[:2]
        focal_loss = focal_loss.reshape(n, c, -1).mean(dim=2)
        if self.loss_args.get("class_weights", None) is not None:
            focal_loss = focal_loss * self.loss_args["class_weights"]
        focal_loss = focal_loss.mean() * self.scale
        loss_dict = {"dice_loss": dice_loss, "focal_loss": focal_loss}
        loss_dict["loss"] = (
            self.dice_weight * dice_loss + self.focal_weight * focal_loss
        )
        return loss_dict


class TverskyLoss(DiceLoss):
    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        loss_dict = {"loss": _tversky_loss(p, t, **self.loss_args)}
        return loss_dict


class FocalLoss(_SegmentationLoss):
    def __init__(self, params: Dict):
        super().__init__()
        params = copy.deepcopy(params)
        self.gamma = params.get("gamma", 2.0)
        self.alpha = params.get("alpha", None)
        self.convert_labels_to_onehot = params.get("convert_labels_to_onehot", False)
        self.resize_ground_truth = params.get("resize_ground_truth", None)
        self.invert_background = params.get("invert_background", False)
        self.ignore_background = params.get("ignore_background", False)
        self.batch_size = params.get("batch_size", None)
        self.scale = params.get("scale", 1.0)
        assert not (self.invert_background and self.ignore_background)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = self.format_labels(p, t)
        _check_shapes_equal(p, t)
        loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        if "sample_weight" in batch:
            loss = loss.flatten(1) * batch["sample_weight"].unsqueeze(1)
        return {"loss": self.scale * loss.mean()}


class FocalLossMemoryEfficient(FocalLoss):
    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = self.get_inputs(out, batch)
        t = t.unsqueeze(1)
        # t is NOT one-hot encoded
        # t contains class indices where 0 is background
        num_classes = p.shape[1]
        if self.ignore_background:
            # ignore 0 class, need num_classes + 1 to include last class index
            target_classes = torch.arange(1, num_classes + 1, device=p.device)
        else:
            target_classes = torch.arange(num_classes, device=p.device)
        if p.ndim == 5:
            target_classes = target_classes.view(1, num_classes, 1, 1, 1)
        else:
            target_classes = target_classes.view(1, num_classes, 1, 1)
        # if self.batch_size is not None and p.shape[0] > self.batch_size:
        #     loss = []
        #     for i in range(0, p.shape[0], self.batch_size):
        #         loss.append(
        #             _sigmoid_focal_loss(
        #                 p[i : i + self.batch_size],
        #                 (t[i : i + self.batch_size] == target_classes).float(),
        #                 gamma=self.gamma,
        #                 alpha=self.alpha,
        #             ).mean()
        #         )
        #     loss = torch.stack(loss).mean()
        # else:
        t = (t == target_classes).float()
        if self.invert_background:
            # assume 1st channel is background
            t[:, 0] = 1 - t[:, 0]
        loss = _sigmoid_focal_loss(p, t, gamma=self.gamma, alpha=self.alpha)
        loss = loss.mean()
        # loss = 0
        # for i in range(p.shape[1]):
        #     loss += _sigmoid_focal_loss(
        #         p[:, i], (t == i + 1).float(), gamma=self.gamma, alpha=self.alpha
        #     ).mean()
        # loss = loss / p.shape[1]
        return {"loss": self.scale * loss}


class DeepSupervision(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        params = copy.deepcopy(params)
        self.deep_supervision_weights = params.pop("deep_supervision_weights")
        loss_name = params.pop("loss_name")
        self.loss = globals()[loss_name](params)

    @staticmethod
    def downsample_ground_truth(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            F.interpolate(
                y.unsqueeze(1).float() if y.ndim == logits.ndim - 1 else y.float(),
                size=logits.shape[2:],
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )

    def forward(
        self,
        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        # full resolution loss
        loss_dict["loss0"] = self.loss(out, batch)["loss"]
        # downsampled maps from earlier layers
        # auxiliary losses
        for idx, logits in enumerate(out["aux_logits"]):
            downsampled_y = self.downsample_ground_truth(logits, batch["y"])
            tmp_batch = {"y": downsampled_y}
            if "mask_present" in batch:
                tmp_batch["mask_present"] = batch["mask_present"]
            loss_dict[f"loss{idx + 1}"] = self.loss({"logits": logits}, tmp_batch)[
                "loss"
            ]

        loss_dict["loss"] = torch.stack(
            [
                v * self.deep_supervision_weights[int(i.replace("loss", ""))]
                for i, v in loss_dict.items()
            ]
        ).sum(0)

        return loss_dict
