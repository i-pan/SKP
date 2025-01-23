import torch

from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from torchmetrics import Metric
from typing import Dict

from skp.configs.base import Config


class BaseMetric(Metric):
    """
    Base class for simple classification/regression metrics
    """

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].float())
        self.t.append(batch["y"].float())

    def compute(self):
        raise NotImplementedError


class ScoreBased(BaseMetric):
    """
    Uses model output scores (e.g., logits)
    For example, AUC and AVP
    Can also work with regression tasks, such as MAE
    """

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0)  # (N,C)
        t = torch.cat(self.t, dim=0).cpu()  # (N,1) or (N,C)
        if self.cfg.metric_include_classes:
            assert isinstance(self.cfg.metric_include_classes, list)
            p = p[:, self.cfg.metric_include_classes]
            t = t[:, self.cfg.metric_include_classes]
        assert (
            p.ndim == t.ndim == 2
        ), f"p.ndim [{p.ndim}] and t.ndim [{t.ndim}] must be 2"
        # activation function is not necessarily required, for example for AUC
        if self.cfg.metric_activation_fn == "sigmoid":
            p = p.sigmoid()
        elif self.cfg.metric_activation_fn == "softmax":
            p = p.softmax(dim=1)
        p = p.cpu()
        if p.size(1) == 1:
            # Binary classification (or simple regression)
            return {f"{self.name}_mean": self.metric_func(t, p)}
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            metrics_dict[f"{self.name}{c}"] = self.metric_func(tmp_gt, p[:, c])
        metrics_dict[f"{self.name}_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        return metrics_dict


class ClassBased(BaseMetric):
    """
    Uses class prediction
    For example, accuracy, Kappa score
    This is for multiclass/binary classification, not multilabel classification
    """

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0)  # (N,C)
        t = torch.cat(self.t, dim=0).cpu()  # (N,1) or (N,C)
        assert (
            p.ndim == t.ndim == 2
        ), f"p.ndim [{p.ndim}] and t.ndim [{t.ndim}] must be 2"
        # activation function is not necessarily required
        if self.cfg.metric_activation_fn == "sigmoid" or p.size(1) == 1:
            # p.size(1) = 1 implies binary classification
            p = p.sigmoid()
        elif self.cfg.metric_activation_fn == "softmax":
            p = p.softmax(dim=1)
        p = p.cpu()
        if p.size(1) == 1:
            # binary classification
            threshold = self.cfg.binary_cls_threshold or 0.5
            p = (p >= threshold).float()
            return {f"{self.name}": self.metric_func(t, p)}
        p = torch.argmax(p, dim=1, keepdims=True)
        return {f"{self.name}": self.metric_func(t, p)}


class AUROC(ScoreBased):
    name = "auc"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))


class AVP(ScoreBased):
    name = "avp"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(average_precision_score(y_true=t, y_score=p))


class MAE(ScoreBased):
    name = "mae"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(t - p))


class MSE(ScoreBased):
    name = "mse"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.mean((t - p) ** 2)


class Accuracy(ClassBased):
    name = "accuracy"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (t == p).float().mean()


class Kappa(ClassBased):
    name = "kappa"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(cohen_kappa_score(y1=t, y2=p))


class QWK(ClassBased):
    name = "qwk"

    def metric_func(self, t: torch.Tensor, p: torch.Tensor) -> torch.tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(cohen_kappa_score(y1=t, y2=p, weights="quadratic"))
