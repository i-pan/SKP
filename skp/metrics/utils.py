import torch

from sklearn import metrics


def auc(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if len(t.unique()) == 1:
        return torch.tensor(0.5)
    t, p = t.numpy(), p.numpy()
    return torch.tensor(metrics.roc_auc_score(y_true=t, y_score=p))


def avp(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if len(t.unique()) == 1:
        return torch.tensor(0)
    t, p = t.numpy(), p.numpy()
    return torch.tensor(metrics.average_precision_score(y_true=t, y_score=p))


def kappa(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if len(t.unique()) == 1:
        return torch.tensor(0)
    t, p = t.numpy(), p.numpy()
    return torch.tensor(metrics.cohen_kappa_score(y1=t, y2=p))


def qwk(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if len(t.unique()) == 1:
        return torch.tensor(0)
    t, p = t.numpy(), p.numpy()
    return torch.tensor(metrics.cohen_kappa_score(y1=t, y2=p, weights="quadratic"))


def mae(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(t - p))


def mse(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.mean((t - p) ** 2)


def accuracy(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.mean((t == p).float())
