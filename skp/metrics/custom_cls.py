import numpy as np 
import torch

from collections import defaultdict
from einops import rearrange
from sklearn.metrics import log_loss, roc_auc_score, cohen_kappa_score
from torchmetrics import Metric
from typing import Callable, Dict, Tuple

from skp.configs import Config
from skp.metrics.classification import BaseMetric


class MAE_Accuracy(Metric):
    """
    Originally intended for bone age regression + classification
    """

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p_reg", default=[], dist_reduce_fx=None)
        self.add_state("t_reg", default=[], dist_reduce_fx=None)
        self.add_state("p_cls", default=[], dist_reduce_fx=None)
        self.add_state("t_cls", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p_reg.append(out["logits0"].float())
        self.t_reg.append(batch["y"]["reg"].float())
        self.p_cls.append(out["logits1"].float())
        self.t_cls.append(batch["y"]["cls"])

    def compute(self) -> Dict[str, torch.Tensor]:
        p_reg = torch.cat(self.p_reg, dim=0)
        t_reg = torch.cat(self.t_reg, dim=0)
        p_cls = torch.cat(self.p_cls, dim=0)
        t_cls = torch.cat(self.t_cls, dim=0)

        metrics_dict = {}
        metrics_dict["mae_mean"] = torch.mean(torch.abs(p_reg - t_reg))
        p_cls = p_cls.argmax(dim=1)
        metrics_dict["accuracy"] = (p_cls == t_cls).float().mean()
        metrics_dict["accuracy_offby1"] = ((p_cls - t_cls).abs() <= 1).float().mean()
        metrics_dict["accuracy_offby2"] = ((p_cls - t_cls).abs() <= 2).float().mean()
        return metrics_dict


class DoubleMAE(Metric):
    """
    Originally intended for bone age regression + classification

    Computes MAE using output from both regression and classification heads
    as well as a simple average of the 2 outputs
    """

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p_reg", default=[], dist_reduce_fx=None)
        self.add_state("p_cls", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p_reg.append(out["logits0"].float())
        self.p_cls.append(out["logits1"].float())
        self.t.append(batch["y"].float())

    def compute(self) -> Dict[str, torch.Tensor]:
        p_reg = torch.cat(self.p_reg, dim=0)
        p_cls = torch.cat(self.p_cls, dim=0)
        p_cls = p_cls.softmax(dim=1)
        p_cls = p_cls * torch.arange(p_cls.size(1)).to(p_cls.device)
        p_cls = p_cls.sum(1, keepdims=True)
        t = torch.cat(self.t, dim=0)

        metrics_dict = {}
        metrics_dict["mae_reg"] = torch.mean(torch.abs(p_reg - t))
        metrics_dict["mae_cls"] = torch.mean(torch.abs(p_cls - t))
        metrics_dict["mae_regcls"] = torch.mean(torch.abs((p_reg + p_cls) / 2 - t))
        return metrics_dict


class MAE_AUROC(BaseMetric):
    """
    For training chest X-ray models on:
        1- Age, 2- View (AP/PA/Lateral), 3- Sex (Male/Female)
    """

    def auc(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        p_age, t_age = p[:, 0], t[:, 0]
        p_view, t_view = p[:, 1:4], t[:, 1]
        p_fem, t_fem = p[:, 4], t[:, 2]
        metrics_dict = {"mae_age": torch.abs(p_age - t_age).mean()}
        for idx in range(3):
            metrics_dict[f"auc_view{idx}"] = self.auc(
                (t_view == idx).float(), p_view[:, idx]
            )
        metrics_dict["auc_female"] = self.auc(t_fem, p_fem)
        return metrics_dict


class MAE_Accuracy_AUROC(BaseMetric):
    """
    For training chest X-ray models on:
        1- Age, 2- View (AP/PA/Lateral), 3- Sex (Male/Female)
    """

    def auc(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        p_age, t_age = p[:, 0], t[:, 0]
        p_view, t_view = p[:, 1:4], t[:, 1]
        p_fem, t_fem = p[:, 4], t[:, 2]
        metrics_dict = {"mae_age": torch.abs(p_age - t_age).mean()}
        metrics_dict["acc_view"] = ((p_view.argmax(dim=1) == t_view).float()).mean()
        metrics_dict["auc_female"] = self.auc(t_fem, p_fem)
        return metrics_dict


class RANZCR_AUROC(BaseMetric):
    """
    Match AUROC metric in the RANZCR CLiP Kaggle challenge.
    We predict 3 additional targets (CVC present, NGT present, ETT present)
    which are not part of the original evaluation.
    This metric will calculate the average of the 11 AUROCs
    which are used in the challenge.
    """

    def auc(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        metrics_dict = {f"auc{c}": self.auc(t[:, c], p[:, c]) for c in range(p.size(1))}
        # exclude last 3 classes
        metrics_dict["auc_mean11"] = torch.stack(
            [metrics_dict[f"auc{i}"] for i in range(11)]
        ).mean()
        return metrics_dict


class MURA_AUROC(Metric):
    """
    Aggregate predictions over images to compute study-level AUC in addition
    to image-level AUC.
    """

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("study_index", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].float())
        self.t.append(batch["y"])
        if not self.cfg.metric_study_level:
            self.study_index.append(batch["study_index"])

    def auc(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))

    def aggregate_mean_or_max(
        self,
        p_dict: Dict[int, torch.Tensor],
        t_dict: Dict[int, torch.Tensor],
        agg_func: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert agg_func in {
            torch.mean,
            torch.amax,
        }, "only torch.mean and torch.amax are valid aggregation functions"
        p_agg = torch.stack(
            [agg_func(torch.stack(p_dict[i]), dim=0) for i in p_dict.keys()]
        )
        # labels should be same for all samples in group
        # so mean or max doesn't matter
        t_agg = torch.stack(
            [agg_func(torch.stack(t_dict[i]), dim=0) for i in t_dict.keys()]
        )
        return p_agg, t_agg

    def compute(self, return_data: bool = False) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        if self.cfg.metric_study_level:
            metrics_dict = {"auc_study": self.auc(t[:, 0], p[:, 0])}
            if t.size(1) == 2:
                # if exam_type is present
                metrics_dict["exam_accuracy"] = (
                    (p[:, 1:].argmax(dim=1) == t[:, 1]).float()
                ).mean()
        else:
            study_index = torch.cat(self.study_index, dim=0).cpu()
            metrics_dict = {"auc_image": self.auc(t[:, 0], p[:, 0])}
            # aggregate predictions over images to compute study-level AUC
            p_dict, t_dict = defaultdict(list), defaultdict(list)
            for each_p, each_t, each_index in zip(p, t, study_index):
                p_dict[each_index.item()].append(each_p)
                t_dict[each_index.item()].append(each_t)
            p_mean, t_mean = self.aggregate_mean_or_max(p_dict, t_dict, torch.mean)
            metrics_dict["auc_study_mean"] = self.auc(t_mean[:, 0], p_mean[:, 0])
            p_max, t_max = self.aggregate_mean_or_max(p_dict, t_dict, torch.amax)
            metrics_dict["auc_study_max"] = self.auc(t_max[:, 0], p_max[:, 0])
            metrics_dict["auc_study"] = torch.stack(
                [metrics_dict["auc_study_mean"], metrics_dict["auc_study_max"]]
            ).amax()
            if t.size(1) == 2:
                # if exam_type is present
                metrics_dict["exam_accuracy_image"] = (
                    (p[:, 1:].argmax(dim=1) == t[:, 1]).float()
                ).mean()

                metrics_dict["exam_accuracy_study"] = (
                    (p_mean[:, 1:].argmax(dim=1) == t_mean[:, 1]).float()
                ).mean()

        if return_data:
            metrics_dict["p"] = p
            metrics_dict["t"] = t
            if not self.cfg.metric_study_level:
                metrics_dict["p_mean"] = p_mean
                metrics_dict["t_mean"] = t_mean
                metrics_dict["p_max"] = p_max
                metrics_dict["t_max"] = t_max

        return metrics_dict


class MURA_AUROC_Kappa(MURA_AUROC):
    """
    Aggregate predictions over images to compute study-level AUC/Kappa in addition
    to image-level AUC/Kappa.
    """

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(cfg, dist_sync_on_step)

        self.thresholds = self.cfg.metric_thresholds or [0.5]

    def kappa(
        self, t: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(t.unique()) == 1:
            return torch.tensor(0), torch.tensor(0)
        t, p = t.numpy(), p.numpy()
        kappa_list = []
        for threshold in self.thresholds:
            tmp_p = (p >= threshold).astype(int)
            kappa_list.append(torch.tensor(cohen_kappa_score(y1=t, y2=tmp_p)))
        kappa_list = torch.stack(kappa_list, dim=0)
        best_kappa, best_threshold = (
            kappa_list.amax(dim=0),
            self.thresholds[kappa_list.argmax(dim=0).item()],
        )

        return best_kappa, torch.tensor(best_threshold)

    def compute(self) -> Dict[str, torch.Tensor]:
        metrics_dict = super().compute(return_data=True)
        p, t = metrics_dict.pop("p"), metrics_dict.pop("t")
        p_mean, t_mean = (
            metrics_dict.pop("p_mean", None),
            metrics_dict.pop("t_mean", None),
        )
        p_max, t_max = metrics_dict.pop("p_max", None), metrics_dict.pop("t_max", None)

        if self.cfg.metric_study_level:
            kap, thresh = self.kappa(t[:, 0], p[:, 0])
            metrics_dict["kappa_study"] = kap
            metrics_dict["kappa_th_study"] = thresh
        else:
            kap, thresh = self.kappa(t[:, 0], p[:, 0])
            metrics_dict["kappa_image"] = kap
            metrics_dict["kappa_th_image"] = thresh

            kap, thresh = self.kappa(t_mean[:, 0], p_mean[:, 0])
            metrics_dict["kappa_study_mean"] = kap
            metrics_dict["kappa_th_study_mean"] = thresh

            kap, thresh = self.kappa(t_max[:, 0], p_max[:, 0])
            metrics_dict["kappa_study_max"] = kap
            metrics_dict["kappa_th_study_max"] = thresh

            metrics_dict["kappa_study"] = torch.stack(
                [metrics_dict["kappa_study_mean"], metrics_dict["kappa_study_max"]]
            ).amax()

        return metrics_dict


class MURA_AUROC_V2(MURA_AUROC):
    """
    Aggregate predictions over images to compute study-level AUC in addition
    to image-level AUC.

    This metric is for model which predicts abnormal_<exam_type> classes
    instead of 1 abnormal class +/- exam_type label.

    Overall abnormal prediction is the max over the individual
    abnormal_<exam_type> classes.
    """

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        metrics_dict = {}
        if self.cfg.metric_study_level:
            for c in range(p.size(1)):
                metrics_dict[f"auc{c}"] = self.auc(t[:, c], p[:, c])
            # use max abnormal_<exam_type> prediction as overall prediction
            metrics_dict["auc_study"] = self.auc(t.amax(dim=1), p.amax(dim=1))
        else:
            study_index = torch.cat(self.study_index, dim=0).cpu()
            for c in range(p.size(1)):
                metrics_dict[f"auc{c}"] = self.auc(t[:, c], p[:, c])
            # use max abnormal_<exam_type> prediction as overall prediction
            metrics_dict["auc_image"] = self.auc(t.amax(dim=1), p.amax(dim=1))

            # aggregate predictions over images to compute study-level AUC
            p_dict, t_dict = defaultdict(list), defaultdict(list)
            for each_p, each_t, each_index in zip(p, t, study_index):
                p_dict[each_index.item()].append(each_p)
                t_dict[each_index.item()].append(each_t)

            # mean over images
            p_mean = torch.stack(
                [torch.stack(p_dict[i]).mean(dim=0) for i in p_dict.keys()]
            )
            t_mean = torch.stack(
                [torch.stack(t_dict[i]).mean(dim=0) for i in t_dict.keys()]
            )
            for c in range(p.size(1)):
                metrics_dict[f"auc{c}_study_mean"] = self.auc(
                    t_mean[:, c], p_mean[:, c]
                )
            metrics_dict["auc_study_mean"] = self.auc(
                t_mean.amax(dim=1), p_mean.amax(dim=1)
            )

            # max over images
            p_max = torch.stack(
                [torch.stack(p_dict[i]).amax(dim=0) for i in p_dict.keys()]
            )
            t_max = torch.stack(
                [torch.stack(t_dict[i]).amax(dim=0) for i in t_dict.keys()]
            )
            for c in range(p.size(1)):
                metrics_dict[f"auc{c}_study_mean"] = self.auc(t_max[:, c], p_max[:, c])

            metrics_dict["auc_study_max"] = self.auc(
                t_max.amax(dim=1), p_max.amax(dim=1)
            )

            # take max of mean/max aggregation
            metrics_dict["auc_study"] = torch.stack(
                [metrics_dict["auc_study_mean"], metrics_dict["auc_study_max"]]
            ).amax()

        return metrics_dict


class MURA_AUROC_V3(MURA_AUROC):
    """
    Same as MURA_AUROC_V2 but with additional overall abnormal class.
    """

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        num_classes = p.size(1)
        metrics_dict = {}
        if self.cfg.metric_study_level:
            for c in range(num_classes - 1):
                metrics_dict[f"auc{c}"] = self.auc(t[:, c], p[:, c])
            # use max abnormal_<exam_type> prediction as overall prediction
            metrics_dict["auc_study"] = self.auc(t[:, -1], p[:, -1])
        else:
            study_index = torch.cat(self.study_index, dim=0).cpu()
            for c in range(num_classes - 1):
                metrics_dict[f"auc{c}"] = self.auc(t[:, c], p[:, c])
            # use max abnormal_<exam_type> prediction as overall prediction
            metrics_dict["auc_image"] = self.auc(t[:, -1], p[:, -1])

            # aggregate predictions over images to compute study-level AUC
            p_dict, t_dict = defaultdict(list), defaultdict(list)
            for each_p, each_t, each_index in zip(p, t, study_index):
                p_dict[each_index.item()].append(each_p)
                t_dict[each_index.item()].append(each_t)

            # mean over images
            p_mean = torch.stack(
                [torch.stack(p_dict[i]).mean(dim=0) for i in p_dict.keys()]
            )
            t_mean = torch.stack(
                [torch.stack(t_dict[i]).mean(dim=0) for i in t_dict.keys()]
            )
            for c in range(num_classes - 1):
                metrics_dict[f"auc{c}_study_mean"] = self.auc(
                    t_mean[:, c], p_mean[:, c]
                )
            metrics_dict["auc_study_mean"] = self.auc(t_mean[:, -1], p_mean[:, -1])

            # max over images
            p_max = torch.stack(
                [torch.stack(p_dict[i]).amax(dim=0) for i in p_dict.keys()]
            )
            t_max = torch.stack(
                [torch.stack(t_dict[i]).amax(dim=0) for i in t_dict.keys()]
            )
            for c in range(num_classes - 1):
                metrics_dict[f"auc{c}_study_mean"] = self.auc(t_max[:, c], p_max[:, c])

            metrics_dict["auc_study_max"] = self.auc(t_max[:, -1], p_max[:, -1])

            # take max of mean/max aggregation
            metrics_dict["auc_study"] = torch.stack(
                [metrics_dict["auc_study_mean"], metrics_dict["auc_study_max"]]
            ).amax()

        return metrics_dict


class Mammo_AUROC(Metric):
    """
    Compute multiclass AUROC for mammo dataset.
    """

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("group_index", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].float())
        self.t.append(batch["y"])
        self.group_index.append(batch["group_index"])

    def auc(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if t.size(0) == 0 or p.size(0) == 0:
            return torch.tensor(0.5)
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))

    def compute_aucs(
        self, t: torch.Tensor, p: torch.Tensor, suffix: str
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        for i in range(4):
            metrics_dict[f"auc{i}_{suffix}"] = self.auc(t[:, i], p[:, i])
        birads_t = t[:, 4]
        birads_t_present = birads_t != -1
        for idx, i in enumerate([4, 5, 6]):
            metrics_dict[f"auc{i}_{suffix}"] = self.auc(
                (birads_t[birads_t_present] == idx).float(), p[birads_t_present, i]
            )
        density_t = t[:, 5]
        density_t_present = density_t != -1
        for idx, i in enumerate([7, 8, 9, 10]):
            metrics_dict[f"auc{i}_{suffix}"] = self.auc(
                (density_t[density_t_present] == idx).float(), p[density_t_present, i]
            )
        return metrics_dict

    def aggregate_mean_or_max(
        self,
        p_dict: Dict[int, torch.Tensor],
        t_dict: Dict[int, torch.Tensor],
        agg_func: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert agg_func in {
            torch.mean,
            torch.amax,
        }, "only torch.mean and torch.amax are valid aggregation functions"
        p_agg = torch.stack(
            [agg_func(torch.stack(p_dict[i]), dim=0) for i in p_dict.keys()]
        )
        # labels should be same for all samples in group
        # so mean or max doesn't matter
        t_agg = torch.stack(
            [agg_func(torch.stack(t_dict[i]), dim=0) for i in t_dict.keys()]
        )
        return p_agg, t_agg

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        metrics_dict = self.compute_aucs(t, p, "image")
        # aggregate predictions over images to compute breast-level AUC
        p_dict, t_dict = defaultdict(list), defaultdict(list)
        group_indices = torch.cat(self.group_index, dim=0).cpu()
        for each_p, each_t, each_index in zip(p, t, group_indices):
            p_dict[each_index.item()].append(each_p)
            t_dict[each_index.item()].append(each_t)

        # mean over images
        p_mean, t_mean = self.aggregate_mean_or_max(p_dict, t_dict, torch.mean)
        metrics_dict.update(self.compute_aucs(t_mean, p_mean, "breast_mean"))

        # max over images
        p_max, t_max = self.aggregate_mean_or_max(p_dict, t_dict, torch.amax)
        metrics_dict.update(self.compute_aucs(t_max, p_max, "breast_max"))

        # take max of mean/max aggregation
        metrics_dict["auc0_breast"] = torch.stack(
            [metrics_dict["auc0_breast_mean"], metrics_dict["auc0_breast_max"]]
        ).amax()
        return metrics_dict


class GroupedAUROC(Metric):
    """
    Compute multilabel AUROC for grouped dataset.
    """

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("group_index", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p.append(out["logits"].float())
        self.t.append(batch["y"])
        self.group_index.append(batch["group_index"])

    def auc(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))

    def compute_aucs(
        self, t: torch.Tensor, p: torch.Tensor, suffix: str
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        num_classes = p.size(1)
        for i in range(num_classes):
            metrics_dict[f"auc{i}_{suffix}"] = self.auc(t[:, i], p[:, i])
        metrics_dict[f"auc_mean_{suffix}"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean()
        return metrics_dict

    def aggregate_mean_or_max(
        self,
        p_dict: Dict[int, torch.Tensor],
        t_dict: Dict[int, torch.Tensor],
        agg_func: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert agg_func in {
            torch.mean,
            torch.amax,
        }, "only torch.mean and torch.amax are valid aggregation functions"
        p_agg = torch.stack(
            [agg_func(torch.stack(p_dict[i]), dim=0) for i in p_dict.keys()]
        )
        # labels should be same for all samples in group
        # so mean or max doesn't matter
        t_agg = torch.stack(
            [agg_func(torch.stack(t_dict[i]), dim=0) for i in t_dict.keys()]
        )
        return p_agg, t_agg

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        metrics_dict = self.compute_aucs(t, p, "image")
        # aggregate predictions over images to compute breast-level AUC
        p_dict, t_dict = defaultdict(list), defaultdict(list)
        group_indices = torch.cat(self.group_index, dim=0).cpu()
        for each_p, each_t, each_index in zip(p, t, group_indices):
            p_dict[each_index.item()].append(each_p)
            t_dict[each_index.item()].append(each_t)

        # mean over images
        p_mean, t_mean = self.aggregate_mean_or_max(p_dict, t_dict, torch.mean)
        metrics_dict.update(self.compute_aucs(t_mean, p_mean, "agg_mean"))

        # max over images
        p_max, t_max = self.aggregate_mean_or_max(p_dict, t_dict, torch.amax)
        metrics_dict.update(self.compute_aucs(t_max, p_max, "agg_max"))

        # take max of mean/max aggregation
        metrics_dict["auc_mean_agg"] = torch.stack(
            [metrics_dict["auc_mean_agg_mean"], metrics_dict["auc_mean_agg_max"]]
        ).amax()
        return metrics_dict


class Mammo_pF1(GroupedAUROC):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(cfg, dist_sync_on_step=dist_sync_on_step)
        self.thresholds = self.cfg.metric_thresholds or [0.5]

    @staticmethod
    def pfbeta(
        labels: torch.Tensor, predictions: torch.Tensor, beta: int = 1
    ) -> torch.Tensor:
        # From: https://www.kaggle.com/code/sohier/probabilistic-f-score
        y_true_count = 0
        ctp = 0
        cfp = 0
        for idx in range(len(labels)):
            prediction = min(max(predictions[idx], 0), 1)
            if labels[idx]:
                y_true_count += 1
                ctp += prediction
            else:
                cfp += prediction
        beta_squared = beta * beta
        if ctp + cfp == 0 or y_true_count == 0:
            return torch.tensor(0)
        c_precision = ctp / (ctp + cfp)
        c_recall = ctp / y_true_count
        if c_precision > 0 and c_recall > 0:
            result = (
                (1 + beta_squared)
                * (c_precision * c_recall)
                / (beta_squared * c_precision + c_recall)
            )
            return result
        else:
            return torch.tensor(0)

    def binarized_pf1(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pf1_list = []
        for threshold in self.thresholds:
            p_binarized = (p >= threshold).float()
            pf1_list.append(self.pfbeta(t, p_binarized))
        pf1_list = torch.stack(pf1_list, dim=0)
        best_pf1, best_threshold = (
            pf1_list.amax(dim=0),
            self.thresholds[pf1_list.argmax(dim=0).item()],
        )
        return best_pf1, torch.tensor(best_threshold)

    def compute_f1s(
        self, t: torch.Tensor, p: torch.Tensor, suffix: str
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        metrics_dict[f"pF1_{suffix}"] = self.pfbeta(t, p)
        pf1_bin, pf1_bin_th = self.binarized_pf1(p, t)
        metrics_dict[f"pF1_bin_{suffix}"] = pf1_bin
        metrics_dict[f"pF1_bin_th_{suffix}"] = pf1_bin_th
        return metrics_dict

    def compute(self) -> Dict[str, torch.Tensor]:
        p = torch.cat(self.p, dim=0).cpu().sigmoid()
        t = torch.cat(self.t, dim=0).cpu()
        # assume cancer is first class and only evaluate cancer
        p, t = p[:, 0], t[:, 0]
        metrics_dict = self.compute_f1s(t, p, "image")
        # aggregate predictions over images to compute breast-level AUC
        p_dict, t_dict = defaultdict(list), defaultdict(list)
        group_indices = torch.cat(self.group_index, dim=0).cpu()
        for each_p, each_t, each_index in zip(p, t, group_indices):
            p_dict[each_index.item()].append(each_p)
            t_dict[each_index.item()].append(each_t)

        # mean over images
        p_mean, t_mean = self.aggregate_mean_or_max(p_dict, t_dict, torch.mean)
        metrics_dict.update(self.compute_f1s(t_mean, p_mean, "agg_mean"))

        # max over images
        p_max, t_max = self.aggregate_mean_or_max(p_dict, t_dict, torch.amax)
        metrics_dict.update(self.compute_f1s(t_max, p_max, "agg_max"))

        # take max of mean/max aggregation
        metrics_dict["pF1_agg"] = torch.stack(
            [metrics_dict["pF1_agg_mean"], metrics_dict["pF1_agg_max"]]
        ).amax()
        metrics_dict["pF1_bin_agg"] = torch.stack(
            [metrics_dict["pF1_bin_agg_mean"], metrics_dict["pF1_bin_agg_max"]]
        ).amax()
        return metrics_dict


class MelanomaAUROC(GroupedAUROC):
    def compute(self) -> Dict[str, torch.Tensor]:
        # only take melanoma class
        p = torch.cat(self.p, dim=0).cpu()[:, 0]
        t = torch.cat(self.t, dim=0).cpu()[:, 0]
        metrics_dict = {"auc_mean": self.auc(t, p)}
        # group index is year
        group_indices = torch.cat(self.group_index, dim=0).cpu()
        for year in torch.unique(group_indices):
            metrics_dict[f"auc_{year}"] = self.auc(
                t[group_indices == year], p[group_indices == year]
            )
        return metrics_dict


class ICHSeqClsAUROC(Metric):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg

        self.add_state("p_seq", default=[], dist_reduce_fx=None)
        self.add_state("t_seq", default=[], dist_reduce_fx=None)
        self.add_state("p_cls", default=[], dist_reduce_fx=None)
        self.add_state("t_cls", default=[], dist_reduce_fx=None)
        self.add_state("mask", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        self.p_seq.append(out["logits_seq"].float())
        self.t_seq.append(batch["y_seq"].float())
        self.p_cls.append(out["logits_cls"].float())
        self.t_cls.append(batch["y_cls"].float())
        self.mask.append(batch["mask"])

    def auc(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if len(t.unique()) == 1:
            return torch.tensor(0.5)
        t, p = t.numpy(), p.numpy()
        return torch.tensor(roc_auc_score(y_true=t, y_score=p))

    def compute_aucs(
        self, t: torch.Tensor, p: torch.Tensor, suffix: str
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        num_classes = p.size(1)
        for i in range(num_classes):
            metrics_dict[f"auc{i}_{suffix}"] = self.auc(t[:, i], p[:, i])
        metrics_dict[f"auc_mean_{suffix}"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean()
        return metrics_dict

    def compute(self) -> Dict[str, torch.Tensor]:
        p_cls = torch.cat(self.p_cls, dim=0).cpu()
        t_cls = torch.cat(self.t_cls, dim=0).cpu()
        metrics_dict = self.compute_aucs(t_cls, p_cls, "cls")
        p_seq = torch.cat(self.p_seq, dim=0).cpu()
        t_seq = torch.cat(self.t_seq, dim=0).cpu()
        p_seq = rearrange(p_seq, "b n c -> (b n) c")
        t_seq = rearrange(t_seq, "b n c -> (b n) c")
        mask = torch.cat(self.mask, dim=0).cpu()
        mask = rearrange(mask, "b n -> (b n)")
        p_seq, t_seq = p_seq[~mask], t_seq[~mask]
        metrics_dict.update(self.compute_aucs(t_seq, p_seq, "seq"))
        size = len(p_seq)
        p_seq = torch.cat([p_seq[:, i] for i in range(p_seq.shape[1])], dim=0).sigmoid().numpy()
        t_seq = torch.cat([t_seq[:, i] for i in range(t_seq.shape[1])], dim=0).numpy()
        sample_weights = [2] * size + [1] * (5 * size)
        metrics_dict["log_loss_seq"] = log_loss(t_seq, p_seq, sample_weight=np.asarray(sample_weights))
        return metrics_dict
