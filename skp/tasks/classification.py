import lightning
import numpy as np
import torch.nn as nn
import torch

from collections import defaultdict
from neptune.utils import stringify_unsupported
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Any, Dict

from skp.tasks.utils import build_dataloader


class Task(lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.val_loss = defaultdict(list)

    def set(self, name: str, attr: Any) -> None:
        if name == "metrics":
            attr = nn.ModuleList(attr)
        setattr(self, name, attr)

    def on_train_start(self):
        for obj in [
            "model",
            "datasets",
            "optimizer",
            "scheduler",
            "metrics",
            "val_metric",
        ]:
            assert hasattr(self, obj)

        self.logger.experiment["cfg"] = stringify_unsupported(self.cfg.__dict__)

    def mixup(self, batch: Dict) -> Dict:
        x, y = batch["x"], batch["y"]
        # ensure float tensors
        assert x.dtype == torch.float, f"x.dtype is {x.dtype}, not float"
        assert y.dtype == torch.float, f"y.dtype is {y.dtype}, not float"

        batch_size = y.size(0)
        # generate lambda from beta distribution
        lamb = np.random.beta(self.cfg.mixup, self.cfg.mixup, batch_size)
        lamb = torch.from_numpy(lamb).to(x.device).float()
        # adjust lambda shape for broadcasting when multiplying
        if lamb.ndim < y.ndim:
            for _ in range(y.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        permuted_indices = torch.randperm(batch_size)
        # mixed label
        ymix = lamb * y + (1 - lamb) * y[permuted_indices]
        # adjust lambda shape again for input
        # which should have more dims than label
        if lamb.ndim < x.ndim:
            for _ in range(x.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        assert (
            lamb.ndim == x.ndim
        ), f"lamb has {lamb.ndim} dims whereas x has {x.ndim} dims"
        # mixed input
        xmix = lamb * x + (1 - lamb) * x[permuted_indices]
        # replace original input and label with mixed
        batch["x"] = xmix
        batch["y"] = ymix
        return batch

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if self.cfg.mixup:
            batch = self.mixup(batch)
        out = self.model(batch, return_loss=True)
        for k, v in out.items():
            if "loss" in k:
                self.log(k, v)
        return out["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        out = self.model(batch, return_loss=True)
        for k, v in out.items():
            if "loss" in k:
                self.val_loss[k].append(v)
        for m in self.metrics:
            # passing output and input dicts is the most flexible
            # then the metric can be customized to get the keys they need
            m.update(out, batch)
        return out["loss"]

    def on_train_epoch_end(self) -> None:
        if self.global_rank == 0:
            self.logger.experiment["training/epoch"].append(self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        metrics = {}
        for m in self.metrics:
            metrics.update(m.compute())
        for k, v in self.val_loss.items():
            metrics[k] = torch.stack(v).mean()
        # reset val losses
        self.val_loss = defaultdict(list)

        if isinstance(self.val_metric, list):
            metrics["val_metric"] = torch.sum(
                torch.stack([metrics[_vm.lower()].cpu() for _vm in self.val_metric])
            ).item()
        else:
            metrics["val_metric"] = metrics[self.val_metric.lower()]
            if isinstance(metrics["val_metric"], torch.Tensor):
                metrics["val_metric"] = metrics["val_metric"].item()

        for m in self.metrics:
            m.reset()

        if self.global_rank == 0:
            print("\n========")
            max_strlen = max([len(k) for k in metrics.keys()])
            for k, v in metrics.items():
                print(
                    f"{k.ljust(max_strlen)} | {v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                )

        if (
            self.trainer.state.stage
            != lightning.pytorch.trainer.states.RunningStage.SANITY_CHECKING
        ):  # don't log metrics during sanity check
            for k, v in metrics.items():
                self.logger.experiment[f"val/{k}"].append(v)

            self.log("val_metric", metrics["val_metric"], sync_dist=True)

    def configure_optimizers(self) -> Dict:
        lr_scheduler = {
            "scheduler": self.scheduler,
            "interval": self.cfg.scheduler_interval,
        }
        if isinstance(self.scheduler, ReduceLROnPlateau):
            lr_scheduler["monitor"] = self.val_metric
        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[0], "train")

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[1], "val")
