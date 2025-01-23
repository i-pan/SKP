import torch
import torch.nn as nn

from importlib import import_module
from typing import Dict


def get_loss(loss_name, loss_params) -> nn.Module:
    module, loss = loss_name.split(".")
    module = import_module(f"skp.losses.{module}")
    return getattr(module, loss)(loss_params)


class CombinedLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        # params format:
        # {
        #     loss_name: {
        #         params: {...},
        #         output_key: ...
        #         weight: ...
        #     }
        # }
        self.losses = {k: get_loss(k, v["params"]) for k, v in params.items()}
        self.loss_name2key = {k: v["output_key"] for k, v in params.items()}
        self.loss_weights = {k: v["weight"] for k, v in params.items()}

    def forward(
        self, out: Dict[str, Dict], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # example out format
        # {
        #     cls: {
        #         logits
        #     },
        #     seg: {
        #         logits
        #     }
        # }
        # batch is similar except with expected contents of batch
        loss_dict = {}
        for loss_name, loss in self.losses.items():
            tmp_key = self.loss_name2key[loss_name]
            tmp_loss = loss(out[tmp_key], batch[tmp_key])
            tmp_loss = {f"{tmp_key}_{k}": v for k, v in tmp_loss.items()}
            loss_dict.update(tmp_loss)

        loss_dict["loss"] = 0
        for loss_name, loss in self.losses.items():
            tmp_key = self.loss_name2key[loss_name]
            loss_dict["loss"] += (
                self.loss_weights[loss_name]
                * loss_dict[f"{tmp_key}_loss"]
            )

        return loss_dict
