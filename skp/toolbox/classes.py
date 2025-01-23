import torch
import torch.nn as nn

from typing import Dict, Optional


class Ensemble(nn.Module):
    """
    Simple wrapper to mean-ensemble multiple models during inference
    Activation function, if specified, applied before mean
    """

    def __init__(
        self,
        model_list: nn.ModuleList,
        output_name: str = "logits",
        activation_fn: Optional[str] = None,
    ):
        super().__init__()
        assert isinstance(model_list, nn.ModuleList)
        self.models = model_list
        self.output_name = output_name
        self.activation_fn = activation_fn

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out_list = []
        for model in self.models:
            out = model(batch, return_loss=False)[self.output_name]
            if self.activation_fn == "sigmoid":
                out = out.sigmoid()
            elif self.activation_fn == "softmax":
                out = out.softmax(dim=1)
            out_list.append(out)
        return torch.stack(out_list, dim=0).mean(dim=0)
