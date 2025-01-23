import torch

from typing import Dict


def torch_load_weights(path: str) -> Dict[str, torch.Tensor]:
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    return weights


def filter_weights_by_prefix(
    weights: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    weights = {
        k.replace(prefix, ""): v for k, v in weights.items() if k.startswith(prefix)
    }
    return weights
