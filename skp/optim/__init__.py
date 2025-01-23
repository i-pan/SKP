import math

from torch import nn, optim
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from skp.configs.base import Config
from skp.optim.adamp import AdamP
from skp.optim.linear_warmup_cosine_annealing import LinearWarmupCosineAnnealingLR


def get_optimizer(cfg: Config, model: nn.Module) -> Optimizer:
    if cfg.optimizer == "AdamP":
        return AdamP(model.parameters(), **cfg.optimizer_params)
    return getattr(optim, cfg.optimizer)(
        params=model.parameters(), **cfg.optimizer_params
    )


def get_scheduler(cfg: Config, optimizer: Optimizer) -> LRScheduler:
    if cfg.scheduler in {"CosineAnnealingLR", "LinearWarmupCosineAnnealingLR"}:
        # Determine the total number of training steps
        # This is determined by:
        #   - Specified of iterations or dataset size + batch size
        #   - Number of epochs
        #   - World size (distributed training)
        #   - Number of gradient accumulation steps (default: 1, i.e., no gradient accumulation)
        param = "T_max" if cfg.scheduler == "CosineAnnealingLR" else "total_steps"
        if cfg.num_iterations_per_epoch is not None:
            steps_per_epoch = cfg.num_iterations_per_epoch
        else:
            effective_batch_size = cfg.batch_size * cfg.world_size
            # using ceil will overestimate steps_per_epoch
            # but will prevent scheduler from overstepping
            # results in minimal difference in learning rate schedule
            steps_per_epoch = math.ceil(cfg.n_train / effective_batch_size)

        steps_per_epoch = math.ceil(
            steps_per_epoch / (cfg.accumulate_grad_batches or 1)
        )
        cfg.scheduler_params[param] = steps_per_epoch * cfg.num_epochs

    if cfg.scheduler == "LinearWarmupCosineAnnealingLR":
        scheduler_obj = LinearWarmupCosineAnnealingLR
        # set max_lr to specified lr
        cfg.scheduler_params["max_lr"] = cfg.optimizer_params["lr"]
        if "warmup_epochs" in cfg.scheduler_params:
            assert "pct_start" not in cfg.scheduler_params
            cfg.scheduler_params["pct_start"] = (
                cfg.scheduler_params.pop("warmup_epochs") / cfg.num_epochs
            )
    else:
        scheduler_obj = getattr(lr_scheduler, cfg.scheduler)

    return scheduler_obj(optimizer=optimizer, **cfg.scheduler_params)
