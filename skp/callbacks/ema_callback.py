"""
Based off:
https://github.com/benihime91/gale/blob/master/gale/collections/callbacks/ema.py
"""

import torch

from copy import deepcopy
from lightning.pytorch.callbacks import Callback

from skp.callbacks.ema import ModelEmaV3
from skp.callbacks.utils import _print_rank_zero


class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        update_every_n_steps: int = 1,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2 / 3,
        device: torch.device | None = None,
        foreach: bool = True,
        exclude_buffers: bool = False,
        switch_ema: bool = False,
    ):
        """
        Initialize the EMACallback with optional parameters for exponential moving average.

        Args:
            decay (float): The decay rate for the moving average. Default is 0.9999.
            min_decay (float): The minimum decay rate to be used. Default is 0.0.
            update_after_step (int): Number of steps after which to start updating EMA. Default is 0.
            update_every_n_steps (int): Number of steps between EMA updates. Default is 1.
            use_warmup (bool): Whether to use warmup for decay rate. Default is False.
            warmup_gamma (float): The gamma value for warmup calculation. Default is 1.0.
            warmup_power (float): The power value for warmup calculation. Default is 2/3.
            device (Optional[torch.device]): The device to perform EMA on. Default is None.
            foreach (bool): Whether to use torch's foreach methods for updates. Default is True.
            exclude_buffers (bool): Whether to exclude buffers from EMA. Default is False.
            switch_ema (bool): Whether to switch EMA weights to model training weights after each epoch.
                Default is False.

        Notes:
          The non-EMA weights are saved to the Lightning checkpoint and the EMA weights are
          saved separately as ema_weights.pt in the same folder. For some reason, on_save_checkpoint
          hook does not work.

          Operates over state_dict().values() rather than parameters() since parameters() does not
          include buffers (e.g., running mean, variance of batch norm)
        """
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.update_every_n_steps = update_every_n_steps
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device
        self.exclude_buffers = exclude_buffers
        self.switch_ema = switch_ema

        self.ema = None
        self.global_step = 0

    def on_fit_start(self, trainer, pl_module):
        self.ema = ModelEmaV3(
            model=pl_module,
            decay=self.decay,
            min_decay=self.min_decay,
            update_after_step=self.update_after_step,
            use_warmup=self.use_warmup,
            warmup_gamma=self.warmup_gamma,
            warmup_power=self.warmup_power,
            device=self.device,
            foreach=self.foreach,
            exclude_buffers=self.exclude_buffers,
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if self.update_after_step == 0 and self.global_step == 0:
            _print_rank_zero("Starting EMA at step 0 ...")

        # update EMA
        if self.global_step >= self.update_after_step:
            if self.global_step % self.update_every_n_steps == 0:
                self.ema.update(model=pl_module, step=self.global_step)

        # if current_decay is None, EMA has not started
        # so effectively decay is 0
        pl_module.logger.experiment["training/ema_decay"].append(
            self.ema.current_decay or 0.0
        )

        self.global_step += 1
        if self.global_step == self.update_after_step:
            _print_rank_zero(f"Starting EMA at step {self.update_after_step} ...")
            # once starting step is reached, copy the training model parameters to EMA
            self.copy_to(pl_module, self.ema.module)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.global_step > self.update_after_step:
            # save original parameters before replacing with EMA
            self.store(pl_module)
            # copy EMA parameters to LightningModule
            _print_rank_zero("Switching to EMA weights for validation ...")
            self.copy_to(self.ema.module, pl_module)
            pl_module.eval()

    def on_validation_end(self, trainer, pl_module):
        if self.global_step > self.update_after_step:
            # restore original parameters
            _print_rank_zero("Switching back to training weights ...")
            self.restore(pl_module)
            pl_module.train()

    def on_train_epoch_start(self, trainer, pl_module):
        # switch_ema is done at start or else we end up saving same weights
        # for EMA and non-EMA
        if self.global_step > self.update_after_step:
            if self.switch_ema:
                _print_rank_zero(
                    "switch_ema=True, copying EMA weights to training weights ..."
                )
                # copy EMA parameters to LightningModule
                self.copy_to(self.ema.module, pl_module)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            self.ema.module.load_state_dict(checkpoint["state_dict"])

    def store(self, module):
        # store backup parameters
        self.backup = deepcopy(module.state_dict())

    def restore(self, module):
        # restore backup parameters to module
        for backup_param, param in zip(
            self.backup.values(), module.state_dict().values()
        ):
            param.copy_(backup_param)

    def copy_to(self, model_x, model_y):
        # copy parameters from model_x to model_y
        for x_param, y_param in zip(
            model_x.state_dict().values(), model_y.state_dict().values()
        ):
            y_param.copy_(x_param)
