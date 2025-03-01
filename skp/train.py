import argparse
import copy
import json
import numpy as np
import os
import pickle
import lightning
import logging
import re
import sys
import torch
import uuid

from ast import literal_eval
from importlib import import_module
from lightning.pytorch import callbacks as lightning_callbacks
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.plugins import TorchSyncBatchNorm
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
from timm.layers import convert_sync_batchnorm
from typing import Optional, Tuple

from skp.callbacks import EMACallback, GPUStatsLogger
from skp.configs.base import Config
from skp.optim import get_optimizer, get_scheduler


class TimmSyncBatchNorm(TorchSyncBatchNorm):
    """
    Default SyncBN plugin for Lightning does not work with latest version of timm
    EfficientNets because it uses the native PyTorch `convert_sync_batchnorm` function.

    Use this plugin instead, which uses the timm helper and should work for non-timm
    models as well.
    """

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return convert_sync_batchnorm(model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--kfold", type=str)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--overwrite_run", action="store_true")
    parser.add_argument("--double_cv", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    # this flag is to not change fold in path of pretrained weights
    # e.g. if loading weights trained on another dataset
    parser.add_argument("--keep_pretrained_weights_path", action="store_true")
    # can be used to add notes e.g., for overwritten args
    parser.add_argument("--notes", type=str, default=None)
    # Lightning Trainer arguments
    # any arguments above need to be popped before passing to get_trainer function
    # due to overwrite/unknown arguments, prefer not to use LightningCLI
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    # default to benchmark=True, sync_batchnorm=True
    parser.add_argument("--no_benchmark", action="store_true")
    parser.add_argument("--no_sync_batchnorm", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    return parser.parse_known_args()


@rank_zero_only
def _print_rank_zero(msg: str) -> None:
    print(msg)


def _modify_limit_batches(args: argparse.Namespace) -> argparse.Namespace:
    # float represents percentage, int represents number of batches
    if isinstance(args.limit_train_batches, float):
        if args.limit_train_batches > 1:
            args.limit_train_batches = int(args.limit_train_batches)

    if isinstance(args.limit_val_batches, float):
        if args.limit_val_batches > 1:
            args.limit_val_batches = int(args.limit_val_batches)

    return args


def load_config(args, overwrite_args):
    args = _modify_limit_batches(args)

    cfg_file = args.__dict__.pop("config")
    cfg = import_module(f"skp.configs.{cfg_file}").cfg
    cfg.args = args.__dict__
    cfg.config = cfg_file

    # overwrite config parameters, if specified in command line
    if len(overwrite_args) > 1:
        # assumes all overwrite args are prepended with '--''
        # overwrite_args will be a list following:
        # [arg_name, arg_value, arg_name, arg_value, ...]
        # here we turn it into a dict following: {arg_name: arg_value}
        overwrite_args = {
            k.replace("-", ""): v
            for k, v in zip(overwrite_args[::2], overwrite_args[1::2])
        }

    # some config parameters, such as loss_params or optimizer_params
    # are dictionaries
    # to overwrite these params via CLI, we use dot concatenation to specify
    # params within these dictionaries i.e., optimizer_params.lr
    for key in overwrite_args:
        # require that you can only overwrite an existing config parameter
        # split by . to deal with subparams
        if key in {"image_height", "image_width"}:
            raise Exception(
                f"`{key}` should not be overwritten by CLI as it is most likely referenced in other config parameters"
            )
        if overwrite_args[key].endswith(".json"):
            # for parameters (e.g., cfg.loss_params) which may have complicated
            # nested dictionary structures, if wanting to iterate over multiple
            # parameters, easier to generate a json file for each set
            _print_rank_zero(
                f"overwriting cfg.{key} with parameters in {overwrite_args[key]}"
            )
            with open(overwrite_args[key], "r") as f:
                cfg.__dict__[key] = json.load(f)
            continue
        if key.split(".")[0] in cfg.__dict__:
            if len(key.split(".")) == 1:
                # no subparam, just overwrite
                _print_rank_zero(
                    f"overwriting cfg.{key}: {cfg.__dict__[key]} -> {overwrite_args[key]}"
                )
            else:
                # subparam, need to identify dict param and key-value pair
                param_dict, param_key = key.split(".")
                _print_rank_zero(
                    f"overwriting cfg.{key}: {cfg.__dict__[param_dict][param_key]} -> {overwrite_args[key]}"
                )

            cfg_type = type(cfg.__dict__[key.split(".")[0]])
            # check if param is a dict
            if cfg_type is dict:
                assert len(key.split(".")) > 1
                param_dict, param_key = key.split(".")
                param_type = type(cfg.__dict__[param_dict][param_key])
                if param_type is bool:
                    # note that because we are not using argparse to add arguments
                    # we cannot have `store_true` args, so if it is a boolean, we need
                    # to specify True after the arg flag in command line
                    cfg.__dict__[param_dict][param_key] = overwrite_args[key] == "True"
                elif param_type is None:
                    cfg.__dict__[param_dict][param_key] = overwrite_args[key]
                else:
                    cfg.__dict__[param_dict][param_key] = param_type(
                        overwrite_args[key]
                    )
            # check if param is a list or tuple
            elif cfg_type is list or cfg_type is tuple:
                cfg.__dict__[key] = literal_eval(overwrite_args[key])
            else:
                if cfg_type is bool:
                    cfg.__dict__[key] = overwrite_args[key] == "True"
                elif cfg_type is None:
                    cfg.__dict__[key] = overwrite_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(overwrite_args[key])
        else:
            raise Exception(f"{key} is not specified in config")

    return cfg


@rank_zero_only
def symlink_best_model_path(trainer: lightning.Trainer) -> None:
    best_model_path = None
    for callback in trainer.callbacks:
        if isinstance(callback, lightning_callbacks.ModelCheckpoint):
            best_model_path = os.path.abspath(callback.best_model_path)
            break

    if best_model_path:
        symlink_path = os.path.join(os.path.dirname(best_model_path), "best.ckpt")
        if os.path.exists(symlink_path):
            _ = os.system(f"rm {symlink_path}")
        _ = os.system(f"ln -s {best_model_path} {symlink_path}")


def generate_random_run_id(num_chars: int = 8) -> str:
    return uuid.uuid4().hex[:num_chars]


def generate_experiment_save_dir(cfg: Config, run_id: Optional[str] = None) -> Config:
    save_dir = os.path.abspath(cfg.save_dir)

    if run_id:
        cfg.run_id = run_id
        _print_rank_zero(f"Using manually specified run ID: {run_id}")
    else:
        cfg.run_id = generate_random_run_id()

    save_dir = os.path.join(save_dir, cfg.config, cfg.run_id)
    cfg.save_dir = save_dir

    _print_rank_zero(f"\nRun ID: {cfg.run_id}\n")
    _print_rank_zero(f"Saving experiments to {cfg.save_dir} ...\n")

    return cfg


@rank_zero_only
def create_save_dirs(save_dir: str, overwrite: bool = False) -> None:
    os.makedirs(save_dir, exist_ok=overwrite)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=overwrite)


def get_trainer(cfg: Config) -> Tuple[lightning.Trainer, Config]:
    save_dir = os.path.join(cfg.save_dir, f"fold{cfg.fold}")
    create_save_dirs(save_dir, overwrite=cfg.overwrite_run)
    callbacks = []
    if cfg.ema and cfg.ema["on"]:
        _print_rank_zero("\n>> Using EMA ...\n")
        callbacks.append(EMACallback(**{k: v for k, v in cfg.ema.items() if k != "on"}))
    else:
        # don't save EMA parameters if not using
        _ = cfg.__dict__.pop("ema", None)

    callbacks.extend(
        [
            GPUStatsLogger(),
            lightning_callbacks.ModelCheckpoint(
                # Set dirpath explicitly to save checkpoints in the desired folder
                # This is so that we can keep the desired directory structure and format locally
                dirpath=os.path.join(save_dir, "checkpoints"),
                monitor="val_metric",
                filename="{epoch:03d}-{val_metric:.4f}",
                save_last=True,
                save_weights_only=True,
                mode=cfg.val_track,
                save_top_k=cfg.save_top_k or 1,
            ),
            lightning_callbacks.LearningRateMonitor(logging_interval="step"),
        ]
    )

    if cfg.early_stopping:
        _print_rank_zero(">> Using early stopping ...")
        early_stopping = lightning_callbacks.EarlyStopping(
            patience=cfg.early_stopping_patience,
            monitor="val_metric",
            min_delta=cfg.early_stopping_min_delta,
            verbose=cfg.early_stopping_verbose or False,
            mode=cfg.val_track,
        )
        callbacks.append(early_stopping)

    if cfg.args["strategy"] == "ddp":
        strategy = lightning.pytorch.strategies.DDPStrategy(
            find_unused_parameters=cfg.find_unused_parameters or False
        )
        plugins = [TimmSyncBatchNorm()]
    else:
        strategy = cfg.args["strategy"]
        plugins = None

    neptune_logger = NeptuneLogger(
        project=cfg.project,
        source_files=[
            os.path.join(
                os.path.dirname(__file__), f"configs/{cfg.config.replace('.', '/')}.py"
            ),
            os.path.join(
                os.path.dirname(__file__), f"models/{cfg.model.replace('.', '/')}.py"
            ),
            os.path.join(
                os.path.dirname(__file__),
                f"datasets/{cfg.dataset.replace('.', '/')}.py",
            ),
        ],
        mode=cfg.neptune_mode,
        log_model_checkpoints=False,
    )

    # make copy of args dictionary for strategy
    args_dict = copy.deepcopy(cfg.args)
    args_dict["strategy"] = strategy

    args_dict["sync_batchnorm"] = not args_dict.pop("no_sync_batchnorm", False)
    args_dict["benchmark"] = not args_dict.pop("no_benchmark", False)

    if cfg.args["strategy"] != "ddp" and args_dict["sync_batchnorm"]:
        raise Exception("SyncBatchNorm is only supported with DDP strategy")

    trainer = lightning.Trainer(
        **args_dict,
        max_epochs=cfg.num_epochs,
        callbacks=callbacks,
        plugins=plugins,
        logger=neptune_logger,
        # easier to handle custom samplers if below is False
        # see tasks/samplers.py
        # if running DDP, uses native torch DistributedSampler as default
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.accumulate_grad_batches or 1,
        profiler="simple",
    )

    return trainer, cfg


def get_loss(cfg: Config) -> torch.nn.Module:
    module, loss = cfg.loss.split(".")
    module = import_module(f"skp.losses.{module}")
    return getattr(module, loss)(cfg.loss_params)


def get_metrics(cfg: Config):
    metrics_list = []
    for metric in cfg.metrics:
        module, metric_name = metric.split(".")
        module = import_module(f"skp.metrics.{module}")
        metrics_list.append(getattr(module, metric_name)(cfg))
    return metrics_list


def get_task(cfg: Config) -> Tuple[lightning.LightningModule, Config]:
    model = import_module(f"models.{cfg.model}").Net(cfg)
    if model.criterion is None:
        # for some models (e.g., detection),
        # easier to write loss with model
        loss = get_loss(cfg)
        model.set_criterion(loss)

    ds_class = import_module(f"datasets.{cfg.dataset}").Dataset
    train_dataset = ds_class(cfg, "train")
    val_dataset = ds_class(cfg, "val")

    cfg.n_train, cfg.n_val = len(train_dataset), len(val_dataset)
    _print_rank_zero("\n")
    _print_rank_zero(f"TRAIN : N={cfg.n_train}")
    _print_rank_zero(f"VAL   : N={cfg.n_val}\n")
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    task = import_module(f"tasks.{cfg.task}").Task(cfg)

    task.set("model", model)
    task.set("datasets", [train_dataset, val_dataset])
    task.set("optimizer", optimizer)
    task.set("scheduler", scheduler)
    task.set("metrics", get_metrics(cfg))
    task.set("val_metric", cfg.val_metric)

    # config has been updated with n_train, n_val
    return task, cfg


@rank_zero_only
def sync_neptune_if_offline(cfg: Config) -> None:
    # Avoids multiple uploads by syncing everything at end of run
    # in case using server which would potentially flag
    # as suspicious activity
    if cfg.neptune_mode == "offline":
        _ = os.system(f"neptune sync --project {cfg.project} --offline-only")


@rank_zero_only
def save_config_as_pickle(cfg: Config) -> None:
    # Can think about a solution not using pickle at a later point
    # Mainly to save a copy of the config if it was modified by command line
    # arguments, since the original config would not be correct in that case
    # Although the parameters should be correct in Neptune
    with open(os.path.join(cfg.save_dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg.__dict__, f)  # dump dict because pickle can't load Config obj

    print(f"Completed run {cfg.run_id}")
    print(f"Saved experiment to {cfg.save_dir}")


class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
            )
        )


@rank_zero_only
def print_environment(cfg: Config, args: argparse.Namespace) -> None:
    print("\nENVIRONMENT\n")
    print(f"  Python {sys.version}\n")
    print(f"  torch.__version__              = {torch.__version__}")
    print(f"  torch.version.cuda             = {torch.version.cuda}")
    print(f"  torch.backends.cudnn.version() = {torch.backends.cudnn.version()}\n")
    print(f"  pytorch_lightning.__version__  = {lightning.__version__}\n")
    print(
        f"  world_size={cfg.world_size}, num_nodes={args.num_nodes}, num_gpus={args.devices if args.devices else 1}"
    )
    print("\n")


@rank_zero_only
def save_ema_weights(trainer: lightning.Trainer) -> None:
    ema = False
    for callback in trainer.callbacks:
        if isinstance(callback, EMACallback):
            ema = True
            ema_callback = callback
        if isinstance(callback, lightning_callbacks.ModelCheckpoint):
            last_model_path = os.path.abspath(callback.last_model_path)
    if not ema:
        return

    save_dir = os.path.dirname(last_model_path)
    torch.save(
        {"state_dict": ema_callback.ema.module.state_dict()},
        os.path.join(save_dir, "ema_weights.pt"),
    )


def change_fold_in_pretrained_weights_path(cfg: Config) -> Config:
    """
    Changes pretrained weights fold to current fold in config
    For example, if running k-fold training and wanting to load
    pretrained backbone for each fold, instead of manually
    specifying each individual fold path for each run

    Assumes all folds live in the same run_id
    """
    load_attributes = [
        "load_pretrained_model",
        "load_pretrained_backbone",
        "load_pretrained_encoder",
        "load_pretrained_decoder",
    ]
    for each_att in load_attributes:
        val = getattr(cfg, each_att)
        if val:
            pattern = r"/fold[0-9]+/"
            if bool(re.search(pattern, val)):
                setattr(cfg, each_att, re.sub(pattern, f"/fold{cfg.fold}/", val))

    return cfg


def progressive_resizing(cfg: Config) -> None:
    _print_rank_zero("Running progressive resizing ...\n")
    # progressive resizing parameters live in a dictionary
    # in cfg.progressive_resizing
    num_sizes = None
    for k, v in cfg.progressive_resizing.items():
        assert isinstance(v, list)
        if num_sizes:
            assert len(v) == num_sizes
        else:
            num_sizes = len(v)

    model_state_dict = {}
    cfg.save_dir = os.path.join(cfg.save_dir, "prog_resize0")
    for i in range(num_sizes):
        cfg.prog_resize = i
        for k, v in cfg.progressive_resizing.items():
            _print_rank_zero(f">>setting cfg.{k} to {v[i]}")
            cfg.__dict__[k] = v[i]
        task, cfg = get_task(cfg)
        if len(model_state_dict) > 0:
            # load previously trained model, if exists
            previous_height = cfg.progressive_resizing["image_height"][i - 1]
            previous_width = cfg.progressive_resizing["image_width"][i - 1]
            _print_rank_zero(
                f"Loading previously trained model ({previous_height}x{previous_width}) ..."
            )
            task.model.load_state_dict(model_state_dict)
        if i > 0:
            cfg.save_dir = cfg.save_dir.replace(f"resize{i-1}", f"resize{i}")
        trainer, cfg = get_trainer(cfg)
        trainer.fit(task)
        model_state_dict = copy.deepcopy(task.model.state_dict())
        symlink_best_model_path(trainer)
        trainer.logger.experiment.stop()
        sync_neptune_if_offline(cfg)
        save_ema_weights(trainer)


def hyperparameter_sweep(cfg: Config) -> None:
    from skp.toolbox.halton import generate_search

    num_trials = cfg.hyperparameter_sweep["num_trials"]
    _print_rank_zero(f"Running hyperparameter sweep ({num_trials} trials) ...\n")
    search_space = {
        k: v for k, v in cfg.hyperparameter_sweep.items() if k != "num_trials"
    }
    # set seed or will generate different hyperparams for each DDP process
    np.random.seed(88)
    hyperparameters = generate_search(search_space, num_trials)
    for trial_idx, hyp_tuple in enumerate(hyperparameters):
        for hyp in hyp_tuple._fields:
            hyp_value = getattr(hyp_tuple, hyp)
            if isinstance(hyp_value, float):
                hyp_value_print = f"{hyp_value:0.3g}"
            else:
                hyp_value_print = f"{hyp_value}"
            _print_rank_zero(f">>setting cfg.{hyp} to {hyp_value_print}")
            if len(hyp.split("__")) > 1:
                param_dict = hyp.split("__")[0]
                if getattr(cfg, param_dict) is None:
                    setattr(cfg, param_dict, {})
                cfg.__dict__[param_dict][hyp.split("__")[1]] = hyp_value
            else:
                cfg.__dict__[hyp] = hyp_value

        if trial_idx > 0:
            # generate new run
            save_dir = Path(cfg.save_dir).parent.absolute()
            run_id = generate_random_run_id()
            cfg.save_dir = os.path.join(save_dir, run_id)

        task, cfg = get_task(cfg)
        trainer, cfg = get_trainer(cfg)
        trainer.fit(task)
        symlink_best_model_path(trainer)
        trainer.logger.experiment.stop()
        sync_neptune_if_offline(cfg)
        save_ema_weights(trainer)


def main():
    # small bug in lightning causes neptune logging error (no impact on logging)
    # workaround to suppress printing of this error message
    # see: https://github.com/neptune-ai/neptune-client/issues/1702
    logging.getLogger("neptune").addFilter(_FilterCallback())

    # uses parse_known_args() to separate into specified args
    # in parse_args and unknown args which will be exclusively used for
    # overwriting config parameters
    args, overwrite_args = parse_args()
    kfold = args.__dict__.pop("kfold")

    cfg = load_config(args, overwrite_args)
    cfg.overwrite_run = args.__dict__.pop("overwrite_run")
    cfg.double_cv = args.__dict__.pop("double_cv")
    cfg.world_size = args.num_nodes * (args.devices if args.devices else 1)

    notes = args.__dict__.pop("notes")
    if notes is not None:
        cfg.notes = notes

    debug_mode = args.__dict__.pop("debug")
    if debug_mode:
        cfg.neptune_mode = "debug"
        cfg.num_workers = 1

    cfg = generate_experiment_save_dir(cfg, run_id=args.__dict__.pop("run_id"))
    print_environment(cfg, args)

    torch.set_float32_matmul_precision(cfg.float32_matmul_precision or "high")

    folds = [int(_) for _ in kfold.split(",")] if kfold else [cfg.fold]
    if cfg.double_cv is not None:
        _print_rank_zero(
            f"Running double (nested) cross-validation with outer split {cfg.double_cv} ..."
        )

    keep_pretrained_weights_path = args.__dict__.pop(
        "keep_pretrained_weights_path", False
    )
    for fold in folds:
        _print_rank_zero(
            f"Running k-fold {kfold}, fold {fold} ...\n"
            if kfold
            else f"Running fold {fold} ...\n"
        )
        cfg.fold = fold
        if not keep_pretrained_weights_path:
            cfg = change_fold_in_pretrained_weights_path(cfg)
        if cfg.progressive_resizing is not None:
            progressive_resizing(cfg)
            continue
        if cfg.hyperparameter_sweep is not None:
            hyperparameter_sweep(cfg)
            continue
        task, cfg = get_task(cfg)
        trainer, cfg = get_trainer(cfg)
        trainer.fit(task)
        symlink_best_model_path(trainer)
        trainer.logger.experiment.stop()  # stop after each fold
        # for some reason, on_save_checkpoint hook is not working
        # in EMACallback
        sync_neptune_if_offline(cfg)
        save_ema_weights(trainer)

    save_config_as_pickle(cfg)


if __name__ == "__main__":
    main()
