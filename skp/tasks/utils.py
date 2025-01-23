import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from skp.configs.base import Config
from skp.tasks import samplers


def build_dataloader(cfg: Config, dataset: Dataset, mode: str) -> DataLoader:

    def worker_init_fn(worker_id: int) -> None:                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    dataloader_params = {}
    dataloader_params["num_workers"] = cfg.num_workers
    dataloader_params["drop_last"] = mode == "train"
    dataloader_params["shuffle"] = mode == "train"
    dataloader_params["pin_memory"] = cfg.pin_memory or False
    dataloader_params["persistent_workers"] = cfg.persistent_workers or False
    dataloader_params["collate_fn"] = dataset.collate_fn

    if mode == "train":
        dataloader_params["batch_size"] = cfg.batch_size
    else:
        dataloader_params["batch_size"] = cfg.val_batch_size or cfg.batch_size * 2

    sampler = None
    if cfg.sampler and cfg.sampler != "" and mode == "train":
        sampler = getattr(samplers, cfg.sampler)(dataset=dataset, cfg=cfg)

    if sampler:
        dataloader_params["shuffle"] = False
        if cfg.args["strategy"] == "ddp":
            sampler = samplers.DistributedSamplerWrapper(sampler)
        print(f"Using sampler {sampler} for training ...")
        dataloader_params["sampler"] = sampler
    elif cfg.args["strategy"] == "ddp":
        dataloader_params["shuffle"] = False
        dataloader_params["sampler"] = DistributedSampler(dataset, shuffle=mode == "train")

    loader = DataLoader(dataset,
        **dataloader_params,
        worker_init_fn=worker_init_fn)
    
    return loader
