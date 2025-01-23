import cv2
import glob
import numpy as np
import os
import pandas as pd
import torch

from albumentations import Compose as AlbumentationsCompose
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset, default_collate
from torchvision.transforms.v2 import Compose as TorchvisionCompose

from skp.configs import Config


train_collate_fn = default_collate
val_collate_fn = default_collate


class Dataset(TorchDataset):
    def __init__(self, cfg: Config, mode: str):
        self.cfg = cfg
        self.mode = mode
        df = pd.read_csv(self.cfg.annotations_file)
        assert mode in {
            "train",
            "val",
            "inference",
        }, f"mode [{mode}] must be one of [train, val, inference]"
        if mode == "inference":
            # assume we are inferring over entire DataFrame
            # so no need to use folds
            # can specify inference_transforms in config
            # though most of the time will be same as val_transforms
            self.transforms = self.cfg.inference_transforms or self.cfg.val_transforms
        else:
            if cfg.double_cv is not None:
                # double_cv is an int representing the outer fold
                # exclude the outer fold before doing train/val split
                # as the outer fold will be the test set
                # then reassign fold column to inner
                df = df[df.outer != cfg.double_cv]
                df["fold"] = df[f"inner{cfg.double_cv}"]
            if self.mode == "train":
                df = df[df.fold != self.cfg.fold]
                self.transforms = self.cfg.train_transforms
            elif self.mode == "val":
                df = df[df.fold == self.cfg.fold]
                self.transforms = self.cfg.val_transforms

        self.df_list = [_df for _, _df in df.groupby("study_path")]
        self.num_images = cfg.num_images_per_study or 4

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.df_list)

    def load_image(self, path):
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        if img.ndim == 2:
            img = rearrange(img, "h w -> h w 1")
        return img

    def _get(self, i):
        study = self.df_list[i]
        study_path = study.study_path.values[0]
        images = sorted(glob.glob(os.path.join(self.cfg.data_dir, study_path, "*.png")))
        mask = [False] * self.num_images
        if len(images) > self.num_images:
            if self.mode == "train":
                images = list(np.random.choice(images, self.num_images, replace=False))
            else:
                images = images[: self.num_images]
        x = [self.load_image(path) for path in images]
        if len(x) < self.num_images:
            mask = [False] * len(x) + [True] * (self.num_images - len(x))
            x = x + [np.zeros_like(x[0])] * (self.num_images - len(x))
        assert (
            len(x) == self.num_images
        ), f"number of images in study [{len(x)}] does not equal specified number of images per study {self.num_images}"
        y = study[self.cfg.targets].values
        for c in range(y.shape[1]):
            # all images in study should have same label
            assert len(np.unique(y[:, c])) == 1
        y = y[0]

        return x, y, mask

    def get(self, i):
        if self.cfg.skip_failed_data:
            try:
                return self._get(i)
            except Exception as e:
                print(e)
                return None
        else:
            return self._get(i)

    def __getitem__(self, i):
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, mask = data
        if isinstance(self.transforms, AlbumentationsCompose):
            x = np.stack([self.transforms(image=_)["image"] for _ in x])
            x = torch.from_numpy(x)
            x = rearrange(x, "n h w c -> n c h w")
        elif isinstance(self.transforms, TorchvisionCompose):
            x = [torch.from_numpy(_) for _ in x]
            x = [rearrange(_, "h w c -> c h w") for _ in x]
            x = torch.stack([self.transforms(_) for _ in x])

        y = torch.tensor(y)
        x, y = x.float(), y.float()
        input_dict = {
            "x": x,
            "y": y,
            "mask": torch.tensor(mask),
            "index": torch.tensor(i),
        }

        return input_dict
