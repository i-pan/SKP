"""
Loads 2Dc images for simple classification/regression tasks, along extra variable for embedding, if specified.
2Dc is mainly used for 3D images that have been converted to 2D slices.
For example, if we have a CT, we can convert it to 2Dc by taking 3 contiguous slices and forming
a pseudo RGB channel.
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset, default_collate
from typing import Dict, List, Optional, Tuple

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

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].values
        if self.cfg.vars:
            self.vars = df[self.cfg.vars].values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

    def load_image(self, path: str) -> List[np.ndarray]:
        # format is usually a string with filepaths separated by commas
        paths = path.split(",")
        paths = [os.path.join(self.cfg.data_dir, p) for p in paths]
        imgs = [cv2.imread(p, self.cfg.cv2_load_flag) for p in paths]
        imgs = [rearrange(img, "h w -> h w 1") if img.ndim == 2 else img for img in imgs]
        return imgs

    def _get(self, i: int) -> Tuple[List[np.ndarray], np.ndarray]:
        x = self.load_image(self.inputs[i])
        y = self.labels[i].copy()
        return x, y

    def get(self, i: int) -> Optional[Tuple]:
        if self.cfg.skip_failed_data:
            try:
                return self._get(i)
            except Exception as e:
                print(e)
                return None
        else:
            return self._get(i)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data
        # apply same transforms to each of the images
        x = {"image" if idx == 0 else f"image{idx}": img for idx, img in enumerate(x)}
        x_trf = self.transforms(**x)
        x = [x_trf["image"]] + [x_trf[f"image{idx}"] for idx in range(1, len(x))]
        x = np.stack(x)
        x = rearrange(x, "n h w c -> (n c) h w")

        if self.mode == "train" and self.cfg.channel_reverse_aug is not None:
            if np.random.rand() < self.cfg.channel_reverse_aug:
                x = np.ascontiguousarray(x[::-1])
                
        x = torch.from_numpy(x)
        y = torch.tensor(y)

        x, y = x.float(), y.float()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        if hasattr(self, "vars"):
            input_dict["var"] = torch.tensor(self.vars[i]).long()

        return input_dict
