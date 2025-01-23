"""
For bone age model
Instead of passing female as variable into nn.Embedding
Encode it as a separate image channel where female is all 255 and male is all 0
Grids the image into multiple patches, which can be used for MIL models
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset, default_collate
from typing import Dict, Optional, Tuple

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
        assert (
            "female" in df.columns
        ), f"`female` not present in DataFrame [{df.columns}]"
        self.female = df.female.values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

    @staticmethod
    def clip_outlier_pixels_and_rescale(
        img: np.ndarray, clip_bounds: Tuple[int] = (1, 99)
    ) -> np.ndarray[np.uint8]:
        lower, upper = np.percentile(img, clip_bounds)
        img = np.clip(img, lower, upper)
        img -= lower
        img /= upper - lower
        img *= 255
        img = img.astype("uint8")
        return img

    @staticmethod
    def grid_image_into_square_patches(
        img: np.ndarray, patch_size: int, num_rows: int, num_cols: int
    ) -> np.ndarray:
        assert patch_size < img.shape[0] and patch_size < img.shape[1]
        xs = np.linspace(0, img.shape[1] - patch_size, num_cols).astype("int")
        ys = np.linspace(1, img.shape[0] - patch_size, num_rows).astype("int")
        patches = []
        for xi in xs:
            for yi in ys:
                patches.append(img[yi : yi + patch_size, xi : xi + patch_size])
        return np.stack(patches)

    def load_image(self, path: str) -> np.ndarray:
        path = os.path.join(self.cfg.data_dir, path)
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        if self.cfg.clip_outlier_pixels_and_rescale:
            # some of the images are very dim with poor contrast
            img = self.clip_outlier_pixels_and_rescale(
                img, self.cfg.clip_bounds or (1, 99)
            )
        if img.ndim == 2:
            img = rearrange(img, "h w -> h w 1")
        return img

    def _get(self, i: int) -> Tuple[np.ndarray]:
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
        x = self.transforms(image=x)["image"]
        ch = np.zeros(x.shape[:2])
        if self.female[i]:
            ch[...] = 255
        ch = rearrange(ch, "h w -> h w 1")
        x = np.concatenate([x, ch], axis=2)
        x = self.grid_image_into_square_patches(
            x,
            patch_size=self.cfg.patch_size,
            num_rows=self.cfg.patch_num_rows,
            num_cols=self.cfg.patch_num_cols,
        )
        x = torch.from_numpy(x)
        x = rearrange(x, "n h w c -> n c h w")
        y = torch.tensor(y)

        x, y = x.float(), y.float()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        return input_dict
