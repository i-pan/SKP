"""
For bone age model
Instead of passing female as variable into nn.Embedding
Encode it as a separate image channel where female is all 255 and male is all 0
Provide regression and classification targets
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
        assert "female" in df.columns, f"`female` not present in DataFrame [{df.columns}]"
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

    def load_image(self, path: str) -> np.ndarray:
        path = os.path.join(self.cfg.data_dir, path)
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        if self.cfg.clip_outlier_pixels_and_rescale:
            # some of the images are very dim with poor contrast
            do_clip = True
            if self.cfg.clip_as_data_aug:
                # use as data augmentation
                if self.mode != "train":
                    # if using as data aug, only do during train
                    do_clip = False
                else:
                    do_clip = np.random.binomial(1, self.cfg.clip_proba or 0.5) if self.mode == "train" else False
            if do_clip:
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
        x = torch.from_numpy(x)
        x = rearrange(x, "h w c -> c h w")
        ch = torch.zeros(x.shape[1:])
        if self.female[i]:
            ch[...] = 255
        ch = rearrange(ch, "h w -> 1 h w")
        x = torch.cat([x, ch], dim=0)
        y = {"reg": torch.tensor([y[0]]).float(), "cls": torch.tensor(y[1]).long()}
        x = x.float()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        return input_dict
