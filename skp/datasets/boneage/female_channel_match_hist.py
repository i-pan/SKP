"""
For bone age model
Instead of passing female as variable into nn.Embedding
Encode it as a separate image channel where female is all 255 and male is all 0

Loads a reference image for histogram matching
This may be helpful for applications such as bone age prediction where
images may be of very different contrast levels
Although this may also increase overfitting by reducing variability in the
training data, which can be somewhat compensated for with more aggressive
augmentations
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from skimage.exposure import match_histograms
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
        self.ref_img = cv2.imread(self.cfg.ref_image_match_hist, self.cfg.cv2_load_flag)

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

    def load_image(self, path: str) -> np.ndarray:
        path = os.path.join(self.cfg.data_dir, path)
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        do_match_hist = True
        if self.cfg.match_hist_as_train_aug and self.mode == "train":
            # as aug with proba 0.5 during train
            do_match_hist = np.random.binomial(1, 0.5)
        # always on during val
        if do_match_hist:
            img = match_histograms(img, self.ref_img).astype("uint8")
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
        y = torch.tensor(y)

        x, y = x.float(), y.float()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        return input_dict
