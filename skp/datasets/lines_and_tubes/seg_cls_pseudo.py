"""
Loads 2D images and masks for combined segmentation-classification tasks.
Includes model-generated mask pseudolabels for samples without ground truth masks.
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
        self.masks = df[self.cfg.masks].tolist()
        self.labels = df[self.cfg.targets].values
        if cfg.sampling_weight_col:
            self.sampling_weights = df[cfg.sampling_weight_col].values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

    @staticmethod
    def load_image(
        path: str,
        data_dir: str,
        cv2_load_flag,
        img_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        path = os.path.join(data_dir, path)
        img = cv2.imread(path, cv2_load_flag)
        return img

    def _get(self, i: int) -> Tuple[np.ndarray]:
        x = self.load_image(self.inputs[i], self.cfg.data_dir, self.cfg.cv2_load_flag)
        if x.ndim == 2:
            x = rearrange(x, "h w -> h w 1")
        # mask should be 4 channels saved as PNG RGBA
        y = cv2.imread(os.path.join(self.cfg.seg_data_dir or self.cfg.data_dir, self.masks[i]), cv2.IMREAD_UNCHANGED)
        assert y.shape[2] == 4
        if self.cfg.exclude_swan_ganz:
            # Swan Ganz is last (4th) channel
            y = y[..., :3]
        return x, y

    def get(self, i: int) -> Optional[Tuple[np.ndarray]]:
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
        trf = self.transforms(image=x, mask=y)
        x, y = trf["image"], trf["mask"]
        x = torch.from_numpy(x)
        x = rearrange(x, "h w c -> c h w")
        y = torch.tensor(y)
        y = rearrange(y, "h w c -> c h w")
        x = x.float()
        y = y.float()

        sample_weight = 1.0
        pseudolabel = False
        if "masks-pseudo" in self.masks[i]:
            y = y / 255.0
            sample_weight = self.cfg.mask_pseudolabel_sample_weight or 0.5
            pseudolabel = True

        # shared encoder and x goes through segmentation model first
        # so cls does not need x
        input_dict = {
            "cls": {"y": torch.tensor(self.labels[i])},
            "seg": {
                "x": x,
                "y": y,
                "sample_weight": torch.tensor(sample_weight),
                "pseudolabel": pseudolabel,
            },
        }
        return input_dict
