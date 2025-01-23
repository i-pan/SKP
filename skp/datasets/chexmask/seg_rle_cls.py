"""
Loads 2D images and masks for combined segmentation-classification tasks.
e.g., pneumonia segmentation and classification
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
        if self.cfg.neptune_mode == "debug": 
            # file is very big... so if debugging just load first 100
            df = pd.read_csv(self.cfg.annotations_file, nrows=100)
        else:
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
        self.masks = df[self.cfg.masks].values
        self.mask_present = df.mask_present.tolist()
        self.labels = df[self.cfg.targets].values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

    @staticmethod
    def rle2mask(rle: str, width: int, height: int) -> np.ndarray:
        runs = np.array([int(x) for x in rle.split()])
        starts = runs[::2]
        lengths = runs[1::2]

        mask = np.zeros((height * width), dtype=np.uint8)

        for start, length in zip(starts, lengths):
            start -= 1
            end = start + length
            mask[start:end] = 1

        mask = mask.reshape((height, width))

        return mask

    @staticmethod
    def load_image(
        path: str,
        data_dir: str,
        cv2_load_flag,
        assume_empty_if_not_present: bool = False,
        img_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        path = os.path.join(data_dir, path)
        if assume_empty_if_not_present and not os.path.exists(path):
            return np.zeros(img_shape, dtype=np.uint8)
        img = cv2.imread(path, cv2_load_flag)
        return img

    def create_mask(self, i: int, height: int, width: int) -> np.ndarray:
        if not self.mask_present[i]:
            return np.zeros((height, width), dtype=np.uint8)
        rles = self.masks[i]
        mask = np.zeros((height, width), dtype=np.uint8)
        for rle_idx, each_rle in enumerate(rles):
            tmp_mask = self.rle2mask(each_rle, width=width, height=height)
            # there is some overlap between cardiac silhouette mask
            # and lungs; since heart is going to be at the end of the
            # list, then heart will supersede the lungs, which is fine
            mask[tmp_mask == 1] = rle_idx + 1
        return mask


    def _get(self, i: int) -> Tuple[np.ndarray]:
        x = self.load_image(self.inputs[i], self.cfg.data_dir, self.cfg.cv2_load_flag)
        if x.ndim == 2:
            x = rearrange(x, "h w -> h w 1")
        y = self.create_mask(i, height=x.shape[0], width=x.shape[1])
        assert x.shape[:2] == y.shape[:2]
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

        x = x.float()
        # default long unless otherwise specified
        y = y.type(self.cfg.label_dtype) if self.cfg.label_dtype else y.long()

        # shared encoder and x goes through segmentation model first
        # so cls does not need x
        input_dict = {
            "cls": {"y": torch.tensor(self.labels[i]), "valid_view": "chexpert" in self.inputs[i]},
            "seg": {"x": x, "y": y,  "mask_present": self.mask_present[i]},
        }
        return input_dict
