"""
Loads 2D images and masks for lines and tubes segmentation (RANZCR-CLiP challenge)
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
                df = df.drop_duplicates()
                self.transforms = self.cfg.val_transforms

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df.maskfile.tolist()

        if cfg.sampling_weight_col:
            self.sampling_weights = df[cfg.sampling_weight_col].values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

    def create_mask(
        self, img_shape: Tuple[int], maskfile: str, data_dir: str
    ) -> np.ndarray:
        mask = np.zeros(img_shape + (11,), dtype=np.uint8)
        maskfiles = maskfile.split(",")
        maskfiles = [os.path.join(data_dir, _) for _ in maskfiles]
        for m in maskfiles:
            tmp_mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            ch = os.path.basename(m).split("_")[-1].split(".")[0]
            ch = int(ch)
            mask[:, :, ch] = tmp_mask
        return mask

    @staticmethod
    def load_image(path: str, data_dir: str, cv2_load_flag) -> np.ndarray:
        path = os.path.join(data_dir, path)
        img = cv2.imread(path, cv2_load_flag)
        return img

    def _get(self, i: int) -> Tuple[np.ndarray]:
        x = self.load_image(self.inputs[i], self.cfg.data_dir, self.cfg.cv2_load_flag)
        if x.ndim == 2:
            x = rearrange(x, "h w -> h w 1")
        # 11 channels, one for each class
        y = self.create_mask(
            img_shape=x.shape[:2],
            maskfile=self.labels[i],
            data_dir=self.cfg.seg_data_dir,
        )
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
        if self.cfg.add_cvc_ngt_ett_present_mask:
            cvc_present = np.expand_dims(
                y[..., [0, 3, 6]].sum(-1) > 0, axis=-1
            ).astype("uint8")
            ngt_present = np.expand_dims(
                y[..., [1, 4, 7, 9]].sum(-1) > 0, axis=-1
            ).astype("uint8")
            ett_present = np.expand_dims(
                y[..., [2, 5, 8]].sum(-1) > 0, axis=-1
            ).astype("uint8")
            y = np.concatenate(
                [y, cvc_present, ngt_present, ett_present], axis=-1
            )
        x = torch.from_numpy(x)
        x = rearrange(x, "h w c -> c h w")
        y = torch.tensor(y)
        y = rearrange(y, "h w c -> c h w")

        x = x.float()
        # default long unless otherwise specified
        y = y.float()

        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        return input_dict
