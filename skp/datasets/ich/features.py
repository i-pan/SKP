"""
Loads embeddings and labels to perform sequence modeling.
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

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
        if cfg.double_cv is not None and mode != "inference":
            # double_cv is an int representing the outer fold
            # exclude the outer fold before doing train/val split
            # as the outer fold will be the test set
            # then reassign fold column to inner
            df = df[df.outer != cfg.double_cv]
            df["fold"] = df[f"inner{cfg.double_cv}"]
        if self.mode == "train":
            df = df[df.fold != self.cfg.fold]
        elif self.mode == "val":
            df = df[df.fold == self.cfg.fold]

        # both are lists of numpy files
        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].tolist()

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

    def load_image(self, path: str) -> List[np.ndarray]:
        # format is usually a string with filepaths separated by commas
        paths = path.split(",")
        paths = [os.path.join(self.cfg.data_dir, p) for p in paths]
        imgs = [cv2.imread(p, self.cfg.cv2_load_flag) for p in paths]
        imgs = [
            rearrange(img, "h w -> h w 1") if img.ndim == 2 else img for img in imgs
        ]
        return imgs

    def _get(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.load(os.path.join(self.cfg.data_dir, self.inputs[i]))
        y = np.load(os.path.join(self.cfg.data_dir, self.labels[i]))
        assert len(x) == len(
            y
        ), f"length of x [{len(x)}] and y [{len(y)}] must be equal"
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

    @staticmethod
    def interpolate(x: torch.Tensor, size: int) -> torch.Tensor:
        x = rearrange(x, "n c -> 1 c n")
        x = F.interpolate(x, size=(size,), mode="linear")
        x = rearrange(x, "1 c n -> n c")
        return x

    def pad_truncate_resample_sequence(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = self.cfg.seq_len

        # mask is 1 if PADDING, 0 otherwise
        mask = [0] * x.shape[0]
        if x.shape[0] < seq_len:
            pad_x = torch.zeros((seq_len - x.shape[0], *x.shape[1:]), dtype=x.dtype)
            pad_y = torch.zeros((seq_len - y.shape[0], *y.shape[1:]), dtype=y.dtype)
            x = torch.cat([x, pad_x], dim=0)
            y = torch.cat([y, pad_y], dim=0)
            mask = mask + [1] * pad_x.shape[0]
        elif x.shape[0] > seq_len:
            if self.cfg.truncate_or_resample_sequence == "truncate":
                x = x[:seq_len]
                y = y[:seq_len]
                mask = mask[:seq_len]
            elif self.cfg.truncate_or_resample_sequence == "resample":
                x = self.interpolate(x, seq_len)
                y = self.interpolate(y, seq_len)
                y = y.round()
                mask = mask[:seq_len]

        mask = torch.tensor(mask, dtype=torch.bool)
        assert (
            len(x) == len(y) == len(mask) == seq_len
        ), f"x [{len(x)}], y [{len(y)}], mask [{len(mask)}] must be equal to seq_len [{seq_len}]"
        return x, y, mask

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data

        if self.mode == "train":
            if self.cfg.reverse_sequence_aug is not None:
                if np.random.rand() < self.cfg.reverse_sequence_aug:
                    x = np.ascontiguousarray(x[::-1])
                    y = np.ascontiguousarray(y[::-1])
            if self.cfg.add_embedding_noise_aug is not None:
                # Based on https://arxiv.org/pdf/2310.05914
                # try alpha = {1, 5, 10, 15}
                if np.random.rand() < self.cfg.add_embedding_noise_aug:
                    L, d = x.shape
                    noise = np.random.uniform(-1, 1, x.shape)
                    # scale noise
                    noise = noise * (self.cfg.noise_alpha / (L * d) ** 0.5)
                    x = x + noise

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x, y, mask = self.pad_truncate_resample_sequence(x, y)

        input_dict = {
            "x": x,
            "y_seq": y,
            "y_cls": y.amax(0),
            "mask": mask,
            "index": torch.tensor(i),
        }

        return input_dict
