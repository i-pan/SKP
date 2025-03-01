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

        self.dfs = [_df for _, _df in df.groupby("sid_plane")]

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.dfs)

    def _get(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # each df contains all the augs for that feature
        if self.mode == "train":
            row = self.dfs[i].sample(1).iloc[0]
        else:
            tmp_df = (
                self.dfs[i]
                .reset_index(drop=True)
                .sort_values("features_npy", ascending=True)
            )
            row = tmp_df.iloc[14]
        sid = row["sid"]
        x = np.load(os.path.join(self.cfg.data_dir, row[self.cfg.inputs]))
        y = np.load(os.path.join(self.cfg.data_dir, row[self.cfg.targets]))
        assert x.shape[0] == y.shape[0]
        if self.mode == "train":
            # randomly downsample along slice axis during training
            stride = np.random.choice([1, 2, 3, 4])
            x = x[::stride]
            y = y[::stride]
        if "sagittal" not in sid and self.mode == "train":
            # reverse sequence aug if not sagittal
            # since sagittal needs this to learn left-right
            if self.cfg.reverse_sequence_aug is not None:
                if np.random.rand() < self.cfg.reverse_sequence_aug:
                    x = np.ascontiguousarray(x[::-1])
                    y = np.ascontiguousarray(y[::-1])
        mask = np.ones(x.shape[0], dtype=bool)
        if self.mode == "train":
            # sample random subsequences during training
            seq_len = np.random.randint(1, min(self.cfg.seq_len + 1, x.shape[0] + 1))
            max_seq_len = self.cfg.seq_len
            if seq_len < x.shape[0]:
                start_index = np.random.randint(0, x.shape[0] - seq_len)
                x = x[start_index : start_index + seq_len]
                y = y[start_index : start_index + seq_len]
                mask = mask[start_index : start_index + seq_len]
            if x.shape[0] < max_seq_len:
                pad_num = max_seq_len - x.shape[0]
                x = np.concatenate(
                    [x, np.zeros((pad_num, *x.shape[1:]), dtype=x.dtype)], axis=0
                )
                y = np.concatenate(
                    [y, np.zeros((pad_num, *y.shape[1:]), dtype=y.dtype)], axis=0
                )
                mask = np.concatenate([mask, [False] * pad_num], axis=0)
        return x, y, mask

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

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, mask = data

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
        mask = torch.from_numpy(mask)

        # resample if length greater than self.cfg.seq_len
        if x.shape[0] > self.cfg.seq_len:
            x = self.interpolate(x, self.cfg.seq_len)
            y = self.interpolate(y, self.cfg.seq_len)
            y = y.round()
            mask = torch.ones(self.cfg.seq_len, dtype=bool)

        input_dict = {
            "x": x,
            "y_seq": y,
            "mask": mask,
            "index": torch.tensor(i),
        }

        return input_dict
