"""
This dataset is intended for 3D cross-sectional imaging
where image-level labels are available.

The dataset randomly selects a slab of N slices during training.
During validation, the dataset selects the middle N slices.
Thus the validation performance will not really be representative.

However, this is a precursor step to training on the whole 3D
image. Due to memory limitations, it may not be possible to
train an end-to-end CNN-sequence model on the 3D image at once.

This reduces the memory requirements by selecting only N slices
but still allows training of a CNN-sequence model. Then,
the CNN backbone can be frozen and the model can be trained
on the whole 3D image.
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

from scipy.ndimage import zoom
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

        # group by series
        self.dfs = [_df for _, _df in df.groupby("SeriesInstanceUID")]

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.dfs)

    def _get(self, i: int) -> tuple[list[np.ndarray], np.ndarray, list[bool]]:
        df = self.dfs[i]

        sort_ascending = True
        if self.mode == "train" and self.cfg.reverse_sequence_aug is not None:
            if np.random.rand() < self.cfg.reverse_sequence_aug:
                sort_ascending = False

        df = df.sort_values(self.cfg.inputs, ascending=sort_ascending)

        files_list = df[self.cfg.inputs].tolist()
        files_list = np.asarray(
            [os.path.join(self.cfg.data_dir, f) for f in files_list]
        )

        if self.mode == "train":
            seq_len = self.cfg.seq_len
        else:
            # may want to validate using longer sequence length
            seq_len = self.cfg.val_seq_len or self.cfg.seq_len

        # vast majority of series will be longer than seq_len
        # but still handle edge cases accordingly
        if len(files_list) > seq_len:
            if self.mode == "train":
                # randomly take chunk for train
                start_index = np.random.randint(0, len(files_list) - seq_len)
            else:
                # take middle chunk for val
                start_index = (len(files_list) - seq_len) // 2

            indices = np.arange(start_index, start_index + seq_len, dtype="int")
            files_list = files_list[indices]
            pad_num = 0
            mask = [True] * seq_len
        elif len(files_list) < seq_len:
            indices = np.arange(0, len(files_list), dtype="int")
            pad_num = seq_len - len(files_list)
            mask = [True] * len(files_list) + [False] * pad_num
        elif len(files_list) == seq_len:
            indices = np.arange(0, len(files_list), dtype="int")
            pad_num = 0
            mask = [True] * len(files_list)

        x = [cv2.imread(f, self.cfg.cv2_load_flag) for f in files_list]
        if pad_num > 0:
            x.extend([np.zeros_like(x[0])] * pad_num)

        y = df[self.cfg.targets].values[indices]
        if pad_num > 0:
            pad_y = np.zeros((pad_num, *y.shape[1:]), dtype=y.dtype)
            y = np.concatenate([y, pad_y], axis=0)

        assert (
            len(x) == len(y) == len(mask) == seq_len
        ), f"x [{len(x)}], y [{len(y)}], mask [{len(mask)}] must be equal to seq_len [{seq_len}]"

        return x, y, mask

    def get(self, i: int) -> tuple | None:
        if self.cfg.skip_failed_data:
            try:
                return self._get(i)
            except Exception as e:
                print(e)
                return None
        else:
            return self._get(i)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, mask = data

        # apply same transforms to each of the images
        x = {"image" if idx == 0 else f"image{idx}": img for idx, img in enumerate(x)}
        x_trf = self.transforms(**x)
        x = np.stack(
            [x_trf["image"]] + [x_trf[f"image{idx}"] for idx in range(1, len(x))],
            axis=0,
        )
        x = torch.from_numpy(x).float()
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        x = rearrange(x, "n h w c -> n c h w")
        y = torch.from_numpy(y).float()
        mask = torch.tensor(mask)
        if self.cfg.convert_to_2dc:
            x = torch.cat([x[:-2], x[1:-1], x[2:]], dim=1)
            y = y[1:-1]
            mask = mask[1:-1]

        input_dict = {
            "x": x,
            "y_seq": y,
            "mask": mask,
            "index": torch.tensor(i),
        }

        return input_dict
