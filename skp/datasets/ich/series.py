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

    def load_stack(
        self, files_list: list[str]
    ) -> tuple[list[np.ndarray], np.ndarray, int, list[bool]]:
        # files_list should already be sorted
        files_list = np.asarray(
            [os.path.join(self.cfg.data_dir, f) for f in files_list]
        )
        indices = np.arange(0, len(files_list), dtype="int")
        pad_num = 0
        mask = [True] * len(files_list)
        if len(files_list) > self.cfg.seq_len:
            if self.cfg.truncate_or_resample_sequence == "truncate":
                if self.mode == "train":
                    start_index = np.random.randint(
                        0, len(files_list) - self.cfg.seq_len
                    )
                    indices = np.arange(
                        start_index, start_index + self.cfg.seq_len, dtype="int"
                    )
                    files_list = files_list[indices]
                else:
                    indices = np.arange(0, self.cfg.seq_len, dtype="int")
                    files_list = files_list[indices]
            elif self.cfg.truncate_or_resample_sequence == "resample":
                indices = zoom(
                    indices,
                    zoom=self.cfg.seq_len / len(indices),
                    order=0,
                    prefilter=False,
                )
                files_list = files_list[indices]
            mask = [True] * self.cfg.seq_len
        stack = [cv2.imread(f, self.cfg.cv2_load_flag) for f in files_list]
        if len(stack) < self.cfg.seq_len:
            pad_num = self.cfg.seq_len - len(stack)
            stack = stack + [np.zeros_like(stack[0])] * pad_num
            mask = mask + [False] * pad_num

        return stack, indices, pad_num, mask

    def _get(self, i: int) -> Tuple[np.ndarray, np.ndarray, list[bool]]:
        df = self.dfs[i]
        df = df.sort_values(self.cfg.inputs)
        x, indices, pad_num, mask = self.load_stack(df[self.cfg.inputs].tolist())
        y = df[self.cfg.targets].values[indices]
        if pad_num > 0:
            pad_y = np.zeros((pad_num, *y.shape[1:]), dtype=y.dtype)
            y = np.concatenate([y, pad_y], axis=0)
        assert (
            len(x) == len(y) == len(mask) == self.cfg.seq_len
        ), f"x [{len(x)}], y [{len(y)}], mask [{len(mask)}] must be equal to seq_len [{self.cfg.seq_len}]"
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

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, mask = data

        if self.mode == "train" and self.cfg.reverse_sequence_aug is not None:
            if np.random.rand() < self.cfg.reverse_sequence_aug:
                x = x[::-1]  # x is list
                y = np.ascontiguousarray(y[::-1])  # y is np.ndarray

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
        if self.cfg.data_format == "cthw":
            x = rearrange(x, "n h w c -> c n h w")
        else:
            x = rearrange(x, "n h w c -> n c h w")
        y = torch.from_numpy(y).float()
        mask = torch.tensor(mask)

        input_dict = {
            "x": x,
            "y": y.amax(0),
            "y_seq": y,
            "y_cls": y.amax(0),
            "mask": mask,
            "index": torch.tensor(i),
        }

        return input_dict
