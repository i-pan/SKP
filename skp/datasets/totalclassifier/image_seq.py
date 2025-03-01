import cv2
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset, default_collate

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

        self.dfs = [_df for _, _df in df.groupby("sid_plane")]
        self.label_names = [c for c in df.columns if c.endswith("_label")]
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.dfs)

    def load_image(self, path: str) -> np.ndarray:
        path = os.path.join(self.cfg.data_dir, path)
        # load 4-channel PNG, one for each window
        assert self.cfg.cv2_load_flag == cv2.IMREAD_UNCHANGED
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        assert img.shape[-1] == 4
        return img

    def _get(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.dfs[i]
        plane = df.sid_plane.iloc[0].split("_")[1]
        ascending = True
        if plane != "sagittal":
            # randomly reverse order if not sagittal
            # since sagittal cannot learn left vs. right otherwise
            if self.mode == "train" and np.random.rand() < 0.5:
                ascending = False
        df = df.sort_values("filename", ascending=ascending)
        filepaths = df.filename.tolist()
        x = np.stack([self.load_image(path) for path in filepaths], axis=0)
        y = df[self.label_names].values.copy()

        mask = np.ones(len(x), dtype=bool)
        if self.mode == "train":
            # randomly sample sequence length from 1 to self.cfg.seq_len
            max_seq_len = self.cfg.seq_len
            seq_len = np.random.randint(1, max_seq_len + 1)
            if len(x) < seq_len:
                pad_num = seq_len - len(x)
                x = np.concatenate(
                    [x, np.zeros((pad_num, *x.shape[1:]), dtype=x.dtype)], axis=0
                )
                y = np.concatenate(
                    [y, np.zeros((pad_num, *y.shape[1:]), dtype=y.dtype)], axis=0
                )
                mask = np.concatenate([mask, [False] * pad_num], axis=0)
            elif len(x) > seq_len:
                start_index = np.random.randint(0, len(x) - seq_len)
                x = x[start_index : start_index + seq_len]
                y = y[start_index : start_index + seq_len]
                mask = mask[start_index : start_index + seq_len]
            # then pad to max_seq_len, if needed
            if len(x) < max_seq_len:
                pad_num = max_seq_len - len(x)
                x = np.concatenate(
                    [x, np.zeros((pad_num, *x.shape[1:]), dtype=x.dtype)], axis=0
                )
                y = np.concatenate(
                    [y, np.zeros((pad_num, *y.shape[1:]), dtype=y.dtype)], axis=0
                )
                mask = np.concatenate([mask, [False] * pad_num], axis=0)

        if self.cfg.use_4channels:
            return x, y, mask

        # model accepts 1 input channel
        # each image channel is a window
        # during training, apply augmentation by randomly selecting 1-4 channels
        # if >1 channel selected, take mean
        # for validation, just use mean of 4 channels
        if self.mode == "train":
            num_channels = np.random.randint(1, 5)
            if num_channels < 4:
                channels = np.random.choice([0, 1, 2, 3], num_channels, replace=False)
                x = x[..., channels]
            if x.shape[-1] > 1:
                x = np.mean(x, axis=-1, keepdims=True)
        else:
            x = np.mean(x, axis=-1, keepdims=True)
        return x.astype("uint8"), y, mask

    def get(self, i):
        if self.cfg.skip_failed_data:
            try:
                return self._get(i)
            except Exception as e:
                print(e)
                return None
        else:
            return self._get(i)

    def __getitem__(self, i):
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, mask = data
        if self.mode == "train":
            x = {
                "image" if idx == 0 else f"image{idx}": img for idx, img in enumerate(x)
            }
            x_trf = self.transforms(**x)
            x = np.stack(
                [x_trf["image"]] + [x_trf[f"image{idx}"] for idx in range(1, len(x))],
                axis=0,
            )
        else:
            # only resize being applied
            x = [self.transforms(image=img)["image"] for img in x]
            x = np.stack(x, axis=0)

        x = torch.from_numpy(x)
        x = rearrange(x, "n h w c -> n c h w")

        x, y = x.float(), torch.from_numpy(y).float()
        mask = torch.from_numpy(mask)

        return {
            "x": x,
            "y_seq": y,
            "mask": mask,
            "index": torch.tensor(i),
        }
