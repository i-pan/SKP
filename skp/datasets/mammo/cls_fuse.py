"""
Loads 2D images for simple classification/regression tasks, along extra variable for embedding, if specified.
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch

from albumentations import Compose as AlbumentationsCompose
from collections import defaultdict
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset, default_collate
from torchvision.transforms.v2 import Compose as TorchvisionCompose

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

        if self.cfg.vars:
            self.vars = df[self.cfg.vars].values

        if self.cfg.sampling_weight_col:
            self.sampling_weights = df[self.cfg.sampling_weight_col].values

        df_list = [_df for _, _df in df.groupby(self.cfg.group_index_col)]
        self.dict_list = []
        for _df in df_list:
            tmp_dict = defaultdict(list)
            for row_idx, row in _df.iterrows():
                tmp_dict[row["view"]].append(row[self.cfg.inputs])
            # labels should be same across views
            tmp_dict["label"] = _df[self.cfg.targets].values[0]
            self.dict_list.append(tmp_dict)

        # for this dataset, each sample is its own group
        # this is for compatibility with grouped metrics
        self.group_index = np.arange(len(self.dict_list))

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.dict_list)

    @staticmethod
    def pad_to_aspect_ratio(img: np.ndarray, aspect_ratio: float) -> np.ndarray:
        """
        Pads to specified aspect ratio, only if current aspect ratio is
        greater.
        """
        h, w = img.shape[:2]
        if h / w > aspect_ratio:
            new_w = round(h / aspect_ratio)
            w_diff = new_w - w
            left_pad = w_diff // 2
            right_pad = w_diff - left_pad
            padding = ((0, 0), (left_pad, right_pad))
            if img.ndim == 3:
                padding = padding + ((0, 0),)
            img = np.pad(img, padding, mode="constant", constant_values=0)
        return img

    def load_image(self, path: str) -> np.ndarray:
        path = os.path.join(self.cfg.data_dir, path)
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        if self.cfg.pad_to_aspect_ratio:
            img = self.pad_to_aspect_ratio(img, self.cfg.pad_to_aspect_ratio)
        if img.ndim == 2:
            img = rearrange(img, "h w -> h w 1")
        return img

    def load_cc_and_mlo_views(self, i) -> tuple[np.ndarray, np.ndarray]:
        cc_images = self.dict_list[i]["CC"]
        mlo_images = self.dict_list[i]["MLO"]
        if len(cc_images) == 0:
            cc_images = mlo_images
        if len(mlo_images) == 0:
            mlo_images = cc_images
        cc = np.random.choice(cc_images)
        mlo = np.random.choice(mlo_images)
        cc = self.load_image(cc)
        mlo = self.load_image(mlo)
        return cc, mlo

    def _get(self, i: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        x = self.load_cc_and_mlo_views(i)
        y = self.dict_list[i]["label"].copy()
        return x, y

    def get(self, i: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray] | None:
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

        x, y = data
        if isinstance(self.transforms, AlbumentationsCompose):
            x = [self.transforms(image=_)["image"] for _ in x]
            x = torch.from_numpy(np.stack(x))
            x = rearrange(x, "n h w c -> n c h w")
        elif isinstance(self.transforms, TorchvisionCompose):
            x = torch.stack([self.transforms(_) for _ in x])

        y = torch.tensor(y)
        x, y = x.float(), y.float()
        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}

        if hasattr(self, "vars"):
            input_dict["var"] = torch.tensor(self.vars[i]).long()

        if hasattr(self, "group_index"):
            input_dict["group_index"] = torch.tensor(self.group_index[i])

        return input_dict
