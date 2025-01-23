import cv2
import glob
import numpy as np
import os
import pandas as pd
import torch

from albumentations import Compose as AlbumentationsCompose
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset, default_collate
from torchvision.transforms.v2 import Compose as TorchvisionCompose
from typing import Tuple

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
        self.study_index = df.study_index.values

        if self.mode == "train":
            self.df_list = [_df for _, _df in df.groupby("study_path")]

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        if self.mode == "train":
            return len(self.df_list)

        return len(self.inputs)

    @staticmethod
    def clip_outlier_pixels_and_rescale(
        img: np.ndarray, clip_bounds: Tuple[int] = (1, 99)
    ) -> np.ndarray[np.uint8]:
        lower, upper = np.percentile(img, clip_bounds)
        img = np.clip(img, lower, upper)
        img -= lower
        img /= upper - lower
        img *= 255
        img = img.astype("uint8")
        return img

    def load_image(self, path):
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        if img.ndim == 2:
            img = rearrange(img, "h w -> h w 1")
        return img

    def _get(self, i):
        if self.mode == "train":
            # sample 1 image from study
            study = self.df_list[i]
            study_path = study.study_path.values[0]
            images = glob.glob(os.path.join(self.cfg.data_dir, study_path, "*.png"))
            sampled_image = np.random.choice(images)
            x = self.load_image(sampled_image)
            y = study[self.cfg.targets].values
            for c in range(y.shape[1]):
                # all images in study should have same label
                assert len(np.unique(y[:, c])) == 1
            y = y[0]
        else:
            x = self.load_image(os.path.join(self.cfg.data_dir, self.inputs[i]))
            y = self.labels[i].copy()
        if self.cfg.clip_outlier_pixels is not None:
            x = self.clip_outlier_pixels_and_rescale(x, self.cfg.clip_outlier_pixels)
        return x, y

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

        x, y = data
        if isinstance(self.transforms, AlbumentationsCompose):
            x = self.transforms(image=x)["image"]
            x = torch.from_numpy(x)
            x = rearrange(x, "h w c -> c h w")
        elif isinstance(self.transforms, TorchvisionCompose):
            x = torch.from_numpy(x)
            x = rearrange(x, "h w c -> c h w")
            x = self.transforms(x)

        y = torch.tensor(y)
        x, y = x.float(), y.float()
        input_dict = {"x": x, "y": y, "index": torch.tensor(i)}
        if self.mode != "train":
            input_dict["study_index"] = torch.tensor(self.study_index[i])

        return input_dict
