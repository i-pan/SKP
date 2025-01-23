import albumentations as A
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

        self.fg_crop = A.CropNonEmptyMaskIfExists(
            height=self.cfg.patch_size, width=self.cfg.patch_size, p=1
        )
        self.bg_crop = A.RandomCrop(self.cfg.patch_size, self.cfg.patch_size)

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self) -> int:
        return len(self.inputs)

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
            # pad on right side
            padding = ((0, 0), (0, w_diff))
            if img.ndim == 3:
                padding = padding + ((0, 0),)
            img = np.pad(img, padding, mode="constant", constant_values=0)
        return img

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
            return np.zeros(img_shape)
        img = cv2.imread(path, cv2_load_flag)
        return img

    def _get(self, i: int) -> Tuple[np.ndarray]:
        x = self.load_image(self.inputs[i], self.cfg.data_dir, self.cfg.cv2_load_flag)
        if x.ndim == 2:
            x = rearrange(x, "h w -> h w 1")
        # assumes multiclass segmentation
        # if mask has multiple label channels, this won't work
        y = self.load_image(
            self.masks[i],
            self.cfg.seg_data_dir or self.cfg.data_dir,
            cv2.IMREAD_GRAYSCALE,
            assume_empty_if_not_present=self.cfg.assume_mask_empty_if_not_present,
            img_shape=x.shape[:2],
        )
        if self.cfg.pad_to_aspect_ratio:
            x = self.pad_to_aspect_ratio(x, self.cfg.pad_to_aspect_ratio)
            y = self.pad_to_aspect_ratio(y, self.cfg.pad_to_aspect_ratio)
        if self.cfg.rescale_mask:
            # sometimes for binary masks will save as 255 instead of 1
            y = (y / self.cfg.rescale_mask).astype("int")
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

    def crop_foreground(self, x, y):
        trf = self.fg_crop(image=x, mask=y)
        x_crop, y_crop = trf["image"], trf["mask"]
        return x_crop, y_crop

    def crop_background(self, x, y, max_iter=100):
        trf = self.bg_crop(image=x, mask=y)
        x_crop, y_crop = trf["image"], trf["mask"]
        counter = 0
        while y_crop.sum() > 0 and counter < max_iter:
            trf = self.bg_crop(image=x, mask=y)
            x_crop, y_crop = trf["image"], trf["mask"]
            counter += 1
        return x_crop, y_crop

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        data = self.get(i)
        while data is None:
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data
        trf = self.transforms(image=x, mask=y)
        x, y = trf["image"], trf["mask"]

        # randomly crop either foreground or background patch
        x, y = (
            self.crop_foreground(x, y)
            if np.random.binomial(1, 0.5)
            else self.crop_background(x, y)
        )

        x = torch.from_numpy(x)
        x = rearrange(x, "h w c -> c h w")
        y = torch.tensor(y).long()

        x = x.float()
        y = torch.nn.functional.one_hot(
            y, num_classes=self.cfg.num_classes + 1
        ).float()[..., 1:]
        y = rearrange(y, "h w c -> c h w")
        # shared encoder and x goes through segmentation model first
        # so cls does not need x
        input_dict = {
            "cls": {"y": y.amax((1, 2))},
            "seg": {"x": x, "y": y},
        }
        return input_dict
