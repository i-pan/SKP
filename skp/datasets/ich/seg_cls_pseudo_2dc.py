"""
Loads CT slices in 2Dc format and soft segmentation masks for
classification-segmentation tasks.
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from torch.utils.data import Dataset as TorchDataset, default_collate
from typing import Dict, List, Tuple

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
        self.masks = df[
            ["any_mask", "edh_mask", "iph_mask", "ivh_mask", "sah_mask", "sdh_mask"]
        ].values
        self.labels = df[self.cfg.targets].values
        if cfg.sampling_weight_col:
            self.sampling_weights = df[cfg.sampling_weight_col].values

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

    def _get(self, i: int) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        x = self.load_image(self.inputs[i])
        y_seg = np.zeros(x[0].shape[:2])
        y_seg = np.stack([y_seg] * 6, axis=-1)
        for bleed_idx in range(6):
            fp = self.masks[i, bleed_idx]
            if fp == "empty":
                continue
            y_seg[:, :, bleed_idx] = cv2.imread(
                os.path.join(self.cfg.seg_data_dir, fp), cv2.IMREAD_GRAYSCALE
            )

        y_cls = self.labels[i].copy()
        if np.sum(y_cls) == 0:
            assert np.sum(y_seg) == 0
        return x, y_cls, y_seg

    def get(self, i: int) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray] | None:
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

        x, y_cls, y_seg = data
        # apply same transforms to each of the images
        x = {"image" if idx == 0 else f"image{idx}": img for idx, img in enumerate(x)}
        x["mask"] = y_seg
        x_trf = self.transforms(**x)
        x = [x_trf["image"]] + [
            x_trf[f"image{idx}"] for idx in range(1, len(x) - 1)
        ]  # subtract 1 for mask
        x = np.stack(x)
        x = rearrange(x, "n h w c -> (n c) h w")
        y_seg = x_trf["mask"]
        y_seg = rearrange(y_seg, "h w c -> c h w")

        if self.mode == "train" and self.cfg.channel_reverse_aug is not None:
            if np.random.rand() < self.cfg.channel_reverse_aug:
                x = np.ascontiguousarray(x[::-1])

        x = torch.from_numpy(x).float()
        y_cls = torch.tensor(y_cls).float()
        y_seg = torch.from_numpy(y_seg / 255.0).float()

        if y_seg.sum() > 0:
            # no need to do this for empty masks
            if self.cfg.binarize_pseudolabels is not None:
                assert self.cfg.temperature_scaling is None
                assert len(self.cfg.binarize_pseudolabels) == 6
                for bleed_idx in range(6):
                    y_seg[bleed_idx] = (
                        y_seg[bleed_idx] >= self.cfg.binarize_pseudolabels[bleed_idx]
                    ).float()

            if self.cfg.temperature_scaling is not None:
                assert self.cfg.binarize_pseudolabels is None
                # convert probas to 2 channels
                y_seg = torch.stack([y_seg, 1 - y_seg], dim=0)
                # apply temperature scaling
                if self.cfg.temperature_scaling_threshold is not None:
                    # only apply temperature scaling to values exceeding
                    # specified threshold
                    select = y_seg >= self.cfg.temperature_scaling_threshold
                    y_seg[select] = y_seg[select] ** (1 / self.cfg.temperature_scaling)
                else:
                    y_seg = y_seg ** (1 / self.cfg.temperature_scaling)
                # re-normalize
                y_seg = y_seg / y_seg.sum(dim=0, keepdim=True)
                # back to 1 channel
                y_seg = y_seg[0]

        # shared encoder and x goes through segmentation model first
        # so cls does not need x
        input_dict = {
            "cls": {"y": y_cls},
            "seg": {
                "x": x,
                "y": y_seg,
            },
        }
        return input_dict
