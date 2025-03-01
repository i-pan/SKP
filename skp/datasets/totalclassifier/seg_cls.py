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

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].tolist()  # path to seg file
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs)

    def load_image(self, path):
        path = os.path.join(self.cfg.data_dir, path)
        # load 4-channel PNG, one for each window
        assert self.cfg.cv2_load_flag == cv2.IMREAD_UNCHANGED
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        assert img.shape[-1] == 4
        return img

    def _get(self, i):
        x = self.load_image(self.inputs[i])
        y = cv2.imread(
            os.path.join(self.cfg.seg_data_dir or self.cfg.data_dir, self.labels[i]),
            cv2.IMREAD_GRAYSCALE,
        )
        if self.cfg.use_4channels:
            return x, y
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
        return x.astype("uint8"), y

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
        trf = self.transforms(image=x, mask=y)
        x, y = trf["image"], trf["mask"]
        x = torch.from_numpy(x)
        x = rearrange(x, "h w c -> c h w")

        y_cls = np.zeros((117,))
        seg_labels = np.unique(y)
        seg_labels = seg_labels[seg_labels != 0] - 1
        y_cls[seg_labels] = 1

        x, y_seg = x.float(), torch.from_numpy(y).long()

        assert x.shape[1:] == y_seg.shape
        
        return {
            "cls": {"y": y_cls},
            "seg": {"x": x, "y": y_seg},
            "index": torch.tensor(i),
        }
