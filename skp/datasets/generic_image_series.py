import cv2
import glob
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from scipy.ndimage import zoom
from torch.utils.data import Dataset as TorchDataset, default_collate


train_collate_fn = default_collate
val_collate_fn = default_collate


class Dataset(TorchDataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg 
        self.mode = mode
        df = pd.read_csv(self.cfg.annotations_file)
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
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs) 

    def load_image_stack(self, path):
        path = os.path.join(self.cfg.data_dir, path)
        files = glob.glob(os.path.join(path, "*"))
        files = np.sort(files)
        indices = np.arange(len(files), dtype=np.int64)
        indices = zoom(indices, zoom=self.cfg.max_seq_len / len(indices), order=0, prefilter=False)
        array = np.stack([cv2.imread(files[idx], self.cfg.cv2_load_flag) for idx in indices])
        if self.cfg.random_reverse_series and self.mode == "train":
            if np.random.binomial(1, self.cfg.random_reverse_series):
                array = array[::-1]
        assert len(array) == self.cfg.max_seq_len, f"number of images in array [{len(array)}] does not equal specified sequence length {self.cfg.max_seq_len}"
        if self.cfg.cv2_load_flag == cv2.IMREAD_GRAYSCALE:
            array = rearrange(array, "n h w -> n h w 1")
        return array 
    
    def _get(self, i):
        x = self.load_image_stack(self.inputs[i])
        y = self.labels[i]
        return x.astype(np.float32), y
    
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
        to_transform = {"image": x[0]}
        for idx in range(1, len(x)):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, len(x))], axis=0)
        x = rearrange(x, f"n h w c -> {self.cfg.dims_format or 'n c h w'}")
        x = torch.from_numpy(x)
        y = torch.tensor(y)

        return {"x": x, "y": y, "index": i}
