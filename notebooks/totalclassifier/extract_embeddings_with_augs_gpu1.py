#!/usr/bin/env python
# coding: utf-8

# In[40]:


import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from importlib import import_module
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from skp.toolbox.functions import load_model_from_config


# In[41]:


class ImageDataset(Dataset):
    def __init__(self, df, transforms, data_dir):
        self.transforms = transforms
        self.dfs = [_df for _, _df in df.groupby("sid_plane")]
        self.data_dir = data_dir
        self.label_names = [c for c in df.columns if c.endswith("_label")]

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, i):
        df = self.dfs[i]
        df = df.sort_values("filename", ascending=True)
        filepaths = df.filename.tolist()
        x = [
            cv2.imread(os.path.join(self.data_dir, path), cv2.IMREAD_UNCHANGED)
            for path in filepaths
        ]
        x = np.stack([self.transforms(image=img)["image"] for img in x], axis=0)
        assert x.shape[-1] == 4
        y = df[self.label_names].values.copy()
        x = torch.from_numpy(x).float().permute(0, 3, 1, 2)
        y = torch.from_numpy(y).float()
        return x, y, df.sid_plane.iloc[0]


# In[43]:


device = "cuda:1"

cfg_name = "totalclassifier.cfg_seg_cls_mpr_224"
cfg = import_module(f"skp.configs.{cfg_name}").cfg
cfg.enable_gradient_checkpointing = False
model = load_model_from_config(
    cfg,
    weights_path=cfg.save_dir + cfg_name + "/25866237/fold0/checkpoints/last.ckpt",
    device=device,
    eval_mode=True,
)


# In[44]:

df = pd.read_csv(cfg.annotations_file)
sid_planes = np.sort(df.sid_plane.unique())
sid_planes = sid_planes[len(sid_planes) // 2:]
df = df.loc[df.sid_plane.isin(list(sid_planes))].reset_index(drop=True)
df.head()


# In[45]:


dataset = ImageDataset(df, cfg.val_transforms, cfg.data_dir)
loader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=13, drop_last=False
)


# In[47]:


save_dir = "/home/ian/datasets/totalsegmentator/extracted_embeddings_for_organ_classification_with_augs_v2b0/fold0/"
os.makedirs(save_dir, exist_ok=True)


# In[50]:


inference_batch_size = 256
for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
    x, y, sid_plane = batch
    x, y, sid_plane = x[0], y[0], sid_plane[0]
    x = x.to(device)
    channel_combinations = []
    num_channels = x.shape[1]
    for r in range(1, num_channels + 1):
        for c in combinations(range(num_channels), r):
            channel_combinations.append(c)
    features = []
    for c in channel_combinations:
        with torch.inference_mode():
            tmp_x = x[:, c].mean(1, keepdim=True)
            out = torch.cat(
                [
                    model(
                        {"seg": {"x": tmp_x[bs : bs + inference_batch_size]}},
                        return_features=True,
                    )["seg"]["features"][-1].mean((2, 3))
                    for bs in range(0, len(tmp_x), inference_batch_size)
                ]
            )
            assert len(out) == len(tmp_x)
            features.append(out.cpu().numpy())
            # vflip
            tmp_x = x[:, c].flip(2).mean(1, keepdim=True)
            out = torch.cat(
                [
                    model(
                        {"seg": {"x": tmp_x[bs : bs + inference_batch_size]}},
                        return_features=True,
                    )["seg"]["features"][-1].mean((2, 3))
                    for bs in range(0, len(tmp_x), inference_batch_size)
                ]
            )
            features.append(out.cpu().numpy())
            # transpose
            tmp_x = x[:, c].transpose(2, 3).mean(1, keepdim=True)
            out = torch.cat(
                [
                    model(
                        {"seg": {"x": tmp_x[bs : bs + inference_batch_size]}},
                        return_features=True,
                    )["seg"]["features"][-1].mean((2, 3))
                    for bs in range(0, len(tmp_x), inference_batch_size)
                ]
            )
            features.append(out.cpu().numpy())
            # vflip + transpose
            tmp_x = x[:, c].flip(2).transpose(2, 3).mean(1, keepdim=True)
            out = torch.cat(
                [
                    model(
                        {"seg": {"x": tmp_x[bs : bs + inference_batch_size]}},
                        return_features=True,
                    )["seg"]["features"][-1].mean((2, 3))
                    for bs in range(0, len(tmp_x), inference_batch_size)
                ]
            )
            features.append(out.cpu().numpy())
    features = np.stack(features, axis=0)
    np.save(os.path.join(save_dir, f"{sid_plane}_features.npy"), features)
    np.save(os.path.join(save_dir, f"{sid_plane}_labels.npy"), y.cpu().numpy())
