import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch

from importlib import import_module
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from typing import Dict, Optional, Sequence

from skp.configs import Config


def create_double_cv(
    df: pd.DataFrame,
    id_column: str,
    num_inner: int,
    num_outer: int,
    stratified: Optional[str] = None,
    seed: int = 88,
) -> pd.DataFrame:
    np.random.seed(seed)
    df = df.reset_index(drop=True)
    df["outer"] = -1
    kfold_class = GroupKFold if stratified is None else StratifiedGroupKFold
    if stratified is None:
        stratified = id_column
    else:
        print(f"Stratifying CV folds based on `{stratified}` ...")
    outer_kfold = kfold_class(n_splits=num_outer)
    outer_split = outer_kfold.split(
        X=df[id_column], y=df[stratified], groups=df[id_column]
    )
    for outer_fold, (outer_train, outer_valid) in enumerate(outer_split):
        df.loc[outer_valid, "outer"] = outer_fold
        df[f"inner{outer_fold}"] = -1
        inner_df = df[df.outer != outer_fold].copy()
        inner_kfold = kfold_class(n_splits=num_inner)
        inner_split = inner_kfold.split(
            X=inner_df[id_column], y=inner_df[stratified], groups=inner_df[id_column]
        )
        for inner_fold, (inner_train, inner_valid) in enumerate(inner_split):
            inner_df = inner_df.reset_index(drop=True)
            inner_valid_ids = inner_df.loc[inner_valid, id_column].tolist()
            df.loc[df[id_column].isin(inner_valid_ids), f"inner{outer_fold}"] = (
                inner_fold
            )
    # Do a few checks
    for outer_fold in df.outer.unique():
        train_df = df.loc[df.outer != outer_fold]
        valid_df = df.loc[df.outer == outer_fold]
        assert (
            len(set(train_df[id_column].tolist()) & set(valid_df[id_column].tolist()))
            == 0
        )
        inner_col = f"inner{outer_fold}"
        for inner_fold in df[inner_col].unique():
            inner_train = train_df[train_df[inner_col] != inner_fold]
            inner_valid = train_df[train_df[inner_col] == inner_fold]
            assert (
                len(
                    set(inner_train[id_column].tolist())
                    & set(inner_valid[id_column].tolist())
                )
                == 0
            )
        assert valid_df[f"inner{outer_fold}"].unique() == np.asarray([-1])
    df["fold"] = df["outer"]
    return df


def load_weights_from_path(path: str) -> Dict[str, torch.Tensor]:
    w = torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)[
        "state_dict"
    ]
    w = {
        re.sub(r"^model.", "", k): v
        for k, v in w.items()
        if k.startswith("model.") and "criterion" not in k
    }
    return w


def load_model_from_config(
    cfg: Config,
    weights_path: Optional[str] = None,
    device: str = "cpu",
    eval_mode: bool = True,
) -> torch.nn.Module:
    model = import_module(f"skp.models.{cfg.model}").Net(cfg)
    if weights_path:
        print(f"Loading weights from {weights_path} ...")
        weights = load_weights_from_path(weights_path)
        model.load_state_dict(weights)
    model = model.to(device).train(mode=not eval_mode)
    return model


def load_kfold_ensemble_as_list(
    cfg: Config,
    weights_paths: Sequence[str],
    device: str = "cpu",
    eval_mode: bool = True,
) -> torch.nn.ModuleList:
    # multiple folds for the same model
    # does not work for ensembling different types of models
    # assumes that trained weights are available
    # otherwise why would you load multiple of the same model randomly initialized
    model_list = torch.nn.ModuleList()
    for each_weight in weights_paths:
        model = load_model_from_config(cfg, each_weight, device, eval_mode)
        model_list.append(model)
    return model_list


def draw_bounding_boxes(
    img: NDArray, bboxes: Sequence[int], mode: str = "xyxy"
) -> NDArray:
    assert mode in {"xyxy", "xywh"}, f"mode [{mode}] must be `xyxy` or `xywh`"
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for box in bboxes:
        if mode == "xyxy":
            x1, y1, x2, y2 = box
        elif mode == "xywh":
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
        img = cv2.rectangle(
            img, (x1, y1), (x2, y2), (255, 0, 0), int(0.005 * max(img.shape))
        )
    return img


def overlay_images(image: NDArray, overlay: NDArray, alpha: float = 0.7) -> NDArray:
    overlaid = alpha * image + (1 - alpha) * overlay
    overlaid = overlaid.astype(np.uint8)
    return overlaid


def plot_3d_image(arr: NDArray, num_images: int, axis: int, cmap: str = "gray") -> None:
    for i in range(0, arr.shape[axis], arr.shape[axis] // num_images):
        if axis == 0:
            img = arr[i]
        elif axis == 1:
            img = arr[:, i]
        elif axis == 2:
            img = arr[:, :, i]
        plt.imshow(img, cmap=cmap)
        plt.show()


def plot_3d_image_side_by_side(
    arr1: NDArray, arr2: NDArray, num_images: int, axis: int, cmap: str = "gray"
) -> None:
    assert arr1.shape[:3] == arr2.shape[:3], f"{arr1.shape} does not match {arr2.shape}"
    for i in range(0, arr1.shape[axis], arr1.shape[axis] // num_images):
        if axis == 0:
            img1, img2 = arr1[i], arr2[i]
        elif axis == 1:
            img1, img2 = arr1[:, i], arr2[:, i]
        elif axis == 2:
            img1, img2 = arr1[:, :, i], arr2[:, :, i]
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap=cmap)
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap=cmap)
        plt.show()


def window(x: NDArray, WL: int, WW: int) -> NDArray[np.uint8]:
    # applying windowing to CT
    lower, upper = WL - WW // 2, WL + WW // 2
    x = np.clip(x, lower, upper)
    x = (x - lower) / (upper - lower)
    return (x * 255.0).astype("uint8")


def convert_to_2dc(list_of_files: Sequence[str], size: int = 3) -> Sequence[str]:
    """
    This function converts a list of single image files into "2Dc" format.

    e.g.,
    [a, b, c, d, e] -> [a,a,b, a,b,c, b,c,d, c,d,e, d,e,e]

    Assumes list_of_files is already SORTED.
    """
    original_length = len(list_of_files)
    if not isinstance(list_of_files, list):
        list_of_files = list(list_of_files)
    assert size % 2 == 1, f"size [{size}] should be an odd number"
    pad = size // 2
    # Duplicate first and last slices at the beginning and end of the list
    list_of_files = [list_of_files[0]] * pad + list_of_files + [list_of_files[-1]] * pad
    list_of_files_2dc = []
    for i in range(original_length):
        list_of_files_2dc.append(list_of_files[i : i + size])
    return list_of_files_2dc


def count_parameters(module: torch.nn.Module) -> int:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    return num_params
