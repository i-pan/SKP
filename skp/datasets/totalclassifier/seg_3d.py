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

label_names = {
    1: "adrenal_gland_left",
    2: "adrenal_gland_right",
    3: "aorta",
    4: "atrial_appendage_left",
    5: "autochthon_left",
    6: "autochthon_right",
    7: "brachiocephalic_trunk",
    8: "brachiocephalic_vein_left",
    9: "brachiocephalic_vein_right",
    10: "brain",
    11: "clavicula_left",
    12: "clavicula_right",
    13: "colon",
    14: "common_carotid_artery_left",
    15: "common_carotid_artery_right",
    16: "costal_cartilages",
    17: "duodenum",
    18: "esophagus",
    19: "femur_left",
    20: "femur_right",
    21: "gallbladder",
    22: "gluteus_maximus_left",
    23: "gluteus_maximus_right",
    24: "gluteus_medius_left",
    25: "gluteus_medius_right",
    26: "gluteus_minimus_left",
    27: "gluteus_minimus_right",
    28: "heart",
    29: "hip_left",
    30: "hip_right",
    31: "humerus_left",
    32: "humerus_right",
    33: "iliac_artery_left",
    34: "iliac_artery_right",
    35: "iliac_vena_left",
    36: "iliac_vena_right",
    37: "iliopsoas_left",
    38: "iliopsoas_right",
    39: "inferior_vena_cava",
    40: "kidney_cyst_left",
    41: "kidney_cyst_right",
    42: "kidney_left",
    43: "kidney_right",
    44: "liver",
    45: "lung_lower_lobe_left",
    46: "lung_lower_lobe_right",
    47: "lung_middle_lobe_right",
    48: "lung_upper_lobe_left",
    49: "lung_upper_lobe_right",
    50: "pancreas",
    51: "portal_vein_and_splenic_vein",
    52: "prostate",
    53: "pulmonary_vein",
    54: "rib_left_1",
    55: "rib_left_10",
    56: "rib_left_11",
    57: "rib_left_12",
    58: "rib_left_2",
    59: "rib_left_3",
    60: "rib_left_4",
    61: "rib_left_5",
    62: "rib_left_6",
    63: "rib_left_7",
    64: "rib_left_8",
    65: "rib_left_9",
    66: "rib_right_1",
    67: "rib_right_10",
    68: "rib_right_11",
    69: "rib_right_12",
    70: "rib_right_2",
    71: "rib_right_3",
    72: "rib_right_4",
    73: "rib_right_5",
    74: "rib_right_6",
    75: "rib_right_7",
    76: "rib_right_8",
    77: "rib_right_9",
    78: "sacrum",
    79: "scapula_left",
    80: "scapula_right",
    81: "skull",
    82: "small_bowel",
    83: "spinal_cord",
    84: "spleen",
    85: "sternum",
    86: "stomach",
    87: "subclavian_artery_left",
    88: "subclavian_artery_right",
    89: "superior_vena_cava",
    90: "thyroid_gland",
    91: "trachea",
    92: "urinary_bladder",
    93: "vertebrae_C1",
    94: "vertebrae_C2",
    95: "vertebrae_C3",
    96: "vertebrae_C4",
    97: "vertebrae_C5",
    98: "vertebrae_C6",
    99: "vertebrae_C7",
    100: "vertebrae_L1",
    101: "vertebrae_L2",
    102: "vertebrae_L3",
    103: "vertebrae_L4",
    104: "vertebrae_L5",
    105: "vertebrae_S1",
    106: "vertebrae_T1",
    107: "vertebrae_T10",
    108: "vertebrae_T11",
    109: "vertebrae_T12",
    110: "vertebrae_T2",
    111: "vertebrae_T3",
    112: "vertebrae_T4",
    113: "vertebrae_T5",
    114: "vertebrae_T6",
    115: "vertebrae_T7",
    116: "vertebrae_T8",
    117: "vertebrae_T9",
}

class_map_5_parts = {
    # 24 classes
    "class_map_part_organs": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "small_bowel",
        19: "duodenum",
        20: "colon",
        21: "urinary_bladder",
        22: "prostate",
        23: "kidney_cyst_left",
        24: "kidney_cyst_right",
    },
    # 26 classes
    "class_map_part_vertebrae": {
        1: "sacrum",
        2: "vertebrae_S1",
        3: "vertebrae_L5",
        4: "vertebrae_L4",
        5: "vertebrae_L3",
        6: "vertebrae_L2",
        7: "vertebrae_L1",
        8: "vertebrae_T12",
        9: "vertebrae_T11",
        10: "vertebrae_T10",
        11: "vertebrae_T9",
        12: "vertebrae_T8",
        13: "vertebrae_T7",
        14: "vertebrae_T6",
        15: "vertebrae_T5",
        16: "vertebrae_T4",
        17: "vertebrae_T3",
        18: "vertebrae_T2",
        19: "vertebrae_T1",
        20: "vertebrae_C7",
        21: "vertebrae_C6",
        22: "vertebrae_C5",
        23: "vertebrae_C4",
        24: "vertebrae_C3",
        25: "vertebrae_C2",
        26: "vertebrae_C1",
    },
    # 18
    "class_map_part_cardiac": {
        1: "heart",
        2: "aorta",
        3: "pulmonary_vein",
        4: "brachiocephalic_trunk",
        5: "subclavian_artery_right",
        6: "subclavian_artery_left",
        7: "common_carotid_artery_right",
        8: "common_carotid_artery_left",
        9: "brachiocephalic_vein_left",
        10: "brachiocephalic_vein_right",
        11: "atrial_appendage_left",
        12: "superior_vena_cava",
        13: "inferior_vena_cava",
        14: "portal_vein_and_splenic_vein",
        15: "iliac_artery_left",
        16: "iliac_artery_right",
        17: "iliac_vena_left",
        18: "iliac_vena_right",
    },
    # 23
    "class_map_part_muscles": {
        1: "humerus_left",
        2: "humerus_right",
        3: "scapula_left",
        4: "scapula_right",
        5: "clavicula_left",
        6: "clavicula_right",
        7: "femur_left",
        8: "femur_right",
        9: "hip_left",
        10: "hip_right",
        11: "spinal_cord",
        12: "gluteus_maximus_left",
        13: "gluteus_maximus_right",
        14: "gluteus_medius_left",
        15: "gluteus_medius_right",
        16: "gluteus_minimus_left",
        17: "gluteus_minimus_right",
        18: "autochthon_left",
        19: "autochthon_right",
        20: "iliopsoas_left",
        21: "iliopsoas_right",
        22: "brain",
        23: "skull",
    },
    # 26 classes
    # 12. ribs start from vertebrae T12
    # Small subset of population (roughly 8%) have 13. rib below 12. rib
    #  (would start from L1 then)
    #  -> this has label rib_12
    # Even smaller subset (roughly 1%) has extra rib above 1. rib   ("Halsrippe")
    #  (the extra rib would start from C7)
    #  -> this has label rib_1
    #
    # Quite often only 11 ribs (12. ribs probably so small that not found). Those
    # cases often wrongly segmented.
    "class_map_part_ribs": {
        1: "rib_left_1",
        2: "rib_left_2",
        3: "rib_left_3",
        4: "rib_left_4",
        5: "rib_left_5",
        6: "rib_left_6",
        7: "rib_left_7",
        8: "rib_left_8",
        9: "rib_left_9",
        10: "rib_left_10",
        11: "rib_left_11",
        12: "rib_left_12",
        13: "rib_right_1",
        14: "rib_right_2",
        15: "rib_right_3",
        16: "rib_right_4",
        17: "rib_right_5",
        18: "rib_right_6",
        19: "rib_right_7",
        20: "rib_right_8",
        21: "rib_right_9",
        22: "rib_right_10",
        23: "rib_right_11",
        24: "rib_right_12",
        25: "sternum",
        26: "costal_cartilages",
    },
}


def loop_translate(a: np.ndarray, d: dict) -> np.ndarray:
    n = np.ndarray(a.shape)
    for k in d:
        n[a == k] = d[k]
    return n


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
        if self.mode == "val":
            if self.cfg.val_axials_only:
                self.dfs = [
                    _df for _df in self.dfs if _df.sid_plane.str.contains("axial").any()
                ]
            if self.cfg.divide_val_samples_into_chunks:
                # divide into chunks of self.cfg.num_slices
                df_list = []
                for each_df in self.dfs:
                    each_df = each_df.reset_index(drop=True)
                    for i in range(0, len(each_df), self.cfg.num_slices):
                        df_list.append(each_df.iloc[i : i + self.cfg.num_slices])
                self.dfs = df_list
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

        if self.cfg.classes_subset is not None:
            class_map_subset = class_map_5_parts[self.cfg.classes_subset]
            class_map_subset_reverse = {v: k for k, v in class_map_subset.items()}

            self.label_map = {0: 0}
            label_subset = list(class_map_subset.values())
            for k, v in label_names.items():
                if v not in list(label_subset):
                    self.label_map[k] = 0
                else:
                    self.label_map[k] = class_map_subset_reverse[v]

    def __len__(self) -> int:
        return len(self.dfs)

    def load_image(self, path):
        path = os.path.join(self.cfg.data_dir, path)
        # load 4-channel PNG, one for each window
        assert self.cfg.cv2_load_flag == cv2.IMREAD_UNCHANGED
        img = cv2.imread(path, self.cfg.cv2_load_flag)
        assert img.shape[-1] == 4
        return img

    def _apply_aug_from_config(self, aug_attribute: float | bool | None) -> bool:
        if isinstance(aug_attribute, float):
            assert 0 < aug_attribute < 1
            return np.random.rand() < aug_attribute
        return False

    def _get(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.dfs[i].copy()
        plane = df.sid_plane.iloc[0].split("_")[1]

        ascending = True
        if self.mode == "train" and plane != "sagittal":
            # randomly reverse order if not sagittal
            # since sagittal cannot learn left vs. right otherwise
            if self._apply_aug_from_config(self.cfg.reverse_slices_aug):
                ascending = False
        df = df.sort_values("filename", ascending=ascending).reset_index(drop=True)

        if self.mode == "train":
            if self._apply_aug_from_config(self.cfg.downsample_slices_aug):
                # randomly downsample along slice axis during training
                stride = np.random.choice([1, 2, 3])
                df = df.iloc[::stride].reset_index(drop=True)

        if self.mode == "train":
            if len(df) > self.cfg.num_slices:
                # sampled_num_slices = np.random.randint(1, self.cfg.num_slices + 1)
                sampled_num_slices = self.cfg.num_slices
                start_index = np.random.randint(0, len(df) - sampled_num_slices)
                df = df.iloc[start_index : start_index + sampled_num_slices]
        else:
            if (
                self.cfg.max_val_num_slices is not None
                and len(df) > self.cfg.max_val_num_slices
            ):
                df = df.iloc[: self.cfg.max_val_num_slices]

        # load images and segmentation masks
        filepaths = df.filename.tolist()
        x = np.stack([self.load_image(path) for path in filepaths], axis=0)
        y = np.stack(
            [
                cv2.imread(
                    os.path.join(self.cfg.seg_data_dir or self.cfg.data_dir, path),
                    cv2.IMREAD_GRAYSCALE,
                )
                for path in filepaths
            ],
            axis=0,
        )

        if hasattr(self, "label_map"):
            y = loop_translate(y, self.label_map)

        # pad image stack and masks if needed
        pad_slices = (len(x) < self.cfg.num_slices) and (
            self.mode == "train" or self.cfg.pad_slices_for_val
        )
        if pad_slices:
            pad_num = self.cfg.num_slices - len(x)
            x = np.concatenate(
                [x, np.zeros((pad_num, *x.shape[1:]), dtype=x.dtype)], axis=0
            )
            y = np.concatenate(
                [y, np.zeros((pad_num, *y.shape[1:]), dtype=y.dtype)], axis=0
            )

        assert len(x) == len(y)
        if pad_slices:
            assert len(x) == self.cfg.num_slices

        if self.cfg.use_4channels:
            assert x.shape[-1] == 4
            if self.mode == "train":
                if self._apply_aug_from_config(self.cfg.channel_shuffle_aug):
                    # randomly apply channel shuffle augmentation
                    x = x[..., np.random.permutation(4)]

                if self._apply_aug_from_config(self.cfg.single_channel_aug):
                    # randomly zero out all channels but 1
                    channel_to_keep = np.random.randint(0, 4)
                    channels_to_drop = []
                    for channel_idx in range(4):
                        if channel_idx != channel_to_keep:
                            x[..., channel_idx] = 0
                            channels_to_drop.append(channel_idx)
                    # move empty channels to end
                    channel_order = [channel_to_keep] + channels_to_drop
                    assert len(channel_order) == len(np.unique(channel_order)) == 4
                    x = x[..., channel_order]

                if self._apply_aug_from_config(self.cfg.channel_dropout_aug):
                    num_channels_to_keep = np.random.randint(1, 4)
                    keep_channels = np.random.choice(
                        [0, 1, 2, 3], num_channels_to_keep, replace=False
                    )
                    keep_channels = list(keep_channels)
                    drop_channels = list(set(range(4)) - set(keep_channels))
                    for channel_idx in range(4):
                        if channel_idx not in keep_channels:
                            x[..., channel_idx] = 0
                    # move empty channels to end
                    x = x[..., keep_channels + drop_channels]
        else:
            # model accepts 1 input channel
            # each image channel is a window
            # during training, apply augmentation by randomly selecting 1-4 channels
            # if >1 channel selected, take mean
            # for validation, just use mean of 4 channels
            if self.mode == "train":
                num_channels = np.random.randint(1, 5)
                if num_channels < 4:
                    channels = np.random.choice(
                        [0, 1, 2, 3], num_channels, replace=False
                    )
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

        # apply transforms to 3D image and mask
        if self.transforms is not None:
            if self.mode == "train":
                trf = {
                    "image" if idx == 0 else f"image{idx}": img
                    for idx, img in enumerate(x)
                }
                trf.update(
                    {
                        "mask" if idx == 0 else f"mask{idx}": img
                        for idx, img in enumerate(y)
                    }
                )
                trf = self.transforms(**trf)
                x = np.stack(
                    [trf["image"]] + [trf[f"image{idx}"] for idx in range(1, len(x))],
                    axis=0,
                )
                y = np.stack(
                    [trf["mask"]] + [trf[f"mask{idx}"] for idx in range(1, len(y))],
                    axis=0,
                )
            else:
                # only resize +/- center crop being applied
                trf = [self.transforms(image=img, mask=mask) for img, mask in zip(x, y)]
                x = np.stack([trf[idx]["image"] for idx in range(len(x))], axis=0)
                y = np.stack([trf[idx]["mask"] for idx in range(len(y))], axis=0)

        x = torch.from_numpy(x)
        if self.cfg.data_format == "cthw":
            x = rearrange(x, "n h w c -> c n h w")
        else:
            x = rearrange(x, "n h w c -> n c h w")

        x, y = x.float(), torch.from_numpy(y).float()

        return {
            "x": x,
            "y": y,
            "index": torch.tensor(i),
        }
