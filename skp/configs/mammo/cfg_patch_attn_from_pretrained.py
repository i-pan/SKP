import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/mammo/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "MIL.net2d_transformer"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.load_pretrained_model = os.path.join(
    cfg.save_dir, "mammo.cfg_ddsm_patch_attn", "f800ebc7/fold0/checkpoints/last.ckpt"
)
cfg.ignore_pretrained_linear = True
cfg.freeze_backbone = False
cfg.num_input_channels = 1
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.num_classes = 1 + 1 + 1 + 1 + 3 + 4
cfg.transformer_nhead = 16
cfg.transformer_dropout = 0.1
cfg.transformer_activation = "gelu"
cfg.transformer_num_layers = 1
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.enable_gradient_checkpointing = True

cfg.fold = 0
cfg.dataset = "mammo.cls_patch"
cfg.data_dir = (
    "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/train_cropped_png/"
)
cfg.annotations_file = (
    "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/train_kfold_cv.csv"
)
cfg.inputs = "filename"
cfg.targets = [
    "cancer",
    "biopsy",
    "invasive",
    "difficult_negative_case",
    "BIRADS",
    "density",
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.group_index_col = "breast_id_index"
cfg.num_workers = 16
cfg.pin_memory = True
cfg.persistent_workers = True

cfg.loss = "custom.MammoCancerWithAuxLosses"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["custom_cls.Mammo_AUROC"]
cfg.val_metric = "auc0_breast"
cfg.val_track = "max"

cfg.image_height = 2048
cfg.image_width = 1024
cfg.pad_to_aspect_ratio = cfg.image_height / cfg.image_width

cfg.patch_size = 512
cfg.patches_num_rows = cfg.image_height // cfg.patch_size
cfg.patches_num_cols = cfg.image_width // cfg.patch_size

resize_transforms = [A.Resize(cfg.image_height, cfg.image_width, p=1)]

augmentation_space = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Affine(scale=(0.8, 1.2), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(
        translate_percent={"x": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(
        translate_percent={"y": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Solarize(threshold_range=(0.5, 0.5), p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.5), p=1),
    A.Downscale(scale_range=(0.1, 0.25), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=(3, 5), p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms + [A.OneOf(augmentation_space, p=augment_p)], p=1
)

cfg.val_transforms = A.Compose(resize_transforms)
