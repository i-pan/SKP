import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/ich/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "classification.net2d"
cfg.backbone = "tf_efficientnetv2_b0"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.num_classes = 6
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "simple2d"
cfg.data_dir = "/mnt/stor/datasets/kaggle/rsna-intracranial-hemorrhage-detection/stage_2_train_png/"
cfg.annotations_file = "/mnt/stor/datasets/kaggle/rsna-intracranial-hemorrhage-detection/train_slices_with_2dc_kfold.csv"
cfg.inputs = "filepath"
cfg.targets = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = os.cpu_count() // 2 - 1
cfg.skip_failed_data = True
cfg.pin_memory = True
cfg.persistent_workers = True

cfg.loss = "custom.ICHLogLoss"
cfg.loss_params = {}

cfg.batch_size = 184
cfg.num_epochs = 5
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {
    "pct_start": 0.05,
    "div_factor": 100,
    "final_div_factor": 1_000,
}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["classification.AUROC"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512

resize_transforms = [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]

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
    A.Affine(shear=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(rotate=(-30, 30), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.3, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.25), p=1),
    A.Downscale(scale_range=(0.25, 0.4), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=(3, 5), p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms
    + [
        A.RandomCrop(
            height=int(0.875 * cfg.image_height),
            width=int(0.875 * cfg.image_width),
            p=1,
        )
    ]
    + [
        A.OneOf(augmentation_space, p=augment_p),
    ],
)

cfg.val_transforms = A.Compose(resize_transforms)
