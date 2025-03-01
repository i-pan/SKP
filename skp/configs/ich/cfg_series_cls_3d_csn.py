import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/ich/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "classification.net3d"
cfg.backbone = "csn_r26"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.dim0_strides = [1, 1, 1, 1, 1]
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.num_classes = 6
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.enable_gradient_checkpointing = True

cfg.fold = 0
cfg.dataset = "ich.series"
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
# load each grayscale image then stack
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 8
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.reverse_sequence_aug = 0.5
cfg.truncate_or_resample_sequence = "resample"
cfg.data_format = "cthw"

cfg.loss = "classification.BCEWithLogitsLoss"
cfg.loss_params = {}

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["classification.AUROC"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.num_slices = 40
cfg.image_height = 256
cfg.image_width = 256

cfg.seq_len = cfg.num_slices 

resize_transforms = [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]

additional_targets = {f"image{idx}": "image" for idx in range(1, cfg.num_slices)}
cfg.train_transforms = A.Compose(
    resize_transforms
    + [
        A.RandomCrop(
            height=int(cfg.image_height * 0.875),
            width=int(cfg.image_width * 0.875),
            p=1,
        )
    ]
    + [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.SomeOf(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.2,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.0,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1,
                ),
                A.GaussianBlur(p=1),
                A.GaussNoise(p=1),
                A.RandomBrightnessContrast(
                    contrast_limit=0.3, brightness_limit=0.0, p=1
                ),
                A.RandomBrightnessContrast(
                    contrast_limit=0.0, brightness_limit=0.3, p=1
                ),
            ],
            n=3,
            p=0.9,
            replace=False,
        ),
    ],
    additional_targets=additional_targets,
)

cfg.val_transforms = A.Compose(resize_transforms, additional_targets=additional_targets)
