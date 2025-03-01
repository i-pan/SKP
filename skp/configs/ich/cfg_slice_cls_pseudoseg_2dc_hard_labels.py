import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/ich/"
cfg.project = "gradientecho/SKP"

cfg.task = "cls_seg"

cfg.model = "segmentation.seg_cls"
cfg.decoder_type = "DeepLabV3PlusDecoder"
cfg.backbone = "tf_efficientnetv2_m"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.seg_dropout = 0.1
cfg.cls_dropout = 0.1
cfg.pool = "avg"
cfg.decoder_out_channels = 256
cfg.atrous_rates = (4, 8, 12, 16)
cfg.aspp_separable = False
cfg.aspp_dropout = 0.1
cfg.activation_fn = "sigmoid"
cfg.deep_supervision = False
# cfg.deep_supervision_num_levels = 2
cfg.num_classes = 6
cfg.cls_num_classes = 6
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.enable_gradient_checkpointing = True

cfg.fold = 0
cfg.dataset = "ich.seg_cls_pseudo_2dc"
data_dir = "/mnt/stor/datasets/kaggle/rsna-intracranial-hemorrhage-detection/"
cfg.data_dir = os.path.join(data_dir, "stage_2_train_png/")
cfg.seg_data_dir = os.path.join(data_dir, "segmentation_masks_soft_pseudolabels/")
cfg.annotations_file = os.path.join(data_dir, "train_slices_with_2dc_and_soft_pseudolabels_kfold.csv")
cfg.inputs = "filepath_2dc"
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
cfg.num_workers = 6
cfg.skip_failed_data = True
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.channel_reverse_aug = 0.5
cfg.binarize_pseudolabels = [0.4, 0.1, 0.35, 0.4, 0.35, 0.3]

cfg.loss = "combined.CombinedLoss"
cfg.loss_params = {
    "classification.BCEWithLogitsLoss": {
        "params": {},
        "output_key": "cls",
        "weight": 1.0,
    },
    "segmentation.FocalLoss": {
        "params": {
            "gamma": 2.0,
        },
        "output_key": "seg",
        "weight": 100.0,
    },
}

cfg.batch_size = 64
cfg.num_epochs = 5
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["classification.AUROC"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512

resize_transforms = [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]

additional_targets = {
    f"image{idx}": "image" for idx in range(1, cfg.num_input_channels)
}
cfg.train_transforms = A.Compose(
    resize_transforms
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
