import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/mammo/"
cfg.project = "gradientecho/SKP"

cfg.task = "cls_seg"

cfg.model = "segmentation.seg_cls"
cfg.decoder_type = "DeepLabV3PlusDecoder"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.seg_dropout = 0.1
cfg.cls_dropout = 0.1
cfg.pool = "avg"
cfg.decoder_out_channels = 256
cfg.atrous_rates = (6, 12, 18, 24)
cfg.aspp_separable = False
cfg.aspp_dropout = 0.1
cfg.activation_fn = "sigmoid"
cfg.deep_supervision = False
# cfg.deep_supervision_num_levels = 2
cfg.num_classes = 4
cfg.cls_num_classes = 4
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.enable_gradient_checkpointing = True

cfg.fold = 0
cfg.dataset = "mammo.seg_cls"
cfg.data_dir = "/mnt/stor/datasets/CBIS-DDSM/cropped_png"
cfg.seg_data_dir = "/mnt/stor/datasets/CBIS-DDSM/cropped_masks"
cfg.annotations_file = "/mnt/stor/datasets/CBIS-DDSM/train_kfold_cv.csv"
cfg.inputs = "filepath"
cfg.masks = "filepath"
cfg.targets = [
    "benign_calcification",
    "benign_mass",
    "malignant_calcification",
    "malignant_mass",
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 16
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True

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


cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.1, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["segmentation.MultilabelDiceScore", "classification.AUROC"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 2048
cfg.image_width = 1024
cfg.pad_to_aspect_ratio = cfg.image_height / cfg.image_width

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
