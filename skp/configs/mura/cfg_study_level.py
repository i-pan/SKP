import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/mura/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "MIL.net2d_transformer"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.reduce_feature_dim = 512
cfg.num_classes = 8  # abnormal/normal + 7 exam types
cfg.transformer_nhead = 16
cfg.transformer_dim_feedforward = cfg.reduce_feature_dim
cfg.transformer_dropout = 0.1
cfg.transformer_activation = "gelu"
cfg.transformer_num_layers = 1
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "mura.study"
cfg.data_dir = "/mnt/stor/datasets/mura/"
cfg.annotations_file = "/mnt/stor/datasets/mura/MURA-v1.1/train_kfold_cv.csv"
cfg.targets = ["label", "exam_type_cls"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_images_per_study = 4
cfg.num_workers = 16
cfg.pin_memory = True
cfg.persistent_workers = True

cfg.loss = "custom.MURA_BCE_CELoss"
cfg.loss_params = {}

cfg.batch_size = 8
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["custom_cls.MURA_AUROC"]
cfg.metric_study_level = True
cfg.val_metric = "auc_study"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512

resize_transforms = [
    A.LongestMaxSize(max_size=cfg.image_height, p=1),
    A.PadIfNeeded(
        min_height=cfg.image_height,
        min_width=cfg.image_width,
        border_mode=cv2.BORDER_CONSTANT,
        p=1,
    ),
]

augmentation_space = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(scale=(0.75, 1.25), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(shear=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(
        translate_percent={"x": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(
        translate_percent={"y": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.InvertImg(p=1),
    A.Solarize(threshold_range=(0.5, 0.5), p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.5), p=1),
    A.Downscale(scale_range=(0.1, 0.25), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=(4, 5), p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms + [A.OneOf(augmentation_space, p=augment_p)], p=1
)

cfg.val_transforms = A.Compose(resize_transforms)
