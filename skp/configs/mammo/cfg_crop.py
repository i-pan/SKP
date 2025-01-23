import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/mammo/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "classification.net2d"
cfg.backbone = "mobilenetv3_small_100"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "gem"
cfg.pool_params = {"p": 3}
cfg.dropout = 0.1
cfg.num_classes = 4
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.model_activation_fn = "sigmoid"

cfg.fold = 0
cfg.dataset = "crop2d"
cfg.data_dir = "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/train_png/"
cfg.annotations_file = (
    "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/train_coords_kfold.csv"
)
cfg.inputs = "filename"
cfg.targets = ["x", "y", "w", "h"]
cfg.normalize_crop_coords = True
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 16
cfg.pin_memory = True
cfg.persistent_workers = True

cfg.loss = "classification.L1Loss"
cfg.loss_params = {}

cfg.batch_size = 128
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["classification.MAE", "classification.MSE"]
cfg.val_metric = "mae_mean"
cfg.val_track = "min"

cfg.image_height = 256
cfg.image_width = 256

bbox_params = A.BboxParams(format="coco", min_visibility=0.1)
resize_transforms = [A.Resize(cfg.image_height, cfg.image_width, p=1)]

augmentation_space = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
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
    resize_transforms + [A.OneOf(augmentation_space, p=augment_p)],
    p=1,
    bbox_params=bbox_params,
)

cfg.val_transforms = A.Compose(resize_transforms, p=1, bbox_params=bbox_params)
