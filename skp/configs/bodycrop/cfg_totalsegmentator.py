import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/bodycrop/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "classification.net2d"
cfg.backbone = "mobilenetv3_small_100"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.num_classes = 4
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.model_activation_fn = "sigmoid"

cfg.fold = 0
cfg.dataset = "crop2d"
cfg.data_dir = "/mnt/stor/datasets/totalsegmentator/pngs_for_bbox_background_crop/"
cfg.annotations_file = "/mnt/stor/datasets/totalsegmentator/train_bodycrop_kfold.csv"
cfg.inputs = "filename"
cfg.targets = ["x1", "y1", "w", "h"]
cfg.normalize_crop_coords = True
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 16
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 500

cfg.loss = "classification.L1Loss"
cfg.loss_params = {}

cfg.batch_size = 128
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.1, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["classification.MAE", "classification.MSE"]
cfg.val_metric = "mae_mean"
cfg.val_track = "min"

cfg.image_height = 256
cfg.image_width = 256

bbox_params = A.BboxParams(format="coco")
resize_transforms = [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]

augmentation_space = [
    A.Affine(scale=(0.8, 1.2), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(
        translate_percent={"x": (-0.25, 0.25)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(
        translate_percent={"y": (-0.25, 0.25)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(shear=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(rotate=(-30, 30), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.25), p=1),
    A.Downscale(scale_range=(0.25, 0.4), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=(2, 5), p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1

cfg.train_transforms = A.Compose(
    resize_transforms
    + [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
    ]
    + [
        A.OneOf(augmentation_space, p=augment_p),
    ],
    bbox_params=bbox_params,
)

cfg.val_transforms = A.Compose(
    resize_transforms,
    bbox_params=bbox_params,
)
