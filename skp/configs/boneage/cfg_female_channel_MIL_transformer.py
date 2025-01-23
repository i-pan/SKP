import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/boneage/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "MIL.net2d_attn"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 2
cfg.pool = "gem"
cfg.pool_params = {"p": 3}
cfg.dropout = 0.1
cfg.num_classes = 1
cfg.reduce_feature_dim = 256
cfg.add_transformer = True
cfg.transformer_dropout = 0.0
cfg.transformer_num_layers = 1
cfg.attn_dropout = 0.0
cfg.attn_version = "v1"
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "boneage.female_channel_grid_patch"
cfg.data_dir = "/mnt/stor/datasets/bone-age/cropped_train_plus_valid/"
cfg.annotations_file = "/mnt/stor/datasets/bone-age/train_plus_valid_kfold.csv"
cfg.inputs = "imgfile0"
cfg.targets = ["bone_age_years"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.patch_size = 224
cfg.patch_num_rows = 5
cfg.patch_num_cols = 3
cfg.num_workers = 16
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "classification.L1Loss"
cfg.loss_params = {}

cfg.batch_size = 16
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

cfg.image_height = 560
cfg.image_width = cfg.image_height # not used 

resize_transforms = [
    A.LongestMaxSize(max_size=cfg.image_height, p=1),
]

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
    ]
)

cfg.val_transforms = A.Compose(resize_transforms)
