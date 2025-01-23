import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/mammo/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "classification.net2d"
cfg.backbone = "tf_efficientnetv2_m"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.num_classes = 1 + 1 + 1 + 1 + 3 + 4
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "simple2d"
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
cfg.group_index_col = "breast_id_index"
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 16
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampling_weight_col = "sampling_weight"
cfg.sampler = "WeightedWithDecaySampler"
cfg.sampler_decay_alpha = 0.6

cfg.loss = "custom.MammoCancerWithAuxLosses"
cfg.loss_params = {}

cfg.batch_size = 8
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

cfg.image_height = 1024
cfg.image_width = 1024

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
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms + [A.OneOf(augmentation_space, p=augment_p)], p=1
)

cfg.val_transforms = A.Compose(resize_transforms)
