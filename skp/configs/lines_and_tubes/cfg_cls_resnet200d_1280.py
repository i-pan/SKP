import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/lines_and_tubes/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "classification.net2d"
cfg.backbone = "resnet200d"
cfg.pretrained = True
cfg.load_pretrained_model = os.path.join(cfg.save_dir, "lines_and_tubes.cfg_cls_resnet200d_prog_resize", "8fbb2ad9/prog_resize3/fold0/checkpoints/last.ckpt")
cfg.num_input_channels = 1
cfg.dropout = 0.1
cfg.pool = "avg"
cfg.num_classes = 14
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "simple2d"
data_dir = "/mnt/stor/datasets/kaggle/siim-acr-pneumothorax-segmentation/"
data_dir = "/mnt/stor/datasets/kaggle/ranzcr-clip-catheter-line-classification/"
cfg.data_dir = os.path.join(data_dir, "train")
cfg.annotations_file = os.path.join(data_dir, "train_kfold_cv.csv")
cfg.inputs = "filename"
normal = ["cvc_normal", "ngt_normal", "ett_normal"]
borderline = ["cvc_borderline", "ngt_borderline", "ett_borderline"]
abnormal = ["cvc_abnormal", "ngt_abnormal", "ett_abnormal"]
other = ["ngt_incompletelyimaged", "swan_ganz_present"]
present = ["cvc_present", "ngt_present", "ett_present"]
cfg.targets = normal + borderline + abnormal + other + present
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 12
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True

cfg.loss = "classification.BCEWithLogitsLoss"
cfg.loss_params = {}

cfg.batch_size = 4
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {
    "pct_start": 1.0 / cfg.num_epochs,
    "div_factor": 100,
    "final_div_factor": 1_000,
}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["custom_cls.RANZCR_AUROC"]
cfg.val_metric = "auc_mean11"
cfg.val_track = "max"

cfg.image_height = 1280
cfg.image_width = 1280

resize_transforms = [A.Resize(cfg.image_height, cfg.image_width, p=1)]

augmentation_space = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(scale=(0.75, 1.25), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(shear=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(
        translate_percent={"x": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.InvertImg(p=1),
    A.Solarize(threshold_range=(0.5, 0.5), p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.5), p=1),
    A.Downscale(scale_range=(0.25, 0.25), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=5, p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms + [A.OneOf(augmentation_space, p=augment_p)], p=1
)

cfg.val_transforms = A.Compose(resize_transforms, p=1)
