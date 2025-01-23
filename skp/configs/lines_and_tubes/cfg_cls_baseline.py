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
cfg.backbone = "tf_efficientnetv2_l"
cfg.pretrained = True
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
cfg.scheduler_params = {"pct_start": 0.1, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["custom_cls.RANZCR_AUROC"]
cfg.val_metric = "auc_mean11"
cfg.val_track = "max"

cfg.image_height = 1024
cfg.image_width = 1024

resize_transforms = [A.Resize(cfg.image_height, cfg.image_width, p=1)]

cfg.train_transforms = A.Compose(
    resize_transforms
    + [
        A.HorizontalFlip(p=0.5),
    ],
)

cfg.val_transforms = A.Compose(resize_transforms)
