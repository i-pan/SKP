import cv2

from torchvision.transforms import v2
from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/melanoma/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "classification.net2d"
cfg.backbone = "tf_efficientnetv2_m"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.num_classes = 9
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "simple2d"
cfg.data_dir = "/mnt/stor/datasets/ISIC/2019/"
cfg.annotations_file = "/mnt/stor/datasets/ISIC/train_2019_kfold_cv.csv"
cfg.inputs = "image"
cfg.targets = ["label"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 16
cfg.pin_memory = True
cfg.persistent_workers = True

cfg.loss = "classification.CrossEntropyLoss"
cfg.loss_params = {}

cfg.batch_size = 20
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["classification.AUROC", "classification.Accuracy"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 640
cfg.image_width = 640

resize_transforms = [
    v2.ToImage(),
    v2.Resize(cfg.image_height),
]

cfg.train_transforms = v2.Compose(
    resize_transforms
    + [v2.TrivialAugmentWide(), v2.RandomCrop((cfg.image_height, cfg.image_width))]
)

cfg.val_transforms = v2.Compose(
    resize_transforms + [v2.CenterCrop((cfg.image_height, cfg.image_width))]
)
