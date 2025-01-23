import cv2
import os

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
cfg.load_pretrained_model = os.path.join(cfg.save_dir, "melanoma.cfg_2019_baseline/e1638ae8/fold0/checkpoints/last.ckpt")
cfg.num_input_channels = 3
cfg.pool = "avg"
cfg.dropout = 0.1
cfg.num_classes = 9
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "simple2d"
cfg.data_dir = "/mnt/stor/datasets/ISIC/"
cfg.annotations_file = "/mnt/stor/datasets/ISIC/train_combined_2019_2020_kfold_cv.csv"
cfg.inputs = "image"
cfg.targets = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = os.cpu_count() // 2 - 1
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.group_index_col = "year"

cfg.loss = "classification.SoftTargetCrossEntropy"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["custom_cls.MelanomaAUROC"]
cfg.metric_include_classes = [0]  # melanoma
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512

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
