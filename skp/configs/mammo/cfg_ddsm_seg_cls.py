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
cfg.backbone = "tf_efficientnetv2_m"
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

cfg.fold = 0
cfg.dataset = "mammo.seg_cls"
cfg.data_dir = "/mnt/stor/datasets/CBIS-DDSM/cropped_png"
cfg.seg_data_dir = "/mnt/stor/datasets/CBIS-DDSM/cropped_masks"
cfg.annotations_file = "/mnt/stor/datasets/CBIS-DDSM/train_kfold_cv.csv"
cfg.inputs = "filepath"
cfg.masks = "filepath"
cfg.targets = ["benign_calcification", "benign_mass", "malignant_calcification", "malignant_mass"]
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


cfg.batch_size = 4
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

cfg.image_height = 1536
cfg.image_width = 1024

resize_transforms = [A.Resize(cfg.image_height, cfg.image_width, p=1)]

augmentation_space = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms + [A.OneOf(augmentation_space, p=augment_p)], p=1
)

cfg.val_transforms = A.Compose(resize_transforms)
