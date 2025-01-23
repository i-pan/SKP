import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/lines_and_tubes/"
cfg.project = "gradientecho/SKP"

cfg.task = "cls_seg"

cfg.model = "segmentation.seg_cls"
cfg.backbone = "resnet200d"
cfg.decoder_type = "DeepLabV3PlusDecoder"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.seg_dropout = 0.1
cfg.cls_dropout = 0.1
cfg.decoder_out_channels = 256
cfg.atrous_rates = (6, 12, 18, 24)
cfg.aspp_separable = False
cfg.aspp_dropout = 0.0
cfg.deep_supervision = False
cfg.pool = "avg"
cfg.num_classes = 4
cfg.cls_num_classes = 14
cfg.activation_fn = "sigmoid"
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "lines_and_tubes.seg_cls"
data_dir = "/mnt/stor/datasets/kaggle/ranzcr-clip-catheter-line-classification/"
cfg.data_dir = os.path.join(data_dir, "train")
cfg.seg_data_dir = os.path.join(data_dir, "masks")
cfg.annotations_file = os.path.join(data_dir, "train_anno_kfold_cv.csv")
cfg.inputs = "filename"
cfg.masks = "filename"
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
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 500

cfg.loss = "combined.CombinedLoss"
cfg.loss_params = {
    "classification.BCEWithLogitsLoss": {
        "params": {},
        "output_key": "cls",
        "weight": 1.0,
    },
    "segmentation.DiceLoss": {
        "params": {
            "convert_labels_to_onehot": False,
            "pred_power": 2.0,
            "ignore_background": False,
            "activation_fn": cfg.activation_fn,
            "compute_method": "per_batch",
        },
        "output_key": "seg",
        "weight": 1.0,
    },
}

cfg.batch_size = 32
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
base_lr = 3e-4
cfg.optimizer_params = {"lr": [base_lr, base_lr, base_lr]}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {
    "pct_start": 0.1,
    "div_factor": 100,
    "final_div_factor": 1_000,
}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["segmentation.MultilabelDiceScore", "custom_cls.RANZCR_AUROC"]
cfg.metric_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cfg.val_metric = "auc_mean11"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512

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

image_height = [224, 448, 896]
image_width = [224, 448, 896]
base_lr = 3e-4
cfg.progressive_resizing = {
    "batch_size": [128, 32, 8],
    "num_epochs": [1, 4, 8],
    "image_height": image_height,
    "image_width": image_width,
    "optimizer_params": [{"lr": base_lr}] * 3,
}

cfg.progressive_resizing["val_batch_size"] = cfg.progressive_resizing["batch_size"]
resize_transforms = [A.Resize(h, w, p=1) for h, w in zip(image_height, image_width)]
cfg.progressive_resizing["train_transforms"] = [
    A.Compose([_resize] + A.OneOf(augmentation_space, p=augment_p), p=1)
    for _resize in resize_transforms
]
cfg.progressive_resizing["val_transforms"] = [
    A.Compose([_resize], p=1) for _resize in resize_transforms
]
