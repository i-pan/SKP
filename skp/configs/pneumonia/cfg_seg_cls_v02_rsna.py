import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/pneumonia/"
cfg.project = "gradientecho/SKP"

cfg.task = "cls_seg"

cfg.model = "segmentation.unet_cls"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.seg_dropout = 0.1
cfg.cls_dropout = 0.1
cfg.pool = "avg"
cfg.decoder_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.activation_fn = "sigmoid"
cfg.deep_supervision = True
cfg.deep_supervision_num_levels = 2
cfg.num_classes = 1
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "simple2d_seg_cls"
cfg.data_dir = "/mnt/stor/datasets/pneumonia-seg/"
cfg.annotations_file = os.path.join(cfg.data_dir, "train_rsna_only_kfold_cv.csv")
cfg.inputs = "filename"
cfg.masks = "maskfile"
cfg.targets = ["label"]
cfg.assume_mask_empty_if_not_present = True
cfg.rescale_mask = 255.0
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 16
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 200

cfg.loss = "combined.CombinedLoss"
cfg.loss_params = {
    "losses": {
        "classification.BCEWithLogitsLoss": {
            "params": {},
            "output_key": "cls",
            "weight": 1.0,
        },
        "segmentation.DeepSupervision": {
            "params": {
                "deep_supervision_weights": [1.0, 0.5, 0.25],
                "loss_name": "DiceLoss",
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
}

cfg.batch_size = 32
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

cfg.image_height = 512
cfg.image_width = 512

resize1 = A.Compose(
    [
        A.SmallestMaxSize(cfg.image_height, p=1),
        A.RandomCrop(cfg.image_height, cfg.image_width, p=1),
    ]
)
resize2 = A.Resize(cfg.image_height, cfg.image_width, p=1)
resize3 = A.Compose(
    [
        A.LongestMaxSize(cfg.image_height, p=1),
        A.PadIfNeeded(
            min_height=cfg.image_height,
            min_width=cfg.image_width,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
    ]
)

resize_transforms_train = [
    A.OneOf([resize1, resize2, resize3], p=1),
]
resize_transforms_val = resize2

cfg.train_transforms = A.Compose(
    resize_transforms_train
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
    ],
)

cfg.val_transforms = A.Compose([resize_transforms_val])
