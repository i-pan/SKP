import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/lines_and_tubes/"
cfg.project = "gradientecho/SKP"

cfg.task = "cls_seg"

cfg.model = "segmentation.unet_cls"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.load_pretrained_segmenter = os.path.join(
    cfg.save_dir,
    "lines_and_tubes.cfg_seg_pos_only",
    "4c682c16/fold0/checkpoints/last.ckpt",
)
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
cfg.num_classes = 4  # seg
cfg.cls_num_classes = 14
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "lines_and_tubes.seg_cls"
data_dir = "/mnt/stor/datasets/kaggle/siim-acr-pneumothorax-segmentation/"
data_dir = "/mnt/stor/datasets/kaggle/ranzcr-clip-catheter-line-classification/"
cfg.data_dir = os.path.join(data_dir, "train")
cfg.seg_data_dir = os.path.join(data_dir, "masks")
cfg.annotations_file = os.path.join(data_dir, "train_kfold_cv.csv")
cfg.inputs = "filename"
cfg.masks = "filename"
cvc_targets = ["cvc_present", "cvc_normal", "cvc_borderline", "cvc_abnormal"]
ngt_targets = [
    "ngt_present",
    "ngt_normal",
    "ngt_borderline",
    "ngt_abnormal",
    "ngt_incompletelyimaged",
]
ett_targets = ["ett_present", "ett_normal", "ett_borderline", "ett_abnormal"]
cfg.targets = cvc_targets + ngt_targets + ett_targets + ["swan_ganz_present"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 12
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

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

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.1, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["segmentation.MultilabelDiceScore", "classification.AUROC"]
cfg.metric_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.image_height = 768
cfg.image_width = 768

# limit use of transforms that could result
# in cropping ends of lines/tubes
resize1 = A.Resize(cfg.image_height, cfg.image_width, p=1)
resize2 = A.Compose(
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
    A.OneOf([resize1, resize2], p=1),
]
resize_transforms_val = resize1

cfg.train_transforms = A.Compose(
    resize_transforms_train
    + [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.SomeOf(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.0,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1,
                ),
                A.GaussianBlur(p=1),
                A.GaussNoise(p=1, std_range=(0.01, 0.1)),
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
