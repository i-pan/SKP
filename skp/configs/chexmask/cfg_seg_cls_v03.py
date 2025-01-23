import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/chexmask/"
cfg.project = "gradientecho/SKP"

cfg.task = "cls_seg"

cfg.model = "segmentation.seg_cls"
cfg.decoder_type = "UnetDecoder"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.seg_dropout = 0.1
cfg.cls_dropout = 0.1
cfg.pool = "avg"
cfg.decoder_out_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.activation_fn = "softmax"
# cfg.deep_supervision = True
# cfg.deep_supervision_num_levels = 2
cfg.num_classes = 4
cfg.cls_num_classes = 1 + 3 + 1  # age, view, female
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "chexmask.seg_rle_cls"
cfg.data_dir = "/mnt/stor/datasets/"
cfg.annotations_file = "/mnt/stor/datasets/chexmask/train_nih_chexpert_combined_rle_masks_with_age_view_sex_kfold.csv"
cfg.inputs = "filename"
cfg.masks = ["right_lung", "left_lung", "heart"]
cfg.targets = ["age", "view", "female"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 16
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "combined.CombinedLoss"
cfg.loss_params = {
    "custom.AgeViewFemaleLoss": {
        "params": {"weights": [0.02, 1.0, 1.0]},
        "output_key": "cls",
        "weight": 1.0,
    },
    "segmentation.DiceLoss": {
        "params": {
            "convert_labels_to_onehot": True,
            "pred_power": 1.0,
            "ignore_background": False,
            "activation_fn": cfg.activation_fn,
            "compute_method": "per_sample",
        },
        "output_key": "seg",
        "weight": 1.0,
    },
}


cfg.batch_size = 64
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.1, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["segmentation.MulticlassDiceScore", "custom_cls.MAE_Accuracy_AUROC"]
cfg.val_metric = "dice_mean"
cfg.val_track = "max"

cfg.image_height = 320
cfg.image_width = 320

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
    A.Affine(
        translate_percent={"y": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.InvertImg(p=1),
    A.Solarize(threshold_range=(0.5, 0.5), p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.5), p=1),
    A.Downscale(scale_range=(0.1, 0.25), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=(4, 5), p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms + [A.OneOf(augmentation_space, p=augment_p)], p=1
)

cfg.val_transforms = A.Compose(resize_transforms)
