import cv2
import albumentations as A

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/totalclassifier/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "segmentation.base_effnet_adapt3d"
cfg.decoder_type = "DeepLabV3PlusDecoder3d"
cfg.backbone = "tf_efficientnetv2_b0"
cfg.features_only = True
cfg.freeze_encoder = False
cfg.num_input_channels = 4
cfg.seg_dropout = 0.1
cfg.cls_dropout = 0.1
cfg.pool = "avg"
cfg.adapter_num_blocks = 1
cfg.adapter_kernel_size = (3, 3, 3)
cfg.adapter_strides = [2, 2, 2, 1, 1]
cfg.adapter_separable_conv = True
cfg.atrous_rates = ((1, 2, 2), (2, 4, 4), (3, 6, 6))
cfg.decoder_out_channels = 256
cfg.decoder_norm_layer = "layer_norm"
cfg.decoder_act_layer = "gelu"
cfg.aspp_separable = True
cfg.aspp_dropout = 0.1
cfg.activation_fn = "sigmoid"
cfg.deep_supervision = False
# cfg.deep_supervision_num_levels = 2
cfg.num_classes = 118
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.enable_gradient_checkpointing = True

cfg.fold = 0
cfg.dataset = "totalclassifier.seg_3d"
cfg.data_dir = (
    "/home/ian/datasets/totalsegmentator/pngs_for_slice_organ_classification/"
)
cfg.seg_data_dir = (
    "/home/ian/datasets/totalsegmentator/segs_for_slice_organ_classification/"
)
cfg.annotations_file = (
    "/home/ian/datasets/totalsegmentator/train_val_organ_classification_original_splits.csv"
)
cfg.cv2_load_flag = cv2.IMREAD_UNCHANGED
cfg.data_format = "cthw"
cfg.use_4channels = True
cfg.num_workers = 4
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.use_4channels = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "segmentation.FocalLossMemoryEfficient"
cfg.loss_params = {"gamma": 2.0, "invert_background": True, "scale": 100.0}

cfg.batch_size = 4
cfg.num_epochs = 100
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = []
cfg.val_metric = "loss"
cfg.val_track = "min"

cfg.dim0 = 24
cfg.dim1 = 224
cfg.dim2 = 224

cfg.num_slices = cfg.dim0
cfg.image_height = cfg.dim1
cfg.image_width = cfg.dim2

additional_targets = {f"image{idx}": "image" for idx in range(1, cfg.num_slices)}
additional_targets.update({f"mask{idx}": "mask" for idx in range(1, cfg.num_slices)})

resize_transforms = [
    A.LongestMaxSize(max_size=int(cfg.image_height / 0.875), p=1),
    A.PadIfNeeded(
        min_height=int(cfg.image_height / 0.875),
        min_width=int(cfg.image_width / 0.875),
        border_mode=cv2.BORDER_CONSTANT,
        p=1,
    ),
]

augmentation_space = [
    A.Affine(scale=(0.8, 1.2), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(
        translate_percent={"x": (-0.25, 0.25)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(
        translate_percent={"y": (-0.25, 0.25)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(shear=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(rotate=(-30, 30), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.25), p=1),
    A.Downscale(scale_range=(0.25, 0.4), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=(2, 5), p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1

cfg.train_transforms = A.Compose(
    resize_transforms
    + [
        A.RandomCrop(
            height=cfg.image_height,
            width=cfg.image_width,
            p=1,
        )
    ]
    + [
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
    ]
    + [
        A.OneOf(augmentation_space, p=augment_p),
    ],
    additional_targets=additional_targets,
)

cfg.val_transforms = A.Compose(
    resize_transforms + [A.CenterCrop(cfg.image_height, cfg.image_width)],
    additional_targets=additional_targets,
)
