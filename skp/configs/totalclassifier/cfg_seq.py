import cv2
import albumentations as A

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/totalclassifier/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "embed_seq.cnn2d_rnn"
cfg.backbone = "tf_efficientnetv2_b0"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "avg"
cfg.seq_num_classes = 117
cfg.add_auxiliary_classifier = False
cfg.rnn_class = "GRU"
cfg.rnn_num_layers = 1
cfg.rnn_dropout = 0.1
cfg.linear_seq_dropout = 0.1
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False
cfg.enable_gradient_checkpointing = True
cfg.forward_chunks = True

cfg.fold = 0
cfg.dataset = "totalclassifier.image_seq"
cfg.data_dir = (
    "/home/ian/datasets/totalsegmentator/pngs_for_slice_organ_classification/"
)
cfg.annotations_file = (
    "/home/ian/datasets/totalsegmentator/train_organ_classification_kfold.csv"
)
cfg.inputs = "filename"
cfg.cv2_load_flag = cv2.IMREAD_UNCHANGED
cfg.num_workers = 4
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.use_4channels = False

cfg.loss = "classification.BCEWithLogitsLoss"
cfg.loss_params = {"seq_mode": True}

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1
cfg.metrics = ["classification.ManyClassAUROC"]
cfg.metric_seq_mode = True
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.seq_len = 48
cfg.image_height = 384
cfg.image_width = 384

additional_targets = {f"image{idx}": "image" for idx in range(1, cfg.seq_len)}
resize_transforms = [
    A.LongestMaxSize(max_size=cfg.image_height, p=1),
    A.PadIfNeeded(
        min_height=cfg.image_height,
        min_width=cfg.image_width,
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
            height=int(0.875 * cfg.image_height),
            width=int(0.875 * cfg.image_width),
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

cfg.val_transforms = A.Compose(resize_transforms, additional_targets=additional_targets)
