import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/ich/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "embed_seq.cnn2d_rnn"
cfg.backbone = "tf_efficientnetv2_b0"
cfg.load_pretrained_backbone = os.path.join(
    cfg.save_dir,
    "ich.cfg_simple_slice_classifier/853d0cbd/fold0/checkpoints/last.ckpt",
)
cfg.num_input_channels = 1
cfg.freeze_backbone = True
cfg.backbone_img_size = False
cfg.pool = "avg"
cfg.seq_num_classes = 6
cfg.add_auxiliary_classifier = True
cfg.rnn_class = "GRU"
cfg.rnn_num_layers = 2
cfg.rnn_dropout = 0.1
cfg.linear_seq_dropout = 0.1
cfg.linear_aux_dropout = 0.1
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.enable_gradient_checkpointing = True

cfg.fold = 0
cfg.dataset = "ich.series_random_slab"
cfg.data_dir = "/mnt/stor/datasets/kaggle/rsna-intracranial-hemorrhage-detection/stage_2_train_png/"
cfg.annotations_file = "/mnt/stor/datasets/kaggle/rsna-intracranial-hemorrhage-detection/train_slices_with_2dc_kfold.csv"
cfg.inputs = "filepath"
cfg.targets = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]
# load each grayscale image then stack
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 6
cfg.skip_failed_data = True
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.reverse_sequence_aug = 0.5

cfg.loss = "custom.ICHLogLossSeqWithAux"
cfg.loss_params = {"aux_weight": 0.2}

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.1, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size // 2
cfg.metrics = ["classification.AUROCSeq"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.seq_len = 40
cfg.val_seq_len = 60
cfg.image_height = 512
cfg.image_width = 512

resize_transforms = [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]

additional_targets = {
    f"image{idx}": "image" for idx in range(1, cfg.val_seq_len or cfg.seq_len)
}
augmentation_space = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Affine(scale=(0.8, 1.2), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(
        translate_percent={"x": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(
        translate_percent={"y": (-0.20, 0.20)}, border_mode=cv2.BORDER_CONSTANT, p=1
    ),
    A.Affine(shear=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.Affine(rotate=(-30, 30), border_mode=cv2.BORDER_CONSTANT, p=1),
    A.AutoContrast(p=1),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.3, 0.4), contrast_limit=(0, 0), p=1
    ),
    A.RandomGamma(gamma_limit=(50, 150), p=1),
    A.GaussNoise(std_range=(0.1, 0.25), p=1),
    A.Downscale(scale_range=(0.25, 0.4), p=1),
    A.ImageCompression(quality_range=(20, 100), p=1),
    A.Posterize(num_bits=(3, 5), p=1),
]

augment_p = 1 - (len(augmentation_space) + 1) ** -1
cfg.train_transforms = A.Compose(
    resize_transforms
    + [
        A.RandomCrop(
            height=int(0.8125 * cfg.image_height),
            width=int(0.8125 * cfg.image_width),
            p=1,
        )
    ]
    + [
        A.OneOf(augmentation_space, p=augment_p),
    ],
    additional_targets=additional_targets,
    p=1,
)

cfg.val_transforms = A.Compose(
    resize_transforms, additional_targets=additional_targets, p=1
)
