import albumentations as A
import cv2

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/ich/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "segmentation.unet"
cfg.backbone = "maxvit_tiny_tf_512"
cfg.pretrained = True
cfg.load_pretrained_encoder = cfg.save_dir + "ich.cfg_slice_classifier_2dc/8447489b/fold0/checkpoints/last.ckpt"
cfg.freeze_encoder = True
cfg.num_input_channels = 3
cfg.seg_dropout = 0.1
cfg.decoder_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.activation_fn = "sigmoid"
# cfg.deep_supervision = True
# cfg.deep_supervision_num_levels = 2
cfg.num_classes = 6
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = True

cfg.fold = 0
cfg.dataset = "simple2d_seg"
cfg.data_dir = "/mnt/stor/datasets/BHSD/png/"
cfg.annotations_file = "/mnt/stor/datasets/BHSD/train_positive_slices_png_kfold.csv"
cfg.inputs = "image"
cfg.targets = "label"
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 16
cfg.skip_failed_data = True
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 100

cfg.loss = "segmentation.DiceLoss"
cfg.loss_params = {
    "convert_labels_to_onehot": True,
    "pred_power": 2.0, 
    "invert_background": True,
    "activation_fn": cfg.activation_fn,
    "compute_method": "per_batch"
}

cfg.batch_size = 128
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 1e-3}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["segmentation.MultilabelDiceScore"]
cfg.metric_labels_to_onehot = True
cfg.val_metric = "ml_dice_mean"
cfg.val_track = "max"

cfg.image_height = 512
cfg.image_width = 512

resize_transforms = [A.Resize(height=cfg.image_height, width=cfg.image_width, p=1)]

cfg.train_transforms = A.Compose(
    resize_transforms
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

cfg.val_transforms = A.Compose(resize_transforms)
