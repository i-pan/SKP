import cv2
import albumentations as A

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/totalclassifier/"
cfg.project = "gradientecho/SKP"

cfg.task = "segmentation_3d"

cfg.model = "segmentation.base_3d"
cfg.decoder_type = "Unet3dDecoder"
cfg.backbone = "x3d_l"
cfg.pretrained = True
cfg.freeze_encoder = False
cfg.num_input_channels = 4
cfg.seg_dropout = 0.1
cfg.dim0_strides = [2, 2, 2, 2, 2]
cfg.decoder_n_blocks = 5
cfg.decoder_out_channels = [256, 128, 64, 32, 16]
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_separable_conv = True
cfg.activation_fn = "sigmoid"
cfg.deep_supervision = False
# cfg.deep_supervision_num_levels = 2
cfg.num_classes = 24
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.enable_gradient_checkpointing = True

cfg.fold = 0
cfg.dataset = "totalclassifier.seg_3d"
cfg.data_dir = (
    "/home/ian/datasets/totalsegmentator/pngs_for_slice_organ_classification/"
)
cfg.seg_data_dir = (
    "/home/ian/datasets/totalsegmentator/segs_for_slice_organ_classification/"
)
cfg.annotations_file = "/home/ian/datasets/totalsegmentator/train_val_organ_classification_original_splits.csv"
cfg.cv2_load_flag = cv2.IMREAD_UNCHANGED
cfg.data_format = "cthw"
cfg.num_workers = 8
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.use_4channels = True
cfg.reverse_slices_aug = True 
cfg.downsample_slices_aug = False 
cfg.channel_shuffle_aug = False
cfg.single_channel_aug = False
cfg.channel_dropout_aug = False
cfg.classes_subset = "class_map_part_organs"
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "segmentation.DiceBCELoss"
cfg.loss_params = {
    "compute_method": "per_batch",
    "ignore_background": True,
    "convert_labels_to_onehot": True,
    "activation_fn": cfg.activation_fn,
    # "focal_weight": 10.0,
    # "gamma": 2.0,
    # "alpha": None,
}

cfg.batch_size = 12
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1
cfg.max_val_num_slices = 384
cfg.divide_val_samples_into_chunks = False
cfg.metric_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
cfg.pad_slices_for_val = False
cfg.use_sliding_window_inference = True
cfg.metrics = ["segmentation.TotalSegmentatorForegroundDice"]
cfg.val_metric = "dice_mean"
cfg.val_track = "max"

cfg.dim0 = 128
cfg.dim1 = 128
cfg.dim2 = 128

cfg.num_slices = cfg.dim0
cfg.image_height = cfg.dim1
cfg.image_width = cfg.dim2

additional_targets = {f"image{idx}": "image" for idx in range(1, cfg.num_slices)}
additional_targets.update({f"mask{idx}": "mask" for idx in range(1, cfg.num_slices)})
cfg.train_transforms = A.Compose(
    [
        A.RandomScale(scale_limit=(-0.2, 0.2), p=1),
        A.PadIfNeeded(
            min_height=cfg.image_height,
            min_width=cfg.image_width,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
        A.CropNonEmptyMaskIfExists(height=cfg.image_height, width=cfg.image_width, p=1),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
    ],
    additional_targets=additional_targets,
)

cfg.val_transforms = None
