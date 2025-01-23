import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/lines_and_tubes/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "segmentation.base"
cfg.backbone = "convnextv2_base"
cfg.decoder_type = "DeepLabV3PlusDecoder"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.seg_dropout = 0.1
cfg.decoder_out_channels = 256
cfg.atrous_rates = (6, 12, 18, 24)
cfg.aspp_separable = False
cfg.aspp_dropout = 0.1
cfg.activation_fn = "sigmoid"
cfg.deep_supervision = False
cfg.num_classes = 14
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.backbone_img_size = False

cfg.fold = 0
cfg.dataset = "lines_and_tubes.seg_line_tip"
data_dir = "/mnt/stor/datasets/kaggle/ranzcr-clip-catheter-line-classification/"
cfg.data_dir = os.path.join(data_dir, "train")
cfg.annotations_file = os.path.join(data_dir, "train_anno_tips_kfold_cv.csv")
cfg.inputs = "filename"
cfg.sampling_weight_col = "sampling_weight_max_ent"
cfg.add_cvc_ngt_ett_present_mask = True
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 10
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.sampler = "WeightedSampler"

cfg.loss = "segmentation.DiceLoss"
cfg.loss_params = {
    "convert_labels_to_onehot": False,
    "pred_power": 2.0,
    "ignore_background": False,
    "activation_fn": cfg.activation_fn,
    "compute_method": "per_batch",
}

cfg.batch_size = 4
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["segmentation.MultilabelDiceScore"]
cfg.metric_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cfg.val_metric = "dice_mean"
cfg.val_track = "max"

cfg.image_height = 1024
cfg.image_width = 1024

resize_transforms = [A.Resize(cfg.image_height, cfg.image_width, p=1)]

cfg.train_transforms = A.Compose(
    resize_transforms
    + [
        A.HorizontalFlip(p=0.5),
    ],
)

cfg.val_transforms = A.Compose(resize_transforms)
