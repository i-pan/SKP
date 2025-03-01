import albumentations as A
import cv2
import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/ich/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "embed_seq.seq"
cfg.head_type = "DoubleSkipBiRNNHead"
# cfg.head_type = "BiRNNHead"
cfg.add_skip_connection = True
cfg.feature_dim = 512
cfg.rnn_dropout = 0.1
cfg.rnn_type = "LSTM"
cfg.attention_type = "basic"
cfg.attention_dropout = 0.1
cfg.dropout = 0.1
cfg.seq_num_classes = 6
cfg.cls_num_classes = 6
cfg.seq_len = 60
cfg.seq2seq = True
cfg.seq2cls = True

cfg.fold = 0
cfg.dataset = "ich.features"
data_dir = "/mnt/stor/datasets/kaggle/rsna-intracranial-hemorrhage-detection/"
cfg.data_dir = os.path.join(data_dir, "features_2dc_cls_pseudoseg/fold0/")
cfg.annotations_file = os.path.join(data_dir, "train_kfold_features_cv.csv")
cfg.inputs = "features_npy"
cfg.targets = "labels_npy"
cfg.truncate_or_resample_sequence = "truncate"
cfg.num_workers = os.cpu_count() // 2 - 1
cfg.skip_failed_data = True
cfg.pin_memory = True
cfg.persistent_workers = True
# cfg.add_embedding_noise_aug = 0.5
# cfg.noise_alpha = 10
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 100

cfg.loss = "custom.ICHSeqClsMaskedLoss"
cfg.loss_params = {"cls_weight": 0.1}

cfg.batch_size = 64
cfg.num_epochs = 5
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 0.05}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size
cfg.metrics = ["custom_cls.ICHSeqClsAUROC"]
cfg.val_metric = "auc_mean_cls"
cfg.val_track = "max"
