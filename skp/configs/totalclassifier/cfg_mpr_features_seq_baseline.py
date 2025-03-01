import os

from skp.configs import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "/home/ian/projects/SKP/experiments/totalclassifier/"
cfg.project = "gradientecho/SKP"

cfg.task = "classification"

cfg.model = "embed_seq.seq"
cfg.head_type = "BiRNNHead"
cfg.add_skip_connection = True
cfg.apply_norm_first_to_features = False
cfg.feature_dim = 192
cfg.rnn_dropout = 0.0
cfg.rnn_type = "GRU"
cfg.rnn_num_layers = 1
cfg.dropout = 0.1
cfg.seq_num_classes = 117
cfg.seq_len = 512
cfg.seq2seq = True
cfg.seq2cls = False

cfg.fold = 0
cfg.dataset = "totalclassifier.features"
data_dir = "/home/ian/datasets/totalsegmentator/"
cfg.data_dir = os.path.join(
    data_dir, "extracted_embeddings_for_organ_classification_with_augs_v2b0/fold0/"
)
cfg.annotations_file = os.path.join(
    data_dir, "train_organ_classification_kfold_features.csv"
)
cfg.inputs = "features_npy"
cfg.targets = "labels_npy"
cfg.num_workers = os.cpu_count() // 2 - 1
cfg.skip_failed_data = False
cfg.pin_memory = True
cfg.persistent_workers = True
cfg.reverse_sequence_aug = 0.5
# cfg.add_embedding_noise_aug = 0.5
# cfg.noise_alpha = 10
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 500

cfg.loss = "classification.FocalLoss"
cfg.loss_params = {"seq_mode": True, "gamma": 2.0, "alpha": 0.75}

cfg.batch_size = 256
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
