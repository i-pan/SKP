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
cfg.feature_dim = 1280
cfg.rnn_dropout = 0.1
cfg.rnn_type = "GRU"
cfg.rnn_num_layers = 1
cfg.dropout = 0.1
cfg.seq_num_classes = 117
cfg.seq_len = 256
cfg.seq2seq = True
cfg.seq2cls = False

cfg.fold = 0
cfg.dataset = "totalclassifier.features"
data_dir = "/home/ian/datasets/totalsegmentator/"
cfg.data_dir = os.path.join(
    data_dir, "extracted_embeddings_for_organ_classification_with_augs/fold0/"
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
cfg.num_iterations_per_epoch = 100

cfg.loss = "classification.BCEWithLogitsLoss"
cfg.loss_params = {"seq_mode": True}

cfg.batch_size = 256
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 0.05}

cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "div_factor": 100, "final_div_factor": 1_000}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1
cfg.metrics = ["classification.ManyClassAUROC"]
cfg.metric_seq_mode = True
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.hyperparameter_sweep = {
    "num_trials": 50,
    "rnn_type": {
        "feasible_points": ["GRU", "LSTM"],
    },
    "optimizer_params__lr": {
        "min": 1e-5,
        "max": 1e-3,
        "scaling": "log",
    },
    "optimizer_params__weight_decay": {
        "min": 5e-5,
        "max": 0.05,
        "scaling": "log",
    },
    "optimizer_params__eps": {
        "min": 1e-8,
        "max": 1e-3,
        "scaling": "log",
    },
    "optimizer_params__beta1": {
        "min": 0.5,
        "max": 0.99,
        "scaling": "linear",
    },
    "rnn_dropout": {
        "min": 0,
        "max": 0.5,
        "scaling": "linear",
    },
    "rnn_num_layers": {"feasible_points": [1, 2, 3, 4]},
    "dropout": {
        "min": 0,
        "max": 0.5,
        "scaling": "linear",
    },
    "seq_len": {"feasible_points": [256, 384, 512]},
    "num_iterations_per_epoch": {"feasible_points": [50, 100, 250, 500]},
}
