from skp.configs import Config


cfg = Config()
# use __ to indicate parameter within a dict
# e.g., optimizer_params["lr"] -> optimizer_params__lr
cfg.hyperparameter_sweep = {
    "num_trials": 10,
    "backbone": {
        "feasible_points": [
            "tf_efficientnetv2_b0",
            "tf_efficientnetv2_b1",
            "tf_efficientnetv2_b2",
        ],
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
    "dropout": {
        "min": 0,
        "max": 0.5,
        "scaling": "linear",
    },
}
