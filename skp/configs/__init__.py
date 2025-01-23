from types import SimpleNamespace


class Config(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # allows for updating EMA options thru CLI without having to
        # explicitly specify in each config file
        self.ema = {
            "on": False,
            "decay": 0.9999,
            "update_after_step": 0,
            "update_every_n_steps": 1,
            "use_warmup": False,
            "warmup_gamma": 1.0,
            "warmup_power": 2 / 3,
            "switch_ema": False,
        }
        self.enable_gradient_checkpointing = False

    def __getattribute__(self, value):
        # If attribute not specified in config,
        # return None instead of raise error
        try:
            return super().__getattribute__(value)
        except AttributeError:
            return None

    def __str__(self):
        # pretty print
        string = ["config"]
        string.append("=" * len(string[0]))
        longest_param_name = max([len(k) for k in [*self.__dict__]])
        for k, v in self.__dict__.items():
            string.append(f"{k.ljust(longest_param_name)} : {v}")
        return "\n".join(string)

    def __deepcopy__(self, memo=None):
        return SimpleNamespace()
