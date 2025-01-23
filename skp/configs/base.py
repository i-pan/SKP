from types import SimpleNamespace


class Config(SimpleNamespace):

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
