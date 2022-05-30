import yaml

class Config():
    def __init__(self, atributes):
        for key, value in atributes.items():
            self.__dict__[key] = value

        mandatory = {"cuda_visible_device", "method", "global_method", "config", "arch", "weight_decay", "gr_clipping_max_norm",
                     "alpha", "mu", "gamma", "momentum", "tau", "lr", "mode", "dirichlet_alpha",  "participation_rate", "learning_rate_decay",
                     "set"}

        assert(mandatory.issubset(set(atributes.keys())))


class Args:
    def __init__(self, yaml_dict):
        self.__dict__ = yaml_dict


def load_yaml_conf(yaml_file):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return Args(data)
        
