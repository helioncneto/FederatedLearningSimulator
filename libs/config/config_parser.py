import sys
import yaml
from config import Config

config = yaml.load(open("config.yaml").read(), Loader=yaml.FullLoader)

cg = Config(config)

print(type(cg.additional_experiment_name))