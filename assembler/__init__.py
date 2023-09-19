import importlib.util
import random
import numpy as np
import torch

from types import ModuleType


def set_seed(config: ModuleType):
    seed = config.reproducibility['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Global seed set to {seed}')


def load_config(config_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location('module_name', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    set_seed(config)
    return config
