import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    return torch.device(cfg.device if torch.cuda.is_available() else "cpu")


def save_checkpoint(state_dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state_dict, path)
    print(f"[checkpoint] saved to {path}")
