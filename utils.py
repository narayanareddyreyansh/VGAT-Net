import os
import random
import numpy as np
import torch


def set_seed(seed):
    """Reproducibility (Sec. 4.1.2, seeds {0,1,2})."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val, k=1):
        self.sum += val * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


class EarlyStopping:
    """Stop when validation loss has not improved for `patience` epochs (Sec. 4.1.2)."""

    def __init__(self, patience=15, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, value):
        improved = (self.best is None or
                    (value < self.best if self.mode == "min" else value > self.best))
        if improved:
            self.best = value
            self.count = 0
            return True
        self.count += 1
        if self.count >= self.patience:
            self.should_stop = True
        return False
