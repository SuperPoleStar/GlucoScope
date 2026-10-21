"""Deterministic seed and device helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(use_gpu=True, gpu=0):
    if bool(use_gpu) and torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu)}")
    return torch.device("cpu")
