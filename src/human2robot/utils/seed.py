"""Reproducible seeding for python, numpy, and torch.

One top-level seed drives everything. Vector env workers receive
seed + worker_index so parallel environments never share an RNG.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (CPU and CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(seed: int, worker_index: int) -> int:
    """Derive a distinct seed for a vector env worker."""
    return seed + worker_index


def set_torch_threads(limit: int = 8) -> None:
    """Cap intra-op threads so vector env workers keep CPU headroom."""
    torch.set_num_threads(limit)
    os.environ.setdefault("OMP_NUM_THREADS", str(limit))
