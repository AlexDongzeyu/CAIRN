"""Determinism control. Import and call before anything stochastic."""
from __future__ import annotations

import os
import random

import numpy as np

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # the 10 canonical seeds (protocol §0.3)


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
