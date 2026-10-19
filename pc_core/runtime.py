"""Runtime and checkpoint helpers shared by command-line entry points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def choose_device(requested: str) -> torch.device:
    """Resolve an explicit or automatically selected Torch device."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps was requested, but MPS is unavailable.")
    return device


def make_folds(n_rows: int, n_folds: int, seed: int) -> list[np.ndarray]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two.")
    if n_folds > n_rows:
        raise ValueError("n_folds cannot exceed the number of cells.")
    rng = np.random.default_rng(seed)
    return [
        np.asarray(fold, dtype=np.int64)
        for fold in np.array_split(rng.permutation(n_rows), n_folds)
    ]


def load_torch_checkpoint(path: str | Path) -> dict:
    """Load a local checkpoint while preserving compatibility with Torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location="cpu")
