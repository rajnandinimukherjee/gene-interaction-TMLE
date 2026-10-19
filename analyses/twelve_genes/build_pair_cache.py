"""Cache all 66 panel marginal pairs using the shared PC implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from pc_core.build_pair_cache import build_pair_cache_from_data
from pc_core.runtime import choose_device

from .config import DATA, MODEL_PATH, PAIR_CACHE_PATH
from .data import load_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=PAIR_CACHE_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pair_cache_from_data(
        load_panel(args.data, max_samples=args.max_samples), model_path=args.model,
        output=args.output, device=choose_device(args.device),
    )
