"""Train the selected cross-fitted PC for the curated 12-gene panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from pc_core.runtime import choose_device
from pc_core.train_cross_fitted_pcs import train_cross_fitted_models

from .config import (
    ARCHITECTURE,
    COMPONENTS,
    DATA,
    LEARNING_RATE,
    MODEL_PATH,
    WEIGHT_DECAY,
)
from .data import load_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--architecture", choices=(ARCHITECTURE,), default=ARCHITECTURE)
    parser.add_argument("--components", type=int, default=COMPONENTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--info", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    train_cross_fitted_models(
        load_panel(args.data, max_samples=args.max_samples),
        output=args.output,
        architecture=args.architecture,
        components=args.components,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_folds=args.n_folds,
        seed=args.seed,
        device=choose_device(args.device),
        info=args.info,
    )


if __name__ == "__main__":
    main()
