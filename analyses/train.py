"""Unified cross-fitted PC training for the 12-gene and 999-gene analyses.

Choose the study with ``--analysis 12`` or ``--analysis 999``.  The shared
PC implementation, selected architecture defaults, checkpoint metadata, and
``--info`` calibration live in :mod:`pc_core`; this module only selects the
appropriate gene set and output location.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pc_core.architectures import ARCHITECTURES
from pc_core.data import load_gene_data
from pc_core.runtime import choose_device
from pc_core.train_cross_fitted_pcs import (
    DEFAULT_DATA,
    train_cross_fitted_models,
)

from analyses.nine_nine_nine_genes.config import MODEL_PATH as ALL_GENE_MODEL_PATH
from analyses.defaults import training_defaults
from analyses.naming import tagged_path
from analyses.twelve_genes.config import MODEL_PATH as TWELVE_MODEL_PATH
from analyses.twelve_genes.data import load_panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=int, choices=(12, 999), required=True,
        help="Gene set to analyse: curated 12-gene panel or all 999 named genes.",
    )
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Checkpoint path (defaults to the selected analysis output directory).",
    )
    parser.add_argument(
        "--architecture", choices=ARCHITECTURES, default=None,
        help="Override the architecture selected by this study's 20-epoch sweep.",
    )
    parser.add_argument(
        "--components", type=int, default=None,
        help="Override the component count selected by this study's 20-epoch sweep.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override the learning rate selected by this study's 20-epoch sweep.",
    )
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Development/testing only.")
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Development/testing only; valid with --analysis 999.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--info", action="store_true",
        help="Time one disposable epoch and report an estimated full-run duration; write no checkpoint.",
    )
    return parser.parse_args(argv)


def _load_data(args: argparse.Namespace):
    if args.analysis == 12:
        if args.max_genes is not None:
            raise ValueError("--max-genes is only valid with --analysis 999.")
        return load_panel(args.data, max_samples=args.max_samples)
    return load_gene_data(
        args.data, max_samples=args.max_samples, max_genes=args.max_genes
    )


def _default_output(analysis: int) -> Path:
    if analysis == 12:
        return TWELVE_MODEL_PATH
    return ALL_GENE_MODEL_PATH


def main(args: argparse.Namespace | None = None) -> None:
    """Run one selected study through the single training interface."""

    args = parse_args() if args is None else args
    data = _load_data(args)
    output = tagged_path(args.output, args.analysis) or _default_output(args.analysis)
    selected = training_defaults(args.analysis)
    architecture = args.architecture or selected.architecture
    components = args.components if args.components is not None else selected.components
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else selected.learning_rate
    )
    weight_decay = args.weight_decay if args.weight_decay is not None else selected.weight_decay
    print(f"analysis: {args.analysis}-gene")
    print(
        "PC defaults: "
        f"architecture={architecture}; components={components}; learning_rate={learning_rate:g}"
    )
    train_cross_fitted_models(
        data,
        output=output,
        architecture=architecture,
        components=components,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_folds=args.n_folds,
        seed=args.seed,
        device=choose_device(args.device),
        info=args.info,
    )


if __name__ == "__main__":
    main()
