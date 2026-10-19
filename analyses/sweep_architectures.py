"""Run a 20-epoch PC architecture sweep for the 12- or 999-gene study.

The likelihood winner is written to the selected study's tagged configuration
file.  ``train_cross_fitted_pcs.py`` reads that file automatically on later
runs, so no model setting needs to be copied by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analyses.naming import tagged_path
from analyses.nine_nine_nine_genes import config as all_gene_config
from analyses.twelve_genes import config as panel_config
from analyses.twelve_genes.data import load_panel
from pc_core.architectures import ARCHITECTURES
from pc_core.data import load_gene_data
from pc_core.sweep_pc_hyperparameters import run_sweep


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=int, choices=(12, 999), required=True)
    parser.add_argument("--data", type=Path, default=panel_config.DATA)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument(
        "--components", nargs="+", type=int, default=(4, 8, 16, 32),
        help="Root-component candidates for every non-factorized architecture.",
    )
    parser.add_argument("--learning-rates", nargs="+", type=float, default=(0.005, 0.01, 0.02))
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Epochs per candidate; the default 20 is the selection protocol.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Development/testing only.")
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Development/testing only; valid with --analysis 999.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    parser.add_argument("--output-config", type=Path, default=None)
    return parser.parse_args(argv)


def _study_config(analysis: int):
    return panel_config if analysis == 12 else all_gene_config


def _default_paths(analysis: int) -> tuple[Path, Path, Path]:
    config = _study_config(analysis)
    tag = config.ANALYSIS_TAG
    return (
        config.OUTPUT_DIR / f"pc_architecture_sweep_{tag}.csv",
        config.PLOTS_DIR / f"pc_architecture_sweep_{tag}.png",
        config.ARCHITECTURE_SELECTION_PATH,
    )


def _load_study_data(args: argparse.Namespace):
    if args.analysis == 12:
        if args.max_genes is not None:
            raise ValueError("--max-genes is only valid with --analysis 999.")
        return load_panel(args.data, max_samples=args.max_samples)
    return load_gene_data(args.data, max_samples=args.max_samples, max_genes=args.max_genes)


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    data = _load_study_data(args)
    default_csv, default_plot, default_config = _default_paths(args.analysis)
    args.output_csv = tagged_path(args.output_csv, args.analysis) or default_csv
    args.output_plot = tagged_path(args.output_plot, args.analysis) or default_plot
    args.output_config = tagged_path(args.output_config, args.analysis) or default_config
    args.plot_style = "parallel"
    args.plot_title = f"{args.analysis}-gene PC architecture and hyperparameter sweep"
    print(
        f"analysis: {args.analysis}-gene; {len(args.architectures)} architectures; "
        f"{args.epochs} epochs per candidate"
    )
    winner = run_sweep(args, data=data)
    print(
        "Default for future unified training: "
        f"architecture={winner['architecture']}; components={int(winner['components'])}; "
        f"learning_rate={float(winner['learning_rate']):g}"
    )


if __name__ == "__main__":
    main()
