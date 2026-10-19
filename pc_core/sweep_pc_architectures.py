"""Run the 999-gene architecture/hyperparameter sweep with a parallel plot."""

from __future__ import annotations

from pathlib import Path

from .sweep_pc_hyperparameters import parse_args, run_sweep


def main() -> None:
    args = parse_args()
    # Give the architecture-oriented entry point its own artifacts while still
    # allowing explicit --output-* arguments to override these defaults.
    if args.output_csv == Path("output/pc_hyperparameter_sweep_999_genes.csv"):
        args.output_csv = Path("output/pc_architecture_sweep_999_genes.csv")
    if args.output_plot == Path("plots/pc_hyperparameter_sweep_999_genes.png"):
        args.output_plot = Path("plots/pc_architecture_sweep_999_genes.png")
    if args.output_config == Path("output/pc_selected_config_999_genes.json"):
        args.output_config = Path("output/pc_architecture_selected_config_999_genes.json")
    args.plot_style = "parallel"
    run_sweep(args)


if __name__ == "__main__":
    main()
