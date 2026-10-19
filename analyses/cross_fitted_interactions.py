"""Unified cross-fitted interaction analysis for 12 or 999 genes.

The 12-gene study evaluates every requested-order marginal joint-cell
interaction and makes its comparison plot. The 999-gene study creates/reuses
a pair-marginal cache, applies marginal-TMLE to all pairs, and screens
pairwise q-values/effect sizes before querying higher-order candidates.
Shared options retain the same spelling in both studies; options marked
999-only configure the scalable screened workflow.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

from analyses.nine_nine_nine_genes.config import MODEL_PATH as ALL_GENE_MODEL_PATH
from analyses.nine_nine_nine_genes.config import LOG_DIR as ALL_GENE_LOG_DIR
from analyses.naming import analysis_tag, marginal_estimand_path
from analyses.twelve_genes.config import ANALYSIS_TAG as TWELVE_ANALYSIS_TAG
from analyses.twelve_genes.config import DATA as DEFAULT_DATA
from analyses.twelve_genes.config import LOG_DIR as TWELVE_LOG_DIR
from analyses.twelve_genes.config import MODEL_PATH as TWELVE_MODEL_PATH
from analyses.twelve_genes.config import OUTPUT_DIR as TWELVE_OUTPUT_DIR
from analyses.twelve_genes.cross_fitted_interactions import main as run_twelve_gene_analysis
from pc_core.cross_fitted_interactions import main as run_all_gene_analysis
from pc_core.plaintext_reporting import (
    format_duration,
    print_training_provenance,
    tee_to_plaintext_log,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=int, choices=(12, 999), required=True,
        help="Gene set to analyse: curated 12-gene panel or all 999 named genes.",
    )
    shared = parser.add_argument_group("shared options")
    shared.add_argument("--data", type=Path, default=DEFAULT_DATA)
    shared.add_argument("--model", type=Path, default=None)
    shared.add_argument("--interaction-order", type=int, default=2)
    shared.add_argument(
        "--genes", type=str, default=None,
        help="Comma-separated names for one interaction; the 12-gene default is all combinations.",
    )
    shared.add_argument("--output", type=Path, default=None)
    shared.add_argument(
        "--log", type=Path, default=None,
        help="Optional plaintext-log base path; order and submission timestamp are always appended.",
    )
    shared.add_argument("--batch-size", type=int, default=512)
    shared.add_argument("--minimum-cell-count", type=int, default=5)
    shared.add_argument("--minimum-probability", type=float, default=1e-12)
    shared.add_argument("--bisection-iterations", type=int, default=64)
    shared.add_argument("--max-samples", type=int, default=None)
    shared.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")

    plotting = parser.add_argument_group("interaction plotting")
    plotting.add_argument("--plots-dir", type=Path, default=None)
    plotting.add_argument("--forest-top-n", type=int, default=25)
    plotting.add_argument("--network-max-items", type=int, default=25)
    plotting.add_argument("--network-max-degree", type=int, default=4)
    plotting.add_argument("--no-plots", action="store_true", help="Do not write overview, forest, and network figures.")

    panel = parser.add_argument_group("12-gene legacy comparison plot")
    panel.add_argument("--plot", type=Path, default=None)

    scalable = parser.add_argument_group("999-gene cache and higher-order screening")
    scalable.add_argument("--max-genes", type=int, default=None, help="Development/testing only.")
    scalable.add_argument("--cache", type=Path, default=None)
    scalable.add_argument("--rebuild-cache", action="store_true")
    scalable.add_argument("--pairwise-output", type=Path, default=None)
    scalable.add_argument("--candidates", type=Path, default=None)
    scalable.add_argument("--candidates-csv", type=Path, default=None)
    scalable.add_argument("--screened-pairs-csv", type=Path, default=None)
    scalable.add_argument("--screen-max-q", type=float, default=0.05)
    scalable.add_argument("--screen-min-abs-log-interaction", type=float, default=0.25)
    scalable.add_argument("--max-degree", type=int, default=20)
    scalable.add_argument("--edge-policy", choices=("mutual", "union"), default="mutual")
    scalable.add_argument("--max-candidates", type=int, default=100_000)
    return parser.parse_args(argv)


def _default_model(analysis: int) -> Path:
    if analysis == 12:
        return TWELVE_MODEL_PATH
    return ALL_GENE_MODEL_PATH


def _result_path(args: argparse.Namespace) -> Path:
    """Return the final result table path used to derive the default log."""

    requested_output = marginal_estimand_path(args.output, args.analysis)
    if requested_output is not None:
        return requested_output
    if args.analysis == 12:
        return TWELVE_OUTPUT_DIR / f"{args.interaction_order}_pt_marginal_estimand_interactions_{TWELVE_ANALYSIS_TAG}.pt"
    model = Path(args.model or _default_model(999))
    if args.interaction_order == 2:
        return model.parent / f"{model.stem}_marginal_estimand_pairwise_tmle.pt"
    return model.parent / f"{model.stem}_marginal_estimand_order{args.interaction_order}_tmle.pt"


def _timestamped_log_path(
    path: Path,
    args: argparse.Namespace,
    submitted_at: datetime,
) -> Path:
    """Add order and submit time to a log base path without replacing a log."""

    suffix = path.suffix or ".txt"
    timestamp = submitted_at.strftime("%Y%m%dT%H%M%S.%f%z")
    return path.with_name(
        f"{path.stem}_order_{args.interaction_order}_submitted_{timestamp}{suffix}"
    )


def _next_unused_path(path: Path) -> Path:
    """Return an unused path, retaining all prior logs if a name collides."""

    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _default_log_path(args: argparse.Namespace, submitted_at: datetime) -> Path:
    log_dir = TWELVE_LOG_DIR if args.analysis == 12 else ALL_GENE_LOG_DIR
    base_path = log_dir / f"cross_fitted_marginal_estimand_interactions_{analysis_tag(args.analysis)}.txt"
    return _timestamped_log_path(base_path, args, submitted_at)


def _print_run_header(args: argparse.Namespace, submitted_at: datetime) -> None:
    """Print the stable provenance header retained in every plaintext log."""

    model = Path(args.model or _default_model(args.analysis))
    print("Cross-fitted marginal-estimand interaction analysis")
    print(f"Started: {submitted_at.isoformat(timespec='microseconds')}")
    print(f"Analysis: {args.analysis}-gene")
    print("Estimand: marginal joint-cell log interaction")
    print("Run settings")
    for name, value in vars(args).items():
        print(f"  {name.replace('_', '-')}: {value}")
    print_training_provenance(model)
    print("=" * 78)


def _run_twelve(args: argparse.Namespace) -> None:
    if args.max_genes is not None:
        raise ValueError("--max-genes is only valid with --analysis 999.")
    run_twelve_gene_analysis(
        argparse.Namespace(
            data=args.data,
            model=args.model or _default_model(12),
            interaction_order=args.interaction_order,
            genes=args.genes,
            output=marginal_estimand_path(args.output, 12),
            plot=marginal_estimand_path(args.plot, 12),
            plots_dir=args.plots_dir,
            forest_top_n=args.forest_top_n,
            network_max_items=args.network_max_items,
            network_max_degree=args.network_max_degree,
            no_plots=args.no_plots,
            batch_size=args.batch_size,
            minimum_cell_count=args.minimum_cell_count,
            minimum_probability=args.minimum_probability,
            bisection_iterations=args.bisection_iterations,
            max_samples=args.max_samples,
            device=args.device,
        )
    )


def _run_all_genes(args: argparse.Namespace) -> None:
    run_all_gene_analysis(
        argparse.Namespace(
            data=args.data,
            model=args.model or _default_model(999),
            max_samples=args.max_samples,
            max_genes=args.max_genes,
            interaction_order=args.interaction_order,
            genes=args.genes,
            cache=marginal_estimand_path(args.cache, 999),
            rebuild_cache=args.rebuild_cache,
            output=marginal_estimand_path(args.output, 999),
            pairwise_output=marginal_estimand_path(args.pairwise_output, 999),
            candidates=args.candidates,
            candidates_csv=marginal_estimand_path(args.candidates_csv, 999),
            screened_pairs_csv=marginal_estimand_path(args.screened_pairs_csv, 999),
            plots_dir=args.plots_dir,
            forest_top_n=args.forest_top_n,
            network_max_items=args.network_max_items,
            network_max_degree=args.network_max_degree,
            no_plots=args.no_plots,
            minimum_cell_count=args.minimum_cell_count,
            minimum_probability=args.minimum_probability,
            bisection_iterations=args.bisection_iterations,
            screen_max_q=args.screen_max_q,
            screen_min_abs_log_interaction=args.screen_min_abs_log_interaction,
            max_degree=args.max_degree,
            edge_policy=args.edge_policy,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
            device=args.device,
        )
    )


def main(args: argparse.Namespace | None = None) -> None:
    """Dispatch to the panel or cached all-gene marginal-TMLE workflow."""

    args = parse_args() if args is None else args
    submitted_at = datetime.now().astimezone()
    requested_log = marginal_estimand_path(getattr(args, "log", None), args.analysis)
    log_base = requested_log or _default_log_path(args, submitted_at)
    log_path = _next_unused_path(
        _timestamped_log_path(log_base, args, submitted_at) if requested_log else log_base
    )
    started = time.perf_counter()
    outcome = "completed"
    with tee_to_plaintext_log(log_path):
        _print_run_header(args, submitted_at)
        try:
            if args.analysis == 12:
                _run_twelve(args)
            else:
                _run_all_genes(args)
        except BaseException as error:
            outcome = f"failed ({type(error).__name__}: {error})"
            raise
        finally:
            print("\n" + "=" * 78)
            print(f"Analysis {outcome}.")
            print(f"Interaction-analysis runtime: {format_duration(time.perf_counter() - started)}")
    print(f"Plaintext run log written to {log_path}")


if __name__ == "__main__":
    main()
