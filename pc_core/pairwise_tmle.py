"""Run batched cross-fitted conditional TMLE for every ordered gene pair."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .build_pair_cache import load_cross_fitted_models
from .conditional_tmle import (
    DEFAULT_PROBABILITY_CLIP,
    NORMAL_95_CRITICAL_VALUE,
    run_conditional_tmle,
)
from .data import load_gene_data
from .interaction_plots import generate_interaction_plots
from .plaintext_reporting import print_tmle_result_summary
from .runtime import choose_device


def compute_pairwise_tmle(args: argparse.Namespace) -> None:
    """Estimate standardized conditional log odds ratios for all pairs.

    Pair ``(i, j)`` is ordered: gene ``i`` is the binary outcome and gene
    ``j`` is the exposure.  The remaining genes are the PC-conditioned
    covariates.  This orientation follows the reference analysis; it is not
    the symmetric four-cell association on the marginal branch.
    """

    data = getattr(args, "data_object", None)
    if data is None:
        data = load_gene_data(args.data, max_samples=args.max_samples, max_genes=args.max_genes)
    requested_device = choose_device(args.device)
    device = torch.device("cpu") if requested_device.type == "mps" else requested_device
    _, models, folds = load_cross_fitted_models(args.model, data, device)
    pair_i, pair_j = torch.triu_indices(data.shape[1], data.shape[1], offset=1)
    print(
        f"Conditional TMLE device: {device}; evaluating {len(pair_i):,} ordered outcome/exposure pairs "
        f"across {len(models)} folds"
    )
    estimates = run_conditional_tmle(
        models,
        folds,
        data.values,
        torch.stack((pair_i, pair_j), dim=1),
        device=device,
        target_batch_size=args.batch_size,
        nuisance_sample_batch_size=args.nuisance_sample_batch_size,
        probability_clip=args.probability_clip,
        fluctuation_iterations=args.fluctuation_iterations,
    )
    valid = estimates["tmle_supported"]
    screened = (
        valid
        & (estimates["q_value"] <= args.screen_max_q)
        & (torch.abs(estimates["tmle_log_interaction"]) >= args.screen_min_abs_log_interaction)
    )
    result = {
        "format_version": 3,
        "estimand": "standardized_conditional_log_odds_interaction",
        "source_model": str(Path(args.model)),
        "data_fingerprint": data.fingerprint,
        "gene_names": list(data.gene_names),
        "pair_i": pair_i,
        "pair_j": pair_j,
        **estimates,
        "screened": screened,
        "settings": {
            "probability_clip": args.probability_clip,
            "fluctuation_iterations": args.fluctuation_iterations,
            "nuisance_sample_batch_size": args.nuisance_sample_batch_size,
            "total_heldout_cells": len(data.values),
            "multiple_testing": "Benjamini-Yekutieli",
            "outcome_position": 0,
            "covariates": "all remaining genes",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    names = result["gene_names"]
    print_tmle_result_summary(
        {**result, "output_path": str(output)},
        lambda row: f"{names[int(pair_i[row])]} <- {names[int(pair_j[row])]}",
        max_rows=25,
    )
    print(f"Pairs passing BY/effect screen: {int(screened.sum()):,}")
    print(f"Saved all conditional pairwise results to {output}")

    if not getattr(args, "no_plots", False):
        default_plot_dir = output.parent.parent / "plots" if output.parent.name == "output" else output.parent / "plots"
        plot_paths = generate_interaction_plots(
            result,
            Path(getattr(args, "plots_dir", None) or default_plot_dir),
            output.stem,
            max_q=args.screen_max_q,
            min_abs_log_interaction=args.screen_min_abs_log_interaction,
            forest_top_n=getattr(args, "forest_top_n", 25),
            network_max_items=getattr(args, "network_max_items", 25),
            network_max_degree=getattr(args, "network_max_degree", 4),
        )
        print("Wrote interaction plots: " + ", ".join(str(path) for path in plot_paths))

    if args.screened_pairs_csv is not None:
        write_screened_pairs_csv(result, Path(args.screened_pairs_csv))


def write_screened_pairs_csv(result: dict, path: Path) -> None:
    """Write the directional pairs that pass the BY/effect screen."""

    path.parent.mkdir(parents=True, exist_ok=True)
    keep = torch.nonzero(result["screened"], as_tuple=False).squeeze(1).tolist()
    names = result["gene_names"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("outcome_gene", "exposure_gene", "tmle_log_interaction", "tmle_se", "q_value", "minimum_conditional_propensity"))
        for row in keep:
            writer.writerow((
                names[int(result["pair_i"][row])],
                names[int(result["pair_j"][row])],
                float(result["tmle_log_interaction"][row]),
                float(result["tmle_se"][row]),
                float(result["q_value"][row]),
                float(result["minimum_conditional_propensity"][row]),
            ))
    print(f"Wrote {len(keep):,} screened directional pairs to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("input/trainingData_11950Cells_1000Genes.csv"))
    parser.add_argument("--model", type=Path, default=Path("output/cross_fitted_pcs_999_genes.pt"))
    parser.add_argument("--output", type=Path, default=Path("output/cross_fitted_conditional_pairwise_tmle_999_genes.pt"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512, help="Outcome/exposure targets per conditional-TMLE block.")
    parser.add_argument("--nuisance-sample-batch-size", type=int, default=64)
    parser.add_argument("--probability-clip", type=float, default=DEFAULT_PROBABILITY_CLIP)
    parser.add_argument("--fluctuation-iterations", type=int, default=64)
    parser.add_argument("--screen-max-q", type=float, default=0.05)
    parser.add_argument("--screen-min-abs-log-interaction", type=float, default=0.25)
    parser.add_argument("--screened-pairs-csv", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--forest-top-n", type=int, default=25)
    parser.add_argument("--network-max-items", type=int, default=25)
    parser.add_argument("--network-max-degree", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    compute_pairwise_tmle(parse_args())
