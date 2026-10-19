"""Run cross-fitted conditional TMLE for screened ordered gene sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .build_pair_cache import load_cross_fitted_models
from .conditional_tmle import DEFAULT_PROBABILITY_CLIP, run_conditional_tmle
from .data import load_gene_data
from .plaintext_reporting import print_tmle_result_summary
from .runtime import choose_device, load_torch_checkpoint


def run_higher_order_tmle(args: argparse.Namespace) -> None:
    """Target standardized conditional interactions for screened gene sets.

    The first gene in each candidate set is the binary outcome, and all later
    genes jointly form the exposure vector.  The saved candidate ordering is
    therefore part of the estimand, unlike the symmetric marginal branch.
    """

    candidate = load_torch_checkpoint(args.candidates)
    required = {
        "format_version",
        "data_fingerprint",
        "gene_names",
        "gene_sets",
        "settings",
    }
    missing = required - candidate.keys()
    if missing or candidate.get("format_version") != 1:
        raise ValueError(
            f"Expected version-1 candidate file; missing: {', '.join(sorted(missing))}"
        )
    gene_sets_cpu = candidate["gene_sets"].to(torch.long)
    if gene_sets_cpu.ndim != 2 or len(gene_sets_cpu) == 0:
        raise ValueError("Candidate file has no gene sets to evaluate.")
    order = int(gene_sets_cpu.shape[1])
    if order < 3:
        raise ValueError(
            "This entry point is intended for interactions of order three or greater."
        )

    data = load_gene_data(
        args.data, max_samples=args.max_samples, max_genes=args.max_genes
    )
    requested_device = choose_device(args.device)
    device = torch.device("cpu") if requested_device.type == "mps" else requested_device
    _, models, folds = load_cross_fitted_models(args.model, data, device)
    if (
        candidate["data_fingerprint"] != data.fingerprint
        or tuple(candidate["gene_names"]) != data.gene_names
    ):
        raise ValueError(
            "Candidate file was built from different data or a different gene order."
        )

    print(
        f"Conditional TMLE device: {device}; targeting {len(gene_sets_cpu):,} ordered "
        f"order-{order} gene sets across {len(models)} folds"
    )
    estimates = run_conditional_tmle(
        models,
        folds,
        data.values,
        gene_sets_cpu,
        device=device,
        target_batch_size=args.batch_size,
        nuisance_sample_batch_size=args.nuisance_sample_batch_size,
        probability_clip=args.probability_clip,
        fluctuation_iterations=args.fluctuation_iterations,
        progress_interval_seconds=getattr(args, "progress_interval_seconds", 30.0),
    )
    result = {
        "format_version": 3,
        "estimand": "standardized_conditional_log_odds_interaction",
        "source_candidates": str(Path(args.candidates)),
        "source_model": str(Path(args.model)),
        "data_fingerprint": data.fingerprint,
        "gene_names": list(data.gene_names),
        "gene_sets": gene_sets_cpu,
        **estimates,
        "settings": {
            "probability_clip": args.probability_clip,
            "fluctuation_iterations": args.fluctuation_iterations,
            "nuisance_sample_batch_size": args.nuisance_sample_batch_size,
            "total_heldout_cells": len(data.values),
            "multiple_testing": "Benjamini-Yekutieli",
            "outcome_position": 0,
            "covariates": "all genes outside the ordered target",
            "exploratory_warning": "Candidate sets were selected from the same conditional pairwise data; use an independent confirmation split for confirmatory inference.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    names = result["gene_names"]
    print_tmle_result_summary(
        {**result, "output_path": str(output)},
        lambda row: f"{names[int(gene_sets_cpu[row, 0])]} <- "
        + ",".join(names[int(gene)] for gene in gene_sets_cpu[row, 1:]),
        max_rows=25,
    )
    print(f"Saved conditional higher-order results to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("input/trainingData_11950Cells_1000Genes.csv")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("output/cross_fitted_pcs_999_genes.pt")
    )
    parser.add_argument(
        "--candidates", type=Path, default=Path("output/order3_candidates_999_genes.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/order3_tmle_999_genes.pt")
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--nuisance-sample-batch-size", type=int, default=64)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=30.0,
        help="Emit conditional-TMLE progress and a rough ETA this often; 0 disables periodic updates.",
    )
    parser.add_argument(
        "--probability-clip", type=float, default=DEFAULT_PROBABILITY_CLIP
    )
    parser.add_argument("--fluctuation-iterations", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_higher_order_tmle(parse_args())
