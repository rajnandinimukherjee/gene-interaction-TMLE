"""Query screened higher-order sets and run batched marginal TMLE."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .build_pair_cache import load_cross_fitted_models
from .data import load_gene_data
from .pairwise_tmle import (
    MARGINAL_ESTIMAND,
    NORMAL_95_CRITICAL_VALUE,
    benjamini_hochberg,
)
from .plaintext_reporting import print_tmle_result_summary
from .runtime import choose_device, load_torch_checkpoint


def outcome_signs(
    order: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lexicographic binary outcomes and alternating interaction signs."""

    integers = torch.arange(2**order, device=device)
    shifts = torch.arange(order - 1, -1, -1, device=device)
    outcomes = ((integers[:, None] >> shifts) & 1).to(dtype)
    signs = torch.where(
        ((order - outcomes.sum(dim=1).to(torch.long)) % 2) == 0,
        torch.ones(2**order, dtype=dtype, device=device),
        -torch.ones(2**order, dtype=dtype, device=device),
    )
    return outcomes, signs


@torch.no_grad()
def model_probabilities(
    model, gene_sets: torch.Tensor, outcomes: torch.Tensor
) -> torch.Tensor:
    """Exact PC probabilities for every outcome of each requested gene set."""

    return model.joint_probabilities(gene_sets, outcomes)


@torch.no_grad()
def held_out_counts(
    data: torch.Tensor, gene_sets: torch.Tensor, order: int
) -> torch.Tensor:
    """Build one 2^order contingency table per requested set without Python loops."""

    values = data[:, gene_sets].to(torch.long)  # [cells, sets, order]
    weights = (2 ** torch.arange(order - 1, -1, -1, device=data.device)).to(torch.long)
    code = (values * weights).sum(dim=-1)
    num_sets = gene_sets.shape[0]
    n_outcomes = 2**order
    encoded = code + n_outcomes * torch.arange(num_sets, device=data.device)[None, :]
    return torch.bincount(encoded.reshape(-1), minlength=num_sets * n_outcomes).reshape(
        num_sets, n_outcomes
    )


def universal_tmle_fold(
    probabilities: torch.Tensor,
    counts: torch.Tensor,
    signs: torch.Tensor,
    *,
    minimum_cell_count: int,
    minimum_probability: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target estimable sets and identify those supported for inference."""

    n_sets = len(probabilities)
    dtype, device = probabilities.dtype, probabilities.device
    nan = torch.full((n_sets,), float("nan"), dtype=dtype, device=device)
    min_count = counts.min(dim=1).values
    min_probability_per_set = probabilities.min(dim=1).values
    passes_support_threshold = (
        (min_count >= minimum_cell_count)
        & (min_probability_per_set >= minimum_probability)
        & torch.isfinite(probabilities).all(dim=1)
    )
    # Positivity is enough to calculate a point update.  The configured
    # thresholds below remain mandatory for its standard error and CI.
    targetable = torch.isfinite(probabilities).all(dim=1) & (probabilities > 0.0).all(
        dim=1
    )
    if not targetable.any():
        return nan, nan, nan, torch.zeros_like(targetable)

    selected = torch.nonzero(targetable, as_tuple=False).squeeze(1)
    p = probabilities[selected]
    empirical = counts[selected].to(dtype) / counts[selected].sum(
        dim=1, keepdim=True
    ).to(dtype)
    lower = -p[:, signs > 0].min(dim=1).values
    upper = p[:, signs < 0].min(dim=1).values
    interval = upper - lower
    lower = lower + interval * 1e-12
    upper = upper - interval * 1e-12

    def score(epsilon: torch.Tensor) -> torch.Tensor:
        return (empirical * signs / (p + epsilon[:, None] * signs)).sum(dim=1)

    lower_score, upper_score = score(lower), score(upper)
    root_exists = (
        torch.isfinite(lower_score)
        & torch.isfinite(upper_score)
        & (lower_score > 0.0)
        & (upper_score < 0.0)
    )
    if not root_exists.any():
        return nan, nan, nan, torch.zeros_like(targetable)

    usable = selected[root_exists]
    p, empirical = p[root_exists], empirical[root_exists]
    lo, hi = lower[root_exists], upper[root_exists]
    for _ in range(iterations):
        midpoint = (lo + hi) / 2.0
        score_midpoint = (empirical * signs / (p + midpoint[:, None] * signs)).sum(
            dim=1
        )
        lo = torch.where(score_midpoint > 0.0, midpoint, lo)
        hi = torch.where(score_midpoint > 0.0, hi, midpoint)
    p_star = p + ((lo + hi) / 2.0)[:, None] * signs
    log_interaction = (signs * p_star.log()).sum(dim=1)
    clever_covariate = signs / p_star
    clever_covariate = clever_covariate - (p_star * clever_covariate).sum(
        dim=1, keepdim=True
    )
    usable_counts = counts[usable].to(dtype)

    log_result, sum_result, square_result = nan.clone(), nan.clone(), nan.clone()
    log_result[usable] = log_interaction
    sum_result[usable] = (usable_counts * clever_covariate).sum(dim=1)
    square_result[usable] = (usable_counts * clever_covariate.square()).sum(dim=1)
    supported_mask = torch.zeros_like(targetable)
    supported_mask[usable] = passes_support_threshold[usable]
    return log_result, sum_result, square_result, supported_mask


def run_higher_order_tmle(args: argparse.Namespace) -> None:
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
    candidate_estimand = candidate.get("estimand")
    if candidate_estimand is not None and candidate_estimand != MARGINAL_ESTIMAND:
        raise ValueError("Candidate file was created for a different estimand.")
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

    dtype = torch.float64
    outcomes, signs = outcome_signs(order, device, dtype)
    n_sets = len(gene_sets_cpu)
    plugin_total = torch.zeros(n_sets, dtype=dtype)
    tmle_total = torch.zeros(n_sets, dtype=dtype)
    eif_sum_total = torch.zeros(n_sets, dtype=dtype)
    eif_square_sum_total = torch.zeros(n_sets, dtype=dtype)
    targeted = torch.ones(n_sets, dtype=torch.bool)
    valid = torch.ones(n_sets, dtype=torch.bool)
    min_count = torch.full((n_sets,), torch.iinfo(torch.int32).max, dtype=torch.int32)
    min_probability = torch.full((n_sets,), float("inf"), dtype=dtype)
    fold_sizes: list[int] = []

    print(
        f"Marginal-TMLE device: {device}; querying {n_sets:,} order-{order} candidate sets"
    )
    for fold_number, (model, held_out_indices) in enumerate(
        zip(models, folds), start=1
    ):
        held_out = torch.as_tensor(
            data.values[held_out_indices], dtype=torch.uint8, device=device
        )
        fold_size = len(held_out_indices)
        fold_sizes.append(fold_size)
        fold_log = torch.empty(n_sets, dtype=dtype)
        fold_eif_sum = torch.empty(n_sets, dtype=dtype)
        fold_eif_square_sum = torch.empty(n_sets, dtype=dtype)
        fold_valid = torch.empty(n_sets, dtype=torch.bool)
        for start in range(0, n_sets, args.batch_size):
            stop = min(start + args.batch_size, n_sets)
            gene_sets = gene_sets_cpu[start:stop].to(device)
            probabilities = model_probabilities(model, gene_sets, outcomes)
            counts = held_out_counts(held_out, gene_sets, order)
            plugin_total[start:stop] += (
                fold_size * (probabilities.log() * signs).sum(dim=1).cpu()
            )
            min_count[start:stop] = torch.minimum(
                min_count[start:stop], counts.min(dim=1).values.cpu().to(torch.int32)
            )
            min_probability[start:stop] = torch.minimum(
                min_probability[start:stop], probabilities.min(dim=1).values.cpu()
            )
            log_tmle, eif_sum, eif_square_sum, supported = universal_tmle_fold(
                probabilities,
                counts,
                signs,
                minimum_cell_count=args.minimum_cell_count,
                minimum_probability=args.minimum_probability,
                iterations=args.bisection_iterations,
            )
            fold_log[start:stop] = log_tmle.cpu()
            fold_eif_sum[start:stop] = eif_sum.cpu()
            fold_eif_square_sum[start:stop] = eif_square_sum.cpu()
            fold_valid[start:stop] = supported.cpu()
        valid &= fold_valid
        targeted &= torch.isfinite(fold_log)
        tmle_total += fold_size * torch.nan_to_num(fold_log, nan=0.0)
        eif_sum_total += torch.nan_to_num(fold_eif_sum, nan=0.0)
        eif_square_sum_total += torch.nan_to_num(fold_eif_square_sum, nan=0.0)
        print(
            f"  fold {fold_number}: {int(fold_valid.sum()):,} supported marginal-TMLE targets"
        )

    total_n = sum(fold_sizes)
    plugin_log = plugin_total / total_n
    tmle_log = tmle_total / total_n
    tmle_log[~targeted] = float("nan")
    variance_numerator = eif_square_sum_total - eif_sum_total.square() / total_n
    se = torch.sqrt(torch.clamp(variance_numerator / (total_n - 1) / total_n, min=0.0))
    se[~valid] = float("nan")
    p_value = torch.erfc(
        torch.abs(tmle_log / se) / torch.sqrt(torch.tensor(2.0, dtype=dtype))
    )
    p_value = torch.where(
        (se == 0.0) & (tmle_log == 0.0), torch.ones_like(p_value), p_value
    )
    p_value = torch.where(
        (se == 0.0) & (tmle_log != 0.0), torch.zeros_like(p_value), p_value
    )
    p_value[~valid] = float("nan")
    q_value = benjamini_hochberg(p_value, valid)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "format_version": 2,
        "estimand": MARGINAL_ESTIMAND,
        "source_candidates": str(Path(args.candidates)),
        "data_fingerprint": data.fingerprint,
        "gene_names": list(data.gene_names),
        "gene_sets": gene_sets_cpu,
        "plugin_log_interaction": plugin_log,
        "tmle_log_interaction": tmle_log,
        "tmle_se": se,
        "tmle_log_ci_low": tmle_log - NORMAL_95_CRITICAL_VALUE * se,
        "tmle_log_ci_high": tmle_log + NORMAL_95_CRITICAL_VALUE * se,
        "p_value": p_value,
        "q_value": q_value,
        "minimum_heldout_cell_count": min_count,
        "minimum_model_probability": min_probability,
        "tmle_targeted": targeted,
        "tmle_supported": valid,
        "settings": {
            "minimum_cell_count": args.minimum_cell_count,
            "minimum_probability": args.minimum_probability,
            "total_heldout_cells": total_n,
            "exploratory_warning": "Candidates were selected from the same pairwise data; use an independent confirmation split for confirmatory inference.",
        },
    }
    torch.save(result, output)
    names = result["gene_names"]
    print_tmle_result_summary(
        {**result, "output_path": str(output)},
        lambda row: "-".join(names[int(gene)] for gene in gene_sets_cpu[row]),
        max_rows=25,
    )
    print(f"Saved marginal higher-order results to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("input/trainingData_11950Cells_1000Genes.csv")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("output/cross_fitted_pcs_999_genes.pt")
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("output/marginal_estimand_order3_candidates_999_genes.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/marginal_estimand_order3_tmle_999_genes.pt"),
    )
    parser.add_argument("--minimum-cell-count", type=int, default=5)
    parser.add_argument("--minimum-probability", type=float, default=1e-12)
    parser.add_argument("--bisection-iterations", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Must match training if used."
    )
    parser.add_argument(
        "--max-genes", type=int, default=None, help="Must match training if used."
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_higher_order_tmle(parse_args())
