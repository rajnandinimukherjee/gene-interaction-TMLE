"""Run cached, batched cross-fitted marginal-TMLE for every gene pair."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .interaction_plots import generate_interaction_plots
from .plaintext_reporting import print_tmle_result_summary
from .runtime import choose_device, load_torch_checkpoint


NORMAL_95_CRITICAL_VALUE = 1.959963984540054
MARGINAL_ESTIMAND = "marginal_joint_cell_log_interaction"


def _pair_probabilities(
    fold: dict, pair_i: torch.Tensor, pair_j: torch.Tensor
) -> torch.Tensor:
    """Construct [p00, p01, p10, p11] for all cached pairs."""

    p_one = fold["p_one"]
    p11 = fold["p11"]
    pi = p_one[pair_i]
    pj = p_one[pair_j]
    return torch.stack((1.0 - pi - pj + p11, pj - p11, pi - p11, p11), dim=1)


def _pair_counts(
    fold: dict, pair_i: torch.Tensor, pair_j: torch.Tensor
) -> torch.Tensor:
    """Construct [n00, n01, n10, n11] held-out counts for all pairs."""

    n = int(fold["n_held_out"])
    counts_one = fold["count_one"]
    n11 = fold["count11"]
    ni = counts_one[pair_i]
    nj = counts_one[pair_j]
    return torch.stack((n - ni - nj + n11, nj - n11, ni - n11, n11), dim=1)


def _log_interaction(probabilities: torch.Tensor) -> torch.Tensor:
    return (
        probabilities[:, 0].log()
        + probabilities[:, 3].log()
        - probabilities[:, 1].log()
        - probabilities[:, 2].log()
    )


def _tmle_fold(
    probabilities: torch.Tensor,
    counts: torch.Tensor,
    *,
    minimum_cell_count: int,
    minimum_probability: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target all estimable pair distributions using vectorized bisection.

    The least-favourable path is p_e(a)=p_0(a)+e*s(a), where the signs for
    outcomes (00, 01, 10, 11) are (+1, -1, -1, +1).  The returned support
    mask is deliberately stricter than point-estimate availability: it also
    enforces the pre-specified held-out count and model-probability thresholds
    required for valid unregularized inference.
    """

    device = probabilities.device
    dtype = probabilities.dtype
    n_pairs = len(probabilities)
    signs = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=dtype, device=device)
    nan = torch.full((n_pairs,), float("nan"), dtype=dtype, device=device)
    min_count = counts.min(dim=1).values
    min_probability_per_pair = probabilities.min(dim=1).values
    passes_support_threshold = (
        (min_count >= minimum_cell_count)
        & (min_probability_per_pair >= minimum_probability)
        & torch.isfinite(probabilities).all(dim=1)
    )
    # A finite, strictly positive initial distribution is sufficient to try
    # the targeting update.  It is intentionally less strict than the
    # inferential support rule above, so a finite point estimate can still be
    # reported without a CI for sparse targets.
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

    lower = -p[:, (0, 3)].min(dim=1).values
    upper = p[:, (1, 2)].min(dim=1).values
    # Stay strictly within the positivity interval even for tiny probabilities.
    interval = upper - lower
    lower = lower + interval * 1e-12
    upper = upper - interval * 1e-12

    def score(epsilon: torch.Tensor) -> torch.Tensor:
        return (empirical * signs / (p + epsilon[:, None] * signs)).sum(dim=1)

    lower_score = score(lower)
    upper_score = score(upper)
    root_exists = (
        torch.isfinite(lower_score)
        & torch.isfinite(upper_score)
        & (lower_score > 0.0)
        & (upper_score < 0.0)
    )
    if not root_exists.any():
        return nan, nan, nan, torch.zeros_like(targetable)

    usable = selected[root_exists]
    p = p[root_exists]
    empirical = empirical[root_exists]
    lo, hi = lower[root_exists], upper[root_exists]
    for _ in range(iterations):
        midpoint = (lo + hi) / 2.0
        midpoint_score = (empirical * signs / (p + midpoint[:, None] * signs)).sum(
            dim=1
        )
        lo = torch.where(midpoint_score > 0.0, midpoint, lo)
        hi = torch.where(midpoint_score > 0.0, hi, midpoint)
    epsilon = (lo + hi) / 2.0
    p_star = p + epsilon[:, None] * signs

    log_tmle = _log_interaction(p_star)
    clever_covariate = signs / p_star
    # E_P[H] is zero analytically, but centre explicitly for numerical safety.
    clever_covariate = clever_covariate - (p_star * clever_covariate).sum(
        dim=1, keepdim=True
    )
    selected_counts = counts[usable].to(dtype)
    eif_sum = (selected_counts * clever_covariate).sum(dim=1)
    eif_square_sum = (selected_counts * clever_covariate.square()).sum(dim=1)

    log_result, sum_result, square_result = nan.clone(), nan.clone(), nan.clone()
    log_result[usable] = log_tmle
    sum_result[usable] = eif_sum
    square_result[usable] = eif_square_sum
    supported_mask = torch.zeros_like(targetable)
    supported_mask[usable] = passes_support_threshold[usable]
    return log_result, sum_result, square_result, supported_mask


def benjamini_hochberg(p_values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return BH q-values, with NaN for untestable hypotheses."""

    q_values = torch.full_like(p_values, float("nan"))
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if len(valid_indices) == 0:
        return q_values
    p = p_values[valid_indices].clamp(0.0, 1.0)
    sorted_p, order = torch.sort(p)
    ranks = torch.arange(1, len(p) + 1, dtype=p.dtype, device=p.device)
    raw_q = sorted_p * len(p) / ranks
    monotone_q = torch.flip(
        torch.cummin(torch.flip(raw_q, dims=(0,)), dim=0).values, dims=(0,)
    ).clamp(max=1.0)
    q_values[valid_indices[order]] = monotone_q
    return q_values


def compute_pairwise_tmle(args: argparse.Namespace) -> None:
    cache = load_torch_checkpoint(args.cache)
    required = {
        "format_version",
        "gene_names",
        "pair_i",
        "pair_j",
        "folds",
        "data_fingerprint",
    }
    missing = required - cache.keys()
    if missing or cache.get("format_version") != 1:
        raise ValueError(
            f"Expected a version-1 pair cache; missing: {', '.join(sorted(missing))}"
        )

    requested_device = choose_device(args.device)
    # MPS currently has limited float64 support.  TMLE is inexpensive once the
    # cache exists, so preserve numerical stability by using CPU double there.
    device = torch.device("cpu") if requested_device.type == "mps" else requested_device
    dtype = torch.float64
    pair_i = cache["pair_i"].to(device=device, dtype=torch.long)
    pair_j = cache["pair_j"].to(device=device, dtype=torch.long)
    n_pairs = len(pair_i)

    plugin_log_total = torch.zeros(n_pairs, dtype=dtype, device=device)
    tmle_log_by_fold: list[torch.Tensor] = []
    eif_sum_by_fold: list[torch.Tensor] = []
    eif_square_sum_by_fold: list[torch.Tensor] = []
    valid_by_fold: list[torch.Tensor] = []
    min_count = torch.full(
        (n_pairs,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device
    )
    min_probability = torch.full((n_pairs,), float("inf"), dtype=dtype, device=device)
    fold_sizes: list[int] = []

    print(
        f"Marginal-TMLE device: {device}; evaluating {n_pairs:,} pairs across {len(cache['folds'])} folds"
    )
    for fold_number, raw_fold in enumerate(cache["folds"], start=1):
        fold = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_fold.items()
        }
        probabilities = _pair_probabilities(fold, pair_i, pair_j).to(dtype)
        counts = _pair_counts(fold, pair_i, pair_j).to(torch.int32)
        fold_size = int(fold["n_held_out"])
        fold_sizes.append(fold_size)
        plugin_log_total += fold_size * _log_interaction(probabilities)
        min_count = torch.minimum(min_count, counts.min(dim=1).values)
        min_probability = torch.minimum(
            min_probability, probabilities.min(dim=1).values
        )
        tmle_log, eif_sum, eif_square_sum, valid = _tmle_fold(
            probabilities,
            counts,
            minimum_cell_count=args.minimum_cell_count,
            minimum_probability=args.minimum_probability,
            iterations=args.bisection_iterations,
        )
        tmle_log_by_fold.append(tmle_log)
        eif_sum_by_fold.append(eif_sum)
        eif_square_sum_by_fold.append(eif_square_sum)
        valid_by_fold.append(valid)
        print(
            f"  fold {fold_number}: {int(valid.sum()):,} supported marginal-TMLE targets"
        )

    total_n = sum(fold_sizes)
    plugin_log = plugin_log_total / total_n
    valid = torch.stack(valid_by_fold).all(dim=0)
    targeted = torch.stack([torch.isfinite(values) for values in tmle_log_by_fold]).all(
        dim=0
    )
    # Replacing unavailable fold-level updates by zero is safe only after
    # retaining values that were targeted in every fold.
    tmle_log = (
        sum(
            size * torch.nan_to_num(values, nan=0.0)
            for size, values in zip(fold_sizes, tmle_log_by_fold)
        )
        / total_n
    )
    eif_sum = sum(torch.nan_to_num(values, nan=0.0) for values in eif_sum_by_fold)
    eif_square_sum = sum(
        torch.nan_to_num(values, nan=0.0) for values in eif_square_sum_by_fold
    )
    variance_numerator = eif_square_sum - eif_sum.square() / total_n
    se = torch.sqrt(torch.clamp(variance_numerator / (total_n - 1) / total_n, min=0.0))
    se[~valid] = float("nan")
    tmle_log[~targeted] = float("nan")
    z = tmle_log / se
    p_value = torch.erfc(
        torch.abs(z) / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    )
    # Degenerate EIFs can occur for nearly deterministic variables.
    p_value = torch.where(
        (se == 0.0) & (tmle_log == 0.0), torch.ones_like(p_value), p_value
    )
    p_value = torch.where(
        (se == 0.0) & (tmle_log != 0.0), torch.zeros_like(p_value), p_value
    )
    p_value[~valid] = float("nan")
    q_value = benjamini_hochberg(p_value, valid)
    screened = (
        valid
        & (q_value <= args.screen_max_q)
        & (torch.abs(tmle_log) >= args.screen_min_abs_log_interaction)
    )

    log_ci_low = tmle_log - NORMAL_95_CRITICAL_VALUE * se
    log_ci_high = tmle_log + NORMAL_95_CRITICAL_VALUE * se
    result = {
        "format_version": 2,
        "estimand": MARGINAL_ESTIMAND,
        "source_cache": str(Path(args.cache)),
        "data_fingerprint": cache["data_fingerprint"],
        "gene_names": cache["gene_names"],
        "pair_i": pair_i.cpu(),
        "pair_j": pair_j.cpu(),
        "plugin_log_interaction": plugin_log.cpu(),
        "tmle_log_interaction": tmle_log.cpu(),
        "tmle_se": se.cpu(),
        "tmle_log_ci_low": log_ci_low.cpu(),
        "tmle_log_ci_high": log_ci_high.cpu(),
        "p_value": p_value.cpu(),
        "q_value": q_value.cpu(),
        "minimum_heldout_cell_count": min_count.cpu(),
        "minimum_model_probability": min_probability.cpu(),
        "tmle_targeted": targeted.cpu(),
        "tmle_supported": valid.cpu(),
        "screened": screened.cpu(),
        "settings": {
            "minimum_cell_count": args.minimum_cell_count,
            "minimum_probability": args.minimum_probability,
            "screen_max_q": args.screen_max_q,
            "screen_min_abs_log_interaction": args.screen_min_abs_log_interaction,
            "total_heldout_cells": total_n,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    names = result["gene_names"]
    print_tmle_result_summary(
        {**result, "output_path": str(output)},
        lambda row: f"{names[int(pair_i[row])]}-{names[int(pair_j[row])]}",
        max_rows=25,
    )
    print(f"Pairs passing marginal q/effect screen: {int(screened.sum()):,}")
    print(f"Saved all marginal pairwise results to {output}")

    if not getattr(args, "no_plots", False):
        default_plot_dir = (
            output.parent.parent / "plots"
            if output.parent.name == "output"
            else output.parent / "plots"
        )
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
    """Write only screened pairs, avoiding a half-million-line text report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    keep = torch.nonzero(result["screened"], as_tuple=False).squeeze(1).tolist()
    names = result["gene_names"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "gene_1",
                "gene_2",
                "marginal_tmle_log_interaction",
                "tmle_se",
                "q_value",
                "minimum_heldout_cell_count",
            )
        )
        for row in keep:
            writer.writerow(
                (
                    names[int(result["pair_i"][row])],
                    names[int(result["pair_j"][row])],
                    float(result["tmle_log_interaction"][row]),
                    float(result["tmle_se"][row]),
                    float(result["q_value"][row]),
                    int(result["minimum_heldout_cell_count"][row]),
                )
            )
    print(f"Wrote {len(keep):,} screened marginal pairs to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("output/cross_fitted_marginal_estimand_pair_cache_999_genes.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/cross_fitted_marginal_estimand_pairwise_tmle_999_genes.pt"
        ),
    )
    parser.add_argument("--minimum-cell-count", type=int, default=5)
    parser.add_argument("--minimum-probability", type=float, default=1e-12)
    parser.add_argument("--bisection-iterations", type=int, default=64)
    parser.add_argument("--screen-max-q", type=float, default=0.05)
    parser.add_argument("--screen-min-abs-log-interaction", type=float, default=0.25)
    parser.add_argument("--screened-pairs-csv", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--forest-top-n", type=int, default=25)
    parser.add_argument("--network-max-items", type=int, default=25)
    parser.add_argument("--network-max-degree", type=int, default=4)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write overview, forest, and network figures.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    compute_pairwise_tmle(parse_args())
