"""Plot every marginal order-n log interaction for the curated 12-gene panel.

This is the replacement for the legacy top-level script.  It retains its two
important options, ``--model`` and ``--interaction-order``, but queries the
shared selected-PC checkpoint format rather than the retired Cirkit format.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from pc_core.build_pair_cache import load_cross_fitted_models
from pc_core.higher_order_tmle import (
    held_out_counts,
    model_probabilities,
    outcome_signs,
    universal_tmle_fold,
)
from pc_core.interaction_plots import generate_interaction_plots
from pc_core.pairwise_tmle import MARGINAL_ESTIMAND, NORMAL_95_CRITICAL_VALUE, benjamini_hochberg
from pc_core.plaintext_reporting import print_tmle_result_summary
from pc_core.runtime import choose_device

from analyses.twelve_genes.config import ANALYSIS_TAG, DATA, MODEL_PATH, OUTPUT_DIR, PLOTS_DIR
from analyses.twelve_genes.data import load_panel
from analyses.naming import marginal_estimand_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--interaction-order", type=int, required=True)
    parser.add_argument(
        "--genes", type=str, default=None,
        help="Optional comma-separated gene set. By default every panel combination is evaluated.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional .pt result table.")
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--forest-top-n", type=int, default=25)
    parser.add_argument("--network-max-items", type=int, default=25)
    parser.add_argument("--network-max-degree", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true", help="Do not write overview, forest, and hypergraph figures.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum-cell-count", type=int, default=5)
    parser.add_argument("--minimum-probability", type=float, default=1e-12)
    parser.add_argument("--bisection-iterations", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def plot_log_interactions(
    labels: list[str], plugin: torch.Tensor, tmle: torch.Tensor, se: torch.Tensor,
    tmle_supported: torch.Tensor, empirical: torch.Tensor, interaction_order: int, output: Path,
) -> None:
    """Write the legacy-style vertical comparison for every requested gene set.

    The canonical 12-gene plot deliberately has one labelled row per
    interaction: ``C(12, order)`` rows when no individual gene set is
    requested.  Keep this separate from the scalable overview/forest figures,
    which summarise rather than enumerate the results.
    """

    # Rank gene sets solely by their empirical log interaction.  Finite values
    # appear first in descending order; undefined empirical interactions stay
    # at the bottom, retaining their original (lexicographic) order.
    empirical_values = empirical.numpy()
    empirical_finite = np.isfinite(empirical_values)
    finite_indices = np.flatnonzero(empirical_finite)
    order = np.concatenate((
        finite_indices[np.argsort(-empirical_values[finite_indices], kind="stable")],
        np.flatnonzero(~empirical_finite),
    ))
    y = np.arange(len(labels))
    plugin_values = plugin.numpy()[order]
    tmle_values = tmle.numpy()[order]
    errors = se.numpy()[order]
    supported_values = tmle_supported.numpy()[order]
    empirical_values = empirical.numpy()[order]
    # This is intentionally tall.  The legacy figure showed every labelled
    # interaction vertically rather than collapsing the panel into a summary.
    height = len(labels) * 0.3 + 0.2
    figure, axis = plt.subplots(figsize=(6, height))
    axis.scatter(plugin_values, y, label="marginal PC plug-in", marker="o")
    valid = supported_values & np.isfinite(tmle_values) & np.isfinite(errors)
    axis.errorbar(tmle_values[valid], y[valid], xerr=NORMAL_95_CRITICAL_VALUE * errors[valid],
                  fmt="x", capsize=2, color="tab:orange", label="marginal TMLE (95% CI)")
    unsupported = ~supported_values & np.isfinite(tmle_values)
    axis.scatter(tmle_values[unsupported], y[unsupported], marker="x", color="tab:orange",
                 zorder=3, label="TMLE (unsupported; no CI)")
    empirical_valid = np.isfinite(empirical_values)
    axis.scatter(empirical_values[empirical_valid], y[empirical_valid], marker="*", color="tab:green",
                 zorder=4, label="empirical")

    # Retain the legacy connector between the plug-in and TMLE estimates.
    # Unsupported TMLE values remain visible, but no fabricated CI is drawn.
    connected = np.isfinite(plugin_values) & np.isfinite(tmle_values)
    for index in np.flatnonzero(connected):
        axis.plot([plugin_values[index], tmle_values[index]], [y[index], y[index]],
                  linewidth=1, color="tab:orange", zorder=2)

    # A symmetric log-scale range makes the no-interaction reference (zero)
    # visually comparable across the full vertical panel.
    ci_low = tmle_values[valid] - NORMAL_95_CRITICAL_VALUE * errors[valid]
    ci_high = tmle_values[valid] + NORMAL_95_CRITICAL_VALUE * errors[valid]
    plotted_values = np.concatenate((
        plugin_values[np.isfinite(plugin_values)],
        tmle_values[np.isfinite(tmle_values)],
        ci_low[np.isfinite(ci_low)],
        ci_high[np.isfinite(ci_high)],
        empirical_values[empirical_valid],
    ))
    half_width = max(np.max(np.abs(plotted_values)), 0.1) if plotted_values.size else 0.1
    axis.set_xlim(-1.05 * half_width, 1.05 * half_width)
    axis.axvline(0, color="0.5", linewidth=1, linestyle="--", zorder=0)
    axis.set(yticks=y, yticklabels=[labels[index] for index in order],
             xlabel="marginal log(interaction strength)")
    axis.set_title(f"Cross-fitted marginal {interaction_order}-pt log interactions", pad=0)
    # Matplotlib places y=0 at the bottom by default.  Reverse it so the
    # largest empirical log interaction is the top gene tuple in the plot.
    axis.invert_yaxis()
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend()
    axis.margins(y=0.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(
        left=0.25,
        right=0.95,
        bottom=0.1 / height,
        top=1 - 0.1 / height,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    data = load_panel(args.data, max_samples=args.max_samples)
    order = args.interaction_order
    if not 2 <= order <= len(data.gene_names):
        raise ValueError(f"--interaction-order must be between 2 and {len(data.gene_names)}.")
    device = choose_device(args.device)
    # Float64 TMLE is not available on MPS; the panel is small, so CPU is the
    # reliable fallback while model checkpoints remain device-independent.
    if device.type == "mps":
        device = torch.device("cpu")
    _, models, folds = load_cross_fitted_models(args.model, data, device)
    requested_label = None
    if args.genes is None:
        gene_sets_cpu = torch.tensor(
            list(itertools.combinations(range(len(data.gene_names)), order)), dtype=torch.long
        )
    else:
        requested = tuple(name.strip() for name in args.genes.split(",") if name.strip())
        if len(requested) != order:
            raise ValueError("--genes must contain exactly --interaction-order comma-separated names.")
        if len(set(requested)) != len(requested):
            raise ValueError("--genes cannot contain duplicate names.")
        indices = {name: index for index, name in enumerate(data.gene_names)}
        unknown = sorted(set(requested) - indices.keys())
        if unknown:
            raise ValueError(f"Unknown panel gene(s): {', '.join(unknown)}")
        gene_sets_cpu = torch.tensor([[indices[name] for name in requested]], dtype=torch.long)
        requested_label = "-".join(requested)
    labels = ["-".join(data.gene_names[index] for index in gene_set) for gene_set in gene_sets_cpu.tolist()]
    dtype = torch.float64
    outcomes, signs = outcome_signs(order, device, dtype)
    count = len(gene_sets_cpu)
    plugin_total = torch.zeros(count, dtype=dtype)
    tmle_total = torch.zeros(count, dtype=dtype)
    eif_sum_total = torch.zeros(count, dtype=dtype)
    eif_square_sum_total = torch.zeros(count, dtype=dtype)
    targeted = torch.ones(count, dtype=torch.bool)
    valid = torch.ones(count, dtype=torch.bool)
    minimum_heldout_cell_count = torch.full(
        (count,), torch.iinfo(torch.int32).max, dtype=torch.int32
    )
    minimum_model_probability = torch.full((count,), float("inf"), dtype=dtype)
    fold_sizes: list[int] = []

    print(f"Evaluating {count:,} marginal order-{order} interactions across {len(models)} folds.")
    for fold_number, (model, held_out_indices) in enumerate(zip(models, folds), start=1):
        held_out = torch.as_tensor(data.values[held_out_indices], dtype=torch.uint8, device=device)
        fold_size = len(held_out)
        fold_sizes.append(fold_size)
        fold_log = torch.empty(count, dtype=dtype)
        fold_eif_sum = torch.empty(count, dtype=dtype)
        fold_eif_square_sum = torch.empty(count, dtype=dtype)
        fold_valid = torch.empty(count, dtype=torch.bool)
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            gene_sets = gene_sets_cpu[start:stop].to(device)
            probabilities = model_probabilities(model, gene_sets, outcomes).to(dtype)
            counts = held_out_counts(held_out, gene_sets, order)
            plugin_total[start:stop] += fold_size * (probabilities.log() * signs).sum(dim=1).cpu()
            minimum_heldout_cell_count[start:stop] = torch.minimum(
                minimum_heldout_cell_count[start:stop],
                counts.min(dim=1).values.cpu().to(torch.int32),
            )
            minimum_model_probability[start:stop] = torch.minimum(
                minimum_model_probability[start:stop], probabilities.min(dim=1).values.cpu()
            )
            log_tmle, eif_sum, eif_square_sum, supported = universal_tmle_fold(
                probabilities, counts, signs, minimum_cell_count=args.minimum_cell_count,
                minimum_probability=args.minimum_probability, iterations=args.bisection_iterations,
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
        print(f"  fold {fold_number}: {int(fold_valid.sum()):,}/{count:,} marginal-TMLE targets supported")

    total_n = sum(fold_sizes)
    plugin_log = plugin_total / total_n
    tmle_log = tmle_total / total_n
    tmle_log[~targeted] = float("nan")
    variance = eif_square_sum_total - eif_sum_total.square() / total_n
    se = torch.sqrt(torch.clamp(variance / (total_n - 1) / total_n, min=0.0))
    se[~valid] = float("nan")
    z = tmle_log / se
    p_value = torch.erfc(torch.abs(z) / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
    p_value = torch.where((se == 0.0) & (tmle_log == 0.0), torch.ones_like(p_value), p_value)
    p_value = torch.where((se == 0.0) & (tmle_log != 0.0), torch.zeros_like(p_value), p_value)
    p_value[~valid] = float("nan")
    q_value = benjamini_hochberg(p_value, valid)
    # This deliberately uses every selected cell, matching the empirical
    # comparator in the legacy figure rather than cross-fitted held-out folds.
    full_data = torch.as_tensor(data.values, dtype=torch.uint8, device=device)
    empirical_counts = torch.empty((count, 2 ** order), dtype=torch.int64)
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        empirical_counts[start:stop] = held_out_counts(
            full_data, gene_sets_cpu[start:stop].to(device), order
        ).cpu()
    empirical_probabilities = empirical_counts.to(dtype) / len(full_data)
    # Empty observed cells make an unregularized empirical interaction
    # undefined; expose that as NaN rather than plotting misleading infinity.
    empirical_log = (empirical_probabilities.log() * signs.cpu()).sum(dim=1)
    empirical_log[empirical_counts.eq(0).any(dim=1)] = float("nan")
    # A focused --genes query must not overwrite the canonical all-combinations
    # result or its tall vertical plot.  Explicit output paths retain their
    # usual override behaviour.
    if requested_label is None:
        default_stem = f"{order}_pt_marginal_estimand_interactions_{ANALYSIS_TAG}"
    else:
        default_stem = f"{order}_pt_marginal_estimand_interaction_{requested_label}_{ANALYSIS_TAG}"
    output = marginal_estimand_path(args.output, 12) or OUTPUT_DIR / f"{default_stem}.pt"
    plot = marginal_estimand_path(args.plot, 12) or PLOTS_DIR / f"{default_stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "format_version": 2,
        "estimand": MARGINAL_ESTIMAND,
        "gene_names": list(data.gene_names),
        "gene_sets": gene_sets_cpu,
        "plugin_log_interaction": plugin_log,
        "tmle_log_interaction": tmle_log,
        "tmle_se": se,
        "tmle_log_ci_low": tmle_log - NORMAL_95_CRITICAL_VALUE * se,
        "tmle_log_ci_high": tmle_log + NORMAL_95_CRITICAL_VALUE * se,
        "p_value": p_value,
        "q_value": q_value,
        "empirical_log_interaction": empirical_log,
        "tmle_targeted": targeted,
        "tmle_supported": valid,
        "minimum_heldout_cell_count": minimum_heldout_cell_count,
        "minimum_model_probability": minimum_model_probability,
        "source_model": str(args.model),
        "settings": {
            "minimum_cell_count": args.minimum_cell_count,
            "minimum_probability": args.minimum_probability,
            "total_heldout_cells": total_n,
        },
    }
    torch.save(result, output)
    print_tmle_result_summary({**result, "output_path": str(output)}, labels, max_rows=None)
    plot_log_interactions(labels, plugin_log, tmle_log, se, valid, empirical_log, order, plot)
    if not getattr(args, "no_plots", False):
        plot_paths = generate_interaction_plots(
            result,
            Path(getattr(args, "plots_dir", None) or PLOTS_DIR),
            output.stem,
            forest_top_n=getattr(args, "forest_top_n", 25),
            network_max_items=getattr(args, "network_max_items", 25),
            network_max_degree=getattr(args, "network_max_degree", 4),
        )
        print("Wrote interaction plots: " + ", ".join(str(path) for path in plot_paths))
    print(f"Wrote marginal interaction results to {output} and {plot}")


if __name__ == "__main__":
    main()
