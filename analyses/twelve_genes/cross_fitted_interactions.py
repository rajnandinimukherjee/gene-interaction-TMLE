"""Conditional-TMLE interactions for the curated 12-gene panel."""

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

from analyses.twelve_genes.config import ANALYSIS_TAG, DATA, MODEL_PATH, OUTPUT_DIR, PLOTS_DIR
from analyses.twelve_genes.data import load_panel
from pc_core.build_pair_cache import load_cross_fitted_models
from pc_core.conditional_tmle import (
    DEFAULT_PROBABILITY_CLIP,
    run_conditional_tmle,
)
from pc_core.interaction_plots import generate_interaction_plots
from pc_core.plaintext_reporting import print_tmle_result_summary
from pc_core.runtime import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--interaction-order", type=int, required=True)
    parser.add_argument("--genes", type=str, default=None,
                        help="Ordered outcome,exposure[,exposure...] gene names; default: panel combinations.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--forest-top-n", type=int, default=25)
    parser.add_argument("--network-max-items", type=int, default=25)
    parser.add_argument("--network-max-degree", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--nuisance-sample-batch-size", type=int, default=64)
    parser.add_argument("--probability-clip", type=float, default=DEFAULT_PROBABILITY_CLIP)
    parser.add_argument("--fluctuation-iterations", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def plot_conditional_interactions(
    labels: list[str],
    plugin: torch.Tensor,
    tmle: torch.Tensor,
    ci_low: torch.Tensor,
    ci_high: torch.Tensor,
    targeted: torch.Tensor,
    order: int,
    output: Path,
) -> None:
    """Write the panel's labelled, directional conditional-TMLE forest plot."""

    plugin_values = plugin.cpu().numpy()
    tmle_values = tmle.cpu().numpy()
    low_values = ci_low.cpu().numpy()
    high_values = ci_high.cpu().numpy()
    valid = targeted.cpu().numpy() & np.isfinite(tmle_values) & np.isfinite(low_values) & np.isfinite(high_values)
    ranking = np.argsort(-np.nan_to_num(np.abs(tmle_values), nan=-np.inf), kind="stable")
    y = np.arange(len(labels))
    height = len(labels) * 0.3 + 0.4
    figure, axis = plt.subplots(figsize=(7.5, height))
    axis.scatter(plugin_values[ranking], y, label="PC plug-in", marker="o", color="tab:blue")
    selected = ranking[valid[ranking]]
    if len(selected):
        selected_y = y[valid[ranking]]
        lower = np.maximum(tmle_values[selected] - low_values[selected], 0.0)
        upper = np.maximum(high_values[selected] - tmle_values[selected], 0.0)
        axis.errorbar(
            tmle_values[selected], selected_y, xerr=np.vstack((lower, upper)),
            fmt="x", capsize=2, color="tab:orange", label="cross-targeted TMLE (95% CI)",
        )
    unavailable = ranking[~valid[ranking] & np.isfinite(tmle_values[ranking])]
    if len(unavailable):
        axis.scatter(tmle_values[unavailable], y[~valid[ranking]], marker="x", color="0.45", label="no finite fluctuation")
    finite = np.concatenate((plugin_values[np.isfinite(plugin_values)], low_values[valid], high_values[valid]))
    half_width = max(0.1, float(np.max(np.abs(finite)))) if len(finite) else 0.1
    axis.axvline(0.0, color="0.5", linewidth=1, linestyle="--")
    axis.set(
        yticks=y,
        yticklabels=[labels[index] for index in ranking],
        xlabel="standardized conditional log interaction",
        title=f"Cross-targeted conditional {order}-gene interactions",
        xlim=(-1.05 * half_width, 1.05 * half_width),
    )
    axis.invert_yaxis()
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend()
    axis.margins(y=0.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _gene_sets(args: argparse.Namespace, names: tuple[str, ...]) -> tuple[torch.Tensor, str | None]:
    order = args.interaction_order
    if args.genes is None:
        return torch.tensor(list(itertools.combinations(range(len(names)), order)), dtype=torch.long), None
    requested = tuple(name.strip() for name in args.genes.split(",") if name.strip())
    if len(requested) != order or len(set(requested)) != len(requested):
        raise ValueError("--genes must name exactly --interaction-order distinct genes.")
    index = {name: position for position, name in enumerate(names)}
    unknown = sorted(set(requested) - index.keys())
    if unknown:
        raise ValueError(f"Unknown panel gene(s): {', '.join(unknown)}")
    return torch.tensor([[index[name] for name in requested]], dtype=torch.long), "_".join(requested)


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    data = load_panel(args.data, max_samples=args.max_samples)
    if not 2 <= args.interaction_order <= len(data.gene_names):
        raise ValueError(f"--interaction-order must be between 2 and {len(data.gene_names)}.")
    device = choose_device(args.device)
    if device.type == "mps":
        device = torch.device("cpu")
    _, models, folds = load_cross_fitted_models(args.model, data, device)
    gene_sets, requested_label = _gene_sets(args, data.gene_names)
    labels = [
        f"{data.gene_names[indices[0]]} <- " + ",".join(data.gene_names[index] for index in indices[1:])
        for indices in gene_sets.tolist()
    ]
    print(f"Conditional TMLE: {len(gene_sets):,} ordered targets across {len(models)} folds.")
    estimates = run_conditional_tmle(
        models,
        folds,
        data.values,
        gene_sets,
        device=device,
        target_batch_size=args.batch_size,
        nuisance_sample_batch_size=args.nuisance_sample_batch_size,
        probability_clip=args.probability_clip,
        fluctuation_iterations=args.fluctuation_iterations,
    )
    if requested_label is None:
        stem = f"{args.interaction_order}_pt_conditional_interactions_{ANALYSIS_TAG}"
    else:
        stem = f"{args.interaction_order}_pt_conditional_interaction_{requested_label}_{ANALYSIS_TAG}"
    output = args.output or OUTPUT_DIR / f"{stem}.pt"
    plot = args.plot or PLOTS_DIR / f"{stem}.png"
    result = {
        "format_version": 3,
        "estimand": "standardized_conditional_log_odds_interaction",
        "source_model": str(args.model),
        "data_fingerprint": data.fingerprint,
        "gene_names": list(data.gene_names),
        "gene_sets": gene_sets,
        **estimates,
        "settings": {
            "probability_clip": args.probability_clip,
            "fluctuation_iterations": args.fluctuation_iterations,
            "nuisance_sample_batch_size": args.nuisance_sample_batch_size,
            "total_heldout_cells": len(data.values),
            "multiple_testing": "Benjamini-Yekutieli",
            "outcome_position": 0,
            "covariates": "all genes outside the ordered target",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print_tmle_result_summary({**result, "output_path": str(output)}, labels, max_rows=None)
    if not args.no_plots:
        plot_conditional_interactions(
            labels,
            result["plugin_log_interaction"],
            result["tmle_log_interaction"],
            result["tmle_log_ci_low"],
            result["tmle_log_ci_high"],
            result["tmle_targeted"],
            args.interaction_order,
            plot,
        )
        plot_paths = generate_interaction_plots(
            result,
            args.plots_dir or PLOTS_DIR,
            output.stem,
            forest_top_n=args.forest_top_n,
            network_max_items=args.network_max_items,
            network_max_degree=args.network_max_degree,
        )
        print("Wrote interaction plots: " + ", ".join(str(path) for path in (plot, *plot_paths)))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
