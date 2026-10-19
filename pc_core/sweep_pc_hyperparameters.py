"""Select a PC architecture, width, and learning rate on a gene matrix.

The sweep compares normalized, tractable PC structures implemented by this
package:

* ``factorized``: one product of Bernoulli leaves (the independence baseline);
* ``latent-class``: a root sum over product components, inducing dependence.
* ``mixture-chow-liu``: a mixture of tree PCs over a training-fold topology;
* ``latent-class-residual-tree``: a latent PC with tied sparse tree residuals.

Every candidate uses the full named-gene matrix.  The chosen configuration is
written beside the CSV/plot, while the training CLI defaults are intentionally
kept in source control and updated only after reviewing a completed sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .architectures import ARCHITECTURES, TREE_ARCHITECTURES
from .data import GeneData, load_gene_data
from .runtime import choose_device
from .train_cross_fitted_pcs import effective_components, fit_pc
from .tree_models import learn_chow_liu_tree


matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATA = "input/trainingData_11950Cells_1000Genes.csv"
@torch.no_grad()
def evaluate_nll(model: torch.nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> float:
    """Return held-out mean negative log likelihood without retaining gradients."""

    model.eval()
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    nll_sum = 0.0
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        nll_sum -= model.log_prob(batch).sum().item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return nll_sum / len(x)


def split_data(values: np.ndarray, train_size: int | None, validation_size: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible disjoint train/validation partition."""

    n_rows = len(values)
    if train_size is None and validation_size is None:
        train_size = int(round(0.8 * n_rows))
        validation_size = n_rows - train_size
    elif train_size is None:
        if validation_size is None or not 1 <= validation_size < n_rows:
            raise ValueError("validation_size must be between 1 and n_rows - 1.")
        train_size = n_rows - validation_size
    elif validation_size is None:
        if not 1 <= train_size < n_rows:
            raise ValueError("train_size must be between 1 and n_rows - 1.")
        validation_size = n_rows - train_size
    if train_size is None or validation_size is None or train_size < 1 or validation_size < 1:
        raise ValueError("Both train and validation partitions must be non-empty.")
    if train_size + validation_size > n_rows:
        raise ValueError("Requested train_size + validation_size exceeds available cells.")
    indices = np.random.default_rng(seed).permutation(n_rows)
    return values[indices[:train_size]], values[indices[train_size:train_size + validation_size]]


def plot_results(rows: list[dict[str, str | int | float]], output: Path) -> None:
    """Create the heatmap/learning-rate view used by the original sweep."""

    architectures = list(dict.fromkeys(str(row["architecture"]) for row in rows))
    widths = sorted({int(row["components"]) for row in rows})
    learning_rates = sorted({float(row["learning_rate"]) for row in rows})
    matrix = np.full((len(architectures), len(widths)), np.nan)
    for y, architecture in enumerate(architectures):
        for x, width in enumerate(widths):
            values = [
                float(row["validation_nll"])
                for row in rows
                if row["architecture"] == architecture and int(row["components"]) == width
            ]
            if values:
                matrix[y, x] = min(values)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    image = axes[0].imshow(masked, aspect="auto", cmap="viridis_r")
    axes[0].set(
        title="Best validation NLL over learning rates",
        xlabel="root components",
        ylabel="PC architecture",
        xticks=np.arange(len(widths)),
        xticklabels=widths,
        yticks=np.arange(len(architectures)),
        yticklabels=architectures,
    )
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                axes[0].text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axes[0], label="validation NLL (lower is better)")

    for architecture in architectures:
        best_by_lr = []
        for learning_rate in learning_rates:
            values = [
                float(row["validation_nll"])
                for row in rows
                if row["architecture"] == architecture
                and float(row["learning_rate"]) == learning_rate
            ]
            best_by_lr.append(min(values))
        axes[1].plot(learning_rates, best_by_lr, "o-", label=architecture)
    axes[1].set_xscale("log")
    axes[1].set(
        title="Best architecture width at each learning rate",
        xlabel="AdamW learning rate",
        ylabel="validation NLL (lower is better)",
    )
    axes[1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_architecture_results(
    rows: list[dict[str, str | int | float]],
    winner: dict[str, str | int | float],
    output: Path,
    title: str = "PC architecture and hyperparameter sweep",
) -> None:
    """Write the parallel-coordinates architecture plot used by the 12-gene sweep."""

    architectures = list(dict.fromkeys(str(row["architecture"]) for row in rows))
    architecture_values = {architecture: index for index, architecture in enumerate(architectures)}
    fields = ("architecture", "components", "learning_rate", "parameters", "validation_nll")
    titles = ("PC architecture", "root components", "learning rate", "parameters", "validation NLL")
    columns: list[np.ndarray] = []
    for field in fields:
        if field == "architecture":
            columns.append(np.asarray([architecture_values[str(row[field])] for row in rows], dtype=float))
        elif field == "learning_rate":
            columns.append(np.log10(np.asarray([float(row[field]) for row in rows])))
        else:
            columns.append(np.asarray([float(row[field]) for row in rows]))
    normalized = [
        np.full_like(column, 0.5) if column.min() == column.max()
        else (column - column.min()) / (column.max() - column.min())
        for column in columns
    ]
    nll = columns[-1]
    colour_norm = Normalize(nll.min(), nll.max())
    cmap = plt.get_cmap("viridis_r")
    x = np.arange(len(fields))

    figure, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    for row_index in np.argsort(nll)[::-1]:
        y = np.asarray([column[row_index] for column in normalized])
        axis.plot(x, y, color=cmap(colour_norm(nll[row_index])), alpha=0.45, linewidth=1)
    winner_index = rows.index(winner)
    winner_y = np.asarray([column[winner_index] for column in normalized])
    axis.plot(x, winner_y, color="black", linewidth=3, zorder=5)
    axis.scatter(x, winner_y, color="black", s=30, zorder=6)
    axis.set(
        xlim=(0, len(fields) - 1), ylim=(0, 1), xticks=x, xticklabels=titles,
        yticks=[], title=title,
    )

    for field_index, (field, column) in enumerate(zip(fields, columns)):
        low, high = column.min(), column.max()
        if field == "architecture":
            ticks = np.arange(len(architectures), dtype=float)
            labels = architectures
        elif field == "learning_rate":
            raw = sorted({float(row[field]) for row in rows})
            ticks = np.log10(raw)
            labels = [f"{value:g}" for value in raw]
        elif field in {"components", "parameters"}:
            raw = sorted({int(row[field]) for row in rows})
            ticks = np.asarray(raw, dtype=float)
            labels = [f"{value:,}" for value in raw]
        else:
            ticks = np.linspace(low, high, 5)
            labels = [f"{value:.2f}" for value in ticks]
        positions = np.full(len(ticks), 0.5) if low == high else (ticks - low) / (high - low)
        axis.vlines(field_index, 0, 1, color="0.75", linewidth=1, zorder=0)
        for position, label in zip(positions, labels):
            axis.text(field_index + 0.03, position, label, va="center", fontsize=8, color="0.25")

    selected = (
        "SELECTED DEFAULT\n"
        f"architecture  {winner['architecture']}\n"
        f"components    {int(winner['components'])}\n"
        f"learning rate {float(winner['learning_rate']):g}\n"
        f"parameters    {int(winner['parameters']):,}\n"
        f"validation NLL {float(winner['validation_nll']):.6f}"
    )
    axis.text(
        0.985, 0.98, selected, transform=axis.transAxes, ha="right", va="top",
        fontsize=8, family="monospace", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black", "alpha": 0.92},
    )
    figure.colorbar(ScalarMappable(norm=colour_norm, cmap=cmap), ax=axis, pad=0.02).set_label(
        "validation NLL (lower is better)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_sweep(
    args: argparse.Namespace,
    *,
    data: GeneData | None = None,
) -> dict[str, str | int | float]:
    """Train every candidate and save a likelihood-selected configuration.

    Supplying ``data`` lets analysis-specific callers select a fixed gene panel
    without duplicating sweep/training code.  The direct core CLI keeps its
    all-gene loader behaviour when it is omitted.
    """

    if data is None:
        data = load_gene_data(args.data, max_samples=args.max_samples, max_genes=args.max_genes)
    train, validation = split_data(data.values, args.train_size, args.validation_size, args.seed)
    device = choose_device(args.device)
    print(
        f"device: {device}; named genes: {data.shape[1]:,}; "
        f"train/validation cells: {len(train):,}/{len(validation):,}"
    )

    rows: list[dict[str, str | int | float]] = []
    # The Chow--Liu topology depends only on the training split, not width,
    # learning rate, or tree architecture.  Learn it once rather than 24 times
    # in the standard 20-epoch tree sweep.
    tree_parents = None
    if any(architecture in TREE_ARCHITECTURES for architecture in args.architectures):
        print("Learning one shared Chow--Liu topology for tree candidates.")
        tree_parents = learn_chow_liu_tree(train)
    for architecture in args.architectures:
        requested_widths = (1,) if architecture == "factorized" else tuple(args.components)
        for requested_components in requested_widths:
            components = effective_components(architecture, requested_components)
            for learning_rate in args.learning_rates:
                started = time.perf_counter()
                model, history = fit_pc(
                    train,
                    architecture=architecture,
                    components=components,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=learning_rate,
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                    device=device,
                    parents=tree_parents if architecture in TREE_ARCHITECTURES else None,
                    verbose=False,
                )
                validation_nll = evaluate_nll(model, validation, args.batch_size, device)
                row: dict[str, str | int | float] = {
                    "architecture": architecture,
                    "components": components,
                    "learning_rate": learning_rate,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "validation_nll": validation_nll,
                    "final_train_nll": history[-1],
                    "seconds": time.perf_counter() - started,
                }
                rows.append(row)
                print(
                    f"architecture={architecture:14s} components={components:2d} "
                    f"lr={learning_rate:.4g} parameters={int(row['parameters']):,} "
                    f"validation_NLL={validation_nll:.6f} time={float(row['seconds']):.1f}s"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    winner = min(rows, key=lambda row: float(row["validation_nll"]))
    if args.plot_style == "parallel":
        plot_architecture_results(
            rows, winner, Path(args.output_plot),
            title=getattr(args, "plot_title", "PC architecture and hyperparameter sweep"),
        )
    else:
        plot_results(rows, Path(args.output_plot))
    config_path = Path(args.output_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as handle:
        json.dump(
            {
                "architecture": winner["architecture"],
                "components": winner["components"],
                "learning_rate": winner["learning_rate"],
                "weight_decay": args.weight_decay,
                "validation_nll": winner["validation_nll"],
                "selection_data_shape": data.shape,
                "train_size": len(train),
                "validation_size": len(validation),
                "epochs": args.epochs,
                "seed": args.seed,
                "analysis": getattr(args, "analysis", None),
            },
            handle,
            indent=2,
        )
    print(f"\nSelected configuration: {winner}")
    print(f"Wrote {output_csv}, {args.output_plot}, and {config_path}")
    return winner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument("--components", nargs="+", type=int, default=(2, 4, 8, 16, 32))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=(0.01, 0.02))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Development/testing only.")
    parser.add_argument("--max-genes", type=int, default=None, help="Development/testing only.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-csv", type=Path, default=Path("output/pc_hyperparameter_sweep_999_genes.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("plots/pc_hyperparameter_sweep_999_genes.png"))
    parser.add_argument("--output-config", type=Path, default=Path("output/pc_selected_config_999_genes.json"))
    parser.add_argument("--plot-style", choices=("heatmap", "parallel"), default="heatmap")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
