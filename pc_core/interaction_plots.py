"""Create scalable summaries of cross-fitted log-interaction results.

The module accepts either result-table layout used in this repository:
``pair_i``/``pair_j`` for pairwise studies or ``gene_sets`` for a fixed
interaction order.  It writes three complementary figures:

* an all-result effect-versus-evidence overview;
* a readable forest plot of the strongest positive and negative effects; and
* a sparse signed network (or signed hypergraph for interactions of order >2).

It deliberately never attempts to put every interaction on a categorical
axis.  That remains readable for a small panel but not for an all-gene study.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
# Command-line plotting can be invoked outside either study module, so give
# Matplotlib a writable cache rather than relying on a user's home directory.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pc_interaction_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pc_interaction_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import torch

from .runtime import load_torch_checkpoint


NORMAL_95_CRITICAL_VALUE = 1.959963984540054
POSITIVE_COLOUR = "tab:red"
NEGATIVE_COLOUR = "tab:blue"
NEUTRAL_COLOUR = "0.35"


def _as_numpy(value: object, *, dtype: type | None = None) -> np.ndarray:
    """Detach a saved tensor and return it as a NumPy array."""

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _interaction_indices(result: dict) -> np.ndarray:
    """Return one row of gene indices per interaction in a result table."""

    if "pair_i" in result and "pair_j" in result:
        return np.column_stack((
            _as_numpy(result["pair_i"], dtype=int),
            _as_numpy(result["pair_j"], dtype=int),
        ))
    if "gene_sets" in result:
        indices = _as_numpy(result["gene_sets"], dtype=int)
        if indices.ndim != 2:
            raise ValueError("Expected result['gene_sets'] to have one row per interaction.")
        return indices
    raise ValueError("Result table must contain pair_i/pair_j or gene_sets.")


def _result_arrays(
    result: dict,
) -> tuple[tuple[str, ...], np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract names, memberships, estimates, CIs, support, and q-values."""

    names = tuple(str(name) for name in result["gene_names"])
    memberships = _interaction_indices(result)
    estimates = _as_numpy(result["tmle_log_interaction"], dtype=float)
    supported = _as_numpy(result.get("tmle_supported", np.isfinite(estimates)), dtype=bool)
    if len(memberships) != len(estimates) or len(supported) != len(estimates):
        raise ValueError("Interaction result vectors have inconsistent lengths.")

    if "tmle_log_ci_low" in result and "tmle_log_ci_high" in result:
        ci_low = _as_numpy(result["tmle_log_ci_low"], dtype=float)
        ci_high = _as_numpy(result["tmle_log_ci_high"], dtype=float)
    elif "tmle_se" in result:
        se = _as_numpy(result["tmle_se"], dtype=float)
        ci_low = estimates - NORMAL_95_CRITICAL_VALUE * se
        ci_high = estimates + NORMAL_95_CRITICAL_VALUE * se
    else:
        ci_low = estimates.copy()
        ci_high = estimates.copy()

    q_values = None
    if "q_value" in result:
        candidate_q = _as_numpy(result["q_value"], dtype=float)
        if len(candidate_q) == len(estimates):
            q_values = candidate_q

    conditional = result.get("estimand") == "standardized_conditional_log_odds_interaction"
    if conditional:
        labels = [
            f"{names[int(row[0])]} <- " + ",".join(names[int(index)] for index in row[1:])
            for row in memberships
        ]
    else:
        labels = ["-".join(names[int(index)] for index in row) for row in memberships]
    return names, memberships, labels, estimates, ci_low, ci_high, supported, q_values


def _supported_indices(estimates: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, supported: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        supported & np.isfinite(estimates) & np.isfinite(ci_low) & np.isfinite(ci_high)
    )


def _screened_display_indices(
    supported_indices: np.ndarray,
    estimates: np.ndarray,
    q_values: np.ndarray | None,
    *,
    max_q: float,
    min_abs_log_interaction: float,
) -> np.ndarray:
    """Select display candidates, falling back to all supported estimates."""

    if q_values is None:
        keep = np.abs(estimates[supported_indices]) >= min_abs_log_interaction
    else:
        keep = (
            np.isfinite(q_values[supported_indices])
            & (q_values[supported_indices] <= max_q)
            & (np.abs(estimates[supported_indices]) >= min_abs_log_interaction)
        )
    selected = supported_indices[keep]
    return selected if len(selected) else supported_indices


def _write_empty_figure(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_effect_evidence_overview(
    estimates: np.ndarray,
    supported_indices: np.ndarray,
    q_values: np.ndarray | None,
    *,
    title: str,
    max_q: float,
    min_abs_log_interaction: float,
    output: Path,
) -> None:
    """Plot every supported result without trying to label individual pairs."""

    if not len(supported_indices):
        _write_empty_figure(output, title, "No interactions supported for inference")
        return

    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    values = estimates[supported_indices]
    if q_values is None or not np.isfinite(q_values[supported_indices]).any():
        bins = min(70, max(15, int(np.sqrt(len(values)))))
        axis.hist(values, bins=bins, color=NEUTRAL_COLOUR, edgecolor="white", linewidth=0.35)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.axvline(min_abs_log_interaction, color=POSITIVE_COLOUR, linestyle="--", linewidth=1)
        axis.axvline(-min_abs_log_interaction, color=NEGATIVE_COLOUR, linestyle="--", linewidth=1)
        axis.set(
            title=title,
            xlabel="TMLE log(interaction)",
            ylabel="Number of supported interactions",
        )
    else:
        finite_q = np.isfinite(q_values[supported_indices])
        rows = supported_indices[finite_q]
        q = q_values[rows]
        positive_q = q[q > 0.0]
        floor = positive_q.min() if len(positive_q) else np.finfo(float).tiny
        # Extremely small q-values add no useful vertical resolution to a
        # scatter plot and may underflow to zero.  Cap only the display scale;
        # the saved result table retains the exact q-values.
        evidence_cap = 50.0
        evidence = np.minimum(-np.log10(np.clip(q, floor, 1.0)), evidence_cap)
        positive = estimates[rows] >= 0.0
        # A 12-gene panel has at most hundreds of points, whereas the
        # all-gene pairwise plot has hundreds of thousands.  The latter needs
        # transparency to reveal density; the former needs visible marks.
        if len(rows) <= 1_000:
            marker_size, marker_alpha = 24, 0.82
        elif len(rows) <= 10_000:
            marker_size, marker_alpha = 10, 0.42
        else:
            marker_size, marker_alpha = 5, 0.18
        axis.scatter(
            estimates[rows][~positive], evidence[~positive], s=marker_size, alpha=marker_alpha,
            color=NEGATIVE_COLOUR, linewidths=0, rasterized=True, label="negative estimate",
        )
        axis.scatter(
            estimates[rows][positive], evidence[positive], s=marker_size, alpha=marker_alpha,
            color=POSITIVE_COLOUR, linewidths=0, rasterized=True, label="positive estimate",
        )
        axis.axvline(0.0, color="black", linewidth=1)
        axis.axvline(min_abs_log_interaction, color=POSITIVE_COLOUR, linestyle="--", linewidth=1)
        axis.axvline(-min_abs_log_interaction, color=NEGATIVE_COLOUR, linestyle="--", linewidth=1)
        axis.axhline(-np.log10(max_q), color=NEUTRAL_COLOUR, linestyle="--", linewidth=1)
        axis.set(
            title=title,
            xlabel="TMLE log(interaction)",
            ylabel=f"−log10(BY q-value), capped at {evidence_cap:g}",
        )
        axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", color="0.9", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _top_by_sign(indices: np.ndarray, estimates: np.ndarray, *, positive: bool, top_n: int) -> np.ndarray:
    direction = indices[estimates[indices] >= 0.0] if positive else indices[estimates[indices] < 0.0]
    if positive:
        ordered = sorted(direction.tolist(), key=lambda row: (-estimates[row], row))
    else:
        ordered = sorted(direction.tolist(), key=lambda row: (estimates[row], row))
    return np.asarray(ordered[:top_n], dtype=int)


def plot_ranked_forest(
    labels: list[str],
    estimates: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    display_indices: np.ndarray,
    *,
    title: str,
    top_n: int,
    output: Path,
) -> None:
    """Write side-by-side, signed forest plots for a manageable result subset."""

    positive = _top_by_sign(display_indices, estimates, positive=True, top_n=top_n)
    negative = _top_by_sign(display_indices, estimates, positive=False, top_n=top_n)
    selected = np.concatenate((negative, positive))
    if not len(selected):
        _write_empty_figure(output, title, "No interactions available for the ranked forest plot")
        return

    extent = np.concatenate((ci_low[selected], ci_high[selected], np.array([0.0])))
    extent = extent[np.isfinite(extent)]
    limit = max(abs(float(extent.min())), abs(float(extent.max()))) if len(extent) else 1.0
    limit = max(0.1, limit * 1.08)
    rows = max(len(positive), len(negative), 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, max(3.5, 0.34 * rows + 1.6)), sharex=True)
    for axis, indices, colour, heading in (
        (axes[0], negative, NEGATIVE_COLOUR, "Strongest negative interactions"),
        (axes[1], positive, POSITIVE_COLOUR, "Strongest positive interactions"),
    ):
        axis.axvline(0.0, color="black", linewidth=1, zorder=0)
        if len(indices):
            y = np.arange(len(indices))
            lower_error = np.maximum(estimates[indices] - ci_low[indices], 0.0)
            upper_error = np.maximum(ci_high[indices] - estimates[indices], 0.0)
            axis.errorbar(
                estimates[indices], y, xerr=np.vstack((lower_error, upper_error)),
                fmt="o", color=colour, ecolor=colour, capsize=2, markersize=4,
            )
            axis.set_yticks(y, [labels[row] for row in indices])
            axis.invert_yaxis()
        else:
            axis.text(0.5, 0.5, "None selected", ha="center", va="center", transform=axis.transAxes)
            axis.set_yticks([])
        axis.set(title=heading, xlabel="TMLE log(interaction)", xlim=(-limit, limit))
        axis.grid(axis="x", color="0.9", linewidth=0.8)
    axes[0].set_ylabel("Gene set")
    figure.suptitle(title, y=1.01)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _component_spring_layout(nodes: np.ndarray, edges: list[tuple[int, int]]) -> dict[int, np.ndarray]:
    """Lay out one connected component with a small force simulation."""

    node_count = len(nodes)
    if node_count == 1:
        return {int(nodes[0]): np.array((0.0, 0.0))}
    angles = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    index = {int(node): location for location, node in enumerate(nodes)}
    edge_locations = [(index[i], index[j]) for i, j in edges]
    ideal = 1.4 / np.sqrt(node_count)
    for step in range(180):
        displacement = np.zeros_like(positions)
        for i in range(node_count):
            delta = positions[i] - positions
            distance = np.linalg.norm(delta, axis=1)
            nonzero = distance > 1e-9
            displacement[i] += (
                delta[nonzero] / distance[nonzero, None]
                * (ideal * ideal / distance[nonzero, None])
            ).sum(axis=0)
        for i, j in edge_locations:
            delta = positions[i] - positions[j]
            distance = max(float(np.linalg.norm(delta)), 1e-9)
            pull = delta / distance * (distance * distance / ideal)
            displacement[i] -= pull
            displacement[j] += pull
        temperature = 0.10 * (1.0 - step / 180.0) + 0.005
        magnitude = np.linalg.norm(displacement, axis=1)
        move = displacement / np.maximum(magnitude[:, None], 1e-9)
        positions += move * np.minimum(magnitude, temperature)[:, None]
    positions -= positions.mean(axis=0, keepdims=True)
    radius = np.max(np.abs(positions))
    if radius > 0:
        positions /= radius
    return {int(node): positions[location] for location, node in enumerate(nodes)}


def _spring_layout(nodes: np.ndarray, edges: list[tuple[int, int]]) -> dict[int, np.ndarray]:
    """Return a deterministic layout, packing disconnected components apart."""

    adjacency = {int(node): set() for node in nodes}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: list[list[int]] = []
    unvisited = set(adjacency)
    while unvisited:
        seed = min(unvisited)
        unvisited.remove(seed)
        component = [seed]
        for node in component:
            neighbours = adjacency[node] & unvisited
            unvisited -= neighbours
            component.extend(sorted(neighbours))
        components.append(sorted(component))
    components.sort(key=lambda component: (-len(component), component[0]))
    if len(components) == 1:
        component = np.asarray(components[0], dtype=int)
        return _component_spring_layout(component, edges)

    positions: dict[int, np.ndarray] = {}
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, len(components), endpoint=False)
    for component, angle in zip(components, angles):
        component_set = set(component)
        local_edges = [(first, second) for first, second in edges if first in component_set and second in component_set]
        local = _component_spring_layout(np.asarray(component, dtype=int), local_edges)
        centre = np.array((0.72 * np.cos(angle), 0.72 * np.sin(angle)))
        radius = 0.14 + 0.045 * np.sqrt(len(component))
        positions.update({node: centre + radius * point for node, point in local.items()})
    return positions


def _bounded_pair_network_indices(
    display_indices: np.ndarray,
    memberships: np.ndarray,
    estimates: np.ndarray,
    *,
    max_edges: int,
    max_degree: int,
) -> np.ndarray:
    """Keep strong edges while preventing a single gene from filling the graph."""

    ordered = sorted(display_indices.tolist(), key=lambda row: (-abs(estimates[row]), row))
    degree: dict[int, int] = {}
    selected: list[int] = []
    for row in ordered:
        first, second = (int(value) for value in memberships[row])
        if degree.get(first, 0) >= max_degree or degree.get(second, 0) >= max_degree:
            continue
        selected.append(row)
        degree[first] = degree.get(first, 0) + 1
        degree[second] = degree.get(second, 0) + 1
        if len(selected) == max_edges:
            break
    return np.asarray(selected, dtype=int)


def _plot_pair_network(
    names: tuple[str, ...],
    memberships: np.ndarray,
    estimates: np.ndarray,
    display_indices: np.ndarray,
    *,
    title: str,
    max_edges: int,
    max_degree: int,
    directed: bool,
    output: Path,
) -> None:
    rows = _bounded_pair_network_indices(
        display_indices, memberships, estimates, max_edges=max_edges, max_degree=max_degree,
    )
    if not len(rows):
        _write_empty_figure(output, title, "No interactions available for the network")
        return
    edges = [(int(memberships[row, 0]), int(memberships[row, 1])) for row in rows]
    nodes = np.unique(memberships[rows].reshape(-1))
    # For a compact panel, a circular layout gives every gene an unobscured
    # label.  Larger all-gene subsets benefit from component-aware placement.
    if len(nodes) <= 12:
        angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, len(nodes), endpoint=False)
        positions = {
            int(node): 0.86 * np.array((np.cos(angle), np.sin(angle)))
            for node, angle in zip(nodes, angles)
        }
    else:
        positions = _spring_layout(nodes, edges)
    degrees = {int(node): 0 for node in nodes}
    for first, second in edges:
        degrees[first] += 1
        degrees[second] += 1

    figure, axis = plt.subplots(figsize=(9, 8))
    maximum = max(abs(estimates[rows]).max(), 1e-12)
    for row, (first, second) in zip(rows, edges):
        colour = POSITIVE_COLOUR if estimates[row] >= 0.0 else NEGATIVE_COLOUR
        width = 0.6 + 3.0 * abs(estimates[row]) / maximum
        start, end = positions[first], positions[second]
        if directed:
            # Membership is stored as (outcome, exposure), so arrowheads point
            # from exposure to outcome.  This is a descriptive direction set
            # by the estimand, not a causal-direction claim.
            axis.annotate(
                "", xy=start, xytext=end,
                arrowprops={"arrowstyle": "->", "color": colour, "lw": width, "alpha": 0.68,
                            "shrinkA": 12, "shrinkB": 12},
                zorder=1,
            )
        else:
            axis.plot((start[0], end[0]), (start[1], end[1]), color=colour, linewidth=width, alpha=0.68, zorder=1)
    for node in nodes:
        point = positions[int(node)]
        axis.scatter(point[0], point[1], s=80 + 32 * degrees[int(node)], color="white", edgecolor="black", linewidth=0.8, zorder=2)
        direction = point / max(float(np.linalg.norm(point)), 1e-9)
        axis.text(point[0] + 0.055 * direction[0], point[1] + 0.055 * direction[1], names[int(node)],
                  ha="center", va="center", fontsize=8, zorder=3)
    axis.legend(handles=(
        Line2D([0], [0], color=POSITIVE_COLOUR, linewidth=2, label="positive log interaction"),
        Line2D([0], [0], color=NEGATIVE_COLOUR, linewidth=2, label="negative log interaction"),
    ), frameon=False, loc="upper right")
    axis.set(
        title=f"{title}\n(up to {len(rows)} screened edges; max degree {max_degree})",
        xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_hypergraph(
    names: tuple[str, ...],
    memberships: np.ndarray,
    estimates: np.ndarray,
    display_indices: np.ndarray,
    *,
    title: str,
    max_sets: int,
    output: Path,
) -> None:
    """Represent selected n-way interactions as signed polygons over genes."""

    rows = np.asarray(
        sorted(display_indices.tolist(), key=lambda row: (-abs(estimates[row]), row))[:max_sets], dtype=int,
    )
    if not len(rows):
        _write_empty_figure(output, title, "No interactions available for the hypergraph")
        return
    nodes = np.unique(memberships[rows].reshape(-1))
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, len(nodes), endpoint=False)
    positions = {int(node): np.array((np.cos(angle), np.sin(angle))) for node, angle in zip(nodes, angles)}

    figure, axis = plt.subplots(figsize=(9, 8))
    maximum = max(abs(estimates[rows]).max(), 1e-12)
    for rank, row in enumerate(rows):
        vertices = np.asarray([positions[int(node)] for node in memberships[row]])
        # Stagger equally sized gene sets slightly toward the centre so their
        # outlines remain distinguishable when they share members.
        centre = vertices.mean(axis=0)
        scale = 0.95 - 0.025 * (rank % 12)
        vertices = centre + scale * (vertices - centre)
        colour = POSITIVE_COLOUR if estimates[row] >= 0.0 else NEGATIVE_COLOUR
        width = 0.8 + 1.6 * abs(estimates[row]) / maximum
        # Filled polygons rapidly obscure one another for sets of three or
        # more genes.  Outlines retain each selected hyperedge's membership.
        axis.add_patch(Polygon(vertices, closed=True, facecolor="none", edgecolor=colour,
                               linewidth=width, alpha=0.65, zorder=1))
    for node in nodes:
        point = positions[int(node)]
        axis.scatter(point[0], point[1], s=105, color="white", edgecolor="black", linewidth=0.8, zorder=2)
        axis.text(point[0] * 1.12, point[1] * 1.12, names[int(node)], ha="center", va="center", fontsize=9, zorder=3)
    axis.legend(handles=(
        Line2D([0], [0], color=POSITIVE_COLOUR, linewidth=2, label="positive log interaction"),
        Line2D([0], [0], color=NEGATIVE_COLOUR, linewidth=2, label="negative log interaction"),
    ), frameon=False, loc="upper right")
    axis.set(title=title, xlim=(-1.32, 1.32), ylim=(-1.32, 1.32), aspect="equal")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_interaction_plots(
    result: dict,
    output_dir: Path,
    prefix: str,
    *,
    max_q: float = 0.05,
    min_abs_log_interaction: float = 0.25,
    forest_top_n: int = 25,
    network_max_items: int = 25,
    network_max_degree: int = 4,
) -> tuple[Path, Path, Path]:
    """Generate overview, forest, and signed network plots from one result table."""

    if forest_top_n < 1:
        raise ValueError("forest_top_n must be at least one.")
    if network_max_items < 1 or network_max_degree < 1:
        raise ValueError("Network limits must be at least one.")
    names, memberships, labels, estimates, ci_low, ci_high, supported, q_values = _result_arrays(result)
    supported_indices = _supported_indices(estimates, ci_low, ci_high, supported)
    display_indices = _screened_display_indices(
        supported_indices, estimates, q_values,
        max_q=max_q, min_abs_log_interaction=min_abs_log_interaction,
    )
    order = int(memberships.shape[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = output_dir / f"{prefix}_overview.png"
    forest = output_dir / f"{prefix}_forest.png"
    network = output_dir / f"{prefix}_{'network' if order == 2 else 'hypergraph'}.png"
    conditional = result.get("estimand") == "standardized_conditional_log_odds_interaction"
    quantity = (
        f"Cross-targeted standardized conditional {order}-point log interactions"
        if conditional else f"Cross-fitted {order}-point log interactions"
    )
    plot_effect_evidence_overview(
        estimates, supported_indices, q_values,
        title=(
            f"{quantity}: effect and evidence"
            if q_values is not None and np.isfinite(q_values[supported_indices]).any()
            else f"{quantity}: effect distribution"
        ),
        max_q=max_q,
        min_abs_log_interaction=min_abs_log_interaction,
        output=overview,
    )
    plot_ranked_forest(
        labels, estimates, ci_low, ci_high, display_indices,
        title=f"{quantity}: ranked TMLE estimates (95% CI)",
        top_n=forest_top_n,
        output=forest,
    )
    if order == 2:
        _plot_pair_network(
            names, memberships, estimates, display_indices,
            title=f"{quantity}: sparse signed network",
            max_edges=network_max_items,
            max_degree=network_max_degree,
            directed=conditional,
            output=network,
        )
    else:
        _plot_hypergraph(
            names, memberships, estimates, display_indices,
            title=f"{quantity}: top signed {order}-way sets",
            # Polygon overlays become illegible much sooner than pairwise
            # edges; keep this focused on the strongest few n-way sets.
            max_sets=min(network_max_items, 8),
            output=network,
        )
    return overview, forest, network


def _default_plot_directory(result_path: Path) -> Path:
    if result_path.parent.name == "output":
        return result_path.parent.parent / "plots"
    return result_path.parent / "plots"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Saved .pt interaction result table.")
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--max-q", type=float, default=0.05)
    parser.add_argument("--min-abs-log-interaction", type=float, default=0.25)
    parser.add_argument("--forest-top-n", type=int, default=25)
    parser.add_argument("--network-max-items", type=int, default=25)
    parser.add_argument("--network-max-degree", type=int, default=4)
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    result_path = Path(args.results)
    result = load_torch_checkpoint(result_path)
    outputs = generate_interaction_plots(
        result,
        args.plots_dir or _default_plot_directory(result_path),
        args.prefix or result_path.stem,
        max_q=args.max_q,
        min_abs_log_interaction=args.min_abs_log_interaction,
        forest_top_n=args.forest_top_n,
        network_max_items=args.network_max_items,
        network_max_degree=args.network_max_degree,
    )
    print("Wrote interaction plots: " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
