"""Cached, batched cross-fitted marginal-TMLE analysis for large gene panels.

This mirrors the familiar 12-gene ``cross_fitted_interactions.py`` interface
where possible. At 999 genes, cached pair marginals are essential, so this
entry point creates/reuses the marginal cache and dispatches to the batched
TMLE path instead of invoking a circuit integration query once per pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from .build_pair_cache import build_pair_cache
from .data import GeneData, load_gene_data
from .higher_order_tmle import run_higher_order_tmle
from .pairwise_tmle import compute_pairwise_tmle
from .plaintext_reporting import format_duration
from .runtime import load_torch_checkpoint
from .screen_higher_order import screen_higher_order


DEFAULT_DATA = "input/trainingData_11950Cells_1000Genes.csv"
DEFAULT_MODEL = "output/cross_fitted_pcs_999_genes.pt"


def _default_path(model: Path, suffix: str) -> Path:
    """Keep derived artifacts beside their checkpoint, not in a global folder."""

    return model.parent / f"{model.stem}{suffix}"


def _requested_gene_indices(
    args: argparse.Namespace, data: GeneData
) -> tuple[int, ...] | None:
    if args.genes is None:
        return None
    requested = tuple(name.strip() for name in args.genes.split(",") if name.strip())
    if len(requested) != args.interaction_order:
        raise ValueError(
            "--genes must contain exactly --interaction-order comma-separated names."
        )
    if len(set(requested)) != len(requested):
        raise ValueError("--genes cannot contain duplicate names.")
    name_to_index = {name: index for index, name in enumerate(data.gene_names)}
    unknown = set(requested) - name_to_index.keys()
    if unknown:
        raise ValueError(f"Unknown gene(s): {', '.join(sorted(unknown))}")
    return tuple(name_to_index[name] for name in requested)


def _cache_matches(cache_path: Path, data: GeneData, model_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    try:
        cache = load_torch_checkpoint(cache_path)
    except Exception:
        return False
    return (
        cache.get("format_version") == 1
        and cache.get("data_fingerprint") == data.fingerprint
        and tuple(cache.get("gene_names", ())) == data.gene_names
        and cache.get("source_checkpoint") == str(model_path)
    )


def ensure_pair_cache(
    args: argparse.Namespace, data: GeneData, cache_path: Path
) -> None:
    started = time.perf_counter()
    if not args.rebuild_cache and _cache_matches(cache_path, data, args.model):
        print(f"Using matching marginal pair cache: {cache_path}")
    else:
        print(f"Building marginal pair cache: {cache_path}")
        build_pair_cache(
            argparse.Namespace(
                data=args.data,
                model=args.model,
                output=cache_path,
                max_samples=args.max_samples,
                max_genes=args.max_genes,
                device=args.device,
            )
        )
    print(
        f"Marginal pair-cache stage runtime: {format_duration(time.perf_counter() - started)}"
    )


def run_pairwise(args: argparse.Namespace, cache_path: Path, output_path: Path) -> dict:
    started = time.perf_counter()
    compute_pairwise_tmle(
        argparse.Namespace(
            cache=cache_path,
            output=output_path,
            minimum_cell_count=args.minimum_cell_count,
            minimum_probability=args.minimum_probability,
            bisection_iterations=args.bisection_iterations,
            screen_max_q=args.screen_max_q,
            screen_min_abs_log_interaction=args.screen_min_abs_log_interaction,
            screened_pairs_csv=args.screened_pairs_csv,
            plots_dir=args.plots_dir,
            forest_top_n=args.forest_top_n,
            network_max_items=args.network_max_items,
            network_max_degree=args.network_max_degree,
            no_plots=args.no_plots,
            device=args.device,
        )
    )
    result = load_torch_checkpoint(output_path)
    print(
        f"Marginal pairwise-TMLE stage runtime: {format_duration(time.perf_counter() - started)}"
    )
    return result


def print_requested_pair(result: dict, gene_indices: tuple[int, ...]) -> None:
    i, j = sorted(gene_indices)
    rows = torch.nonzero(
        (result["pair_i"] == i) & (result["pair_j"] == j), as_tuple=False
    ).flatten()
    if len(rows) != 1:
        raise RuntimeError("Requested pair was not found in the pairwise result table.")
    row = int(rows.item())
    names = result["gene_names"]
    print(f"\nCross-fitted marginal interaction: {names[i]}-{names[j]}")
    print(
        f"  PC marginal log interaction: {float(result['plugin_log_interaction'][row]):.6g}"
    )
    if bool(result["tmle_supported"][row]):
        print(
            f"  Marginal-TMLE log interaction: {float(result['tmle_log_interaction'][row]):.6g}"
        )
        print(
            "  Marginal-TMLE 95% CI (log scale): "
            f"[{float(result['tmle_log_ci_low'][row]):.6g}, "
            f"{float(result['tmle_log_ci_high'][row]):.6g}]"
        )
        print(f"  BH q-value: {float(result['q_value'][row]):.6g}")
    else:
        print("  Marginal TMLE: not estimable under the configured support threshold")


def write_direct_candidate(
    path: Path,
    data: GeneData,
    gene_indices: tuple[int, ...],
) -> None:
    """Write one explicit higher-order set in the candidate-file protocol."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "data_fingerprint": data.fingerprint,
            "gene_names": list(data.gene_names),
            "gene_sets": torch.tensor([gene_indices], dtype=torch.int64),
            "estimand": "marginal_joint_cell_log_interaction",
            "settings": {"order": len(gene_indices), "source": "--genes"},
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    if args.interaction_order < 2:
        raise ValueError("--interaction-order must be at least 2.")
    data = load_gene_data(
        args.data, max_samples=args.max_samples, max_genes=args.max_genes
    )
    gene_indices = _requested_gene_indices(args, data)
    cache_path = args.cache or _default_path(
        args.model, "_marginal_estimand_pair_cache.pt"
    )
    ensure_pair_cache(args, data, cache_path)

    if args.interaction_order == 2:
        output_path = args.output or _default_path(
            args.model, "_marginal_estimand_pairwise_tmle.pt"
        )
        result = run_pairwise(args, cache_path, output_path)
        if gene_indices is not None:
            print_requested_pair(result, gene_indices)
        print(f"Pairwise cross-fitted marginal interactions written to {output_path}")
        return

    pairwise_output = args.pairwise_output or _default_path(
        args.model, "_marginal_estimand_pairwise_tmle.pt"
    )
    run_pairwise(args, cache_path, pairwise_output)
    if gene_indices is not None:
        if args.candidates is not None:
            raise ValueError(
                "Use either --genes or --candidates for higher-order analysis, not both."
            )
        candidate_path = _default_path(
            args.model,
            f"_marginal_estimand_order{args.interaction_order}_direct_candidate.pt",
        )
        write_direct_candidate(candidate_path, data, gene_indices)
    elif args.candidates is not None:
        candidate_path = args.candidates
    else:
        candidate_path = _default_path(
            args.model,
            f"_marginal_estimand_order{args.interaction_order}_candidates.pt",
        )
        screen_higher_order(
            argparse.Namespace(
                pairwise_results=pairwise_output,
                output=candidate_path,
                order=args.interaction_order,
                max_q=args.screen_max_q,
                min_abs_log_interaction=args.screen_min_abs_log_interaction,
                max_degree=args.max_degree,
                edge_policy=args.edge_policy,
                max_candidates=args.max_candidates,
                csv=args.candidates_csv,
            )
        )

    output_path = args.output or _default_path(
        args.model, f"_marginal_estimand_order{args.interaction_order}_tmle.pt"
    )
    higher_order_started = time.perf_counter()
    run_higher_order_tmle(
        argparse.Namespace(
            data=args.data,
            model=args.model,
            candidates=candidate_path,
            output=output_path,
            minimum_cell_count=args.minimum_cell_count,
            minimum_probability=args.minimum_probability,
            bisection_iterations=args.bisection_iterations,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_genes=args.max_genes,
            device=args.device,
        )
    )
    print(
        f"Marginal higher-order-TMLE stage runtime: {format_duration(time.perf_counter() - higher_order_started)}"
    )
    print(
        f"Order-{args.interaction_order} cross-fitted marginal interactions written to {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Compatible with the 12-gene command-line interface.
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--interaction-order", type=int, default=2)
    parser.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Comma-separated genes for one interaction.",
    )
    # Scalable-only controls.
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--pairwise-output",
        type=Path,
        default=None,
        help="Intermediate all-pair marginal-TMLE output used for higher-order screening.",
    )
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--screened-pairs-csv", type=Path, default=None)
    parser.add_argument("--minimum-cell-count", type=int, default=5)
    parser.add_argument("--minimum-probability", type=float, default=1e-12)
    parser.add_argument("--bisection-iterations", type=int, default=64)
    parser.add_argument("--screen-max-q", type=float, default=0.05)
    parser.add_argument("--screen-min-abs-log-interaction", type=float, default=0.25)
    parser.add_argument("--max-degree", type=int, default=20)
    parser.add_argument("--edge-policy", choices=("mutual", "union"), default="mutual")
    parser.add_argument("--max-candidates", type=int, default=100_000)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--forest-top-n", type=int, default=25)
    parser.add_argument("--network-max-items", type=int, default=25)
    parser.add_argument("--network-max-degree", type=int, default=4)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write pairwise overview, forest, and network figures.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-genes", type=int, default=None, help="Development/testing only."
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
