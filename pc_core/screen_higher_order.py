"""Screen pairwise TMLE hits into bounded higher-order candidate cliques."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .runtime import load_torch_checkpoint


def _screen_edges(
    result: dict, max_q: float, min_abs_log_interaction: float
) -> tuple[list[tuple[int, int, float]], dict[tuple[int, int], int]]:
    supported = result["tmle_supported"].bool()
    q_values = result["q_value"]
    effects = result["tmle_log_interaction"]
    keep = (
        supported
        & torch.isfinite(q_values)
        & (q_values <= max_q)
        & (torch.abs(effects) >= min_abs_log_interaction)
    )
    edges: list[tuple[int, int, float]] = []
    lookup: dict[tuple[int, int], int] = {}
    for row in torch.nonzero(keep, as_tuple=False).squeeze(1).tolist():
        i, j = int(result["pair_i"][row]), int(result["pair_j"][row])
        edge = (i, j) if i < j else (j, i)
        edges.append((edge[0], edge[1], abs(float(effects[row]))))
        lookup[edge] = row
    return edges, lookup


def _bounded_adjacency(
    n_genes: int,
    edges: list[tuple[int, int, float]],
    max_degree: int,
    policy: str,
) -> list[set[int]]:
    """Keep each gene's strongest q/effect-screened neighbours."""

    ranked: list[list[tuple[float, int]]] = [[] for _ in range(n_genes)]
    for i, j, strength in edges:
        ranked[i].append((strength, j))
        ranked[j].append((strength, i))
    selected = [
        {
            neighbour
            for _, neighbour in sorted(values, key=lambda item: (-item[0], item[1]))[
                :max_degree
            ]
        }
        for values in ranked
    ]
    adjacency = [set() for _ in range(n_genes)]
    for i, neighbours in enumerate(selected):
        for j in neighbours:
            keep = (i in selected[j]) if policy == "mutual" else True
            if keep:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def enumerate_cliques(
    adjacency: list[set[int]], order: int, max_candidates: int
) -> list[tuple[int, ...]]:
    """Enumerate unique cliques, aborting safely if the cap is reached."""

    candidates: list[tuple[int, ...]] = []

    def extend(prefix: tuple[int, ...], possible: list[int]) -> None:
        if len(prefix) == order:
            candidates.append(prefix)
            if len(candidates) > max_candidates:
                raise RuntimeError(
                    f"Candidate cap ({max_candidates:,}) reached. Tighten q/effect thresholds "
                    "or reduce --max-degree before proceeding."
                )
            return
        for position, gene in enumerate(possible):
            remainder = possible[position + 1 :]
            extend(
                prefix + (gene,),
                [other for other in remainder if other in adjacency[gene]],
            )

    extend((), list(range(len(adjacency))))
    return candidates


def screen_higher_order(args: argparse.Namespace) -> None:
    if args.order < 3:
        raise ValueError("Higher-order screening requires --order of at least 3.")
    if args.max_degree < args.order - 1:
        raise ValueError("--max-degree must be at least --order - 1.")
    result = load_torch_checkpoint(args.pairwise_results)
    required = {
        "format_version",
        "gene_names",
        "pair_i",
        "pair_j",
        "tmle_supported",
        "q_value",
        "tmle_log_interaction",
    }
    missing = required - result.keys()
    if (
        missing
        or result.get("format_version") != 3
        or result.get("estimand") != "standardized_conditional_log_odds_interaction"
    ):
        raise ValueError(
            "Expected version-3 standardized-conditional pairwise results; "
            f"missing: {', '.join(sorted(missing))}"
        )

    edges, row_lookup = _screen_edges(result, args.max_q, args.min_abs_log_interaction)
    adjacency = _bounded_adjacency(
        len(result["gene_names"]), edges, args.max_degree, args.edge_policy
    )
    retained_edge_count = sum(len(neighbours) for neighbours in adjacency) // 2
    candidates = enumerate_cliques(adjacency, args.order, args.max_candidates)

    candidate_tensor = (
        torch.tensor(candidates, dtype=torch.int64)
        if candidates
        else torch.empty((0, args.order), dtype=torch.int64)
    )
    pair_q = result["q_value"]
    pair_effect = result["tmle_log_interaction"]
    min_pair_effect: list[float] = []
    max_pair_q: list[float] = []
    for genes in candidates:
        rows = [
            row_lookup[(i, j)]
            for index, i in enumerate(genes)
            for j in genes[index + 1 :]
        ]
        min_pair_effect.append(min(abs(float(pair_effect[row])) for row in rows))
        max_pair_q.append(max(float(pair_q[row]) for row in rows))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "source_pairwise_results": str(Path(args.pairwise_results)),
            "data_fingerprint": result.get("data_fingerprint"),
            "gene_names": result["gene_names"],
            "gene_sets": candidate_tensor,
            "minimum_abs_pair_effect": torch.tensor(
                min_pair_effect, dtype=torch.float64
            ),
            "maximum_pair_q_value": torch.tensor(max_pair_q, dtype=torch.float64),
            "settings": {
                "order": args.order,
                "max_q": args.max_q,
                "min_abs_log_interaction": args.min_abs_log_interaction,
                "max_degree": args.max_degree,
                "edge_policy": args.edge_policy,
                "max_candidates": args.max_candidates,
                "screened_edges_before_degree_cap": len(edges),
                "retained_edges_after_degree_cap": retained_edge_count,
            },
        },
        output,
    )
    print(f"Pairwise q/effect hits: {len(edges):,}")
    print(
        f"Edges retained after {args.edge_policy} degree cap: {retained_edge_count:,}"
    )
    print(f"Order-{args.order} candidates: {len(candidates):,}")
    print(f"Saved candidates to {output}")
    if args.csv is not None:
        write_candidates_csv(output, args.csv)


def write_candidates_csv(candidate_path: Path, csv_path: Path) -> None:
    candidate = load_torch_checkpoint(candidate_path)
    names = candidate["gene_names"]
    sets = candidate["gene_sets"].tolist()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        order = len(sets[0]) if sets else int(candidate["settings"]["order"])
        writer.writerow(
            [f"gene_{index + 1}" for index in range(order)]
            + ["minimum_abs_pair_effect", "maximum_pair_q_value"]
        )
        for index, genes in enumerate(sets):
            writer.writerow(
                [names[gene] for gene in genes]
                + [
                    float(candidate["minimum_abs_pair_effect"][index]),
                    float(candidate["maximum_pair_q_value"][index]),
                ]
            )
    print(f"Wrote candidate table to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairwise-results",
        type=Path,
        default=Path("output/cross_fitted_pairwise_tmle_999_genes.pt"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/order3_candidates_999_genes.pt")
    )
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--max-q", type=float, default=0.05)
    parser.add_argument("--min-abs-log-interaction", type=float, default=0.25)
    parser.add_argument("--max-degree", type=int, default=20)
    parser.add_argument("--edge-policy", choices=("mutual", "union"), default="mutual")
    parser.add_argument("--max-candidates", type=int, default=100_000)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    screen_higher_order(parse_args())
