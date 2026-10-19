"""Materialize exact pair marginals and held-out contingency counts once."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .architectures import restore_pc
from .data import GeneData, load_gene_data
from .runtime import choose_device, load_torch_checkpoint


DEFAULT_DATA = "input/trainingData_11950Cells_1000Genes.csv"


def load_cross_fitted_models(
    checkpoint_path: str | Path,
    data: GeneData,
    device: torch.device,
) -> tuple[dict, list[torch.nn.Module], list[np.ndarray]]:
    """Restore every fold model after checking input identity and order."""

    checkpoint = load_torch_checkpoint(checkpoint_path)
    required = {
        "format_version", "model_type", "gene_names", "data_shape", "data_fingerprint",
        "fold_indices", "state_dicts", "model",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing: {', '.join(sorted(missing))}")
    is_legacy_latent = (
        checkpoint["format_version"] == 2
        and checkpoint["model_type"] == "latent_class_pc"
    )
    is_previous_shared_pc = (
        checkpoint["format_version"] == 3
        and checkpoint["model_type"] == "scalable_pc"
    )
    is_shared_pc = (
        checkpoint["format_version"] == 4
        and checkpoint["model_type"] == "cross_fitted_pc"
    )
    if not (is_legacy_latent or is_previous_shared_pc or is_shared_pc):
        raise ValueError(
            "Expected a format-version 2 latent checkpoint, version-3 scalable PC checkpoint, "
            "or version-4 cross-fitted PC checkpoint."
        )
    if tuple(checkpoint["data_shape"]) != data.shape or checkpoint["data_fingerprint"] != data.fingerprint:
        raise ValueError("The supplied data do not exactly match this checkpoint.")
    if tuple(checkpoint["gene_names"]) != data.gene_names:
        raise ValueError("Checkpoint gene order does not match the supplied data.")

    folds = [fold.cpu().numpy().astype(np.int64, copy=False) for fold in checkpoint["fold_indices"]]
    states = checkpoint["state_dicts"]
    if len(folds) < 2 or len(folds) != len(states):
        raise ValueError("Checkpoint has invalid cross-fitting folds.")
    if len(np.unique(np.concatenate(folds))) != len(data.values):
        raise ValueError("Held-out folds must partition every input cell exactly once.")

    components = int(checkpoint["model"]["components"])
    architecture = checkpoint["model"].get("architecture", "latent-class")
    models: list[torch.nn.Module] = []
    for state in states:
        models.append(restore_pc(architecture, data.shape[1], components, state, device))
    return checkpoint, models, folds


def held_out_pair_counts(
    held_out: np.ndarray,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-gene and pair-11 counts from a held-out binary fold."""

    x = torch.as_tensor(held_out, dtype=torch.float32, device=device)
    count_one = x.sum(dim=0).round().to(torch.int32).cpu()
    # Float32 represents all integer counts in this data exactly.  It is much
    # faster than integer GEMM on CUDA and MPS.
    all_count11 = (x.T @ x).round().to(torch.int32)
    count11 = all_count11[pair_i.to(device), pair_j.to(device)].cpu()
    return count_one, count11


def build_pair_cache_from_data(
    data: GeneData,
    *,
    model_path: str | Path,
    output: str | Path,
    device: torch.device,
) -> None:
    """Cache pair marginals for any :class:`GeneData` selection."""

    checkpoint, models, folds = load_cross_fitted_models(model_path, data, device)
    del checkpoint

    pair_i, pair_j = torch.triu_indices(data.shape[1], data.shape[1], offset=1)
    fold_caches: list[dict[str, torch.Tensor | int]] = []
    print(f"device: {device}; caching {len(pair_i):,} pairs for {len(models)} folds")

    for fold, (model, held_out_indices) in enumerate(zip(models, folds), start=1):
        # Exact model cache: use double precision on CPU even when fitting took
        # place on GPU.  This protects rare-cell probabilities during TMLE.
        cpu_model = model.to("cpu")
        p_one, p11_matrix = cpu_model.pair_marginals(dtype=torch.float64)
        count_one, count11 = held_out_pair_counts(
            data.values[held_out_indices], pair_i, pair_j, device
        )
        fold_caches.append(
            {
                "n_held_out": int(len(held_out_indices)),
                "p_one": p_one.cpu(),
                "p11": p11_matrix[pair_i, pair_j].cpu(),
                "count_one": count_one,
                "count11": count11,
            }
        )
        print(f"  fold {fold}: cached {len(held_out_indices):,} held-out cells")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "source_checkpoint": str(Path(model_path)),
            "data_shape": data.shape,
            "data_fingerprint": data.fingerprint,
            "gene_names": list(data.gene_names),
            "pair_i": pair_i,
            "pair_j": pair_j,
            "folds": fold_caches,
        },
        output,
    )
    print(f"Saved pairwise probability/count cache to {output}")


def build_pair_cache(args: argparse.Namespace) -> None:
    """CLI adapter for the all-gene workflow."""

    data = load_gene_data(args.data, max_samples=args.max_samples, max_genes=args.max_genes)
    build_pair_cache_from_data(
        data, model_path=args.model, output=args.output,
        device=choose_device(args.device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--model", type=Path, default=Path("output/cross_fitted_pcs_999_genes.pt"))
    parser.add_argument("--output", type=Path, default=Path("output/cross_fitted_pair_cache_999_genes.pt"))
    parser.add_argument("--max-samples", type=int, default=None, help="Must match training if used.")
    parser.add_argument("--max-genes", type=int, default=None, help="Must match training if used.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    build_pair_cache(parse_args())
