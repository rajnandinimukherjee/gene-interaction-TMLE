"""Factories and restoration for every shared PC architecture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .model import LatentClassPC
from .tree_models import LatentResidualTreePC, MixtureChowLiuPC, learn_chow_liu_tree


ARCHITECTURES = (
    "factorized",
    "latent-class",
    "mixture-chow-liu",
    "latent-class-residual-tree",
)
TREE_ARCHITECTURES = frozenset(("mixture-chow-liu", "latent-class-residual-tree"))


def build_pc(
    architecture: str,
    data: np.ndarray,
    components: int,
    device: torch.device,
    *,
    parents: Sequence[int] | None = None,
) -> nn.Module:
    """Build and initialize a PC using only the supplied training matrix."""

    n_genes = data.shape[1]
    x = torch.as_tensor(data, dtype=torch.float32, device=device)
    if architecture in {"factorized", "latent-class"}:
        model = LatentClassPC(n_genes, components).to(device)
        model.initialize_from_marginals(x)
        return model
    if architecture in TREE_ARCHITECTURES:
        tree_parents = learn_chow_liu_tree(data) if parents is None else np.asarray(parents, dtype=np.int64)
        if architecture == "mixture-chow-liu":
            model = MixtureChowLiuPC(n_genes, components, tree_parents).to(device)
        else:
            model = LatentResidualTreePC(n_genes, components, tree_parents).to(device)
        model.initialize_from_data(x)
        return model
    raise ValueError(f"Unknown architecture {architecture!r}.")


def restore_pc(
    architecture: str,
    num_genes: int,
    components: int,
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device,
) -> nn.Module:
    """Rebuild a saved PC without relearning a tree from potentially new data."""

    if architecture in {"factorized", "latent-class"}:
        model: nn.Module = LatentClassPC(num_genes, components)
    elif architecture in TREE_ARCHITECTURES:
        try:
            parents = state_dict["parents"].detach().cpu().numpy()
        except KeyError as error:
            raise ValueError("Tree PC checkpoint is missing its learned parents buffer.") from error
        if architecture == "mixture-chow-liu":
            model = MixtureChowLiuPC(num_genes, components, parents)
        else:
            model = LatentResidualTreePC(num_genes, components, parents)
    else:
        raise ValueError(f"Unknown architecture {architecture!r}.")
    model = model.to(device)
    model.load_state_dict(dict(state_dict))
    model.eval()
    return model
