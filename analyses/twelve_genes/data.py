"""Load the curated panel through the shared all-gene CSV loader."""

from pathlib import Path

from pc_core.data import GeneData, load_gene_data, select_genes

from .config import GENE_NAMES


def load_panel(path: str | Path, *, max_samples: int | None = None) -> GeneData:
    return select_genes(load_gene_data(path, max_samples=max_samples), GENE_NAMES)
