"""Input validation and reproducible metadata for the gene matrix."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeneData:
    """Binary observations and their named gene columns."""

    values: np.ndarray
    gene_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Gene data must be a two-dimensional matrix.")
        if self.values.shape[1] != len(self.gene_names):
            raise ValueError("The number of gene names does not match the matrix.")
        if self.values.dtype != np.uint8:
            raise ValueError("Gene matrix must use uint8 values.")
        if not np.isin(self.values, (0, 1)).all():
            raise ValueError("This workflow requires binary (0/1) gene data.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def fingerprint(self) -> str:
        """Fingerprint values and column order, protecting checkpoint alignment."""

        digest = hashlib.sha256()
        digest.update("\0".join(self.gene_names).encode("utf-8"))
        digest.update(np.ascontiguousarray(self.values).tobytes())
        return digest.hexdigest()


def _read_gene_names(path: Path) -> list[str]:
    """Read the second CSV line, which is the column-name row in this data."""

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)  # Dataset label line; intentionally unused.
            header = next(reader)
        except StopIteration as error:
            raise ValueError(
                f"{path} does not contain the expected two header rows."
            ) from error

    if len(header) < 2 or header[0] != "":
        raise ValueError(
            "Expected an unnamed leading metadata column followed by named genes."
        )
    if any(not name for name in header[1:]):
        raise ValueError("Every selected gene must have a non-empty column name.")
    if len(set(header[1:])) != len(header) - 1:
        raise ValueError("Gene names must be unique.")
    return header[1:]


def load_gene_data(
    path: str | Path,
    *,
    max_samples: int | None = None,
    max_genes: int | None = None,
) -> GeneData:
    """Load the 999 named binary gene columns, excluding metadata column zero.

    The supplied CSV has 1,000 numeric columns.  Its first column is unnamed
    and constant (currently all ones), so it is metadata rather than a usable
    gene.  This loader always excludes it instead of accidentally fitting a
    degenerate variable.
    """

    path = Path(path)
    names = _read_gene_names(path)
    usecols = tuple(range(1, len(names) + 1))
    values = np.loadtxt(
        path, delimiter=",", dtype=np.uint8, skiprows=2, usecols=usecols
    )
    if values.ndim == 1:
        values = values[None, :]

    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        values = values[:max_samples]
    if max_genes is not None:
        if max_genes < 2:
            raise ValueError("max_genes must be at least two.")
        values = values[:, :max_genes]
        names = names[:max_genes]

    return GeneData(np.ascontiguousarray(values), tuple(names))


def select_genes(data: GeneData, gene_names: tuple[str, ...] | list[str]) -> GeneData:
    """Return named columns in the requested order.

    Keeping selection next to loading means every analysis uses the same CSV
    validation, binary checks, and checkpoint fingerprint semantics.  It also
    makes a small curated panel a view of the all-gene dataset rather than a
    second data-loading implementation.
    """

    requested = tuple(gene_names)
    if not requested:
        raise ValueError("At least one gene must be selected.")
    if len(set(requested)) != len(requested):
        raise ValueError("Selected gene names must be unique.")
    positions = {name: index for index, name in enumerate(data.gene_names)}
    missing = [name for name in requested if name not in positions]
    if missing:
        raise ValueError(f"Selected genes are not in the input: {', '.join(missing)}")
    indices = [positions[name] for name in requested]
    return GeneData(np.ascontiguousarray(data.values[:, indices]), requested)
