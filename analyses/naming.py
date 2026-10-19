"""Keep generated artifacts visibly partitioned by study size."""

from __future__ import annotations

from pathlib import Path


def analysis_tag(analysis: int) -> str:
    """Return the mandatory filename marker for a supported study."""

    if analysis not in (12, 999):
        raise ValueError(f"Unsupported analysis {analysis!r}; expected 12 or 999.")
    return f"{analysis}_genes"


def tagged_path(path: Path | None, analysis: int) -> Path | None:
    """Add the study marker before a filename extension when it is absent.

    Input checkpoints are intentionally not passed through this helper: old
    checkpoints remain readable.  Every artifact created by the unified CLIs
    is passed through it, including an explicitly supplied output path.
    """

    if path is None:
        return None
    path = Path(path)
    tag = analysis_tag(analysis)
    if tag in path.name:
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")
