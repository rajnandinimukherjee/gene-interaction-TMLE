"""Load each study's likelihood-selected PC defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pc_core.architectures import ARCHITECTURES

from analyses.nine_nine_nine_genes import config as all_gene_config
from analyses.twelve_genes import config as panel_config


@dataclass(frozen=True)
class TrainingDefaults:
    architecture: str
    components: int
    learning_rate: float
    weight_decay: float


def _fallback_config(analysis: int):
    if analysis == 12:
        return panel_config
    if analysis == 999:
        return all_gene_config
    raise ValueError(f"Unsupported analysis {analysis!r}; expected 12 or 999.")


def training_defaults(analysis: int) -> TrainingDefaults:
    """Use the saved 20-epoch winner when available, else a safe fallback."""

    config = _fallback_config(analysis)
    fallback = TrainingDefaults(
        architecture=config.ARCHITECTURE,
        components=config.COMPONENTS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    selection_path = Path(config.ARCHITECTURE_SELECTION_PATH)
    if not selection_path.is_file():
        return fallback
    try:
        with selection_path.open() as handle:
            selected = json.load(handle)
        architecture = str(selected["architecture"])
        components = int(selected["components"])
        learning_rate = float(selected["learning_rate"])
        weight_decay = float(selected.get("weight_decay", fallback.weight_decay))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid selected-PC configuration: {selection_path}"
        ) from error
    if (
        architecture not in ARCHITECTURES
        or components < 1
        or learning_rate <= 0
        or weight_decay < 0
    ):
        raise ValueError(f"Invalid selected-PC configuration: {selection_path}")
    return TrainingDefaults(architecture, components, learning_rate, weight_decay)
