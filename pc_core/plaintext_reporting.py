"""Plaintext reporting helpers for reproducible interaction runs.

The numerical result tables remain Torch checkpoints so downstream code can
load them efficiently.  These helpers deliberately duplicate the essential
provenance, support diagnostics, and a human-sized result table in a text log
that can be inspected without Python or Torch.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO

import torch

from .runtime import load_torch_checkpoint


def format_duration(seconds: float) -> str:
    """Format a wall-clock duration compactly for a run log."""

    hours, remainder = divmod(max(seconds, 0.0), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    if minutes:
        return f"{int(minutes)}m {seconds:.2f}s"
    return f"{seconds:.2f}s"


class _Tee:
    """Small text stream that preserves console output while writing a log."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self._streams)


@contextmanager
def tee_to_plaintext_log(path: Path) -> Iterator[None]:
    """Send stdout and stderr to both their usual streams and ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is a final guard against replacing a historical log
    # if two launches happen to select the same timestamped filename.
    with path.open("x", encoding="utf-8") as handle:
        with (
            redirect_stdout(_Tee(sys.stdout, handle)),
            redirect_stderr(_Tee(sys.stderr, handle)),
        ):
            yield


def print_training_provenance(model_path: Path) -> None:
    """Print checkpoint training metadata that is useful beside TMLE output."""

    print("Training provenance")
    print(f"  Cross-fitted PC checkpoint: {model_path}")
    try:
        checkpoint = load_torch_checkpoint(model_path)
    except Exception as error:  # The analysis will report its own load failure.
        print(f"  Unable to read checkpoint metadata: {error}")
        return

    print(f"  Checkpoint format: {checkpoint.get('format_version', 'unknown')}")
    data_shape = checkpoint.get("data_shape")
    if data_shape is not None:
        print(f"  Training data shape: {tuple(data_shape)} (cells, genes)")
    model = checkpoint.get("model", {})
    if model:
        details = "; ".join(f"{key}={value}" for key, value in model.items())
        print(f"  Model: {details}")

    fold_indices = checkpoint.get("fold_indices", ())
    if fold_indices:
        held_out_sizes = ", ".join(str(len(indices)) for indices in fold_indices)
        print(f"  Cross-fitting folds: {len(fold_indices)} (held out: {held_out_sizes} cells)")

    training = checkpoint.get("training", {})
    if not training:
        print("  Training metadata: not recorded in this checkpoint")
        return
    fields = ("epochs", "batch_size", "learning_rate", "weight_decay", "seed", "device")
    details = "; ".join(
        f"{field}={training[field]}" for field in fields if field in training
    )
    if details:
        print(f"  Training settings: {details}")
    if "elapsed_seconds" in training:
        print(f"  Training runtime: {format_duration(float(training['elapsed_seconds']))}")
    histories = training.get("nll_by_fold")
    if histories:
        final_nll = ", ".join(
            f"fold {fold + 1}: {float(history[-1]):.6f}"
            for fold, history in enumerate(histories) if history
        )
        if final_nll:
            print(f"  Final training NLL: {final_nll}")


def _format_number(value: torch.Tensor | float | int, *, signed: bool = False) -> str:
    """Render a scalar compactly while making unavailable results explicit."""

    number = float(value)
    if not torch.isfinite(torch.tensor(number)):
        return "NA"
    return f"{number:+.6e}" if signed else f"{number:.6e}"


def print_tmle_result_summary(
    result: dict,
    labels: Sequence[str] | Callable[[int], str],
    *,
    max_rows: int | None,
) -> None:
    """Print TMLE support diagnostics and a readable subset of result rows.

    ``max_rows=None`` prints every row, which is suitable for the small
    curated panel.  Large analyses receive the supported rows with the
    greatest absolute TMLE log interaction instead.
    """

    targeted = result["tmle_targeted"].to(torch.bool).cpu()
    supported = result["tmle_supported"].to(torch.bool).cpu()
    tmle = result["tmle_log_interaction"].cpu()
    plugin = result["plugin_log_interaction"].cpu()
    se = result["tmle_se"].cpu()
    ci_low = result["tmle_log_ci_low"].cpu()
    ci_high = result["tmle_log_ci_high"].cpu()
    q_value = result["q_value"].cpu()
    total = len(tmle)

    print("\nTMLE support")
    print(f"  Finite point estimates in every fold: {int(targeted.sum()):,} / {total:,}")
    print(f"  Supported unregularized targets: {int(supported.sum()):,} / {total:,}")
    print(f"  Unsupported targets: {int((~supported).sum()):,}")
    settings = result.get("settings", {})
    if settings:
        minimum_cell_count = settings.get("minimum_cell_count")
        minimum_probability = settings.get("minimum_probability")
        if minimum_cell_count is not None or minimum_probability is not None:
            print(
                "  Support thresholds: "
                f"minimum held-out cell count={minimum_cell_count}; "
                f"minimum model probability={minimum_probability}"
            )
    minimum_counts = result.get("minimum_heldout_cell_count")
    if minimum_counts is not None and settings.get("minimum_cell_count") is not None:
        below_count = minimum_counts.cpu() < settings["minimum_cell_count"]
        print(f"  Below held-out cell-count threshold in any fold: {int(below_count.sum()):,}")
    minimum_probabilities = result.get("minimum_model_probability")
    if minimum_probabilities is not None and settings.get("minimum_probability") is not None:
        below_probability = minimum_probabilities.cpu() < settings["minimum_probability"]
        print(f"  Below model-probability threshold in any fold: {int(below_probability.sum()):,}")

    supported_indices = torch.nonzero(supported, as_tuple=False).flatten()
    if max_rows is None:
        rows = torch.arange(total)
        description = f"all {total:,} targets"
    elif len(supported_indices):
        ranked = torch.argsort(torch.abs(tmle[supported_indices]), descending=True)
        rows = supported_indices[ranked[:max_rows]]
        description = f"top {len(rows):,} supported targets by absolute TMLE log interaction"
    else:
        rows = torch.nonzero(targeted, as_tuple=False).flatten()[:max_rows]
        description = f"up to {len(rows):,} targetable but unsupported targets"

    print(f"\nTMLE results on the log-interaction scale ({description})")
    print(
        f"{'Target':<46}  {'Plug-in':>13}  {'TMLE':>13}  {'TMLE 95% CI':>31}  "
        f"{'q value':>10}  {'SE':>11}  {'Targeted':>9}  {'Supported':>9}"
    )
    print("-" * 166)
    for row in rows.tolist():
        label = labels(row) if callable(labels) else labels[row]
        if bool(supported[row]):
            interval = f"[{_format_number(ci_low[row], signed=True)}, {_format_number(ci_high[row], signed=True)}]"
        else:
            interval = "not supported"
        print(
            f"{label:<46.46}  "
            f"{_format_number(plugin[row], signed=True):>13}  "
            f"{_format_number(tmle[row], signed=True):>13}  "
            f"{interval:>31}  "
            f"{_format_number(q_value[row]):>10}  "
            f"{_format_number(se[row]):>11}  "
            f"{str(bool(targeted[row])):>9}  {str(bool(supported[row])):>9}"
        )
    if max_rows is not None and total > len(rows):
        print(f"  Full numerical results remain in the Torch table: {result.get('output_path', 'see saved output above')}")
