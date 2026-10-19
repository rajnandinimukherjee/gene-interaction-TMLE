"""Train cross-fitted latent-class PCs over every named gene column."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .architectures import ARCHITECTURES, build_pc
from .data import GeneData, load_gene_data
from .runtime import choose_device, make_folds


DEFAULT_DATA = "input/trainingData_11950Cells_1000Genes.csv"
DEFAULT_ARCHITECTURE = "latent-class-residual-tree"
DEFAULT_COMPONENTS = 32
# The direct all-gene core CLI uses the 999-gene 20-epoch sweep winner.  The
# unified CLI loads the separate 12-gene winner from its study configuration.
DEFAULT_LEARNING_RATE = 0.01


def effective_components(architecture: str, requested_components: int) -> int:
    """Map a PC architecture and width option to its actual root width."""

    if architecture == "factorized":
        return 1
    if architecture in {"latent-class", "mixture-chow-liu", "latent-class-residual-tree"}:
        if requested_components < 2:
            raise ValueError("latent-class architecture requires at least two components.")
        return requested_components
    raise ValueError(f"Unknown architecture {architecture!r}.")


def architecture_parameter_count(architecture: str, components: int, num_genes: int) -> int:
    """Return trainable parameters for the selected normalized PC."""

    if architecture in {"factorized", "latent-class"}:
        return components * (num_genes + 1)
    if architecture == "mixture-chow-liu":
        return components * (2 * num_genes + 2)
    if architecture == "latent-class-residual-tree":
        return components * (num_genes + 1) + 2 * num_genes
    raise ValueError(f"Unknown architecture {architecture!r}.")


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unavailable"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _physical_memory_bytes() -> int | None:
    """Read installed RAM without making psutil a project dependency."""

    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _synchronize(device: torch.device) -> None:
    """Make device timing measure submitted GPU/MPS work rather than queue time."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def report_device_info(device: torch.device) -> None:
    """Print CPU/RAM plus accelerator capacity used by the calibration estimate."""

    cpu_description = platform.processor() or platform.machine() or "unknown CPU"
    print("System/device information")
    print(f"  CPU: {cpu_description}; logical cores: {os.cpu_count() or 'unavailable'}")
    print(f"  System memory: {_format_bytes(_physical_memory_bytes())}")
    print(f"  Selected Torch device: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        print(f"  GPU: {properties.name}")
        print(f"  GPU memory: {_format_bytes(total)} total; {_format_bytes(free)} currently free")
        print(f"  CUDA capability: {properties.major}.{properties.minor}")
    elif device.type == "mps":
        print("  GPU: Apple Metal (MPS; unified memory with the system RAM shown above)")
    else:
        print("  GPU: not selected")


def _one_epoch_calibration(
    train_data: np.ndarray,
    *,
    architecture: str,
    components: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    """Time model setup and one disposable optimizer epoch without saving state."""

    torch.manual_seed(seed)
    setup_started = time.perf_counter()
    model = build_pc(architecture, train_data, components, device)
    x = torch.as_tensor(train_data, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    _synchronize(device)
    setup_seconds = time.perf_counter() - setup_started

    epoch_started = time.perf_counter()
    model.train()
    permutation = torch.randperm(len(x), device=device)
    for start in range(0, len(x), batch_size):
        batch = x[permutation[start:start + batch_size]]
        loss = -model.log_prob(batch).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _synchronize(device)
    epoch_seconds = time.perf_counter() - epoch_started

    # The calibration PC is intentionally discarded: --info never writes a
    # checkpoint or retains a trained model in process memory.
    del optimizer, model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return setup_seconds, epoch_seconds


def print_training_estimate(
    data: GeneData,
    folds: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    components: int,
) -> None:
    """Calibrate one unsaved epoch and extrapolate the requested full run."""

    all_indices = np.arange(len(data.values))
    calibration_held_out = folds[0]
    calibration_train = data.values[
        np.setdiff1d(all_indices, calibration_held_out, assume_unique=False)
    ]
    parameter_count = architecture_parameter_count(args.architecture, components, data.shape[1])
    resident_data_bytes = len(calibration_train) * data.shape[1] * 4
    optimizer_lower_bound = parameter_count * 4 * 4  # parameter, grad, Adam m/v
    print("\nTraining estimate (calibration only; no checkpoint will be created)")
    print(f"  Data: {data.shape[0]:,} cells x {data.shape[1]:,} named genes")
    print(f"  Architecture: {args.architecture}; {components} root components; {parameter_count:,} trainable parameters")
    print(f"  Largest fold resident input: about {_format_bytes(resident_data_bytes)}")
    print(f"  Optimizer/model lower bound: about {_format_bytes(optimizer_lower_bound)}")
    print(f"  Timed calibration: one epoch on fold 1 ({len(calibration_train):,} training cells)")

    setup_seconds, epoch_seconds = _one_epoch_calibration(
        calibration_train,
        architecture=args.architecture,
        components=components,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    total_training_cell_epochs = args.epochs * sum(
        data.shape[0] - len(held_out) for held_out in folds
    )
    calibrated_cell_epochs = len(calibration_train)
    estimated_seconds = (
        len(folds) * setup_seconds
        + epoch_seconds * total_training_cell_epochs / calibrated_cell_epochs
    )
    low_seconds, high_seconds = estimated_seconds * 0.7, estimated_seconds * 1.5
    print(f"  Calibration setup: {setup_seconds:.2f} s; one epoch: {epoch_seconds:.2f} s")
    print(f"  Requested work: {len(folds)} folds x {args.epochs} epochs = {total_training_cell_epochs:,} cell-epochs")
    print(f"  Estimated wall time: {estimated_seconds / 60:.1f} minutes")
    print(f"  Planning range: {low_seconds / 60:.1f}–{high_seconds / 60:.1f} minutes")
    print("  The range reflects accelerator warm-up, competing load, and fold-to-fold variation.")


def fit_pc(
    train_data: np.ndarray,
    *,
    architecture: str,
    components: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    parents: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[torch.nn.Module, list[float]]:
    """Fit one selected normalized PC using minibatch maximum likelihood."""

    torch.manual_seed(seed)
    x = torch.as_tensor(train_data, dtype=torch.float32, device=device)
    model = build_pc(architecture, train_data, components, device, parents=parents)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x), device=device)
        total_nll = 0.0
        for start in range(0, len(x), batch_size):
            batch = x[permutation[start:start + batch_size]]
            loss = -model.log_prob(batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_nll += loss.detach().item() * len(batch)
        mean_nll = total_nll / len(x)
        history.append(mean_nll)
        if verbose:
            print(f"    epoch {epoch:03d}/{epochs}: mean NLL {mean_nll:.6f}")

    model.eval()
    return model, history


def train_cross_fitted_models(
    data: GeneData,
    *,
    output: Path,
    architecture: str,
    components: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    n_folds: int,
    seed: int,
    device: torch.device,
    info: bool = False,
) -> None:
    """Fit and checkpoint cross-fitted PCs for any validated gene selection."""

    folds = make_folds(len(data.values), n_folds, seed)
    components = effective_components(architecture, components)
    print(
        f"device: {device}; cells: {data.shape[0]:,}; named genes: {data.shape[1]:,}; "
        f"architecture: {architecture}; root components: {components}"
    )

    if info:
        args = argparse.Namespace(
            architecture=architecture, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay, seed=seed,
        )
        report_device_info(device)
        print_training_estimate(data, folds, args, device, components)
        return

    state_dicts: list[dict[str, torch.Tensor]] = []
    histories: list[list[float]] = []
    all_indices = np.arange(len(data.values))
    started = time.perf_counter()

    for fold, held_out in enumerate(folds):
        train_indices = np.setdiff1d(all_indices, held_out, assume_unique=False)
        print(f"\nFold {fold + 1}/{len(folds)}: {len(train_indices):,} training / {len(held_out):,} held out")
        model, history = fit_pc(
            data.values[train_indices],
            architecture=architecture,
            components=components,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + fold,
            device=device,
        )
        state_dicts.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        histories.append(history)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 4,
            "model_type": "cross_fitted_pc",
            "gene_names": list(data.gene_names),
            "data_shape": tuple(data.shape),
            "data_fingerprint": data.fingerprint,
            "fold_indices": [torch.as_tensor(fold, dtype=torch.int64) for fold in folds],
            "state_dicts": state_dicts,
            "model": {"architecture": architecture, "components": components},
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "device": str(device),
                "elapsed_seconds": time.perf_counter() - started,
                "nll_by_fold": histories,
            },
        },
        output,
    )
    print(f"\nSaved {len(folds)} cross-fitted PCs to {output}")


def train_cross_fitted_pcs(args: argparse.Namespace) -> None:
    """CLI adapter for the all-gene workflow."""

    data = load_gene_data(args.data, max_samples=args.max_samples, max_genes=args.max_genes)
    train_cross_fitted_models(
        data,
        output=Path(args.output), architecture=args.architecture,
        components=args.components, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        n_folds=args.n_folds, seed=args.seed, device=choose_device(args.device),
        info=args.info,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--output", type=Path, default=Path("output/cross_fitted_pcs_999_genes.pt"))
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default=DEFAULT_ARCHITECTURE,
        help="factorized is a product PC; latent-class is a root sum of product PCs.",
    )
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Development/testing only.")
    parser.add_argument("--max-genes", type=int, default=None, help="Development/testing only.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Report device/memory details and estimate runtime from one disposable epoch; do not save a checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_cross_fitted_pcs(parse_args())
