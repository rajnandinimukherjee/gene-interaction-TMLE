"""Sweep all-gene PC architectures using shared implementation."""

from pc_core.sweep_pc_architectures import parse_args, run_sweep


if __name__ == "__main__":
    run_sweep(parse_args())
