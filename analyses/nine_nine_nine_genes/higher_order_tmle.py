"""Evaluate all-gene higher-order candidates using shared implementation."""

from pc_core.higher_order_tmle import parse_args, run_higher_order_tmle


if __name__ == "__main__":
    run_higher_order_tmle(parse_args())
