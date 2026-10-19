"""Run all-gene pairwise TMLE using shared implementation."""

from pc_core.pairwise_tmle import compute_pairwise_tmle, parse_args


if __name__ == "__main__":
    compute_pairwise_tmle(parse_args())
