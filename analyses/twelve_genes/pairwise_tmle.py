"""Run panel pairwise TMLE through the shared cached-TMLE implementation."""

from pathlib import Path

from pc_core.pairwise_tmle import compute_pairwise_tmle, parse_args

from .config import PAIR_CACHE_PATH, PAIRWISE_TMLE_PATH


if __name__ == "__main__":
    args = parse_args()
    # The reusable parser has all-gene defaults; retain them only for its
    # workflow and place panel results beside the panel checkpoint/cache.
    if args.cache == Path("output/cross_fitted_pair_cache_999_genes.pt"):
        args.cache = PAIR_CACHE_PATH
    if args.output == Path("output/cross_fitted_pairwise_tmle_999_genes.pt"):
        args.output = PAIRWISE_TMLE_PATH
    compute_pairwise_tmle(args)
