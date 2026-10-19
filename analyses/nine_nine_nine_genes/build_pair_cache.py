"""Build the all-gene pair cache using shared implementation."""

from pc_core.build_pair_cache import build_pair_cache, parse_args


if __name__ == "__main__":
    build_pair_cache(parse_args())
