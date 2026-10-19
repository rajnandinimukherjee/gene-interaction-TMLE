"""Screen all-gene higher-order candidates using shared implementation."""

from pc_core.screen_higher_order import parse_args, screen_higher_order


if __name__ == "__main__":
    screen_higher_order(parse_args())
