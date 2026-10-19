"""Train all 999 named genes using shared scalable-PC components."""

from pc_core.train_cross_fitted_pcs import parse_args, train_cross_fitted_pcs


if __name__ == "__main__":
    train_cross_fitted_pcs(parse_args())
