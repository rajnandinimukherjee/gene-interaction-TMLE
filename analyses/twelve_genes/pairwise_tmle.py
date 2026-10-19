"""Run panel conditional pairwise TMLE through the shared implementation."""

from pathlib import Path

from pc_core.pairwise_tmle import compute_pairwise_tmle, parse_args

from .config import DATA, MODEL_PATH, PAIRWISE_TMLE_PATH
from .data import load_panel


if __name__ == "__main__":
    args = parse_args()
    # The reusable parser has all-gene defaults; redirect its default model
    # and result path to the fixed panel study.
    if args.data == Path("input/trainingData_11950Cells_1000Genes.csv"):
        args.data = DATA
    if args.model == Path("output/cross_fitted_pcs_999_genes.pt"):
        args.model = MODEL_PATH
    if args.output == Path("output/cross_fitted_conditional_pairwise_tmle_999_genes.pt"):
        args.output = PAIRWISE_TMLE_PATH
    args.data_object = load_panel(args.data, max_samples=args.max_samples)
    compute_pairwise_tmle(args)
