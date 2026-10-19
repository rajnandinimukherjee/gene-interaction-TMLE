"""Fixed study definition for the curated 12-gene analysis."""

from pathlib import Path

# This is the exact ordered ``CONSIDERED_GENES`` panel from the original
# ``data_utils.py``.  Keeping it here makes the --analysis 12 selection
# reproducible independently of the order of columns in the input CSV.
GENE_NAMES = (
    "Spc24", "Gmnn", "Prim2", "Gins2", "Cks2", "Cdca8",
    "Pbk", "Ube2c", "Ccna2", "Apitd1", "Hells", "Mcm6",
)

DATA = Path("input/trainingData_11950Cells_1000Genes.csv")
OUTPUT_DIR = Path("analyses/twelve_genes/output")
PLOTS_DIR = Path("analyses/twelve_genes/plots")
LOG_DIR = Path("analyses/twelve_genes/logs")
ANALYSIS_TAG = "12_genes"
MODEL_PATH = OUTPUT_DIR / f"cross_fitted_pcs_{ANALYSIS_TAG}.pt"
PAIR_CACHE_PATH = OUTPUT_DIR / f"pair_cache_marginal_estimand_{ANALYSIS_TAG}.pt"
PAIRWISE_TMLE_PATH = OUTPUT_DIR / f"pairwise_marginal_estimand_tmle_{ANALYSIS_TAG}.pt"
ARCHITECTURE_SELECTION_PATH = OUTPUT_DIR / f"pc_architecture_selected_config_{ANALYSIS_TAG}.json"

# Winner of the 20-epoch 12-gene architecture sweep.  The matching saved
# selection file is read by the unified CLI and is retained for provenance.
ARCHITECTURE = "mixture-chow-liu"
COMPONENTS = 32
LEARNING_RATE = 0.02
WEIGHT_DECAY = 1e-5
