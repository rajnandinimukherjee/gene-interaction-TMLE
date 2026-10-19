"""Paths and stable naming for the 999-named-gene study."""

from pathlib import Path


DATA = Path("input/trainingData_11950Cells_1000Genes.csv")
OUTPUT_DIR = Path("analyses/nine_nine_nine_genes/output")
PLOTS_DIR = Path("analyses/nine_nine_nine_genes/plots")
LOG_DIR = Path("analyses/nine_nine_nine_genes/logs")
ANALYSIS_TAG = "999_genes"
MODEL_PATH = OUTPUT_DIR / f"cross_fitted_pcs_{ANALYSIS_TAG}.pt"
ARCHITECTURE_SELECTION_PATH = OUTPUT_DIR / f"pc_architecture_selected_config_{ANALYSIS_TAG}.json"

# Winner of the 20-epoch 999-gene architecture sweep.  The matching saved
# selection file is read by the unified CLI and is retained for provenance.
ARCHITECTURE = "latent-class-residual-tree"
COMPONENTS = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-5
