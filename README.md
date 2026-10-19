# Cross-fitted gene-interaction analysis

## Install from a fresh clone

This repository includes the input dataset and saved analysis artifacts.  You
need Python 3.10 through 3.13 and an internet connection the first time so `pip`
can download the tested PyTorch, NumPy, and Matplotlib versions.  From the repository root, the
cross-platform bootstrap command creates an isolated `.venv` and installs this
checkout together with all runtime dependencies:

```bash
python scripts/bootstrap.py
```

Activate the environment after it completes:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

The installed commands can then be run from the repository root:

```bash
train-cross-fitted-pcs --analysis 12 --epochs 100
sweep-pc-architectures --analysis 12
cross-fitted-interactions --analysis 12 --interaction-order 3
```

If you prefer to manage environments yourself, `python -m pip install .`
installs the same package and commands; `python -m pip install -r
requirements.txt` is an equivalent requirements-file workflow.  The original
`python train_cross_fitted_pcs.py`, `python sweep_pc_architectures.py`, and
`python cross_fitted_interactions.py` entry points remain supported.

By default `--device auto` uses CUDA, Apple Metal (MPS), or CPU in that order.
CPU works without additional configuration.  For a particular NVIDIA/CUDA
installation, use the PyTorch wheel selector if its required CUDA build differs
from the default wheel that `pip` selects.

There are two public commands.  Select the gene set with `--analysis`; the
implementation and checkpoint format are shared internally by `pc_core`.

```bash
# A device/memory report plus a disposable one-epoch timing calibration.
# No checkpoint is written by --info.
python train_cross_fitted_pcs.py --analysis 999 --epochs 100 --device auto --info

# Train either study. Defaults come from that study's saved 20-epoch winner.
python train_cross_fitted_pcs.py --analysis 12 --epochs 100
python train_cross_fitted_pcs.py --analysis 999 --epochs 100 --device cuda

# Repeat the 20-epoch architecture-selection protocol for either study.
python sweep_pc_architectures.py --analysis 12
python sweep_pc_architectures.py --analysis 999 --batch-size 256

# The 12-gene workflow evaluates every selected-order panel combination and
# writes its result table and plot. --genes limits it to one set.
python cross_fitted_interactions.py --analysis 12 --interaction-order 3

# The 999-gene workflow cross-targets conditional outcome/propensity nuisance
# functions, runs directional pairwise TMLE, then screens BY q-values/effect
# sizes before evaluating order > 2.
python cross_fitted_interactions.py --analysis 999 --interaction-order 3 \
  --screen-max-q 0.05 --screen-min-abs-log-interaction 0.25
```

Artifacts are written to the selected study directory and always carry its
tag, including when an explicit output path is supplied: for example,
`cross_fitted_pcs_12_genes.pt`,
`3_pt_conditional_interactions_12_genes.pt`, or
`cross_fitted_pcs_999_genes_conditional_pairwise_tmle.pt`. On this branch, pair labels
mean `outcome <- exposure` and the estimand is a standardized conditional
log-odds interaction; see the study guides for its covariate definition.

Every root `cross_fitted_interactions.py` command also writes a new plaintext,
timestamped log in `analyses/twelve_genes/logs/` or
`analyses/nine_nine_nine_genes/logs/`.  Each filename includes the interaction
order and submit time, and existing logs are never replaced.  `--log PATH`
sets a base path but still appends those identifiers.  Logs record the
checkpoint's training settings, final NLLs and training duration; the analysis
settings; per-fold TMLE targeting diagnostics; a readable TMLE result table; and the total
analysis runtime.
For the 999-gene workflow the table is the 25 largest finite conditional-TMLE effects, with
the full numerical table retained in the `.pt` artifact.

Every interaction analysis also writes three figures beside the study results:

- `*_overview.png`: all finite conditional estimates, shown against BY q-value when
  inferential q-values are available;
- `*_forest.png`: the strongest positive and negative TMLE estimates with
  95% confidence intervals; and
- `*_network.png` for pairs, or `*_hypergraph.png` for panel interactions of
  order greater than two.

The network is deliberately sparse (by default at most 25 screen-passing
edges and degree at most 4; an n-way hypergraph is capped at eight polygons).
Change its display limits with
`--network-max-items` and `--network-max-degree`; use `--forest-top-n` for the
ranked panels or `--no-plots` to suppress the new figures.  Existing saved
tables can be plotted without rerunning TMLE:

```bash
python -m pc_core.interaction_plots \
  --results analyses/nine_nine_nine_genes/output/cross_fitted_pcs_999_genes_conditional_pairwise_tmle.pt
```

Files without `_conditional_` in their interaction-result names predate this
branch and contain the marginal estimand; use `marginal-estimand` to inspect
them. They are intentionally not inputs to this workflow.

The 20-epoch architecture sweeps selected different defaults for the studies:

- 12 genes: `mixture-chow-liu`, 32 components, learning rate `0.02`
  (validation NLL `4.081627`).
- 999 genes: `latent-class-residual-tree`, 32 components, learning rate `0.01`
  (validation NLL `241.895630`).

The tagged JSON selection files are read by the unified training command, and
the same values are also recorded in the study configuration fallbacks.

For a single higher-order gene set rather than screened candidates, use:

```bash
python cross_fitted_interactions.py --analysis 999 --interaction-order 3 \
  --genes Spc24,Gmnn,Prim2
```
