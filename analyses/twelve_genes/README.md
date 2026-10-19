# Curated 12-gene workflow

This analysis selects, in this exact order:

`Spc24, Gmnn, Prim2, Gins2, Cks2, Cdca8, Pbk, Ube2c, Ccna2, Apitd1, Hells, Mcm6`.

It uses the shared PC implementation and the winner of its dedicated 20-epoch
architecture sweep: `mixture-chow-liu`, 32 components, learning rate `0.02`.

Run from the repository root:

```bash
python train_cross_fitted_pcs.py --analysis 12
python sweep_pc_architectures.py --analysis 12
python -m analyses.twelve_genes.build_pair_cache
python -m analyses.twelve_genes.pairwise_tmle
```

To reproduce the former all-combinations plot for a chosen order, use:

```bash
python cross_fitted_interactions.py --analysis 12 \
  --model analyses/twelve_genes/output/cross_fitted_pcs_12_genes.pt \
  --interaction-order 3
```

It evaluates all `C(12, order)` sets, writes the log-interaction estimates to
`output/<order>_pt_interactions_12_genes.pt`, and retains the former
all-combinations comparison plot at `plots/<order>_pt_interactions_12_genes.png`.
That figure is a vertical, labelled row for every interaction (for example,
66 rows at order 2 and 220 at order 3).  A focused `--genes` query receives a
gene-set-specific default filename, so it cannot overwrite the canonical
all-combinations figure or result table.
It also writes `*_overview.png`, `*_forest.png`, and either `*_network.png`
(pairwise) or `*_hypergraph.png` (higher-order) in the same plots directory.

New files are written under `analyses/twelve_genes/output` and plots under
`analyses/twelve_genes/plots`.  Earlier untagged artifacts were cleared for a
fresh start; superseded Cirkit-only source remains in `legacy_code` as a
reference for the original panel definition.
