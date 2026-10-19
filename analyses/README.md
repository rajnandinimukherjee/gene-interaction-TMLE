# Study definitions

This directory holds the two study definitions and their outputs.  Use the two
repository-root commands, not per-study scripts:

```bash
python train_cross_fitted_pcs.py --analysis 12 [training options]
python train_cross_fitted_pcs.py --analysis 999 [training options]

python cross_fitted_interactions.py --analysis 12 --interaction-order 3
python cross_fitted_interactions.py --analysis 999 --interaction-order 3

python sweep_pc_architectures.py --analysis 12
python sweep_pc_architectures.py --analysis 999 --batch-size 256
```

Both commands use the internal `pc_core` package; no second implementation is
maintained for either gene set. The 12-gene command evaluates all panel
combinations (or one ordered `--genes` set), while the 999-gene command uses
the batched conditional-TMLE and higher-order screening pipeline.

Detailed workflow documentation is kept with each study:

- [12-gene analysis guide](twelve_genes/ANALYSIS_GUIDE.md)
- [999-gene analysis guide](nine_nine_nine_genes/ANALYSIS_GUIDE.md)

Their default filenames are study-tagged (`*_12_genes.*` or `*_999_genes.*`)
to prevent mixing checkpoints, figures, and TMLE tables between the
two analyses.

Each interaction run writes an all-result overview, a signed ranked forest
plot, and a sparse signed network.  For 12-gene interactions of order greater
than two, the last figure is a signed hypergraph, where each polygon denotes a
selected gene set.  The all-gene study generates these figures only for its
pairwise TMLE table.

Previous untagged run artifacts were cleared for a fresh start.  The
`twelve_genes/legacy_code` source remains as a non-executable reference for
the original study definition.
