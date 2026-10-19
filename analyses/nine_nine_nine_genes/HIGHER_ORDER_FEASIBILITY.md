# Higher-order conditional-TMLE feasibility: 999 genes

## Summary

This branch uses the conditional-outcome estimand from the reference PC/TMLE analysis, not the former marginal contingency-table interaction. A target is an ordered tuple `(Y, A_1, ..., A_k)`: the first gene is the binary outcome, the later genes are a joint exposure, and the other genes are the conditioning variables `W`.

For every exposure configuration `a`, the held-out-fold PC supplies

```text
Q_a(W) = P(Y=1 | A=a, W)       g_a(W) = P(A=a | W)
m_a    = mean_i Q_a(W_i).
```

The reported effect is

```text
theta = sum_a (-1)^(k-sum(a)) logit(m_a).
```

Therefore a pair is a standardized conditional log odds ratio and a triplet is an alternating conditional log-odds interaction. The result is directional. The exhaustive pair scan follows the reference code's canonical ordering: for the pair of columns `(i, j)` with `i < j`, gene `i` is the outcome and gene `j` is the exposure. This ordering is a convention for the estimand, not evidence of a biological direction or a causal effect.

The full pair scan is computationally far more expensive than it was on the `marginal-estimand` branch. Higher orders still need screening. Do not interpret a fast marginal-cache run as a runtime estimate for this branch.

## Two-fold cross-targeting is retained

The trained PCs and folds have not been collapsed into a single in-sample analysis:

```text
PC fit on fold 2 -> Q, g, TMLE fluctuation on held-out fold 1
PC fit on fold 1 -> Q, g, TMLE fluctuation on held-out fold 2
```

Each held-out fold has its own nuisance predictions and logistic fluctuation. Fold effects are combined in proportion to held-out fold size; EIF sums and squared sums are pooled over all held-out cells. This is the same two-fold cross-targeting layout used by the previous analysis, with conditional nuisance functions in place of a marginal `2^r` cell table.

## What makes the conditional scan costly

The old pair cache was sufficient for a marginal pair because four model probabilities and four held-out counts determine its effect. It cannot recover `Q_a(W_i)` or `g_a(W_i)`: those quantities change with each cell's remaining 997 gene values. Consequently, there is no exact reuse of the old all-pair marginal cache for the conditional target.

For an order-`r` target, the implementation must evaluate two outcome completions for every one of `2^(r-1)` exposure patterns for every held-out cell, in both folds. The target count also grows combinatorially:

| Analysis | Target count | Practical advice |
| --- | ---: | --- |
| Canonical pairs | C(999, 2) = 498,501 | Calibrate first with `--max-genes`; budget a long batched PC-evaluation job. |
| Canonical triples | C(999, 3) = 165,668,499 | Do not scan exhaustively. |
| Four-gene tuples | C(999, 4) = 41,251,456,251 | Not a practical exhaustive analysis. |

The implementation avoids avoidable overhead: it restores each cross-fitted PC once, evaluates all binary completions for a target block together, chunks held-out cells, uses a vectorized bisection rather than one GLM object per stratum, and vectorizes EIF/CI/BY calculations. These are constant-factor speed-ups; they do not remove the cell-level conditional computation.

`--batch-size` controls the target block and `--nuisance-sample-batch-size` controls the held-out cell chunk. Smaller values reduce memory use at the cost of more launches/loops. The code also bounds blocks internally to avoid retaining an unbounded target-by-cell-by-pattern-by-gene tensor.

## Higher-order workflow

For orders three and above, the program first targets canonical pairs, then screens them using BY-adjusted q-values and absolute conditional effect size, caps graph degree, and evaluates the resulting candidate cliques. The first gene in each saved candidate is again the outcome. This makes a bounded analysis possible but does not turn results into confirmatory discoveries: the same data select candidates and estimate their higher-order effects. Use an independent confirmation split or a selection-aware design for a confirmatory claim.

## TMLE, uncertainty, and overlap

For pattern `a`, the held-out-fold fluctuation solves

```text
sum_i I(A_i=a) H_a(W_i) [Y_i - expit(logit(Q_a(W_i)) + epsilon_a H_a(W_i))] = 0

H_a(W) = (-1)^(k-sum(a)) / [m_a(1-m_a)g_a(W)].
```

The pooled efficient influence-function values give the source-compatible population-variance Wald standard error,

```text
SE = sqrt({P_n[D^2] - P_n[D]^2} / N),
```

and the saved 95% interval is `theta +/- 1.959964 SE` on the log-interaction scale. P-values receive Benjamini--Yekutieli (BY) correction.

Unlike the marginal branch, this procedure has no four-cell empirical-count support gate. Probabilities are clipped to `[1e-6, 1-1e-6]`, and a target is unavailable when an exposure stratum is absent or has complete outcome separation in a fold, so its logistic fluctuation is not finite. This does not guarantee usable conditional overlap. Inspect the recorded minimum raw `Q` and `g` values; a finite, clipped estimate with a very small propensity is still sensitive to model extrapolation.

## Hardware and a safe calibration sequence

The conditional-TMLE part uses float64. CUDA can accelerate the batched PC queries; MPS intentionally falls back to CPU for the TMLE computation because the required float64 operations are not reliable there. A CUDA device needs enough memory for the selected target and cell chunks, not the full scan.

Before a 999-gene run, train a matching reduced checkpoint and exercise the same options, for example:

```bash
python train_cross_fitted_pcs.py --analysis 999 --max-genes 12 \
  --output /private/tmp/cross_fitted_pcs_12_gene_smoke_999_genes.pt --epochs 1
python cross_fitted_interactions.py --analysis 999 --max-genes 12 \
  --model /private/tmp/cross_fitted_pcs_12_gene_smoke_999_genes.pt \
  --interaction-order 2 --no-plots
```

The reduced data selection and checkpoint must match exactly: checkpoint metadata deliberately rejects a full 999-gene checkpoint paired with `--max-genes`. After calibration, increase the gene count and tune the two batch controls before committing to the full all-pair job.

For the full statistical definition and result fields, see [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md). The previous marginal cache and its empirical four-cell support rule remain available unchanged on the `marginal-estimand` branch.
