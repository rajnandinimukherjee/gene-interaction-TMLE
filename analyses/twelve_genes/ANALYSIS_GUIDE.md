# 12-gene conditional-estimand analysis

Result filenames on this branch include `_conditional_`. Existing files
without that marker are historical marginal-estimand artifacts; they are not
used by the conditional workflow.

This branch replaces the former symmetric marginal cell-probability interaction with the conditional-outcome TMLE used by the referenced PC/TMLE analysis. The former analysis remains on the `marginal-estimand` branch.

The 12-gene workflow evaluates every ordered panel tuple of the requested size. Its first gene is the outcome; its remaining genes are the joint exposure vector. This order is part of the estimand.

```bash
python train_cross_fitted_pcs.py --analysis 12
python cross_fitted_interactions.py --analysis 12 --interaction-order 2
python cross_fitted_interactions.py --analysis 12 --interaction-order 3

# One explicitly ordered target: outcome first, then exposure(s).
python cross_fitted_interactions.py --analysis 12 --interaction-order 3 \
  --genes Spc24,Gmnn,Prim2
```

For a default panel combination, the genes are in panel order, so the first member of that combination is the outcome. An explicit `--genes` list preserves the specified outcome/exposure order.

## Fresh full-panel run and timings

On 19 October 2026, the complete 12-gene conditional workflow was run on all
11,950 cells with a fresh two-fold `mixture-chow-liu` checkpoint (32
components, 20 epochs, batch size 128, seed 0, CPU). The interaction commands
used `--batch-size 512`, `--nuisance-sample-batch-size 64`, 64 fluctuation
bisections, the default probability clip of `1e-6`, and the default full and
summary plots. No prior conditional result table or conditional interaction
cache was used.

The table reports complete `/usr/bin/time -p` wall-clock time (`real`) for
each Python command. It includes process startup, input/checkpoint loading,
conditional nuisance evaluation, targeting, result/log saving, and rendering
all four plots. The fresh-fit column adds the once-only training command; do
not sum it down the rows. `Finite targets` is the number with a finite
unregularized cross-targeted fluctuation (`tmle_targeted`), not a count of
statistically significant interactions.

| Order | Ordered panel targets | Finite targets | Fresh PC training (once) | Full interaction command | Fresh fit + one order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2-point | 66 | 66 | 1.97 s | 4.03 s | 6.00 s |
| 3-point | 220 | 220 | 1.97 s | 19.99 s | 21.96 s |
| 4-point | 495 | 494 | 1.97 s | 77.09 s | 79.06 s |
| 5-point | 792 | 619 | 1.97 s | 247.13 s | 249.10 s |
| 6-point | 924 | 64 | 1.97 s | 622.78 s | 624.75 s |
| 7-point | 792 | 0 | 1.97 s | 1,060.62 s | 1,062.59 s |

Training once and running every order took 2,033.61 s (33 m 53.61 s) in
total; the six interaction commands alone took 2,031.64 s (33 m 51.64 s).
The detailed, timestamped per-order logs are in `logs/` with the name pattern
`cross_fitted_interactions_12_genes_order_<order>_submitted_<timestamp>.txt`.

Higher-order availability is limited by missing treatment strata or complete
outcome separation in at least one held-out fold: this run has one unavailable
order-4 target, 173 unavailable order-5 targets, 860 unavailable order-6
targets, and no finite order-7 fluctuations. These rows remain in the saved
tables and plots with `NA` inference; they were not discarded or regularized.

## Estimand

For an ordered set `(Y, A_1, ..., A_k)`, let `A` denote the binary exposure vector and let `W` be every gene outside that set. The cross-fitted PC estimates

```text
Q_a(W) = P(Y=1 | A=a, W)
g_a(W) = P(A=a | W)
m_a    = E[Q_a(W)].
```

The reported log interaction is the alternating cross-product of the standardized outcome odds:

```text
theta = sum_a s(a) logit(m_a)
s(a) = (-1)^(k - sum(a)).
```

For a pair `(Y, A)`, this is

```text
theta = logit(m_1) - logit(m_0),
```

the standardized conditional log odds ratio. For `(Y, A, B)`, it is

```text
logit(m_00) - logit(m_01) - logit(m_10) + logit(m_11).
```

This differs from the old marginal target `log[p00 p11 / (p01 p10)]`. It is directional and depends on the covariate definition. In the current expression-only workflow, conditioning on all other genes makes this an adjusted descriptive parameter. It is not a causal gene-regulation estimate: other genes may be mediators, colliders, or descendants. A causal analysis should replace `W` with a prespecified set of defensible pre-exposure confounders.

## Two-fold cross-targeting

The earlier two-fold design is retained:

```text
fold 1 PC trains on fold 2  -> predicts and targets fold 1
fold 2 PC trains on fold 1  -> predicts and targets fold 2
```

Each fold gets its own conditional TMLE fluctuation. The final plug-in and TMLE log interactions are held-out-fold-size-weighted averages; EIF first and second moments are pooled across the two folds. Thus no PC is evaluated or targeted on a cell used to fit it.

For each held-out cell, the PC is evaluated under every `(Y, A)` completion. Normalizing the two `Y` completions gives `Q_a(W)`; summing over `Y` and normalizing over all `A` patterns gives `g_a(W)`. The implementation batches target tuples, exposure patterns, and cells, and solves all scalar fluctuations together in Torch instead of constructing a statsmodels GLM for each tuple/pattern.

## TMLE adjustment and score

Within a fold, let the initial standardized mean be `m_a = P_n Q_a(W)`. For each exposure pattern `a`, the clever covariate is

```text
H_a(W) = s(a) / [m_a (1-m_a) g_a(W)].
```

The outcome regression is updated along the logistic fluctuation

```text
Q_a*(W) = expit(logit(Q_a(W)) + epsilon_a H_a(W)).
```

The fold-specific coefficient `epsilon_a` solves the score equation

```text
sum_i I(A_i=a) H_a(W_i) [Y_i - Q_a*(W_i)] = 0.
```

The source implementation solves this with one intercept-free logistic GLM per pattern. This branch solves the identical monotone scalar scores by 64 batched bisection iterations. It then recalculates `m_a* = P_n Q_a*(W)` and `theta*` from the updated means.

As in the source code, PC conditional outcomes and propensities are clipped to `[1e-6, 1-1e-6]` before the fluctuation. This is numerical stabilization, not empirical support repair.

## EIF and confidence interval

The old marginal EIF was a function of one observed joint cell. The conditional EIF for the new target is

```text
D(O) = sum_a s(a) / [m_a*(1-m_a*)] * {
         I(A=a)/g_a(W) * [Y-Q_a*(W)] + Q_a*(W)-m_a*
       }.
```

The residual term is inverse-propensity weighted because only cells with observed `A=a` inform that pattern's outcome regression. The `Q_a*(W)-m_a*` term standardizes over the held-out covariate distribution. The logit derivative supplies `1/[m_a*(1-m_a*)]` and the alternating signs.

Across both cross-targeted folds, the code pools `sum D_i` and `sum D_i^2` and uses the source pipeline's population-variance convention:

```text
var_hat(D) = P_n[D^2] - (P_n[D])^2
SE(theta*) = sqrt(var_hat(D) / N).
```

It reports the Wald interval

```text
theta* +/- 1.959964 SE(theta*).
```

Exponentiating its endpoints gives the interval on the cross-product/odds-ratio scale. The two-sided normal p-value tests `theta=0`. Benjamini--Yekutieli, matching the source pipeline, adjusts those p-values across the target family.

## Positivity and numerical availability

Unlike the marginal branch, this workflow does not calculate empirical `2^r` pattern frequencies or use a four-cell support gate. It records the smallest raw PC `Q` and `g` values as diagnostics, then clips model values at the stated floor.

A result is unavailable only when a fold lacks an exposure pattern or has complete outcome separation within one observed exposure pattern, so the corresponding unregularized logistic fluctuation has no finite solution. Such rows receive `NaN` inference. A finite result is not evidence of adequate conditional positivity. Inspect the stored `minimum_conditional_propensity` field and perform domain-specific overlap checks before drawing strong conclusions.

## Saved outputs

The result table stores the ordered `gene_sets`, PC plug-in and TMLE log interactions, log-scale SE/CI, p- and BY q-values, targeting status, and minimum raw conditional outcome/propensity predictions. Pair labels use `outcome <- exposure`; higher-order labels use `outcome <- exposure_1,exposure_2,...`.

The overview and forest plot this standardized conditional log interaction. Pair networks use arrows from exposure to outcome to preserve the selected estimand direction; the arrows do not assert biological causality.
