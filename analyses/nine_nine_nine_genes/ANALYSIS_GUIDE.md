# 999-gene conditional-estimand analysis

This branch replaces the prior marginal joint-cell interaction with the conditional-outcome PC/TMLE estimand from the referenced GitHub analysis. The original cached marginal workflow remains intact on `marginal-estimand`.

Interaction-result filenames on this branch include `_conditional_`. The
pre-existing files without that marker are retained historical marginal
artifacts, not conditional results or workflow inputs.

```bash
python train_cross_fitted_pcs.py --analysis 999

# Conditional pairwise analysis: canonical outcome is the first member of a pair.
python cross_fitted_interactions.py --analysis 999 --interaction-order 2

# Screen pairwise conditional results, then target selected ordered triples.
python cross_fitted_interactions.py --analysis 999 --interaction-order 3

# Development-scale validation: the reduced checkpoint must match the reduced data.
python train_cross_fitted_pcs.py --analysis 999 --max-genes 12 --epochs 1 \
  --output /private/tmp/cross_fitted_pcs_12_gene_smoke_999_genes.pt
python cross_fitted_interactions.py --analysis 999 --max-genes 12 \
  --model /private/tmp/cross_fitted_pcs_12_gene_smoke_999_genes.pt \
  --interaction-order 2 --no-plots
```

## Ordered conditional target

For an ordered set `(Y, A_1, ..., A_k)`, `Y` is the outcome, `A` is the binary treatment vector, and every gene outside the set is `W`. A held-out-fold PC provides

```text
Q_a(W) = P(Y=1 | A=a, W)
g_a(W) = P(A=a | W)
m_a    = E[Q_a(W)].
```

The target is

```text
theta = sum_a (-1)^(k-sum(a)) logit(m_a).
```

For a pair, it is the standardized conditional log odds ratio of the first gene under the two states of the second. For a triple, it is the alternating conditional log-odds interaction across 00, 01, 10, and 11 of its two exposures. The order is stored in the result table and is part of the meaning; a pair is displayed as `outcome <- exposure`.

This is not the old symmetric marginal `log[p00 p11/(p01 p10)]`, and it is not automatically causal. With all remaining genes used as `W`, it is an adjusted descriptive parameter. For a causal analysis, replace that covariate set with prespecified, defensible pre-exposure confounders rather than mechanically conditioning on every other gene.

## Cross-targeting, TMLE, EIF, and CI

The two fold protocol is retained exactly in structure:

```text
PC fitted on fold 2 -> nuisance predictions and TMLE fluctuation on fold 1
PC fitted on fold 1 -> nuisance predictions and TMLE fluctuation on fold 2
```

The fold estimates are weighted by held-out fold size. Their EIF sums and squared sums are pooled for uncertainty. This preserves the original workflow's cross-targeting protection while using the new conditional nuisance functions.

For each exposure pattern `a`, the fold-level adjustment is

```text
H_a(W) = s(a) / [m_a(1-m_a) g_a(W)]
Q_a*(W) = expit(logit(Q_a(W)) + epsilon_a H_a(W))
```

where `s(a)=(-1)^(k-sum(a))`. The scalar `epsilon_a` solves

```text
sum_i I(A_i=a) H_a(W_i) [Y_i-Q_a*(W_i)] = 0.
```

The source code solved one logistic GLM per pattern. Here that same score is solved by batched Torch bisection, eliminating per-target Python and statsmodels overhead.

The EIF is

```text
D(O) = sum_a s(a)/[m_a*(1-m_a*)] * {
         I(A=a)/g_a(W) [Y-Q_a*(W)] + Q_a*(W)-m_a*
       }.
```

The standard error uses the source pipeline's population variance convention over the pooled cross-targeted EIF values:

```text
SE = sqrt({P_n[D^2] - P_n[D]^2} / N).
```

The saved 95% CI is on the log-interaction scale, `theta* +/- 1.959964 SE`; exponentiate endpoints for the multiplicative scale. Two-sided normal p-values are adjusted with Benjamini--Yekutieli (BY), matching the source analysis.

The conditional nuisance probabilities are clipped to `[1e-6, 1-1e-6]`, as in the source. There is no empirical contingency-table support filter on this branch. A row is unavailable only if an exposure stratum is absent or has complete outcome separation in a fold, leaving no finite logistic fluctuation. The result table records minimum raw PC conditional outcome and propensity values; a finite CI should still be assessed for practical conditional-overlap problems.

## Scaling and retained speed-ups

The former all-pair cache was specific to marginal probabilities: `p_i`, `p_ij`, and four observed counts determine a pair's old target. It cannot determine `Q_a(W)` or `g_a(W)`, which vary cell by cell after conditioning on the remaining 997 genes. The conditional workflow therefore does not reuse the old pair-marginal cache.

It does retain and extend the useful computational ideas from the marginal implementation:

- two saved cross-fitted PCs are restored once and reused;
- all `(Y,A)` binary completions for a target batch are evaluated together;
- held-out cells are chunked so a full target-by-cell-by-pattern-by-gene tensor is never retained;
- conditional patterns, fluctuations, EIF moments, p-values, and BY adjustment are vectorized in Torch; and
- only a bounded target block and its two held-out folds reside in memory.

This target is intrinsically much more expensive than the old exact pair-marginal cache. It requires cell-level conditional evaluations for every pair, so an exhaustive 999-gene pair analysis should be budgeted and first calibrated with `--max-genes` and the intended device. `--batch-size` controls the target block; `--nuisance-sample-batch-size` controls the cell chunk. The code reduces either internally when needed to bound working memory.

## Higher orders

An all-gene scan beyond pairs remains combinatorial. For order three and above, the workflow first computes conditional pairwise results, screens using BY q-values and effect size, limits graph degree, and evaluates candidate cliques. For each retained tuple, its first gene is the outcome and the other genes form the joint exposure vector.

Because the same data choose pairwise candidates and estimate their higher-order conditional effects, those higher-order CIs and q-values are exploratory. Use an independent confirmation split or a selection-aware design for confirmatory claims.

## Result files and plots

Pairwise results retain `pair_i`/`pair_j` for screening but now mean `outcome`/`exposure`; the `estimand` field is `standardized_conditional_log_odds_interaction`. Higher-order results retain ordered `gene_sets`. All tables contain plug-in/TMLE effects, SEs, log-scale CIs, p-values, BY q-values, numerical-targeting flags, and minimum raw conditional nuisance probabilities.

The pair plot draws arrows from exposure to outcome. This is estimand direction only, not evidence of a regulatory arrow.
