# 999-gene interaction analysis

This guide explains the scalable analysis that considers all 999 genes. It can calculate every pairwise interaction, but it cannot practically test every possible three-gene or larger set. Higher-order results therefore come from a clearly labelled screening workflow and should be treated as exploratory.

## At a glance

The workflow has two different scopes:

- **Pairs:** calculate all 498,501 unordered gene pairs exactly under the fitted PC, then perform cross-fitted TMLE inference where the data support it.
- **Orders 3 and above:** use the pairwise results to select a manageable set of candidate groups, then evaluate those candidates exactly. This is not an exhaustive all-gene search and it is not confirmatory inference.

An interaction measures association among binary gene-expression states. It does not by itself show direct regulation, a biological mechanism, or causation.

## Data and saved files

The source CSV contains 11,950 cells and 1,000 numeric columns. The unnamed first column is constant metadata and is removed. All 999 remaining named columns are kept in their CSV order and must be binary (0/1). The current analysis uses every cell and every gene; it applies no cell or gene filter.

Development options such as `--max-samples` and `--max-genes` change the data fingerprint. A checkpoint or cache from a development run therefore cannot silently be used with the full dataset.

The fingerprint hashes the ordered gene names and their binary values. The code requires an exact match before loading a checkpoint or reusing a pair cache. This prevents a changed input file or reordered columns from producing an apparently valid but incorrect analysis.

| File | What it contains |
| --- | --- |
| `output/pc_architecture_sweep_999_genes.csv` | Results for all architecture-selection candidates. |
| `output/pc_architecture_selected_config_999_genes.json` | Settings selected by validation likelihood. |
| `output/cross_fitted_pcs_999_genes.pt` | Two trained fold-specific PCs and run provenance. |
| `output/cross_fitted_pcs_999_genes_pair_cache.pt` | Exact PC pair marginals and held-out pair counts. |
| `output/cross_fitted_pcs_999_genes_pairwise_tmle.pt` | Results for all `C(999,2) = 498,501` pairs. |
| `plots/*pairwise_tmle_{overview,forest,network}.png` | Pairwise summary figures. |
| `*_order{r}_candidates.pt`, `*_order{r}_tmle.pt` | Screened higher-order candidates and their results. |

Run the usual workflow from the repository root:

~~~bash
python sweep_pc_architectures.py --analysis 999
python train_cross_fitted_pcs.py --analysis 999
python cross_fitted_interactions.py --analysis 999 --interaction-order 2 --rebuild-cache
python cross_fitted_interactions.py --analysis 999 --interaction-order 3
~~~

## 1. Select the PC model

Architecture selection is separate from interaction inference. With `seed=0`, the 11,950 cells are randomly divided into 9,560 selection-training cells and 2,390 validation cells. Each candidate trains for 20 epochs on the first group and is ranked by mean validation negative log likelihood (NLL) on the second. Lower NLL is better.

The selection grid has 39 candidates:

- four architectures: factorized, latent-class, mixture-Chow--Liu, and latent-class-residual-tree;
- 4, 8, 16, or 32 components for non-factorized models (the factorized model has one);
- AdamW learning rates of 0.005, 0.01, and 0.02; and
- a batch size of 1,024 cells, weight decay of `1e-5`, and seed 0.

For each tree-based candidate, the sweep learns one smoothed Chow--Liu maximum-mutual-information tree from the selection-training cells only. It roots the tree at a high-entropy gene and reuses that topology for the sweep's tree candidates. The final cross-fitted PCs relearn a separate topology from their own training halves.

The saved winner was:

| Selected setting | Value |
| --- | --- |
| Architecture | latent-class-residual-tree |
| Components | 32 |
| AdamW learning rate | 0.01 |
| Weight decay | `1e-5` |
| Validation NLL | 241.895630 |
| Trainable parameters | 33,998 |

The selected PC has 32 mixture components. Each component has its own baseline Bernoulli logit for every gene. Tree-edge residual logits are shared across components:

~~~text
logit P(X_j=1 | X_parent(j)=u, Z=z)
    = base_logit[z,j] + residual_edge_logit[j,u]
~~~

In plainer terms, the components can represent broad cell-state patterns, while the shared tree residuals represent compact gene-to-gene dependence. The root uses its component-specific baseline, and softmax-normalized weights combine the components. The parameter count is `32 x (999 + 1) + 2 x 999 = 33,998`.

No interaction result, p-value, or q-value was used to choose this model.

## 2. Cross-fit the final models

The selection sweep does not describe the final checkpoint training in full. The saved checkpoint was trained as follows:

| Checkpoint setting | Recorded value |
| --- | --- |
| PC / components | latent-class residual tree / 32 |
| Cross-fitting folds | 2 |
| Training / held-out cells per fold | 5,975 / 5,975 |
| Epochs | 10 |
| Training minibatch | 1,024 cells |
| AdamW learning rate / weight decay | 0.01 / `1e-5` |
| Fold seeds | 0 and 1 |
| Device | CPU |
| Training time | 11.31 s model-fit/save timer; 12.16 s complete fresh training command |
| Final training NLL, folds 1 and 2 | 246.000771, 246.297290 |

Each epoch shuffles cells and AdamW minimizes the mean NLL. A minibatch always contains **cells**; it does not split the 999 genes into smaller analyses.

Cross-fitting means each cell is assessed by a model that did not train on it:

~~~text
All 11,950 cells -- seeded split --> F1 (5,975) + F2 (5,975)

PC_1 learns from F2  --> predicts probabilities and targets F1
PC_2 learns from F1  --> predicts probabilities and targets F2

Combine fold estimates by held-out sample size and pool EIF moments for uncertainty.
~~~

Together, the two held-out folds include every cell exactly once.

## 3. Calculate all pairs with one exact cache

There are 498,501 unordered pairs, but the workflow does not fit or separately query 498,501 PCs. With `--rebuild-cache`, it checks the data/checkpoint fingerprint, loads the two fold PCs, and creates a cache containing the following for each fold:

1. `p_one[i] = P(X_i=1)`, a length-999 vector;
2. `p11[i,j] = P(X_i=1, X_j=1)` for the upper triangle;
3. held-out `count_one[i]` and `count11[i,j]`; and
4. the held-out fold size.

For this tree-mixture PC, one-gene and two-gene probabilities have exact analytic solutions. The code propagates component means and covariances along the tree, averages them using the mixture weights, and builds one 999 x 999 pair-probability matrix per fold. It never enumerates the impossible `2^999` full joint state space.

Observed counts are also calculated in bulk. One-gene counts are column sums; all 11 counts come from `X.T @ X`, an optimized dense matrix multiplication. The remaining three cells for a pair follow from identities:

~~~text
p00 = 1 - p_i - p_j + p11     n00 = n - n_i - n_j + n11
p01 = p_j - p11               n01 = n_j - n11
p10 = p_i - p11               n10 = n_i - n11
p11 = p11                     n11 = n11
~~~

### Recorded fresh end-to-end runtime

The current saved run trained a new two-fold checkpoint from the full source CSV, then forced a new pair-cache build before all-pairs TMLE. No checkpoint, cache, or pairwise result from an earlier run was read. It used `minimum-cell-count=0`, `minimum-probability=1e-12`, 64 bisection iterations, and default plots.

| Stage | Time | What the timed stage contains |
| --- | ---: | --- |
| Fresh training command | 12.16 s | Complete external Python-command time, including startup/imports, input loading, both PC fits, and checkpoint saving. The checkpoint's internal model-fit/save timer was 11.31 s. |
| Build and save the pair cache | 1.24 s | Program stage timer within the fresh pairwise command: restore the newly trained PCs; generate exact one- and two-gene probability tables and held-out count tables; and write the cache `.pt` file. |
| All-pairs TMLE, final save, and plots | 1.30 s | Program stage timer within the fresh pairwise command: construct 498,501 four-cell tables; do two-fold TMLE, SE/p-value/BH calculation; write the final pairwise `.pt` result; print the report; and render the overview, forest, and network images. |
| Fresh pairwise command, complete | 3.50 s | Complete external Python-command time for cache construction, TMLE, final save, and plots, including startup/imports, input loading, validation, result reload, and wrapper overhead. Its internal interaction timer was 2.65 s. |
| Fresh sequential pipeline | 15.66 s | `12.16 + 3.50`: one new PC fit followed by one new cache build and the all-pairs analysis. |

The complete external command times are the `real` values from `/usr/bin/time -p`; the two substage rows are the program's own logged timers. Thus the cache-construction row is the recorded **data-generation and cache-saving** time, while the all-pairs row is the recorded **TMLE, final-result saving, and plotting** time. The log does not separately time each operation inside those rows, so it would be misleading to assign finer-grained values. The master log, `logs/end_to_end_fresh_999_gene_pairwise_20261019T093921+0100.txt`, includes the training and pairwise commands in sequence. A later run can reuse a matching cache, but this recorded run did not.

### Why this is faster than the legacy workflow

The retired 12-gene code repeatedly used a generic circuit-integration query for each target, binary pattern, and fold. It also scanned cell rows in Python for counts and TMLE diagnostics.

The scalable workflow instead uses a calculation specific to pairwise marginals in this model, matrix arithmetic for observed counts, and vectorized operations across 498,501 four-cell tables. This is an algorithmic speed-up, not evidence that hundreds of thousands of PCs were fitted. Results are not expected to reproduce the retired implementation exactly, because the PC class, tree learning, and saved checkpoint differ.

## 4. Pairwise interaction, TMLE, and support

For a pair, the target is the log odds ratio:

~~~text
theta(P) = log[p00 p11 / (p01 p10)]
~~~

Zero means no interaction on this binary odds-ratio scale. Positive and negative values mean positive and negative association. This is not a causal estimate and does not establish a direct biological mechanism.

### What TMLE does

For each cross-fitted fold, TMLE starts with the PC's four-cell table `p_0` and makes a targeted update:

~~~text
p_epsilon(a) = p_0(a) + epsilon s(a),    s = (+1, -1, -1, +1)
~~~

`epsilon` must keep all four probabilities strictly positive. Its allowable interval is:

~~~text
max(-p00,-p11) < epsilon < min(p01,p10)
~~~

After shifting the endpoints slightly inward, the code runs 64 vectorized bisection steps to solve:

~~~text
sum_a P_n(a) s(a) / p_epsilon(a) = 0
~~~

`P_n` is the observed distribution in that held-out fold. This is the efficient-influence-function (EIF) score equation: TMLE chooses the update whose empirical EIF mean is zero. The EIF is **not** minimized.

At the updated table `p_star`, the centered EIF is:

~~~text
H(a) = s(a)/p_star(a) - E_pstar[s(A)/p_star(A)]
~~~

Fold log estimates are weighted by held-out sample size. Pooled EIF values give the SE, a 95% Wald confidence interval, and a normal-approximation p-value. Benjamini--Hochberg (BH) then adjusts p-values over the supported pair family.

### When the table reports uncertainty

The result table distinguishes two statuses:

- **Targeted:** The initial table is finite and strictly positive, and bisection has a valid root in both folds. It has a finite point estimate.
- **Supported:** The result is targeted and, in each fold, the smallest of the four observed counts is at least `--minimum-cell-count`, every initial model probability is at least `--minimum-probability`, and the table is finite.

Only supported pairs receive an SE, CI, p-value, and q-value. A finite targeted pair that fails support is descriptive only. Support is an unregularized positivity and stability requirement; it is not a score of biological importance.

> **What “supported unregularized” means here.** Inference uses the raw held-out four-cell table. It does not add a pseudocount, continuity correction, probability floor, ridge penalty, or variance regularization to repair an empty or rare cell. The thresholds are gates: they decide whether inference is shown, but they do not modify a count or clip a probability.
>
> In the current rerun, `minimum-cell-count=0`. The held-out count rule therefore does not exclude a pair with an empty 00, 01, 10, or 11 cell; it still applies independently to F1 and F2 rather than pooling counts across all 11,950 cells. In every fold, the PC must also assign all four cells a probability of at least `1e-12`; lower values are excluded, not raised to that threshold.
>
> “Targeted” is weaker than “supported.” A finite, strictly positive model table can have a valid TMLE solution even if one held-out cell is zero or otherwise below the threshold. Such a pair deliberately receives no SE, CI, p-value, or q-value.

The most recent table used `minimum-cell-count=0` and `minimum-probability=1e-12`. Of 498,501 pairs, 389,779 have finite targeted estimates and all 389,779 are supported. The remaining 108,722 pairs are not targeted because TMLE did not yield a finite point estimate in both folds; none is excluded by the held-out count or `1e-12` model-probability threshold.

Lowering the minimum count increases coverage, but it permits inference based on smaller cells. That is a scientific design choice, not a speed optimization.

### What the BH q-value in the overview plot means

BH is applied **only** to the 389,779 supported pairs in this run; targeted-but-unsupported pairs have no p-value or q-value and are not counted.  Let `m = 389,779`, let `p_h` be the two-sided Wald p-value for pair `h` under its null hypothesis of zero log interaction, and write the ordered p-values as

~~~text
p_(1) <= p_(2) <= ... <= p_(m).
~~~

For a chosen false-discovery-rate (FDR) level `alpha`, the BH procedure finds the largest rank `k` for which

~~~text
p_(k) <= k * alpha / m,
~~~

then calls ranks `1, ..., k` discoveries.  The reported value for the pair at rank `i` is the corresponding monotone BH-adjusted p-value:

~~~text
q_(i) = min_{j=i,...,m} min(1, m * p_(j) / j).
~~~

The value is then returned to its original pair.  Thus `q_h <= alpha` means exactly that pair `h` belongs to the BH discovery set if BH is run at level `alpha`; equivalently, `q_h` is the smallest BH FDR level at which it would be selected.  The name “q-value” here means this BH-adjusted p-value, not a posterior probability or a local false-discovery rate.

In the overview plot, pair `h` has vertical coordinate

~~~text
y_h = -log10(q_h),
~~~

with values above 50 visually capped at 50.  Higher points have smaller BH-adjusted p-values: `q=0.05`, `0.01`, and `0.001` map to about `1.301`, `2`, and `3`, respectively.  The horizontal line is therefore at `-log10(0.05) = 1.301`; points on or above it meet the BH `q <= 0.05` criterion.  The x-coordinate and colour describe the estimated TMLE log interaction and its sign, respectively.  They do not change the q-value, which comes from a two-sided p-value.

BH controls the FDR of the *set* selected at a fixed cutoff, where

~~~text
FDP = V / max(R, 1),       FDR = E[FDP],
~~~

`R` is the number of BH discoveries and `V` is the (unobserved) number of those discoveries whose null hypotheses are true.  Under valid null p-values and the usual BH dependence conditions (independence, or positive regression dependence on the true nulls), selecting every pair with `q <= 0.05` gives `FDR <= 0.05`.  It does **not** say that five percent of a particular subset, one plotted point, or one named edge is false, and it does not establish causation.  The pair tests share cells and genes, so they are dependent; the standard BH guarantee is not a guarantee under arbitrary dependence.  A confirmatory analysis needing protection under arbitrary dependence would need a different prespecified correction, such as Benjamini--Yekutieli.

References: Benjamini and Hochberg, “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B* 57(1), 289--300 (1995), [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x); Benjamini and Yekutieli, “The Control of the False Discovery Rate in Multiple Testing under Dependency,” *Annals of Statistics* 29(4), 1165--1188 (2001), [doi:10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998).

The default display screen additionally requires `|TMLE log interaction| >= 0.25`; 132,177 pairs pass both cuts in the current run.

## 5. Read the pairwise table and figures

The pairwise table keeps every result: PC plug-in and TMLE log effects, SE, CIs, p-values, q-values, support flags, and the smallest count/probability. Exponentiate a pair's log effect to get its odds ratio. Consider its CI, q-value, raw support, and external validation before drawing a biological conclusion.

The three figures are intentionally readable summaries rather than complete graphs.  They begin with pairs that are supported and have finite TMLE estimates and confidence intervals.  With the default settings, their display set is

~~~text
D = {h : supported, finite estimate/CI, q_h <= 0.05,
          and |TMLE log interaction_h| >= 0.25}.
~~~

The two cuts have different jobs: `q <= 0.05` is the multiple-testing evidence screen, whereas `|log interaction| >= 0.25` is an effect-size screen (odds ratio at least `exp(0.25) ≈ 1.28`, or at most `exp(-0.25) ≈ 0.78`).  If no pair satisfies **both** cuts, the forest and network fall back to all supported finite pairs, so a figure is still available.  A fallback figure is therefore a ranking of supported pairs, not a set that passed both default cuts.  The overview does not use this screen: it always includes every supported pair.

- **Overview:** Includes all supported pairs. The x-axis is the TMLE log interaction; the y-axis is `-log10(BH q-value)`, capped at 50 only for display. Red means positive and blue means negative. Dashed vertical lines mark +/-0.25; the horizontal line at 1.301 marks `q=0.05`. The BH family is all supported pairs, not only pairs passing the effect-size display cut.
- **Forest:** Separates `D` into negative estimates (`theta < 0`) and non-negative estimates (`theta >= 0`). Within each sign it orders by the estimate itself—most negative first on the left and most positive first on the right—and takes `min(25, number available)`. The two panels are selected independently, not balanced to contain 25 each. Thus fewer than 25 points on a side means fewer than 25 display-set pairs have that sign (or, after the no-hit fallback, fewer than 25 supported finite pairs have that sign). It does not mean that a 25th pair failed an extra confidence-interval or significance test. A sign with no eligible pair is labelled “None selected.” Points are TMLE log effects, bars are 95% CIs, and the black line marks zero. The default 25-point limit can be changed with `--forest-top-n`.
- **Network:** Starts with the same `D` (or the fallback set), ranks candidate pairs by decreasing `|TMLE log interaction|`, and processes them greedily. It adds a pair only while fewer than 25 edges have been kept **and** both of its genes currently have degree below 4; it then increments both degrees. A pair can therefore be absent despite passing the two display cuts because a gene in the pair already has four stronger retained neighbours. Fewer than 25 edges can remain if candidates run out or degree caps block the rest. These are sparse readability rules, not a global graph optimization or extra statistical tests. Red/blue marks sign, edge width shows relative absolute effect, and node size shows retained degree.

The all-gene network may contain two or more **disjoint components**.  Only genes incident to a retained edge are drawn, and the code does not require retained edges to form a connected graph.  Separate components arise when high-ranking retained pairs do not share genes, when potential bridging pairs fail the q-value or effect-size cuts, or when a bridge is skipped after one of its endpoints reaches the degree cap (or the 25-edge limit is reached).  The plotting code deliberately lays disconnected components apart so their labels and edges remain readable.  Their separation is not evidence that the gene groups are biologically independent, nor evidence that no association exists between them; it is a consequence of screening and sparsification.

## 6. Screened higher-order interactions

For `--interaction-order r >= 3`, the code first runs or reuses the all-pair stage. It then chooses a manageable candidate set:

1. keep pairs that are supported and finite, have `q <= 0.05`, and have `|log interaction| >= 0.25`;
2. for each gene, retain its 20 neighbours with the largest absolute pair effect;
3. by default, keep an edge only if both genes selected each other (`edge-policy=mutual`); `edge-policy=union` keeps an edge if either endpoint selected it;
4. enumerate `r`-cliques in the retained graph, stopping with an error above `max-candidates=100,000`; and
5. save each candidate's smallest absolute constituent-pair effect and largest constituent-pair q-value.

For every selected candidate, the PC then calculates all `2^r` binary-pattern probabilities exactly by batched tree message passing. The held-out patterns are counted with binary coding and `bincount`. The same TMLE update, support checks, pooled EIF variance, p-value, and BH calculation are then applied.

The higher-order target is:

~~~text
theta_r(P) = sum_{a in {0,1}^r} (-1)^(r-sum(a)) log P(X_S=a)
~~~

This is not a complete all-gene high-order search. The same pairwise data select candidates and are later used to estimate them. The saved higher-order result therefore carries an explicit exploratory warning. Do not present its q-values as confirmatory FDR evidence; use an independent confirmation split or a selection-aware design.

## 7. Why exhaustive high-order scans do not scale

The number of possible gene sets grows very quickly:

| Order r | All-gene set count `C(999,r)` |
| ---: | ---: |
| 2 | 498,501 |
| 3 | 165,668,499 |
| 4 | 41,251,456,251 |
| 5 | 8,209,039,793,949 |

Each order-`r` set also has `2^r` binary pattern cells. A general tree query takes work roughly proportional to `batch size x 2^r x 999 x 32` per fold, plus count construction. TMLE then performs 64 bisection updates across those `2^r` cells. Larger batches reduce loop overhead, but they use more memory and do not reduce the number of sets.

- The exact analytic pair cache makes a complete order-2 analysis practical on CPU. Its special pairwise identities do not exist for arbitrary larger sets.
- An exhaustive order-3 scan is not practical in this implementation on a CPU or a typical GPU. It starts with 166 million sets before accounting for eight patterns, two folds, PC queries, counts, TMLE, or result storage. A multi-field results table alone would need many tens of gigabytes.
- Orders 4 and 5 are much larger still. A normal GPU cannot remove the combinatorial `C(999,r)` growth or the `2^r` pattern growth.
- A CUDA GPU can speed model fitting and a **screened, bounded** set of higher-order candidates. With enough memory, it can also allow a larger candidate batch. It cannot make exhaustive high-order inference feasible.
- On Apple MPS, pairwise and higher-order TMLE run on CPU because the implementation needs reliable float64 arithmetic and MPS float64 support is limited.

The practical approach is to calculate all pairs exactly, screen using predeclared q-value, effect-size, and degree limits, cap the candidate set, inspect support carefully, and independently confirm selected higher-order findings.
