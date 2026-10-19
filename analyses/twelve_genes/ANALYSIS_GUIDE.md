# 12-gene interaction analysis

This guide explains the current analysis of a fixed panel of 12 genes. It uses the shared probabilistic-circuit (PC) models and batched TMLE code. Files in `legacy_code/` are kept only for historical reference; they did not create the current results.

## At a glance

The workflow does four things:

1. selects a PC architecture using validation likelihood only;
2. trains two PCs on opposite halves of the cells, so each cell is evaluated by a model that did not see it during training;
3. estimates the association for every requested gene set; and
4. reports uncertainty only when every needed cell has enough raw data and the fitted model assigns it a usable probability.

An interaction here is an association in binary gene-expression states. It is not evidence that one gene directly regulates another, and it is not a causal effect.

## Data, panel, and saved files

The panel has a fixed order. That order is part of the analysis and must not change.

~~~text
Spc24  Gmnn  Prim2  Gins2  Cks2  Cdca8
Pbk    Ube2c Ccna2  Apitd1 Hells Mcm6
~~~

The source CSV contains 11,950 cells and 1,000 numeric columns. Its unnamed first column is constant metadata, so the analysis removes it. The remaining 999 named columns must contain binary values (0 or 1). The panel loader takes the 12 names above, in the displayed order. It does not filter cells or genes: all 11,950 cells are used.

`--max-samples` is only for development. It changes the data fingerprint, so a reduced-data run cannot accidentally reuse a full-data checkpoint.

Every checkpoint and result file stores a SHA-256 fingerprint of the ordered gene names and binary values. Loading fails if the input file has changed, its columns have been reordered, or a different panel is supplied.

| File | What it contains |
| --- | --- |
| `output/pc_architecture_sweep_12_genes.csv` | Results for every model-selection candidate. |
| `output/pc_architecture_selected_config_12_genes.json` | The settings selected by validation likelihood. |
| `output/cross_fitted_pcs_12_genes.pt` | The two fitted fold-specific PCs and run provenance. |
| `output/{2,3,4,5,6,7}_pt_interactions_12_genes.pt` | Machine-readable interaction results for one order. |
| `plots/{order}_pt_interactions_12_genes.png` | A labelled comparison plot for every requested gene set. |
| `plots/*_overview.png`, `*_forest.png`, `*_network.png`, `*_hypergraph.png` | Smaller summary figures. |
| `logs/*.txt` | Plain-text run details, support checks, and results. |

Run the usual workflow from the repository root:

~~~bash
python sweep_pc_architectures.py --analysis 12
python train_cross_fitted_pcs.py --analysis 12
python cross_fitted_interactions.py --analysis 12 --interaction-order 2
~~~

## 1. Choose the PC model

Model selection is separate from interaction estimation. With `seed=0`, the 11,950 cells are randomly split into 9,560 selection-training cells and 2,390 validation cells (80/20). Each candidate learns from the first group and is ranked by its mean validation negative log likelihood (NLL) on the second. Lower NLL is better.

The 20-epoch selection run evaluates 39 candidates:

- four architectures: factorized, latent-class, mixture-Chow--Liu, and latent-class-residual-tree;
- 4, 8, 16, or 32 components for each non-factorized architecture (the factorized baseline has one component);
- AdamW learning rates of 0.005, 0.01, or 0.02; and
- a batch size of 1,024 cells, weight decay of `1e-5`, and seed 0.

For tree-based candidates, a Chow--Liu tree is learned once using only the selection-training cells. It uses smoothed pairwise mutual information, chooses the maximum-spanning tree, and roots it at the gene with highest entropy. The validation cells play no part in learning this tree.

The saved winner was:

| Selected setting | Value |
| --- | --- |
| Architecture | mixture-Chow--Liu |
| Mixture components | 32 |
| AdamW learning rate | 0.02 |
| Weight decay | `1e-5` |
| Validation NLL | 4.081627 |
| Trainable parameters | 832 |

No interaction estimate, p-value, or q-value was used to select these settings.

### What the selected PC represents

The selected PC is a weighted mixture of 32 binary tree distributions:

~~~text
P(x) = sum_z pi_z P(x_root | z) product_{j != root} P(x_j | x_parent(j), z)
~~~

In plain language, each mixture component represents a broad expression pattern. Within a component, the genes follow the same tree layout, but their probabilities may differ. The component weights `pi_z` are normalized with softmax. Each component has its own root probability and two conditional probabilities for every non-root gene (one for each parent state). Starting values come from smoothed empirical probabilities with small noise, keeping probabilities away from exactly zero and one.

## 2. Train two fold-specific models

The selection sweep chooses the architecture and a few hyperparameters. It is not the final training record. The saved checkpoint was trained with these settings:

| Checkpoint setting | Recorded value |
| --- | --- |
| PC / components | mixture-Chow--Liu / 32 |
| Cross-fitting folds | 2 |
| Training / held-out cells per fold | 5,975 / 5,975 |
| Epochs | 20 |
| Training minibatch | 128 cells |
| AdamW learning rate / weight decay | 0.02 / `1e-5` |
| Fold seeds | 0 and 1 |
| Device | CPU |
| Training time | 1.19 s model-fit/save timer; 2.13 s complete fresh training command |
| Final training NLL, folds 1 and 2 | 4.024783, 4.044821 |

During every epoch, the training cells are shuffled and AdamW minimizes the average NLL. A training batch always means a number of **cells**; it never means a group of the 12 genes.

Cross-fitting works as follows:

~~~text
All 11,950 cells -- seeded split --> F1 (5,975) + F2 (5,975)

Train PC_1 on F2  --> estimate and target results for F1
Train PC_2 on F1  --> estimate and target results for F2

Combine the two fold results, weighting each by its held-out sample size.
~~~

This protects against evaluating a cell with a model that memorized it: every cell is evaluated by a PC trained on the other fold. The two held-out folds cover every cell exactly once.

### Recorded fresh end-to-end runtimes

The following are complete wall-clock timings from a fresh full-panel run on 19 October 2026. It trained the two 20-epoch fold PCs from the source CSV before any interaction work; no checkpoint, interaction result, or cache from an earlier run was read. Each interaction command then read the checkpoint produced moments earlier in that same master run. The run used `minimum-cell-count=1`, `minimum-probability=1e-12`, `batch-size=512`, 64 TMLE bisection iterations, default plots, and no reduced data.

| Order | Full-panel targets | Fresh PC training (once) | Full interaction command | Fresh fit + one order |
| ---: | ---: | ---: | ---: | ---: |
| 2-point | 66 | 2.13 s | 1.60 s | 3.73 s |
| 3-point | 220 | 2.13 s | 2.01 s | 4.14 s |
| 4-point | 495 | 2.13 s | 2.95 s | 5.08 s |
| 5-point | 792 | 2.13 s | 4.47 s | 6.60 s |
| 6-point | 924 | 2.13 s | 5.28 s | 7.41 s |
| 7-point | 792 | 2.13 s | 4.82 s | 6.95 s |

The table reports the `real` time from `/usr/bin/time -p` around each whole Python command, so it includes interpreter startup and imports, loading the CSV and freshly written checkpoint, querying every gene set, exact PC probabilities and held-out counts, TMLE plus SE/p-value/BH calculation, result/report saving, and rendering the tall full-panel figure plus overview, forest, and network/hypergraph figures. The per-order program logs also retain their smaller internal analysis timer; the table deliberately uses the complete command timer rather than assigning unsupported substep timings. The master log, `logs/end_to_end_fresh_12_gene_orders_2_to_7_20261019T093419+0100.txt`, contains training and all six analyses in sequence. Training once and then running all six orders took 23.26 s by summing the recorded command times (`2.13 + 1.60 + 2.01 + 2.95 + 4.47 + 5.28 + 4.82`), rather than fitting six separate sets of PCs.

## 3. Calculate model probabilities and observed counts

For interaction order `r`, the default run examines every `C(12,r)` gene set:

| Order | Sets evaluated |
| ---: | ---: |
| 2 (pairs) | 66 |
| 3 (triples) | 220 |
| 4 | 495 |
| 5 | 792 |
| 6 | 924 |
| 7 | 792 |

Use `--genes` when you want one named set instead of the full panel.

For each gene set `S`, the PC calculates the probability of every binary pattern `a` in `{0,1}^r`. For example, a pair has four patterns: 00, 01, 10, and 11. These are exact marginal probabilities under the fitted PC, not probabilities estimated by simulating cells. The tree PC uses batched message passing: it fixes the selected gene states, sums over the remaining tree nodes, then averages across mixture components.

The interaction command's `--batch-size` is a number of **gene sets** processed together. It is not a count of genes or cells.

For the held-out cells, the code counts every pattern for every requested set in one batch. It encodes each pattern as a binary integer and uses `torch.bincount`, rather than looping over cells in Python.

## 4. Interaction estimate, TMLE, and uncertainty

For a set `S` of order `r`, the reported interaction is:

~~~text
theta(P) = sum_a s(a) log P(X_S=a)
s(a) = (-1)^(r - sum(a))
~~~

For a pair, this simplifies to:

~~~text
log[p00 p11 / (p01 p10)]
~~~

That is the log odds ratio. Zero means no interaction on this scale; positive and negative values indicate positive and negative association. For sets larger than pairs, exponentiating the result gives an alternating cross-product ratio. It is not a correlation and it is not a causal effect.

### What TMLE changes

For each fold and target, TMLE starts from the PC probability table `p_0` and makes a small, targeted update:

~~~text
p_epsilon(a) = p_0(a) + epsilon s(a)
~~~

`epsilon` is restricted so that every updated probability remains strictly positive. The code moves the endpoints of that valid interval slightly inward, then uses 64 vectorized bisection steps to solve:

~~~text
sum_a P_n(a) s(a) / p_epsilon(a) = 0
~~~

Here, `P_n` is the observed distribution in the held-out fold. This equation says that the empirical efficient-influence-function (EIF) mean is zero. The EIF is **not** minimized.

At the updated table `p_star`, the centered EIF is:

~~~text
H(a) = s(a)/p_star(a) - E_pstar[s(A)/p_star(A)]
~~~

The two fold estimates are combined using their held-out sample sizes. The pooled EIF values give the standard error (SE):

~~~text
SE = sqrt([sum H^2 - (sum H)^2/N] / [(N-1) N])
~~~

The result table then gives a 95% Wald interval (`theta +/- 1.959964 SE`), a two-sided normal p-value for the null value zero, and a Benjamini--Hochberg (BH) q-value.

### When a result is eligible for inference

The table uses two deliberately different statuses:

- **Targeted:** The initial model table is finite and strictly positive, and the bisection equation has a valid solution in both folds. This produces a finite point estimate.
- **Supported for unregularized inference:** The target is targeted and, in each fold, every one of its `2^r` observed patterns has at least `--minimum-cell-count` cells, every initial model probability is at least `--minimum-probability`, and the table is finite.

Only supported results receive an SE, confidence interval, p-value, and q-value. A targeted but unsupported result can still be useful as a descriptive point estimate, but it is intentionally not used for unregularized statistical inference.

For the freshly saved pairwise panel result, `minimum-cell-count=1` and `minimum-probability=1e-12`; all 66 pairs are supported. At larger interaction orders, sparse pattern counts—not computation—usually become the main scientific limit.

> **What “supported unregularized TMLE” means here.** The raw held-out counts are used exactly as observed. The code does not add pseudocounts, a continuity correction, a probability floor, a ridge penalty, or variance regularization to repair a rare or empty pattern. For a pair in this saved run, each of 00, 01, 10, and 11 must appear at least once in F1 and at least once in F2. The fitted PC must also assign each pattern probability at least `1e-12` in both folds. These thresholds only decide whether inference is reported; they do not change counts or probabilities.
>
> A result may be targeted without being supported. For example, the score equation may have a finite solution even though one pattern is too rare for the prespecified inference rule. Such a result has no SE, CI, p-value, or q-value. If no strictly positive bisection solution exists, it is not targeted either.

### What the BH q-value in an overview plot means

BH is applied **only** to the supported targets in the result being plotted; unsupported targets have no p-value or q-value and are not counted.  Let `m` be that number of supported targets, let `p_h` be the two-sided Wald p-value for target `h` under its null hypothesis of zero log interaction, and write the ordered p-values as

~~~text
p_(1) <= p_(2) <= ... <= p_(m).
~~~

For a chosen false-discovery-rate (FDR) level `alpha`, the BH procedure finds the largest rank `k` for which

~~~text
p_(k) <= k * alpha / m,
~~~

then calls ranks `1, ..., k` discoveries.  The reported value for the target at rank `i` is the corresponding monotone BH-adjusted p-value:

~~~text
q_(i) = min_{j=i,...,m} min(1, m * p_(j) / j).
~~~

The value is then returned to its original target.  Thus `q_h <= alpha` means exactly that target `h` belongs to the BH discovery set if BH is run at level `alpha`; equivalently, `q_h` is the smallest BH FDR level at which it would be selected.  The name “q-value” here means this BH-adjusted p-value, not a posterior probability or a local false-discovery rate.

In the overview plot, a point for target `h` has vertical coordinate

~~~text
y_h = -log10(q_h),
~~~

with values above 50 visually capped at 50.  Higher points have smaller BH-adjusted p-values: `q=0.05`, `0.01`, and `0.001` map to about `1.301`, `2`, and `3`, respectively.  The horizontal line is therefore at `-log10(0.05) = 1.301`; points on or above it meet the BH `q <= 0.05` criterion.  The x-coordinate and colour describe the estimated TMLE log interaction and its sign, respectively.  They do not change the q-value, which comes from a two-sided p-value.

BH controls the FDR of the *set* selected at a fixed cutoff, where

~~~text
FDP = V / max(R, 1),       FDR = E[FDP],
~~~

`R` is the number of BH discoveries and `V` is the (unobserved) number of those discoveries whose null hypotheses are true.  Under valid null p-values and the usual BH dependence conditions (independence, or positive regression dependence on the true nulls), selecting every target with `q <= 0.05` gives `FDR <= 0.05`.  It does **not** say that five percent of a particular subset, one plotted point, or one named gene set is false, and it does not establish causation.  The targets share cells and genes, so their tests are dependent; the standard BH guarantee is not a guarantee under arbitrary dependence.  A confirmatory analysis needing protection under arbitrary dependence would need a different prespecified correction, such as Benjamini--Yekutieli.

References: Benjamini and Hochberg, “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B* 57(1), 289--300 (1995), [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x); Benjamini and Yekutieli, “The Control of the False Discovery Rate in Multiple Testing under Dependency,” *Annals of Statistics* 29(4), 1165--1188 (2001), [doi:10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998).

`empirical_log_interaction` is included only as a full-data descriptive comparison. It uses all 11,950 cells, is `NaN` if any observed pattern is empty, is not cross-fitted, and is not used for TMLE inference.

## 5. Read the result table and plots

The `.pt` result table includes the gene set, PC plug-in effect, TMLE effect, SE, confidence interval, p-value, q-value, smallest observed count, smallest model probability, and targeted/supported flags. For pairs, exponentiate the log effect to obtain the odds ratio. Do not treat a screened result as proof of direct regulation or causation.

### Full-panel comparison: `{order}_pt_interactions_12_genes.png`

This tall figure intentionally gives every requested set its own row.

- Rows are ordered from the largest finite empirical log interaction to the smallest; sets with undefined empirical values appear last.
- A circle shows the PC plug-in estimate.
- An orange x with a bar shows a supported TMLE estimate and its 95% CI. A finite, unsupported TMLE estimate is an orange x without a CI.
- A green star shows the full-data empirical comparison when it is defined.
- An orange line connects the plug-in and finite TMLE estimates.
- The dashed vertical line at zero marks no log interaction.

### Overview, forest, network, and hypergraph figures

These smaller figures begin with results that are supported and have finite TMLE estimates and confidence intervals.  With the default settings, their display set is

~~~text
D = {h : supported, finite estimate/CI, q_h <= 0.05,
          and |TMLE log interaction_h| >= 0.25}.
~~~

The two cuts serve different purposes: `q <= 0.05` is the multiple-testing evidence screen, whereas `|log interaction| >= 0.25` is an effect-size screen (an odds ratio at least `exp(0.25) ≈ 1.28`, or at most `exp(-0.25) ≈ 0.78`, for pairs).  If `D` is empty, the forest, pair network, and hypergraph fall back to all supported finite results so that the figure is not blank.  Consequently, a fallback figure is a ranking of supported results, not a set that passed both default cuts.  This is a display and ranking rule; the full result table remains the complete record.  The overview is different: it always shows every supported result.

- **Overview:** Shows every supported effect against `-log10(BH q-value)`, capped at 50 only for drawing. Red means positive and blue means negative. Dashed vertical lines mark +/-0.25; the horizontal line at 1.301 marks `q=0.05`. The BH family is all supported targets in that plotted result, not only points passing the effect-size display cut.
- **Forest:** Separates `D` into negative estimates (`theta < 0`) and non-negative estimates (`theta >= 0`). Within each sign it orders by the estimate itself—most negative first on the left and most positive first on the right—and takes `min(25, number available)`. Thus the two panels are selected independently, not balanced to contain 25 each. A panel has fewer than 25 points when fewer than 25 display-set results have that sign (or, after the no-hit fallback, fewer than 25 supported finite results have that sign); it says nothing about a 25th result being removed for a confidence interval or an additional significance test. A panel with no eligible results says “None selected.” Points are TMLE log effects, bars are 95% CIs, and the black line is zero. The default 25-point limit can be changed with `--forest-top-n`.
- **Pair network:** Starts with the same `D` (or the fallback set), ranks candidate pairs by decreasing `|TMLE log interaction|`, and processes them greedily. It adds a pair only while fewer than 25 edges have been kept **and** both genes currently have degree below 4; then it increments both degrees. A strong pair can therefore be omitted because an endpoint has already reached degree 4, and fewer than 25 edges can remain if there are too few candidates or the degree cap prevents further additions. This is a sparse readability rule, not a global optimization or a second statistical test. Red/blue marks sign, edge width shows relative absolute effect, and node size shows retained degree. It is intentionally not a graph of every significant pair.
- **Hypergraph (orders above 2):** Draws at most the eight strongest screened sets as outline polygons. Red/blue marks sign and line width shows relative magnitude. Polygon overlap has no extra statistical meaning.

## 6. Why the current workflow is faster

The speed improvement changes the implementation, not the question being estimated. The retired code processed one set at a time, queried a generic circuit integrator separately for each binary pattern, and repeatedly scanned held-out cells in Python for counts, diagnostics, and TMLE.

The current code batches gene sets and patterns, uses exact tree message passing, builds contingency tables with tensors, and solves the TMLE updates in batches. It still calculates exact requested marginals for the fitted PC and uses the same universal least-favourable TMLE update. It is not expected to match the retired Cirkit checkpoint numerically, because the PC class, tree learning, and fitted checkpoint are different.

The code can enumerate all panel sets for the listed orders, but the `2^r` pattern table gets sparse as `r` grows. That loss of support is the practical limit for unregularized high-order inference.
