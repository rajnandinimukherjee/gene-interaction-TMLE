# 999-gene interaction analysis

This guide documents the scalable all-gene workflow. It explains why all 498,501 pairwise interactions can be evaluated quickly, why that pairwise shortcut does not make an exhaustive high-order scan feasible, and how screened higher-order results must be interpreted.

## Scope, data, and artifacts

The source CSV contains 11,950 cells and 1,000 numeric columns. The unnamed first column is constant metadata and is removed; all 999 named genes remain in CSV order and must be binary (0/1). The current analysis has no cell or gene filtering: it uses all 11,950 x 999 values. Development options such as --max-samples and --max-genes alter the fingerprint and cannot be mixed with the saved checkpoint/cache.

The fingerprint hashes both the ordered names and data values. Checkpoint loading and pair-cache reuse require an exact match, preventing silent analysis of a changed file or reordered genes.

| Artifact | Meaning |
| --- | --- |
| output/pc_architecture_sweep_999_genes.csv | All architecture-selection candidates. |
| output/pc_architecture_selected_config_999_genes.json | Likelihood-selected settings. |
| output/cross_fitted_pcs_999_genes.pt | Two trained fold PCs and provenance. |
| output/cross_fitted_pcs_999_genes_pair_cache.pt | Exact pairwise PC marginals and held-out pair counts. |
| output/cross_fitted_pcs_999_genes_pairwise_tmle.pt | All C(999,2)=498,501 pairwise results. |
| plots/*pairwise_tmle_{overview,forest,network}.png | Pairwise summary figures. |
| *_order{r}_candidates.pt and *_order{r}_tmle.pt | Screened higher-order candidates and results. |

Usual commands, from the repository root:

~~~bash
python sweep_pc_architectures.py --analysis 999
python train_cross_fitted_pcs.py --analysis 999
python cross_fitted_interactions.py --analysis 999 --interaction-order 2 --rebuild-cache
python cross_fitted_interactions.py --analysis 999 --interaction-order 3
~~~

## 1. Architecture selection and model

Selection is independent of interaction inference. With seed=0, a random permutation makes a 9,560-cell sweep-training set and a 2,390-cell validation set. Candidate PCs are ranked only by validation mean negative log likelihood (NLL; lower is better) after 20 epochs.

The selection grid contains 39 candidates:

- factorized, latent-class, mixture-chow-liu, and latent-class-residual-tree architectures;
- 4, 8, 16, and 32 components for non-factorized architectures; factorized has one;
- AdamW learning rates 0.005, 0.01, and 0.02;
- batch size 1,024 cells, weight decay 1e-5, and seed 0.

For tree candidates, the sweep derives one smoothed Chow--Liu maximum-mutual-information tree from sweep-training rows only, roots it at a high-entropy gene, and reuses it across tree candidates. The final cross-fitted fits relearn their own topology from their own training halves.

The saved winner is:

| Selected quantity | Value |
| --- | --- |
| Architecture | latent-class-residual-tree |
| Components | 32 |
| AdamW learning rate | 0.01 |
| Weight decay | 1e-5 |
| Validation NLL | 241.895630 |
| Trainable parameters | 33,998 |

The selected PC has 32 mixture components. Each component has a base Bernoulli logit for every gene; every learned tree edge has two residual logits that are shared across components:

~~~text
logit P(X_j=1 | X_parent(j)=u, Z=z)
    = base_logit[z,j] + residual_edge_logit[j,u].
~~~

The root uses its component base logit and component weights are softmax-normalized. This gives component-specific broad cell-state effects plus compact, shared tree residual dependencies. The parameter count is 32 x (999 + 1) + 2 x 999 = 33,998.

## 2. Actual checkpoint training and cross-fitting

The sweep chooses the architecture, component count, learning rate, and weight decay; it is not the complete checkpoint-training record. The saved checkpoint was actually fitted as follows:

| Checkpoint setting | Recorded value |
| --- | --- |
| PC / components | latent-class residual tree / 32 |
| Cross-fitting folds | 2 |
| Training / held-out cells per fold | 5,975 / 5,975 |
| Epochs | 10 |
| Training minibatch | 1,024 cells |
| AdamW learning rate / weight decay | 0.01 / 1e-5 |
| Fold seeds | 0 and 1 |
| Device | CPU |
| Training time | 11.78 s |
| Final training NLL, folds 1 and 2 | 246.000771, 246.297290 |

Each epoch shuffles cells and AdamW minimizes mean negative log likelihood. The training batch is a number of **cells**, never a partition of the 999 genes.

~~~text
All 11,950 cells -- seeded permutation --> F1 (5,975) + F2 (5,975)

             PC_1: learn topology and fit on F2
F1 counts  <---- predict P_1(X_S=a) for F1 targets

             PC_2: learn topology and fit on F1
F2 counts  <---- predict P_2(X_S=a) for F2 targets

              fold-specific TMLE / EIF moments
                           |
                           v
        held-out-size-weighted log estimate and pooled EIF variance
~~~

Every cell is targeted with a PC that was not fitted on it, and every cell contributes once across the two held-out folds.

## 3. All-pair cache: exact and fast

There are 498,501 unordered pairs, but the workflow does not fit or query a PC 498,501 times. With --rebuild-cache, it validates the data/checkpoint fingerprint, restores the two fold PCs, and writes one cache containing for each fold:

1. p_one[i] = P(X_i=1), a length-999 vector;
2. p11[i,j] = P(X_i=1, X_j=1), retained for the upper triangle;
3. held-out count_one[i] and count11[i,j]; and
4. held-out fold size.

For this tree-mixture PC, one- and two-gene probabilities are exact analytic marginals. Component means and covariances are propagated along the tree, averaged with mixture weights, and assembled into one 999 x 999 pair-marginal matrix per fold. This never enumerates the 2^999 full joint state space.

Held-out counts are bulk calculations: one-gene counts are column sums and all pairwise 11 counts are X.T @ X, a dense optimized matrix multiply. Upper-triangle extraction gives the cache. The other table entries are identities:

~~~text
p00 = 1 - p_i - p_j + p11     n00 = n - n_i - n_j + n11
p01 = p_j - p11               n01 = n_j - n11
p10 = p_i - p11               n10 = n_i - n11
p11 = p11                     n11 = n11.
~~~

The current run really rebuilt this cache: --rebuild-cache bypasses reuse, the log records “Building pair cache,” processes both 5,975-cell held-out folds, and saves the cache. That run's cache stage took 1.16 seconds; the all-pairs TMLE stage took 1.26 seconds; total interaction-analysis time was 2.53 seconds. Those numbers exclude PC fitting (11.78 seconds) and Python process/import startup.

### Why this is so much faster than legacy code

The retired 12-gene code repeatedly invoked a generic circuit-integration query for every target, outcome, and fold, then scanned rows in Python for counts and TMLE diagnostics. The scalable workflow uses a structure-specific all-pairs marginal calculation, optimized matrix arithmetic for counts, and vectorized tensor operations for 498,501 four-cell tables.

This is an algorithmic speed-up, not evidence that 498,501 PCs were fitted. It also does not mean the new estimates should reproduce legacy results: the PC class, topology learning, and saved checkpoint differ.

## 4. Pairwise TMLE, EIF, and support

For a pair, the target is the log odds ratio:

~~~text
theta(P) = log[p00 p11 / (p01 p10)].
~~~

Positive/negative values mean positive/negative association on this binary odds-ratio scale; zero is the no-interaction reference. This is not a causal estimate or proof of a direct biological mechanism.

For each cross-fitted fold, the PC's initial table p_0 is fluctuated along:

~~~text
p_epsilon(a) = p_0(a) + epsilon s(a),    s = (+1, -1, -1, +1).
~~~

The strict positivity interval is max(-p00,-p11) < epsilon < min(p01,p10). After moving its endpoints slightly inward, 64 vectorized bisection steps solve:

~~~text
sum_a P_n(a) s(a) / p_epsilon(a) = 0.
~~~

This is the EIF score/estimating equation. **The EIF is not minimized.** TMLE finds the epsilon whose empirical EIF mean is zero. At p_star, the centred EIF

~~~text
H(a) = s(a)/p_star(a) - E_pstar[s(A)/p_star(A)]
~~~

gives held-out sums and squared sums. Fold log estimates are weighted by held-out sample size; pooled EIF moments provide the SE, 95% Wald CI, and normal-approximation p-value. Benjamini--Hochberg (BH) adjusts p-values among supported pairs.

The result table distinguishes:

- **targeted:** the initial table is finite and strictly positive and bisection has a valid root in every fold, yielding a finite point estimate;
- **supported:** targeted and, in every fold, the smallest of the four held-out counts is at least minimum-cell-count, every initial model probability is at least minimum-probability, and the table is finite.

Only supported pairs receive SEs, CIs, p-values, and q-values. A finite targeted estimate that fails support is descriptive only. Thus TMLE support is an unregularized positivity/stability condition, not a measure of biological relevance.

> **Plain-language meaning in this project.** The log's “supported
> unregularized targets” are targets for which the code can make inference from
> the raw held-out four-cell table without adding a pseudocount, continuity
> correction, probability floor, ridge penalty, or variance regularization. The
> support threshold is a gate; it does not change a count, smooth a table, or
> clip a probability before estimating the effect.
>
> In the current rerun, minimum-cell-count=1 has a very literal per-fold
> meaning: for a pair to be supported, 00, 01, 10, and 11 must each occur at
> least once in F1 **and** at least once in F2. Counts are not pooled across
> the 11,950 cells for that decision. In each fold, the fitted PC must also
> assign all four cells at least 1e-12 probability; values below the threshold
> are excluded, not raised to 1e-12.
>
> “Targeted” is weaker than “supported.” A pair is targeted if its finite,
> strictly positive model table admits a bisection root for the raw TMLE score.
> It may still have a zero or otherwise under-threshold held-out cell and is
> then denied an SE, CI, p-value, and q-value. In the current result, 389,779
> pairs are targeted but only 389,725 are supported: 54 finite point estimates
> therefore remain intentionally non-inferential. The 108,776 unsupported
> pairs all fail the held-out count rule in at least one fold; none fails the
> 1e-12 model-probability rule.

The most recent table was deliberately rerun with minimum-cell-count=1 and minimum-probability=1e-12. It has 498,501 pairs, 389,779 finite targeted estimates, and 389,725 supported estimates. Lowering the threshold increased coverage but permits inference based on smaller cells; it is a scientific design choice, not a speed optimization.

A BH q-value is a false-discovery-rate-adjusted p-value within this supported pair family. A threshold q <= 0.05 controls the expected false-discovery proportion under standard BH assumptions; it is neither a posterior probability that an edge is false nor a causal guarantee. The default screen also requires |TMLE log interaction| >= 0.25. In the current run, 132,169 pairs pass both cuts.

## 5. Reading the pairwise results and plots

The pairwise table retains every result, including the plug-in and TMLE log effects, SE, CIs, p/q-values, support flags, and the minimum count/probability. Exponentiating a pair effect gives its odds ratio. Strong values should still be read with their CI, support, counts, q-value, and external validation in mind.

The three figures are deliberately sparse/readable summaries:

- **Overview:** all supported pairs. The x-axis is TMLE log interaction; the y-axis is -log10(BH q-value), capped at 50 only for display. Red is positive and blue negative. Vertical dashed lines are +/-0.25; the horizontal dashed line is q=0.05.
- **Forest:** starts from pairs passing both display cuts (or all supported pairs if none pass), then shows the 25 most negative and 25 most positive. Points are TMLE log effects, bars are 95% CIs, and zero is the black reference line.
- **Network:** starts from the same display set, ranks by absolute effect, and retains at most 25 edges while capping each gene at degree 4. Red/blue gives sign; edge width is relative absolute effect; node size is retained degree. It is not the full screened graph.

## 6. Screened higher-order interactions

For --interaction-order r >= 3, the workflow first runs or reuses the pairwise stage, then proposes a tractable candidate set:

1. retain pairwise targets that are supported and finite, with q <= 0.05 and |log interaction| >= 0.25;
2. for each gene, keep its strongest absolute-effect max-degree=20 neighbours;
3. by default (edge-policy=mutual), keep an edge only if both endpoints selected each other; edge-policy=union keeps either endpoint's choice;
4. enumerate r-cliques in the retained graph, stopping with an error above max-candidates=100,000;
5. store every candidate's minimum absolute constituent pair effect and largest constituent-pair q-value.

Each candidate is then evaluated exactly under the PC for all 2^r binary configurations by batched tree message passing. Held-out tables use binary coding and bincount; the same least-favourable TMLE path, support checks, pooled EIF variance, p-value, and BH calculation are applied.

The high-order estimand is:

~~~text
theta_r(P) = sum_{a in {0,1}^r} (-1)^(r-sum(a)) log P(X_S=a).
~~~

This is not a complete all-gene high-order scan. Candidate selection reuses the same pairwise data later used for high-order estimation. The saved high-order result therefore has an explicit exploratory warning. That two-stage pipeline should not be presented as confirmatory FDR inference: use an independent confirmation split or a selection-aware design.

## 7. Scaling and CPU/GPU feasibility

The full set count alone grows explosively:

| Order r | Full all-gene set count C(999,r) |
| ---: | ---: |
| 2 | 498,501 |
| 3 | 165,668,499 |
| 4 | 41,251,456,251 |
| 5 | 8,209,039,793,949 |

Every order-r target has 2^r table cells. A general tree query has work roughly proportional to batch-size x 2^r x 999 x 32 per fold, plus held-out count construction; TMLE then makes 64 bisection updates across those 2^r cells. Larger batches reduce loop overhead but consume more memory and do not reduce the number of sets.

Therefore:

- The analytic all-pairs cache makes full r=2 inference practical on CPU. It uses pairwise identities unavailable for arbitrary gene sets.
- A full r=3 scan is not practical in this implementation on either CPU or a normal GPU: it has 166 million sets before eight configurations, two folds, PC queries, counts, TMLE, and result storage. A multi-field results table alone would require many tens of gigabytes.
- Orders 4 and 5 are categorically larger. No ordinary GPU removes the C(999,r) factor or 2^r growth.
- A CUDA GPU can materially speed fitting and a **screened, bounded** batch of high-order candidates, and may permit a larger candidate batch if it has enough memory. It cannot make exhaustive high-order inference feasible.
- On Apple MPS, pairwise and higher-order TMLE deliberately run on CPU because the implementation requires reliable float64 arithmetic and MPS float64 support is limited.

The practical strategy is: calculate all pairs exactly; screen with predeclared q/effect/degree limits; cap candidates; inspect support; and treat selected high-order findings as exploratory until independent confirmation.

