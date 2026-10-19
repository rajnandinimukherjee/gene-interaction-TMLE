# 12-gene interaction analysis

This guide documents the current 12-gene workflow. It is separate from the retired implementation in legacy_code/: current outputs use the shared pc_core models and batched TMLE code. The old files are historical reference, not the implementation that created the present artifacts.

## Scope, data, and artifacts

The fixed, ordered panel is:

~~~text
Spc24  Gmnn  Prim2  Gins2  Cks2  Cdca8
Pbk    Ube2c Ccna2  Apitd1 Hells Mcm6
~~~

The source CSV has 11,950 cells and 1,000 numeric columns. The unnamed first column is constant metadata and is removed; the 999 named columns must be binary. The panel loader selects the 12 names above in that order. There is no cell or gene filter in the saved analysis: all 11,950 cells are used. --max-samples is development-only and changes the data fingerprint, so it cannot silently be mixed with a full-data checkpoint.

Every checkpoint/artifact stores a SHA-256 fingerprint of the ordered names and binary data. A mismatch is rejected, protecting against changed data, a reordered CSV, or a substituted panel.

| Artifact | Meaning |
| --- | --- |
| output/pc_architecture_sweep_12_genes.csv | Candidate sweep results. |
| output/pc_architecture_selected_config_12_genes.json | Likelihood-selected architecture settings. |
| output/cross_fitted_pcs_12_genes.pt | Two fitted fold PCs and provenance. |
| output/{2,3,4,5}_pt_interactions_12_genes.pt | Machine-readable interaction tables. |
| plots/{order}_pt_interactions_12_genes.png | The labelled all-panel comparison plot. |
| plots/*_overview.png, *_forest.png, *_network.png or *_hypergraph.png | Scalable summary views. |
| logs/*.txt | Plaintext provenance, support, and result logs. |

Run the normal workflow from the repository root:

~~~bash
python sweep_pc_architectures.py --analysis 12
python train_cross_fitted_pcs.py --analysis 12
python cross_fitted_interactions.py --analysis 12 --interaction-order 2
~~~

## 1. Architecture and hyperparameter selection

Architecture selection is independent of the interaction analysis. With seed=0, a random permutation splits 11,950 cells into 9,560 sweep-training and 2,390 validation cells (80/20). Every candidate is trained on the former and ranked by mean validation negative log likelihood (NLL; lower is better) on the latter.

The 20-epoch selection protocol has 39 candidates:

- architectures: factorized, latent-class, mixture-chow-liu, and latent-class-residual-tree;
- component candidates 4, 8, 16, and 32 for every non-factorized architecture (the factorized baseline has one component);
- AdamW learning rates 0.005, 0.01, and 0.02;
- batch size 1,024 cells, weight decay 1e-5, and seed 0.

For tree candidates, a Chow--Liu topology is learned once from the sweep-training partition, never validation rows, then reused by those candidates. It is a smoothed (add-one) pairwise-mutual-information maximum-spanning tree, rooted at the highest-entropy gene.

The saved winner is:

| Selected quantity | Value |
| --- | --- |
| Architecture | mixture-chow-liu |
| Mixture components | 32 |
| AdamW learning rate | 0.02 |
| Weight decay | 1e-5 |
| Validation NLL | 4.081627 |
| Trainable parameters | 832 |

This is a likelihood-only selection; no interaction estimate chose these settings.

### Selected PC

The winning PC is a normalized mixture of 32 binary Chow--Liu tree distributions:

~~~text
P(x) = sum_z pi_z P(x_root | z) product_{j != root} P(x_j | x_parent(j), z).
~~~

Components share the topology within a fold, while each has its own root Bernoulli logit and two conditional Bernoulli logits per non-root gene. Softmax-normalized pi_z mix components. Initialization uses smoothed empirical marginals/conditionals plus small noise, keeping probabilities away from zero and one.

## 2. Saved checkpoint and cross-fitting

The sweep chooses architecture, width, learning rate, and weight decay; it is not the full checkpoint-training record. The existing checkpoint records this actual run:

| Checkpoint setting | Recorded value |
| --- | --- |
| PC / components | mixture-Chow--Liu / 32 |
| Cross-fitting folds | 2 |
| Training / held-out cells per fold | 5,975 / 5,975 |
| Epochs | 20 |
| Training minibatch | 128 cells |
| AdamW learning rate / weight decay | 0.02 / 1e-5 |
| Fold seeds | 0 and 1 |
| Device | MPS |
| Training time | 4.84 s |
| Final training NLL, folds 1 and 2 | 4.025127, 4.042978 |

Each epoch shuffles cells and AdamW minimizes mean negative log likelihood in minibatches. The training batch size is a number of **cells**, never a partition of the 12 genes.

~~~text
All 11,950 cells -- seeded permutation --> held-out F1 (5,975) + F2 (5,975)

                 train on F2                     train on F1
                    --------                        --------
F1 target rows --> | PC_1 | --> model probabilities | PC_2 | <-- F2 target rows
                    --------                        --------
       counts/TMLE on F1                              counts/TMLE on F2
                     \                                /
                      \-- sample-size-weighted ----/
                           cross-fitted result
~~~

Each cell is targeted using a PC that was not fitted on that cell. The two held-out folds partition the data, so every cell contributes once.

## 3. PC querying and observed counts

At interaction order r, the default panel run evaluates every C(12,r) set: 66 pairs, 220 triples, 495 quadruples, or 792 five-gene sets. --genes instead evaluates one explicit named set.

For each set S and every binary configuration a in {0,1}^r, the PC returns the exact marginal P(X_S=a); it does not simulate cells. The tree PC uses batched sum-product/message passing: it places evidence on selected genes, integrates the other tree nodes, then averages components with pi. The interaction CLI's --batch-size here is a batch of **interaction sets**, not genes or cells.

For the corresponding held-out fold, all 2^r observed-cell counts are made in a batch. A selected configuration is encoded as a binary integer and torch.bincount forms the contingency table without Python cell loops.

## 4. Estimand, TMLE, and cross-fitted inference

The estimand is the log interaction:

~~~text
theta(P) = sum_a s(a) log P(X_S=a),
s(a) = (-1)^(r - sum(a)).
~~~

For a pair, this is log[p00 p11 / (p01 p10)], the log odds ratio. Zero is the no-interaction reference; positive/negative are positive/negative association on this scale. For r > 2, exp(theta) is an alternating cross-product ratio, not a correlation or causal effect.

For each fold and target, the initial PC table p_0 is updated on the universal least-favourable path:

~~~text
p_epsilon(a) = p_0(a) + epsilon s(a).
~~~

The allowed epsilon interval keeps every probability strictly positive. Its endpoints are moved slightly inward, then 64 vectorized bisection steps solve:

~~~text
sum_a P_n(a) s(a) / p_epsilon(a) = 0,
~~~

where P_n is the held-out empirical cell distribution. This is the EIF score/estimating equation. **The EIF is not minimized.** TMLE selects epsilon so the empirical EIF mean is zero.

At p_star, the centred EIF is accumulated over held-out counts:

~~~text
H(a) = s(a)/p_star(a) - E_pstar[s(A)/p_star(A)].
~~~

Fold log estimates are weighted by held-out size. Pooled EIF sums and squared sums give:

~~~text
SE = sqrt([sum H^2 - (sum H)^2/N] / [(N-1) N]).
~~~

The table then reports a 95% Wald interval theta +/- 1.959964 SE, a two-sided normal p-value for theta=0, and a Benjamini--Hochberg (BH) q-value.

### What TMLE support means

Two statuses are intentionally distinct.

- **Targeted / finite point estimate:** all initial PC cell probabilities are finite and strictly positive, and a valid bisection root exists in every fold.
- **Supported for unregularized inference:** targeted and, in every fold, every one of the 2^r observed cells has at least --minimum-cell-count observations, every initial PC probability is at least --minimum-probability, and the table is finite.

Only supported targets receive an SE, confidence interval, p-value, and q-value. A finite but unsupported point can be descriptive but is not used for unregularized inference. The saved 2-point panel result uses minimum-cell-count=5 and minimum-probability=1e-12; all 66 pairs are supported. At high order, sparse 2^r tables are usually the scientific limit before compute is.

BH adjusts p-values across supported targets in that run. A BH q-value controls the expected false-discovery proportion at a threshold such as 0.05 under its usual assumptions; it is not the posterior probability that a named edge is false and does not make an association causal.

The stored empirical_log_interaction is only a full-data descriptive comparator. It uses all 11,950 cells, becomes NaN when any observed cell is empty, and is neither cross-fitted nor used for TMLE inference.

## 5. Result interpretation and plots

The .pt table includes gene sets, plug-in and TMLE log effects, SEs, CIs, p/q-values, minimum observed/model probabilities, and targeted/supported flags. For a pair, exponentiate the log effect for an odds ratio. Do not interpret a screen hit as a direct regulatory or causal relationship.

### Full panel comparison: {order}_pt_interactions_12_genes.png

This intentionally tall plot has one row for every requested set.

- Rows sort by decreasing finite empirical log interaction; undefined empirical values go last.
- A circle is the PC plug-in estimate.
- An orange x and bar is a supported TMLE estimate and 95% CI. A finite unsupported estimate is an orange x with no CI.
- A green star is the full-data empirical comparator when defined.
- An orange connector joins plug-in and finite TMLE estimates.
- The dashed vertical zero line is no log interaction.

### Overview, forest, and network/hypergraph

These figures display supported results. Their display screen is q <= 0.05 and |log interaction| >= 0.25; if nothing passes, the code displays all supported results instead. That screen is a display/ranking rule, not a replacement for the full table.

- **Overview:** every supported effect against -log10(BH q-value), capped at 50 only for drawing. Red is positive; blue is negative. Vertical dashed lines are +/-0.25; the horizontal line is q=0.05.
- **Forest:** at most the 25 strongest positive and 25 strongest negative display-screened effects. Points are TMLE log effects and bars are 95% CIs; the black line is zero.
- **Pair network:** ranks display-screened pairs by absolute effect, retains at most 25 while limiting every gene to degree 4, and draws red/blue by sign. Edge width is relative absolute effect and node size is retained degree. It is intentionally not all significant pairs.
- **Hypergraph (r > 2):** draws at most eight strongest display-screened sets as outline polygons. Red/blue shows sign and line width relative magnitude. Polygon overlap has no additional statistical meaning.

## 6. Why the current code is faster than legacy_code/

The improvement is algorithmic, not a shortcut in the estimand. The retired code handled each set separately, invoked a generic circuit-integration query for each of its 2^r outcomes, and repeatedly scanned held-out cells in Python for counts, diagnostics, and TMLE.

The current code batches sets/outcomes, uses exact tree message passing, constructs contingency tables with tensors, and solves bisection updates in a batch. It remains exact for the fitted PC's requested marginals and uses the same universal least-favourable TMLE fluctuation. It is not intended to reproduce numerical values from the retired Cirkit checkpoint: the PC class, topology learning, and fitted checkpoint differ.

All panel orders can be enumerated combinatorially, but increasing 2^r sparsity makes unregularized support the practical scientific constraint.

