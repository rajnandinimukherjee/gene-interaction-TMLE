# Higher-order interaction feasibility: 999 genes

## The short version

The complete all-gene **pairwise** analysis is practical.  The complete
all-gene **higher-order** analysis is not.

That conclusion applies to the **current estimand and implementation**:
the marginal interaction
`p(X_S = a)`, estimated with the current cross-fitted marginal-table TMLE.
There is a materially different exact-state/MFI estimand,
`p(X_S = a, X_-S = 0)`, where every non-selected gene is off.  This document
discusses it only to compare the present analysis with the Gene Flows study:
it is **not** a candidate replacement estimand or implementation route for
this analysis.  It does not make the present data support that estimand, make
it causal by itself, or make exhaustive orders four and above small.  The
comparison begins in
[Exact-state/MFI comparator](#exact-statemfi-all-other-genes-off-comparator-not-a-proposed-analysis).

This is not because the input data file is unusually large.  It is because
the number of gene groups grows extremely quickly once groups contain three
or more genes.

The practical approach is:

1. evaluate every pair exactly;
2. retain strong, statistically supported pairwise signals;
3. use those signals to form a bounded set of three- or higher-gene
   candidates;
4. run the exact higher-order TMLE only for those candidates; and
5. regard the resulting higher-order hits as exploratory until they are
   confirmed in independent data.

For the current pairwise result and default screen, this produces **4,789
three-gene candidates**, rather than all 165.7 million possible triples.

## What is feasible?

| Analysis | Number of gene sets | Practical status |
| --- | ---: | --- |
| All pairs | C(999, 2) = 498,501 | Practical on CPU; the cached current run took about 2.5 seconds. |
| All triples | C(999, 3) = 165,668,499 | Not practical in the current implementation. |
| All four-gene sets | C(999, 4) = 41,251,456,251 | Not realistic. |
| All five-gene sets | C(999, 5) = 8,209,039,793,949 | Categorically infeasible. |
| Default screened triples | 4,789 | Practical; suitable for a CUDA GPU and potentially manageable on CPU. |

The fast pairwise result does **not** mean that an all-triple scan will also
be fast.  Pairs have a special shortcut: this probabilistic-circuit model can
calculate all one- and two-gene marginals analytically, and the held-out
pairwise counts come from one matrix multiplication.  There is no equivalent
all-combinations shortcut for arbitrary three- or higher-gene sets.

## Why higher-order analysis becomes expensive

For every selected set of `r` genes, there are `2^r` possible on/off patterns:

| Interaction order | Patterns per gene set |
| ---: | ---: |
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |
| 5 | 32 |

For each gene set, the workflow must:

- predict the probability of every pattern under each held-out-fold model;
- count every observed pattern in that fold;
- perform the TMLE targeting update (64 bisection iterations); and
- pool the two folds and calculate uncertainty and a multiple-testing value.

The selected model has 999 genes and 32 mixture components.  Its general
higher-order query uses tree message passing through the 999-gene model.
Thus a triple is much more than one small eight-cell table: it invokes a
substantial model calculation twice, once for each cross-fitting fold.

The important problem is the **number of sets**, not the number of cells or
the size of the trained model.  Training the current model is quick; running
hundreds of millions of inference targets is not.

## Memory: two different kinds

It helps to separate memory used for one GPU batch from memory needed to hold
all results of an exhaustive scan.

### GPU memory for one batch

With the current default higher-order batch size of 512, the tree-message
workspace approximately doubles for each added gene:

| Order | Approximate live model workspace | Sensible VRAM budget |
| ---: | ---: | ---: |
| 3 | about 2 GiB | 4–8 GiB |
| 4 | about 4 GiB | 8–16 GiB |
| 5 | about 8 GiB | 16+ GiB |

These are working-memory estimates, not guarantees: CUDA's allocator,
temporary tensors, and other processes need headroom.  Reducing
`--batch-size` reduces this VRAM requirement approximately linearly, but does
not reduce the total amount of computation.

A 16–24 GB CUDA GPU has ample headroom for the screened triple analysis.  A
40–80 GB data-centre GPU can use larger batches and provides more margin for
higher-order screened work.

### RAM and disk for a *complete* triple scan

Even if batches are processed sequentially, the current code stores the
candidate list, result fields, intermediate totals, and all p-values needed
for the final Benjamini--Hochberg correction.

| Exhaustive analysis | Raw final result table alone |
| --- | ---: |
| Triples | about 14.5 GiB |
| Four-gene sets | about 3.8 TiB |
| Five-gene sets | about 821 TiB |

For all triples, the candidate list itself is about 3.5 GiB.  The current
in-memory implementation would need several intermediate arrays as well, so
a realistic full-triple job would require roughly 32–64 GB of host RAM and
at least tens of GB of fast temporary/output storage.  This is before
considering runtime.

For the screened 4,789 triples, result storage is tiny.  The per-batch
model workspace, rather than the final result table, is the relevant memory
cost.

## Why a laptop struggles

An ordinary laptop can comfortably load the 11,950 by 999 binary data matrix
and the trained model.  It struggles with exhaustive higher-order inference
because it combines:

- hundreds of millions (or more) separate targets;
- two cross-fitting folds per target;
- twice as many table patterns for every additional gene;
- repeated double-precision TMLE calculations;
- large intermediate/result arrays for a full scan; and
- generally limited GPU memory and sustained compute compared with a
  data-centre GPU.

There is also a statistical limitation that faster hardware cannot fix.  The
earlier pairwise run already had 108,776 pairs that did not meet the
unregularized support rule with a minimum cell count of one.  A triple has
eight cells rather than four, so rare gene patterns will leave many more
triples without reliable standard errors, confidence intervals, or p-values.
Orders four and five are sparser still.

## MPS versus CUDA

### Why MPS previously failed or did not accelerate the analysis

The TMLE calculation deliberately uses `float64` (double precision).  This
is important when probabilities for rare configurations are very small.
Apple's MPS backend has limited/reliable float64 support for the required
operations.

The current implementation therefore deliberately sends pairwise and
higher-order TMLE work to the **CPU** when `--device mps` is requested.  This
avoids an MPS failure, but it also means that MPS does not accelerate this
part of the analysis.

### Can it run on CUDA?

Yes.  CUDA supports the required float64 tensor operations, including the
tree message passing, contingency-table counting, bisection update, and
error-function calculation.  The higher-order code keeps CUDA as its device;
it does not apply the MPS-to-CPU fallback.

Requirements are:

- an NVIDIA GPU with sufficient VRAM;
- a CUDA-enabled PyTorch installation and compatible NVIDIA driver; and
- a conservative starting batch size (for example, 128 or 512) that can be
  increased after measuring peak VRAM.

CUDA success does not remove the combinatorial explosion.  If a CUDA job runs
out of memory, lowering `--batch-size` can make it run, but it will not make a
complete all-triple scan small or quick.

Data-centre GPUs such as NVIDIA A100 or H100 are preferable because they have
large memory and strong double-precision performance.  Consumer GeForce GPUs
can still run the calculation correctly, but their double-precision speed is
often much lower.

## Would a GPU cluster make exhaustive higher-order analysis possible?

For the screened workflow, a cluster is unnecessary: a single suitable CUDA
GPU is enough for the current triple candidate set.

For an **exhaustive triple** scan, a GPU cluster could make the task
technically conceivable only after an engineering redesign.  The present
program is single-device: allocating a cluster to it will still use just one
GPU.  A distributed version would need to:

1. generate and split the triple combinations across jobs without retaining
   all combinations in one process;
2. stream each job's results to durable storage;
3. merge the results; and
4. calculate the final multiple-testing correction across the whole family.

That would be a substantial compute-and-software project.  It could be
considered for triples if there were a compelling scientific reason, but it
does not make orders four and above reasonable.  More importantly, it does
not solve sparse contingency tables or make unsupported effects inferential.

## Recommended use of compute

1. Keep the exact all-pair analysis; it is already cheap and complete.
2. Run the pre-specified pairwise screen and degree cap.
3. Evaluate the resulting triples on CUDA, if available.
4. For fourth-order work, impose even stronger biological or pairwise
   pre-screening and keep the candidate cap.
5. Confirm interesting higher-order results in an independent split or an
   external data set.  The same pairwise data are used for screening and for
   higher-order estimation here, so the selected results are exploratory and
   should not be described as confirmatory FDR findings.

In short: CUDA is a good way to accelerate bounded, screened higher-order
analysis for the current marginal-TMLE target.  It is not a way to turn all
`C(999, r)` marginal targets into a practical or statistically well-supported
analysis.  The exact-state/MFI comparator below explains why the Gene Flows
GPU result does not alter that conclusion for the present estimand.

# Exact-state/MFI (“all other genes off”) comparator

## The target under discussion

For a selected gene set `S` of order `r`, let `A` be the subset of selected
genes that are on in binary pattern `a`.  Define the **exact-state mass**

```text
q(A) = p(X_A = 1, X_-A = 0) = p(X = 1_A).
```

For a pair `{i, j}`, the four components are therefore

```text
q({})       = p(X_i=0, X_j=0, X_-(i,j)=0)
q({j})      = p(X_i=0, X_j=1, X_-(i,j)=0)
q({i})      = p(X_i=1, X_j=0, X_-(i,j)=0)
q({i,j})    = p(X_i=1, X_j=1, X_-(i,j)=0).
```

The corresponding order-`r` MFI/log interaction is

```text
I_S = sum_{A subseteq S} (-1)^(r-|A|) log q(A).
```

This differs fundamentally from the current higher-order target,
`sum_a (-1)^(r-sum(a)) log p(X_S=a)`, which marginalises all genes outside
`S`.  If `R = X_-S`, then normalising the exact-state masses gives
`p(X_S=a | R=0) = q(A) / sum_(B subseteq S) q(B)`.  The normalising constant
cancels from the alternating log contrast.  Thus `I_S` is the interaction
conditional on the very specific all-other-genes-off stratum.

This is an observational conditional association, not automatically a causal
quantity.  `p(X_S=a, X_-S=0)` is not an interventional distribution.  A claim
about `p(... | do(X_-S=0))`, or about regulation of one gene by another,
would require a causal graph/structural model and identification assumptions
that this PC analysis does not supply.  Intervening to fix every gene would
also require a carefully stated outcome; it is not equivalent to observing a
rare joint expression state.

## Why the exact-state target has much more cache reuse

For this target, a probability is determined solely by its **active set**
`A`; it does not depend on which additional selected genes happen to be off.
For example, `q({i})` is reused in every pair containing `i`, and `q({i,j})`
is reused in every triple containing `{i,j}`.  This reuse is unavailable for
the current marginal probabilities: `p(X_i=1, X_j=0)` and
`p(X_i=1, X_j=0, X_k=0)` marginalise different sets of variables.

An exhaustive order-`r` exact-state scan therefore needs at most

```text
sum_(k=0)^r C(999, k)
```

distinct model probabilities, rather than evaluating `2^r C(999,r)` pattern
rows independently.  The interaction values still number `C(999,r)` and must
be emitted or otherwise reduced, so this is a large constant-factor saving,
not a removal of combinatorics.

| Order | Interaction sets | Direct pattern rows | Distinct exact-state masses `q(A)` with reuse |
| ---: | ---: | ---: | ---: |
| 2 | 498,501 | 1,994,004 | 499,501 |
| 3 | 165,668,499 | 1,325,347,992 | 166,168,000 |
| 4 | 41,251,456,251 | 660,023,300,016 | 41,417,624,251 |

For pairs, this is the particularly simple cache
`q({}), q({i}), q({i,j})`.  For triples, a singleton is shared by
`C(998,2)` triples and a pair by 997 triples.  A streamed implementation does
not need to materialise every triple mass: it can retain the lower-order
caches, calculate a triple mass, assemble that triple's interaction, and
write only the final shard.

## What the data says: this exact-state estimand has no raw support here

The 11,950 by 999 input matrix was checked directly.  Its smallest row sum is
**29**: there are no all-zero, singleton, doubleton, or tripleton cell
profiles.  Consequently, for every pair and every triple, each exact-state
cell needed by the target has observed count zero.  The same is true for every
order below 29.

This has three consequences.

1. The empirical conditional table given `X_-S=0` is undefined: its
   denominator has count zero.
2. The present `universal_tmle_fold` cannot be reused by substituting `q(A)`
   for its probability table.  It expects a probability vector summing to one
   and nonzero held-out table counts.  Even after conditionalising the `q(A)`
   values, there are no observations in the conditioning stratum with which
   to target the update or estimate its EIF variance.
3. A PC can still assign positive exact-state mass because its Bernoulli
   parameters are smoothed.  Such an estimate would be a **model extrapolation**
   into an entirely unobserved part of the 999-gene state space.  It cannot
   carry the current unregularised TMLE confidence intervals, p-values, or
   BH q-values.

Accordingly, this analysis will not add an unscreened exact-state/MFI run.  It
would be a model extrapolation rather than a supported estimate and is not the
marginal estimand selected for this study.

## What Bossemeyer and Belo did in the Gene Flows study

In Section 2.1 and Sections 4.6--4.7,
Leonie **Bossemeyer** and Alex **Belo** use this same all-other-genes-off MFI
formulation.  Their two-factor formula contains
`p(X_i,X_j, X_-(i,j)=0)` for each of the four on/off patterns; their
three-factor formula extends it to eight exact full-state masses.

Their ability to avoid screening is computational rather than statistical:

- They train a MADE-based autoregressive/flow density model that can score a
  complete 1,000-gene binary input in one GPU forward pass.
- They enumerate combinations in batches, begin each batch as an all-zero
  Boolean tensor on the GPU, and set the selected on-gene columns to one.
  For higher orders they use a generator, avoiding a giant in-memory list of
  combinations.
- Their simple “option A” predicts every binary pattern for every gene set.
  It is a brute-force streamed scan, not a combinatorial shortcut.
- They distribute non-all-zero pattern classes across GPUs: three GPUs for
  order 2, seven for order 3, and 15 for order 4.  The shared all-off mass is
  evaluated once.
- They preallocate outputs and use large batches; their performance tests use
  fixed MADE masks, a compiled model, and float16 inference.  The reported
  fastest order-3 all-on bottleneck is 50.686 seconds for all
  `C(1000,3)` inputs on one A100 at batch size 200,000.  They report the
  complete order-3 scan in under 10 minutes.
- Their “option B” recognises the cache described above: calculate all-off,
  singleton-on, then doubleton-on probabilities once and copy/reindex lower
  order values to form pair tables.  They implemented this reuse only for
  order 2.  They note that lookup structures for orders 4+ become a RAM/disk
  problem.

Their results also show the boundary of the approach, rather than eliminating
it.  They report that order-4 probabilities require 15 GPUs, under four hours
of computation, and about 1.3 TB of storage; that storage requirement stopped
them from calculating the final order-4 MFI scores.  They project order 5 as
impractical without many more GPUs and further engineering.

### Parallels and differences with this analysis

| Topic | Bossemeyer & Belo | Current 999-gene analysis |
| --- | --- | --- |
| Primary high-order target | Exact-state MFI: selected pattern with every other gene off. | Marginal alternating log interaction: unselected genes integrated out. |
| Density model | MADE-based autoregressive/flow model, scored by dense GPU forward passes. | 32-component latent-class residual Chow--Liu tree PC, with exact tree marginal queries. |
| Unscreened calculation | Stream every combination/pattern through CUDA, then retain/shard outputs. | Only all pairs are exhaustive; order 3+ is pair-screened before general marginal tree-message queries. |
| Probability reuse | Their documented reuse is pairwise only; triples/quadruples use mostly direct pattern scans. | The current marginal target has no equivalent active-subset cache; that reuse belongs to their different exact-state target. |
| Precision and hardware | A100 CUDA, large batches, float16 performance experiment, multiple GPUs. | Float64 TMLE; MPS deliberately falls back to CPU; CUDA is supported but the present higher-order program is single-device. |
| Statistical procedure | Model plug-in MFI and delta-method variance/z testing. | Two-fold cross-fitted PC plug-in plus TMLE, held-out contingency counts, EIF SEs, and BH only for supported targets. |

The comparison offers useful engineering lessons, but not a transfer of
scientific validity.  Their neural-model likelihood benchmark does not show
that the model is calibrated in the extreme all-other-off region, and their
plug-in/delta-method procedure does not repair zero observed support.  Nor can
their float16/A100 timing be treated as a prediction for this PC: the model
architectures, numerical precision, batching internals, and cross-fitting
requirements differ.

## What the 12-gene calculations imply, and what they do not

The existing 12-gene workflow is a useful check on the **current generic
marginal-TMLE implementation**, not a direct benchmark for the different
exact-state GPU computation in Gene Flows.  With all 11,950 cells, two folds,
batch size 512, minimum held-out cell count 1, and CPU double precision when
MPS is selected, the recent all-combination runs took:

| Order | Sets `C(12,r)` | Pattern tables across two folds | Recorded end-to-end runtime |
| ---: | ---: | ---: | ---: |
| 2 | 66 | 528 | 0.65 s |
| 3 | 220 | 3,520 | 1.13 s |
| 4 | 495 | 15,840 | 2.03 s |
| 5 | 792 | 50,688 | 3.61 s |
| 6 | 924 | 118,272 | 4.49 s |
| 7 | 792 | 202,752 | 3.96 s |

These timings include fixed startup, result reporting, and plotting, so they
are not linear microbenchmarks.  They do establish that the 12-gene code can
evaluate all panel combinations, but its tree-message calculation costs
roughly `batch x 2^r x genes x components` per fold.  It has no path for
all-gene exact-state reuse.

As a deliberately crude sanity check, scaling the recorded 12-gene marginal
wall time by both the ratio of set counts and `999/12` genes gives the
following **non-forecasts** for the current generic CPU algorithm:

| Order | All 999-gene sets | Crude CPU-equivalent time | Why this is not a forecast |
| ---: | ---: | ---: | --- |
| 3 | 165,668,499 | about 2.2 years | GPU batching, I/O, and fixed overhead are ignored. |
| 4 | 41,251,456,251 | about 446 years | Same caveats; output storage alone is substantial. |
| 5 | 8,209,039,793,949 | about 98,700 years | The combinatorial family is already decisive. |

The pairwise row is deliberately omitted: the 999-gene pairwise calculation
has a special analytic cache and actually takes about 2.5 seconds, so it
would make a misleading extrapolation.  Conversely, the table above
does not describe the Gene Flows exact-state computation: a direct full-state
evaluator can reuse active subsets and avoid the `2^r` repeated marginal
queries.  That distinction is precisely why its GPU timing does not transfer
to the chosen marginal target.  The 12-gene results therefore justify the
conclusion that the current route cannot be scaled, not a particular GPU
budget for the different exact-state functional.

The closest published compute guide for an exact-state triple scan is instead
the A100 experiment in Gene Flows: its 1,000-gene triple count is
within 0.3% of `C(999,3)`.  It is useful for explaining how their different
functional can be evaluated, but is not a compute budget or implementation
precedent for this analysis.

## Why the Gene Flows no-screen GPU strategy does not transfer

The Gene Flows GPU pipeline exploits a property of its exact-state target:
the same probability `q(A)` is shared by every selected set that contains
the active subset `A`.  The paper can therefore construct an all-zero input,
turn on the genes in `A`, and score a complete state.  Its documented direct
pipeline nevertheless enumerates the combinations and patterns; large CUDA
batches and multiple A100s make the enumeration fast enough for triples.

For the selected marginal estimand here, the probability needed for a gene
set is `p(X_S=a)`, which sums over different complements for different `S`.
There is no corresponding active-subset cache.  For example,
`p(X_i=1, X_j=0)` cannot be recovered from, or reused as,
`p(X_i=1, X_j=0, X_k=0)`, because the latter restricts rather than
marginalises `X_k`.  The generic PC tree-message query and the held-out
`2^r` contingency table are therefore still required for each marginal target.

CUDA can reduce the constant factor for batched marginal tree messages, but
it cannot remove the `C(999,r) 2^r` target/pattern family, produce the missing
held-out support, or replicate the exact-state reuse that made the Gene Flows
triple calculation possible.  No exact-state cache, MFI evaluator,
tree-delta kernel, or new causal/exact-state code is proposed for this
repository.

## Practical recommendation

Continue with the selected marginal cross-fitted TMLE: retain the all-pair
exact scan, the pairwise screen and degree cap for higher orders, its support
rules, and independent confirmation of selected findings.  The Gene Flows
study is informative about GPU streaming and the hard boundary imposed by
order-4 storage, but it does not provide a no-screen route for this
analysis's marginal estimand.
