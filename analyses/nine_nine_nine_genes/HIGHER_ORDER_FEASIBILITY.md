# Higher-order interaction feasibility: 999 genes

## The short version

The complete all-gene **pairwise** analysis is practical.  The complete
all-gene **higher-order** analysis is not.

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
analysis.  It is not a way to turn all `C(999, r)` gene sets into a practical
or statistically well-supported analysis.
