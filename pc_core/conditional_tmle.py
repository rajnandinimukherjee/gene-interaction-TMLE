"""Cross-fitted conditional-outcome TMLE for ordered gene interactions.

For an ordered target ``(Y, A_1, ..., A_k)`` this module follows the
conditional procedure used by the reference PC/TMLE analysis.  The remaining
genes are covariates ``W``.  A cross-fitted PC supplies

``Q_a(W) = P(Y=1 | A=a, W)`` and ``g_a(W) = P(A=a | W)``.

The target is the alternating log cross-product of the standardized outcome
odds ``m_a = E[Q_a(W)]``.  With one exposure this is the standardized log odds
ratio; with two exposures it is the reference repository's three-gene
interaction.  Computation is deliberately batched over targets and samples
and uses a vectorized bisection fluctuation rather than fitting a Python/GLM
object for every treatment stratum.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


NORMAL_95_CRITICAL_VALUE = 1.959963984540054
DEFAULT_PROBABILITY_CLIP = 1e-6


def treatment_patterns(
    order: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lexicographic exposure patterns and alternating log-odds signs.

    The first member of each ordered gene set is the binary outcome.  The
    remaining ``order - 1`` members form the treatment vector.  Thus pair
    ``(Y, A)`` receives signs ``(-1, +1)`` for treatments ``0, 1`` and a
    triplet ``(Y, A, B)`` receives ``(+1, -1, -1, +1)`` for 00, 01, 10, 11.
    """

    if order < 2:
        raise ValueError(
            "Conditional interactions require an outcome and at least one exposure."
        )
    treatment_order = order - 1
    integers = torch.arange(2**treatment_order, device=device)
    shifts = torch.arange(treatment_order - 1, -1, -1, device=device)
    patterns = ((integers[:, None] >> shifts) & 1).to(dtype)
    signs = torch.where(
        ((treatment_order - patterns.sum(dim=1).to(torch.long)) % 2) == 0,
        torch.ones(len(patterns), dtype=dtype, device=device),
        -torch.ones(len(patterns), dtype=dtype, device=device),
    )
    return patterns, signs


def _observed_outcome_and_treatment(
    data: torch.Tensor,
    gene_sets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return observed Y and lexicographic A codes as [targets, samples]."""

    outcome = data[:, gene_sets[:, 0]].T.to(torch.float64)
    treatment = data[:, gene_sets[:, 1:]].to(torch.long)
    shifts = torch.arange(
        gene_sets.shape[1] - 2, -1, -1, device=data.device, dtype=torch.long
    )
    codes = (treatment * (2**shifts)).sum(dim=-1).T
    return outcome, codes


def _assignment_layout(
    gene_sets: torch.Tensor,
    patterns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten target/pattern/outcome assignments for batched PC evaluation."""

    target_count, order = gene_sets.shape
    n_patterns = len(patterns)
    outcome_values = torch.arange(2, device=gene_sets.device, dtype=patterns.dtype)
    genes = gene_sets[:, None, None, :].expand(target_count, n_patterns, 2, order)
    exposure = patterns[None, :, None, :].expand(target_count, n_patterns, 2, order - 1)
    outcome = outcome_values[None, None, :, None].expand(target_count, n_patterns, 2, 1)
    values = torch.cat((outcome, exposure), dim=-1)
    return genes.reshape(-1, order), values.reshape(-1, order)


@torch.no_grad()
def conditional_nuisances(
    model: torch.nn.Module,
    held_out: torch.Tensor,
    gene_sets: torch.Tensor,
    patterns: torch.Tensor,
    *,
    probability_clip: float,
    sample_batch_size: int,
    max_likelihood_elements: int = 16_000_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate PC conditional outcomes and propensities for a target batch.

    The returned arrays have shape ``[targets, held-out cells, treatments]``.
    A PC is evaluated on the two outcome completions for every treatment
    pattern; normalizing those joint likelihoods gives ``Q`` and ``g``.  The
    operation is chunked over cells so memory scales with the specified target
    batch rather than all target/cell combinations at once.
    """

    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must lie strictly between 0 and 0.5.")
    if sample_batch_size < 1:
        raise ValueError("sample_batch_size must be positive.")
    if gene_sets.ndim != 2 or gene_sets.shape[1] < 2:
        raise ValueError(
            "gene_sets must be a [targets, outcome-plus-exposures] matrix."
        )

    device = held_out.device
    target_count = len(gene_sets)
    n_cells = len(held_out)
    n_patterns = len(patterns)
    dtype = torch.float64
    q = torch.empty((target_count, n_cells, n_patterns), dtype=dtype, device=device)
    g = torch.empty_like(q)
    flat_genes, flat_values = _assignment_layout(gene_sets, patterns)
    configuration_count = len(flat_genes)
    # The cloned completed-input tensor is the largest temporary allocation.
    # Adapt the cell chunk to retain the existing vectorized target/outcome
    # speed-up without allocating a target x cell x pattern x gene cube for a
    # complete held-out fold.
    per_cell = configuration_count * held_out.shape[1]
    adaptive_batch_size = max(1, max_likelihood_elements // max(per_cell, 1))
    effective_sample_batch_size = min(sample_batch_size, adaptive_batch_size)
    rows = torch.arange(configuration_count, device=device)

    for start in range(0, n_cells, effective_sample_batch_size):
        stop = min(start + effective_sample_batch_size, n_cells)
        observed = held_out[start:stop]
        completed = (
            observed[:, None, :].expand(len(observed), configuration_count, -1).clone()
        )
        for position in range(gene_sets.shape[1]):
            completed[:, rows, flat_genes[:, position]] = flat_values[:, position].to(
                completed.dtype
            )
        log_joint = model.log_prob(completed.reshape(-1, completed.shape[-1])).reshape(
            len(observed), target_count, n_patterns, 2
        )
        # Q_a(W) is normalized only over Y.  g_a(W) first sums Y and then
        # normalizes over all treatment patterns, exactly as in the source
        # pair/triplet implementation.
        q_chunk = torch.sigmoid(log_joint[..., 1] - log_joint[..., 0])
        treatment_log_mass = torch.logsumexp(log_joint, dim=-1)
        g_chunk = torch.softmax(treatment_log_mass, dim=-1)
        q[:, start:stop] = q_chunk.permute(1, 0, 2).to(dtype)
        g[:, start:stop] = g_chunk.permute(1, 0, 2).to(dtype)

    raw_minimum_q = q.amin(dim=(1, 2))
    raw_minimum_g = g.amin(dim=(1, 2))
    return (
        q.clamp(probability_clip, 1.0 - probability_clip),
        g.clamp(probability_clip, 1.0 - probability_clip),
        raw_minimum_q,
        raw_minimum_g,
    )


def _log_interaction_from_means(
    means: torch.Tensor, signs: torch.Tensor, clip: float
) -> torch.Tensor:
    """Return the alternating standardized log-odds interaction."""

    return (torch.logit(means.clamp(clip, 1.0 - clip)) * signs).sum(dim=1)


def conditional_tmle_update(
    q: torch.Tensor,
    g: torch.Tensor,
    outcome: torch.Tensor,
    treatment_code: torch.Tensor,
    signs: torch.Tensor,
    *,
    probability_clip: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply source-compatible, stratum-specific logistic TMLE updates.

    For treatment pattern ``a``, the fluctuation solves the empirical score

    ``sum_i I(A_i=a) H_a(W_i) [Y_i - expit(logit Q_a(W_i)+eps_a H_a(W_i))] = 0``

    where ``H_a = sign_a / [m_a(1-m_a)g_a]``.  A vectorized bisection solver
    replaces one statsmodels GLM fit per target/pattern while retaining the
    same intercept-free logistic fluctuation and offsets.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if q.shape != g.shape or q.ndim != 3:
        raise ValueError("q and g must both have shape [targets, cells, treatments].")
    if outcome.shape != q.shape[:2] or treatment_code.shape != q.shape[:2]:
        raise ValueError("Observed outcome/treatment arrays must align with q and g.")

    target_count, _, n_patterns = q.shape
    if len(signs) != n_patterns:
        raise ValueError(
            "The treatment-sign vector does not match the nuisance arrays."
        )
    means = q.mean(dim=1)
    plugin_log = _log_interaction_from_means(means, signs, probability_clip)
    denominators = means * (1.0 - means)
    h = signs[None, None, :] / (denominators[:, None, :] * g)
    observed = torch.nn.functional.one_hot(treatment_code, num_classes=n_patterns).to(
        q.dtype
    )
    observed_count = observed.sum(dim=1)
    observed_success = (observed * outcome[:, :, None]).sum(dim=1)
    # The reference GLMs have no finite solution under complete outcome
    # separation in a treatment stratum.  Mark that numerical situation rather
    # than allowing an arbitrary very-large coefficient to masquerade as a
    # regular TMLE estimate.
    targetable = (
        torch.isfinite(q).all(dim=(1, 2))
        & torch.isfinite(g).all(dim=(1, 2))
        & (observed_count > 0).all(dim=1)
        & (observed_success > 0).all(dim=1)
        & (observed_success < observed_count).all(dim=1)
    )

    minimum_abs_h = h.abs().amin(dim=1).clamp_min(torch.finfo(q.dtype).tiny)
    # At these endpoints every observed stratum has |epsilon H| >= 40, so the
    # monotone logistic score has effectively reached its two limits whenever
    # both outcome states are present.
    lower = -40.0 / minimum_abs_h
    upper = 40.0 / minimum_abs_h
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        q_midpoint = torch.sigmoid(torch.logit(q) + midpoint[:, None, :] * h)
        score = (observed * h * (outcome[:, :, None] - q_midpoint)).sum(dim=1)
        lower = torch.where(score > 0.0, midpoint, lower)
        upper = torch.where(score > 0.0, upper, midpoint)
    epsilon = (lower + upper) / 2.0
    q_star = torch.sigmoid(torch.logit(q) + epsilon[:, None, :] * h)
    means_star = q_star.mean(dim=1)
    tmle_log = _log_interaction_from_means(means_star, signs, probability_clip)

    # This is the log-interaction EIF used by the source pair/triplet code:
    # the inverse-propensity residual term plus the standardization term, then
    # the delta-method derivative of each logit(m_a).
    first_order = observed * (outcome[:, :, None] - q_star) / g
    eif_by_pattern = first_order + q_star - means_star[:, None, :]
    eif = (
        eif_by_pattern
        * signs[None, None, :]
        / (means_star[:, None, :] * (1.0 - means_star[:, None, :]))
    ).sum(dim=2)
    eif_sum = eif.sum(dim=1)
    eif_square_sum = eif.square().sum(dim=1)
    nan = torch.full_like(tmle_log, float("nan"))
    tmle_log = torch.where(targetable, tmle_log, nan)
    eif_sum = torch.where(targetable, eif_sum, nan)
    eif_square_sum = torch.where(targetable, eif_square_sum, nan)
    return plugin_log, tmle_log, eif_sum, eif_square_sum, targetable


def benjamini_yekutieli(p_values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return BY-adjusted p-values, with NaN for untargetable hypotheses."""

    q_values = torch.full_like(p_values, float("nan"))
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if len(valid_indices) == 0:
        return q_values
    p = p_values[valid_indices].clamp(0.0, 1.0)
    sorted_p, order = torch.sort(p)
    m = len(p)
    ranks = torch.arange(1, m + 1, dtype=p.dtype, device=p.device)
    harmonic = (1.0 / ranks).sum()
    raw_q = sorted_p * m * harmonic / ranks
    monotone_q = torch.flip(
        torch.cummin(torch.flip(raw_q, dims=(0,)), dim=0).values, dims=(0,)
    ).clamp(max=1.0)
    q_values[valid_indices[order]] = monotone_q
    return q_values


def run_conditional_tmle(
    models: Sequence[torch.nn.Module],
    folds: Sequence[object],
    values: object,
    gene_sets_cpu: torch.Tensor,
    *,
    device: torch.device,
    target_batch_size: int,
    nuisance_sample_batch_size: int,
    probability_clip: float = DEFAULT_PROBABILITY_CLIP,
    fluctuation_iterations: int = 64,
) -> dict[str, torch.Tensor]:
    """Run fold-targeted, cross-fitted conditional TMLE for gene sets.

    Each PC predicts only its held-out fold and that fold receives its own
    fluctuation.  Fold estimates are then weighted by held-out size and their
    EIF moments pooled.  This retains the two-fold cross-targeting protocol of
    the marginal workflow while changing the fold-level target to the
    conditional-outcome estimand.
    """

    if target_batch_size < 1:
        raise ValueError("target_batch_size must be positive.")
    if len(models) != len(folds):
        raise ValueError("Each restored model must have one held-out fold.")
    if gene_sets_cpu.ndim != 2 or len(gene_sets_cpu) == 0:
        raise ValueError("gene_sets_cpu must contain at least one ordered target.")

    dtype = torch.float64
    order = gene_sets_cpu.shape[1]
    patterns, signs = treatment_patterns(order, device, dtype)
    n_targets = len(gene_sets_cpu)
    n_patterns = len(patterns)
    total_n = sum(len(fold) for fold in folds)
    # q and g are retained for only this block.  Leave headroom for the
    # completed-input temporary allocation in conditional_nuisances.
    max_block_elements = 16_000_000
    bounded_batch_size = min(
        target_batch_size,
        max(1, max_block_elements // max(total_n * n_patterns * 2, 1)),
    )

    plugin_log = torch.empty(n_targets, dtype=dtype)
    tmle_log = torch.empty(n_targets, dtype=dtype)
    eif_sum = torch.empty(n_targets, dtype=dtype)
    eif_square_sum = torch.empty(n_targets, dtype=dtype)
    targeted = torch.empty(n_targets, dtype=torch.bool)
    minimum_q = torch.empty(n_targets, dtype=dtype)
    minimum_g = torch.empty(n_targets, dtype=dtype)

    for start in range(0, n_targets, bounded_batch_size):
        stop = min(start + bounded_batch_size, n_targets)
        gene_sets = gene_sets_cpu[start:stop].to(device=device, dtype=torch.long)
        q_minima: list[torch.Tensor] = []
        g_minima: list[torch.Tensor] = []
        block_plugin_total = torch.zeros(stop - start, dtype=dtype, device=device)
        block_tmle_total = torch.zeros_like(block_plugin_total)
        block_eif_sum_total = torch.zeros_like(block_plugin_total)
        block_eif_square_sum_total = torch.zeros_like(block_plugin_total)
        block_targeted = torch.ones(stop - start, dtype=torch.bool, device=device)
        for model, fold_indices in zip(models, folds):
            held_out = torch.as_tensor(
                values[fold_indices], dtype=torch.float32, device=device
            )
            q, g, raw_q_minimum, raw_g_minimum = conditional_nuisances(
                model,
                held_out,
                gene_sets,
                patterns,
                probability_clip=probability_clip,
                sample_batch_size=nuisance_sample_batch_size,
            )
            outcome, treatment_code = _observed_outcome_and_treatment(
                held_out, gene_sets
            )
            fold_plugin, fold_tmle, fold_eif_sum, fold_eif_square_sum, fold_targeted = (
                conditional_tmle_update(
                    q,
                    g,
                    outcome,
                    treatment_code,
                    signs,
                    probability_clip=probability_clip,
                    iterations=fluctuation_iterations,
                )
            )
            fold_size = len(held_out)
            block_plugin_total += fold_size * fold_plugin
            block_tmle_total += fold_size * torch.nan_to_num(fold_tmle, nan=0.0)
            block_eif_sum_total += torch.nan_to_num(fold_eif_sum, nan=0.0)
            block_eif_square_sum_total += torch.nan_to_num(fold_eif_square_sum, nan=0.0)
            block_targeted &= fold_targeted
            q_minima.append(raw_q_minimum)
            g_minima.append(raw_g_minimum)
        block_plugin = block_plugin_total / total_n
        block_tmle = block_tmle_total / total_n
        block_tmle[~block_targeted] = float("nan")
        plugin_log[start:stop] = block_plugin.cpu()
        tmle_log[start:stop] = block_tmle.cpu()
        eif_sum[start:stop] = block_eif_sum_total.cpu()
        eif_square_sum[start:stop] = block_eif_square_sum_total.cpu()
        targeted[start:stop] = block_targeted.cpu()
        minimum_q[start:stop] = torch.stack(q_minima).amin(dim=0).cpu()
        minimum_g[start:stop] = torch.stack(g_minima).amin(dim=0).cpu()

    variance_numerator = eif_square_sum - eif_sum.square() / total_n
    # The reference code uses np.var(phi) (ddof=0), then sqrt(var / N).
    se = torch.sqrt(torch.clamp(variance_numerator / (total_n * total_n), min=0.0))
    se[~targeted] = float("nan")
    z = tmle_log / se
    p_value = torch.erfc(torch.abs(z) / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
    p_value = torch.where(
        (se == 0.0) & (tmle_log == 0.0), torch.ones_like(p_value), p_value
    )
    p_value = torch.where(
        (se == 0.0) & (tmle_log != 0.0), torch.zeros_like(p_value), p_value
    )
    p_value[~targeted] = float("nan")
    q_value = benjamini_yekutieli(p_value, targeted)
    return {
        "plugin_log_interaction": plugin_log,
        "tmle_log_interaction": tmle_log,
        "tmle_se": se,
        "tmle_log_ci_low": tmle_log - NORMAL_95_CRITICAL_VALUE * se,
        "tmle_log_ci_high": tmle_log + NORMAL_95_CRITICAL_VALUE * se,
        "p_value": p_value,
        "q_value": q_value,
        "tmle_targeted": targeted,
        # Kept for plotting/reporting compatibility.  Unlike the marginal
        # branch, this is a numerical-targeting status rather than an
        # empirical four-cell support gate.
        "tmle_supported": targeted.clone(),
        "minimum_conditional_outcome_probability": minimum_q,
        "minimum_conditional_propensity": minimum_g,
    }
