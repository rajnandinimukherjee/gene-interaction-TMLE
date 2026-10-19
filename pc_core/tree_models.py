"""Tree-structured mixture probabilistic circuits for binary gene data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def learn_chow_liu_tree(data: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Learn a maximum-mutual-information binary tree using only training rows.

    The returned parent array represents an undirected Chow--Liu tree oriented
    away from a high-entropy root.  It is deliberately learned before model
    optimization, avoiding any validation-fold leakage in architecture sweeps
    and cross-fitting.
    """

    if data.ndim != 2 or len(data) < 2:
        raise ValueError("Chow--Liu learning requires at least two observations.")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")
    x = np.asarray(data, dtype=np.float64)
    n, d = x.shape
    # NumPy linked against some Accelerate builds can emit spurious overflow
    # warnings for this binary GEMM. Torch's CPU implementation is stable and
    # keeps the calculation deterministic across the training architectures.
    x_torch = torch.as_tensor(x, dtype=torch.float64, device="cpu")
    count_one = x_torch.sum(dim=0).numpy()
    count11 = (x_torch.T @ x_torch).numpy()
    count10 = count_one[:, None] - count11
    count01 = count_one[None, :] - count11
    count00 = n - count_one[:, None] - count_one[None, :] + count11
    denominator = n + 4.0 * smoothing
    p11 = (count11 + smoothing) / denominator
    p10 = (count10 + smoothing) / denominator
    p01 = (count01 + smoothing) / denominator
    p00 = (count00 + smoothing) / denominator
    px1, px0 = p11 + p10, p01 + p00
    py1, py0 = p11 + p01, p10 + p00
    mutual_information = (
        p11 * np.log(p11 / (px1 * py1))
        + p10 * np.log(p10 / (px1 * py0))
        + p01 * np.log(p01 / (px0 * py1))
        + p00 * np.log(p00 / (px0 * py0))
    )
    np.fill_diagonal(mutual_information, -np.inf)

    marginal = ((count_one + smoothing) / (n + 2.0 * smoothing)).clip(1e-12, 1.0 - 1e-12)
    entropy = -marginal * np.log(marginal) - (1.0 - marginal) * np.log(1.0 - marginal)
    root = int(np.argmax(entropy))
    parents = np.full(d, -1, dtype=np.int64)
    selected = np.zeros(d, dtype=bool)
    selected[root] = True
    best_edge = mutual_information[root].copy()
    parents[:] = root
    parents[root] = -1
    best_edge[root] = -np.inf
    for _ in range(d - 1):
        candidate_scores = np.where(selected, -np.inf, best_edge)
        node = int(np.argmax(candidate_scores))
        selected[node] = True
        update = (~selected) & (mutual_information[node] > best_edge)
        best_edge[update] = mutual_information[node, update]
        parents[update] = node
    return parents


def _tree_order(parents: np.ndarray) -> tuple[int, list[int], list[list[int]]]:
    roots = np.flatnonzero(parents < 0)
    if len(roots) != 1:
        raise ValueError("A tree parent array must contain exactly one root.")
    root = int(roots[0])
    d = len(parents)
    children: list[list[int]] = [[] for _ in range(d)]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(child)
    order: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children[node]))
    if len(order) != d:
        raise ValueError("Tree parent array is disconnected.")
    return root, order, children


def _empirical_tree_logits(data: torch.Tensor, parents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothed root and conditional Bernoulli logits for one tree."""

    x = data.float()
    d = x.shape[1]
    root = int(torch.nonzero(parents < 0, as_tuple=False).item())
    root_probability = ((x[:, root].sum() + 1.0) / (len(x) + 2.0)).clamp(1e-4, 1.0 - 1e-4)
    conditional_probability = torch.full((d, 2), 0.5, device=x.device)
    for child in range(d):
        parent = int(parents[child].item())
        if parent < 0:
            continue
        parent_values = x[:, parent]
        for parent_outcome in (0, 1):
            mask = parent_values == parent_outcome
            conditional_probability[child, parent_outcome] = (
                (x[mask, child].sum() + 1.0) / (mask.sum() + 2.0)
            )
    return torch.logit(root_probability), torch.logit(conditional_probability.clamp(1e-4, 1.0 - 1e-4))


def _tree_pair_marginals(
    pi: torch.Tensor,
    root_logits: torch.Tensor,
    conditional_logits: torch.Tensor,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute all exact pair marginals of a mixture of binary tree models.

    For binary tree distributions, covariance propagated across an edge is the
    incoming covariance multiplied by that edge's directed regression slope.
    Propagating from every source supplies every pair marginal without calling
    a generic circuit integration query per pair.
    """

    pi_np = pi.detach().cpu().double().numpy()
    root_np = torch.sigmoid(root_logits.detach()).cpu().double().numpy()
    conditional_np = torch.sigmoid(conditional_logits.detach()).cpu().double().numpy()
    parents_np = parents.detach().cpu().numpy().astype(np.int64, copy=False)
    root, _, children = _tree_order(parents_np)
    k, d, _ = conditional_np.shape

    means = np.empty((k, d), dtype=np.float64)
    means[:, root] = root_np
    traversal = [root]
    for node in traversal:
        traversal.extend(children[node])
    for node in traversal:
        for child in children[node]:
            p0 = conditional_np[:, child, 0]
            p1 = conditional_np[:, child, 1]
            means[:, child] = p0 + (p1 - p0) * means[:, node]

    variances = np.maximum(means * (1.0 - means), 1e-15)
    down_slope = np.zeros((k, d), dtype=np.float64)
    up_slope = np.zeros((k, d), dtype=np.float64)
    for child in range(d):
        parent = parents_np[child]
        if parent < 0:
            continue
        down_slope[:, child] = conditional_np[:, child, 1] - conditional_np[:, child, 0]
        up_slope[:, child] = down_slope[:, child] * variances[:, parent] / variances[:, child]

    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(d)]
    for child, parent in enumerate(parents_np):
        if parent < 0:
            continue
        adjacency[int(parent)].append((child, down_slope[:, child]))
        adjacency[child].append((int(parent), up_slope[:, child]))

    p11 = np.empty((d, d), dtype=np.float64)
    for source in range(d):
        covariance = np.zeros((k, d), dtype=np.float64)
        covariance[:, source] = variances[:, source]
        stack: list[tuple[int, int]] = [(source, -1)]
        while stack:
            node, previous = stack.pop()
            for neighbour, slope in adjacency[node]:
                if neighbour == previous:
                    continue
                covariance[:, neighbour] = covariance[:, node] * slope
                stack.append((neighbour, node))
        p11[source] = np.sum(
            pi_np[:, None] * (covariance + means[:, source, None] * means), axis=0
        )
    p_one = pi_np @ means
    p11 = (p11 + p11.T) / 2.0
    np.fill_diagonal(p11, p_one)
    return torch.from_numpy(p_one), torch.from_numpy(p11)


class _TreeMixturePC(nn.Module):
    """Shared tree likelihood and marginal-query mechanics."""

    def __init__(self, num_genes: int, num_components: int, parents: Sequence[int]) -> None:
        super().__init__()
        parent_tensor = torch.as_tensor(parents, dtype=torch.long)
        if parent_tensor.shape != (num_genes,):
            raise ValueError("Tree parents must contain one entry per gene.")
        self.num_genes = int(num_genes)
        self.num_components = int(num_components)
        self.register_buffer("parents", parent_tensor)
        root = torch.nonzero(parent_tensor < 0, as_tuple=False).flatten()
        if len(root) != 1:
            raise ValueError("Tree must contain exactly one root.")
        self.root = int(root.item())
        self.logits_pi = nn.Parameter(torch.zeros(num_components))

    @property
    def pi(self) -> torch.Tensor:
        return torch.softmax(self.logits_pi, dim=0)

    @property
    def root_logits(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def conditional_logits(self) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim != 2 or data.shape[1] != self.num_genes:
            raise ValueError(f"Expected [batch, {self.num_genes}] observations.")
        x = data.float()
        parent_values = x[:, self.parents.clamp_min(0)].unsqueeze(1)
        conditional_logits = self.conditional_logits
        logits0, logits1 = conditional_logits[:, :, 0], conditional_logits[:, :, 1]
        selected_logits = logits0.unsqueeze(0) + parent_values * (logits1 - logits0).unsqueeze(0)
        leaf_log_prob = (
            x.unsqueeze(1) * torch.nn.functional.logsigmoid(selected_logits)
            + (1.0 - x).unsqueeze(1) * torch.nn.functional.logsigmoid(-selected_logits)
        )
        leaf_log_prob[:, :, self.root] = 0.0
        root_values = x[:, self.root].unsqueeze(1)
        root_log_prob = (
            root_values * torch.nn.functional.logsigmoid(self.root_logits).unsqueeze(0)
            + (1.0 - root_values) * torch.nn.functional.logsigmoid(-self.root_logits).unsqueeze(0)
        )
        component_log_prob = leaf_log_prob.sum(dim=2) + root_log_prob + torch.log_softmax(self.logits_pi, dim=0)
        return torch.logsumexp(component_log_prob, dim=1)

    @torch.no_grad()
    def pair_marginals(self, *, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
        p_one, p11 = _tree_pair_marginals(
            self.pi, self.root_logits, self.conditional_logits, self.parents
        )
        return p_one.to(dtype=dtype), p11.to(dtype=dtype)

    @torch.no_grad()
    def joint_probabilities(
        self,
        gene_sets: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate arbitrary low-order marginals by batched tree message passing.

        The input contains many different evidence sets.  One upward pass over
        the learned tree evaluates all their configurations for every mixture
        component simultaneously, so selected triples/higher-order candidates
        stay practical without enumerating the 999-gene joint state space.
        """

        if gene_sets.ndim != 2 or outcomes.ndim != 2:
            raise ValueError("gene_sets and outcomes must both be matrices.")
        if gene_sets.shape[1] != outcomes.shape[1]:
            raise ValueError("Gene-set and outcome orders must match.")
        batch_size, order = gene_sets.shape
        n_outcomes = outcomes.shape[0]
        dtype, device = outcomes.dtype, outcomes.device
        parents_np = self.parents.detach().cpu().numpy().astype(np.int64, copy=False)
        root, traversal, children = _tree_order(parents_np)
        if root != self.root:
            raise RuntimeError("Tree root metadata is inconsistent.")

        # likelihood[b, o, gene, state] is one for an unobserved gene and an
        # indicator for observed evidence.  It does not depend on component.
        likelihood = torch.ones(
            (batch_size, n_outcomes, self.num_genes, 2), dtype=dtype, device=device
        )
        batch_index = torch.arange(batch_size, device=device)[:, None]
        outcome_index = torch.arange(n_outcomes, device=device)[None, :]
        for position in range(order):
            evidence_gene = gene_sets[:, position][:, None]
            opposite_value = (1.0 - outcomes[:, position])[None, :].to(torch.long)
            likelihood[batch_index, outcome_index, evidence_gene, opposite_value] = 0.0

        conditional_probability = torch.sigmoid(self.conditional_logits).to(dtype=dtype)
        # [component, child, parent_state, child_state]
        transition = torch.stack((1.0 - conditional_probability, conditional_probability), dim=3)
        messages: dict[int, torch.Tensor] = {}
        for node in reversed(traversal):
            incoming = likelihood[:, :, node, :].unsqueeze(2)
            for child in children[node]:
                incoming = incoming * messages[child]
            if node != root:
                messages[node] = torch.einsum(
                    "bokx,kyx->boky", incoming, transition[:, node, :, :]
                )
            else:
                root_probability = torch.sigmoid(self.root_logits).to(dtype=dtype)
                root_distribution = torch.stack((1.0 - root_probability, root_probability), dim=1)
                component_probability = (incoming * root_distribution[None, None, :, :]).sum(dim=3)
                return (component_probability * self.pi.to(dtype=dtype)[None, None, :]).sum(dim=2)
        raise RuntimeError("Tree message pass did not reach its root.")


class MixtureChowLiuPC(_TreeMixturePC):
    """Mixture of Chow--Liu trees sharing a training-fold topology."""

    def __init__(self, num_genes: int, num_components: int, parents: Sequence[int]) -> None:
        super().__init__(num_genes, num_components, parents)
        self._root_logits = nn.Parameter(torch.zeros(num_components))
        self._conditional_logits = nn.Parameter(torch.zeros(num_components, num_genes, 2))

    @property
    def root_logits(self) -> torch.Tensor:
        return self._root_logits

    @property
    def conditional_logits(self) -> torch.Tensor:
        return self._conditional_logits

    def initialize_from_data(self, data: torch.Tensor, noise_scale: float = 0.20) -> None:
        root_logit, conditional_logits = _empirical_tree_logits(data, self.parents)
        with torch.no_grad():
            self._root_logits.copy_(root_logit + noise_scale * torch.randn_like(self._root_logits))
            self._conditional_logits.copy_(
                conditional_logits.unsqueeze(0) + noise_scale * torch.randn_like(self._conditional_logits)
            )
            self.logits_pi.zero_()


class LatentResidualTreePC(_TreeMixturePC):
    """Latent-class PC with a tied sparse Chow--Liu residual structure.

    Component-specific leaf logits model global cell-state variation.  A pair
    of logits on each learned tree edge supplies a shared residual dependency
    after conditioning on the latent class, yielding a compact alternative to
    fully component-specific mixture trees.
    """

    def __init__(self, num_genes: int, num_components: int, parents: Sequence[int]) -> None:
        super().__init__(num_genes, num_components, parents)
        self.base_logits = nn.Parameter(torch.zeros(num_components, num_genes))
        self.residual_edge_logits = nn.Parameter(torch.zeros(num_genes, 2))

    @property
    def root_logits(self) -> torch.Tensor:
        return self.base_logits[:, self.root]

    @property
    def conditional_logits(self) -> torch.Tensor:
        return self.base_logits.unsqueeze(2) + self.residual_edge_logits.unsqueeze(0)

    def initialize_from_data(self, data: torch.Tensor, noise_scale: float = 0.20) -> None:
        root_logit, conditional_logits = _empirical_tree_logits(data, self.parents)
        marginal = data.float().mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
        base_logits = torch.logit(marginal)
        residual = conditional_logits - base_logits.unsqueeze(1)
        with torch.no_grad():
            self.base_logits.copy_(
                base_logits.unsqueeze(0) + noise_scale * torch.randn_like(self.base_logits)
            )
            self.base_logits[:, self.root].copy_(
                root_logit + noise_scale * torch.randn_like(self.base_logits[:, self.root])
            )
            self.residual_edge_logits.copy_(residual)
            self.residual_edge_logits[self.root].zero_()
            self.logits_pi.zero_()
