"""A normalized, GPU-friendly latent-class probabilistic circuit."""

from __future__ import annotations

import torch
from torch import nn


class LatentClassPC(nn.Module):
    """A root sum node over products of Bernoulli leaves.

    ``p(x) = sum_z pi[z] prod_j Bernoulli(x[j]; theta[z, j])`` is a smooth,
    decomposable probabilistic circuit.  It has O(components * genes)
    parameters, evaluates full data batches efficiently on GPUs, and exposes
    exact pair marginals through a matrix multiplication.
    """

    def __init__(self, num_genes: int, num_components: int) -> None:
        super().__init__()
        if num_genes < 2:
            raise ValueError("At least two genes are required.")
        if num_components < 1:
            raise ValueError("num_components must be positive.")
        self.num_genes = int(num_genes)
        self.num_components = int(num_components)
        self.logits_pi = nn.Parameter(torch.zeros(num_components))
        self.logits_theta = nn.Parameter(torch.zeros(num_components, num_genes))

    @property
    def pi(self) -> torch.Tensor:
        return torch.softmax(self.logits_pi, dim=0)

    @property
    def theta(self) -> torch.Tensor:
        return torch.sigmoid(self.logits_theta)

    def initialize_from_marginals(
        self,
        data: torch.Tensor,
        *,
        noise_scale: float = 0.25,
    ) -> None:
        """Initialize leaves near empirical marginals without zero probabilities."""

        with torch.no_grad():
            marginal = data.float().mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
            base_logits = torch.logit(marginal)
            self.logits_theta.copy_(
                base_logits.unsqueeze(0)
                + noise_scale * torch.randn_like(self.logits_theta)
            )
            self.logits_pi.zero_()

    def log_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Return one log probability per binary observation."""

        if data.ndim != 2 or data.shape[1] != self.num_genes:
            raise ValueError(
                f"Expected [batch, {self.num_genes}] observations, got {tuple(data.shape)}."
            )
        x = data.float()
        log_theta = torch.nn.functional.logsigmoid(self.logits_theta)
        log_not_theta = torch.nn.functional.logsigmoid(-self.logits_theta)
        component_log_prob = (
            x @ log_theta.T + (1.0 - x) @ log_not_theta.T + torch.log_softmax(self.logits_pi, dim=0)
        )
        return torch.logsumexp(component_log_prob, dim=1)

    @torch.no_grad()
    def joint_probabilities(
        self,
        gene_sets: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact probabilities for each outcome of each gene set."""

        theta = self.theta.to(dtype=outcomes.dtype)
        pi = self.pi.to(dtype=outcomes.dtype)
        selected_theta = theta[:, gene_sets]  # [components, sets, order]
        factors = torch.where(
            outcomes[None, None, :, :].bool(),
            selected_theta[:, :, None, :],
            1.0 - selected_theta[:, :, None, :],
        )
        return (pi[:, None, None] * factors.prod(dim=-1)).sum(dim=0)

    @torch.no_grad()
    def pair_marginals(self, *, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
        """Return p(X_i=1) and p(X_i=1, X_j=1) for every i,j.

        Computation occurs in the requested precision.  The cache builder uses
        float64 on CPU so probabilities for very rare configurations retain
        enough precision for the TMLE fluctuation.
        """

        theta = self.theta.detach().to(dtype=dtype)
        pi = self.pi.detach().to(dtype=dtype)
        p_one = pi @ theta
        p11 = (theta * pi.unsqueeze(1)).T @ theta
        return p_one, p11
