"""
Training losses for score-based diffusion models.

Denoising Score Matching (DSM) loss (Vincent 2011, Song et al. 2020):

    L_DSM = E_{t, x_0, x_t} [ λ(t) ‖ s_θ(x_t, t) − ∇_{x_t} log p_t(x_t|x_0) ‖² ]

where:
    x_t ~ p_t(x_t | x_0) = N(α(t) x_0, σ²(t) I)
    ∇_{x_t} log p_t(x_t | x_0) = −(x_t − α(t) x_0) / σ²(t)
                                 = −ε / σ(t)       with ε ~ N(0,I)

The callable passed in must be score-valued. UPDATED: with the ε-parameterised
setup the raw UNet predicts ε and is wrapped in sde.EpsilonScoreModel
(s_θ = -ε̂/σ(t)) before reaching this loss — so the score-space math below is
unchanged. Note the weighting then simplifies analytically: score-space error
is ‖Δε‖²/σ², so e.g. min_snr → min(1/α², 1/σ²)·‖Δε‖² ≈ the standard ε-MSE.

Weighting strategies:
  - "likelihood": λ(t) = σ²(t)  [standard DSM, uniform importance across t]
  - "snr":        λ(t) = σ²(t) / α²(t)  [signal-to-noise ratio weighting]
  - "ones":       λ(t) = 1       [unweighted, useful for debugging]
"""

import torch
import torch.nn as nn
# from typing import Callable, Optional

from sde import SDE


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def dsm_loss(
    score_net: nn.Module,
    sde: SDE,
    x0: torch.Tensor,
    t: torch.Tensor,
    weighting: str = "likelihood",
    eps_t: float = 1e-5,
) -> torch.Tensor:
    """
    Denoising Score Matching loss.

    The score network predicts the score directly:
        s_θ(x_t, t) ≈ ∇_{x_t} log p_t(x_t)

    The target score from the conditional:
        ∇_{x_t} log p_t(x_t | x_0) = −ε / σ(t)

    Loss:
        λ(t) ‖ s_θ(x_t, t) + ε / σ(t) ‖²

    Args:
        score_net:  score network s_θ(x, t)
        sde:        SDE defining the forward process
        x0:         clean data batch  (B, C, H, W)
        t:          optional pre-sampled times (B,)
        weighting:  one of "likelihood" | "snr" | "ones"
        eps_t:      minimum time to avoid singularity at t=0

    Returns:
        scalar loss averaged over the batch
    """
    # B = x0.shape[0]
    # device = x0.device

    # Forward diffusion: sample x_t ~ p_t(x_t | x_0)
    x_t, noise = sde.marginal_sample(x0, t)
    mean_scale, std = sde.marginal_params(t)
    while std.dim() < noise.dim():
        std = std.unsqueeze(-1)
        mean_scale = mean_scale.unsqueeze(-1)

    target_score = (mean_scale * x0 - x_t) / std**2

    # Network score prediction
    pred_score = score_net(x_t, t)  # (B, C, H, W)

    # Per-sample squared error, summed over spatial/channel dims
    error = (pred_score - target_score) ** 2
    # Reduce over all non-batch dims → (B,)
    error = error.flatten(1).mean(1)

    # Weighting λ(t)
    # mean_scale, std = sde.marginal_params(t)

    if weighting == "likelihood":
        # λ(t) = σ²(t): down-weights high-noise timesteps
        lam = std**2

    elif weighting == "snr":
        # Most widely used weighting strategy
        # λ(t) = (σ/α)²
        lam = (std / (mean_scale + 1e-8)) ** 2

    elif weighting == "min_snr":
        # Efficient Diffusion Training via Min-SNR Weighting Strategy, Hang et. al. 2023
        # λ(t) = min( (σ/α)², 1)
        snr = (std / (mean_scale + 1e-8)) ** 2
        lam = torch.minimum(snr, torch.ones_like(snr))

    elif weighting == "ones":
        lam = torch.ones_like(std)

    else:
        raise ValueError(
            f"Unknown weighting '{weighting}'. Choose: likelihood | snr | min_snr | ones"
        )

    loss = (lam.squeeze() * error).mean()
    return loss


# ---------------------------------------------------------------------------
# Convenience wrapper: callable loss object (useful in training loops)
# ---------------------------------------------------------------------------


class DSMLoss(nn.Module):
    """
    Callable DSM loss module.

    Usage::

        criterion = DSMLoss(sde, weighting="likelihood")
        loss = criterion(score_net, x0)
    """

    def __init__(self, sde: SDE, weighting: str = "likelihood", eps_t: float = 1e-5):
        super().__init__()
        self.sde = sde
        self.weighting = weighting
        self.eps_t = eps_t

    def forward(
        self,
        score_net: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return dsm_loss(
            score_net,
            self.sde,
            x0,
            t=t,
            weighting=self.weighting,
            eps_t=self.eps_t,
        )
