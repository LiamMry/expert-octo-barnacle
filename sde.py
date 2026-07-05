"""
Stochastic Differential Equations (SDEs) for diffusion models.

Each SDE defines the forward noising process:
    dx = f(x, t) dt + g(t) dW

and provides the tools needed for score-based generation:
  - marginal distributions p_t(x | x_0)
  - reverse-time SDE coefficients
  - probability flow ODE
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class SDE(ABC):
    """Abstract base class for SDEs used in diffusion models."""

    @property
    @abstractmethod
    def T(self) -> float:
        """Terminal time of the forward SDE."""

    @abstractmethod
    def sde(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Drift f(x, t) and diffusion g(t) of the forward SDE.

        Returns:
            drift:     shape (B, ...) — same as x
            diffusion: shape (B,)     — scalar per sample
        """

    @abstractmethod
    def marginal_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters of the marginal p_t(x | x_0) = N(mean_scale * x_0, std^2 I).

        Returns:
            mean_scale: shape (B,)
            std:        shape (B,)
        """

    def marginal_sample(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ p_t(x | x_0).

        Returns:
            x_t:   noisy sample, shape (B, ...)
            noise: the injected Gaussian noise, shape (B, ...)
        """
        mean_scale, std = self.marginal_params(t)
        # Reshape for broadcasting over spatial/channel dims
        while mean_scale.dim() < x0.dim():
            mean_scale = mean_scale.unsqueeze(-1)
            std = std.unsqueeze(-1)
        noise = torch.randn_like(x0)
        x_t = mean_scale * x0 + std * noise
        return x_t, noise

    def reverse_sde(
        self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Drift and diffusion of the reverse-time SDE (Anderson 1982):
            dx = [f(x,t) - g²(t) ∇log p_t(x)] dt + g(t) dW̄

        Returns:
            drift:     shape (B, ...)
            diffusion: shape (B,)
        """
        drift, diffusion = self.sde(x, t)
        g2 = diffusion**2
        while g2.dim() < score.dim():
            g2 = g2.unsqueeze(-1)
        reverse_drift = drift - g2 * score
        return reverse_drift, diffusion

    def probability_flow_ode(
        self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """
        Drift of the probability flow ODE (deterministic, same marginals):
            dx/dt = f(x,t) - ½ g²(t) ∇log p_t(x)

        Returns:
            drift: shape (B, ...)
        """
        drift, diffusion = self.sde(x, t)
        g2 = diffusion**2
        while g2.dim() < score.dim():
            g2 = g2.unsqueeze(-1)
        return drift - 0.5 * g2 * score


# ---------------------------------------------------------------------------
# Concrete SDE implementations
# ---------------------------------------------------------------------------


class VPSDE(SDE):
    """
    Variance-Preserving SDE (Ho et al., DDPM):
        dx = -½ β(t) x dt + √β(t) dW

    β(t) interpolates linearly from β_min to β_max.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self._T = T
        self.beta_min = beta_min
        self.beta_max = beta_max

    @property
    def T(self) -> float:
        return self._T

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x, t):
        beta_t = self.beta(t)
        while beta_t.dim() < x.dim():
            beta_t = beta_t.unsqueeze(-1)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(self.beta(t))
        return drift, diffusion

    def marginal_params(self, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean_scale = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean_scale, std


class VESDE(SDE):
    """
    Variance-Exploding SDE (Song et al., NCSN):
        dx = σ(t) √(2 d log σ / dt) dW

    σ(t) = σ_min (σ_max/σ_min)^t
    """

    def __init__(
        self, sigma_min: float = 0.01, sigma_max: float = 50.0, T: float = 1.0
    ):
        self._T = T
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def T(self) -> float:
        return self._T

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x, t):
        sigma_t = self.sigma(t)
        log_ratio = np.log(self.sigma_max / self.sigma_min)
        diffusion = sigma_t * torch.sqrt(2.0 * torch.tensor(log_ratio, device=t.device))
        drift = torch.zeros_like(x)
        return drift, diffusion

    def marginal_params(self, t):
        mean_scale = torch.ones_like(t)
        std = self.sigma(t)
        return mean_scale, std


# ---------------------------------------------------------------------------
# ε-parameterised score model
# ---------------------------------------------------------------------------


class EpsilonScoreModel(torch.nn.Module):
    """
    Wraps an ε-prediction network into a score model:

        s_θ(x, t) = -ε_θ(x, t) / σ(t)

    Why: the DSM target score −ε/σ(t) blows up like 1/σ as t → 0 (magnitudes
    ~10³–10⁴ at t = 1e-4), which a raw UNet output cannot track — this was the
    source of the speckle in generated surfaces. Here the UNet regresses the
    injected noise ε ~ N(0, I), an O(1) target at every t, and the exact 1/σ
    blow-up comes analytically from the division.

    Downstream code is unchanged: this module still *is* a score function, so
    the DSM loss, all samplers and the DPS conditioning keep working as-is.

    Note: EMA and checkpointing must keep operating on the wrapped `model`
    (this wrapper holds no parameters of its own), so checkpoint format is
    unchanged.
    """

    def __init__(self, model: torch.nn.Module, sde: SDE):
        super().__init__()
        self.model = model
        self.sde = sde

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        _, std = self.sde.marginal_params(t)
        while std.dim() < x.dim():
            std = std.unsqueeze(-1)
        eps = self.model(x, t) if y is None else self.model(x, t, y)
        return -eps / std


SDE_REGISTRY = {"vp": VPSDE, "ve": VESDE}


def build_sde(name: str, **kwargs) -> SDE:
    if name not in SDE_REGISTRY:
        raise ValueError(f"Unknown SDE '{name}'. Choose from: {list(SDE_REGISTRY)}")
    return SDE_REGISTRY[name](**kwargs)
