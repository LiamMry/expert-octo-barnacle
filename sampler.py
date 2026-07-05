"""
Modular SDE/ODE samplers for inference.

All samplers share the same interface:

    sampler = EulerMaruyamaSampler(sde, score_net, num_steps=1000)
    samples = sampler.sample(shape, y=None)

    # With conditioning (DPS):
    sampler = EulerMaruyamaSampler(sde, score_net, num_steps=1000,
                                   conditioning='dps', physics=physics, scale=1.0)
    samples = sampler.sample(shape, y=y)

    # Deterministic probability flow ODE:
    sampler = EulerODESampler(sde, score_net, num_steps=1000)
    samples = sampler.sample(shape)

    # 2nd-order DPM-Solver (deterministic, 2 NFE/step):
    sampler = DPMSolver2Sampler(sde, score_net, num_steps=100)
    samples = sampler.sample(shape)
"""

import torch

# import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from sde import SDE
from conditioning.physics import Physics
from conditioning.conditioning import get_conditioning_method

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_grid(
    T: float, num_steps: int, device: torch.device, eps: float = 1e-3
) -> torch.Tensor:
    """Linearly spaced descending times from T to eps."""
    return torch.linspace(T, eps, num_steps + 1, device=device)


__SAMPLER_REGISTRY__ = {}


class Sampler(ABC):
    @abstractmethod
    def sample(self, shape, y=None, score_fn=None) -> torch.Tensor:
        pass


def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER_REGISTRY__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER_REGISTRY__[name] = cls
        return cls

    return wrapper


def build_sampler(name: str, sde: SDE, score_fn: ScoreFn, device, **kwargs) -> Sampler:
    if __SAMPLER_REGISTRY__.get(name, None) is None:
        raise NameError(
            f"Unknown sampler '{name}'. Available: {list(__SAMPLER_REGISTRY__)}"
        )
    return __SAMPLER_REGISTRY__[name](
        sde=sde, score_fn=score_fn, device=device, **kwargs
    )


# ---------------------------------------------------------------------------
# 1. Euler-Maruyama sampler
# ---------------------------------------------------------------------------


@register_sampler(name="euler_maruyama")
class EulerMaruyamaSampler(Sampler):
    """
    First-order Euler-Maruyama discretisation of the reverse SDE:
        x_{t-dt} = x_t - [f(x_t,t) - g²(t) s_θ(x_t,t)] dt + g(t) √|dt| z
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        num_steps: int = 1000,
        eps: float = 1e-4,
        device: str = "cpu",
        physics: Optional[Physics] = None,
        conditioning: Optional[str] = "none",
        **kwargs,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.num_steps = num_steps
        self.eps = eps
        self.device = device

        if conditioning is not None and conditioning != "none":
            if physics is None:
                physics = Physics(operator="id", noise="none")
            self.conditioning = get_conditioning_method(conditioning, physics, **kwargs)
        else:
            self.conditioning = None

    def sample(self, shape, y=None, score_fn=None):
        score_fn = score_fn or self.score_fn
        times = _time_grid(self.sde.T, self.num_steps, self.device, self.eps)
        x = torch.randn(shape, device=self.device)

        # import time

        ctx = torch.enable_grad() if self.conditioning is not None else torch.no_grad()
        with ctx:
            for i in tqdm(range(self.num_steps)):
                t_cur = times[i]
                t_next = times[i + 1]
                dt = t_next - t_cur  # negative (reverse time)

                t_batch = t_cur.expand(shape[0])

                if self.conditioning is not None:
                    x = x.detach().requires_grad_(True)

                # --- Prior score ---
                prior_score = score_fn(x, t_batch)

                # --- Get reverse SDE  ---
                drift, diffusion = self.sde.reverse_sde(x, t_batch, prior_score)
                g = diffusion
                while g.dim() < x.dim():
                    g = g.unsqueeze(-1)

                if self.conditioning is not None:
                    alpha_t, std_t = self.sde.marginal_params(t_batch)
                    while alpha_t.dim() < x.dim():
                        alpha_t = alpha_t.unsqueeze(-1)
                        std_t = std_t.unsqueeze(-1)
                    x0hat = ((x + std_t**2 * prior_score) / (alpha_t + 1e-8)).clamp(
                        -1, 1
                    )

                    likelihood_score = self.conditioning.likelihood_score(
                        x, y, t_batch, x0hat=x0hat, idx=self.num_steps - 1 - i
                    )

                z = torch.randn_like(x)
                x = (x + drift * dt + g * torch.sqrt(torch.abs(dt)) * z).detach()

                if self.conditioning is not None:
                    x = x - (g**2 * likelihood_score * dt).detach()

        return x


# ---------------------------------------------------------------------------
# 2. Probability Flow ODE sampler (Euler)
# ---------------------------------------------------------------------------


@register_sampler(name="euler_ode")
class EulerODESampler(Sampler):
    """
    Euler discretisation of the probability flow ODE (Song et al. 2021):
        dx/dt = f(x,t) - ½ g²(t) s_θ(x_t,t)

    Deterministic — no noise injection.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        num_steps: int = 500,  # recommanded : start at 500 and reduce
        eps: float = 1e-4,
        device: str = "cpu",
        physics: Optional[Physics] = None,
        conditioning: Optional[str] = "none",
        **kwargs,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.num_steps = num_steps
        self.eps = eps
        self.device = device

        if conditioning is not None and conditioning != "none":
            if physics is None:
                physics = Physics(operator="id", noise="none")
            self.conditioning = get_conditioning_method(conditioning, physics, **kwargs)
        else:
            self.conditioning = None

    def sample(self, shape, y=None, score_fn=None):
        score_fn = score_fn or self.score_fn
        times = _time_grid(self.sde.T, self.num_steps, self.device, self.eps)
        x = torch.randn(shape, device=self.device)

        ctx = torch.enable_grad() if self.conditioning is not None else torch.no_grad()
        with ctx:
            for i in tqdm(range(self.num_steps)):
                t_cur = times[i]
                t_next = times[i + 1]
                dt = t_next - t_cur  # negative (reverse time)

                t_batch = t_cur.expand(shape[0])

                if self.conditioning is not None:
                    x = x.detach().requires_grad_(True)

                prior_score = score_fn(x, t_batch)

                # Probability flow ODE drift: f(x,t) - ½ g²(t) s_θ
                ode_drift = self.sde.probability_flow_ode(x, t_batch, prior_score)

                if self.conditioning is not None:
                    alpha_t, std_t = self.sde.marginal_params(t_batch)
                    while alpha_t.dim() < x.dim():
                        alpha_t = alpha_t.unsqueeze(-1)
                        std_t = std_t.unsqueeze(-1)
                    x0hat = ((x + std_t**2 * prior_score) / (alpha_t + 1e-8)).clamp(
                        -1, 1
                    )

                    _, diffusion = self.sde.sde(x, t_batch)
                    g = diffusion
                    while g.dim() < x.dim():
                        g = g.unsqueeze(-1)

                    likelihood_score = self.conditioning.likelihood_score(
                        x, y, t_batch, x0hat=x0hat, gt2=0.5 * g**2
                    )
                    ode_drift = ode_drift - 0.5 * g**2 * likelihood_score

                x = (x + ode_drift * dt).detach()

        return x


# ---------------------------------------------------------------------------
# 3. Heun ODE sampler (2nd-order Runge-Kutta / trapezoidal rule)
# ---------------------------------------------------------------------------


@register_sampler(name="heun_ode")
class HeunODESampler(Sampler):
    """
    2nd-order Heun (improved Euler) discretisation of the probability flow ODE.

    Two NFE per step:
        d_i   = f(x_i,   t_i)               [slope at current point]
        x̃     = x_i + dt · d_i              [Euler predictor]
        d̃     = f(x̃,    t_{i+1})            [slope at predicted point]
        x_{i+1} = x_i + dt/2 · (d_i + d̃)  [trapezoidal corrector]

    Halve num_steps vs EulerODESampler for the same NFE budget.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        num_steps: int = 250,
        eps: float = 1e-3,
        device: str = "cpu",
        physics: Optional[Physics] = None,
        conditioning: Optional[str] = "none",
        **kwargs,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.num_steps = num_steps
        self.eps = eps
        self.device = device

        if conditioning is not None and conditioning != "none":
            if physics is None:
                physics = Physics(operator="id", noise="none")
            self.conditioning = get_conditioning_method(conditioning, physics, **kwargs)
        else:
            self.conditioning = None

    def sample(self, shape, y=None, score_fn=None):
        score_fn = score_fn or self.score_fn
        times = _time_grid(self.sde.T, self.num_steps, self.device, self.eps)
        x = torch.randn(shape, device=self.device)

        ctx = torch.enable_grad() if self.conditioning is not None else torch.no_grad()
        with ctx:
            for i in tqdm(range(self.num_steps)):
                t_cur = times[i]
                t_next = times[i + 1]
                dt = t_next - t_cur  # negative (reverse time)

                t_b = t_cur.expand(shape[0])
                t_next_b = t_next.expand(shape[0])

                if self.conditioning is not None:
                    x = x.detach().requires_grad_(True)

                # --- Slope at current point ---
                prior_score_i = score_fn(x, t_b)
                d_i = self.sde.probability_flow_ode(x, t_b, prior_score_i)

                # --- Euler predictor ---
                x_pred = (x + d_i * dt).detach()

                # --- DPS nudge on predictor (guides corrector slope) ---
                if self.conditioning is not None:
                    alpha_t, std_t = self.sde.marginal_params(t_b)
                    while alpha_t.dim() < x_pred.dim():
                        alpha_t = alpha_t.unsqueeze(-1)
                        std_t = std_t.unsqueeze(-1)
                    _, diffusion = self.sde.sde(x, t_b)
                    g = diffusion
                    while g.dim() < x_pred.dim():
                        g = g.unsqueeze(-1)
                    x0hat = ((x + std_t**2 * prior_score_i) / (alpha_t + 1e-8)).clamp(
                        -1, 1
                    )
                    likelihood_score = self.conditioning.likelihood_score(
                        x, y, t_b, x0hat=x0hat, gt2=0.5 * g**2
                    )
                    x_pred = (
                        x_pred - 0.5 * g**2 * likelihood_score.detach() * dt
                    ).detach()

                # --- Slope at guided predictor ---
                prior_score_pred = score_fn(x_pred, t_next_b)
                d_pred = self.sde.probability_flow_ode(
                    x_pred, t_next_b, prior_score_pred
                )

                # --- Trapezoidal corrector ---
                x = (x.detach() + 0.5 * dt * (d_i.detach() + d_pred)).detach()

        return x


# ---------------------------------------------------------------------------
# 4. RK3 ODE sampler (3rd-order classical Runge-Kutta)
# ---------------------------------------------------------------------------


@register_sampler(name="rk3_ode")
class RK3ODESampler(Sampler):
    """
    Classical 3rd-order Runge-Kutta discretisation of the probability flow ODE.

    Three NFE per step (Simpson's-rule weights 1/6, 4/6, 1/6):
        k1 = f(x_i,                   t_i       )
        k2 = f(x_i + dt/2 · k1,       t_i + dt/2)
        k3 = f(x_i − dt·k1 + 2dt·k2,  t_i + dt  )
        x_{i+1} = x_i + dt/6 · (k1 + 4·k2 + k3)

    Use num_steps ≈ N/3 vs EulerODESampler for the same NFE budget.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        num_steps: int = 166,
        eps: float = 1e-3,
        device: str = "cpu",
        physics: Optional[Physics] = None,
        conditioning: Optional[str] = "none",
        **kwargs,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.num_steps = num_steps
        self.eps = eps
        self.device = device

        if conditioning is not None and conditioning != "none":
            if physics is None:
                physics = Physics(operator="id", noise="none")
            self.conditioning = get_conditioning_method(conditioning, physics, **kwargs)
        else:
            self.conditioning = None

    def sample(self, shape, y=None, score_fn=None):
        score_fn = score_fn or self.score_fn
        times = _time_grid(self.sde.T, self.num_steps, self.device, self.eps)
        x = torch.randn(shape, device=self.device)

        ctx = torch.enable_grad() if self.conditioning is not None else torch.no_grad()
        with ctx:
            for i in tqdm(range(self.num_steps)):
                t_cur = times[i]
                t_next = times[i + 1]
                dt = t_next - t_cur  # negative (reverse time)
                t_mid = t_cur + 0.5 * dt

                t_b = t_cur.expand(shape[0])
                t_mid_b = t_mid.expand(shape[0])
                t_next_b = t_next.expand(shape[0])

                if self.conditioning is not None:
                    x = x.detach().requires_grad_(True)

                # --- k1: slope at current point (score reused for conditioning) ---
                prior_score_i = score_fn(x, t_b)
                k1 = self.sde.probability_flow_ode(x, t_b, prior_score_i)

                # --- k2: slope at midpoint (Euler half-step with k1) ---
                x_mid = x.detach() + 0.5 * dt * k1.detach()
                k2 = self.sde.probability_flow_ode(
                    x_mid, t_mid_b, score_fn(x_mid, t_mid_b)
                )

                # --- k3: slope at endpoint (extrapolated with k1, k2) ---
                x_end = x.detach() - dt * k1.detach() + 2.0 * dt * k2.detach()
                k3 = self.sde.probability_flow_ode(
                    x_end, t_next_b, score_fn(x_end, t_next_b)
                )

                # --- Simpson's-rule weighted update ---
                x = (
                    x.detach()
                    + (dt / 6.0) * (k1.detach() + 4.0 * k2.detach() + k3.detach())
                ).detach()

                # --- DPS conditioning (reuses prior_score_i from k1, no extra NFE) ---
                if self.conditioning is not None:
                    alpha_t, std_t = self.sde.marginal_params(t_b)
                    while alpha_t.dim() < x.dim():
                        alpha_t = alpha_t.unsqueeze(-1)
                        std_t = std_t.unsqueeze(-1)
                    _, diffusion = self.sde.sde(x, t_b)
                    g = diffusion
                    while g.dim() < x.dim():
                        g = g.unsqueeze(-1)
                    x0hat = ((x + std_t**2 * prior_score_i) / (alpha_t + 1e-8)).clamp(
                        -1, 1
                    )
                    likelihood_score = self.conditioning.likelihood_score(
                        x, y, t_b, x0hat=x0hat, gt2=0.5 * g**2
                    )
                    x = (x - 0.5 * g**2 * likelihood_score * dt).detach()

        return x


# ---------------------------------------------------------------------------
# 5. DPM-Solver-2 (single-step, 2nd order)
# ---------------------------------------------------------------------------


@register_sampler(name="dpm_solver_2")
class DPMSolver2Sampler(Sampler):
    """
    2nd-order single-step DPM-Solver (Lu et al., NeurIPS 2022).

    Two NFE per step via a predictor at the log-SNR midpoint λ_s = λ_i + h/2:

        ε_i     = −σ_i · s_θ(x_i, t_i)
        x_s     = (α_s/α_i) x_i − σ_s (e^{h/2}−1) ε_i          [predictor]
        ε_s     = −σ_s · s_θ(x_s, t_s)
        x_{i+1} = (α_{i+1}/α_i) x_i − σ_{i+1} (e^h−1) ε_s      [corrector]

    where h = λ_{i+1} − λ_i > 0,  λ_t = log(α_t / σ_t),
    and t_s is found by bisecting λ(t) = λ_i + h/2.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        num_steps: int = 100,
        eps: float = 1e-4,
        device: str = "cpu",
        physics: Optional[Physics] = None,
        conditioning: Optional[str] = "none",
        **kwargs,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.num_steps = num_steps
        self.eps = eps
        self.device = device

        if conditioning is not None and conditioning != "none":
            if physics is None:
                physics = Physics(operator="id", noise="none")
            self.conditioning = get_conditioning_method(conditioning, physics, **kwargs)
        else:
            self.conditioning = None

    def _lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Scalar log-SNR: log(α_t / σ_t) for scalar t."""
        alpha, sigma = self.sde.marginal_params(t.unsqueeze(0))
        return torch.log(alpha.squeeze() / (sigma.squeeze() + 1e-8))

    def _t_from_lambda(
        self,
        lam_target: torch.Tensor,
        t_lo: torch.Tensor,
        t_hi: torch.Tensor,
        n_iter: int = 30,
    ) -> torch.Tensor:
        """Bisection: find t ∈ [t_lo, t_hi] s.t. λ(t) = lam_target.
        λ is monotone decreasing in t, so λ(t_lo) > λ(t_hi)."""
        for _ in range(n_iter):
            t_mid = 0.5 * (t_lo + t_hi)
            if self._lambda(t_mid) > lam_target:
                t_lo = t_mid  # λ too high → t too small → push lower bound up
            else:
                t_hi = t_mid
        return 0.5 * (t_lo + t_hi)

    def sample(self, shape, y=None, score_fn=None):
        score_fn = score_fn or self.score_fn
        times = _time_grid(self.sde.T, self.num_steps, self.device, self.eps)
        x = torch.randn(shape, device=self.device)
        ndim = x.dim()

        def bcast(v: torch.Tensor) -> torch.Tensor:
            while v.dim() < ndim:
                v = v.unsqueeze(-1)
            return v

        # Precompute all midpoint times (bisection is sequential Python — do it once)
        midpoint_times = []
        with torch.no_grad():
            for i in range(self.num_steps):
                lam_i = self._lambda(times[i])
                lam_next = self._lambda(times[i + 1])
                t_s = self._t_from_lambda(
                    lam_i + 0.5 * (lam_next - lam_i), times[i + 1], times[i]
                )
                midpoint_times.append(t_s)

        ctx = torch.enable_grad() if self.conditioning is not None else torch.no_grad()
        with ctx:
            for i in tqdm(range(self.num_steps)):
                t_cur = times[i]
                t_next = times[i + 1]

                t_b = t_cur.expand(shape[0])
                t_next_b = t_next.expand(shape[0])

                alpha_i, sigma_i = self.sde.marginal_params(t_b)
                alpha_next, sigma_next = self.sde.marginal_params(t_next_b)
                alpha_i, sigma_i = bcast(alpha_i), bcast(sigma_i)
                alpha_next, sigma_next = bcast(alpha_next), bcast(sigma_next)

                lam_i = self._lambda(t_cur)
                lam_next = self._lambda(t_next)
                h = lam_next - lam_i

                t_s = midpoint_times[i]
                t_s_b = t_s.expand(shape[0])
                alpha_s, sigma_s = self.sde.marginal_params(t_s_b)
                alpha_s, sigma_s = bcast(alpha_s), bcast(sigma_s)

                # --- Score and noise prediction at t_cur ---
                x_i = x.detach().requires_grad_(True)
                prior_score_i = score_fn(x_i, t_b)
                eps_i = -sigma_i * prior_score_i

                # --- Predictor: x_i → x_s (DPM-Solver-1 half-step in λ-space) ---
                x_s = (alpha_s / alpha_i) * x_i - sigma_s * (
                    torch.exp(0.5 * h) - 1.0
                ) * eps_i
                x_s = x_s.detach()

                # --- Score and noise prediction at midpoint t_s ---
                prior_score_s = score_fn(x_s, t_s_b)
                eps_s = (-sigma_s * prior_score_s).detach()

                # --- Corrector: x_i → x_{i+1} using midpoint noise estimate ---
                x_next = (alpha_next / alpha_i) * x_i.detach() - sigma_next * (
                    torch.exp(h) - 1.0
                ) * eps_s

                # --- DPS conditioning (from x_i state, applied post-step) ---
                if self.conditioning is not None:
                    x0hat = (
                        (x_i + sigma_i**2 * prior_score_i) / (alpha_i + 1e-8)
                    ).clamp(-1, 1)
                    _, diffusion = self.sde.sde(x_i.detach(), t_b)
                    g = bcast(diffusion)
                    likelihood_score = self.conditioning.likelihood_score(
                        x_i, y, t_b, x0hat=x0hat, gt2=g**2
                    )
                    dt = t_next - t_cur  # negative
                    x_next = x_next.detach() - (
                        0.5 * g**2 * likelihood_score.detach() * dt
                    )

                x = x_next.detach()

        return x


# ---------------------------------------------------------------------------
# 4. Ancestral (DDPM-style) sampler
# ---------------------------------------------------------------------------


# @register_sampler(name="ancestral")
# class AncestralSampler(Sampler):
#     """
#     Discrete ancestral sampling (Ho et al., DDPM 2020).
#         x̂_0 = (x_t + σ²(t) s_θ(x_t,t)) / α(t)     [Tweedie]
#         x_{t-1} ~ q(x_{t-1} | x_t, x̂_0)             [posterior]
#     """

#     def __init__(
#         self,
#         sde: SDE,
#         score_fn: ScoreFn,
#         num_steps: int = 1000,
#         eps: float = 1e-3,
#         device: str = "cpu",
#         physics: Optional[Physics] = Physics(operator="id", noise="none"),
#         conditioning: Optional[str] = "none",
#         **kwargs,
#     ):
#         self.sde = sde
#         self.score_fn = score_fn
#         self.num_steps = num_steps
#         self.eps = eps
#         self.device = device
#         self.conditioning = get_conditioning_method(conditioning, physics, **kwargs)

#     def sample(self, shape, y=None, score_fn=None):
#         score_fn = score_fn or self.score_fn
#         times = _time_grid(self.sde.T, self.num_steps, self.device, self.eps)
#         x = torch.randn(shape, device=self.device)

#         for i in tqdm(range(self.num_steps)):
#             t_cur = times[i]
#             t_next = times[i + 1]

#             t_batch = t_cur.expand(shape[0])
#             t_next_batch = t_next.expand(shape[0])

#             alpha_t, std_t = self.sde.marginal_params(t_batch)
#             alpha_s, std_s = self.sde.marginal_params(t_next_batch)

#             while alpha_t.dim() < x.dim():
#                 alpha_t = alpha_t.unsqueeze(-1)
#                 std_t = std_t.unsqueeze(-1)
#                 alpha_s = alpha_s.unsqueeze(-1)
#                 std_s = std_s.unsqueeze(-1)

#             x = x.detach().requires_grad_(True)

#             # --- Prior score ---
#             prior_score = score_fn(x, t_batch)

#             # --- Tweedie's estimator ---
#             x0hat = (x + std_t**2 * prior_score) / (alpha_t + 1e-8)
#             x0hat = x0hat.clamp(0, 1)

#             # --- g²(t) for DPS coeff ---
#             _, diffusion = self.sde.sde(x, t_batch)
#             gt2 = (diffusion**2).mean().item()

#             # --- Likelihood score (zeros for unconditional) ---
#             likelihood_score = self.conditioning.likelihood_score(
#                 x, y, t_batch, x0hat=x0hat, gt2=gt2
#             )

#             score = prior_score + likelihood_score

#             # --- DDPM posterior update ---
#             # Re-derive x_0_hat from total score for the posterior mean
#             x_0_hat = (x + std_t**2 * score) / (alpha_t + 1e-8)
#             x_0_hat = x_0_hat.clamp(0, 1)

#             r = alpha_t / (alpha_s + 1e-8)
#             beta_tilde = std_s**2 * (1.0 - r**2 * std_t**2 / (std_s**2 + 1e-8))
#             beta_tilde = beta_tilde.clamp(min=0)
#             mu = alpha_s * x_0_hat + r * (x - alpha_t * x_0_hat)

#             z = torch.randn_like(x) if i < self.num_steps - 1 else torch.zeros_like(x)
#             x = (mu + torch.sqrt(beta_tilde) * z).detach()

#         return x
