from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from conditioning.physics import GaussianNoise, PoissonNoise, NoNoise
# import numpy as np

__CONDITIONING_METHOD__ = {}

EPS = 1e-7


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, physics, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](physics=physics, **kwargs)


class ConditioningMethod(ABC):
    """
    Abstract base class for plug-and-play conditioning operators.

    Subclasses implement likelihood_score(), which returns an additive
    correction to the prior score at each reverse-diffusion step:

        score_total = score_prior + conditioning.likelihood_score(x_t, t)

    This is the only method samplers call — everything else (measurement
    loading, gradient computation, hyperparameters) lives inside the subclass.
    """

    def __init__(self, physics, **kwargs):
        self.physics = physics

    @abstractmethod
    def likelihood_score(
        self,
        x_t: torch.Tensor,  # current noisy sample  (B, C, ...)
        y: torch.Tensor,  # observation sample    (B, ...)
        t: torch.Tensor,  # batch of timesteps    (B,)
        **kwargs,  # additional args
    ) -> torch.Tensor:
        """
        Returns  ∇_{x_t} log p(y | x_t),  same shape as x_t.

        This is added to the unconditional score before the diffusion update.
        Return zeros_like(x_t) for an identity / no-op operator.
        """


@register_conditioning_method(name="none")
class NoConditioning(ConditioningMethod):
    """Identity operator — returns zero correction (unconditional sampling)."""

    def __init__(self, physics, **kwargs):
        super().__init__(physics)

    def likelihood_score(self, x_t, y, t, **kwargs):
        return torch.zeros_like(x_t)


# ------------------------------------------------------ #
# ---------------- Gaussian conditioning ---------------- #
# ------------------------------------------------------ #


@register_conditioning_method(name="dps")
class DiffusionPosteriorSamplingGaussian(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [DiffusionPosteriorSamplingGaussian] -> [conditioning()] -> 'x0hat' is not set."
        )

        err_sino = y - self.physics.operator.forward(x0hat)  # (B, C, H, W)
        err_img = self.physics.operator.transpose(err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_img)[0]

        return grad / (self.physics.noise.sigma**2)


@register_conditioning_method(name="cdps")
class CalibratedDiffusionPosteriorSamplingGaussian(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)

        self.a_star = kwargs.get("a_star", None)
        assert self.scale is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingGaussian] -> [__init__()] -> 'a_star' is not set."
        )

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingGaussian] -> [conditioning()] -> 'x0hat' is not set."
        )

        idx = kwargs.get("idx", None)
        assert idx is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingGaussian] -> [conditioning()] -> 'idx' is not set."
        )

        err_sino = y - self.physics.operator.forward(x0hat)  # (B, C, H, W)
        err_img = self.physics.operator.transpose(err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_img)[0]

        return self.a_star[idx] * grad / (self.physics.noise.sigma**2)


@register_conditioning_method(name="dps_fbp")
class DiffusionPosteriorSamplingGaussianFBP(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [DiffusionPosteriorSampling] -> [conditioning()] -> 'x0hat' is not set."
        )

        err_sino = y - self.physics.operator.forward(x0hat)  # (B, C, H, W)
        filtered_err_sino = self.physics.operator.ramp_filter(err_sino)
        err_fbp = self.physics.operator.transpose(filtered_err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_fbp)[0]  # fmt: skip

        return grad / (self.physics.noise.sigma**2)


@register_conditioning_method(name="cdps_fbp")
class CalibratedDiffusionPosteriorSamplingGaussianFBP(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)

        self.a_star = kwargs.get("a_star", None)
        assert self.scale is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingGaussianFBP] -> [__init__()] -> 'a_star' is not set."
        )

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingGaussianFBP] -> [conditioning()] -> 'x0hat' is not set."
        )

        idx = kwargs.get("idx", None)
        assert idx is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingGaussianFBP] -> [conditioning()] -> 'idx' is not set."
        )

        err_sino = y - self.physics.operator.forward(x0hat)  # (B, C, H, W)
        filtered_err_sino = self.physics.operator.ramp_filter(err_sino)
        err_fbp = self.physics.operator.transpose(filtered_err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_fbp)[0]  # fmt: skip

        return self.a_star[idx] * grad / (self.physics.noise.sigma**2)


# ------------------------------------------------------ #
# ---------------- Poisson conditioning ---------------- #
# ------------------------------------------------------ #


@register_conditioning_method(name="dps_poisson")
class DiffusionPosteriorSamplingPoisson(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)

        self.a = self.physics.operator.a

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [DiffusionPosteriorSamplingPoisson] -> [conditioning()] -> 'x0hat' is not set."
        )

        err_sino = (y / self.physics.operator.forward(x0hat, **kwargs) - 1)  # fmt: skip
        err_img = 0.5 * self.a * self.physics.operator.transpose(err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_img)[0]  # fmt: skip

        return grad  # fmt: skip


@register_conditioning_method(name="dps_poisson_fbp")
class DiffusionPosteriorSamplingPoissonFBP(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)
        self.a = self.physics.operator.a

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [DiffusionPosteriorSamplingPoisson] -> [conditioning()] -> 'x0hat' is not set."
        )

        err_sino = (y / self.physics.operator.forward(x0hat, **kwargs) - 1)  # fmt: skip
        filtered_err_sino = self.physics.operator.ramp_filter(err_sino)
        err_fbp = 0.5 * self.a * self.physics.operator.transpose(filtered_err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_fbp)[0]  # fmt: skip

        return grad  # fmt: skip


@register_conditioning_method(name="cdps_poisson")
class CalibratedDiffusionPosteriorSamplingPoisson(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)
        self.a = self.physics.operator.a

        self.a_star = kwargs.get("a_star", None)
        assert self.scale is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingPoisson] -> [__init__()] -> 'a_star' is not set."
        )

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingPoisson] -> [conditioning()] -> 'x0hat' is not set."
        )

        idx = kwargs.get("idx", None)
        assert idx is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingPoisson] -> [conditioning()] -> 'idx' is not set."
        )

        err_sino = (y / self.physics.operator.forward(x0hat, **kwargs) - 1)  # fmt: skip
        err_img = 0.5 * self.a * self.physics.operator.transpose(err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_img)[0]  # fmt: skip

        return self.a_star[idx] * grad  # fmt: skip


@register_conditioning_method(name="cdps_poisson_fbp")
class CalibratedDiffusionPosteriorSamplingPoissonFBP(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)
        self.a = self.physics.operator.a

        self.a_star = kwargs.get("a_star", None)
        assert self.scale is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingPoisson] -> [__init__()] -> 'a_star' is not set."
        )

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingPoisson] -> [conditioning()] -> 'x0hat' is not set."
        )

        idx = kwargs.get("idx", None)
        assert idx is not None, (
            "[Assertion Error] -> [CalibratedDiffusionPosteriorSamplingPoisson] -> [conditioning()] -> 'idx' is not set."
        )

        err_sino = (y / self.physics.operator.forward(x0hat, **kwargs) - 1)  # fmt: skip
        filtered_err_sino = self.physics.operator.ramp_filter(err_sino)
        err_fbp = 0.5 * self.a * self.physics.operator.transpose(filtered_err_sino)
        grad = torch.autograd.grad(outputs=x0hat, inputs=x_t, grad_outputs=err_fbp)[0]  # fmt: skip

        return self.a_star[idx] * grad  # fmt: skip


@register_conditioning_method(name="dps_sct")
class DiffusionPosteriorSampling_SCT(ConditioningMethod):
    def __init__(self, physics, **kwargs):
        super().__init__(physics)

        self.scale = kwargs.get("scale", None)
        assert self.scale is not None, (
            "[Assertion Error] -> [DiffusionPosteriorSampling] -> [__init__()] -> 'scale' is not set."
        )

    def likelihood_score(self, x_t, y, t, **kwargs):

        x0hat = kwargs.get("x0hat", None)
        assert x0hat is not None, (
            "[Assertion Error] -> [DiffusionPosteriorSampling] -> [conditioning()] -> 'x0hat' is not set."
        )

        gt2 = kwargs.get("gt2", None)
        assert gt2 is not None, (
            "[Assertion Error] -> [DiffusionPosteriorSampling] -> [conditioning()] -> 'gt2' is not set."
        )

        Htx0hat, kept_angles = self.physics.operator.forward(x0hat, t)
        yt = self.physics.operator.decimate_observation(y, kept_angles)

        # print(Htx0hat.shape, yt.shape, y.shape)

        # plt.figure()
        # plt.imshow(Htx0hat[0, 0, ...].detach().cpu(), cmap="gray")
        # plt.colorbar()
        # plt.savefig("./tmp/Htx0hat.png")

        # plt.figure()
        # plt.imshow(yt[0, 0, ...].detach().cpu(), cmap="gray")
        # plt.colorbar()
        # plt.savefig("./tmp/yt.png")

        # plt.figure()
        # plt.imshow(y[0, 0, ...].detach().cpu(), cmap="gray")
        # plt.colorbar()
        # plt.savefig("./tmp/y.png")

        # plt.figure()
        # plt.imshow(((y - yt) ** 2)[0, 0, ...].detach().cpu(), cmap="gray")
        # plt.colorbar()
        # plt.savefig("./tmp/diff_y.png")

        # exit()

        diff = yt - Htx0hat  # (B, C, kept_angles, n_det)
        L2 = diff.reshape(diff.shape[0], -1).norm(dim=1)  # (B, 1)
        # del Htx0hat, yt, diff
        # torch.cuda.empty_cache()
        grad = torch.autograd.grad(
            outputs=L2, inputs=x_t, grad_outputs=torch.ones_like(L2)
        )[0]  # (B, C, H, W)

        m = y.shape[2]
        k = kept_angles.shape[0]

        # Grad is multiplied by to to cancel the 1/2 factor arising
        # from the autograd of L2 and not L2^2.

        return -(1 / gt2) * self.scale * (m / k) * 2 * grad
