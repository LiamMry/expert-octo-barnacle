"""This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n."""

from abc import ABC, abstractmethod

# from functools import partial
# import yaml
# from torch.nn import functional as F
from torchvision import torch
import numpy as np
import scipy.sparse


# =================
# Main physics class
# =================
class Physics:
    def __init__(self, operator: str = "id", noise: str = "none", **kwargs):

        self.operator = get_operator(operator, **kwargs)
        self.noise = get_noise(noise, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):

        y = self.operator(x, **kwargs)
        yn = self.noise(y, **kwargs)

        return yn


# =================
# Operator classes
# =================

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    @abstractmethod
    def forward(self, x, **kwargs):
        # calculate Hx
        pass


@register_operator(name="id")
class IdentityOperator(Operator):
    """This operator returns the input."""

    def __init__(self, **kwargs):
        pass

    def forward(self, x, **kwargs):
        return x


@register_operator(name="inpainting")
class InpaintingOperator(Operator):
    """This operator get pre-defined mask and return masked image."""

    def __init__(self, mask, **kwargs):
        self.mask = mask

    def forward(self, x, **kwargs):
        return x * self.mask.to(x.device)


@register_operator(name="ct")
class CtOperator(Operator):
    def __init__(self, **kwargs):
        self.device = "cuda:0"
        self.H = self.load(kwargs.get("fname")).to(self.device)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    def load(self, fname):

        loaded = np.load(fname)
        data, ir, jc = loaded["data"], loaded["ir"], loaded["jc"]
        shape = tuple(loaded["shape"])

        # Rebuild as scipy sparse if needed
        sparse_matrix = scipy.sparse.csc_matrix((data, ir, jc), shape=shape)

        # Or convert to torch sparse
        coo = sparse_matrix.tocoo()
        del sparse_matrix
        indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.int64)
        values = torch.tensor(coo.data, dtype=torch.float32)

        return torch.sparse_coo_tensor(indices, values, (shape[0] + 8, shape[1])).coalesce()  # fmt: skip

    def forward(self, x, **kwargs):

        # Straight-through estimator:
        # forward sees clamped (physically valid) values,
        # gradient flows as identity through the clamp
        # x_clamped = x.clamp(min=-1)
        # x_ste = x + (x_clamped - x).detach()
        # x_scaled = x_ste * 0.5 + 0.5  # [-1, ∞) → [0, ∞), grad = 0.5 * I
        x = (x + 1) / 2

        # Rescale x from [-1, ?] to [0, 0.5*?+0.5]
        # x_scaled = x.clamp(min=-1) * 0.5 + 0.5

        Hx = torch.sparse.mm(self.H, x.view(x.shape[0], -1).T).T

        return (self.a * Hx.reshape(Hx.shape[0], 1, 360, 182) + self.b).clamp(
            min=self.b
        )

    def transpose(self, x, **kwargs):

        B = x.shape[0]
        x_flat = x.view(B, -1)  # (B, n_meas)
        HTx = torch.sparse.mm(self.H.t(), x_flat.T).T  # (B, n_pixels)
        n_pixels = self.H.shape[1]
        side = int(n_pixels**0.5)
        return HTx.reshape(B, 1, side, side)

    def ramp_filter(self, y: torch.Tensor) -> torch.Tensor:
        """Apply a Ram-Lak ramp filter along the detector dimension (last dim).

        y:       (..., n_det) sinogram tensor
        returns: (..., n_det) filtered sinogram
        """
        n_det = y.shape[-1]
        freqs = torch.fft.rfftfreq(n_det, device=y.device, dtype=y.dtype)
        return torch.fft.irfft(torch.fft.rfft(y, dim=-1) * freqs.abs(), n=n_det, dim=-1)


@register_operator(name="sct")
class StochasticCtOperator(Operator):
    """CT operator with randomly decimated rows.

    tau in [0, 1]: fraction of rows to DROP.
      tau ~ 0 → barely decimated (most rows kept)
      tau ~ 1 → severely decimated (few rows kept)
    """

    def __init__(self, **kwargs):
        self.device = "cuda:0"
        self.mode = kwargs.get("mode")
        self._H_rows, self._H_cols, self._H_vals, self._H_shape, self._H_boundaries = (
            self._load(kwargs.get("fname"))
        )
        # Move COO data to GPU once — avoids a CPU→GPU transfer on every _decimate call
        self._H_rows = self._H_rows.to(self.device)
        self._H_cols = self._H_cols.to(self.device)
        self._H_vals = self._H_vals.to(self.device)
        # _H_boundaries stays on CPU: used only for Python-side .item() index lookups
        self._n_det = 182
        self._n_angles = 360

    def _load(self, fname):
        loaded = np.load(fname)
        data, ir, jc = loaded["data"], loaded["ir"], loaded["jc"]
        shape = tuple(loaded["shape"])
        n_det = 182
        n_angles = 360

        sparse_matrix = scipy.sparse.csc_matrix((data, ir, jc), shape=shape)
        coo = sparse_matrix.tocoo()
        del sparse_matrix

        # Sort COO by row once so _decimate can slice per angle without scanning all nnz
        order = np.argsort(coo.row, kind="stable")
        rows_np = coo.row[order]
        cols_np = coo.col[order]
        vals_np = coo.data[order].astype(np.float32)

        # boundaries[a] = first position in sorted arrays with row >= a * n_det
        boundaries = np.searchsorted(rows_np, np.arange(n_angles + 1) * n_det).astype(
            np.int32
        )

        # In return, 8 rows of zeros are added so that the number of rows is a multiple of n_det

        return (
            torch.from_numpy(rows_np.astype(np.int32)),
            torch.from_numpy(cols_np.astype(np.int32)),
            torch.from_numpy(vals_np),
            (shape[0] + 8, shape[1]),  # fmt: skip
            torch.from_numpy(boundaries),
        )

    def _decimate(self, tau: float, n_det: int = 182):
        """Return a decimated version of H that keeps only a random subset of projection angles.

        H has shape (n_angles * n_det, n_pixels).  Each block of n_det consecutive
        rows corresponds to one projection angle (one full detector readout).
        Decimation drops whole angle-blocks so the resulting operator is self-consistent:
        every kept angle still contributes all n_det detector measurements.

        tau (in [0, 1]) controls the fraction of angles KEPT:
          tau = 0  →  no angles kept  (only 1 kept, the minimum)
          tau = 1  →  all angles kept (full sinogram)

        The n_keep kept angles are chosen uniformly at random (without replacement) and
        sorted so the output rows are in the same angular order as the original H.

        Efficient implementation: at load time the COO arrays are sorted by row and
        per-angle slice boundaries are stored.  _decimate therefore only touches the
        O(nnz_kept) entries it actually needs — no scan of the full matrix.

        Returns:
          Ht          — sparse COO tensor of shape (n_keep * n_det, n_pixels)
          n_keep      — number of angles kept
          kept_angles — 1-D tensor of the kept angle indices (into the original 0..n_angles-1)
        """
        n_angles = self._n_angles
        n_keep = max(1, round(tau * n_angles))
        kept_angles = torch.randperm(n_angles)[:n_keep].sort().values

        # Gather COO slices for kept angles — O(nnz_kept) instead of O(nnz_total)
        chunks_rows, chunks_cols, chunks_vals = [], [], []
        for new_idx, a in enumerate(kept_angles.tolist()):
            start = self._H_boundaries[a].item()
            end = self._H_boundaries[a + 1].item()
            orig_rows = self._H_rows[start:end]
            # Re-index: original row a*n_det+k → new row new_idx*n_det+k
            chunks_rows.append(orig_rows - a * n_det + new_idx * n_det)
            chunks_cols.append(self._H_cols[start:end])
            chunks_vals.append(self._H_vals[start:end])

        new_indices = torch.stack([torch.cat(chunks_rows), torch.cat(chunks_cols)])
        new_values = torch.cat(chunks_vals)
        del chunks_rows, chunks_cols, chunks_vals

        return (
            torch.sparse_coo_tensor(
                new_indices, new_values, (n_keep * n_det, self._H_shape[1])
            ).coalesce(),
            n_keep,
            kept_angles,
        )

    def schedule(self, t: float, mode: str = "linear") -> float:
        """Map diffusion time t to decimation tau.
        Tau controls the quantity of information in the sinogram.
        tau = 1 -> all information,
        tau = 0 -> no information.

        Modes (all satisfy f(0)=1, f(1)=0):
          linear : uniform info introduction throughout sampling
          log    : fast intro early in denoising, slows near the end
          exp    : slow intro early in denoising, fast burst near the end
          sin    : smooth — sin(π/2 · t)
        """
        import math

        assert 0.0 <= t <= 1.0, "t must be in [0, 1]"

        tau = 1 - t

        if mode == "linear":
            return tau
        elif mode == "log":
            return math.log1p(tau) / math.log(2)
        elif mode == "exp":
            return (math.exp(tau) - 1) / (math.e - 1)
        elif mode == "sin":
            return math.sin(math.pi / 2 * tau)
        # Percentage
        elif mode == "10%":
            return 0.1
        elif mode == "25%":
            return 0.25
        elif mode == "33%":
            return 0.33
        elif mode == "50%":
            return 0.5
        elif mode == "66%":
            return 0.66
        elif mode == "75%":
            return 0.75
        elif mode == "100%":
            return 1.0
        else:
            raise ValueError(
                f"Unknown schedule mode '{mode}'. Choose from: linear, log, exp, sin"
            )

    def decimate_observation(
        self, y: torch.Tensor, kept_angles: torch.Tensor, n_det: int = 182
    ) -> torch.Tensor:
        """Decimate a full sinogram y using the same kept_angles as were used to build Ht.

        y:            (B, 1, n_angles, n_det) full sinogram
        kept_angles:  1-D tensor of kept angle indices returned by _decimate
        returns:      (B, 1, n_keep, n_det) decimated sinogram
        """
        B = y.shape[0]
        # View as column vector (B, n_angles * n_det), select kept rows, reshape
        y_vec = y.view(B, -1, n_det)  # (B, n_angles, n_det)
        y_dec = y_vec[:, kept_angles.cpu(), :]  # (B, n_keep, n_det)
        return y_dec.unsqueeze(1)  # (B, 1, n_keep, n_det)

    # def transpose(self, y, kept_angles, **kwargs):
    #     """Apply H_t^T to y, where H_t is the decimated operator for the given kept_angles.

    #     y:            (B, 1, n_keep, n_det) in decimated sinogram space
    #     kept_angles:  1-D tensor of kept angle indices returned by forward
    #     returns:      (B, 1, side, side) in image space
    #     """
    #     B = y.shape[0]
    #     n_det = self._n_det

    #     chunks_rows, chunks_cols, chunks_vals = [], [], []
    #     for new_idx, a in enumerate(kept_angles.tolist()):
    #         start = self._H_boundaries[a].item()
    #         end = self._H_boundaries[a + 1].item()
    #         orig_rows = self._H_rows[start:end]
    #         chunks_rows.append(orig_rows - a * n_det + new_idx * n_det)
    #         chunks_cols.append(self._H_cols[start:end])
    #         chunks_vals.append(self._H_vals[start:end])

    #     new_indices = torch.stack([torch.cat(chunks_rows), torch.cat(chunks_cols)])
    #     new_values = torch.cat(chunks_vals)
    #     n_keep = kept_angles.shape[0]
    #     Ht = torch.sparse_coo_tensor(
    #         new_indices, new_values, (n_keep * n_det, self._H_shape[1])
    #     ).coalesce()

    #     y_flat = y.view(B, -1)  # (B, n_keep * n_det)
    #     HTy = torch.sparse.mm(Ht.t().coalesce(), y_flat.T).T  # (B, n_pixels)
    #     n_pixels = self._H_shape[1]
    #     side = int(n_pixels**0.5)
    #     return HTy.reshape(B, 1, side, side)

    # def fbp(self, y, kept_angles, **kwargs):
    #     """Filtered Back Projection: apply a Ram-Lak ramp filter then back-project.

    #     y:            (B, 1, n_keep, n_det) decimated sinogram
    #     kept_angles:  1-D tensor of kept angle indices returned by forward
    #     returns:      (B, 1, side, side) reconstructed image
    #     """
    #     return self.transpose(ramp_filter(y), kept_angles)

    # def full_forward(self, x):

    #     # Rescale x from [-1, 1] to [0, 1]
    #     x = x.clamp(min=-1) * 0.5 + 0.5
    #     Ht, n_keep, _ = self._decimate(tau=1.0)
    #     Hx = torch.sparse.mm(Ht, x.view(x.shape[0], -1).T).T
    #     return Hx.reshape(x.shape[0], 1, n_keep, 182)

    def forward(self, x, t=torch.tensor(0.5), **kwargs):
        # t may arrive as a batch tensor (B,) — all elements are the same timestep
        t_scalar = float(t.flatten()[0])

        # Rescale x from [-?, ?] to [0, ?]
        x = x.clamp(min=-1) * 0.5 + 0.5
        tau = self.schedule(t=t_scalar, mode=self.mode)
        Ht, n_keep, kept_angles = self._decimate(tau)
        Hx = torch.sparse.mm(Ht, x.view(x.shape[0], -1).T).T
        return Hx.reshape(
            x.shape[0], 1, n_keep, 182
        ), kept_angles  # (B, 1, n_keep_angles, 182)


# =============
# Noise classes
# =============

__NOISE__: dict = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name="none")
class NoNoise(Noise):
    def forward(self, data):
        return data


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    def __init__(self, sigma, **kwargs):
        self.name = "gaussian"
        self.sigma = sigma

    def forward(self, y):
        return y + torch.randn_like(y, device=y.device) * self.sigma


@register_noise(name="poisson")
class PoissonNoise(Noise):
    def __init__(self, **kwargs):
        pass

    def forward(self, y):
        # x: sinogram in [0, ∞) — already positive, no remapping needed
        # rate: photon count scale (higher → less noise)
        return torch.poisson(y)


def main():
    import matplotlib.pyplot as plt
    import time

    sct_op = StochasticCtOperator(fname="/users/lm4057/CT/op/H_f32.npz", mode="100%")
    x = torch.load("/users/lm4057/CT/obs_ct/observation.clean.pt", map_location="cpu")[:1].to(sct_op.device)  # fmt: skip
    y = torch.load("/users/lm4057/CT/obs_ct/observation.pt", map_location="cpu")[:1].to(sct_op.device)  # fmt: skip

    ## Decimation check

    # timesteps = torch.linspace(1, 0, 10).tolist()

    # n_angles_full = 360
    # n_det = 182

    # fig, axes = plt.subplots(2, 10, figsize=(20, 8))
    # t_loop_start = time.perf_counter()
    # for i, t in enumerate(timesteps):
    #     t_start = time.perf_counter()
    #     y, kept_angles = sct_op.forward(x, t=torch.tensor(t))  # (1, 1, n_keep, 182)
    #     elapsed = time.perf_counter() - t_start

    #     # Row 0: decimated sinogram (zero-filled)
    #     img_full = torch.zeros(n_angles_full, n_det)
    #     img_full[kept_angles] = y[0, 0].cpu()
    #     im0 = axes[0, i].imshow(img_full.numpy(), aspect="auto", cmap="gray")
    #     axes[0, i].set_title(
    #         f"t={t:.2f}\n{kept_angles.shape[0]} angles\n{elapsed * 1e3:.1f} ms",
    #         fontsize=8,
    #     )
    #     axes[0, i].axis("off")
    #     plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

    #     # Row 1: FBP reconstruction + PSNR
    #     fbp_img = sct_op.fbp(y, kept_angles)[0, 0].detach().cpu()
    #     fbp_img = (fbp_img - fbp_img.min()) / (
    #         fbp_img.max() - fbp_img.min()
    #     )  # -> [0, 1]
    #     fbp_img = (fbp_img - 0.5) / 0.5
    #     x_clean = x[0, 0].detach().cpu()
    #     # print(x_clean.max(), x_clean.min())
    #     mse = torch.mean((fbp_img - x_clean) ** 2)
    #     psnr = 10 * torch.log10(4 / mse)
    #     im1 = axes[1, i].imshow(fbp_img.numpy(), cmap="gray")
    #     axes[1, i].set_title(f"FBP\nPSNR={psnr:.1f}dB", fontsize=8)
    #     axes[1, i].axis("off")
    #     plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

    #     print(
    #         f"[{i + 1:2d}/10] t={t:.2f}  {kept_angles.shape[0]} angles  {elapsed * 1e3:.1f} ms  PSNR={psnr:.2f}dB"
    #     )
    # print(f"Total loop time: {(time.perf_counter() - t_loop_start) * 1e3:.1f} ms")

    # plt.tight_layout()
    # plt.savefig("/users/lm4057/CT/schedule_grid.png", dpi=100)
    # print("Saved schedule_grid.png")

    ## Adjoint test  <Hx, v> == <x, H^T v>
    # Tests the pure linear map H, without the affine rescaling in forward()
    x = torch.randn(1, 1, 256, 256).to(sct_op.device)
    tau = sct_op.schedule(t=0.5, mode=sct_op.mode)
    Ht_mat, n_keep, kept_angles = sct_op._decimate(tau)
    Hx = torch.sparse.mm(Ht_mat, x.view(1, -1).T).T.reshape(1, 1, n_keep, 182)
    v = torch.randn_like(Hx)

    lhs = (Hx * v).sum()  # <Hx, v>
    HT_v = sct_op.transpose(v, kept_angles)
    rhs = (x * HT_v).sum()  # <x, H^T v>

    print(
        f"Adjoint test — lhs: {lhs.item():.4f}  rhs: {rhs.item():.4f}  diff: {abs(lhs.item() - rhs.item()):.2e}"
    )

    ## Forward / decimate consistency
    # y, kept_angles = sct_op.forward(x, t=torch.tensor(0.0))  # tau=1
    # Hx, kept_angles = sct_op.forward(x, t=torch.tensor(0.0))
    # y_dec = sct_op.decimate_observation(y, kept_angles)

    # print(torch.allclose(y_dec, Hx, atol=1e-5))  # should be True


if __name__ == "__main__":
    main()
