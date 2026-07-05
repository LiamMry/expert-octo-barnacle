"""
eval.py — Image quality evaluation for generative imaging models.

Image-to-image metrics (paired, require ground-truth):
    psnr, ssim, rmse, mae, lpips

Distribution-level metrics (unpaired, compare two image sets):
    fid, kid, mmd, cmmd, inception_score

Feature extraction:
    extract_inception_features   — 2048-d InceptionV3 pool3 features

Convenience wrappers:
    evaluate_paired        — run all paired metrics at once
    evaluate_distribution  — run all distribution metrics at once

Conventions
-----------
- Tensors are (N, C, H, W) PyTorch float32.
- Paired metrics accept images in any consistent range; pass data_range
  accordingly (1.0 for [0,1], 2.0 for [-1,1]).
- Distribution metrics internally normalise to [-1,1] / [0,1] as needed.
- Grayscale (C=1) is automatically expanded to 3 channels for models that
  require RGB (InceptionV3, CLIP, LPIPS).

Optional dependencies
---------------------
- lpips   : pip install lpips
- open_clip: pip install open-clip-torch   (required for cmmd)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from conditioning.physics import Physics

# =============================================================================
# IMAGE-TO-IMAGE METRICS (paired)
# =============================================================================


def psnr(pred: Tensor, target: Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Args:
        pred       : (N, C, H, W) reconstructed images
        target     : (N, C, H, W) ground-truth images
        data_range : max signal value — 1.0 for [0,1], 2.0 for [-1,1]

    Returns:
        Mean PSNR in dB across the batch (higher is better).
    """

    data_range = pred.max() - pred.min()
    with torch.no_grad():
        mse_per_image = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        psnr_vals = 10.0 * torch.log10(data_range ** 2 / (mse_per_image + 1e-10))  # fmt: skip
    return psnr_vals.mean().item()


def ssim(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Structural Similarity Index Measure (Wang et al., 2004).

    Args:
        pred        : (N, C, H, W) reconstructed images
        target      : (N, C, H, W) ground-truth images
        data_range  : dynamic range (1.0 or 2.0)
        window_size : Gaussian window side length (must be odd)
        sigma       : standard deviation of the Gaussian window

    Returns:
        Mean SSIM in [−1, 1] across the batch (higher is better).
    """

    data_range = pred.max() - pred.min()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Build 2-D Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)  # (W, W)
    C_in = pred.shape[1]
    kernel = (
        kernel_2d.unsqueeze(0).unsqueeze(0).expand(C_in, 1, window_size, window_size)
    )
    pad = window_size // 2

    with torch.no_grad():
        mu_x = F.conv2d(pred, kernel, padding=pad, groups=C_in)
        mu_y = F.conv2d(target, kernel, padding=pad, groups=C_in)

        mu_xx = mu_x**2
        mu_yy = mu_y**2
        mu_xy = mu_x * mu_y

        sig_xx = F.conv2d(pred * pred, kernel, padding=pad, groups=C_in) - mu_xx
        sig_yy = F.conv2d(target * target, kernel, padding=pad, groups=C_in) - mu_yy
        sig_xy = F.conv2d(pred * target, kernel, padding=pad, groups=C_in) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
            (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2) + 1e-10
        )
    return ssim_map.mean().item()


def rmse(pred: Tensor, target: Tensor) -> float:
    """Root Mean Squared Error (lower is better)."""
    with torch.no_grad():
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred: Tensor, target: Tensor) -> float:
    """Mean Absolute Error (lower is better)."""
    with torch.no_grad():
        return torch.mean(torch.abs(pred - target)).item()


# =============================================================================
# DISTRIBUTION-LEVEL METRICS (unpaired)
# =============================================================================


def extract_inception_features(
    imgs: Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract 2048-d InceptionV3 pool3 features.

    Args:
        imgs       : (N, C, H, W) images in [-1, 1]; grayscale → 3-channel
        device     : target device (defaults to imgs.device)
        batch_size : images per forward pass

    Returns:
        (N, 2048) float32 numpy array.
    """
    if device is None:
        device = imgs.device
    model = _get_inception_pool(device)
    return _inception_pool_forward(imgs, model, device, batch_size)


def fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    Fréchet Inception Distance (Heusel et al., 2017).

    Args:
        real_feats : (N, D) InceptionV3 features from real images
        fake_feats : (M, D) InceptionV3 features from generated images

    Returns:
        FID score (lower is better).
    """
    from scipy.linalg import sqrtm

    mu_r = real_feats.mean(0)
    mu_f = fake_feats.mean(0)
    sig_r = np.cov(real_feats, rowvar=False)
    sig_f = np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sig_r + sig_f - 2.0 * covmean))


def kid(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    n_subsets: int = 100,
    subset_size: int = 1000,
) -> tuple[float, float]:
    """
    Kernel Inception Distance (Bińkowski et al., 2018).

    Uses an unbiased polynomial kernel MMD estimator averaged over random subsets.

    Args:
        real_feats  : (N, D) InceptionV3 features from real images
        fake_feats  : (M, D) InceptionV3 features from generated images
        n_subsets   : number of random subsets for variance estimation
        subset_size : size of each subset (capped at min(N, M))

    Returns:
        (mean_KID, std_KID) — lower mean is better.
    """
    subset_size = min(subset_size, len(real_feats), len(fake_feats))
    scores = []
    for _ in range(n_subsets):
        idx_r = np.random.choice(len(real_feats), subset_size, replace=False)
        idx_f = np.random.choice(len(fake_feats), subset_size, replace=False)
        r = real_feats[idx_r].astype(np.float64)
        f = fake_feats[idx_f].astype(np.float64)
        scores.append(_polynomial_mmd(r, f))
    return float(np.mean(scores)), float(np.std(scores))


def mmd(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> float:
    """
    Maximum Mean Discrepancy with an RBF (Gaussian) or linear kernel.

    Args:
        real_feats : (N, D) features from real images
        fake_feats : (M, D) features from generated images
        kernel     : "rbf" (default) | "linear"
        gamma      : RBF bandwidth; if None, uses the median heuristic

    Returns:
        MMD² estimate (lower is better).
    """
    X = torch.tensor(real_feats, dtype=torch.float32)
    Y = torch.tensor(fake_feats, dtype=torch.float32)

    if kernel == "linear":
        k_xx = (X @ X.T).mean()
        k_yy = (Y @ Y.T).mean()
        k_xy = (X @ Y.T).mean()
    else:
        if gamma is None:
            # Median heuristic on a capped sub-sample
            n = min(500, len(X), len(Y))
            Z = torch.cat([X[:n], Y[:n]], dim=0)
            dists = torch.cdist(Z, Z, p=2)
            median_dist = dists.median().item()
            gamma = 1.0 / (2.0 * median_dist**2 + 1e-10)
        k_xx = _rbf_kernel(X, X, gamma).mean()
        k_yy = _rbf_kernel(Y, Y, gamma).mean()
        k_xy = _rbf_kernel(X, Y, gamma).mean()

    return float(k_xx + k_yy - 2.0 * k_xy)


def cmmd(
    real_imgs: Tensor,
    fake_imgs: Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
) -> float:
    """
    CLIP Maximum Mean Discrepancy (Jayasumana et al., 2024).

    Replaces Inception features with CLIP ViT embeddings and computes
    an RBF-kernel MMD. More semantically sensitive than FID/KID.

    Requires ``pip install open-clip-torch``.

    Args:
        real_imgs      : (N, C, H, W) in [-1, 1]; grayscale → 3-channel
        fake_imgs      : (M, C, H, W)
        device         : target device (defaults to real_imgs.device)
        batch_size     : images per CLIP forward pass
        clip_model     : OpenCLIP model tag (e.g. "ViT-B-32", "ViT-L-14")
        clip_pretrained: pretrained weights tag (e.g. "openai", "laion2b_s34b_b79k")

    Returns:
        CMMD score (lower is better).
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("Install open_clip: pip install open-clip-torch")

    if device is None:
        device = real_imgs.device

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained
    )
    model = model.to(device).eval()

    def _extract_clip(imgs: Tensor) -> np.ndarray:
        imgs_3ch = _to_3ch_01(imgs)
        # Resize to 224×224 as expected by standard CLIP
        if imgs_3ch.shape[-1] != 224 or imgs_3ch.shape[-2] != 224:
            imgs_3ch = F.interpolate(
                imgs_3ch, size=(224, 224), mode="bilinear", align_corners=False
            )
        feats = []
        for i in range(0, len(imgs_3ch), batch_size):
            batch = imgs_3ch[i : i + batch_size].to(device)
            with torch.no_grad():
                f = model.encode_image(batch)
                f = F.normalize(f, dim=-1)
            feats.append(f.cpu().float())
        return torch.cat(feats, dim=0).numpy()

    r = _extract_clip(real_imgs)
    f = _extract_clip(fake_imgs)
    return mmd(r, f, kernel="rbf")


def inception_score(
    imgs: Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    splits: int = 10,
) -> tuple[float, float]:
    """
    Inception Score (Salimans et al., 2016).

    Measures both quality (low entropy per-image) and diversity
    (high entropy of the marginal).

    Args:
        imgs       : (N, C, H, W) generated images in [-1, 1]
        device     : target device (defaults to imgs.device)
        batch_size : images per Inception forward pass
        splits     : number of splits for mean/std estimation

    Returns:
        (mean_IS, std_IS) — higher mean is better.
    """
    if device is None:
        device = imgs.device

    model = _get_inception_cls(device)
    preds = _inception_cls_forward(imgs, model, device, batch_size)  # (N, 1000)

    N = len(preds)
    chunk = N // splits
    scores = []
    for k in range(splits):
        part = preds[k * chunk : (k + 1) * chunk]
        p_y = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores)), float(np.std(scores))


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================


def to_01(t: torch.Tensor) -> torch.Tensor:
    """Map model output (≈ [-1,1]) to [0,1]."""
    return t.clamp(-1.0, 1.0).add(1.0).div(2.0)


def compute_metrics(recon: torch.Tensor, label: torch.Tensor, device) -> dict:
    """PSNR, SSIM, MAE (paired, data_range=1.0) + pixel-space MMD."""
    metrics = evaluate_paired(recon, label, compute_lpips=False, device=device)
    r_flat = label.view(label.shape[0], -1).numpy()
    f_flat = recon.view(recon.shape[0], -1).numpy()
    metrics["mmd"] = mmd(r_flat, f_flat, kernel="linear")
    return metrics


def evaluate_paired(
    pred: Tensor,
    target: Tensor,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compute all paired image-to-image metrics in one call.

    Args:
        pred          : (N, C, H, W) reconstructed / generated images
        target        : (N, C, H, W) ground-truth images
        data_range    : signal range (1.0 for [0,1], 2.0 for [-1,1])
        compute_lpips : set False to skip LPIPS (requires the lpips package)
        device        : device for LPIPS computation

    Returns:
        dict with keys: psnr, ssim, rmse, mae, [lpips]
    """
    results: dict = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
    }
    return results


def evaluate_distribution(
    real_imgs: Tensor,
    fake_imgs: Tensor,
    device: Optional[torch.device] = None,
    compute_cmmd: bool = True,
    compute_is: bool = True,
    batch_size: int = 64,
) -> dict:
    """
    Compute all distribution-level metrics in one call.

    Inception features are extracted once and reused for FID, KID, and MMD.

    Args:
        real_imgs    : (N, C, H, W) real / reference images in [-1, 1]
        fake_imgs    : (M, C, H, W) generated images in [-1, 1]
        device       : target device (defaults to real_imgs.device)
        compute_cmmd : set False to skip CMMD (requires open-clip-torch)
        compute_is   : set False to skip Inception Score
        batch_size   : images per Inception / CLIP forward pass

    Returns:
        dict with keys: fid, kid_mean, kid_std, mmd, [is_mean, is_std], [cmmd]
    """
    if device is None:
        device = real_imgs.device

    real_feats = extract_inception_features(
        real_imgs, device=device, batch_size=batch_size
    )
    fake_feats = extract_inception_features(
        fake_imgs, device=device, batch_size=batch_size
    )

    kid_mean, kid_std = kid(real_feats, fake_feats)
    results: dict = {
        "fid": fid(real_feats, fake_feats),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "mmd": mmd(real_feats, fake_feats),
    }

    if compute_is:
        is_mean, is_std = inception_score(
            fake_imgs, device=device, batch_size=batch_size
        )
        results["is_mean"] = is_mean
        results["is_std"] = is_std

    if compute_cmmd:
        try:
            results["cmmd"] = cmmd(
                real_imgs, fake_imgs, device=device, batch_size=batch_size
            )
        except ImportError:
            pass

    return results


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _to_3ch_neg1_1(x: Tensor) -> Tensor:
    """Expand grayscale → 3-channel and clamp to [-1, 1]."""
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    # Normalise whatever range to [-1, 1] using per-batch min/max
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = 2.0 * (x - mn) / (mx - mn) - 1.0
    return x.clamp(-1.0, 1.0)


def _to_3ch_01(x: Tensor) -> Tensor:
    """Expand grayscale → 3-channel and normalise to [0, 1]."""
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    return x.clamp(0.0, 1.0)


def _get_inception_pool(device: torch.device) -> nn.Module:
    """InceptionV3 with fc replaced by Identity → returns 2048-d pool3 features."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.aux_logits = False
    return model.to(device).eval()


def _get_inception_cls(device: torch.device) -> nn.Module:
    """Full InceptionV3 classifier → returns 1000-d logits."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.aux_logits = False
    return model.to(device).eval()


def _inception_pool_forward(
    imgs: Tensor,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Batch-forward through the pool-feature InceptionV3; returns (N, 2048)."""
    imgs_3ch = _to_3ch_neg1_1(imgs)
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(imgs_3ch), batch_size):
            batch = imgs_3ch[i : i + batch_size].to(device)
            if batch.shape[-1] < 75:
                batch = F.interpolate(
                    batch, size=(75, 75), mode="bilinear", align_corners=False
                )
            feats = model(batch)
            all_feats.append(feats.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)


def _inception_cls_forward(
    imgs: Tensor,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Batch-forward through the classifier InceptionV3; returns (N, 1000) softmax probs."""
    imgs_3ch = _to_3ch_neg1_1(imgs)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(imgs_3ch), batch_size):
            batch = imgs_3ch[i : i + batch_size].to(device)
            if batch.shape[-1] < 75:
                batch = F.interpolate(
                    batch, size=(75, 75), mode="bilinear", align_corners=False
                )
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().float().numpy())
    return np.concatenate(all_preds, axis=0)


def _rbf_kernel(X: Tensor, Y: Tensor, gamma: float) -> Tensor:
    """Gaussian RBF kernel k(x, y) = exp(−γ ‖x − y‖²)."""
    XX = (X * X).sum(dim=1, keepdim=True)
    YY = (Y * Y).sum(dim=1, keepdim=True)
    sq_dists = XX + YY.T - 2.0 * (X @ Y.T)
    return torch.exp(-gamma * sq_dists.clamp(min=0.0))


def _polynomial_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0,
) -> float:
    """
    Unbiased polynomial MMD² estimator (used by KID).
    k(x, y) = (γ <x, y> + c₀)^d
    """
    m, n = len(X), len(Y)

    def poly(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (gamma * (A @ B.T) + coef0) ** degree

    kxx = poly(X, X)
    kyy = poly(Y, Y)
    kxy = poly(X, Y)

    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(
        kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2.0 * kxy.mean()
    )


# =============================================================================
# IV-SURFACE ARBITRAGE CHECKS (static no-arbitrage, roadmap Phase 7)
# =============================================================================
#
# Both checks operate on surfaces in RAW implied-vol units (annualised, e.g.
# 0.2 = 20%) — NOT on z-scored model outputs. Denormalise generated samples
# first with IVSurfaceDataset._denorm().
#
# Grid convention (same as dataset.create_surfaces):
#   rows (H, axis -2) = maturity  (dte_axis, calendar days)
#   cols (W, axis -1) = log-moneyness k = log(K/S)  (logm_axis)
#
# Everything is expressed in total variance  w(k, τ) = σ²(k, τ) · τ  (τ in
# years), the natural quantity for both conditions (Gatheral & Jacquier 2014).


def _to_numpy_iv(surfaces) -> np.ndarray:
    """(N, 1, H, W) or (N, H, W), torch or numpy → (N, H, W) float64 array."""
    if torch.is_tensor(surfaces):
        surfaces = surfaces.detach().cpu().numpy()
    surfaces = np.asarray(surfaces, dtype=np.float64)
    if surfaces.ndim == 4:
        surfaces = surfaces[:, 0]
    if (surfaces <= 0).any():
        raise ValueError(
            "Non-positive IVs found — surfaces look normalised. "
            "Pass raw implied vols (use IVSurfaceDataset._denorm on samples)."
        )
    return surfaces


def calendar_arbitrage(surfaces, dte_axis: np.ndarray, tol: float = 0.0) -> dict:
    """
    Calendar-spread check: total variance w = σ²·τ must be non-decreasing in
    maturity at every fixed log-moneyness. A violation is an adjacent maturity
    pair with w(τ_{i+1}) < w(τ_i) - tol.

    Returns dict:
        cell_rate    — violating (maturity-pair, moneyness) cells / all cells
        surface_rate — fraction of surfaces with ≥ 1 violation
        mean_mag     — mean size of w-decrease over violating cells (0 if none)
    """
    iv = _to_numpy_iv(surfaces)
    tau = np.asarray(dte_axis, dtype=np.float64) / 365.0  # years
    w = iv**2 * tau[None, :, None]  # (N, H, W)

    dw = np.diff(w, axis=1)  # w(τ_{i+1}) − w(τ_i)
    viol = dw < -tol
    return {
        "cell_rate": float(viol.mean()),
        "surface_rate": float(viol.any(axis=(1, 2)).mean()),
        "mean_mag": float(-dw[viol].mean()) if viol.any() else 0.0,
    }


def butterfly_arbitrage(
    surfaces, logm_axis: np.ndarray, dte_axis: np.ndarray, tol: float = 0.0
) -> dict:
    """
    Butterfly check via Durrleman's condition: for each maturity slice, the
    risk-neutral density is non-negative iff

        g(k) = (1 - k·w'/(2w))² - (w'²/4)(1/w + 1/4) + w''/2  >=  0

    with w(k) the total variance and ' = d/dk. Derivatives are central finite
    differences, so only interior moneyness points are checked.

    Returns dict with the same keys as calendar_arbitrage (cells are interior
    (maturity, moneyness) grid points; magnitude is |g| over violations).
    """
    iv = _to_numpy_iv(surfaces)
    k = np.asarray(logm_axis, dtype=np.float64)
    tau = np.asarray(dte_axis, dtype=np.float64) / 365.0
    w = iv**2 * tau[None, :, None]  # (N, H, W)

    dk = k[1] - k[0]  # uniform grid
    wp = (w[..., 2:] - w[..., :-2]) / (2.0 * dk)  # w'
    wpp = (w[..., 2:] - 2.0 * w[..., 1:-1] + w[..., :-2]) / dk**2  # w''
    wm = w[..., 1:-1]
    km = k[None, None, 1:-1]

    g = (1.0 - km * wp / (2.0 * wm)) ** 2 - (wp**2 / 4.0) * (1.0 / wm + 0.25) + wpp / 2.0  # fmt: skip
    viol = g < -tol
    return {
        "cell_rate": float(viol.mean()),
        "surface_rate": float(viol.any(axis=(1, 2)).mean()),
        "mean_mag": float(-g[viol].mean()) if viol.any() else 0.0,
    }


def arbitrage_report(
    surfaces, logm_axis: np.ndarray, dte_axis: np.ndarray, tol: float = 0.0
) -> dict:
    """Run both static-arbitrage checks; returns a flat {check_metric: value} dict."""
    cal = calendar_arbitrage(surfaces, dte_axis, tol)
    bfly = butterfly_arbitrage(surfaces, logm_axis, dte_axis, tol)
    return {f"calendar_{k}": v for k, v in cal.items()} | {
        f"butterfly_{k}": v for k, v in bfly.items()
    }
