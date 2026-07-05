"""
exp_stylized.py — stylized-facts panel for the README.

Does the model reproduce the cross-sectional statistics quants care about,
rather than just visually plausible individual surfaces?

Outputs (--out_dir, default assets/):
    stylized_hists.png — histograms of ATM level, short-dated skew and
                         ATM term-structure slope: train vs test vs generated
    pca_modes.png      — mean surface + top-3 PCA eigenmodes, real vs
                         generated (the classic level / skew / term modes,
                         cf. Cont & da Fonseca 2002)
    rn_density.png     — implied risk-neutral density (Breeden–Litzenberger
                         via Durrleman's g(k), Gatheral & Jacquier 2014) of a
                         typical generated surface vs a typical market surface

Usage:
    python exp_stylized.py [--samples uncond_samples/samples_euler_ode_steps500.pt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import IVSurfaceDataset


def summary_stats(iv: np.ndarray, k: np.ndarray, dte: np.ndarray) -> dict:
    """Per-surface scalars: ATM level (~30d), skew (~30d), term slope."""
    atm = int(np.argmin(np.abs(k)))
    r30 = int(np.argmin(np.abs(dte - 30)))
    lo, hi = atm - 3, atm + 3  # slope over a ~±6% moneyness window
    return {
        "ATM vol (30d)": iv[:, r30, atm],
        "skew (30d)": (iv[:, r30, hi] - iv[:, r30, lo]) / (k[hi] - k[lo]),
        "term slope (1y − 30d, ATM)": iv[:, -1, atm] - iv[:, r30, atm],
    }


def pca_modes(iv: np.ndarray, n_modes: int = 3):
    """Mean surface, top eigen-surfaces and explained-variance ratios."""
    flat = iv.reshape(len(iv), -1)
    mean = flat.mean(axis=0)
    _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    evr = s**2 / (s**2).sum()
    H, W = iv.shape[1:]
    return mean.reshape(H, W), vt[:n_modes].reshape(n_modes, H, W), evr[:n_modes]


def durrleman_g(iv_slice: np.ndarray, k: np.ndarray, tau: float):
    """g(k) on interior points and total variance w — Durrleman's condition."""
    w = iv_slice**2 * tau
    dk = k[1] - k[0]
    wp = (w[2:] - w[:-2]) / (2 * dk)
    wpp = (w[2:] - 2 * w[1:-1] + w[:-2]) / dk**2
    wm, km = w[1:-1], k[1:-1]
    g = (1 - km * wp / (2 * wm)) ** 2 - (wp**2 / 4) * (1 / wm + 0.25) + wpp / 2
    return g, wm


def rn_density(iv_slice: np.ndarray, k: np.ndarray, tau: float):
    """Risk-neutral density in log-moneyness: p(k) = g(k)·φ(d₋)/√w."""
    g, w = durrleman_g(iv_slice, k, tau)
    d2 = -k[1:-1] / np.sqrt(w) - np.sqrt(w) / 2
    return g * np.exp(-0.5 * d2**2) / np.sqrt(2 * np.pi * w)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/iv_surfaces.pt")
    p.add_argument("--samples", default="uncond_samples/samples_euler_ode_steps500.pt")
    p.add_argument("--out_dir", default="assets")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(args.data, weights_only=False)
    k = np.asarray(d["logm_axis"], dtype=np.float64)
    dte = np.asarray(d["dte_axis"], dtype=np.float64)

    # normalize=True computes the train z-score stats needed by _denorm;
    # .surfaces itself always stays in raw vol units.
    ds = IVSurfaceDataset(path=args.data, split="train")
    train = ds.surfaces[:, 0].numpy()
    test = IVSurfaceDataset(path=args.data, split="test", normalize=False).surfaces[:, 0].numpy()  # fmt: skip
    gen = ds._denorm(torch.load(args.samples))[:, 0].numpy()

    # =======================================================================
    # Figure 1 — histograms of surface summary statistics
    # =======================================================================
    stats = {name: summary_stats(x, k, dte) for name, x in
             [("train", train), ("test", test), ("generated", gen)]}
    keys = list(stats["train"])

    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2))
    for ax, key in zip(axes, keys):
        lo = min(s[key].min() for s in stats.values())
        hi = max(s[key].max() for s in stats.values())
        bins = np.linspace(lo, hi, 40)
        ax.hist(stats["train"][key], bins=bins, density=True, color="0.8", label="train")
        ax.hist(stats["test"][key], bins=bins, density=True, histtype="step",
                color="k", lw=1.2, label="test")
        ax.hist(stats["generated"][key], bins=bins, density=True, histtype="step",
                color="tab:blue", lw=1.6, label="generated")
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Surface summary statistics — real vs generated", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "stylized_hists.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_dir / 'stylized_hists.png'}")

    # =======================================================================
    # Figure 2 — PCA eigenmodes, real vs generated
    # =======================================================================
    mean_r, modes_r, evr_r = pca_modes(train)
    mean_g, modes_g, evr_g = pca_modes(gen)
    # sign-align generated modes with the real ones (PCA sign is arbitrary)
    for i in range(len(modes_g)):
        if (modes_g[i] * modes_r[i]).sum() < 0:
            modes_g[i] = -modes_g[i]

    n = 1 + len(modes_r)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.0))
    for row, (mean, modes, evr, name) in enumerate(
        [(mean_r, modes_r, evr_r, "real (train)"), (mean_g, modes_g, evr_g, "generated")]
    ):
        panels = [(mean, "mean surface")] + [
            (modes[i], f"PC{i + 1} ({evr[i]:.0%} var)") for i in range(len(modes))
        ]
        for col, (img, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(img, origin="lower", aspect="auto", cmap="viridis",
                           extent=[k[0], k[-1], dte[0], dte[-1]])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name} — {title}", fontsize=8)
            if col == 0:
                ax.set_ylabel("DTE")
            ax.set_xlabel("log-moneyness", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("PCA of the surface cross-section — level / skew / term-structure modes",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_dir / 'pca_modes.png'}")

    # =======================================================================
    # Figure 3 — implied risk-neutral density, generated vs market
    # =======================================================================
    # "typical" = surface at the median mean-IV of each set
    pick = lambda x: x[np.argsort(x.mean(axis=(1, 2)))[len(x) // 2]]
    surf_g, surf_r = pick(gen), pick(test)

    rows = [int(np.argmin(np.abs(dte - m))) for m in (30, 91, 182)]
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.2), sharey=True)
    for ax, row in zip(axes, rows):
        tau = dte[row] / 365.0
        ax.plot(k[1:-1], rn_density(surf_r[row], k, tau), "k--", lw=1.2, label="market (test)")
        ax.plot(k[1:-1], rn_density(surf_g[row], k, tau), color="tab:blue", lw=1.6,
                label="generated")
        ax.axhline(0.0, color="r", lw=0.8, alpha=0.5)
        ax.set_title(f"{dte[row]:.0f} days to expiry", fontsize=10)
        ax.set_xlabel("log-moneyness k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("risk-neutral density p(k)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Implied risk-neutral density (Breeden–Litzenberger) — "
                 "p(k) < 0 would be a butterfly arbitrage", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "rn_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_dir / 'rn_density.png'}")


if __name__ == "__main__":
    main()
