"""
exp_hero.py — README hero figures (roadmap: repo front page).

Real vs. generated surfaces, plotted the way a derivatives person reads them:
3-D vol surfaces and smile slices, in raw implied-vol units (not z-scores).

Outputs (--out_dir, default assets/):
    hero_surfaces.png — 2 rows x 4 cols of 3-D surfaces: real market days
                        (top, dated, picked across vol regimes) vs generated
                        samples (bottom, picked at the same vol quantiles)
    hero_smiles.png   — smile slices at three maturities: real test-set
                        surfaces (grey) vs generated samples (coloured)

Usage:
    python exp_hero.py [--samples uncond_samples/samples_euler_ode_steps500.pt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import IVSurfaceDataset

# Quantiles of the mean-IV distribution used to pick the displayed surfaces:
# calm → typical → stressed → crisis (1.0 = the single highest-vol day).
REGIME_QUANTILES = [0.10, 0.50, 0.90, 1.0]


def pick_by_level(iv: np.ndarray, quantiles) -> list:
    """Indices of surfaces sitting at the given quantiles of mean IV."""
    order = np.argsort(iv.mean(axis=(1, 2)))
    return [order[int(q * (len(order) - 1))] for q in quantiles]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/iv_surfaces.pt")
    p.add_argument("--samples", default="uncond_samples/samples_euler_ode_steps500.pt")
    p.add_argument("--out_dir", default="assets")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Real surfaces (raw vols) + grid axes ---
    d = torch.load(args.data, weights_only=False)
    real = d["surfaces"][:, 0].numpy()  # (N, H, W), raw IV
    dates = np.array(d["dates"])
    k = np.asarray(d["logm_axis"], dtype=np.float64)
    dte = np.asarray(d["dte_axis"], dtype=np.float64)
    gm, gd = np.meshgrid(k, dte)

    # --- Generated samples, mapped back to vol units with TRAIN stats ---
    ds = IVSurfaceDataset(path=args.data, split="train")
    gen = ds._denorm(torch.load(args.samples))[:, 0].numpy()  # (M, H, W)

    # =======================================================================
    # Figure 1 — 3-D surfaces, real (top) vs generated (bottom)
    # =======================================================================
    idx_r = pick_by_level(real, REGIME_QUANTILES)
    idx_g = pick_by_level(gen, REGIME_QUANTILES)
    shown = np.concatenate([real[idx_r], gen[idx_g]])
    vmin, vmax = float(shown.min()), float(shown.max())

    ncol = len(REGIME_QUANTILES)
    fig, axes = plt.subplots(
        2, ncol, figsize=(3.4 * ncol, 6.6), subplot_kw={"projection": "3d"}
    )
    for col in range(ncol):
        for row, (surf, title) in enumerate(
            [
                (real[idx_r[col]], f"market — {str(dates[idx_r[col]])[:10]}"),
                (gen[idx_g[col]], "generated"),
            ]
        ):
            ax = axes[row, col]
            ax.plot_surface(
                gm, gd, surf, cmap="viridis", vmin=vmin, vmax=vmax,
                linewidth=0, antialiased=True,
            )
            ax.set_title(title, fontsize=9)
            if row == 1:  # axis labels on the bottom row only — avoids clutter
                ax.set_xlabel("log-moneyness", fontsize=7)
                ax.set_ylabel("DTE", fontsize=7)
            ax.set_zlim(vmin, vmax)
            ax.tick_params(labelsize=6)
            ax.view_init(elev=25, azim=-60)
    fig.suptitle(
        "SPY implied-vol surfaces — market days across vol regimes (top) "
        "vs unconditional diffusion samples (bottom)",
        fontsize=11,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.08, top=0.90)
    fig.savefig(out_dir / "hero_surfaces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_dir / 'hero_surfaces.png'}")

    # =======================================================================
    # Figure 2 — smile slices at three maturities, real vs generated
    # =======================================================================
    ds_test = IVSurfaceDataset(path=args.data, split="test", normalize=False)
    real_test = ds_test.surfaces[:, 0].numpy()

    rows = [int(np.argmin(np.abs(dte - m))) for m in (30, 91, 182)]
    n_show = 12
    rng = np.random.default_rng(0)
    ir = rng.choice(len(real_test), n_show, replace=False)
    ig = rng.choice(len(gen), n_show, replace=False)

    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.4), sharey=True)
    for ax, row in zip(axes, rows):
        for j in ir:
            ax.plot(k, real_test[j, row], color="0.6", lw=0.8,
                    label="market (test)" if j == ir[0] else None)
        for j in ig:
            ax.plot(k, gen[j, row], color="tab:blue", lw=0.8, alpha=0.8,
                    label="generated" if j == ig[0] else None)
        ax.set_title(f"{dte[row]:.0f} days to expiry", fontsize=10)
        ax.set_xlabel("log-moneyness k = log(K/S)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("implied vol")
    axes[0].legend(fontsize=8)
    fig.suptitle("Smile slices — market test-set surfaces vs generated samples", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "hero_smiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_dir / 'hero_smiles.png'}")


if __name__ == "__main__":
    main()
