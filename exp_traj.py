"""
exp_traj.py — reverse-diffusion trajectory strip (and GIF) for the README.

Integrates the probability-flow ODE for one sample and snapshots the state at
a few noise levels: pure noise at t = T → smooth vol surface at t ≈ 0.

Outputs (--out_dir, default assets/):
    diffusion_strip.png — one row of 3-D surfaces along the trajectory
    diffusion.gif       — the same trajectory animated (needs pillow)

Usage:
    python exp_traj.py --ckpt trained_models_v3/best_val.pt

Note: use best_val.pt / latest.pt, NOT the best_mmd_* checkpoints — MMD is
lowest in the first ~50 epochs (metric artifact), so those checkpoints are
early models that still sample noise.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.unet import create_model
from sde import build_sde, EpsilonScoreModel
from dataset import (
    IVSurfaceDataset,
    LOGM_MIN,
    LOGM_MAX,
    DTE_MIN,
    DTE_MAX,
    GRID_H,
    GRID_W,
)

# Noise levels to snapshot (VP-SDE: most of the visible denoising is at low t)
FRAME_TIMES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def euler_ode_trajectory(sde, score_fn, num_steps, device, seed=0):
    """Euler probability-flow ODE for one sample; returns [(t, x), ...] frames."""
    torch.manual_seed(seed)
    times = torch.linspace(sde.T, 1e-4, num_steps + 1, device=device)
    # step index closest to each requested snapshot time
    frame_idx = {int(torch.argmin((times - t).abs())): None for t in FRAME_TIMES}

    x = torch.randn((1, 1, GRID_H, GRID_W), device=device)
    frames = []
    with torch.no_grad():
        for i in tqdm(range(num_steps + 1)):
            if i in frame_idx:
                frames.append((float(times[i]), x[0, 0].cpu().clone()))
            if i == num_steps:
                break
            t_b = times[i].expand(1)
            drift = sde.probability_flow_ode(x, t_b, score_fn(x, t_b))
            x = x + drift * (times[i + 1] - times[i])
    return frames


def plot_frame(ax, gm, gd, surf, t):
    ax.plot_surface(gm, gd, surf, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(f"t = {t:.2f}", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.view_init(elev=25, azim=-60)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="trained_models_v3/best_val.pt")
    p.add_argument("--data", default="data/iv_surfaces.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="assets")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model + SDE (same loading path as uncond_inference.py) ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    net = create_model(**cfg["model"]).to(device)
    state = ckpt.get("ema_state") or ckpt["model_state"]
    net.load_state_dict({k: v.float() for k, v in state.items()})
    net.eval()
    sde = build_sde(**cfg["sde"])
    score_fn = EpsilonScoreModel(net, sde)

    frames = euler_ode_trajectory(sde, score_fn, args.steps, device, args.seed)

    # denormalise every frame to vol units with train stats
    ds = IVSurfaceDataset(path=args.data, split="train")
    frames = [(t, ds._denorm(x).numpy()) for t, x in frames]

    m_axis = np.linspace(LOGM_MIN, LOGM_MAX, GRID_W)
    d_axis = np.linspace(DTE_MIN, DTE_MAX, GRID_H)
    gm, gd = np.meshgrid(m_axis, d_axis)

    # --- Strip ---
    n = len(frames)
    fig, axes = plt.subplots(
        1, n, figsize=(2.2 * n, 2.6), subplot_kw={"projection": "3d"}
    )
    for ax, (t, surf) in zip(axes, frames):
        plot_frame(ax, gm, gd, surf, t)
    fig.suptitle(
        "Reverse diffusion: noise → implied-vol surface (probability-flow ODE)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "diffusion_strip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_dir / 'diffusion_strip.png'}")

    # --- GIF (optional — skipped if pillow is unavailable) ---
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig = plt.figure(figsize=(4, 3.6))
        ax = fig.add_subplot(projection="3d")

        def draw(i):
            ax.clear()
            plot_frame(ax, gm, gd, frames[i][1], frames[i][0])

        anim = FuncAnimation(fig, draw, frames=n)
        anim.save(out_dir / "diffusion.gif", writer=PillowWriter(fps=2))
        plt.close(fig)
        print(f"saved → {out_dir / 'diffusion.gif'}")
    except ImportError as e:
        print(f"GIF skipped ({e})")


if __name__ == "__main__":
    main()
