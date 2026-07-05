import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
from torchvision.utils import make_grid


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_t_histogram(
    t_values: list, save_path: Path, eps_t: float, T: float, mode: str
):
    """Save a histogram of sampled diffusion timesteps t collected over one epoch."""
    t_np = np.concatenate(
        [t.numpy() if hasattr(t, "numpy") else np.array(t) for t in t_values]
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(
        t_np,
        bins=50,
        range=(eps_t, T),
        color="steelblue",
        edgecolor="none",
        density=True,
    )
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.set_title(f"t sampling distribution  [mode: {mode}]")
    ax.set_xlim(eps_t, T)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def save_loss_plot(train_losses: list, val_losses: list, save_path: Path):
    """Save a train/val loss curve"""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="train", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="val", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("DSM Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_lr_plot(lr_history: list, save_path: Path):
    """Save the learning rate schedule curve to disk."""
    epochs = range(1, len(lr_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lr_history, label="lr", color="green", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")  # log scale makes warmup + decay much clearer
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_mmd_plot(mmd_history, save_path: Path, every: int = 1):
    """Create and save the MMD plot to disk.

    UPDATED for multi-sampler generation:
      - `mmd_history` is a dict {method: [values]} — one curve per sampler on
        the same axes, so samplers can be compared directly. A plain list (the
        OLD single-sampler format) still works.
      - `every` = epochs between generations, so the x-axis shows true
        training epochs instead of the generation index.
    """
    if not isinstance(mmd_history, dict):
        mmd_history = {"mmd": mmd_history}  # OLD single-list format

    fig, ax = plt.subplots(figsize=(8, 4))
    # UPDATED: distinct linestyles — the deterministic (ODE) samplers integrate
    # the same probability-flow ODE from the same seed, so at generous step
    # counts their MMD values coincide to ~5 decimals and solid curves would
    # hide each other completely. Dashes let coincident curves interleave.
    linestyles = ["-", "--", "-.", ":"]
    for k, (method, values) in enumerate(mmd_history.items()):
        epochs = [every * (i + 1) for i in range(len(values))]
        ax.plot(
            epochs, values, label=method, marker="o", markersize=3,
            linestyle=linestyles[k % len(linestyles)], alpha=0.9,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD")
    ax.set_title("Maximum Mean Discrepancy")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_arbitrage_plot(
    arb_history: dict, real_baseline: dict, save_path: Path, every: int = 1
):
    """Evolution of static-arbitrage violation rates on generated samples.

    arb_history   : {method: {"calendar": [rates], "butterfly": [rates]}}
    real_baseline : {"calendar": rate, "butterfly": rate} measured on the real
                    surfaces, drawn as a dashed black reference line — raw
                    market quotes are NOT arbitrage-free, so matching the real
                    level (not zero) is the target.
    every         : epochs between generations (true epochs on the x-axis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    linestyles = ["-", "--", "-.", ":"]  # coincident ODE curves stay visible
    for ax, check in zip(axes, ["calendar", "butterfly"]):
        for k, (method, hist) in enumerate(arb_history.items()):
            values = [100.0 * v for v in hist[check]]
            epochs = [every * (i + 1) for i in range(len(values))]
            ax.plot(
                epochs, values, label=method, marker="o", markersize=3,
                linestyle=linestyles[k % len(linestyles)], alpha=0.9,
            )
        ax.axhline(
            100.0 * real_baseline[check], color="black", linestyle="--",
            linewidth=1.0, label="real data",
        )
        ax.set_title(f"{check} violations")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("cell violation rate (%)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_samples_grid(
    samples: torch.Tensor,  # (N, C, H, W) for images or (N, 1, L)/(N, L) for signals
    save_path: Path,
    nrow: int = 8,
    epoch: int = 0,
    method: str = "",
):
    is_1d = samples.dim() == 2 or (samples.dim() == 3 and samples.shape[-2] == 1)

    if is_1d:
        # --- 1-D branch ---
        if samples.dim() == 3:
            samples = samples.squeeze(1)  # (N, 1, L) → (N, L)

        N, L = samples.shape
        ncols = nrow
        nrows = math.ceil(N / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 2, nrows * 1.5),
            sharex=True,
            sharey=False,
        )
        axes = np.array(axes).reshape(-1)

        for i, ax in enumerate(axes):
            if i < N:
                ax.plot(samples[i].cpu().numpy(), linewidth=0.8)
            else:
                ax.set_visible(False)
            ax.axis("off")

    else:
        # --- image branch ---
        # Autoscale to the batch's own range: IV surfaces occupy a narrow
        # slice of [-1,1], so the fixed (x+1)/2 map renders them near-black.
        lo, hi = samples.min(), samples.max()
        samples = (samples.clamp(lo, hi) - lo) / (hi - lo + 1e-8)
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        npimg = grid.permute(1, 2, 0).cpu().numpy()
        if npimg.shape[-1] == 1:
            npimg = npimg[..., 0]

        fig, ax = plt.subplots(
            figsize=(nrow * 1.2, math.ceil(samples.shape[0] / nrow) * 1.2)
        )
        ax.imshow(npimg, cmap="viridis" if samples.shape[1] == 1 else None)
        ax.axis("off")

    fig.suptitle(f"Generated samples — epoch {epoch}  [{method.upper()}]", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
