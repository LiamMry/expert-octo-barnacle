import math
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def save_recon_grid(
    label: torch.Tensor,
    obs: torch.Tensor,
    recon: torch.Tensor,
    reprj_recon: torch.Tensor,
    path: Path,
    title: str,
    n_show: int = 8,
) -> None:
    """5-row diagnostic grid: label | obs | recon | (recon − label)^2 | (reprj_recon - obs)^2."""
    n_show = min(n_show, label.shape[0])
    cmap = "gray" if label.shape[1] == 1 else None
    rows = [
        ("label", label[:n_show].cpu(), cmap),
        ("obs", obs[:n_show].cpu(), cmap),
        ("recon", recon[:n_show].cpu(), cmap),
        ("|err|", (recon[:n_show] - label[:n_show]).cpu() ** 2, "jet"),
        ("|reprj_err|", (reprj_recon[:n_show] - obs[:n_show]).cpu() ** 2, "jet"),
    ]
    fig, axes = plt.subplots(
        len(rows), n_show, figsize=(n_show * 1.5, len(rows) * 1.7), squeeze=False
    )
    fig.subplots_adjust(left=0.10, hspace=0.05, wspace=0.05)
    for row_idx, (row_label, src, cm) in enumerate(rows):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=8)
        for col in range(n_show):
            ax = axes[row_idx, col]
            im = ax.imshow(src[col].permute(1, 2, 0).squeeze().numpy(), cmap=cm)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def save_grid(
    tensor: torch.Tensor,
    path: Path,
    title: str,
    n: int = 16,
    colorbar: bool = True,
) -> None:
    """Save a (N,C,H,W) tensor as a simple image grid."""
    tensor = tensor[:n].cpu()
    n = tensor.shape[0]
    ncols = min(8, n)
    nrows = math.ceil(n / ncols)
    cmap = "gray" if tensor.shape[1] == 1 else None

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5), squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = tensor[i].permute(1, 2, 0).squeeze().numpy()
            im = ax.imshow(img, cmap=cmap)
            if colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")
