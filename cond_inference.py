"""
Conditional reconstruction benchmark across samplers.

Usage
-----
    # Default (CT task, DPS, all 3 samplers):
    python cond_inference.py --ckpt ./checkpoints/best.pt

    # Custom task / observation directory:
    python cond_inference.py --ckpt ./checkpoints/best.pt \\
        --task_cfg config/task_config.yaml \\
        --obs_dir  data/ \\
        --samplers euler_ode dpm_solver_2

    # Override steps for all samplers:
    python cond_inference.py --ckpt ./checkpoints/best.pt --steps 200

Outputs (written to --out_dir, default: cond_samples/)
------------------------------------------------------
    recon_<name>_steps<N>.png   — label | obs | recon | |err| grid per sampler
    recon_<name>_steps<N>.pt    — raw reconstructed (N,C,H,W) tensor in [-1,1]
    comparison.png              — all samplers in one figure with metrics annotations
    metrics.json                — PSNR / SSIM / MAE / MMD per sampler
"""

import argparse
import json
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from model.unet import create_model
from sde import build_sde, EpsilonScoreModel
from sampler import build_sampler, __SAMPLER_REGISTRY__
from conditioning.physics import Physics
from utils.vis_utils import (
    load_yaml as _load_yaml,
    to_01 as _to_01,
    compute_metrics as _compute_metrics,
    save_recon_grid as _save_recon_grid,
)


_DEFAULT_STEPS = {
    "euler_maruyama": 1000,
    "euler_ode": 500,
    "dpm_solver_2": 100,
}
_NFE_PER_STEP = {"euler_maruyama": 1, "euler_ode": 1, "dpm_solver_2": 2}
ALL_SAMPLERS = ["euler_maruyama", "euler_ode", "dpm_solver_2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_comparison(
    label: torch.Tensor,
    obs: torch.Tensor,
    results: dict,
    out_dir: Path,
    n_show: int = 8,
) -> None:
    """
    Multi-row comparison figure:
        row 0        — ground truth label
        row 1        — observation y
        row 2k       — reconstruction by sampler k
        row 2k+1     — |recon − label| for sampler k
    Row labels include PSNR / SSIM / MAE for each sampler.
    """
    n_show = min(n_show, label.shape[0])
    cmap = "gray" if label.shape[1] == 1 else None
    names = list(results.keys())
    n_rows = 2 + 2 * len(names)

    fig, axes = plt.subplots(
        n_rows,
        n_show,
        figsize=(n_show * 1.9, n_rows * 1.9),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.18, hspace=0.15, wspace=0.35)

    def _show_row(row_idx, tensors, row_label, cm=None):
        axes[row_idx, 0].set_ylabel(
            row_label, fontsize=6, rotation=0, ha="right", va="center"
        )
        for col in range(n_show):
            ax = axes[row_idx, col]
            im = ax.imshow(
                tensors[col].permute(1, 2, 0).squeeze().numpy(), cmap=cm or cmap
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")

    _show_row(0, label[:n_show].cpu(), "label")
    _show_row(1, obs[:n_show].cpu(), "obs")

    for k, name in enumerate(names):
        info = results[name]
        recon = info["recon"][:n_show].cpu()
        diff = (recon - label[:n_show].cpu()).abs()
        m = info["metrics"]

        row_label = (
            f"{name}\n"
            f"{info['steps']} steps\n"
            f"PSNR {m['psnr']:.1f} dB\n"
            f"SSIM {m['ssim']:.3f}"
        )
        _show_row(2 + 2 * k, recon, row_label)
        _show_row(2 + 2 * k + 1, diff, "|err|", cm="hot")

    fig.suptitle("Conditional reconstruction — sampler comparison", fontsize=10)
    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[cond_inference] saved → {path}")


def _print_summary(results: dict) -> None:
    w = 22
    print()
    print("=" * 74)
    print(
        f"{'Sampler':<{w}} {'Steps':>6} {'NFE':>5} {'Time(s)':>8}"
        f" {'PSNR(dB)':>9} {'SSIM':>7} {'MAE':>8} {'MMD':>10}"
    )
    print("-" * 74)
    for name, info in results.items():
        m = info["metrics"]
        nfe = info["steps"] * _NFE_PER_STEP.get(name, 1)
        print(
            f"{name:<{w}} {info['steps']:>6} {nfe:>5} {info['elapsed']:>8.1f}"
            f" {m['psnr']:>9.2f} {m['ssim']:>7.4f} {m['mae']:>8.5f} {m['mmd']:>10.4e}"
        )
    print("=" * 74)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Conditional reconstruction benchmark across diffusion samplers.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (best.pt / latest.pt).",
    )
    p.add_argument(
        "--task_cfg",
        type=str,
        default="config/ct_task_config.yaml",
        help="Task config (physics + conditioning).  Default: ct_task_config.yaml.",
    )
    p.add_argument(
        "--obs_dir",
        type=str,
        default="data",
        help="Directory with observation.pt and observation.clean.pt.",
    )
    p.add_argument(
        "--samplers",
        nargs="+",
        default=ALL_SAMPLERS,
        metavar="NAME",
        help=f"Samplers to run.  Default: all three {ALL_SAMPLERS}.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Override step count for all samplers.  Defaults: {_DEFAULT_STEPS}.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="Number of observations to reconstruct (default 8).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="cond_samples")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cond_inference] device   : {device}")

    # Validate sampler names early
    available = list(__SAMPLER_REGISTRY__.keys())
    unknown = [n for n in args.samplers if n not in available]
    if unknown:
        raise ValueError(f"Unknown sampler(s): {unknown}. Available: {available}")

    # --- Configs ---
    ckpt_data = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt_data["config"]
    task_cfg = _load_yaml(args.task_cfg)

    # --- Model ---
    score_net = create_model(**model_cfg["model"]).to(device)
    score_net.load_state_dict(ckpt_data["model_state"])
    score_net.eval()
    print(
        f"[cond_inference] model    : {sum(p.numel() for p in score_net.parameters()):,} params"
    )

    # --- SDE ---
    sde = build_sde(**model_cfg["sde"])

    # UPDATED: ε-parameterisation — checkpoints now store an ε-predictor, so
    # wrap it into a score function before handing it to the samplers.
    score_net = EpsilonScoreModel(score_net, sde)

    # --- Physics ---
    physics = Physics(**task_cfg["physics"])

    # --- Conditioning method ---
    cond_name = task_cfg["conditioning"]["name"]
    cond_kwargs = {k: v for k, v in task_cfg["conditioning"].items() if k != "name"}
    print(f"[cond_inference] task_cfg : {args.task_cfg}")
    print(f"[cond_inference] cond     : {cond_name}  {cond_kwargs}")

    # --- Observations ---
    obs_dir = Path(args.obs_dir)
    y_all = torch.load(obs_dir / "observation.pt", map_location="cpu")
    x_all = torch.load(obs_dir / "observation.clean.pt", map_location="cpu")
    n = min(args.n_samples, y_all.shape[0])
    y = y_all[:n].to(device)
    x = x_all[:n]  # stays on CPU — used as ground truth for metrics
    print(
        f"[cond_inference] data     : {n} samples  y{tuple(y.shape)}  x{tuple(x.shape)}"
    )

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Generation shape ---
    shape = (
        n,
        model_cfg["model"].get("in_channels", 1),
        model_cfg["model"]["image_size"],
        model_cfg["model"]["image_size"],
    )

    results: dict = {}

    for sampler_name in args.samplers:
        num_steps = (
            args.steps
            if args.steps is not None
            else _DEFAULT_STEPS.get(sampler_name, 500)
        )
        nfe = num_steps * _NFE_PER_STEP.get(sampler_name, 1)
        print(f"\n[cond_inference] ── {sampler_name}  steps={num_steps}  NFE={nfe} ──")

        sampler = build_sampler(
            name=sampler_name,
            sde=sde,
            score_fn=score_net,
            num_steps=num_steps,
            device=device,
            physics=physics,
            conditioning=cond_name,
            **cond_kwargs,
        )

        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        recon = sampler.sample(shape=shape, y=y).detach().cpu()
        elapsed = time.perf_counter() - t0

        print(
            f"[cond_inference] done in {elapsed:.2f}s  "
            f"range=[{recon.min():.3f}, {recon.max():.3f}]"
        )

        # --- Metrics ---
        metrics = _compute_metrics(recon, x, device)
        print(
            f"[cond_inference] PSNR={metrics['psnr']:.2f} dB  "
            f"SSIM={metrics['ssim']:.4f}  "
            f"MAE={metrics['mae']:.5f}  "
            f"MMD={metrics['mmd']:.4e}"
        )

        # --- Save per-sampler outputs ---
        tag = f"{sampler_name}_steps{num_steps}"
        # recon_01 = _to_01(recon)

        _save_recon_grid(
            x,
            y.cpu(),
            recon,
            out_dir / f"recon_{tag}.png",
            title=f"{sampler_name}  ·  {num_steps} steps  ·  {elapsed:.1f}s  ·  PSNR={metrics['psnr']:.1f}dB",
            n_show=min(8, n),
        )

        torch.save(recon, out_dir / f"recon_{tag}.pt")
        print(f"[cond_inference] saved → {out_dir / f'recon_{tag}.pt'}")

        results[sampler_name] = {
            "recon": recon,
            "steps": num_steps,
            "elapsed": elapsed,
            "metrics": metrics,
        }

    # --- Comparison figure ---
    _save_comparison(x, y.cpu(), results, out_dir, n_show=min(8, n))

    # --- Metrics JSON ---
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                name: {
                    **info["metrics"],
                    "steps": info["steps"],
                    "elapsed_s": info["elapsed"],
                }
                for name, info in results.items()
            },
            f,
            indent=2,
        )
    print(f"[cond_inference] saved → {json_path}")

    _print_summary(results)


if __name__ == "__main__":
    main()
