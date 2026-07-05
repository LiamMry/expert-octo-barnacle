"""
Unconditional generation benchmark — runs selected samplers and saves results.

Usage
-----
    # All samplers, default step counts:
    python uncond_inference.py --ckpt ./checkpoints/best.pt

    # Specific samplers:
    python uncond_inference.py --ckpt ./checkpoints/best.pt \\
        --samplers euler_maruyama euler_ode

    # Override steps for all selected samplers:
    python uncond_inference.py --ckpt ./checkpoints/best.pt --steps 200

Outputs (written to --out_dir, default: uncond_samples/)
---------------------------------------------------------
    samples_<name>_steps<N>.png   — 4×4 grid for each sampler
    samples_<name>_steps<N>.pt    — raw (16, C, H, W) tensor
    comparison.png                — all samplers side-by-side (1 row each)
"""

import argparse
import math
import time
from pathlib import Path

import yaml
import torch
import matplotlib.pyplot as plt

from model.unet import create_model
from sde import build_sde, EpsilonScoreModel
from sampler import build_sampler, __SAMPLER_REGISTRY__
from dataset import IVSurfaceDataset


# Sensible default step counts per sampler (used when --steps is not given)
_DEFAULT_STEPS = {
    "euler_maruyama": 1000,
    "euler_ode": 500,
    "dpm_solver_2": 100,
}

# NFE (network evaluations) consumed per ODE/SDE step
_NFE_PER_STEP = {
    "euler_maruyama": 1,
    "euler_ode": 1,
    "dpm_solver_2": 2,
}

ALL_SAMPLERS = ["euler_maruyama", "euler_ode", "dpm_solver_2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_comparison(results: dict, out_dir: Path, n_show: int = 8) -> None:
    """
    One row per sampler, n_show columns.
    Row label shows sampler name, step count, and wall time.
    """
    names = list(results.keys())
    n_rows = len(names)

    first_samples = results[names[0]]["samples"]
    cmap = "gray" if first_samples.shape[1] == 1 else None

    fig, axes = plt.subplots(
        n_rows,
        n_show,
        figsize=(n_show * 1.6, n_rows * 2.0),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.18, hspace=0.1, wspace=0.05)

    for row, name in enumerate(names):
        info = results[name]
        samples = info["samples"][:n_show].cpu()
        steps = info["steps"]
        elapsed = info["elapsed"]
        nfe = steps * _NFE_PER_STEP.get(name, 1)

        row_label = f"{name}\n{steps} steps · {nfe} NFE\n{elapsed:.1f} s"
        axes[row, 0].set_ylabel(
            row_label, fontsize=7, rotation=0, ha="right", va="center", labelpad=5
        )

        for col in range(n_show):
            ax = axes[row, col]
            if col < len(samples):
                img = samples[col].permute(1, 2, 0).squeeze().numpy()
                ax.imshow(img, cmap=cmap)
            ax.axis("off")

    fig.suptitle("Unconditional samples — sampler comparison", fontsize=10)
    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[uncond_inference] saved → {path}")


def _print_summary(results: dict) -> None:
    w = 22
    print()
    print("=" * 60)
    print(
        f"{'Sampler':<{w}}  {'Steps':>6}  {'NFE':>6}  {'Time (s)':>9}  {'mean±std':>14}"
    )
    print("-" * 60)
    for name, info in results.items():
        nfe = info["steps"] * _NFE_PER_STEP.get(name, 1)
        s = info["samples"]
        stat = f"{s.mean():.3f}±{s.std():.3f}"
        print(
            f"{name:<{w}}  {info['steps']:>6}  {nfe:>6}  {info['elapsed']:>9.2f}  {stat:>14}"
        )
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unconditional generation benchmark across diffusion samplers.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (best.pt / latest.pt).",
    )
    p.add_argument(
        "--samplers",
        nargs="+",
        default=ALL_SAMPLERS,
        metavar="NAME",
        help=(
            "Samplers to run (space-separated).\n"
            f"Available: {ALL_SAMPLERS}\n"
            "Default: all three."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Override step count for every selected sampler.\n"
            f"Defaults per sampler: {_DEFAULT_STEPS}."
        ),
    )
    p.add_argument(
        "--n_samples", type=int, default=64, help="Samples to generate (default 64)."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_dir", type=str, default="uncond_samples", help="Output directory."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[uncond_inference] device : {device}")

    # Validate sampler names before loading anything heavy
    available = list(__SAMPLER_REGISTRY__.keys())
    unknown = [n for n in args.samplers if n not in available]
    if unknown:
        raise ValueError(f"Unknown sampler(s): {unknown}. Available: {available}")

    # --- Checkpoint ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    # --- Model ---
    score_net = create_model(**cfg["model"]).to(device)
    # Prefer EMA weights (what the in-training sample grids use); the raw
    # model_state is much noisier and samples poorly.
    if ckpt.get("ema_state") is not None:
        state = {k: v.float() for k, v in ckpt["ema_state"].items()}
        print("[uncond_inference] using EMA weights")
    else:
        state = ckpt["model_state"]
        print("[uncond_inference] WARNING: no ema_state in ckpt — using raw weights")
    score_net.load_state_dict(state)
    score_net.eval()
    n_params = sum(p.numel() for p in score_net.parameters())
    print(f"[uncond_inference] model  : {n_params:,} params")

    # --- SDE ---
    sde = build_sde(**cfg["sde"])

    # UPDATED: ε-parameterisation — checkpoints now store an ε-predictor, so
    # wrap it into a score function before handing it to the samplers.
    score_net = EpsilonScoreModel(score_net, sde)

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Sample shape ---
    shape = (
        args.n_samples,
        cfg["model"].get("in_channels", 1),
        cfg["model"]["image_size"],
        cfg["model"]["image_size"],
    )

    results: dict = {}

    for name in args.samplers:
        num_steps = (
            args.steps if args.steps is not None else _DEFAULT_STEPS.get(name, 500)
        )
        nfe = num_steps * _NFE_PER_STEP.get(name, 1)
        print(f"\n[uncond_inference] ── {name}  steps={num_steps}  NFE={nfe} ──")

        sampler = build_sampler(
            name=name,
            sde=sde,
            score_fn=score_net,
            num_steps=num_steps,
            device=device,
        )

        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        samples = sampler.sample(shape=shape).detach().cpu()
        elapsed = time.perf_counter() - t0

        print(
            f"[uncond_inference] done in {elapsed:.2f}s   "
            f"range=[{samples.min():.3f}, {samples.max():.3f}]"
        )

        tag = f"{name}_steps{num_steps}"

        IVSurfaceDataset(path=None)._visGridImage(
            samples, ncol=8, save_path=out_dir / f"samples_{tag}.png"
        )

        torch.save(samples, out_dir / f"samples_{tag}.pt")
        print(f"[uncond_inference] saved → {out_dir / f'samples_{tag}.pt'}")

        results[name] = {"samples": samples, "steps": num_steps, "elapsed": elapsed}

    # --- Comparison figure (only meaningful with ≥2 samplers) ---
    if len(results) >= 2:
        _save_comparison(results, out_dir, n_show=min(8, args.n_samples))

    _print_summary(results)


if __name__ == "__main__":
    main()
