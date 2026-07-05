"""
Inference script — unconditional or conditional generation.

Unconditional
-------------
    python3 inference.py --ckpt ./checkpoints/best.pt

Conditional (DPS)
-----------------
    python3 inference.py \
        --ckpt     ./checkpoints/best.pt \
        --task_cfg ./config/task_config.yaml \
        --obs      ./data/observations.pt \
        --scale    1.0

task_config.yaml format
------------------------
    task:
      name: ct
      fname: /path/to/H_f32.npz
    noise:
      name: gaussian
      noise_sigma: 0.05
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from model.unet import create_model
from sde import build_sde, EpsilonScoreModel
from sampler import build_sampler
from conditioning.physics import Physics
from conditioning.conditioning import get_conditioning_method
from utils.vis_utils import load_yaml, save_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_task_cfg(raw):
    """Extract operator/noise names and their kwargs from the nested yaml."""
    task_section = dict(raw.get("task", {}))
    noise_section = dict(raw.get("noise", {}))

    operator_name = task_section.pop("name", "id")
    noise_name = noise_section.pop("name", "none")

    if "noise_sigma" in noise_section:
        noise_section["sigma"] = noise_section.pop("noise_sigma")
    if "noise_rate" in noise_section:
        noise_section["rate"] = noise_section.pop("noise_rate")

    return operator_name, noise_name, task_section, noise_section


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (best.pt or latest.pt).",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of observation to invert. If nothing is given then do all inversions.",
    )

    p.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Diffusion steps.",
    )

    p.add_argument(
        "--method",
        type=str,
        required=True,
        help="Sampler: {euler_maruyama | ancestral}.",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--obs_dir",
        type=str,
        default=None,
        help="Observations .pt file  (N,C,H,W) in [0,1]",
    )

    p.add_argument(
        "--task_cfg",
        type=str,
        default=None,
        help="Task config (physics + conditioning). Enables conditioning when set.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device: {device}")

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    method = args.method
    num_steps = args.steps

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    score_net = create_model(**cfg["model"]).to(device)
    score_net.load_state_dict(ckpt["model_state"])
    score_net.eval()
    print(
        f"[inference] model loaded  ({sum(p.numel() for p in score_net.parameters()):,} params)"
    )

    # ------------------------------------------------------------------
    # 3. SDE
    # ------------------------------------------------------------------
    sde = build_sde(**cfg["sde"])

    # UPDATED: ε-parameterisation — checkpoints now store an ε-predictor, so
    # wrap it into a score function before handing it to the samplers.
    score_net = EpsilonScoreModel(score_net, sde)

    # ------------------------------------------------------------------
    # 4. Conditioning  (skipped when --task_cfg is not provided)
    # ------------------------------------------------------------------
    conditioning = None
    # physics = Physics(operator="id", noise="none")  # no-op default
    y = None
    # task_cfg = args.task_cfg
    # obs_dir = args.obs_dir
    # n_samples = 16

    if args.task_cfg is None:
        pass
    else:
        task_cfg = load_yaml(args.task_cfg)

        # 4.2 - Set physics and conditioning method
        physics = Physics(**task_cfg["physics"])
        conditioning = get_conditioning_method(
            physics=physics, **task_cfg["conditioning"]
        )

        print(
            f"[INFO] - inference.py - Task cfg: {args.task_cfg}"
        )

        # 4.3 - Load observations (y), labels (x)
        if args.obs_dir is None:
            raise ValueError("--obs_dir is required when --task_cfg is set.")
        obs_dir = Path(args.obs_dir)

        y = torch.load(obs_dir / "observations.pt", map_location="cpu").to(device)
        if args.n_samples is None:
            n_samples = y.shape[0]
        else:
            n_samples = args.n_samples
            y = y[:n_samples, ...]
        print(f"[INFO] - inference.py - Obs images loaded from {obs_dir / 'observations.pt'}")  # fmt: skip

        x = torch.load(obs_dir / "observations.clean.pt", map_location="cpu")[:n_samples]  # fmt: skip
        print(f"[INFO] - inference.py - Label images loaded from {obs_dir / 'observations.clean.pt'}")  # fmt: skip

    # if task_cfg is not None:
    #     raw = load_yaml(task_cfg)
    #     operator_name, noise_name, op_kwargs, noise_kwargs = parse_task_cfg(raw)

    #     physics = Physics(
    #         operator=operator_name, noise=noise_name, **op_kwargs, **noise_kwargs
    #     )
    #     conditioning = "dps"
    #     scale = 200.0

    #     if obs is None:
    #         raise ValueError("--obs is required when --task_cfg is set.")
    #     y = torch.load(obs, map_location="cpu")[:n_samples].to(device)
    #     # Load clean label if it exists alongside the observations
    #     label_path = obs.replace(".pt", ".clean.pt")
    #     label = None
    #     if Path(label_path).exists():
    #         label = torch.load(label_path, map_location="cpu")[:n_samples]
    #         print(f"[inference] label images loaded from {label_path}")
    #     print(
    #         f"[inference] conditioning: DPS  operator={operator_name}  "
    #         f"noise={noise_name}  scale={scale}"
    #     )
    # else:
    #     print("[inference] conditioning: none  (unconditional)")

    # ------------------------------------------------------------------
    # 5. Sampler
    # ------------------------------------------------------------------
    sampler = build_sampler(
        name=method,
        sde=sde,
        score_fn=score_net,
        num_steps=num_steps,
        device=device,
        # physics=physics,
        conditioning=conditioning,
        # scale=scale,
    )

    gen_shape = (
        n_samples,
        cfg["model"].get("in_channels", 1),
        cfg["model"]["image_size"],
        cfg["model"]["image_size"],
    )

    # ------------------------------------------------------------------
    # 6. Generate
    # ------------------------------------------------------------------
    seed = 0
    torch.manual_seed(seed)
    print(
        f"[INFO] - inference.py - sampling {n_samples} images."
        f"[INFO] - infernce.py - sampler: {method}  steps: {num_steps}."
    )

    # DPS needs autograd through x_t — no torch.no_grad() wrapper here.
    # Samplers handle their own detach() after each step.
    samples = sampler.sample(shape=gen_shape, y=y)
    samples = samples.detach().cpu()

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out_dir = args.out_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{method}_n{n_samples}_steps{num_steps}"

    save_grid(samples, out_dir / f"samples_{tag}.png", title="Samples")

    if y is not None:
        save_grid(
            y.cpu(),
            out_dir / f"observations_{tag}.png",
            title="Observations  y = A(x) + n",
        )

        # Build comparison grid: obs | recon | label | diff
        rows = {
            "label": x,
            "obs": y.cpu(),
            "recon": samples,
            "diff": (samples - x).abs(),
        }

        n_show = min(n_samples, 8)
        n_rows = len(rows)
        cmap = "gray" if samples.shape[1] == 1 else None

        fig, axes = plt.subplots(
            n_rows, n_show, figsize=(n_show * 1.5, n_rows * 1.6), squeeze=False
        )

        for row_idx, (row_label, src) in enumerate(rows.items()):
            for col in range(n_show):
                ax = axes[row_idx, col]
                img = src[col].permute(1, 2, 0).squeeze().numpy()
                if row_label == "diff":
                    # diff: raw values, let colorbar show magnitude
                    im = ax.imshow(img, cmap="hot")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    im = ax.imshow(img, cmap=cmap)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis("off")
            axes[row_idx, 0].set_ylabel(row_label, fontsize=8)

        fig.suptitle(f"method={method}   steps={num_steps}", fontsize=9)
        fig.tight_layout()
        sbs_path = out_dir / f"obs_vs_recon_{tag}.png"
        fig.savefig(sbs_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[inference] saved → {sbs_path}")


if __name__ == "__main__":
    main()
