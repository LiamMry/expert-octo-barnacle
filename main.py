import argparse

import itertools
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim

import numpy as np

from model.unet import create_model
from sampler import build_sampler
from sde import build_sde, EpsilonScoreModel
from loss import DSMLoss
from dataset import IVSurfaceDataset

# Training utils
from utils.train_utils import EMA, sample_t
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

# Utils
from utils.utils import (
    load_yaml,
    save_loss_plot,
    save_lr_plot,
    save_samples_grid,
    save_mmd_plot,
    save_arbitrage_plot,  # UPDATED: arbitrage-rate evolution plot
    save_t_histogram,
)

# Evaluation
from eval import mmd as compute_mmd
from eval import arbitrage_report  # UPDATED: calendar/butterfly checks


def main():

    # --- Parse arguments --- #
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="./config/model_config.yaml")
    parser.add_argument("--train_cfg", type=str, default="./config/train_config.yaml")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Call training functions
    train(args)


def train(args):

    # --- Set the seed --- #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # --- Set device --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load config --- #
    cfg = load_yaml(args.cfg)
    model_cfg = cfg["model"]
    sde_cfg = cfg["sde"]
    # data_cfg = cfg["data"]
    train_cfg = load_yaml(args.train_cfg)
    gen_cfg = train_cfg["generation"]

    # --- Create score net --- #
    score_net = create_model(**model_cfg)
    score_net = score_net.to(device)
    if train_cfg["train"]["ema"]:
        ema = EMA(score_net, decay=0.9999)

    # --- Sde config --- #
    sde = build_sde(**sde_cfg)

    # --- ε-parameterisation (UPDATED) --- #
    # The UNet now regresses the noise ε (O(1) target at every t); the wrapper
    # turns it into a score s(x,t) = -ε̂/σ(t) so the DSM loss and the samplers
    # keep seeing a score function. EMA + checkpoints stay on the raw
    # `score_net` — the wrapper holds no parameters of its own.
    score_model = EpsilonScoreModel(score_net, sde)

    # --- Sde sanity check --- #
    # bx = next(iter(train_loader))
    # x0 = bx[0, ...]
    # fig, axs = plt.subplots(4, 5, figsize=(10, 9))
    # for i in range(20):
    #     t = (i+1) / 20
    #     xt, _ = sde.marginal_sample(x0, torch.Tensor([t]))
    #     axs[i//5, i - i//5 * 5].imshow(xt[0, ...])
    #     axs[i//5, i - i//5 * 5].set_title(f"t = {t}")
    #     axs[i//5, i - i//5 * 5].set_axis_off()
    # fig.savefig('./tmp/sde_check.png')
    # exit()

    # --- Load data --- #
    train_dataset = IVSurfaceDataset(
        path="/users/lm4057/expert-octo-barnacle/data/iv_surfaces.pt", split="train"
    )
    valid_dataset = IVSurfaceDataset(
        path="/users/lm4057/expert-octo-barnacle/data/iv_surfaces.pt", split="valid"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["train"]["batch_size"],
        shuffle=train_cfg["train"]["shuffle"],
        num_workers=train_cfg["train"]["num_workers"],
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=train_cfg["valid"]["batch_size"],
        shuffle=train_cfg["valid"]["shuffle"],
        num_workers=train_cfg["valid"]["num_workers"],
    )

    # --- Real-data arbitrage baseline (UPDATED) --- #
    # Violation rates of the REAL interpolated surfaces (raw IV units) — the
    # honest reference level for generated samples: market quotes put on a
    # grid are not arbitrage-free, so the target is this level, not zero.
    _rep = arbitrage_report(
        valid_dataset.surfaces, valid_dataset.logm_axis, valid_dataset.dte_axis
    )
    arb_real = {
        "calendar": _rep["calendar_cell_rate"],
        "butterfly": _rep["butterfly_cell_rate"],
    }
    print(
        f"[arb] real-data baseline: calendar {arb_real['calendar']:.2%}, "
        f"butterfly {arb_real['butterfly']:.2%}"
    )

    # --- DSM loss --- #
    criterion = DSMLoss(
        sde, weighting=train_cfg["dsm"]["weighting"], eps_t=train_cfg["dsm"]["eps_t"]
    )
    t_sampling = train_cfg["dsm"].get("t_sampling", "uniform")

    # --- Optimizer & scheduler --- #
    optimizer = optim.Adam(score_net.parameters(), lr=train_cfg["optimizer"]["lr"])
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=5,
    )

    plateau = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg["optimizer"].get("plateau_factor", 0.5),
        patience=train_cfg["optimizer"].get("plateau_patience", 10),
        min_lr=train_cfg["optimizer"]["lr"] * 0.01,
    )

    # --- Generation config --- #
    gen_every = gen_cfg["every_n_epochs"]
    gen_n_samples = gen_cfg["n_samples"]
    gen_batch_size = gen_cfg.get("batch_size", 16)
    # gen_num_steps = gen_cfg["num_steps"]  # OLD: single sampler
    # gen_method = gen_cfg["method"]        # OLD
    # UPDATED: {method: num_steps} dict — every listed sampler gets its own
    # grid + MMD curve each gen epoch. Falls back to the old single-method keys
    # so existing configs keep working.
    gen_methods: dict = gen_cfg.get("methods") or {gen_cfg["method"]: gen_cfg["num_steps"]}  # fmt: skip

    # --- Early stopping config --- #
    # patience: epochs with no improvement before stopping (0 = disabled)
    # # min_delta: minimum decrease in val loss that counts as improvement
    # es_patience = train_cfg["train"].get("es_patience", 0)
    # es_min_delta = train_cfg["train"].get("es_min_delta", 0.0)

    gen_img_shape = (
        cfg["model"].get("in_channels", 1),
        cfg["model"]["image_size"],
        cfg["model"]["image_size"],
    )

    # --- Output dirs --- #
    save_dir = Path(args.save_dir)
    samples_dir = save_dir / "samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # --- History (for plotting) --- #
    train_losses: list[float] = []
    val_losses: list[float] = []
    lr_history: list[float] = []
    # mmd_history: list[float] = []  # OLD: single sampler
    mmd_history: dict[str, list[float]] = {m: [] for m in gen_methods}  # UPDATED
    # UPDATED: arbitrage-violation rates per sampler, tracked like MMD
    arb_history: dict[str, dict[str, list[float]]] = {
        m: {"calendar": [], "butterfly": []} for m in gen_methods
    }

    # --- Training --- #
    global_step = 0
    best_val_loss = math.inf
    # best_mmd = math.inf     # OLD: single sampler
    # current_mmd = math.inf  # OLD
    # UPDATED: tracked per sampler — each gets its own best_mmd_{method}.pt
    best_mmd = {m: math.inf for m in gen_methods}
    current_mmd = {m: math.inf for m in gen_methods}
    # patience_counter = 0
    max_epochs = train_cfg["train"].get("max_epochs", 10_000)
    prev_lr = train_cfg["optimizer"]["lr"]

    epoch_bar = tqdm(itertools.count(1), desc="epochs", unit="ep", position=0)
    for epoch in epoch_bar:
        if epoch > max_epochs:
            epoch_bar.write(f"  [max epochs] reached {max_epochs} — stopping.")
            break
        score_net.train()
        epoch_loss = 0.0
        t_samples: list = []

        step_bar = tqdm(
            train_loader,
            desc=f"  train e{epoch:03d}",
            unit="batch",
            position=1,
            leave=False,  # disappears after each epoch
        )
        for x0, date in step_bar:
            x0 = x0.to(device)
            B = x0.shape[0]

            optimizer.zero_grad()

            t = sample_t(B, sde.T, criterion.eps_t, t_sampling, device)
            t_samples.append(t.detach().cpu())
            # loss = criterion(score_net, x0, t=t)  # OLD: net predicted the score directly
            loss = criterion(score_model, x0, t=t)  # UPDATED: ε-parameterised wrapper
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=3.0)

            optimizer.step()
            if train_cfg["train"]["ema"]:
                ema.update(score_net)

            epoch_loss += loss.item()
            global_step += 1

            # Live loss + lr in the step bar suffix
            step_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_train = epoch_loss / len(train_loader)

        # --- Validation --- #
        score_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0, date in valid_loader:
                x0 = x0.to(device)
                B = x0.shape[0]

                # Sample uniformly t : t ~ U([eps_t, T])
                t = (torch.rand(B, device=device) * (sde.T - criterion.eps_t) + criterion.eps_t)  # fmt: skip
                # val_loss += criterion(score_net, x0, t).item()  # OLD
                val_loss += criterion(score_model, x0, t).item()  # UPDATED: ε-param wrapper

        avg_val = val_loss / len(valid_loader)

        # -- Generation --- #
        # UPDATED: loops over every sampler in gen_methods instead of a single
        # one. The EMA swap and RNG save/restore happen ONCE around the loop;
        # each sampler is re-seeded with 42 so all methods consume identical
        # noise (grids comparable across methods AND epochs); MMD is per method.
        if epoch % gen_every == 0:
            epoch_bar.write(
                f"  [gen] epoch {epoch} — "
                + ", ".join(f"{m} ({s} steps)" for m, s in gen_methods.items())
                + f" — {gen_n_samples} samples each"
            )

            # During generation — use EMA weights:
            if train_cfg["train"]["ema"]:
                saved_weights = {
                    k: v.clone() for k, v in score_net.state_dict().items()
                }
                ema.apply(score_net)

            score_net.eval()

            # UPDATED: save BOTH CPU and CUDA RNG state. manual_seed() reseeds
            # both generators, but the OLD code restored only the CPU one — so
            # training silently resumed with a reset CUDA noise stream after
            # every generation epoch.
            saved_rng = torch.get_rng_state()
            saved_rng_cuda = torch.cuda.get_rng_state_all()

            # --- MMD reference batch: built once, shared by all methods --- #
            real_imgs = []
            for x, date in valid_loader:
                real_imgs.append(x.cpu())
                if sum(r.shape[0] for r in real_imgs) >= gen_n_samples:
                    break
            real_imgs = torch.cat(real_imgs, dim=0)[:gen_n_samples]
            real_flat = real_imgs.flatten(start_dim=1).numpy()

            for gen_method, gen_num_steps in gen_methods.items():
                sampler = build_sampler(
                    name=gen_method,
                    sde=sde,
                    score_fn=score_model,  # ε-param wrapper (EMA weights show through)
                    num_steps=gen_num_steps,
                    device=device,
                )

                # --- Sampling --- #
                # Fixed seed PER SAMPLER: identical noise for every method/epoch
                torch.manual_seed(42)

                all_samples = []
                remaining = gen_n_samples
                with torch.no_grad():
                    while remaining > 0:
                        n = min(gen_batch_size, remaining)
                        batch = sampler.sample(shape=(n, *gen_img_shape), y=None)
                        all_samples.append(batch.cpu())
                        remaining -= n
                samples = torch.cat(all_samples, dim=0)

                sample_path = samples_dir / f"epoch_{epoch:04d}_{gen_method}.png"
                train_dataset._visGridImage(samples[:64], ncol=8, save_path=sample_path)  # fmt: skip

                # --- MMD evaluation (per method) --- #
                fake_flat = samples.flatten(start_dim=1).numpy()
                current_mmd[gen_method] = compute_mmd(real_flat, fake_flat)
                mmd_history[gen_method].append(current_mmd[gen_method])
                epoch_bar.write(
                    f"  [eval] MMD[{gen_method}] = {current_mmd[gen_method]:.6f}"
                )

                # --- Arbitrage checks (UPDATED) --- #
                # Back to raw IV units first; clamp guards against negative
                # vols from a still-undertrained model in early epochs.
                gen_iv = train_dataset._denorm(samples).clamp_min(1e-4)
                arb = arbitrage_report(
                    gen_iv, train_dataset.logm_axis, train_dataset.dte_axis
                )
                arb_history[gen_method]["calendar"].append(arb["calendar_cell_rate"])
                arb_history[gen_method]["butterfly"].append(arb["butterfly_cell_rate"])
                epoch_bar.write(
                    f"  [eval] arb[{gen_method}]: calendar {arb['calendar_cell_rate']:.2%}, "
                    f"butterfly {arb['butterfly_cell_rate']:.2%} "
                    f"(real: {arb_real['calendar']:.2%} / {arb_real['butterfly']:.2%})"
                )

            if train_cfg["train"]["ema"]:
                score_net.load_state_dict(saved_weights)  # restore training weights
            torch.set_rng_state(saved_rng)  # restore training RNG (CPU)
            torch.cuda.set_rng_state_all(saved_rng_cuda)  # ...and CUDA (UPDATED)

        # --- History & plot --- #
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        lr_history.append(optimizer.param_groups[0]["lr"])
        if epoch <= 5:
            warmup.step()
        else:
            plateau.step(avg_val)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < prev_lr:
                # patience_counter = 0
                epoch_bar.write(
                    f"  [lr] reduced to {current_lr:.2e} — resetting early-stopping counter"
                )
            prev_lr = current_lr
        save_loss_plot(train_losses, val_losses, save_dir / "loss_curve.png")
        # save_mmd_plot(mmd_history, save_dir / "mmd.png")  # OLD: single curve
        # UPDATED: one curve per sampler on the same axes; `every` puts true
        # training epochs on the x-axis instead of the generation index.
        save_mmd_plot(mmd_history, save_dir / "mmd.png", every=gen_every)
        # UPDATED: arbitrage-violation evolution vs the real-data level
        save_arbitrage_plot(arb_history, arb_real, save_dir / "arbitrage.png", every=gen_every)  # fmt: skip
        save_lr_plot(lr_history, save_dir / "lr_curve.png")
        save_t_histogram(t_samples, save_dir / "t_histogram.png", criterion.eps_t, sde.T, t_sampling)  # fmt: skip

        # --- Epoch bar suffix --- #
        epoch_bar.set_postfix(
            train=f"{avg_train:.4f}",
            val=f"{avg_val:.4f}",
            best=f"{best_val_loss:.4f}",
            # mmd=f"{current_mmd:.4f}" if current_mmd < math.inf else "n/a",  # OLD
            mmd="|".join(  # UPDATED: latest value per sampler
                f"{m}:{v:.3f}" for m, v in current_mmd.items() if v < math.inf
            )
            or "n/a",
        )

        # --- Checkpoint --- #
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": score_net.state_dict(),
            "ema_state": ema.shadow if train_cfg["train"]["ema"] else None,
            "optimizer_state": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "mmd_history": mmd_history,
            "arb_history": arb_history,  # UPDATED: per-sampler arbitrage rates
            "args": vars(args),
            "config": cfg,
        }
        torch.save(ckpt, save_dir / "latest.pt")

        # --- Save best val-loss ckpt --- #
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, save_dir / "best_val.pt")
            epoch_bar.write(
                f"    new best val loss: {best_val_loss:.4f}  (epoch {epoch})"
            )

        # --- Save best mmd ckpt --- #
        # OLD: single best_mmd.pt driven by the one configured sampler.
        # UPDATED: one best_mmd_{method}.pt per sampler — the same model state
        # can be best under one sampler and not another, and comparing the
        # selected checkpoints tells you how much the sampler choice matters.
        for m in gen_methods:
            if current_mmd[m] < best_mmd[m]:
                best_mmd[m] = current_mmd[m]
                torch.save(ckpt, save_dir / f"best_mmd_{m}.pt")
                epoch_bar.write(
                    f"    new best MMD[{m}]: {best_mmd[m]:.6f}  (epoch {epoch})"
                )

    print(f"[train] Done. Best val loss = {best_val_loss:.4f}")
    print(f"[train] Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()
