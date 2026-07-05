"""
IV-surface dataset for the diffusion project.

Two stages (see roadmap Phase 2):

1.  build_surfaces()  — offline preprocessing. Reads the raw SPY EOD parquet
    files, builds one implied-volatility surface per trading day on a fixed
    (log-moneyness x maturity) grid, and caches everything to a single .pt file.
    Run once:

        python datasets.py --raw "data/SPY Options EOD Data (2010-2023) - raw" \
                           --out data/iv_surfaces.pt

2.  IVSurfaceDataset — a thin torch Dataset that memory-maps the cached tensor
    and yields (surface, cond) per day, with a date-based train/val/test split.
"""

import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


# ------------------------------------------ #
# ----- *.parquet file characteristics ----- #
# ------------------------------------------ #

# Each row = one option contract
#
# [QUOTE_UNIXTIME]      : Quote timestamp, Unix epoch seconds
# [QUOTE_READTIME]      : Same instant, human-readable
# [QUOTE_DATE]          : Quote date
# [QUOTE_TIME_HOURS]    : Hour of day of the snapshot (16.0 = 4 PM close)
# [UNDERLYING_LAST]     : SPY last (spot) price S at quote time
# [EXPIRE_DATE]         : Option expiration date
# [EXPIRE_UNIX]         : Expiration as Unix epoch seconds
# [DTE]                 : Days to expiry (calendar days: EXPIRE − QUOTE)
# [STRIKE]              : Strike price K (the row's contract)
#
## Position helpers
# [STRIKE_DISTANCE]     : Absolute distance of strike from spot, |K − S|
# [STRIKE_DISTANCE_PCT] : Same as a fraction of spot, |K − S| / S
#
## Call side
# [C_BID] / [C_ASK]     : Best bid / ask for the call; mid = (bid+ask)/2
# [C_LAST]              : Last traded price of the call
# [C_SIZE]              : Bid size × ask size as a string, e.g. "305 x 270" (contracts available at bid / ask)
# [C_VOLUME]            : Contracts traded that day
# [C_IV]                : Implied volatility (annualized) backed out from the call price
# [C_DELTA]             : ∂price/∂S — sensitivity to spot (calls: 0→1)
# [C_GAMMA]             : ∂delta/∂S — curvature (shared magnitude with the put)
# [C_VEGA]              : ∂price/∂σ — sensitivity to implied vol
# [C_THETA]             : ∂price/∂t — time decay (per day, negative)
# [C_RHO]               : ∂price/∂r — sensitivity to interest rate
#
## Put side (identical to the call side)

# -------------------------------------------- #
# ----- Helpers functions (load / clean) ----- #
# -------------------------------------------- #


def load_clean_chains(root_dir: str) -> pd.DataFrame:
    """Load the raw .parquet files, clean them and return a single frame"""
    files = os.listdir(root_dir)
    l_df = [filter(pd.read_parquet(os.path.join(root_dir, f))) for f in files]

    return pd.concat(l_df, ignore_index=True)


def filter(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe"""

    S = df["[UNDERLYING_LAST]"]  # Spot
    logm = np.log(df["[STRIKE]"] / S)  # Moneyness

    otm_put = logm < 0.0
    bid = np.where(otm_put, df["[P_BID]"], df["[C_BID]"])
    ask = np.where(otm_put, df["[P_ASK]"], df["[C_ASK]"])
    iv = np.where(otm_put, df["[P_IV]"], df["[C_IV]"])

    mid = (bid + ask) / 2.0
    rel_spread = (ask - bid) / np.where(mid > 0, mid, np.nan)

    keep = (
        np.isfinite(iv)
        & (iv > 1e-3)
        & (iv < 5.0)  # sane implied volatility
        & (bid > 0.0)
        & (ask > 0.0)
        & (ask >= bid)  # two-sided quote
        & (rel_spread < 1.0)  # not blown-out / stale
        & (df["[DTE]"] >= 7.0)  # drop expiry-day noise
    )

    return df[keep].copy()


# ----------------------------------- #
# ----- Grid params + functions ----- #
# ----------------------------------- #

LOGM_MIN, LOGM_MAX = -0.35, 0.25  # log(K/S): ~0.70x .. 1.28x moneyness
DTE_MIN, DTE_MAX = 7.0, 365.0  # 1 week .. 1 year to expiry
GRID_H, GRID_W = 32, 32  # (maturity, moneyness) resolution


def day_surface(
    day: pd.DataFrame, gm: np.ndarray, gd: np.ndarray
) -> Optional[np.ndarray]:
    """
    Build one (GRID_H, GRID_W) IV surface from a single quote date's option
    chain, or return None if the day has too few usable quotes.

    gm, gd are the meshgrid target coordinates (log-moneyness, dte).
    """
    spot = day["[UNDERLYING_LAST]"].iloc[0]
    logm = np.log(day["[STRIKE]"].to_numpy() / spot)

    # OTM side: puts below spot, calls above — the liquid, reliable quotes.
    iv = np.where(logm < 0.0, day["[P_IV]"].to_numpy(), day["[C_IV]"].to_numpy())  # fmt: skip
    dte = day["[DTE]"].to_numpy()

    # keep only sane, in-window points
    ok = (
        np.isfinite(iv)
        & (logm >= LOGM_MIN)
        & (logm <= LOGM_MAX)  # grid window
        & (dte >= DTE_MIN)
        & (dte <= DTE_MAX)  # grid window
    )
    logm, dte, iv = logm[ok], dte[ok], iv[ok]
    if logm.size < 50:
        # Too few points
        return None

    pts = np.column_stack([logm, dte])
    # linear interpolation inside the convex hull, nearest to fill the edges
    surf = griddata(pts, iv, (gm, gd), method="linear")
    holes = ~np.isfinite(surf)
    if holes.any():
        surf[holes] = griddata(pts, iv, (gm[holes], gd[holes]), method="nearest")
    return surf.astype(np.float32)


def create_surfaces(filepath: str, outpath: str) -> None:

    df = pd.read_parquet(os.path.join(filepath))

    m_axis = np.linspace(LOGM_MIN, LOGM_MAX, GRID_W)
    d_axis = np.linspace(DTE_MIN, DTE_MAX, GRID_H)
    gm, gd = np.meshgrid(m_axis, d_axis)  # (GRID_H, GRID_W)

    surfaces, dates = [], []
    for date, day in df.groupby("[QUOTE_DATE]", sort=True):
        # `day` is a sub-DataFrame containing only that date's rows
        surf = day_surface(day, gm, gd)
        if surf is None:
            continue
        surfaces.append(surf)
        dates.append(date)

    surfaces = torch.from_numpy(np.stack(surfaces))[:, None]  # (N,1,H,W)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    torch.save(
        {
            "surfaces": surfaces,
            "dates": dates,
            "logm_axis": m_axis.astype(np.float32),
            "dte_axis": d_axis.astype(np.float32),
        },
        outpath,
    )

    return None


class IVSurfaceDataset(Dataset):
    """
    Daily implied-volatility surfaces for diffusion training.

    surface : (1, GRID_H, GRID_W) float32  -- IV grid, optionally normalised

    Split is by date (chronological), never random — consecutive surfaces are
    highly correlated, so a random split would leak future into past.
    """

    def __init__(
        self,
        path: str = None,
        split: str = "train",
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        normalize: bool = True,
    ):
        if path is not None:
            data_dict: dict = torch.load(path, weights_only=False)
            surfaces: torch.Tensor = data_dict["surfaces"]
            dates: np.ndarray = np.array(data_dict["dates"])
            # UPDATED: keep the grid axes — needed by eval.arbitrage_report
            self.logm_axis = np.asarray(data_dict["logm_axis"])
            self.dte_axis = np.asarray(data_dict["dte_axis"])

            # chronological split
            order = np.argsort(dates)
            n = len(order)
            n_test = int(round(test_frac * n))
            n_val = int(round(val_frac * n))
            n_train = n - n_val - n_test
            bounds = {
                "train": order[:n_train],
                "valid": order[n_train : n_train + n_val],
                "test": order[n_train + n_val :],
            }
            if split not in bounds:
                raise ValueError(f"split must be one of {list(bounds)}, got {split}")
            idx = bounds[split]

            self.surfaces = surfaces[idx]
            self.dates = dates[idx]

            # Standardise (z-score) using TRAIN-set stats only (fit once, reuse).
            #
            # UPDATED — why z-score instead of min/max scaling:
            # iv_max ≈ 1.44 comes from a single crisis spike (COVID 2020) while
            # typical IV ≈ 0.2, so min/max scaling squashed the whole dataset
            # into a thin sliver around -0.7 with std ≈ 0.12. The diffusion
            # prior is N(0, 1): the model had to collapse that prior into ~6%
            # of its variance, so any residual network/sampler error was huge
            # relative to the signal — visible as speckle in generated
            # surfaces, and as near-constant colors in the sample grids.
            # Z-scoring gives the data zero mean / unit variance, i.e. the same
            # scale the prior and the ε-target live on.
            self.normalize = normalize
            if normalize:
                train_surf = surfaces[bounds["train"]]
                # OLD (min/max scaling):
                # self.iv_min = float(train_surf.min())
                # self.iv_max = float(train_surf.max())
                self.iv_mean = float(train_surf.mean())
                self.iv_std = float(train_surf.std())

    def __len__(self) -> int:
        return self.surfaces.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        surf = self.surfaces[i]
        date = self.dates[i]
        if self.normalize:
            surf = self._norm(surf)
        return surf, date

    ### Helpers function ###

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # OLD (min/max scaling to [-1, 1]):
        # x = (x - self.iv_min) / (self.iv_max - self.iv_min)  # -> [0,1]
        # return x * 2.0 - 1.0  # -> [-1,1]
        # UPDATED: z-score — zero mean, unit variance (see __init__ for why).
        return (x - self.iv_mean) / self.iv_std

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalised surfaces back to implied-vol units (for eval/plots)."""
        return x * self.iv_std + self.iv_mean

    def _visGridImage(
        self,
        bx: torch.Tensor,
        ncol: int = 8,
        save_path: str = None,
    ) -> None:
        """Plot a batch of IV surfaces as a grid of heatmaps."""

        bx = bx.detach().cpu().numpy()

        n = bx.shape[0]
        ncol = min(ncol, n)
        nrow = math.ceil(n / ncol)
        vmin, vmax = float(bx.min()), float(bx.max())

        fig, axes = plt.subplots(
            nrow, ncol, figsize=(1.6 * ncol, 1.6 * nrow), squeeze=False
        )
        im = None
        for i, ax in enumerate(axes.flat):
            if i < n:
                im = ax.imshow(
                    bx[i, 0],
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            ax.axis("off")

        # One shared colorbar in its own axis, so the tile grid stays uniform.
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, w, h]
        fig.colorbar(im, cax=cax, label="implied vol")

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

    def _visGridSurface(
        self,
        bx: torch.Tensor,
        ncol: int = 4,
        save_path: str = None,
    ) -> None:
        """Plot a batch of IV surfaces as a grid of 3-D surfaces."""

        bx = bx.detach().cpu().numpy()

        n = bx.shape[0]
        ncol = min(ncol, n)
        nrow = math.ceil(n / ncol)
        vmin, vmax = float(bx.min()), float(bx.max())

        # real coordinate axes (same grid the surfaces were built on)
        m_axis = np.linspace(LOGM_MIN, LOGM_MAX, GRID_W)
        d_axis = np.linspace(DTE_MIN, DTE_MAX, GRID_H)
        gm, gd = np.meshgrid(m_axis, d_axis)  # (H, W)

        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(3.0 * ncol, 2.6 * nrow),
            squeeze=False,
            subplot_kw={"projection": "3d"},
        )
        for i, ax in enumerate(axes.flat):
            if i < n:
                ax.plot_surface(
                    gm,
                    gd,
                    bx[i, 0],
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    linewidth=0,
                    antialiased=True,
                )
                ax.set_title(str(self.dates[i]), fontsize=7)
                ax.set_xlabel("log-moneyness", fontsize=6)
                ax.set_ylabel("DTE", fontsize=6)
                ax.set_zlim(vmin, vmax)
                ax.tick_params(labelsize=5)
                ax.view_init(elev=25, azim=-60)
            else:
                ax.axis("off")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)


def main():

    # Load raw data, clean them, and save as .parquet file
    if not os.path.exists("./data/spy_clean.parquet"):
        cleaned_df = load_clean_chains("./data/SPY Options EOD Data (2010-2023) - raw")  # fmt: skip
        cleaned_df.to_parquet("data/spy_clean.parquet", index=False)

    # Create surfaces data
    if not os.path.exists("./data/iv_surfaces.pt"):
        create_surfaces(
            filepath="./data/spy_clean.parquet",
            outpath="./data/iv_surfaces.pt",
        )

    dataset = IVSurfaceDataset(path="./data/iv_surfaces.pt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    i, (x, date) = next(enumerate(dataloader))
    dataset._visGridImage(x, 2, "./tmp/check_dataset.png")
    dataset._visGridSurface(x, 2, "./tmp/check_dataset_surf.png")


if __name__ == "__main__":
    main()
