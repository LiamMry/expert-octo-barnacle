# Project Plan: Diffusion-Based Generative Modeling of the Implied Volatility Surface

**Goal.** Train a score-based / DDPM generative model to produce arbitrage-consistent implied
volatility surfaces directly from historical SPX option data, benchmark it against Heston and
rough Bergomi calibration, and (stretch goal) extend it into a DPS-style guided-sampling daily
calibration tool.

**Positioning.** This is the desk-quant / derivatives flagship. Ship Phases 0–8 to a complete,
benchmarked state first — that alone is a strong, self-contained project and paper. Phase 9 is a
genuine research extension; attempt it only once Phases 0–8 are done and written up.

---

## Phase 0 — Environment & Tooling Setup

- [x] Repo skeleton
- [ ] Environment: PyTorch (your existing stack), `wandb` or `mlflow` for experiment tracking
- [ ] Baseline pricing/vol libraries: `py_vollib` or `QuantLib` (Black-Scholes inversion), a
      Heston pricer (own implementation via characteristic functions, or `QuantLib`), a rough
      Bergomi simulator

---

## Phase 1 — Data Acquisition

**Primary option (if accessible):** OptionMetrics via WRDS. This is the dataset used in most of the papers uses.

**Fallback options (fully public, no institutional access needed):**
- CBOE DataShop — free historical SPX options snapshots (limited history/granularity vs. OptionMetrics)
- `yfinance` option chains — free, but only current/near-term chains, no deep historical surfaces
- **Deribit API (BTC/ETH options)** — fully free, high-quality, liquid, and a legitimate,
  increasingly common dataset in the academic literature; a good honest substitute if SPX history
  is unavailable, and arguably a differentiator since most competing papers use SPX only

**Decision point:** confirm data source before writing any model code — the conditioning
variables and preprocessing in Phase 2 depend on what you can actually get.

**Exit criterion:** raw historical option chain data (price, strike, maturity, type, underlying
price, date) downloaded and stored, spanning at minimum 1–2 years for a reasonable train/val/test split.

---

## Phase 2 — Data Cleaning & Preprocessing

- [ ] Filter for liquidity: minimum open interest / volume thresholds, remove stale quotes
- [ ] Filter maturity/moneyness ranges to a sensible grid (e.g. 1 week – 1 year, 0.7–1.3 moneyness)
- [ ] Convert mid-prices to implied volatility via Black-Scholes inversion (Newton-Raphson or Brent)
- [ ] Detect and handle static arbitrage violations in the raw quotes (calendar spread, butterfly
      spread violations) — either discard or use as an early no-arbitrage sanity check
- [ ] Interpolate/regularize onto a fixed (strike, maturity) grid per trading day, OR keep as
      irregular point sets if using a point-cloud / attention-based architecture instead of a
      fixed-grid "image" representation — **this choice determines your Phase 5 architecture**,
      decide it here
- [ ] Train / validation / test split **by date**, not randomly (avoid look-ahead leakage —
      surfaces are highly autocorrelated day-to-day)

**Exit criterion:** a clean tensor dataset of daily implied volatility surfaces (or raw option
point sets), with an explicit, documented train/val/test date split.

---

## Phase 3 — Baseline Models (for later comparison)

- [ ] Implement Heston calibration to the market surface (per-day least-squares fit via
      characteristic-function pricing) — this is your existing planned flagship, reuse it here as
      the baseline rather than duplicating effort
- [ ] Implement rough Bergomi calibration (hybrid scheme simulation + optimization, or a fast
      neural surrogate à la Horvath–Muguruza–Tomas if calibration speed becomes a bottleneck)
- [ ] Record baseline repricing RMSE and stability-across-days metrics for both models on your
      dataset — these numbers are what the diffusion model needs to beat or match

**Exit criterion:** two working baseline calibrators with logged benchmark numbers on your actual
dataset (not just literature numbers — you need them computed on *your* train/test split).

---

## Phase 4 — Synthetic Sanity Check (before touching real data)

- [ ] Generate synthetic surfaces from your Heston and rough Bergomi implementations across a
      range of parameters
- [ ] Train the diffusion model (Phase 5 architecture) on this synthetic data only
- [ ] Verify the model recovers known qualitative features: smile curvature, skew direction,
      term-structure decay, without ever having seen real market data
- [ ] Sanity-check no-arbitrage: do generated synthetic surfaces respect butterfly/calendar
      constraints, even before you've added explicit penalties for it?

**Why this step matters:** if the architecture can't reproduce surfaces from a *known* generative
process, debugging on real (noisy, irregular) market data will be much harder to diagnose. This
step isolates architecture bugs from data bugs.

**Exit criterion:** diffusion model trained on synthetic data visually/quantitatively reproduces
known Heston/rough Bergomi surface shapes.

---

## Phase 5 — Diffusion Model Architecture & Arbitrage-Aware Training

- [ ] Choose representation: fixed-grid "image" (strike × maturity, vol as pixel value) vs.
      point-set/attention-based — grid is simpler and matches most prior work (easier to compare
      against), point-set is more faithful to irregular real quote grids
- [ ] Choose conditioning: date/regime features, underlying spot level, realized vol, VIX level —
      decide what the model conditions on vs. generates unconditionally
- [ ] Score network architecture: U-Net (if grid representation) or a transformer/set-based score
      network (if point-set) — given your background, this is the most familiar part of the whole
      project
- [ ] Noise schedule: reuse your existing VP/VE-SDE expertise directly here — no new theory needed
- [ ] Arbitrage-aware loss terms: add butterfly-spread and calendar-spread soft penalties to the
      training objective (same idea as the Dupire-regularized local-vol papers), or explore hard
      architectural constraints if soft penalties prove insufficient
- [ ] Training loop, logging, checkpointing

**Exit criterion:** training pipeline runs end-to-end on synthetic data (reuse Phase 4 as the
integration test) with logged loss curves and arbitrage-violation metrics tracked during training.

---

## Phase 6 — Training on Real Market Data

- [ ] Train the same architecture (minimal changes from Phase 4/5) on the real preprocessed
      dataset from Phase 2
- [ ] Monitor arbitrage-violation rate on generated samples throughout training, not just at the end
- [ ] Generate held-out-date surfaces and visually compare against actual market surfaces for
      those dates

**Exit criterion:** model generates visually plausible, day-conditioned (or unconditional but
realistic) surfaces on real data, with arbitrage-violation rate quantified.

---

## Phase 7 — Arbitrage & Stability Validation

- [ ] Formal no-arbitrage check on generated surfaces: convert to prices, verify monotonicity in
      strike, convexity (butterfly), monotonicity in maturity (calendar) — quantify violation rate
      and magnitude
- [ ] Day-to-day stability: generate surfaces for consecutive trading days, check the calibrated
      surface doesn't jump unrealistically (same diagnostic used in the Deep Local Volatility papers)
- [ ] Compare arbitrage-violation rate against the Heston/rough Bergomi baselines (parametric
      models are arbitrage-free *by construction* — this is a real disadvantage of the generative
      approach you should report honestly, not hide)

**Exit criterion:** a table quantifying arbitrage violations, and stability across time, for the
diffusion model vs. both baselines.

---

## Phase 8 — Benchmarking Against Baselines (this is the paper's core results table)

- [ ] Repricing RMSE: diffusion model vs. Heston vs. rough Bergomi, on held-out strikes/maturities
- [ ] Out-of-sample generalization: performance on the held-out date range from Phase 2's split
- [ ] Greeks accuracy: compute Delta/Gamma/Vega from the generated surfaces (via the
      pathwise/likelihood-ratio estimators from your own derivations) and compare against baseline
      model Greeks
- [ ] Calibration speed: wall-clock time to produce a calibrated surface for a new day, diffusion
      vs. Heston/rough Bergomi optimization
- [ ] Honest limitations section: where does the diffusion model underperform? (Likely candidates:
      extreme/illiquid strikes with sparse training data, arbitrage violations vs. parametric
      models, interpretability of what's driving a given generated shape)

**Exit criterion:** the core results table and honest limitations discussion — this is the
deliverable that makes the project interview-ready even if Phase 9 never happens.

---

## Phase 9 (Stretch) — DPS-Style Guided Daily Calibration

Only attempt after Phase 8 is complete and written up.

- [ ] Formulate daily calibration as posterior sampling: pretrained diffusion prior over surfaces,
      guided at inference time by that day's actual sparse observed quotes
- [ ] Build the guidance gradient (residual between generated surface's implied prices and observed
      market quotes) — directly analogous to your DPS guidance gradient for CT reconstruction
- [ ] Compare guided-sampling calibration against direct Heston/rough Bergomi optimization on the
      same day's data: does guidance-based calibration match observed quotes as well, faster, or
      with better generalization to unquoted strikes?
- [ ] If covariate shift appears (guided trajectories diverge from the unconditional model's
      training marginals), apply your backward-calibration correction and quantify the improvement

**Exit criterion:** a working guided-calibration pipeline with a quantitative comparison against
Phase 8's direct-optimization baselines — this becomes the paper's "extension" section.

---

## Phase 10 — Writeup & Publication

- [ ] 6–10 page paper: question, method, baselines, results, honest limitations (your existing
      writing standard)
- [ ] Clean GitHub repo with README, reproducible training scripts, and the benchmark table
- [ ] Link from your academic site (liammry.github.io)
- [ ] Prepare the 2–3 minute interview pitch version of this project, per your interview-prep plan

---

## Risk Notes (revisit if timeline slips)

- **Data access risk (Phase 1)** is the single biggest unknown — resolve it in week 1, not after
  architecture work has started, since it determines the representation choice in Phase 5.
- **Arbitrage constraints (Phase 5/7)** are the hardest technical part — budget real time here;
  soft penalties may not fully eliminate violations, and that's a legitimate, reportable limitation
  rather than a failure.
- **Phase 9 is genuinely open-ended research** — treat its exit criterion as "informative result,
  positive or negative," not "must beat the baseline." A clearly-explained negative result here is
  still a strong section of the paper.