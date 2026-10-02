"""Probabilistic radar nowcast (0-3 h) with pysteps STEPS, and its blending with NWP.

STEPS (Bowler, Pierce and Seed 2006; pysteps: Pulkkinen et al. 2019):
  - motion from Lucas-Kanade optical flow on the last scans,
  - the rain field is split into a cascade of spatial scales; each scale evolves as an
    AR(2) process whose memory is estimated from the scans (small scales forget in
    minutes, large ones in hours), with spatially correlated noise replacing the part
    of the field that is no longer predictable,
  - the advection velocity is perturbed too (growing error with lead time),
  - each ensemble member is forced back to the observed rain-rate distribution.
The spread of the ensemble is therefore a statement about predictability, not a guess.

Extrapolation does not create or dissolve storms, so its skill for convection is gone
after 1-3 h. Beyond that the nowcast hands over to the convection-permitting model
runs with a weight that decays with lead time (see `blend_weights`).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from . import qpe

STEP_MIN = 10
RAIN_THR = 0.1        # mm/h
ZEROVALUE_DB = -15.0


def steps_ensemble(rates: list[tuple[datetime, np.ndarray]], n_steps: int = 18, n_members: int = 20,
                   seed: int | None = None, workers: int = 2) -> dict:
    """rates: at least 3 rain-rate scans, 10 min apart, oldest first (mm/h, NaN = no coverage).

    Returns dict(t0, rate (M, n_steps, ny, nx) mm/h float32, motion, method, wet_fraction).
    With (almost) no rain on the radar the result is an all-zero deterministic member.
    """
    from pysteps import nowcasts
    from pysteps.utils import transformation
    rates = sorted(rates, key=lambda x: x[0])[-3:]
    t0 = rates[-1][0]
    stack = np.stack([np.nan_to_num(r, nan=0.0) for _, r in rates]).astype(np.float64)
    wet = float((stack[-1] >= RAIN_THR).mean())
    ny, nx = stack.shape[1:]
    regular = len(rates) == 3 and all(
        abs((rates[k + 1][0] - rates[k][0]).total_seconds() / 60.0 - STEP_MIN) < 2.5 for k in range(2))
    if wet < 0.003 or not regular:
        return dict(t0=t0, rate=np.zeros((1, n_steps, ny, nx), np.float32), motion=np.zeros((2, ny, nx)),
                    method="none (no significant echo)" if regular else "none (irregular scans)", wet_fraction=wet)
    v = qpe.motion_field([r for _, r in rates], coarse=1)      # the caller already works on a 2-km grid
    db, _ = transformation.dB_transform(stack, threshold=RAIN_THR, zerovalue=ZEROVALUE_DB)
    method = "pysteps STEPS"
    try:
        fc = nowcasts.get_method("steps")(
            db, v, n_steps, n_ens_members=n_members, n_cascade_levels=6, precip_thr=10.0 * np.log10(RAIN_THR),
            kmperpixel=1.0, timestep=STEP_MIN, noise_method="nonparametric", vel_pert_method="bps",
            mask_method="incremental", probmatching_method="cdf", seed=seed, num_workers=workers)
        rate = transformation.dB_transform(fc, threshold=10.0 * np.log10(RAIN_THR), inverse=True)[0]
    except Exception as e:
        # too little rain for a stable cascade / noise estimate: plain semi-Lagrangian extrapolation
        from pysteps import extrapolation
        det = extrapolation.get_method("semilagrangian")(stack[-1], v, n_steps, outval=0.0)
        rate = det[None]
        method = f"extrapolation only ({type(e).__name__})"
    rate = np.nan_to_num(rate, nan=0.0).astype(np.float32)
    rate[rate < RAIN_THR] = 0.0
    return dict(t0=t0, rate=rate, motion=v, method=method, wet_fraction=wet)


def hourly_from_steps(rate: np.ndarray, t0: datetime, already_mm: np.ndarray | None = None,
                      minutes_done: float = 0.0) -> tuple[list[datetime], np.ndarray]:
    """(M, n_steps, ny, nx) 10-min rain rates -> hourly accumulations on clock hours.

    The first clock hour is usually under way: `already_mm` (ny, nx) is the observed
    accumulation since the top of the hour and `minutes_done` how long that is.
    Returns (t_end list, acc (M, H, ny, nx) mm).
    """
    M, n, ny, nx = rate.shape
    step_h = STEP_MIN / 60.0
    top = t0.replace(minute=0, second=0, microsecond=0)
    t_ends, accs = [], []
    k = 0
    hour_end = top + timedelta(hours=1)
    cur = np.zeros((M, ny, nx), np.float32)
    if already_mm is not None:
        cur += np.nan_to_num(already_mm, nan=0.0)[None]
    t = t0
    while k < n:
        t = t + timedelta(minutes=STEP_MIN)
        cur += rate[:, k] * step_h
        k += 1
        if t >= hour_end:
            t_ends.append(hour_end); accs.append(cur)
            cur = np.zeros((M, ny, nx), np.float32)
            hour_end = hour_end + timedelta(hours=1)
    return t_ends, (np.stack(accs, axis=1) if accs else np.zeros((M, 0, ny, nx), np.float32))


def blend_weights(lead_h: np.ndarray, full_until_h: float = 1.0, zero_at_h: float = 4.0) -> np.ndarray:
    """Weight of the radar extrapolation against NWP as a function of lead time (hours).

    1 up to `full_until_h`, then a smooth (cosine) decay to 0 at `zero_at_h`. The two
    numbers are tuned on the hindcast as the lead times where the extrapolation's
    fractions skill score drops below that of the model.
    """
    x = np.clip((np.asarray(lead_h, float) - full_until_h) / max(zero_at_h - full_until_h, 1e-6), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def to_analysis_max(field: np.ndarray) -> np.ndarray:
    """(..., 340, 320) -> (..., 68, 64): maximum of each 5 x 5 block (event-in-cell semantics)."""
    s = field.shape
    return field.reshape(*s[:-2], s[-2] // 5, 5, s[-1] // 5, 5).max(axis=(-3, -1))


def to_analysis_mean(field: np.ndarray) -> np.ndarray:
    s = field.shape
    return np.nanmean(field.reshape(*s[:-2], s[-2] // 5, 5, s[-1] // 5, 5), axis=(-3, -1))
