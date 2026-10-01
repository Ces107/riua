"""EFI / SOT — reviewed copy of ``backend/riua/core/efi.py`` (the original is left untouched).

Checked against ECMWF's own reference implementation, ``earthkit-meteo``
(``src/earthkit/meteo/extreme/array/efi.py`` and ``sot.py``, fetched 2026-10-01 from GitHub, copy kept in
``scratch/c1-climate/ek_*.py``), which implements Lalaurette (2003) with the Anderson-Darling weight of
Zsoter (2006):

    EFI = (2 / pi) * integral_0^1 (p - F_f(p)) / sqrt(p (1 - p)) dp ,   F_f(p) = fraction of members <= q_c(p)
    SOT = (Q_f(90) - Q_c(99)) / (Q_c(99) - Q_c(90))

What was wrong / fragile in the original (details and numbers in ``backend/riua/static/climate_README.md``):

1. Dry climates (the important one for precipitation in a Mediterranean climate). Where the climate quantile is 0
   for a large part of the distribution, ``F_f`` jumps between 0 and 1 for trivial amounts, so the plain formula
   answers "is it raining at all?" rather than "is the rain extreme?". ECMWF integrates only over the part of the
   climate distribution that is wet (``q_c > eps``) and normalises by the largest value the integral can take over
   that part (``eps > 0`` branch of earthkit's ``efi``). Implemented here as ``eps`` (default 1 mm for rain).
2. Truncated integral. The original integrates with the trapezoid rule from p = 0.01 to p = 0.999 only. The
   weight 1/sqrt(p(1-p)) is singular (integrable) at both ends, and [0.999, 1] + [0, 0.01] carry 2.9 % + 0.04 %
   of the maximum, while the trapezoid rule over a convex integrand overshoots elsewhere. Here the integral is done
   analytically on each interval with F_f linear in p (exactly ECMWF's scheme, generalised to unequal steps) and
   the range is closed with p = 0 and p = 1.
3. Climate maximum. ECMWF's top node is the M-climate maximum (percentile 100). ``CLIM_P`` stops at 0.999; the
   climatology file also carries the pooled maximum (``pr_max_*``), pass it as ``clim_max``; without it the
   99.9th percentile is used as the top node (EFI reaches +1 when every member exceeds it).
4. SOT. Formula correct. The original floors the denominator at 0.5 mm and does not bound the result, so a
   5 mm forecast in a dry-season climate (Q_c(99) ~ Q_c(90) ~ 0) returns SOT = 10 or more. ECMWF returns
   missing where the denominator vanishes and clips to [-10, 10]. Here: NaN below ``min_den`` (or the old floor
   with ``mode="floor"``), clipped to +-10, and amounts below ``eps`` set to 0 first, as ECMWF does.

Not an error but a caveat that code cannot fix: the reference climate is ERA5 (IFS 41r2, 31 km), not the
ensemble's own reforecast climate (the ENS has been 9 km since June 2023 and is only regridded to 0.25 deg).
A finer model makes heavier rain, so EFI/SOT against ERA5 are biased high in the tail; see the README.
"""
from __future__ import annotations

import numpy as np

CLIM_P = np.concatenate([np.arange(1, 100) / 100.0, [0.995, 0.999]])  # same vector as backend/riua/core/efi.py


def _nodes(members, clim_q, clim_p, clim_max):
    """Sorted p nodes closed with 0 and 1 and F_f at each node."""
    m = np.asarray(members, np.float64)
    q = np.asarray(clim_q, np.float64)
    p = np.asarray(clim_p, np.float64)
    M = m.shape[0]
    frac = np.empty(q.shape, np.float64)
    for k in range(q.shape[0]):
        frac[k] = (m <= q[k][None]).sum(axis=0) / M
    # p = 0 node: by construction nothing lies below the climate minimum that is not also <= q(p_first);
    # holding F_f constant on [0, p_first] is the conservative choice (the weight there is negligible for p_first=0.01)
    q_top = q[-1] if clim_max is None else np.maximum(np.asarray(clim_max, np.float64), q[-1])
    f_top = (m <= q_top[None]).sum(axis=0) / M
    pp = np.concatenate([[0.0], p, [1.0]])
    ff = np.concatenate([frac[:1], frac, f_top[None]])
    qq = np.concatenate([q[:1], q, q_top[None]])
    return pp, ff, qq


def efi(members: np.ndarray, clim_q: np.ndarray, clim_p: np.ndarray = CLIM_P, eps: float = 1.0,
        clim_max: np.ndarray | None = None) -> np.ndarray:
    """Extreme Forecast Index.

    members (M, ...), clim_q (K, ...) climate quantiles at levels ``clim_p`` (ascending, inside (0, 1)).
    ``eps`` > 0: precipitation mode (ECMWF): only the part of the climate distribution with q_c > eps (same units
    as the data) is integrated and the result is normalised by the maximum attainable there. ``eps`` <= 0: plain
    formula over the whole distribution (temperature, TCWV...).
    ``clim_max`` (...): climate maximum, used as the p = 1 node.
    Returns EFI (...) in [-1, 1]; NaN where the climate is dry at every stored level (eps mode).
    """
    pp, ff, qq = _nodes(members, clim_q, clim_p, clim_max)
    A = np.arcsin(np.sqrt(pp))            # d/dp = 1 / (2 sqrt(p (1 - p)))
    S = np.sqrt(pp * (1.0 - pp))
    dA, dS, dp = np.diff(A), np.diff(S), np.diff(pp)
    shape = ff.shape[1:]
    total = np.zeros(shape)
    vmax = np.zeros(shape)
    for i in range(len(dp)):
        s = (ff[i + 1] - ff[i]) / dp[i]   # F_f linear in p on [p_i, p_{i+1}]
        # integral of (p - F_f) / sqrt(p (1 - p)) over the interval, in closed form
        d = (1.0 - 2.0 * ff[i]) * dA[i] - dS[i] + s * ((2.0 * pp[i] - 1.0) * dA[i] + dS[i])
        dmax = dA[i] - dS[i]              # same integral with F_f = 0 (every member above)
        if eps > 0:
            wet = qq[i + 1] > eps
            total += np.where(wet, d, 0.0)
            vmax += np.where(wet, dmax, 0.0)
        else:
            total += d
            vmax += dmax
    out = np.where(vmax > 1e-9, total / np.maximum(vmax, 1e-9), np.nan)
    return np.clip(out, -1.0, 1.0)


def sot(members: np.ndarray, clim_q: np.ndarray, clim_p: np.ndarray = CLIM_P, q: float = 0.9, q_tail: float = 0.99,
        eps: float = 1.0, min_den: float = 1.0, mode: str = "nan") -> np.ndarray:
    """Shift of Tails, upper tail: (Q_f(q) - Q_c(q_tail)) / (Q_c(q_tail) - Q_c(q)), clipped to [-10, 10].

    ``eps``: amounts below it are set to 0 before taking percentiles (ECMWF practice for rain; <= 0 disables).
    ``min_den``: smallest meaningful denominator (data units). ``mode="nan"``: missing below it (ECMWF);
    ``mode="floor"``: denominator floored at ``min_den`` (behaviour of the original, kept for maps without holes).
    """
    m = np.asarray(members, np.float64)
    cq = np.asarray(clim_q, np.float64)
    if eps > 0:
        m = np.where(m < eps, 0.0, m)
        cq = np.where(cq < eps, 0.0, cq)
    qf = np.quantile(m, q, axis=0)
    p = np.asarray(clim_p, np.float64)
    kt = int(np.argmin(np.abs(p - q_tail)))
    kq = int(np.argmin(np.abs(p - q)))
    den = cq[kt] - cq[kq]
    if mode == "floor":
        out = (qf - cq[kt]) / np.maximum(den, min_den)
    else:
        out = np.where(den >= min_den, (qf - cq[kt]) / np.where(den >= min_den, den, 1.0), np.nan)
    return np.clip(out, -10.0, 10.0)


def clim_at(day_of_year: int, clim_doy: np.ndarray, clim_q_doy: np.ndarray) -> np.ndarray:
    """Climate quantiles for one day of year from the thinned file (nodes every 5 days, circular, period 366).

    clim_doy (D,) node days (0-based, 0 = 1 Jan, 59 = 29 Feb), clim_q_doy (D, K, ...). Returns (K, ...).
    """
    d = np.asarray(clim_doy, np.float64)
    x = float(day_of_year) % 366.0
    j = int(np.searchsorted(d, x, side="right")) - 1
    j2 = (j + 1) % len(d)
    span = (d[j2] - d[j]) % 366.0
    w = ((x - d[j]) % 366.0) / span if span > 0 else 0.0
    return (1.0 - w) * clim_q_doy[j].astype(np.float64) + w * clim_q_doy[j2].astype(np.float64)
