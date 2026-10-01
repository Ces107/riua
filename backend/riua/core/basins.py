"""Catchment aggregation: rain that falls upstream is what floods downstream.

On 29 October 2024 it barely rained in Paiporta or Catarroja; the water came down the
Rambla del Poyo from Chiva and Cheste. A cell-by-cell map cannot show that. So every
scenario is also integrated over catchments:

  own scope   mean rain over the basin unit itself
  up scope    mean rain over the unit plus everything that drains into it

and each is judged over 1, 3, 6 and 12 h against the AEMET thresholds of the area,
(a) interpolated in duration with a power law between the official 1-h and 12-h values
and (b) reduced by the areal reduction factor of the Spanish drainage standard
(Norma 5.2-IC): K_A = 1 - log10(A) / 15, because a threshold written for a rain gauge
is harder to reach as an average over hundreds of km2.

For the upstream scope the windows are allowed to end up to one response time before
the frame starts (tc = a * A^b hours): water already on its way counts.

Level 5 has a second, hydrological way in: an estimate of the unit peak discharge
(SCS-type losses with runoff threshold P0, then mean net intensity over tc) compared
with the envelope of the largest Mediterranean flash floods, q = 100 * A^-0.4
m3/s/km2 (Gaume et al. 2009). The Poyo flood of 2024 sat on that envelope.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from . import grid
from .risk import LEVELS, Member, decide, dress, member_weight, weighted_quantile, _bw


@dataclass
class BasinSet:
    ids: list[str]
    names: list[str]
    area: np.ndarray        # (B,) km2, own area
    up_area: np.ndarray     # (B,) km2, own + all upstream
    W: np.ndarray           # (B, NY*NX) float32, rows sum to 1: basin-mean operator
    U: np.ndarray           # (B, B) float32, rows sum to 1: area-weighted mean over own + upstream
    t1h: np.ndarray         # (3, B) yellow, orange, red 1-h thresholds representative of the basin
    t12h: np.ndarray        # (3, B)

    @property
    def n(self) -> int:
        return len(self.ids)


def arf(area_km2: np.ndarray) -> np.ndarray:
    """Areal reduction factor of Norma 5.2-IC."""
    a = np.maximum(np.asarray(area_km2, np.float64), 1.0)
    return np.clip(1.0 - np.log10(a) / 15.0, 0.5, 1.0)


def duration_thresholds(t1h: np.ndarray, t12h: np.ndarray, d: float) -> np.ndarray:
    """Power-law interpolation T(d) = T1 * d^alpha through the 1-h and 12-h thresholds."""
    alpha = np.log(t12h / t1h) / np.log(12.0)
    return t1h * d ** alpha


def response_time_h(area_km2: np.ndarray, hp: dict) -> np.ndarray:
    return np.clip(hp["tc_a"] * np.asarray(area_km2, np.float64) ** hp["tc_b"], 1.0, hp["tc_max_h"])


def runoff_mm(p_mm: np.ndarray, p0: float, s: float | None = None) -> np.ndarray:
    """Cumulative runoff for cumulative rain P: (P - P0)^2 / (P - P0 + S) once P > P0 (S = 5 P0: SCS law)."""
    x = np.maximum(p_mm - p0, 0.0)
    return x ** 2 / (x + (5.0 * p0 if s is None else s))


def _rolling(cs: np.ndarray, d: int) -> np.ndarray:
    """cs: cumulative sums (T+1, B). Returns (T, B) sums over the d hours ending at each step."""
    T = cs.shape[0] - 1
    idx = np.arange(1, T + 1)
    return cs[idx] - cs[np.maximum(idx - d, 0)]


def member_basin_ratios(p: np.ndarray, t_end: np.ndarray, frames, bs: BasinSet, params: dict, fam: dict,
                        obs: np.ndarray | None = None, bias: float = 1.0):
    """One scenario -> hazard ratios (4, F, B), plus diagnostics per frame.

    Returns ratio, own12 (F,B) largest 12-h mean rain over the unit, up_best (F,B) the
    largest ratio-driving upstream mean rain (mm) and its duration, q (F,B) unit discharge.
    """
    hp, ex = params["hydro"], params["extreme"]
    T = p.shape[0]
    flat = np.nan_to_num(p.reshape(T, -1), nan=0.0)
    own = flat @ bs.W.T                      # (T, B) mm per hour, mean over the unit
    up = own @ bs.U.T                        # (T, B) mean over unit + upstream
    cs_own = np.concatenate([np.zeros((1, bs.n)), np.cumsum(own, axis=0)])
    cs_up = np.concatenate([np.zeros((1, bs.n)), np.cumsum(up, axis=0)])
    # measured hours (o) and forecast hours (f) apart: family factor and bias act on the forecast only
    o = (np.zeros(T, bool) if obs is None else obs)[:, None]
    cum = lambda a: np.concatenate([np.zeros((1, bs.n)), np.cumsum(a, axis=0)])
    cs_own_o, cs_own_f, cs_up_o, cs_up_f = cum(own * o), cum(own * ~o), cum(up * o), cum(up * ~o)
    tc = response_time_h(bs.up_area, hp)
    k_own, k_up = arf(bs.area), arf(bs.up_area)
    red1, red12 = bs.t1h[2], bs.t12h[2]
    lvl1 = np.vstack([bs.t1h, red1 * ex["x1h"]])       # (4, B)
    lvl12 = np.vstack([bs.t12h, red12 * ex["x12h"]])

    gate_mm = 0.5 * params.get("obs_gate_mm", 20.0)        # basin means are about half the point amounts
    # hourly series of the largest ratio over durations, per scope: (4, T, B)
    r_own = np.zeros((4, T, bs.n))
    r_up = np.zeros((4, T, bs.n))
    for d in hp["durations_h"]:
        if d < 12 and not fam.get("use_1h", True):
            continue  # coarse/long-step scenarios are judged on 12 h only
        s = fam["s12h"] if d >= 6 else fam["s1h"] if d == 1 else 0.5 * (fam["s1h"] + fam["s12h"])
        thr = duration_thresholds(lvl1, lvl12, float(d))          # (4, B)
        # rain already fallen counts only in so far as the scenario still brings rain (see risk.predictors)
        fo, fu = s * _rolling(cs_own_f, d), s * _rolling(cs_up_f, d)
        go, gu = np.clip(fo / gate_mm, 0.0, 1.0), np.clip(fu / gate_mm, 0.0, 1.0)
        r_own = np.maximum(r_own, (go * _rolling(cs_own_o, d) + bias * fo)[None] / (thr * k_own)[:, None, :])
        r_up = np.maximum(r_up, (gu * _rolling(cs_up_o, d) + bias * fu)[None] / (thr * k_up)[:, None, :])

    # unit peak discharge from the upstream-scope rain: losses on the trailing 24 h total
    wet_o, wet_f = _rolling(cs_up_o, 24), fam["s12h"] * _rolling(cs_up_f, 24)
    e = runoff_mm(wet_o + wet_f, hp["p0_mm"], hp.get("s_mm"))   # (T, B) cumulative runoff of the event so far
    share_f = wet_f / np.maximum(wet_o + wet_f, 1e-9)
    tci = np.maximum(np.rint(tc).astype(int), 1)
    q = np.zeros((T, bs.n))
    for lag in np.unique(tci):
        g = tci == lag
        prev = np.zeros((T, int(g.sum())))
        if lag < T:
            prev[lag:] = e[:-lag][:, g]
        q[:, g] = np.maximum(e[:, g] - prev, 0.0) / lag * 0.278     # mm/h -> m3/s/km2
    q_env = ex["envelope_c"] * np.maximum(bs.up_area, 1.0) ** ex["envelope_exp"]
    r_q = q / (ex["envelope_fraction"] * q_env) * (1.0 + (bias - 1.0) * share_f)

    F = len(frames)
    ratio = np.zeros((4, F, bs.n), np.float32)
    own12 = np.zeros((F, bs.n), np.float32)
    up12 = np.zeros((F, bs.n), np.float32)
    qf = np.zeros((F, bs.n), np.float32)
    share_up = np.zeros((F, bs.n), np.float32)
    valid = np.zeros(F, bool)
    roll12_own, roll12_up = _rolling(cs_own, 12), _rolling(cs_up, 12)
    hours_before = np.ceil(tc).astype(int)
    for f, (t0, t1) in enumerate(frames):
        sel = (t_end > t0) & (t_end <= t1)
        if not sel.any():
            continue
        valid[f] = True
        ro = r_own[:, sel].max(axis=1)                          # (4, B)
        # upstream windows may end up to tc hours before the frame starts
        ru = np.zeros((4, bs.n))
        qq = np.zeros(bs.n)
        for hb in np.unique(hours_before):
            g = hours_before == hb
            selb = (t_end > t0 - np.timedelta64(int(hb), "h")) & (t_end <= t1)
            ru[:, g] = r_up[:, selb][:, :, g].max(axis=1)
            qq[g] = q[selb][:, g].max(axis=0)
            ru[3, g] = np.maximum(ru[3, g], r_q[selb][:, g].max(axis=0))
        ratio[:, f] = np.maximum(ro, ru)
        own12[f] = roll12_own[sel].max(axis=0)
        up12[f] = roll12_up[sel].max(axis=0)
        qf[f] = qq
        share_up[f] = (ru[2] > ro[2]).astype(np.float32)  # is upstream rain the bigger problem (at red scale)?
    return ratio, own12, up12, qf, share_up, valid


@dataclass
class BasinProduct:
    prob: np.ndarray      # (4, F, B)
    level: np.ndarray     # (F, B)
    own12: np.ndarray     # (2, F, B) median / p90 of the 12-h mean rain over the unit (mm)
    up12: np.ndarray      # (2, F, B) same over unit + upstream
    q: np.ndarray         # (2, F, B) median / p90 unit peak discharge (m3/s/km2)
    upstream_driven: np.ndarray  # (F, B) weighted share of scenarios where upstream rain dominates
    valid: np.ndarray


def basin_product(members: list[Member], frames, bs: BasinSet, params: dict, horizon: str,
                  now: datetime) -> BasinProduct:
    F = len(frames)
    sigma, bias = params["sigma"][horizon], params["bias"][horizon]
    shift_km = params["basin_shift_km"][horizon]
    dj = int(round(shift_km / grid.DY_KM))
    di = int(round(shift_km / grid.DX_KM))
    psum = np.zeros((4, F, bs.n))
    wsum = np.zeros(F)
    O, Uu, Q, S, Ws = [], [], [], [], []
    for m in members:
        w = member_weight(m, params, horizon, now)
        if w <= 0:
            continue
        fam = params["families"][m.family]
        # displacement uncertainty: a convection-permitting scenario is also tried
        # moved one step N, S, E and W; the unmoved field keeps 40 % of the weight.
        if m.family in ("cp", "radar") and (dj or di):
            variants = [(0, 0, 0.4), (dj, 0, 0.15), (-dj, 0, 0.15), (0, di, 0.15), (0, -di, 0.15)]
        else:
            variants = [(0, 0, 1.0)]
        for sj, si, share in variants:
            ratio, own12, up12, q, su, valid = member_basin_ratios(
                grid.shift(m.area, sj, si), m.t_end, frames, bs, params, fam, obs=m.observed(), bias=bias)
            if not valid.any():
                break
            wv = w * share * valid
            psum += wv[None, :, None] * dress(ratio, sigma)      # the bias is already inside the ratio
            wsum += wv
            O.append(own12); Uu.append(up12); Q.append(q); S.append(su); Ws.append(wv)
    ok = wsum > 0
    den = np.where(ok, wsum, 1.0)
    prob = np.minimum.accumulate((psum / den[None, :, None]).astype(np.float32), axis=0)
    level = decide(prob, params["tau"][horizon])
    level[~ok] = 0
    z2 = np.zeros((2, F, bs.n), np.float32)
    if not Ws:
        return BasinProduct(prob, level, z2, z2.copy(), z2.copy(), np.zeros((F, bs.n), np.float32), ok)
    ws = np.stack(Ws)                                   # (M, F)
    def q2(lst):
        v = np.stack(lst)                               # (M, F, B)
        out = np.zeros((2, F, bs.n), np.float32)
        for f in range(F):
            use = ws[:, f] > 0
            if use.any():
                out[:, f] = weighted_quantile(v[use, f], _bw(ws[use, f], (bs.n,)), (0.5, 0.9))
        return out
    s = np.stack(S)
    upd = (s * ws[:, :, None]).sum(axis=0) / den[:, None]
    return BasinProduct(prob, level, q2(O), q2(Uu), q2(Q), upd.astype(np.float32), ok)
