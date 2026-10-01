"""Rain to discharge to water level at the control points (ravines and rivers at towns).

Chain, for every rainfall scenario:

1. Losses, per grid cell. Wetness W is the rain remembered by the soil,
   W(t) = W(t-1) * exp(-1/tau) + p(t), and the share of the hour's rain that runs off
   is the marginal runoff coefficient of the SCS / Norma 5.2-IC loss law
   E = (P - P0)^2 / (P + 4 P0):
       c(W) = (W - P0) (W + 9 P0) / (W + 4 P0)^2   for W > P0, else 0.
   Dry ground absorbs the first P0 mm; ground that has already taken 150 mm sheds
   more than 80 % of what follows.

2. Routing, time-area (isochrone) method. The catchment of each control point was cut
   from the terrain model into travel-time bands. Net rain falling on the band that is
   k hours away arrives k hours later:
       Q(t) = 0.278 * sum_k sum_cells A[k, cell] * net(t - k, cell)      [m3/s]
   (A in km2, net in mm/h), then a linear reservoir (Clark) smooths the wave.

3. Hydraulics. A cross-section cut from LiDAR terrain and Manning's equation give a
   rating curve h(Q), the bankfull capacity, and for larger flows how far above the
   bank the water stands.

Levels at a control point compare the peak discharge of a frame with the channel:
   2  flow >= f2 x capacity      the ravine is running hard
   3  flow >= f3 x capacity      close to the brim
   4  flow >= capacity           overflow
   5  water >= d5 m above the bank (or flow >= f5 x capacity if no section is usable)
and the same probabilistic rule as everywhere else turns scenarios into a level.

What this cannot see: dam releases, blocked bridges (decisive in October 2024),
sewer surcharge, backwater from the sea or from a main river. It is an estimate of
natural-regime runoff, not a hydraulic model of the town.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import sparse
from scipy.signal import lfilter

from . import grid
from .risk import Member, decide, dress, member_weight, weighted_quantile, _bw


@dataclass
class HydroNet:
    ids: list[str]
    area: np.ndarray                 # (P,) km2 contributing (unregulated, inside the box)
    tc_h: np.ndarray                 # (P,)
    lags: list[sparse.csr_matrix]    # lags[k]: (P, NY*NX) km2 arriving k hours later
    q_bankfull: np.ndarray           # (P,) m3/s, NaN if unknown
    rating_q: list[np.ndarray]       # per point, discharge table (m3/s), ascending
    rating_h: list[np.ndarray]       # per point, depth above thalweg (m)
    h_bank: np.ndarray               # (P,) bankfull depth above thalweg (m), NaN if unknown

    @property
    def n(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, ids, area, tc_h, point_idx, cell, lag_h, area_km2, q_bankfull, rating_q, rating_h, h_bank):
        P, n = len(ids), grid.NY * grid.NX
        kmax = int(lag_h.max()) if len(lag_h) else 0
        lags = []
        for k in range(kmax + 1):
            s = lag_h == k
            lags.append(sparse.csr_matrix((area_km2[s].astype(np.float32), (point_idx[s], cell[s])), shape=(P, n)))
        return cls(list(ids), np.asarray(area, float), np.asarray(tc_h, float), lags,
                   np.asarray(q_bankfull, float), list(rating_q), list(rating_h), np.asarray(h_bank, float))


def runoff_coefficient(w_mm: np.ndarray, p0: float) -> np.ndarray:
    w = np.maximum(w_mm, 0.0)
    return np.where(w > p0, (w - p0) * (w + 9.0 * p0) / (w + 4.0 * p0) ** 2, 0.0)


def net_rain(p: np.ndarray, p0: float, tau_h: float, w0: np.ndarray | float = 0.0) -> np.ndarray:
    """p (T, ncell) mm/h -> net rain (T, ncell) mm/h."""
    decay = float(np.exp(-1.0 / tau_h))
    w = np.zeros(p.shape[1], np.float32) + w0
    out = np.empty_like(p)
    for t in range(p.shape[0]):
        w_new = w * decay + p[t]
        out[t] = p[t] * runoff_coefficient(0.5 * (w * decay + w_new), p0)
        w = w_new
    return out


def route(net: np.ndarray, net_: HydroNet, k_frac: float) -> np.ndarray:
    """net (T, ncell) mm/h -> Q (T, P) m3/s."""
    T = net.shape[0]
    q = np.zeros((T, net_.n), np.float64)
    for k, A in enumerate(net_.lags):
        if k >= T or A.nnz == 0:
            continue
        q[k:] += (A @ net[: T - k].T).T
    q *= 0.278
    if k_frac > 0:
        for b in range(net_.n):
            K = max(k_frac * net_.tc_h[b], 0.25)
            c1 = 1.0 - np.exp(-1.0 / K)
            q[:, b] = lfilter([c1], [1.0, -(1.0 - c1)], q[:, b])
    return q


def stage(q: np.ndarray, net_: HydroNet) -> np.ndarray:
    """Depth above the thalweg (m) for discharges q (..., P). NaN where no rating exists."""
    out = np.full(q.shape, np.nan, np.float32)
    for b in range(net_.n):
        rq, rh = net_.rating_q[b], net_.rating_h[b]
        if rq is not None and len(rq) > 1:
            out[..., b] = np.interp(q[..., b], rq, rh)
    return out


def level_thresholds(net_: HydroNet, hp: dict) -> np.ndarray:
    """(4, P) discharges at which levels 2..5 start."""
    qb = net_.q_bankfull
    thr = np.stack([hp["f2"] * qb, hp["f3"] * qb, qb, hp["f5"] * qb])
    for b in range(net_.n):
        rq, rh = net_.rating_q[b], net_.rating_h[b]
        if rq is not None and len(rq) > 1 and np.isfinite(net_.h_bank[b]):
            h5 = net_.h_bank[b] + hp["d5_m"]
            if h5 <= rh[-1]:
                thr[3, b] = max(float(np.interp(h5, rh, rq)), 1.15 * qb[b])
    return thr


@dataclass
class HydroProduct:
    t_end: np.ndarray        # (T,) common hourly axis
    q_series: np.ndarray     # (3, T, P) p10, p50, p90 of discharge
    prob: np.ndarray         # (4, F, P)
    level: np.ndarray        # (F, P)
    q_peak: np.ndarray       # (2, F, P) median / p90 of the frame's peak discharge
    h_over: np.ndarray       # (2, F, P) median / p90 height above bank (m, negative = below)
    p_overflow: np.ndarray   # (F, P) plain probability of exceeding capacity (= prob[2])
    valid: np.ndarray


def hydro_product(members: list[Member], frames, net_: HydroNet, params: dict, horizon: str,
                  now: datetime, t_axis: np.ndarray) -> HydroProduct:
    hp = params["hydro"]
    sigma = hp["sigma"][horizon]
    shift_km = params["basin_shift_km"][horizon]
    dj, di = int(round(shift_km / grid.DY_KM)), int(round(shift_km / grid.DX_KM))
    thr = level_thresholds(net_, hp)                                  # (4, P)
    F, T, P = len(frames), len(t_axis), net_.n
    psum, wsum = np.zeros((4, F, P)), np.zeros(F)
    QS, QP, WF, WT = [], [], [], []
    for m in members:
        w = member_weight(m, params, horizon, now)
        if w <= 0:
            continue
        fam = params["families"][m.family]
        variants = [(0, 0, 0.4), (dj, 0, 0.15), (-dj, 0, 0.15), (0, di, 0.15), (0, -di, 0.15)] \
            if fam.get("neigh") and (dj or di) else [(0, 0, 1.0)]
        for sj, si, share in variants:
            p = np.nan_to_num(grid.shift(m.area, sj, si), nan=0.0).reshape(len(m.t_end), -1) * fam["s12h"]
            q = route(net_rain(p, hp["p0_mm"], hp["wet_memory_h"]), net_, hp["clark_k"])      # (Tm, P)
            # onto the common axis
            pos = np.searchsorted(t_axis, m.t_end)
            inside = (pos < T) & (t_axis[np.minimum(pos, T - 1)] == m.t_end)
            qa = np.full((T, P), np.nan, np.float32)
            qa[pos[inside]] = q[inside]
            peak = np.zeros((F, P)); valid = np.zeros(F, bool)
            for f, (t0, t1) in enumerate(frames):
                sel = (m.t_end > t0) & (m.t_end <= t1)
                if sel.any():
                    valid[f] = True
                    peak[f] = q[sel].max(axis=0)
            if not valid.any():
                break
            wv = w * share * valid
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(np.isfinite(thr)[:, None, :], peak[None] / thr[:, None, :], 0.0)
            psum += wv[None, :, None] * dress(ratio, sigma)
            wsum += wv
            QS.append(qa); QP.append(peak); WF.append(wv); WT.append(w * share)
    ok = wsum > 0
    den = np.where(ok, wsum, 1.0)
    prob = np.minimum.accumulate((psum / den[None, :, None]).astype(np.float32), axis=0)
    level = decide(prob, params["tau"][horizon])
    level[~ok] = 0
    level[:, ~np.isfinite(net_.q_bankfull)] = 0
    qser = np.zeros((3, T, P), np.float32)
    qpk = np.zeros((2, F, P), np.float32)
    if QS:
        qs, wt = np.stack(QS), np.asarray(WT)                         # (M, T, P)
        for t in range(T):
            use = np.isfinite(qs[:, t, 0])
            if use.any():
                qser[:, t] = weighted_quantile(qs[use, t], _bw(wt[use], (P,)), (0.1, 0.5, 0.9))
        qp, wf = np.stack(QP), np.stack(WF)
        for f in range(F):
            use = wf[:, f] > 0
            if use.any():
                qpk[:, f] = weighted_quantile(qp[use, f], _bw(wf[use, f], (P,)), (0.5, 0.9))
    h_over = stage(qpk, net_) - net_.h_bank[None, None, :].astype(np.float32)
    return HydroProduct(t_axis, qser, prob, level, qpk, h_over, prob[2], ok)
