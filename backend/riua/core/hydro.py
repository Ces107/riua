"""Rain to discharge to water level at the control points (ravines and rivers at towns).

Chain, for every rainfall scenario:

1. Losses, per grid cell. Wetness W is the rain remembered by the soil,
   W(t) = W(t-1) * exp(-1/tau) + p(t), and the share of the hour's rain that runs off
   is the marginal runoff coefficient of the loss law E = (W - P0)^2 / (W - P0 + S)
   (the SCS / Norma 5.2-IC law when S = 5 P0):
       c(W) = x (x + 2 S) / (x + S)^2,  x = W - P0,  for W > P0, else 0.
   P0 and S are fitted to measured flows: most of the ground swallows about 150 mm before
   it sheds anything, while a small share alpha of each catchment (2-18 %, fitted per
   gauged stream) runs after 10 mm. That share is what makes one ravine run and its
   neighbour stay dry under the same rain.

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

from . import flood_ml, grid
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
    qT: np.ndarray | None = None     # (P, 6) natural flood peaks for T = 2, 5, 10, 25, 100, 500 years (CAUMAX)
    alpha: np.ndarray | None = None  # (NY*NX,) share of each cell that runs after p0b mm (geo/hydro/loss_params.json)

    @property
    def n(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, ids, area, tc_h, point_idx, cell, lag_h, area_km2, q_bankfull, rating_q, rating_h, h_bank, qT=None):
        P, n = len(ids), grid.NY * grid.NX
        kmax = int(lag_h.max()) if len(lag_h) else 0
        lags = []
        for k in range(kmax + 1):
            s = lag_h == k
            lags.append(sparse.csr_matrix((area_km2[s].astype(np.float32), (point_idx[s], cell[s])), shape=(P, n)))
        return cls(list(ids), np.asarray(area, float), np.asarray(tc_h, float), lags,
                   np.asarray(q_bankfull, float), list(rating_q), list(rating_h), np.asarray(h_bank, float),
                   None if qT is None else np.asarray(qT, float))


def runoff_coefficient(w_mm: np.ndarray, p0: float, s: float | None = None) -> np.ndarray:
    """Marginal runoff coefficient dE/dW of E = (W - P0)^2 / (W - P0 + S): nothing below the
    runoff threshold P0, then a share that grows towards 1 as the retention S fills.
    S = 5 P0 is the SCS / Norma 5.2-IC law (initial abstraction 0.2 S)."""
    s = 5.0 * p0 if s is None else s
    x = np.maximum(w_mm - p0, 0.0)
    return x * (x + 2.0 * s) / (x + s) ** 2


def net_rain(p: np.ndarray, p0: float, tau_h: float, w0: np.ndarray | float = 0.0, phi: float | None = None,
             s: float | None = None, alpha: np.ndarray | None = None, p0b: float = 10.0, sb: float = 100.0) -> np.ndarray:
    """p (T, ncell) mm/h -> net rain (T, ncell) mm/h.

    Three ways of producing runoff, as in Mediterranean ravines: the ground fills up
    (saturation excess, share c(W) of the rain); it rains harder than the ground can take in
    (infiltration excess: what exceeds phi mm/h runs off even on dry ground); and a share alpha
    of every cell (paved and compacted ground, valley bottoms, the channels) runs after only
    p0b mm. alpha (per cell) is what differs between catchments.
    """
    decay = float(np.exp(-1.0 / tau_h))
    w = np.zeros(p.shape[1], np.float32) + w0
    out = np.empty_like(p)
    for t in range(p.shape[0]):
        w_new = w * decay + p[t]
        wm = 0.5 * (w * decay + w_new)
        c = runoff_coefficient(wm, p0, s)
        e = p[t] * c
        if phi is not None:
            e = e + (1.0 - c) * np.maximum(p[t] - phi, 0.0)
        if alpha is not None:
            e = (1.0 - alpha) * e + alpha * p[t] * runoff_coefficient(wm, p0b, sb)
        out[t] = e
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
    """(4, P) discharges at which levels 2..5 start.

    With a channel capacity Qb: 2 at f2*Qb, 3 at f3*Qb, 4 at Qb (overflow), 5 when the
    water stands d5 m above the bank (or f5*Qb without rating curve).
    Without a usable capacity: fractions of the envelope of the largest Mediterranean
    flash floods for that area, Q_env = c * A^(1+e) (Gaume et al. 2009): the unit
    discharge is the standard severity measure of a flash flood.
    """
    qb = net_.q_bankfull
    thr = np.stack([hp["f2"] * qb, hp["f3"] * qb, qb, hp["f5"] * qb])
    env = hp["env_c"] * np.maximum(net_.area, 1.0) ** (1.0 + hp["env_e"])
    no_cap = ~np.isfinite(qb)
    for k, frac in enumerate(hp["env_fractions"]):
        thr[k, no_cap] = frac * env[no_cap]
    # better, when the official flood quantiles exist: levels 2..5 at the T-year floods of hp["rp_levels"]
    if net_.qT is not None:
        col = {2: 0, 5: 1, 10: 2, 25: 3, 100: 4, 500: 5}
        have = no_cap & np.all(np.isfinite(net_.qT), axis=1)
        for k, T in enumerate(hp["rp_levels"]):
            thr[k, have] = net_.qT[have, col[int(T)]]
        # with a capacity: a big flood is dangerous even inside a large channel (fords, bridges,
        # riverside roads), so levels 2 and 3 also start at the 2- and 5-year floods; 4 and 5
        # stay tied to the channel overflowing
        both = ~no_cap & np.all(np.isfinite(net_.qT), axis=1)
        for k in (0, 1):
            thr[k, both] = np.minimum(thr[k, both], net_.qT[both, col[int(hp["rp_levels"][k])]])
    # no level for a trickle: where CAUMAX is implausibly low (semi-endorheic basins such as the Tarafa)
    # levels 2 and 3 need at least 0.6 and 1.5 x A^0.6 m3/s (20 and 50 m3/s for 340 km2)
    a06 = np.maximum(net_.area, 1.0) ** 0.6
    thr[0] = np.maximum(thr[0], hp.get("min_q2", 0.6) * a06)
    thr[1] = np.maximum(thr[1], hp.get("min_q3", 1.5) * a06)
    ok = np.isfinite(qb)
    thr[1, ok] = np.minimum(thr[1, ok], qb[ok])      # a floor never moves the overflow above a known capacity
    thr[0] = np.minimum(thr[0], thr[1])
    thr[2] = np.maximum(thr[2], thr[1])
    thr[3] = np.maximum(thr[3], thr[2])
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
    rp: np.ndarray | None = None   # (2, F, P) return period (years) of the median / p90 peak; 1 = below T2


def hydro_product(members: list[Member], frames, net_: HydroNet, params: dict, horizon: str,
                  now: datetime, t_axis: np.ndarray, extra=None) -> HydroProduct:
    """extra: (outflow per scenario key (name, sj, si) -> (T, P) m3/s, its hours): dam releases and spills
    added to the points whose catchment the dam cuts (core/reservoirs.outflow_to_points)."""
    hp = params["hydro"]
    sigma = hp["sigma"][horizon]
    shift_km = params["basin_shift_km"][horizon]
    dj, di = int(round(shift_km / grid.DY_KM)), int(round(shift_km / grid.DX_KM))
    thr = level_thresholds(net_, hp)                                  # (4, P)
    fml = flood_ml.load() if hp.get("ml_vote", True) else None
    F, T, P = len(frames), len(t_axis), net_.n
    psum, wsum = np.zeros((4, F, P)), np.zeros(F)
    QS, QP, WF, WT = [], [], [], []
    for m in members:
        w = member_weight(m, params, horizon, now)
        if w <= 0:
            continue
        fam = params["families"][m.family]
        variants = [(0, 0, 0.4), (dj, 0, 0.15), (-dj, 0, 0.15), (0, di, 0.15), (0, -di, 0.15)] \
            if m.family in ("cp", "radar") and (dj or di) else [(0, 0, 1.0)]
        for sj, si, share in variants:
            p = np.nan_to_num(grid.shift(m.area, sj, si), nan=0.0).reshape(len(m.t_end), -1)
            p = p * np.where(m.observed(), 1.0, fam["s12h"])[:, None]      # measured rain is not rescaled
            q = route(net_rain(p, hp["p0_mm"], hp["wet_memory_h"], phi=hp.get("phi_mmh"), s=hp.get("s_mm"),
                               alpha=net_.alpha, p0b=hp.get("p0b_mm", 10.0), sb=hp.get("sb_mm", 100.0)),
                      net_, hp["clark_k"])      # (Tm, P)
            q0 = route(p, net_, hp["clark_k"]) if fml is not None else None      # zero-loss: rain arriving at each point
            if extra is not None and (m.name, sj, si) in extra[0]:
                add, t_add = extra[0][(m.name, sj, si)], extra[1]          # what the dams above let through
                k = np.searchsorted(t_add, m.t_end)
                ok = (k < len(t_add)) & (t_add[np.minimum(k, len(t_add) - 1)] == m.t_end)
                q[ok] += add[k[ok]]
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
            pd_ = dress(ratio, sigma)
            if fml is not None:     # statistical vote: may raise levels 2-3, never lowers (coord/findings/q11-ml.md)
                pd_[:2] = np.maximum(pd_[:2], flood_ml.prob(fml, peak, q0, m.t_end, frames, thr, net_.area) * valid[None, :, None])
            psum += wv[None, :, None] * pd_
            wsum += wv
            QS.append(qa); QP.append(peak); WF.append(wv); WT.append(w * share)
    ok = wsum > 0
    den = np.where(ok, wsum, 1.0)
    prob = np.minimum.accumulate((psum / den[None, :, None]).astype(np.float32), axis=0)
    level = decide(prob, params["tau"][horizon])
    level[~ok] = 0
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
    return HydroProduct(t_axis, qser, prob, level, qpk, h_over, prob[2], ok, return_period(qpk, net_))


RP_T = np.array([2.0, 5.0, 10.0, 25.0, 100.0, 500.0])


def return_period(q: np.ndarray, net_: HydroNet) -> np.ndarray | None:
    """Return period (years) of peaks q (..., P), interpolated log-log between the CAUMAX quantiles.
    1 = below the 2-year flood; values above T500 are extrapolated and capped at 1000."""
    if net_.qT is None:
        return None
    out = np.full(q.shape, np.nan, np.float32)
    lt = np.log(RP_T)
    for b in range(net_.n):
        qt = net_.qT[b]
        if not np.all(np.isfinite(qt)) or qt[0] <= 0:
            continue
        lq = np.log(np.maximum(qt, 1e-3))
        x = np.log(np.maximum(q[..., b], 1e-3))
        t = np.interp(x, lq, lt, left=-np.inf, right=np.nan)
        slope = (lt[-1] - lt[-2]) / max(lq[-1] - lq[-2], 1e-6)
        t = np.where(np.isnan(t), lt[-1] + slope * (x - lq[-1]), t)      # beyond T500
        out[..., b] = np.where(q[..., b] < qt[0], 1.0, np.minimum(np.exp(t), 1000.0))
    return out
