"""From precipitation scenarios to calibrated probabilities and to a level, per grid cell.

Vocabulary
----------
member    one precipitation scenario on the Riuà grid: a model run, an ensemble member or
          a radar-extrapolation member. Hourly accumulations, mm.
frame     a time window (t0, t1] of a product.
amounts   per member and frame, the two quantities AEMET's thresholds are written for:
          a1  = largest 1-h accumulation ending inside the frame,
          a12 = largest 12-h accumulation ending inside the frame (the window may start
                before the frame: a ravine does not care when the frame began).
levels    2 yellow, 3 orange, 4 red (Meteoalerta thresholds of the warning zone),
          5 extreme (multiples of red, params["extreme"]).

Chain
-----
1. Neighbourhood: for convection-permitting members a1 and a12 are replaced by their
   maximum within R km. A 2.5-km model that puts the storm 15 km from where it falls is
   a good forecast, and must count as one (Schwartz and Sobash 2017; Roberts and Lean 2008).
2. Ensemble summary per cell: weighted mean and weighted 90th percentile of the members'
   amounts. Weights = family weight x age decay of the run.
3. Calibration: censored shifted gamma EMOS (core/emos.py) maps (mean, q90) to the
   predictive distribution of the amount that will be OBSERVED in the cell (largest value
   of the 1-km radar-gauge analysis inside the cell). Its coefficients come from the
   hindcast, separately for each horizon and lead class.
4. P(>= L) = P(a1 >= T1h_L or a12 >= T12h_L), the union taken with a Gaussian copula.
5. Decision: the highest L whose probability reaches tau_L; tau decreases with severity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.special import ndtr

from . import grid
from .emos import CSGD, UnionCopula

LEVELS = (2, 3, 4, 5)


@dataclass
class Member:
    name: str                 # e.g. "AROME-HD 2026-10-01 06Z" or "IFS-ENS m17 2026-10-01 00Z"
    family: str               # key of params["families"]
    model: str                # model id, for the audit trail
    run: datetime             # initialisation time (UTC, naive)
    t_end: np.ndarray         # (T,) datetime64[h], end of each hourly accumulation
    p: np.ndarray             # (T, NY, NX) float32 mm in the hour ending at t_end; no NaN
    native_step_h: int = 1    # 1 = true hourly data; 3/6 = disaggregated uniformly
    weight: float = 1.0       # extra multiplier set by the pipeline
    p_area: np.ndarray | None = None   # areal-mean version for catchment volumes (radar members)
    meta: dict = field(default_factory=dict)

    @property
    def area(self) -> np.ndarray:
        return self.p if self.p_area is None else self.p_area


@dataclass
class Thresholds:
    """Per-cell thresholds, mm. Arrays (4, NY, NX): yellow, orange, red, extreme."""
    t1h: np.ndarray
    t12h: np.ndarray

    @classmethod
    def from_zone_values(cls, y1, o1, r1, y12, o12, r12, params: dict) -> "Thresholds":
        ex = params["extreme"]
        t1 = np.stack([y1, o1, r1, r1 * ex["x1h"]]).astype(np.float32)
        t12 = np.stack([y12, o12, r12, r12 * ex["x12h"]]).astype(np.float32)
        return cls(t1, t12)


def frame_amounts(p: np.ndarray, t_end: np.ndarray, frames) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, ...) hourly amounts -> a1, a12 (F, ...) and valid (F,). Hours before the series count as dry."""
    T = len(t_end)
    cs = np.concatenate([np.zeros((1, *p.shape[1:]), np.float32), np.cumsum(p, axis=0, dtype=np.float32)])
    idx = np.arange(1, T + 1)
    lo = np.maximum(idx - 12, 0)
    a1 = np.zeros((len(frames), *p.shape[1:]), np.float32)
    a12 = np.zeros_like(a1)
    valid = np.zeros(len(frames), bool)
    for f, (t0, t1) in enumerate(frames):
        sel = np.nonzero((t_end > t0) & (t_end <= t1))[0]
        # a frame is covered only if the member spans (almost) all of it
        need = int((t1 - t0) / np.timedelta64(1, "h"))
        if sel.size == 0 or sel.size < max(1, int(0.75 * need)):
            continue
        valid[f] = True
        a1[f] = p[sel].max(axis=0)
        a12[f] = (cs[idx[sel]] - cs[lo[sel]]).max(axis=0)
    return a1, a12, valid


def member_weight(m: Member, params: dict, horizon: str, now: datetime) -> float:
    fam = params["families"][m.family]
    age_h = max(0.0, (now - m.run).total_seconds() / 3600.0)
    decay = 0.5 ** (age_h / params["age_halflife_h"][horizon])
    return float(fam["weight"] * m.weight * decay)


def weighted_stats(values: np.ndarray, weights: np.ndarray, q: float = 0.9):
    """values (M, ...) (NaN = member not usable there), weights (M,) -> weighted mean and quantile (...)."""
    ok = np.isfinite(values)
    w = np.where(ok, weights.reshape(-1, *([1] * (values.ndim - 1))), 0.0)
    v = np.where(ok, values, 0.0)
    wsum = w.sum(axis=0)
    safe = np.maximum(wsum, 1e-12)
    mean = (w * v).sum(axis=0) / safe
    order = np.argsort(np.where(ok, values, np.inf), axis=0)
    vs = np.take_along_axis(v, order, axis=0)
    ws = np.take_along_axis(w, order, axis=0)
    cw = np.cumsum(ws, axis=0) / safe
    k = np.argmax(cw >= q - 1e-9, axis=0)
    quant = np.take_along_axis(vs, k[None], axis=0)[0]
    return (np.where(wsum > 0, mean, np.nan).astype(np.float32),
            np.where(wsum > 0, quant, np.nan).astype(np.float32), wsum)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """values (M, ...), weights (M, ...) broadcastable -> (len(qs), ...)."""
    order = np.argsort(values, axis=0)
    v = np.take_along_axis(values, order, axis=0)
    w = np.take_along_axis(np.broadcast_to(weights, values.shape), order, axis=0)
    cw = np.cumsum(w, axis=0, dtype=np.float64)
    cw /= np.maximum(cw[-1], 1e-12)
    return np.stack([np.take_along_axis(v, np.argmax(cw >= q - 1e-9, axis=0)[None], axis=0)[0] for q in qs])


def _bw(w: np.ndarray, tail: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(w.reshape(-1, *([1] * len(tail))), (w.size, *tail))


@dataclass
class Predictors:
    """Ensemble summaries per frame and cell, the inputs of the calibration."""
    m1: np.ndarray      # (F, NY, NX) weighted mean of neighbourhood 1-h amount (NaN: no hourly member)
    q1: np.ndarray      # weighted 90th percentile
    m12: np.ndarray
    q12: np.ndarray
    valid: np.ndarray   # (F,) at least one member covers the frame
    has_1h: np.ndarray  # (F,) at least one member with true hourly data
    audit: list[dict]   # one row per member
    a1: np.ndarray | None = None    # (M, F, NY, NX) per-member neighbourhood amounts (NaN = not usable)
    a12: np.ndarray | None = None
    w: np.ndarray | None = None     # (M, F) normalised weights


def predictors(members: list[Member], frames, params: dict, horizon: str, now: datetime,
               sample_mask: np.ndarray | None = None, keep_members: bool = True) -> Predictors:
    """sample_mask (NY, NX) bool: cells where convection-permitting / regional members are sampled
    (the 0.1 deg lattice used when the archive only allows that density). Elsewhere the member is
    treated as unknown and the neighbourhood maximum bridges the gap.
    """
    F = len(frames)
    radius = params["radius_km"][horizon]
    A1, A12, W, audit = [], [], [], []
    for m in members:
        w = member_weight(m, params, horizon, now)
        if w <= 0:
            continue
        fam = params["families"][m.family]
        a1, a12, valid = frame_amounts(m.p, m.t_end, frames)
        if not valid.any():
            continue
        if sample_mask is not None and fam.get("sampled", False):
            a1 = np.where(sample_mask, a1, -1.0)
            a12 = np.where(sample_mask, a12, -1.0)
        r = radius if fam.get("neigh") else (params.get("lattice_fill_km", 8.0) if (sample_mask is not None and fam.get("sampled")) else 0.0)
        if r > 0:
            a1 = grid.neighbourhood_max(a1, r)
            a12 = grid.neighbourhood_max(a12, r)
        a1 = np.where(a1 < 0, np.nan, a1)
        a12 = np.where(a12 < 0, np.nan, a12)
        if m.native_step_h > 1:
            a1 = np.full_like(a1, np.nan)       # no information on hourly intensity
        a1[~valid] = np.nan
        a12[~valid] = np.nan
        A1.append(a1); A12.append(a12); W.append(w * valid)
        audit.append({"name": m.name, "family": m.family, "model": m.model,
                      "run": m.run.strftime("%Y-%m-%dT%H:%MZ"), "step_h": m.native_step_h,
                      "neigh_km": r, "w_raw": round(w, 5)})
    shape = (F, grid.NY, grid.NX)
    if not A1:
        nan = np.full(shape, np.nan, np.float32)
        return Predictors(nan, nan.copy(), nan.copy(), nan.copy(), np.zeros(F, bool), np.zeros(F, bool), [])
    a1s, a12s, ws = np.stack(A1), np.stack(A12), np.stack(W)       # (M, F, ...), (M, F)
    m1 = np.full(shape, np.nan, np.float32); q1 = m1.copy(); m12 = m1.copy(); q12 = m1.copy()
    for f in range(F):
        m1[f], q1[f], _ = weighted_stats(a1s[:, f], ws[:, f])
        m12[f], q12[f], _ = weighted_stats(a12s[:, f], ws[:, f])
    valid = ws.sum(axis=0) > 0
    has_1h = np.isfinite(m1).any(axis=(1, 2))
    tot = np.maximum(ws.sum(axis=0, keepdims=True), 1e-12)
    for k, row in enumerate(audit):
        row["w"] = [round(float(x), 4) for x in (ws[k] / tot[0])]
    return Predictors(m1, q1, m12, q12, valid, has_1h, audit,
                      a1s if keep_members else None, a12s if keep_members else None, ws / tot)


@dataclass
class Calibration:
    """Fitted post-processing for one horizon / lead class."""
    c1: CSGD | None        # observed a1  | (m1, q1)
    c1_from12: CSGD | None  # observed a1 | (m12, q12), used when no member has hourly data
    c12: CSGD              # observed a12 | (m12, q12)
    rho: float = 0.7
    _cop: UnionCopula | None = None

    def copula(self) -> UnionCopula:
        if self._cop is None:
            self._cop = UnionCopula(self.rho)
        return self._cop

    def to_dict(self) -> dict:
        return {"c1": self.c1.to_dict() if self.c1 else None,
                "c1_from12": self.c1_from12.to_dict() if self.c1_from12 else None,
                "c12": self.c12.to_dict(), "rho": float(self.rho)}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        mk = lambda x: CSGD(**x) if x else None
        return cls(mk(d.get("c1")), mk(d.get("c1_from12")), mk(d["c12"]), float(d.get("rho", 0.7)))


def calibrated_probabilities(pred: Predictors, cal_by_frame: list[Calibration], thr: Thresholds):
    """-> prob (4, F, NY, NX), p1 (4, F, ...), p12 (4, F, ...): union and the two marginals."""
    F = pred.m12.shape[0]
    prob = np.zeros((4, F, grid.NY, grid.NX), np.float32)
    p1 = np.zeros_like(prob); p12 = np.zeros_like(prob)
    for f in range(F):
        if not pred.valid[f]:
            continue
        cal = cal_by_frame[f]
        m12, q12 = np.nan_to_num(pred.m12[f]), np.nan_to_num(pred.q12[f])
        use1 = pred.has_1h[f] and cal.c1 is not None
        m1, q1 = np.nan_to_num(pred.m1[f]), np.nan_to_num(pred.q1[f])
        for k in range(4):
            pb = cal.c12.exceed(m12, q12, thr.t12h[k])
            if use1:
                pa = cal.c1.exceed(m1, q1, thr.t1h[k])
            elif cal.c1_from12 is not None:
                pa = cal.c1_from12.exceed(m12, q12, thr.t1h[k])
            else:
                pa = np.zeros_like(pb)
            p1[k, f], p12[k, f] = pa, pb
            prob[k, f] = cal.copula().union(pa, pb)
    prob = np.minimum.accumulate(prob, axis=0)      # nested in severity
    return prob, p1, p12


def dress(ratio: np.ndarray, sigma: float, bias: float = 1.0) -> np.ndarray:
    """Probability that the truth reaches the threshold given a scenario at `ratio` of it
    (log-normal error around the scenario; used for catchment and discharge scenarios)."""
    with np.errstate(divide="ignore"):
        z = np.log(np.maximum(bias * ratio, 1e-9)) / sigma
    return ndtr(z).astype(np.float32)


def decide(prob: np.ndarray, tau: dict) -> np.ndarray:
    """prob: (4, ...) P(>=2), P(>=3), P(>=4), P(>=5). Returns levels 1..5 (uint8)."""
    level = np.ones(prob.shape[1:], np.uint8)
    for k, L in enumerate(LEVELS):
        level = np.where(prob[k] >= float(tau[str(L)]), L, level)
    return level.astype(np.uint8)
