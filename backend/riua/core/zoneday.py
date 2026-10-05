"""Days 2-7 by warning zone and day: P(level L is reached somewhere in the zone that day).

Why: a 25-km ensemble cannot place a storm within 12 km days ahead, and the warning services forecast days ahead
by zone and day. Scored out of sample on the GitHub hindcast (hindcast/long_data.py, long_zone.py, long_score.py;
coord/findings/q13-long.md).

Per zone-day, from the scenarios (ECMWF ENS + IFS runs, production weights):
  r_m,L   largest scaled amount / threshold of level L over the zone's cells, scenario m (risk.predictors amounts)
  summaries of ln(1.6 r): weighted mean, share of the weight above 1 and above 0.5, upper decile
  rank    percentile of the yellow-level mean and upper decile in the MODEL'S OWN climate of that zone and season
          (an EFI/SOT-like relative signal: the coarse model under-forecasts extremes, but its unusual days are unusual)
  P(>= L) = logistic(features), one ridge-fitted model per level; the coefficients and the climate tables are in
            zoneday_model.json, written by `python hindcast/long_score.py fit`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit

MODEL_FILE = Path(__file__).resolve().parents[1] / "zoneday_model.json"
B = 1.6


def _lg(f, eps):
    return np.log((f + eps) / (1.0 + eps - f))


def summaries(lR: np.ndarray, Wn: np.ndarray, lead_h: np.ndarray) -> np.ndarray:
    """lR (U, M) ln of the zone ratio of one level (anything for members with Wn = 0), Wn (U, M) normalised weights,
    lead_h (U,) hours from the issue time to the end of the day -> (U, 5) mean, fr1, fr5, q90, lead."""
    lr = lR + np.log(B)
    on = Wn > 0
    mean = np.log(np.maximum((Wn * np.exp(np.where(on, lr, -50))).sum(1), 1e-3))
    fr1 = (Wn * (lr >= 0)).sum(1)
    fr5 = (Wn * (lr >= np.log(0.5))).sum(1)
    v = np.where(on, lr, -99.0)
    order = np.argsort(v, axis=1)
    vs = np.take_along_axis(v, order, 1)
    cw = np.cumsum(np.take_along_axis(Wn, order, 1), 1)
    q90 = np.take_along_axis(vs, np.argmax(cw >= 0.9 - 1e-9, axis=1)[:, None], 1)[:, 0]
    return np.stack([mean, _lg(fr1, 0.02), _lg(fr5, 0.02), np.maximum(q90, -6.0), (lead_h - 48.0) / 96.0], 1)


def rank_lg(r: np.ndarray) -> np.ndarray:
    return _lg(r, 0.01)


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> dict | None:
    p = Path(path) if path else MODEL_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def climate_rank(model: dict, which: str, zone: np.ndarray, month: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Percentile of `value` in the climate table of (zone, month): table[which][zone][month-1] = quantiles at
    model['climate']['probs']."""
    pr = np.array(model["climate"]["probs"])
    out = np.empty(len(value))
    for i in range(len(value)):
        q = np.array(model["climate"][which][int(zone[i])][int(month[i]) - 1])
        out[i] = np.interp(value[i], q, pr, left=0.0, right=1.0)
    return out


def features(R: np.ndarray, Wn: np.ndarray, lead_h, zone, month, model: dict) -> dict:
    """R (U, M, 4) zone ratios -> {level index k: X (U, 7)} with the climate ranks from the model's tables."""
    lR = np.log(np.maximum(np.nan_to_num(R, nan=1e-6), 1e-6))
    Sy = summaries(lR[:, :, 0], Wn, lead_h)
    rk = rank_lg(climate_rank(model, "mean", zone, month, Sy[:, 0]))
    rk90 = rank_lg(climate_rank(model, "q90", zone, month, Sy[:, 3]))
    out = {}
    for k in range(4):
        S = Sy if k == 0 else summaries(lR[:, :, k], Wn, lead_h)
        out[k] = np.hstack([S, rk[:, None], rk90[:, None]])
    return out


def probabilities(R: np.ndarray, Wn: np.ndarray, lead_h, zone, month, model: dict) -> np.ndarray:
    """-> (4, U) P(>= 2..5), nested. Levels without a fitted model get 0."""
    X = features(R, Wn, np.asarray(lead_h, float), np.asarray(zone), np.asarray(month), model)
    P = np.zeros((4, R.shape[0]))
    for k in range(4):
        m = model["levels"].get(str(k + 2))
        if not m:
            continue
        Xs = (X[k] - np.array(m["mu"])) / np.array(m["sd"])
        P[k] = expit(m["beta"][0] + Xs @ np.array(m["beta"][1:]))
    return np.minimum.accumulate(P, axis=0)


def zone_ratios(pred, thr, params: dict, zone_idx: np.ndarray, n_zones: int):
    """From risk.Predictors: R (F, Z, M, 4) largest scaled 12-h (and 1-h, where known) amount / threshold over the
    cells of each zone, and the normalised weights Wn (F, Z, M)."""
    M, F = pred.w.shape
    s12 = np.array([params["families"][a["family"]]["s12h"] for a in pred.audit], np.float32)
    s1 = np.array([params["families"][a["family"]]["s1h"] for a in pred.audit], np.float32)
    A12 = pred.a12.reshape(M, F, -1) * s12[:, None, None]
    A1 = pred.a1.reshape(M, F, -1) * s1[:, None, None]
    t12, t1 = thr.t12h.reshape(4, -1), thr.t1h.reshape(4, -1)
    zi = zone_idx.ravel()
    R = np.full((F, n_zones, M, 4), np.nan, np.float32)
    for z in range(n_zones):
        cs = zi == z
        if not cs.any():
            continue
        for k in range(4):
            r12 = np.where(np.isfinite(A12[:, :, cs]), A12[:, :, cs] / t12[k, cs], -1.0).max(axis=2)
            r1 = np.where(np.isfinite(A1[:, :, cs]), A1[:, :, cs] / t1[k, cs], -1.0).max(axis=2)
            r = np.maximum(r12, r1)                                      # (M, F)
            R[:, z, :, k] = np.where(r >= 0, r, np.nan).T
    ok = np.isfinite(R[..., 0])
    W = np.where(ok, pred.w.T[:, None, :], 0.0)
    Wn = W / np.maximum(W.sum(axis=2, keepdims=True), 1e-12)
    return R, Wn
