"""Independent statistical vote on flood levels 2 and 3 at the control points (inference only, pure numpy).

Trained by hindcast/ml/train.py on 380 measured rain events at 26 gauged ravine catchments (coord/findings/q11-ml.md).
Per scenario, frame and point it returns P(peak >= threshold) from three numbers that hydro_product already has or
gets with one extra zero-loss routing:

    x_sim = ln((q_sim + 0.1 c) / T)      q_sim: the conceptual peak of the frame (with losses)
    x_pot = ln((q_pot + 0.1 c) / T)      q_pot: largest 12-h mean of the zero-loss routed flow in the frame
    l_dep = ln(1 + d24)                  d24:   largest 24-h depth of rain ARRIVING at the point (zero-loss flow, mm)
    P = 1 / (1 + exp(-(b0 + b1 x_sim + b2 x_pot + b3 l_dep)))          c = 5 (A / 184)^0.75 m3/s

Use it only as P_final = max(P_dressed, P) for levels 2 and 3: it can raise a level, never lower one, and it says
nothing about levels 4-5 (no measured event above level 3 to learn from).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[3] / "geo" / "hydro" / "flood_ml.json"


def load(path: Path | None = None) -> dict | None:
    """the coefficients, or None when the file is missing (then the vote is simply off)"""
    f = Path(path or MODEL)
    if not f.exists():
        return None
    m = json.loads(f.read_text(encoding="utf-8"))
    m["beta"] = np.asarray(m["beta"], float)
    return m


def _trailing(x: np.ndarray, n: int) -> np.ndarray:
    """(T, P) -> sum of the last n hours ending at each hour (hours before the series count as 0)"""
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    i = np.arange(1, x.shape[0] + 1)
    return cs[i] - cs[np.maximum(i - n, 0)]


def prob(model: dict, q_peak: np.ndarray, q0: np.ndarray, t_end: np.ndarray, frames, thr: np.ndarray,
         area: np.ndarray) -> np.ndarray:
    """q_peak (F, P) conceptual frame peaks; q0 (Tm, P) zero-loss routed flow of the same scenario on its hourly axis
    t_end; thr (4, P) level thresholds; area (P,) km2.  Returns (len(levels), F, P) probabilities for levels 2, 3
    (0 in frames the scenario does not cover)."""
    b = model["beta"]
    area = np.maximum(np.asarray(area, float), 1.0)
    c = 5.0 * (area / 184.0) ** 0.75
    q0 = np.maximum(np.nan_to_num(np.asarray(q0, float)), 0.0)
    pot = _trailing(q0, int(model.get("pot_hours", 12))) / float(model.get("pot_hours", 12))
    dep = _trailing(q0, int(model.get("dep_hours", 24))) * 3.6 / area[None, :]
    F, P = q_peak.shape
    levels = model.get("levels", [2, 3])
    out = np.zeros((len(levels), F, P), np.float32)
    for f, (t0, t1) in enumerate(frames):
        sel = (t_end > t0) & (t_end <= t1)
        if not sel.any():
            continue
        qp, dp = pot[sel].max(axis=0), dep[sel].max(axis=0)
        for j, L in enumerate(levels):
            T = thr[L - 2]
            ok = np.isfinite(T) & (T > 0)
            Ts = np.where(ok, T, 1.0)
            z = b[0] + b[1] * np.log((q_peak[f] + 0.1 * c) / Ts) + b[2] * np.log((qp + 0.1 * c) / Ts) + b[3] * np.log1p(dp)
            out[j, f] = np.where(ok, 1.0 / (1.0 + np.exp(-z)), 0.0)
    return out
