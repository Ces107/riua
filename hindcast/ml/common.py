"""Shared paths and small helpers of hindcast/ml (flood detection at the control points: diagnosis + light ML)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
CACHE = HERE / "cache"
GAUGES = ROOT / "geo" / "hydro" / "gauges"
FLOWS = ROOT / "hindcast" / "obs" / "flows"
MODEL_JSON = ROOT / "geo" / "hydro" / "flood_ml.json"
for d in (OUT, CACHE):
    d.mkdir(exist_ok=True)
for p in (ROOT / "backend", ROOT / "hindcast", GAUGES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

A_REF, C_REF = 184.0, 5.0
UA = "riua-hindcast/1.0 (flood ML validation; github.com/Ces107/riua)"


def c_off(area):
    """q7's 'the ravine runs' threshold: 5 m3/s at 184 km2, scaled like a flood peak (A^0.75)"""
    return C_REF * (np.asarray(area, float) / A_REF) ** 0.75


def read_json(f):
    return json.loads(Path(f).read_text(encoding="utf-8"))


def write_json(f, obj, **kw):
    Path(f).write_text(json.dumps(obj, ensure_ascii=False, default=float, **kw), encoding="utf-8")


def counts(alert, event):
    """hits, misses, false alarms, correct negatives + POD, FAR (false-alarm ratio), CSI"""
    alert, event = np.asarray(alert, bool), np.asarray(event, bool)
    h, m, f = int((alert & event).sum()), int((~alert & event).sum()), int((alert & ~event).sum())
    n = int((~alert & ~event).sum())
    return dict(hits=h, misses=m, false=f, neg=n, pod=h / max(h + m, 1), far=f / max(h + f, 1), csi=h / max(h + m + f, 1))


def auc(score, event):
    from scipy.stats import rankdata
    event = np.asarray(event, bool)
    n1, n0 = int(event.sum()), int((~event).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
