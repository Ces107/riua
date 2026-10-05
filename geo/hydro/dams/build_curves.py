"""Level-volume curve of every reservoir from the pairs its SAIH publishes: V = a (h - h0)^b.

SAIH Júcar gives level (m a.s.l.) and volume every 5 min; the pairs cached by fetch_history.py sample two years of
levels. Where SAIH clips the volume at the maximum normal level (Forata on 29 Oct 2024: 37.34 hm3 while the level
went 5.7 m above the spillway) the clipped pairs are dropped and the power law is what gives the volume above it.
SAIH Segura gives a level above the gauge zero (not above sea level): the curve is in that scale.

Output: geo/hydro/dams/out/curves.json  {id: {a, b, h0, n, level_min, level_max, v_max, rms_hm3, rel_rms, clipped_at_hm3 | null}}

    py -3.11 geo/hydro/dams/build_curves.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / "scratch" / "q10-dams" / "series"


def pairs(prefix, lvar, vvar):
    fl, fv = CACHE / f"{prefix}_{lvar}.npz", CACHE / f"{prefix}_{vvar}.npz"
    if not fl.exists() or not fv.exists():
        return None
    zl, zv = np.load(fl), np.load(fv)
    _, i, j = np.intersect1d(zl["t"], zv["t"], return_indices=True)
    h, v = zl["v"][i].astype(float), zv["v"][j].astype(float)
    ok = np.isfinite(h) & np.isfinite(v) & (h > 0.01) & (v > 1e-4)
    return h[ok], v[ok]


def fit(h, v):
    """Least squares in log space for a, b, with h0 scanned below the lowest level seen. Returns dict or None."""
    # one point per centimetre of level (median volume): long flat periods must not weigh more than a flood
    key = np.round(h, 2)
    u = np.unique(key)
    hv = np.array([[k, np.median(v[key == k])] for k in u])
    h, v = hv[:, 0], hv[:, 1]
    # drop sensor glitches: a pair far from the running median of its neighbours in level
    if len(h) >= 9:
        med = np.array([np.median(v[max(0, k - 4):k + 5]) for k in range(len(v))])
        keep = np.abs(v - med) <= np.maximum(0.15 * med, 0.02)
        h, v = h[keep], v[keep]
    if len(h) < 4 or h.max() - h.min() < 0.3:
        return None
    clipped = None
    top = v.max()
    at_top = v >= top * 0.9995
    if at_top.sum() >= 3 and h[at_top].max() - h[at_top].min() > 0.25:      # the volume stands still while the level moves
        clipped = float(top)
        first = h[at_top].min()
        keep = h <= first + 1e-9
        h, v = h[keep], v[keep]
    best = None
    span = max(h.max() - h.min(), 1.0)
    for h0 in h.min() - np.concatenate([np.linspace(0.05, 3.0, 30), np.geomspace(3.2, 40.0 * span, 80)]):
        x, y = np.log(h - h0), np.log(v)
        b, la = np.polyfit(x, y, 1)
        if not (1.0 <= b <= 6.0):
            continue
        pred = np.exp(la + b * x)
        err = float(np.sqrt(np.mean((pred - v) ** 2)))
        if best is None or err < best["rms_hm3"]:
            best = dict(a=float(np.exp(la)), b=float(b), h0=float(h0), rms_hm3=err)
    if best is None:
        return None
    best.update(n=int(len(h)), level_min=float(h.min()), level_max=float(h.max()), v_min=float(v.min()), v_max=float(v.max()),
                rel_rms=best["rms_hm3"] / max(float(v.max()), 1e-6), clipped_at_hm3=clipped)
    return best


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sites = json.loads((HERE / "dams_sites.json").read_text(encoding="utf-8"))["dams"]
    out = {}
    for s in sites:
        prefix = "chj" if s["saih"].startswith("chj:") else "seg"
        p = pairs(prefix, s["vars"]["level"], s["vars"]["volume"])
        if p is None:
            print(f"{s['id']:15s} no cached series")
            continue
        c = fit(*p)
        if c is None:
            print(f"{s['id']:15s} not enough spread of levels ({len(p[0])} pairs)")
            continue
        out[s["id"]] = {k: (float(f"{v:.8g}") if isinstance(v, float) else v) for k, v in c.items()}
        print(f"{s['id']:15s} n={c['n']:5d} level {c['level_min']:8.2f}..{c['level_max']:8.2f}  V {c['v_min']:8.3f}..{c['v_max']:8.3f}  "
              f"a={c['a']:.5g} b={c['b']:.3f} h0={c['h0']:.2f}  rms={c['rms_hm3']:.4f} hm3 ({100 * c['rel_rms']:.2f} %)"
              + (f"  CLIPPED at {c['clipped_at_hm3']:.2f}" if c["clipped_at_hm3"] else ""))
    (HERE / "out" / "curves.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(len(out), "curves")
