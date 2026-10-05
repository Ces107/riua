"""Hourly -> sub-hourly relation for forecasts that only give hourly rain: the mean excess above phi inside an hour of cell
mean p, as X = p G(phi / p), fitted on the radar excess ladders of the gauge period (1 km / 10 min OPERA, 2024-2026, the
radar production uses). Scale-free by construction (a multiplicative correction of the hour scales X exactly so), so G
is a function of the ratio r = u / X(0) alone; the check by intensity class is printed.

    py -3.11 hindcast/subhourly/relation.py        -> prints the table and the dependence on the hourly class
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUB = ROOT / "hindcast" / "cache" / "subhourly"
R_GRID = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0])


def main():
    days = json.loads((HERE / "days.json").read_text(encoding="utf-8"))
    ordinary = set(days["ordinary"]) - set(days["big"])
    rs, gs, x0s = [], [], []
    for d in sorted(ordinary):
        f = SUB / f"{d.replace('-', '')}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        lad = z["ladder"].astype(np.float32)               # (24, L, 68, 64)
        u = z["u"]
        x0 = lad[:, 0]
        ok = np.isfinite(x0) & (x0 >= 1.0)
        if not ok.any():
            continue
        x0v = x0[ok]
        for j in range(1, len(u)):
            xv = np.nan_to_num(lad[:, j][ok])
            rs.append(u[j] / x0v); gs.append(xv / x0v); x0s.append(x0v)
    r, g, x0 = np.concatenate(rs), np.concatenate(gs), np.concatenate(x0s)
    print(f"{len(r) // 9} cell-hours with >= 1 mm of radar rain")
    lr, edges = np.log(R_GRID), np.log(R_GRID)
    half = 0.5 * (edges[1] - edges[0])
    table = []
    for c in lr:
        m = np.abs(np.log(r) - c) <= half
        table.append(float(g[m].mean()) if m.sum() >= 50 else np.nan)
    print("ratio phi/p : mean G (all) | G by hourly class 1-5, 5-15, 15-40, >= 40 mm")
    for c, gv in zip(R_GRID, table):
        m = np.abs(np.log(r) - np.log(c)) <= half
        cls = [g[m & (x0 >= a) & (x0 < b)] for a, b in ((1, 5), (5, 15), (15, 40), (40, 1e9))]
        print(f"  {c:5.1f} : {gv:.4f} | " + "  ".join(f"{v.mean():.4f} (n {v.size})" if v.size >= 30 else "   -   " for v in cls))
    ok = np.isfinite(table)
    out = dict(ratios=[float(v) for v in R_GRID[ok]], G=[round(float(v), 5) for v in np.array(table)[ok]])
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
