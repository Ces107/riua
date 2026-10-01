"""Systematic check of an accumulation raster against the official CEDEX/DGA 1:25,000
basins of the Jucar demarcation: for every named river whose Pfafstetter basin
(sum of CuencaKm2 over PFAFCUEN starting with the river code) is > MIN km2, sample the
accumulated area near the river mouth and compare.

Usage: py -3.11 06_validate_cedex.py D [min_km2]
"""
import glob
import os
import sys
import numpy as np
import geopandas as gpd
from shapely.ops import linemerge
from common import SCR, ROOT, NROW, NCOL, rc_of

tag = sys.argv[1] if len(sys.argv) > 1 else "D"
MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
acc = np.load(os.path.join(SCR, f"acc_{tag}.npy"), mmap_mode="r")
dem = np.load(os.path.join(SCR, "dem_cop30_3s.npy"), mmap_mode="r")
base = os.path.join(ROOT, "scratch", "r5-geo", "chj")
riv = gpd.read_file(glob.glob(os.path.join(base, "F850*", "*.shp"))[0])
sub = gpd.read_file(glob.glob(os.path.join(base, "F851*", "*.shp"))[0])
sub["PFAFCUEN"] = sub["PFAFCUEN"].astype(str)
riv["PFAFRIO"] = riv["PFAFRIO"].astype(str)
codes = np.sort(sub.PFAFCUEN.values)
areas = sub.set_index("PFAFCUEN").CuencaKm2.sort_index()
cum = np.concatenate([[0], np.cumsum(areas.values)])
keys = areas.index.values


def basin(code):
    a = np.searchsorted(keys, code, "left")
    b = np.searchsorted(keys, code + "￿", "right")
    return cum[b] - cum[a]


r4 = riv.dissolve("PFAFRIO", aggfunc={"NomRio": "first", "LongRioKm": "sum"}).to_crs(4326)
rows = []
for code, row in r4.iterrows():
    a = basin(code)
    if a < MIN:
        continue
    g = row.geometry
    if g.geom_type == "MultiLineString":
        g = linemerge(g)
    parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
    # mouth = lowest end among all parts
    best = None
    for p in parts:
        for end, pt in ((0, p.coords[0]), (1, p.coords[-1])):
            r, c = rc_of(pt[1], pt[0])
            if 0 <= r < NROW and 0 <= c < NCOL:
                zz = float(np.nan_to_num(dem[r, c], nan=0.0))
                if best is None or zz < best[0]:
                    best = (zz, p, end)
    if best is None:
        continue
    _, p, end = best
    L = p.length
    vals = []
    for d_m in (300, 500, 700, 900, 1200, 1500):
        dd = d_m / 100000.0
        if dd > L * 0.8:
            continue
        pt = p.interpolate(L - dd if end == 1 else dd)
        r, c = rc_of(pt.y, pt.x)
        vals.append(float(acc[max(0, r - 2):r + 3, max(0, c - 2):c + 3].max()))
    if not vals:
        continue
    v = float(np.median(vals))
    rows.append((code, row.NomRio, a, v, v / a, p.coords[-1 if end else 0]))
rows.sort(key=lambda x: -x[2])
bad = 0
print(f"{'code':14s} {'river':42s} {'official':>9s} {'acc_' + tag:>9s} ratio  mouth(lon,lat)")
for code, nm, a, v, q, m in rows:
    flag = "" if 0.9 <= q <= 1.1 else ("  <-- " if 0.8 <= q <= 1.25 else "  <== BAD")
    bad += flag != ""
    if flag or a > 300:
        print(f"{code:14s} {str(nm)[:42]:42s} {a:9.1f} {v:9.1f} {q:5.2f}  ({m[0]:.3f},{m[1]:.3f}){flag}")
q = np.array([r[4] for r in rows])
print(f"n={len(rows)}; within 10%: {(np.abs(q - 1) <= 0.1).mean():.2%}; within 20%: {(np.abs(q - 1) <= 0.2).mean():.2%}; median ratio {np.median(q):.3f}")
