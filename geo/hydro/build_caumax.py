"""Return-period flood peaks at every control point from CEDEX CAUMAX v3.0.

CAUMAX ("Mapa de caudales máximos", CEDEX for MITECO) gives natural-regime peak discharge
for T = 2, 5, 10, 25, 100 and 500 years on 500 m river cells with a catchment of 50 km2 or
more. Source: https://ceh.cedex.es/caumax/caumax_v30.zip (q*.tif, dir.tif), EPSG:25830.

For each control point:
  1. flow accumulation from CAUMAX's own D8 directions gives the catchment area of each cell,
  2. the cell within 2 km whose area best matches the point's natural catchment area is taken
     (score = |ln(A_cell / A_point)| + distance / 4 km), only if the areas agree within x1.6,
  3. points under the 50 km2 CAUMAX limit, or without a matching cell, borrow the nearest
     matching cell upstream or downstream on the same stream and scale it with Q ~ A^0.75,
  4. points below large dams: the flows are scaled to the unregulated area, Q ~ A^0.75
     (the risk model only routes rain from below the dams).

Output: geo/hydro/caumax_points.json  {id: {T2..T500 (m3/s), area_cell_km2, how, dist_km}}
Only needed when the control points change; the pipeline reads the JSON.
"""
import json
import math
from pathlib import Path

import numpy as np
import pyflwdir
import rasterio
from pyproj import Transformer

H = Path(__file__).resolve().parent
REPO = H.parents[1]
SRC = REPO / "scratch" / "h2-sections" / "misc" / "caumax" / "data"
TS = (2, 5, 10, 25, 100, 500)

cps = json.loads((H / "catchments" / "out" / "control_points.json").read_text(encoding="utf-8"))
cps = cps["points"] if isinstance(cps, dict) else cps
tr = Transformer.from_crs(4326, 25830, always_xy=True)

# window covering every catchment that reaches the Comunitat Valenciana (Júcar, Turia, Segura, Mijares...)
x0, y0 = tr.transform(-4.6, 37.2)
x1, y1 = tr.transform(1.0, 41.4)
with rasterio.open(SRC / "dir.tif") as r:
    win = rasterio.windows.from_bounds(x0, y0, x1, y1, r.transform).round_offsets().round_lengths()
    d8 = r.read(1, window=win)
    tf = r.window_transform(win)
    nod = r.nodata
q = {}
for T in TS:
    with rasterio.open(SRC / f"q{T}.tif") as r:
        # the q rasters do not share dir.tif's origin: shift the window by the origin difference
        dc = int(round((tf.c - r.transform.c) / r.transform.a))
        dr = int(round((tf.f - r.transform.f) / r.transform.e))
        qwin = rasterio.windows.Window(dc, dr, win.width, win.height)
        a = r.read(1, window=qwin, boundless=True, fill_value=r.nodata).astype(np.float64)
        a[(a < 0) | (a > 1e6)] = np.nan
        q[T] = a
vals = np.unique(d8[d8 != nod])
print("d8 codes:", vals[:12])
print("cells with code 3:", int((d8 == 3).sum()))
d8u = np.where(d8 == nod, 247, np.where(d8 == 3, 0, d8)).astype(np.uint8)     # 3 is not a D8 code: treat as a pit
flw = pyflwdir.from_array(d8u, ftype="d8", transform=tf, latlon=False)
area = flw.upstream_area(unit="km2")
print("max area in window km2:", float(np.nanmax(area)))

ny, nx = d8.shape
ys = tf.f + (np.arange(ny) + 0.5) * tf.e
xs = tf.c + (np.arange(nx) + 0.5) * tf.a
out = {}
for p in cps:
    A = float(p["area_km2"])
    Au = float(p.get("area_unregulated_km2") or A)
    px, py = tr.transform(p["lon"], p["lat"])
    i0, j0 = int((px - tf.c) / tf.a), int((py - tf.f) / tf.e)
    best, how = None, None
    for rad, limit in ((4, 1.6), (8, 2.5)):            # 2 km, then 4 km with a looser area match
        sl = (slice(max(j0 - rad, 0), j0 + rad + 1), slice(max(i0 - rad, 0), i0 + rad + 1))
        jj, ii = np.mgrid[sl]
        cand = np.isfinite(q[2][sl]) & (area[sl] > 0)
        if not cand.any():
            continue
        dist = np.hypot(xs[ii] - px, ys[jj] - py) / 1000.0
        score = np.abs(np.log(area[sl] / max(A, 1.0))) + dist / 4.0
        score[~cand] = np.inf
        k = np.unravel_index(np.argmin(score), score.shape)
        ratio = area[sl][k] / max(A, 1.0)
        if np.isfinite(score[k]) and 1 / limit <= ratio <= limit:
            best = (jj[k], ii[k], float(dist[k]))
            how = "caumax" if rad == 4 else "caumax (4 km)"
            break
    rec = {"area_point_km2": round(A, 1)}
    if best is not None:
        j, i, d = best
        Ac = float(area[j, i])
        f = (A / Ac) ** 0.75 * (Au / A) ** 0.75          # to the point's area, then to its unregulated part
        rec.update({f"T{T}": round(float(q[T][j, i]) * f, 1) for T in TS})
        rec.update(area_cell_km2=round(Ac, 1), dist_km=round(d, 2), how=how + (" + presas" if Au < 0.95 * A else ""))
    out[p["id"]] = rec

# points left without a cell (small ravines): nearest point on the same stream that has one
by_stream = {}
for p in cps:
    by_stream.setdefault((p.get("stream") or "").split(" (")[0], []).append(p)
for p in cps:
    rec = out[p["id"]]
    if "T2" in rec:
        continue
    A = float(p["area_km2"]); Au = float(p.get("area_unregulated_km2") or A)
    sib = [s for s in by_stream.get((p.get("stream") or "").split(" (")[0], []) if "T2" in out[s["id"]]]
    if sib:
        s = min(sib, key=lambda s: abs(math.log(float(s["area_km2"]) / A)))
        f = (Au / float(s.get("area_unregulated_km2") or s["area_km2"])) ** 0.75
        rec.update({f"T{T}": round(out[s["id"]][f"T{T}"] * f, 1) for T in TS})
        rec.update(how=f"escalado desde {s['id']}")
# still nothing (tiny coastal ravines): nearest point that has values, scaled by area
have_pts = [p for p in cps if "T2" in out[p["id"]]]
for p in cps:
    rec = out[p["id"]]
    if "T2" in rec:
        continue
    s = min(have_pts, key=lambda s: (s["lat"] - p["lat"]) ** 2 + (s["lon"] - p["lon"]) ** 2)
    f = (float(p.get("area_unregulated_km2") or p["area_km2"]) / float(s.get("area_unregulated_km2") or s["area_km2"])) ** 0.75
    rec.update({f"T{T}": round(out[s["id"]][f"T{T}"] * f, 1) for T in TS})
    rec.update(how=f"escalado desde {s['id']} (vecino)")
(H / "caumax_points.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
have = sum(1 for v in out.values() if "T2" in v)
print(f"{have} of {len(out)} points with CAUMAX flows")
for k, v in out.items():
    print(f"  {k:24s} A {v['area_point_km2']:8.0f}  " + ("  ".join(f"T{T} {v.get(f'T{T}', float('nan')):7.0f}" for T in TS)) + f"  {v.get('how', 'NONE')}")
