"""Catchment attributes for every gauged catchment AND the 60 control points (same code path), from
geo/hydro/gauges/out/masks.npz (scope-1 catchments on the 9 arc-second lattice).

Rasters (9", 1680 x 1800, origin lon -3.5 / lat 41.2), built by sub-agent q7b-attr into scratch/q7-hydro/attr/:
  litho_9s.npy   IGME Mapa de Permeabilidades 1:200.000 (ArcGIS REST .../Cartografia_Tematica/IGME_Permeabilidad_200/MapServer/0)
                 code = lithotype*10 + permeability rank (1 very high .. 5 very low); 1x carbonate, 2x detrital,
                 3x Quaternary, 4x evaporitic, 5x meta-detrital, 6x/7x igneous
  clc_9s.npy     CORINE Land Cover 2018 (EEA discomap CLC2018_WM), 3-digit code
  map_9s.npy     CHELSA v2.1 bio12 1981-2010, mm/yr
plus slope from the Copernicus GLO-30 DEM of h1 (scratch/h1-catchments/dem_cop30_3s.npy), the CEDEX P0i raster as Riuà
cell means (scratch/q5-science/p0_cells.npy, built by q5-science) and the CAUMAX 2-year flood (scratch/h2-sections/misc/caumax).

Output: geo/hydro/gauges/out/attributes.json   {point id: {...}}

    py -3.11 geo/hydro/gauges/build_attributes.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ATTR = os.path.join(ROOT, "scratch", "q7-hydro", "attr")
OUT = os.path.join(HERE, "out")
N9R, N9C = 1680, 1800
RES9 = 9.0 / 3600.0

litho = np.load(os.path.join(ATTR, "litho_9s.npy")).ravel()
clc = np.load(os.path.join(ATTR, "clc_9s.npy")).ravel()
mapr = np.load(os.path.join(ATTR, "map_9s.npy")).ravel()

# terrain slope (m/m) at 3", averaged to 9"
f_sl = os.path.join(ATTR, "slope_9s.npy")
if os.path.exists(f_sl):
    slope = np.load(f_sl).ravel()
else:
    dem = np.nan_to_num(np.load(os.path.join(ROOT, "scratch", "h1-catchments", "dem_cop30_3s.npy")), nan=0.0).astype(np.float32)
    lat = 41.2 - (np.arange(dem.shape[0]) + 0.5) * 3.0 / 3600.0
    dy = 6371008.8 * np.deg2rad(3.0 / 3600.0)
    dx = dy * np.cos(np.deg2rad(lat))[:, None]
    gy, gx = np.gradient(dem)
    sl = np.hypot(gx / dx, gy / dy).astype(np.float32)
    del gy, gx, dem
    slope = sl.reshape(N9R, 3, N9C, 3).mean(axis=(1, 3))
    np.save(f_sl, slope)
    slope = slope.ravel()

z = np.load(os.path.join(OUT, "masks.npz"))
ids, ptr, cell9, n9 = [str(x) for x in z["ids"]], z["ptr"], z["cell9"], z["n"]
lat9 = 41.2 - (np.arange(N9R) + 0.5) * RES9
a9 = (6371.0088 * np.deg2rad(RES9)) ** 2 * np.cos(np.deg2rad(lat9))          # km2 of a full 9" cell per row

gp = {p["id"]: p for p in json.load(open(os.path.join(OUT, "gauge_points.json"), encoding="utf-8"))["points"]}
cp = {p["id"]: p for p in json.load(open(os.path.join(HERE, "..", "catchments", "out", "control_points.json"), encoding="utf-8"))["points"]}
caumax_cp = json.load(open(os.path.join(HERE, "..", "caumax_points.json"), encoding="utf-8"))

# ---- CAUMAX 2-year flood at the gauges (same matching rule as geo/hydro/build_caumax.py) ----------------
def caumax_q2(points):
    import pyflwdir
    import rasterio
    from pyproj import Transformer
    src = os.path.join(ROOT, "scratch", "h2-sections", "misc", "caumax", "data")
    tr = Transformer.from_crs(4326, 25830, always_xy=True)
    x0, y0 = tr.transform(-4.6, 37.2)
    x1, y1 = tr.transform(1.0, 41.4)
    with rasterio.open(os.path.join(src, "dir.tif")) as r:
        win = rasterio.windows.from_bounds(x0, y0, x1, y1, r.transform).round_offsets().round_lengths()
        d8 = r.read(1, window=win); tf = r.window_transform(win); nod = r.nodata
    with rasterio.open(os.path.join(src, "q2.tif")) as r:
        dc = int(round((tf.c - r.transform.c) / r.transform.a)); dr = int(round((tf.f - r.transform.f) / r.transform.e))
        q2 = r.read(1, window=rasterio.windows.Window(dc, dr, win.width, win.height), boundless=True, fill_value=r.nodata).astype(np.float64)
        q2[(q2 < 0) | (q2 > 1e6)] = np.nan
    d8u = np.where(d8 == nod, 247, np.where(d8 == 3, 0, d8)).astype(np.uint8)
    area = pyflwdir.from_array(d8u, ftype="d8", transform=tf, latlon=False).upstream_area(unit="km2")
    ny, nx = d8.shape
    ys = tf.f + (np.arange(ny) + 0.5) * tf.e
    xs = tf.c + (np.arange(nx) + 0.5) * tf.a
    out = {}
    for p in points:
        A = float(p["area_km2"]); Au = float(p.get("area_unregulated_km2") or A)
        px, py = tr.transform(p["lon"], p["lat"])
        i0, j0 = int((px - tf.c) / tf.a), int((py - tf.f) / tf.e)
        for rad, limit in ((4, 1.6), (8, 2.5)):
            sl_ = (slice(max(j0 - rad, 0), j0 + rad + 1), slice(max(i0 - rad, 0), i0 + rad + 1))
            jj, ii = np.mgrid[sl_]
            cand = np.isfinite(q2[sl_]) & (area[sl_] > 0)
            if not cand.any():
                continue
            dist = np.hypot(xs[ii] - px, ys[jj] - py) / 1000.0
            score = np.abs(np.log(area[sl_] / max(A, 1.0))) + dist / 4.0
            score[~cand] = np.inf
            k = np.unravel_index(np.argmin(score), score.shape)
            ratio = area[sl_][k] / max(A, 1.0)
            if np.isfinite(score[k]) and 1 / limit <= ratio <= limit:
                out[p["id"]] = float(q2[sl_][k]) * (A / float(area[sl_][k])) ** 0.75 * (Au / A) ** 0.75
                break
    return out


q2_g = caumax_q2(list(gp.values()))

# ---- CEDEX P0i per Riuà cell, weighted with the time-area tables --------------------------------------------
p0c = np.load(os.path.join(ROOT, "scratch", "q5-science", "p0_cells.npy")).ravel()


def p0i_mean(ta, pid_list):
    out = {}
    use = ta["scope"] == 1
    for k, pid in enumerate(pid_list):
        s = use & (ta["point_idx"] == k)
        w = np.bincount(ta["cell"][s].astype(int), ta["area_km2"][s], p0c.size)
        ok = (w > 0) & np.isfinite(p0c)
        if ok.any():
            out[pid] = float((w[ok] * p0c[ok]).sum() / w[ok].sum())
    return out


ta_g = np.load(os.path.join(OUT, "time_area.npz"))
ta_c = np.load(os.path.join(HERE, "..", "catchments", "out", "time_area.npz"))
p0i = {**p0i_mean(ta_g, [str(x) for x in ta_g["point_ids"]]), **p0i_mean(ta_c, [str(x) for x in ta_c["point_ids"]])}

LITHO = {"carb_high": (11, 12), "carb_med": (13,), "marl_clay": (14, 15, 24, 25, 43, 44, 45), "detrital": (21, 22, 23),
         "quaternary": (31, 32, 33, 34), "hard_rock": (54, 55, 64, 65, 74, 75)}
out = {}
for k, pid in enumerate(ids):
    c, n = cell9[ptr[k]:ptr[k + 1]], n9[ptr[k]:ptr[k + 1]].astype(np.float64)
    w = n / 9.0 * a9[c // N9C]
    p = gp.get(pid) or cp[pid]
    rec = dict(kind="gauge" if pid in gp else "control_point",
               area_km2=round(float(p.get("area_unregulated_km2") or p["area_km2"]), 1), area_mask_km2=round(float(w.sum()), 1),
               lat=p["lat"], lon=p["lon"], elev_mean_m=p.get("elev_mean_m"),
               tc_h=p.get("tc_unregulated_h") or p.get("tc_h"), t_longest_h=p.get("t_longest_unregulated_h") or p.get("t_longest_h"))
    li = litho[c]
    ok = (li > 0) & (li != 90)
    wl = w[ok].sum()
    for name, codes in LITHO.items():
        rec[name] = round(float(w[ok & np.isin(li, codes)].sum() / wl), 3) if wl > 0 else None
    rank = li % 10
    rec["perm_high"] = round(float(w[ok & (rank <= 2)].sum() / wl), 3) if wl > 0 else None       # very high + high permeability
    rec["perm_low"] = round(float(w[ok & (rank >= 4)].sum() / wl), 3) if wl > 0 else None        # low + very low
    rec["perm_rank"] = round(float((w[ok] * rank[ok]).sum() / wl), 2) if wl > 0 else None        # 1 (very high) .. 5 (very low)
    cl = clc[c]
    okc = cl > 0
    wc = w[okc].sum()
    for name, lo, hi in (("urban", 100, 199), ("crops", 200, 299), ("forest", 310, 319), ("scrub", 320, 339)):
        rec[name] = round(float(w[okc & (cl >= lo) & (cl <= hi)].sum() / wc), 3) if wc > 0 else None
    rec["map_mm"] = round(float((w * mapr[c]).sum() / w.sum()))
    rec["slope"] = round(float((w * slope[c]).sum() / w.sum()), 4)
    rec["p0i_mm"] = round(p0i[pid], 1) if pid in p0i else None
    q2 = q2_g.get(pid) if pid in gp else (caumax_cp.get(pid) or {}).get("T2")
    rec["q2_m3s"] = round(float(q2), 1) if q2 else None
    rec["q2_spec"] = round(float(q2) / max(rec["area_km2"], 1.0) ** 0.75, 3) if q2 else None      # m3/s per km^1.5
    out[pid] = rec

with open(os.path.join(OUT, "attributes.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
keys = ["area_km2", "carb_high", "carb_med", "marl_clay", "detrital", "quaternary", "perm_rank", "urban", "crops", "forest", "scrub", "map_mm", "slope", "p0i_mm", "q2_spec"]
print(f"{'id':24s}" + "".join(f"{k[:9]:>10s}" for k in keys))
for pid, r in out.items():
    print(f"{pid:24s}" + "".join(f"{(r[k] if r[k] is not None else float('nan')):10.3f}" for k in keys))
