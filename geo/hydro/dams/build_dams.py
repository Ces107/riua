"""Catchments and travel-time bands of the reservoirs, with the same flow network and flood velocity law as the
60 control points (geo/hydro/catchments/build) and the gauges (geo/hydro/gauges/build_gauges.py).

The catchment of a dam STOPS at the modelled dams above it ("own" catchment): what those dams release or spill is
passed down explicitly by the reservoir model (core/reservoirs.py), with the travel time dam -> dam computed here.

Inputs : scratch/h1-catchments/{dir_D.npy, acc_D.npy, dem_cop30_3s.npy, work_points.json}   (built by h1-catchments)
         geo/hydro/dams/dams_sites.json
Outputs: geo/hydro/dams/out/dam_points.json   per dam: snapped position, natural and own area, upstream dams with lag,
                                              next dam downstream, control points downstream with lag and area share
         geo/hydro/dams/out/time_area.npz     point_ids, point_idx, cell, lag_h, area_km2   (own catchment, Riuà grid cells)
         geo/hydro/dams/out/catchments.geojson  own catchment polygons, simplified (for the page)
         scratch/q10-dams/dam_table.txt

    py -3.11 geo/hydro/dams/build_dams.py          # ~4 min, ~2.5 GB RAM
    py -3.11 geo/hydro/dams/build_dams.py probe    # only the snapping table of the dams that are not in the h1 inventory
"""
import json
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "catchments", "build"))
from common import (SCR, ROOT, NROW, NCOL, W_LON0, W_LAT1, RES, G_LON0, G_LAT0, G_D, G_NX, G_NY, rc_of, row_lat, col_lon,   # noqa: E402
                    cell_area_rows, cell_dx_rows, cell_dy_km, downstream_index, topo_order)

OUT = os.path.join(HERE, "out")
SCR10 = os.path.join(ROOT, "scratch", "q10-dams")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SCR10, exist_ok=True)
PROBE = len(sys.argv) > 1 and sys.argv[1] == "probe"

# identical to geo/hydro/catchments/build/08_build_outputs.py
A_CHANNEL, V_HILL, V_MIN, V_MAX, B_AREA, K_FLOOD, SLOPE_REACH = 0.10, 0.40, 1.0, 4.0, 0.20, 15.0, 12


@njit(cache=True)
def step_lengths(ds, ncol, dxrow, dy):
    n = ds.size
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        j = ds[i]
        r = i // ncol
        if j < 0:
            out[i] = dy
            continue
        dr = j // ncol - r
        dc = (j - (j // ncol) * ncol) - (i - r * ncol)
        if dr != 0 and dc != 0:
            out[i] = (dxrow[r] ** 2 + dy ** 2) ** 0.5
        elif dr != 0:
            out[i] = dy
        else:
            out[i] = dxrow[r]
    return out


@njit(cache=True)
def monotone_profile(ds, order, z):
    zc = z.copy()
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        j = ds[i]
        if j >= 0 and zc[j] > zc[i]:
            zc[i] = zc[j]
    return zc


@njit(cache=True)
def reach_slope(ds, zc, steplen, nstep):
    n = ds.size
    s = np.zeros(n, dtype=np.float32)
    for i in range(n):
        j = i
        dist = 0.0
        for _ in range(nstep):
            if ds[j] < 0:
                break
            dist += steplen[j]
            j = ds[j]
        if dist > 0:
            s[i] = (zc[i] - zc[j]) / (dist * 1000.0)
    return s


@njit(cache=True)
def upstream_own(ds, order, outlet, steplen, v1, stop):
    """mask of the cells draining to `outlet` without crossing a cell flagged in `stop` (other dams), and the
    travel time (h) of each to the outlet; `nat` = the same without stopping"""
    n = ds.size
    nat = np.zeros(n, dtype=np.bool_)
    own = np.zeros(n, dtype=np.bool_)
    t1 = np.zeros(n, dtype=np.float32)
    nat[outlet] = True
    own[outlet] = True
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        j = ds[i]
        if j >= 0 and nat[j]:
            nat[i] = True
            t1[i] = t1[j] + steplen[i] / v1[i] / 3.6
            if own[j] and not stop[i]:
                own[i] = True
    return nat, own, t1


@njit(cache=True)
def walk_down(ds, start, steplen, v1, target, maxn):
    """follow the flow path from `start`; return (cells that are flagged in target, hours to reach each)"""
    cells = np.empty(64, dtype=np.int64)
    hours = np.empty(64, dtype=np.float64)
    k = 0
    t = 0.0
    i = start
    for _ in range(maxn):
        t += steplen[i] / v1[i] / 3.6
        i = ds[i]
        if i < 0:
            break
        if target[i] and k < 64:
            cells[k] = i
            hours[k] = t
            k += 1
    return cells[:k], hours[:k]


t0 = time.time()
d = np.load(os.path.join(SCR, "dir_D.npy"))
acc2 = np.load(os.path.join(SCR, "acc_D.npy"))
acc = acc2.ravel()
dem = np.nan_to_num(np.load(os.path.join(SCR, "dem_cop30_3s.npy")), nan=0.0).ravel()
ds = downstream_index(d)
order = topo_order(ds)
area = np.repeat(cell_area_rows()[:, None], NCOL, axis=1).ravel().astype(np.float32)
area[d.ravel() == 255] = 0
wp = json.load(open(os.path.join(SCR, "work_points.json"), encoding="utf-8"))
H1 = {q["id"]: q for q in wp["dams"]}
sites = json.load(open(os.path.join(HERE, "dams_sites.json"), encoding="utf-8"))["dams"]


def snap(lat, lon, r=4, amax=None, amin=None):
    r0, c0 = rc_of(lat, lon)
    cands = []
    for rr in range(max(0, r0 - r), min(NROW, r0 + r + 1)):
        for cc in range(max(0, c0 - r), min(NCOL, c0 + r + 1)):
            a = float(acc2[rr, cc])
            if (amax is not None and a > amax) or (amin is not None and a < amin):
                continue
            cands.append((a, rr, cc))
    cands.sort(reverse=True)
    return cands


lines = []
for s in sites:
    if s.get("h1"):
        q = H1[s["id"]]
        s.update(cell=int(q["cell"]), lat_snap=q["lat_snap"], lon_snap=q["lon_snap"], moved_m=0)
        continue
    c = snap(s["lat"], s["lon"], s.get("r", 4), s.get("amax"), s.get("amin"))
    a, rr, cc = c[0]
    s.update(cell=int(rr * NCOL + cc), lat_snap=float(row_lat(rr)), lon_snap=float(col_lon(cc)))
    s["moved_m"] = round(float(np.hypot((s["lat_snap"] - s["lat"]) * 111200, (s["lon_snap"] - s["lon"]) * 111200 * np.cos(np.deg2rad(s["lat"])))))
    lines.append(f"{s['id']:14s} snapped {a:8.1f} km2, moved {s['moved_m']:4d} m; other candidates (km2): "
                 + ", ".join(f"{x[0]:.0f}" for x in c[1:40:6]))
print("\n".join(lines))
if PROBE:
    sys.exit(0)

steplen = step_lengths(ds, NCOL, cell_dx_rows(), cell_dy_km())
S = reach_slope(ds, monotone_profile(ds, order, dem), steplen, SLOPE_REACH)
v1 = np.clip(K_FLOOD * np.sqrt(np.maximum(S, 0.0)) * np.power(np.maximum(acc, 1e-3), B_AREA), V_MIN, V_MAX).astype(np.float32)
v1[acc < A_CHANNEL] = V_HILL
del S
lat_r, lon_c = row_lat(np.arange(NROW)), col_lon(np.arange(NCOL))
gj = np.floor((lat_r - G_LAT0) / G_D).astype(np.int32)
gi = np.floor((lon_c - G_LON0) / G_D).astype(np.int32)
gj[(gj < 0) | (gj >= G_NY)] = -1
gi[(gi < 0) | (gi >= G_NX)] = -1
gidx = np.where((gj[:, None] >= 0) & (gi[None, :] >= 0), gj[:, None] * G_NX + gi[None, :], -1).astype(np.int32).ravel()
print("loaded", round(time.time() - t0, 1), "s")

dam_cell = {s["cell"]: s["id"] for s in sites}
is_dam = np.zeros(ds.size, dtype=np.bool_)
is_dam[list(dam_cell)] = True
cp_cell = {int(p["cell"]): p["id"] for p in wp["points"]}
is_target = is_dam.copy()
is_target[list(cp_cell)] = True
cps = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "geo", "hydro", "catchments", "out", "control_points.json"), encoding="utf-8"))["points"]}
inv = {q["id"]: q for q in json.load(open(os.path.join(ROOT, "geo", "hydro", "catchments", "out", "control_points.json"), encoding="utf-8"))["meta"]["dams_inventory"]}

ta = {k: [] for k in ("point_idx", "cell", "lag_h", "area_km2")}
points, polys = [], []
hdr = f"{'id':14s} {'A_nat':>8s} {'A_own':>8s} {'inbox':>6s} {'tmax':>5s}  upstream dams (lag h) | next dam | control points below (lag h, share)"
tab = [hdr]
for pi, s in enumerate(sites):
    stop = is_dam.copy()
    stop[s["cell"]] = False
    nat, own, t1 = upstream_own(ds, order, s["cell"], steplen, v1, stop)
    a_nat, a_own = float(area[nat].sum()), float(area[own].sum())
    # dams directly above: inside the natural catchment, and their own downstream cell belongs to this dam's own catchment
    ups = []
    for c, did in dam_cell.items():
        if did != s["id"] and nat[c] and ds[c] >= 0 and own[ds[c]]:
            ups.append(dict(id=did, lag_h=round(float(t1[c]), 1)))
    # downstream: next dam and control points, with travel times along the flow path
    cells, hours = walk_down(ds, s["cell"], steplen, v1, is_target, 80000)
    nxt, below, through = None, [], []
    for c, h in zip(cells, hours):
        c = int(c)
        if c in dam_cell:
            if nxt is None:
                nxt = dict(id=dam_cell[c], lag_h=round(float(h), 1))
            through.append(dam_cell[c])
        if c in cp_cell:
            p = cps[cp_cell[c]]
            below.append(dict(id=p["id"], lag_h=round(float(h), 1), share=round(min(a_nat / p["area_km2"], 1.0), 3),
                              through=list(through)))
    g = gidx[own]
    ok = g >= 0
    lag = np.floor(t1[own][ok]).astype(np.int64)
    w = np.bincount(g[ok].astype(np.int64) * 4096 + lag, weights=area[own][ok].astype(np.float64))
    nz = np.nonzero(w)[0]
    ta["point_idx"].append(np.full(nz.size, pi, dtype=np.int16))
    ta["cell"].append((nz // 4096).astype(np.int16))
    ta["lag_h"].append((nz % 4096).astype(np.int16))
    ta["area_km2"].append(w[nz].astype(np.float32))
    h1 = inv.get(s["id"])
    points.append(dict(
        id=s["id"], name=s["name"], river=s["river"], lat=round(s["lat_snap"], 5), lon=round(s["lon_snap"], 5), moved_m=s["moved_m"],
        area_km2=round(a_nat, 1), own_km2=round(a_own, 1), inside_box_fraction=round(float(w.sum()) / max(a_own, 1e-9), 4),
        t_longest_h=round(float(t1[own].max()), 1), elev_m=round(float(dem[s["cell"]])),
        cuts=bool(h1["cuts"]) if h1 else False, upstream=sorted(ups, key=lambda u: u["id"]), next_dam=nxt, points=below))
    polys.append((s["id"], a_own, own.reshape(NROW, NCOL)))
    tab.append(f"{s['id']:14s} {a_nat:8.1f} {a_own:8.1f} {points[-1]['inside_box_fraction']:6.3f} {points[-1]['t_longest_h']:5.1f}  "
               + ",".join(f"{u['id']}({u['lag_h']})" for u in points[-1]["upstream"]) + " | " + (f"{nxt['id']}({nxt['lag_h']})" if nxt else "-")
               + " | " + ",".join(f"{b['id']}({b['lag_h']},{b['share']})" for b in below))
    print(tab[-1], flush=True)

with open(os.path.join(SCR10, "dam_table.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines + tab) + "\n")
meta = dict(generated=time.strftime("%Y-%m-%d"), agent="q10-dams",
            method="same rasters and flood velocity law (k=15) as geo/hydro/catchments/out/README.md; own catchment = cells that reach the "
                   "dam without crossing another modelled dam; lags are travel times along the D8 flow path",
            grid=dict(lon0=G_LON0, lat0=G_LAT0, d=G_D, nx=G_NX, ny=G_NY, flat_index="j*64+i"))
with open(os.path.join(OUT, "dam_points.json"), "w", encoding="utf-8") as f:
    json.dump(dict(meta=meta, points=points), f, ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(OUT, "time_area.npz"), point_ids=np.array([p["id"] for p in points]),
                    **{k: np.concatenate(v) for k, v in ta.items()})

# own-catchment polygons, simplified
try:
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import shape, mapping, Polygon, MultiPolygon
    from shapely.ops import unary_union

    def rnd(o):
        if isinstance(o, (list, tuple)):
            if o and isinstance(o[0], (int, float)):
                return [round(o[0], 4), round(o[1], 4)]
            return [rnd(x) for x in o]
        return o

    feats = []
    for pid, a, m2 in polys:
        rr, cc = np.nonzero(m2.any(axis=1))[0], np.nonzero(m2.any(axis=0))[0]
        r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
        sub = m2[r0:r1, c0:c1].astype(np.uint8)
        tr = from_origin(W_LON0 + c0 * RES, W_LAT1 - r0 * RES, RES, RES)
        g = unary_union([shape(x) for x, v in features.shapes(sub, mask=sub.astype(bool), transform=tr, connectivity=8)])
        g = g.simplify(float(np.clip(0.0016 * np.sqrt(max(a, 1.0) / 40.0), 0.0012, 0.016)), preserve_topology=True)
        parts = [q for q in (g.geoms if isinstance(g, MultiPolygon) else [g]) if q.area > 0]
        big = max(q.area for q in parts)
        parts = [Polygon(q.exterior) for q in parts if q.area >= 0.03 * big]
        gg = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        gm = mapping(gg)
        feats.append(dict(type="Feature", properties=dict(id=pid, area_km2=round(a, 1)),
                          geometry=dict(type=gm["type"], coordinates=rnd(gm["coordinates"]))))
    txt = json.dumps(dict(type="FeatureCollection", features=feats), separators=(",", ":"))
    with open(os.path.join(OUT, "catchments.geojson"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("catchments.geojson", len(txt), "bytes")
except Exception as e:                                         # noqa: BLE001 - polygons are optional
    print("polygons skipped:", type(e).__name__, e)
print("dams", len(points), "done", round(time.time() - t0, 1), "s")
