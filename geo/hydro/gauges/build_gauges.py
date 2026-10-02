"""Catchments and travel-time bands of the river gauges, with the same flow network, dams,
regulation rule and velocity law as the 60 control points (geo/hydro/catchments/build).

Inputs : scratch/h1-catchments/{dir_D.npy, acc_D.npy, dem_cop30_3s.npy, work_points.json}   (built by h1-catchments)
         geo/hydro/gauges/gauges_src.json        the live river gauges (copied from the published snapshot)
         geo/hydro/gauges/gauges_decisions.json  per gauge: use / why not / snapping overrides / control point on the same stream
Outputs: geo/hydro/gauges/out/gauge_points.json  meta + points (same keys as control_points.json where they apply)
         geo/hydro/gauges/out/time_area.npz      point_ids, point_idx, scope (0 natural, 1 unregulated), cell, lag_h, area_km2
         geo/hydro/gauges/out/masks.npz          per point (gauges AND control points): W-grid cells of the scope-1 catchment
                                                 aggregated to 9 arc-seconds (for build_attributes.py)
         scratch/q7-hydro/gauge_table.txt

    py -3.11 geo/hydro/gauges/build_gauges.py            # ~3 min, ~2.5 GB RAM
    py -3.11 geo/hydro/gauges/build_gauges.py all        # table for EVERY gauge of the source list (to take the decisions)
"""
import json
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "catchments", "build"))
from common import (SCR, ROOT, NROW, NCOL, G_LON0, G_LAT0, G_D, G_NX, G_NY, rc_of, row_lat, col_lon,      # noqa: E402
                    cell_area_rows, cell_dx_rows, cell_dy_km, downstream_index, topo_order, label_upstream)
from cedexref import official_upstream                                                                      # noqa: E402

OUT = os.path.join(HERE, "out")
SCR7 = os.path.join(ROOT, "scratch", "q7-hydro")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SCR7, exist_ok=True)
ALL = len(sys.argv) > 1 and sys.argv[1] == "all"

# identical to geo/hydro/catchments/build/08_build_outputs.py
A_CHANNEL, V_HILL, V_MIN, V_MAX, B_AREA, K_FLOOD, SLOPE_REACH, EQUIV_MM = 0.10, 0.40, 1.0, 4.0, 0.20, 15.0, 12, 20.0


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
def upstream_fields(ds, order, outlet, steplen, v1):
    n = ds.size
    m = np.zeros(n, dtype=np.bool_)
    dist = np.zeros(n, dtype=np.float32)
    t1 = np.zeros(n, dtype=np.float32)
    m[outlet] = True
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        j = ds[i]
        if j >= 0 and m[j]:
            m[i] = True
            dist[i] = dist[j] + steplen[i]
            t1[i] = t1[j] + steplen[i] / v1[i] / 3.6
    return m, dist, t1


def temez(L_km, J):
    return 0.3 * (L_km / max(J, 1e-5) ** 0.25) ** 0.76


t0 = time.time()
d = np.load(os.path.join(SCR, "dir_D.npy"))
acc2 = np.load(os.path.join(SCR, "acc_D.npy"))
acc = acc2.ravel()
dem = np.nan_to_num(np.load(os.path.join(SCR, "dem_cop30_3s.npy")), nan=0.0).ravel()
ds = downstream_index(d)
order = topo_order(ds)
area = np.repeat(cell_area_rows()[:, None], NCOL, axis=1).ravel().astype(np.float32)
area[d.ravel() == 255] = 0
steplen = step_lengths(ds, NCOL, cell_dx_rows(), cell_dy_km())
S = reach_slope(ds, monotone_profile(ds, order, dem), steplen, SLOPE_REACH)
v1 = np.clip(K_FLOOD * np.sqrt(np.maximum(S, 0.0)) * np.power(np.maximum(acc, 1e-3), B_AREA), V_MIN, V_MAX).astype(np.float32)
v1[acc < A_CHANNEL] = V_HILL
lat_r, lon_c = row_lat(np.arange(NROW)), col_lon(np.arange(NCOL))
gj = np.floor((lat_r - G_LAT0) / G_D).astype(np.int32)
gi = np.floor((lon_c - G_LON0) / G_D).astype(np.int32)
gj[(gj < 0) | (gj >= G_NY)] = -1
gi[(gi < 0) | (gi >= G_NX)] = -1
gidx = np.where((gj[:, None] >= 0) & (gi[None, :] >= 0), gj[:, None] * G_NX + gi[None, :], -1).astype(np.int32).ravel()
print("loaded", round(time.time() - t0, 1), "s")

# ---- dams: which ones cut (same loop as 08_build_outputs.py) -----------------------------------
wp = json.load(open(os.path.join(SCR, "work_points.json"), encoding="utf-8"))
DAMS = sorted(wp["dams"], key=lambda q: q["area_km2"])
first1 = np.full(ds.size, -1, dtype=np.int8)
dam_cells = np.array([q["cell"] for q in DAMS])
for qi, q in enumerate(DAMS):
    m = label_upstream(ds, order, q["cell"])
    own = float(area[m & (first1 < 0)].sum())
    q["equiv_mm"] = float(q["capacity_hm3"] / own * 1000.0) if own > 0 else 0.0
    q["cuts"] = bool(q["equiv_mm"] >= EQUIV_MM and not q.get("no_regulation"))
    if q["cuts"]:
        first1[m & (first1 < 0)] = qi
print("dams", round(time.time() - t0, 1), "s")

# ---- gauges ------------------------------------------------------------------------------------
src = json.load(open(os.path.join(HERE, "gauges_src.json"), encoding="utf-8"))["gauges"]
dec_f = os.path.join(HERE, "gauges_decisions.json")
_dec = json.load(open(dec_f, encoding="utf-8")) if os.path.exists(dec_f) else {}
DEC = _dec.get("gauges", {})
if not ALL:
    # virtual points: places with a published flood figure but no live gauge (reservoir inflows); natural catchment
    for v in _dec.get("virtual", []):
        src.append(dict(id=v["id"], name=v["name"], river=v.get("river", ""), lat=v["lat"], lon=v["lon"], source="virtual", kind="virtual"))
        DEC[v["id"]] = dict(use=True, natural=True, note=v.get("note", ""), r=v.get("r", 6))


def snap(lat, lon, r=4, amax=None, amin=None):
    r0, c0 = rc_of(lat, lon)
    best = None
    for rr in range(max(0, r0 - r), min(NROW, r0 + r + 1)):
        for cc in range(max(0, c0 - r), min(NCOL, c0 + r + 1)):
            a = acc2[rr, cc]
            if (amax is not None and a > amax) or (amin is not None and a < amin):
                continue
            if best is None or a > best[0]:
                best = (float(a), rr, cc)
    return best


def add_ta(store, pi, scope, sel, tt):
    g = gidx[sel]
    ok = g >= 0
    lag = np.floor(tt[sel][ok]).astype(np.int64)
    w = np.bincount(g[ok].astype(np.int64) * 4096 + lag, weights=area[sel][ok].astype(np.float64))
    nz = np.nonzero(w)[0]
    store["point_idx"].append(np.full(nz.size, pi, dtype=np.int16))
    store["scope"].append(np.full(nz.size, scope, dtype=np.int8))
    store["cell"].append((nz // 4096).astype(np.int16))
    store["lag_h"].append((nz % 4096).astype(np.int16))
    store["area_km2"].append(w[nz].astype(np.float32))


def mask9(sel):
    """catchment mask aggregated 3x3 -> 9 arc-seconds: flat indices (1680 x 1800) and cell counts 1..9"""
    m = sel.reshape(NROW // 3, 3, NCOL // 3, 3).sum(axis=(1, 3)).ravel().astype(np.uint8)
    nz = np.nonzero(m)[0]
    return nz.astype(np.int32), m[nz]


ta = {k: [] for k in ("point_idx", "scope", "cell", "lag_h", "area_km2")}
points, lines, masks = [], [], {}
hdr = f"{'id':12s} {'name':30s} {'river':22s} {'A_nat':>8s} {'A_unreg':>8s} {'CEDEX lo..hi':>19s} {'tmax':>5s} {'tc':>5s} moved_m  dams (cutting)"
lines.append(hdr)
pi = 0
for g in src:
    dc = DEC.get(g["id"], {})
    if not ALL and not dc.get("use"):
        continue
    if ALL and g["kind"] != "gauge":
        continue
    s = snap(dc.get("lat", g["lat"]), dc.get("lon", g["lon"]), dc.get("r", 4), dc.get("amax"), dc.get("amin"))
    if s is None:
        lines.append(f"{g['id']:12s} {g['name'][:30]:30s} NO SNAP")
        continue
    a, r, c = s
    cell = r * NCOL + c
    lat, lon = float(row_lat(r)), float(col_lon(c))
    moved = float(np.hypot((lat - g["lat"]) * 111200, (lon - g["lon"]) * 111200 * np.cos(np.deg2rad(lat))))
    m, dist, t1 = upstream_fields(ds, order, cell, steplen, v1)
    dam_in = np.append(m[dam_cells], False)
    u1 = m.copy() if dc.get("natural") else m & ~dam_in[first1]
    a_nat, a_u1 = float(area[m].sum()), float(area[u1].sum())
    off = official_upstream(lat, lon)
    cs = f"{off['lo']:9.1f}..{off['hi']:8.1f}" if off else f"{'-':>19s}"
    head = int(np.argmax(np.where(m, dist, -1)))
    L = float(dist[head])
    J = max((float(dem[head]) - float(dem[cell])) / (max(L, 0.05) * 1000.0), 1e-4)
    head_u = int(np.argmax(np.where(u1, dist, -1)))
    L_u = float(dist[head_u])
    J_u = max((float(dem[head_u]) - float(dem[cell])) / (max(L_u, 0.05) * 1000.0), 1e-4)
    inside = [q for q in DAMS if m[q["cell"]]]
    dl = [dict(id=q["id"], name=q["name"], capacity_hm3=q["capacity_hm3"], catchment_km2=round(q["area_km2"], 1),
               equiv_mm=round(q["equiv_mm"], 1), cuts=q["cuts"], travel_h=round(float(t1[q["cell"]]), 1)) for q in inside]
    lines.append(f"{g['id']:12s} {g['name'][:30]:30s} {(g['river'] or '')[:22]:22s} {a_nat:8.1f} {a_u1:8.1f} {cs} {float(t1[u1].max()):5.1f} "
                 f"{temez(L_u, J_u):5.1f} {moved:6.0f}  " + ",".join(q["id"] + ("*" if q["cuts"] else "") for q in inside))
    if ALL:
        continue
    pid = {"saih_chj": "chj-", "saih_segura": "seg-", "virtual": ""}.get(g["source"], "ebr-") + g["id"]
    for scope, sel in ((0, m), (1, u1)):
        add_ta(ta, pi, scope, sel, t1)
    masks[pid] = mask9(u1)
    zs = dem[u1]
    points.append(dict(
        id=pid, var=g["id"], source=g["source"], name=g["name"], stream=g["river"], lat=round(lat, 5), lon=round(lon, 5),
        gauge_lat=g["lat"], gauge_lon=g["lon"], moved_m=round(moved),
        area_km2=round(a_nat, 1), area_unregulated_km2=round(a_u1, 1),
        inside_box_fraction=round(float(area[m & (gidx >= 0)].sum()) / a_nat, 4),
        L_km=round(L, 1), slope=round(J, 5), tc_h=round(temez(L, J), 2),
        L_unregulated_km=round(L_u, 1), slope_unregulated=round(J_u, 5), tc_unregulated_h=round(temez(L_u, J_u), 2),
        t_longest_h=round(float(t1[m].max()), 1), t_longest_unregulated_h=round(float(t1[u1].max()), 1),
        elev_outlet_m=round(float(dem[cell])), elev_mean_m=round(float(np.average(zs, weights=np.maximum(area[u1], 1e-9)))),
        cedex_bracket_km2=[round(off["lo"], 1), round(off["hi"], 1)] if off else None,
        dams=dl, control_point=dc.get("control_point"), regulated=dc.get("regulated", False), note=dc.get("note", "")))
    pi += 1

txt = "\n".join(lines)
with open(os.path.join(SCR7, "gauge_table_all.txt" if ALL else "gauge_table.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print(txt)
if ALL:
    sys.exit(0)

# control-point masks too (scope 1), so the attributes of gauged and ungauged catchments come from one code path
for p in wp["points"]:
    m = label_upstream(ds, order, p["cell"])
    dam_in = np.append(m[dam_cells], False)
    masks[p["id"]] = mask9(m & ~dam_in[first1])

meta = dict(generated=time.strftime("%Y-%m-%d"), agent="q7-hydro",
            method="identical to geo/hydro/catchments/out/README.md (same rasters, dams, 20 mm regulation rule, flood velocity law k=15)",
            grid=dict(lon0=G_LON0, lat0=G_LAT0, d=G_D, nx=G_NX, ny=G_NY, flat_index="j*64+i"),
            scopes={"0": "natural", "1": "unregulated (below dams storing >= 20 mm of their own catchment)"})
with open(os.path.join(OUT, "gauge_points.json"), "w", encoding="utf-8") as f:
    json.dump(dict(meta=meta, points=points), f, ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(OUT, "time_area.npz"), point_ids=np.array([p["id"] for p in points]),
                    **{k: np.concatenate(v) for k, v in ta.items()})
ids = list(masks)
ptr = np.cumsum([0] + [masks[i][0].size for i in ids])
np.savez_compressed(os.path.join(OUT, "masks.npz"), ids=np.array(ids), ptr=ptr,
                    cell9=np.concatenate([masks[i][0] for i in ids]), n=np.concatenate([masks[i][1] for i in ids]))
print("points", len(points), "masks", len(ids), "done", round(time.time() - t0, 1), "s")
