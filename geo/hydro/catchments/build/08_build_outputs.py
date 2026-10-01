"""Catchments, dams, Temez, travel-time field, time-area tables and all outputs.

Inputs : scratch/h1-catchments/{dir_D.npy, acc_D.npy, dem_cop30_3s.npy, burn_D.npy, work_points.json}
Outputs: geo/hydro/catchments/out/{control_points.json, time_area.npz, catchments.geojson, streams.geojson}
         scratch/h1-catchments/{overview.png, final_table.txt, travel_checks.txt}

Usage: py -3.11 08_build_outputs.py            (full build)
       py -3.11 08_build_outputs.py calib      (print calibration diagnostics for several k, no outputs)
"""
import json
import os
import sys
import time
import numpy as np
from numba import njit
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import shape, mapping, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from common import (SCR, OUT, NROW, NCOL, W_LON0, W_LAT1, RES, G_LON0, G_LAT0, G_D, G_NX, G_NY,
                    row_lat, col_lon, cell_area_rows, cell_dx_rows, cell_dy_km,
                    downstream_index, topo_order, label_upstream, trace_down)

# ----------------------------------------------------------------------------- parameters
A_CHANNEL = 0.10      # km2: cells with less accumulated area are "hillslope"
V_HILL = 0.40         # m/s on hillslope cells
V_MIN, V_MAX = 1.0, 4.0   # m/s bounds for channel cells (flood calibration)
B_AREA = 0.20         # exponent of accumulated area (km2)
K_FLOOD = 15.0        # v = K * S^0.5 * A^0.2 ; calibrated on the 29-Oct-2024 Poyo wave (see README)
K_SLOW = 5.0           # Temez-calibrated variant (median longest travel time = Temez tc)
V_MIN_SLOW = 0.5
SLOPE_REACH = 12      # cells (~1.1-1.5 km) over which the channel slope is measured
EQUIV_MM = 20.0       # a dam "cuts" (scope 1) if capacity / own unregulated catchment >= 20 mm
os.makedirs(OUT, exist_ok=True)
CALIB = len(sys.argv) > 1 and sys.argv[1] == "calib"


# ----------------------------------------------------------------------------- numba kernels
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
    """zc[i] = max(z[i], zc[downstream]) : non-increasing in the flow direction"""
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
def upstream_fields(ds, order, outlet, steplen, v1, v2):
    """mask of the catchment of `outlet`, distance (km) and two travel times (h) to it"""
    n = ds.size
    m = np.zeros(n, dtype=np.bool_)
    dist = np.zeros(n, dtype=np.float32)
    t1 = np.zeros(n, dtype=np.float32)
    t2 = np.zeros(n, dtype=np.float32)
    m[outlet] = True
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        j = ds[i]
        if j >= 0 and m[j]:
            m[i] = True
            dist[i] = dist[j] + steplen[i]
            t1[i] = t1[j] + steplen[i] / v1[i] / 3.6
            t2[i] = t2[j] + steplen[i] / v2[i] / 3.6
    return m, dist, t1, t2


def temez(L_km, J):
    return 0.3 * (L_km / max(J, 1e-5) ** 0.25) ** 0.76


# ----------------------------------------------------------------------------- load
t0 = time.time()
d = np.load(os.path.join(SCR, "dir_D.npy"))
acc = np.load(os.path.join(SCR, "acc_D.npy")).ravel()
dem = np.nan_to_num(np.load(os.path.join(SCR, "dem_cop30_3s.npy")), nan=0.0).ravel()
burn = np.load(os.path.join(SCR, "burn_D.npy")).ravel()
ds = downstream_index(d)
order = topo_order(ds)
area = np.repeat(cell_area_rows()[:, None], NCOL, axis=1).ravel().astype(np.float32)
area[d.ravel() == 255] = 0
steplen = step_lengths(ds, NCOL, cell_dx_rows(), cell_dy_km())
zc = monotone_profile(ds, order, dem)
S = reach_slope(ds, zc, steplen, SLOPE_REACH)
del zc


def velocity(k, vmin):
    v = k * np.sqrt(np.maximum(S, 0.0)) * np.power(np.maximum(acc, 1e-3), B_AREA)
    v = np.clip(v, vmin, V_MAX).astype(np.float32)
    v[acc < A_CHANNEL] = V_HILL
    return v


# Riua grid index of every W cell (-1 outside the box)
lat_r = row_lat(np.arange(NROW))
lon_c = col_lon(np.arange(NCOL))
gj = np.floor((lat_r - G_LAT0) / G_D).astype(np.int32)
gi = np.floor((lon_c - G_LON0) / G_D).astype(np.int32)
gj[(gj < 0) | (gj >= G_NY)] = -1
gi[(gi < 0) | (gi >= G_NX)] = -1
gidx = np.where((gj[:, None] >= 0) & (gi[None, :] >= 0), gj[:, None] * G_NX + gi[None, :], -1).astype(np.int32).ravel()

wp = json.load(open(os.path.join(SCR, "work_points.json"), encoding="utf-8"))
P, DAMS = wp["points"], wp["dams"]
print("loaded", round(time.time() - t0, 1), "s")

# ----------------------------------------------------------------------------- calibration mode
if CALIB:
    names = {p["id"]: p for p in P}
    for k in (8.0, 10.0, 11.0, 12.0, 14.0):
        v1 = velocity(k, V_MIN)
        v2 = velocity(K_SLOW, V_MIN_SLOW)
        pa = names["poyo-paiporta"]
        m, dist, t1, t2 = upstream_fields(ds, order, pa["cell"], steplen, v1, v2)
        print(f"k={k}: to Paiporta from Chiva {t1[names['poyo-chiva']['cell']]:.2f} h, Cheste {t1[names['poyo-cheste']['cell']]:.2f} h, "
              f"A-3 gauge {t1[names['poyo-ribarroja']['cell']]:.2f} h; tmax {t1[m].max():.2f} h ; slow: Chiva {t2[names['poyo-chiva']['cell']]:.1f} h tmax {t2[m].max():.1f}")
    for ks, vm in ((1.5, 0.25), (2.0, 0.25), (2.4, 0.25), (3.0, 0.25), (3.0, 0.5), (4.0, 0.5)):
        v2 = velocity(ks, vm)
        v1 = velocity(K_FLOOD, V_MIN)
        ratios = []
        ratios_f = []
        for p in P:
            if p["area_km2"] > 3000:
                continue
            m, dist, t1, t2 = upstream_fields(ds, order, p["cell"], steplen, v1, v2)
            head = int(np.argmax(dist))
            L = float(dist[head])
            J = max((dem[head] - dem[p["cell"]]) / (L * 1000.0), 1e-4)
            tc = temez(L, J)
            ratios.append(float(t2[m].max()) / tc)
            ratios_f.append(float(t1[m].max()) / tc)
        print(f"slow k={ks} vmin={vm}: tmax/tc median {np.median(ratios):.2f} p10 {np.percentile(ratios, 10):.2f} p90 {np.percentile(ratios, 90):.2f}"
              f" | flood k={K_FLOOD}: median {np.median(ratios_f):.2f}")
    sys.exit(0)

v1 = velocity(K_FLOOD, V_MIN)
v2 = velocity(K_SLOW, V_MIN_SLOW)

# ----------------------------------------------------------------------------- dams: which ones cut
DAMS.sort(key=lambda q: q["area_km2"])
# first1[x] / first2[x] = index (in DAMS, sorted upstream -> downstream by area) of the first
# cutting dam met when going downstream from cell x (-1 = none). A cell is regulated with
# respect to a control point iff that first dam lies inside the point's catchment.
first1 = np.full(ds.size, -1, dtype=np.int8)   # scope 1: dams holding >= EQUIV_MM
first2 = np.full(ds.size, -1, dtype=np.int8)   # scope 2: every inventoried dam (except no_regulation)
dam_cells = np.array([q["cell"] for q in DAMS])
for qi, q in enumerate(DAMS):
    m = label_upstream(ds, order, q["cell"])
    own = float(area[m & (first1 < 0)].sum())
    q["own_unregulated_km2"] = own
    q["equiv_mm"] = float(q["capacity_hm3"] / own * 1000.0) if own > 0 else 0.0
    q["cuts"] = bool(q["equiv_mm"] >= EQUIV_MM and not q.get("no_regulation"))
    q["upstream_dams"] = [DAMS[k]["id"] for k in np.nonzero(m[dam_cells])[0] if DAMS[k]["id"] != q["id"]]
    if q["cuts"]:
        first1[m & (first1 < 0)] = qi
    if not q.get("no_regulation"):
        first2[m & (first2 < 0)] = qi
    print(f"dam {q['id']:16s} A={q['area_km2']:8.1f} own={own:8.1f} cap={q['capacity_hm3']:6.1f} equiv={q['equiv_mm']:6.1f} mm cuts={q['cuts']}")
dam_by_id = {q["id"]: q for q in DAMS}

# ----------------------------------------------------------------------------- per point
inbox = gidx >= 0
out_cells = {p["cell"]: p["id"] for p in P}
ta = {k: [] for k in ("point_idx", "scope", "cell", "lag_h", "area_km2")}
ta_slow = {k: [] for k in ("point_idx", "scope", "cell", "lag_h", "area_km2")}
polys = []
lines = []
checks = []
tr_full = from_origin(W_LON0, W_LAT1, RES, RES)


def add_ta(store, pi, scope, sel, tt):
    g = gidx[sel]
    ok = g >= 0
    lag = np.floor(tt[sel][ok]).astype(np.int64)
    key = g[ok].astype(np.int64) * 4096 + lag
    w = np.bincount(key, weights=area[sel][ok].astype(np.float64))
    nz = np.nonzero(w)[0]
    store["point_idx"].append(np.full(nz.size, pi, dtype=np.int16))
    store["scope"].append(np.full(nz.size, scope, dtype=np.int8))
    store["cell"].append((nz // 4096).astype(np.int16))
    store["lag_h"].append((nz % 4096).astype(np.int16))
    store["area_km2"].append(w[nz].astype(np.float32))
    return float(w.sum())


def polygon_of(mask_flat):
    m2 = mask_flat.reshape(NROW, NCOL)
    rr = np.nonzero(m2.any(axis=1))[0]
    cc = np.nonzero(m2.any(axis=0))[0]
    r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
    sub = m2[r0:r1, c0:c1].astype(np.uint8)
    tr = from_origin(W_LON0 + c0 * RES, W_LAT1 - r0 * RES, RES, RES)
    geoms = [shape(g) for g, v in features.shapes(sub, mask=sub.astype(bool), transform=tr, connectivity=8)]
    return unary_union(geoms)


def path_line(cells):
    r = cells // NCOL
    c = cells % NCOL
    return np.column_stack([col_lon(c), row_lat(r)])


for pi, p in enumerate(P):
    t1s = time.time()
    m, dist, t1, t2 = upstream_fields(ds, order, p["cell"], steplen, v1, v2)
    a_nat = float(area[m].sum())
    dam_in = np.append(m[dam_cells], False)          # last slot = "no dam" (index -1)
    u1 = m & ~dam_in[first1]
    u2 = m & ~dam_in[first2]
    a_u1 = float(area[u1].sum())
    a_u2 = float(area[u2].sum())
    a_in = float(area[m & inbox].sum())
    # main channel = longest flow path
    head = int(np.argmax(np.where(m, dist, -1)))
    L = float(dist[head])
    z_out = float(dem[p["cell"]])
    z_head = float(dem[head])
    J = max((z_head - z_out) / (L * 1000.0), 1e-4)
    tc = temez(L, J)
    head_u = int(np.argmax(np.where(u1, dist, -1)))
    L_u = float(dist[head_u])
    J_u = max((float(dem[head_u]) - z_out) / (max(L_u, 0.05) * 1000.0), 1e-4)
    tc_u = temez(L_u, J_u)
    zs = dem[m]
    # dams inside the natural catchment
    dl = []
    inside = [q for q in DAMS if m[q["cell"]]]
    for q in inside:
        nearest = not any((q["id"] in o["upstream_dams"]) and o["cuts"] for o in inside if o["id"] != q["id"])
        dl.append(dict(id=q["id"], name=q["name"], river=q["river"], capacity_hm3=q["capacity_hm3"],
                       catchment_km2=round(q["area_km2"], 1), equiv_mm=round(q["equiv_mm"], 1),
                       cuts=q["cuts"], nearest=bool(nearest and q["cuts"]),
                       lat=round(q["lat_snap"], 4), lon=round(q["lon_snap"], 4),
                       travel_h=round(float(t1[q["cell"]]), 1),
                       **({"note": q["cap_note"]} if "cap_note" in q else {})))
    dl.sort(key=lambda x: -x["catchment_km2"])
    # next control point downstream
    path = trace_down(ds, p["cell"], 40000)
    nxt = None
    for c in path[1:]:
        if int(c) in out_cells:
            nxt = out_cells[int(c)]
            break
    p.update(area_nat=a_nat, area_u1=a_u1, area_u2=a_u2, area_in=a_in, L=L, J=J, tc=tc, L_u=L_u, J_u=J_u, tc_u=tc_u,
             z_out=z_out, z_max=float(zs.max()), z_mean=float(np.average(zs, weights=np.maximum(area[m], 1e-9))),
             z_head=z_head, dams_out=dl, next_down=nxt, tmax=float(t1[m].max()), tmax_slow=float(t2[m].max()),
             tmax_u1=float(t1[u1].max()), head=head)
    # time-area tables
    for scope, sel in ((0, m), (1, u1), (2, u2)):
        s_in = add_ta(ta, pi, scope, sel, t1)
        add_ta(ta_slow, pi, scope, sel, t2)
        tot_in = float(area[sel & inbox].sum())
        assert abs(s_in - tot_in) < 1e-3 * max(1.0, tot_in), (p["id"], scope, s_in, tot_in)
    # polygons
    polys.append((p["id"], "natural", a_nat, polygon_of(m)))
    if a_u1 < 0.995 * a_nat:
        polys.append((p["id"], "unregulated", a_u1, polygon_of(u1)))
    # main channel line: head -> outlet -> sea / confluence / end of mapped channel
    up = trace_down(ds, head, 60000)
    k_out = int(np.nonzero(up == p["cell"])[0][0])
    end = k_out
    gap = 0
    last_burn = k_out
    for k in range(k_out + 1, up.size):
        c = int(up[k])
        if acc[c] > 3.0 * acc[int(up[k - 1])] + 5.0:
            end = k
            last_burn = k
            break
        if burn[c] > 0:
            last_burn = k
            gap = 0
        else:
            gap += 1
            if gap > 15:
                break
        end = k
    end = min(end, last_burn)
    lines.append((p["id"], path_line(up[:end + 1]), k_out))
    p["cells_to_end"] = int(end - k_out)
    print(f"{pi:2d} {p['id']:24s} A={a_nat:9.1f} unreg={a_u1:9.1f} strict={a_u2:9.1f} L={L:6.1f} J={J:.4f} tc={tc:5.1f} tmax={p['tmax']:5.1f} slow={p['tmax_slow']:6.1f}  {time.time() - t1s:.1f}s")
    # travel checks between consecutive points (stored on the downstream one)
    if p["id"] == "poyo-paiporta":
        ids = {q["id"]: q for q in P}
        for src in ("poyo-chiva", "poyo-cheste", "poyo-ribarroja", "horteta-torrent", "poyo-torrent"):
            checks.append(f"{src} -> poyo-paiporta: {dist[ids[src]['cell']]:.1f} km, flood law {t1[ids[src]['cell']]:.2f} h "
                          f"({dist[ids[src]['cell']] / t1[ids[src]['cell']] / 3.6:.2f} m/s), slow law {t2[ids[src]['cell']]:.2f} h")

with open(os.path.join(SCR, "travel_checks.txt"), "w") as f:
    f.write("\n".join(checks) + "\n")
print("\n".join(checks))

# upstream links
ids = [p["id"] for p in P]
ups = {i: [] for i in ids}
for p in P:
    if p["next_down"]:
        ups[p["next_down"]].append(p["id"])


def stream_key(s):
    s = s.lower()
    for a, b in (("rambla del poyo", "poyo"), ("barranco de chiva", "poyo"), ("río magro", "magro"), ("río júcar", "jucar"),
                 ("río turia", "turia"), ("carraixet", "carraixet"), ("serpis", "serpis"), ("vinalopó", "vinalopo")):
        if a in s:
            return b
    return s


# ----------------------------------------------------------------------------- control_points.json
MANUAL = json.load(open(os.path.join(os.path.dirname(__file__), "points_notes.json"), encoding="utf-8"))
cps = []
for p in P:
    lo, hi = p.get("cedex_lo"), p.get("cedex_hi")
    conf = "high"
    reasons = []
    if lo is not None and p["id"] not in MANUAL.get("skip_cedex", []):
        a = p["area_nat"]
        if a < 0.9 * lo or a > 1.1 * hi + 2:
            conf = "low"
            reasons.append(f"area {a:.1f} km2 outside the official CEDEX bracket {lo:.1f}-{hi:.1f}")
        elif a < 0.95 * lo or a > 1.05 * hi + 1:
            conf = "medium"
            reasons.append(f"area {a:.1f} km2 vs official CEDEX bracket {lo:.1f}-{hi:.1f} (5-10 % off)")
    mn = MANUAL["points"].get(p["id"], {})
    if "confidence" in mn:
        order_c = {"high": 0, "medium": 1, "low": 2}
        if order_c[mn["confidence"]] > order_c[conf]:
            conf = mn["confidence"]
    same_up = [u for u in ups[p["id"]] if stream_key(next(q for q in P if q["id"] == u)["stream"]) == stream_key(p["stream"])]
    notes = []
    if p.get("seed") is None:
        notes.append("Added by h1-catchments: " + p.get("why", ""))
    elif p.get("moved_km") is not None and p["moved_km"] > 1.5:
        notes.append(f"Moved {p['moved_km']:.1f} km from the seed coordinate ({p['seed_lat']}, {p['seed_lon']}).")
    if mn.get("note"):
        notes.append(mn["note"])
    notes += reasons
    cps.append(dict(
        id=p["id"], stream=p["stream"], town=p["town"], lat=round(p["lat"], 5), lon=round(p["lon"], 5),
        area_km2=round(p["area_nat"], 1), area_unregulated_km2=round(p["area_u1"], 1),
        area_unregulated_strict_km2=round(p["area_u2"], 1),
        inside_box_fraction=round(p["area_in"] / p["area_nat"], 4),
        L_km=round(p["L"], 1), slope=round(p["J"], 5), tc_h=round(p["tc"], 2),
        L_unregulated_km=round(p["L_u"], 1), slope_unregulated=round(p["J_u"], 5), tc_unregulated_h=round(p["tc_u"], 2),
        elev_outlet_m=round(p["z_out"], 0), elev_max_m=round(p["z_max"], 0), elev_mean_m=round(p["z_mean"], 0),
        elev_head_m=round(p["z_head"], 0),
        t_longest_h=round(p["tmax"], 1), t_longest_unregulated_h=round(p["tmax_u1"], 1), t_longest_slow_h=round(p["tmax_slow"], 1),
        dams=p["dams_out"],
        downstream_of=(sorted(same_up, key=lambda u: -next(q for q in P if q["id"] == u)["area_nat"])[0] if same_up else None),
        upstream_points=ups[p["id"]], next_downstream=p["next_down"],
        ref_area_km2=p.get("ref_km2"), ref_source=p.get("ref_src"),
        cedex_bracket_km2=[round(lo, 1), round(hi, 1)] if (lo is not None and p["id"] not in MANUAL.get("skip_cedex", [])) else None,
        confidence=conf, seed_id=p.get("seed"), notes=" ".join(notes).strip()))
meta = dict(
    generated="2026-10-01", agent="h1-catchments",
    grid=dict(lon0=G_LON0, lat0=G_LAT0, d=G_D, nx=G_NX, ny=G_NY, flat_index="j*64+i"),
    flow_network="Copernicus DEM GLO-30 averaged to 3 arc-seconds + OSM waterway burn-in + CEDEX closed basins as sinks + priority-flood D8",
    velocity_law=dict(hillslope_ms=V_HILL, channel_threshold_km2=A_CHANNEL, formula="v = k * S^0.5 * A_km2^0.2",
                      k_flood=K_FLOOD, bounds_flood_ms=[V_MIN, V_MAX], k_slow=K_SLOW, bounds_slow_ms=[V_MIN_SLOW, V_MAX],
                      slope_reach_cells=SLOPE_REACH),
    scopes={"0": "natural", "1": f"unregulated: downstream of dams whose capacity is >= {EQUIV_MM:.0f} mm over their own catchment",
            "2": "unregulated_strict: downstream of every inventoried dam except Isbert"},
    downstream_of="id of the nearest control point UPSTREAM on the same stream (this point is downstream of it); see also upstream_points / next_downstream",
    dams_inventory=[dict(id=q["id"], name=q["name"], river=q["river"], lat=round(q["lat_snap"], 4), lon=round(q["lon_snap"], 4),
                         capacity_hm3=q["capacity_hm3"], catchment_km2=round(q["area_km2"], 1),
                         own_unregulated_km2=round(q["own_unregulated_km2"], 1), equiv_mm=round(q["equiv_mm"], 1), cuts=q["cuts"])
                    for q in sorted(DAMS, key=lambda q: q["id"])])
with open(os.path.join(OUT, "control_points.json"), "w", encoding="utf-8") as f:
    json.dump(dict(meta=meta, points=cps), f, ensure_ascii=False, indent=1)

# ----------------------------------------------------------------------------- time_area.npz
arrs = {k: np.concatenate(v) for k, v in ta.items()}
arrs_s = {"slow_" + k: np.concatenate(v) for k, v in ta_slow.items()}
np.savez_compressed(os.path.join(OUT, "time_area.npz"), point_ids=np.array(ids), **arrs, **arrs_s)
print("time_area entries", arrs["cell"].size, "slow", arrs_s["slow_cell"].size, "max lag", arrs["lag_h"].max(), arrs_s["slow_lag_h"].max())


# ----------------------------------------------------------------------------- geojson
def clean_poly(g, tol, min_hole_km2=4.0):
    g = g.simplify(tol, preserve_topology=True)
    parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    parts = [q for q in parts if q.area > 0]
    big = max(q.area for q in parts)
    outp = []
    for q in parts:
        if q.area < 0.02 * big:
            continue
        holes = [h for h in q.interiors if Polygon(h).area * 111.2 * 86.0 > min_hole_km2]
        outp.append(Polygon(q.exterior, holes))
    return outp[0] if len(outp) == 1 else MultiPolygon(outp)


def rnd(obj, nd=4):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(obj[0], nd), round(obj[1], nd)]
        return [rnd(o, nd) for o in obj]
    return obj


def dedupe(coords):
    out = [coords[0]]
    for c in coords[1:]:
        if c != out[-1]:
            out.append(c)
    return out


scale = 1.0
while True:
    feats = []
    for pid, scope, a, g in polys:
        tol = float(np.clip(0.0012 * np.sqrt(a / 40.0), 0.0010, 0.012)) * scale
        gg = clean_poly(g, tol)
        gm = mapping(gg)
        gm = dict(type=gm["type"], coordinates=rnd(gm["coordinates"]))
        feats.append(dict(type="Feature", properties=dict(id=pid, scope=scope, area_km2=round(a, 1)), geometry=gm))
    txt = json.dumps(dict(type="FeatureCollection", features=feats), separators=(",", ":"))
    if len(txt.encode()) < 390_000:
        break
    scale *= 1.2
with open(os.path.join(OUT, "catchments.geojson"), "w", encoding="utf-8") as f:
    f.write(txt)
print("catchments.geojson", len(txt), "bytes, scale", round(scale, 2))

scale = 1.0
pmap = {p["id"]: p for p in P}
while True:
    feats = []
    for pid, xy, k_out in lines:
        ln = LineString(xy).simplify(0.0008 * scale, preserve_topology=False)
        co = dedupe(rnd([list(c) for c in ln.coords]))
        feats.append(dict(type="Feature", properties=dict(id=pid, name=pmap[pid]["stream"], length_km=round(pmap[pid]["L"], 1)),
                          geometry=dict(type="LineString", coordinates=co)))
    txt = json.dumps(dict(type="FeatureCollection", features=feats), ensure_ascii=False, separators=(",", ":"))
    if len(txt.encode()) < 290_000:
        break
    scale *= 1.2
with open(os.path.join(OUT, "streams.geojson"), "w", encoding="utf-8") as f:
    f.write(txt)
print("streams.geojson", len(txt.encode()), "bytes, scale", round(scale, 2))

# ----------------------------------------------------------------------------- table + overview
rows = [f"{'id':24s} {'area':>9s} {'ref':>8s} {'CEDEX lo..hi':>19s} {'unreg':>9s} {'strict':>9s} {'inbox':>6s} {'L_km':>6s} {'J':>7s} {'tc_h':>5s} {'tmax':>5s} {'slow':>6s} conf"]
for c, p in zip(cps, P):
    br = f"{c['cedex_bracket_km2'][0]:9.1f}..{c['cedex_bracket_km2'][1]:8.1f}" if c["cedex_bracket_km2"] else f"{'-':>19s}"
    ref = f"{c['ref_area_km2']:8.1f}" if c["ref_area_km2"] else f"{'-':>8s}"
    rows.append(f"{c['id']:24s} {c['area_km2']:9.1f} {ref} {br} {c['area_unregulated_km2']:9.1f} {c['area_unregulated_strict_km2']:9.1f} "
                f"{c['inside_box_fraction']:6.3f} {c['L_km']:6.1f} {c['slope']:7.4f} {c['tc_h']:5.1f} {c['t_longest_h']:5.1f} {c['t_longest_slow_h']:6.1f} {c['confidence']}")
with open(os.path.join(SCR, "final_table.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(rows))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 14))
step = 6
dem2 = np.load(os.path.join(SCR, "dem_cop30_3s.npy"), mmap_mode="r")[::step, ::step]
ax.imshow(dem2, extent=(W_LON0, W_LON0 + NCOL * RES, W_LAT1 - NROW * RES, W_LAT1), cmap="terrain", vmin=-300, vmax=2200, alpha=0.5, aspect="auto")
cm = plt.get_cmap("tab20")
for k, (pid, scope, a, g) in enumerate(sorted(polys, key=lambda x: -x[2])):
    gg = g.simplify(0.004)
    for q in (gg.geoms if hasattr(gg, "geoms") else [gg]):
        x, y = q.exterior.xy
        if scope == "natural":
            ax.fill(x, y, color=cm(k % 20), alpha=0.18)
            ax.plot(x, y, color=cm(k % 20), lw=0.8)
        else:
            ax.plot(x, y, color="k", lw=0.5, ls=":")
for pid, xy, k_out in lines:
    ax.plot(xy[:, 0], xy[:, 1], color="navy", lw=0.7)
for q in DAMS:
    ax.plot(q["lon_snap"], q["lat_snap"], "^", color="red" if q["cuts"] else "orange", ms=5, mec="k", mew=0.3)
for c in cps:
    ax.plot(c["lon"], c["lat"], "o", color="yellow", mec="k", ms=4)
    ax.annotate(c["id"], (c["lon"], c["lat"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
ax.plot([G_LON0, G_LON0 + G_NX * G_D, G_LON0 + G_NX * G_D, G_LON0, G_LON0], [G_LAT0, G_LAT0, G_LAT0 + G_NY * G_D, G_LAT0 + G_NY * G_D, G_LAT0], "k--", lw=1)
ax.set_xlim(-3.2, 0.9)
ax.set_ylim(37.3, 41.1)
ax.set_title("Riua control points: natural catchments (colour), unregulated scope (dotted), main channels, dams (red = cuts, orange = minor), Riua box (dashed)")
plt.tight_layout()
plt.savefig(os.path.join(SCR, "overview.png"), dpi=110)
print("done", round(time.time() - t0, 1), "s")
