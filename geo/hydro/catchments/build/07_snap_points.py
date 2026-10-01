"""Snap control points and dams onto the product-D flow network and compare every
catchment with the official CEDEX reference. Writes scratch/h1-catchments/work_points.json
and prints the area table (also saved as area_table.txt).
"""
import json
import os
import numpy as np
from common import (SCR, ROOT, NROW, NCOL, rc_of, row_lat, col_lon, cell_area_rows,
                    downstream_index, topo_order, label_upstream, trace_down)
from cedexref import official_upstream, raster_idx

B = os.path.join(ROOT, "geo", "hydro", "catchments", "build")
d = np.load(os.path.join(SCR, "dir_D.npy"))
acc2 = np.load(os.path.join(SCR, "acc_D.npy"))
acc = acc2.ravel()
dem = np.load(os.path.join(SCR, "dem_cop30_3s.npy"))
ds = downstream_index(d)
order = topo_order(ds)
area = np.repeat(cell_area_rows()[:, None], NCOL, axis=1).ravel()
sub_idx = raster_idx().ravel()
seed = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "geo", "hydro", "control_points_seed.json"), encoding="utf-8"))["points"]}
pdef = json.load(open(os.path.join(B, "points_def.json"), encoding="utf-8"))["points"]
ddef = json.load(open(os.path.join(B, "dams_def.json"), encoding="utf-8"))["dams"]


def snap(lat, lon, r=3, amax=None, amin=None):
    r0, c0 = rc_of(lat, lon)
    best = None
    for rr in range(max(0, r0 - r), min(NROW, r0 + r + 1)):
        for cc in range(max(0, c0 - r), min(NCOL, c0 + r + 1)):
            a = acc2[rr, cc]
            if amax is not None and a > amax:
                continue
            if amin is not None and a < amin:
                continue
            if best is None or a > best[0]:
                best = (float(a), rr, cc)
    return best


def km(lat1, lon1, lat2, lon2):
    return float(np.hypot((lat1 - lat2) * 111.2, (lon1 - lon2) * 111.2 * np.cos(np.deg2rad(lat1))))


pts = []
lines = []
hdr = f"{'id':24s} {'lat':>8s} {'lon':>8s} {'moved_km':>8s} {'area':>9s} {'ref':>8s} {'CEDEX lo..hi':>19s} {'in-not-off':>10s} {'off-not-in':>10s}  flag"
lines.append(hdr)
for p in pdef:
    s = snap(p["lat"], p["lon"], p.get("r", 3), p.get("amax"), p.get("amin"))
    if s is None:
        raise SystemExit("no snap for " + p["id"])
    a, r, c = s
    lat, lon = float(row_lat(r)), float(col_lon(c))
    out = r * NCOL + c
    m = label_upstream(ds, order, out)
    a_chk = float(area[m].sum())
    sd = seed.get(p.get("seed") or "", None)
    moved = km(lat, lon, sd["lat"], sd["lon"]) if sd else None
    off = official_upstream(lat, lon)
    rec = dict(p)
    rec.update(lat=lat, lon=lon, cell=int(out), area_km2=a_chk, moved_km=moved,
               seed_lat=sd["lat"] if sd else None, seed_lon=sd["lon"] if sd else None)
    flag = ""
    if off is not None:
        ids = np.zeros(sub_idx.max() + 2, dtype=bool)
        ids[off["ids"]] = True
        offm = ids[sub_idx]
        own = sub_idx == off["own"]
        a_in_not_off = float(area[m & ~offm & ~own & (sub_idx > 0)].sum())
        a_off_not_in = float(area[offm & ~m].sum())
        rec.update(cedex_code=off["code"], cedex_lo=off["lo"], cedex_hi=off["hi"],
                   extra_vs_cedex_km2=a_in_not_off, missing_vs_cedex_km2=a_off_not_in)
        if a_chk < 0.9 * off["lo"] or a_chk > 1.1 * off["hi"] + 2:
            flag = "CHECK"
        cs = f"{off['lo']:9.1f}..{off['hi']:8.1f}"
        xs = f"{a_in_not_off:10.1f} {a_off_not_in:10.1f}"
    else:
        cs = f"{'(outside CHJ)':>19s}"
        xs = f"{'':10s} {'':10s}"
    lines.append(f"{p['id']:24s} {lat:8.4f} {lon:8.4f} {moved if moved is not None else float('nan'):8.2f} {a_chk:9.1f} "
                 f"{p.get('ref_km2', float('nan')):8.1f} {cs} {xs}  {flag}")
    pts.append(rec)

# ---- dams ------------------------------------------------------------------
dams = []
lines.append("")
lines.append(f"{'dam':16s} {'lat':>8s} {'lon':>8s} {'area':>9s} {'cap_hm3':>8s}")
for q in ddef:
    a, r, c = snap(q["lat"], q["lon"], 6 if q["kind"] == "dam" else 15)
    if q["kind"] == "res":
        z0 = float(dem[r, c])
        path = trace_down(ds, r * NCOL + c, 400)
        last = path[0]
        for i in path:
            rr, cc = divmod(int(i), NCOL)
            if abs(float(dem[rr, cc]) - z0) > 1.5:
                break
            last = i
        r, c = divmod(int(last), NCOL)
        a = float(acc2[r, c])
    rec = dict(q)
    rec.update(lat_snap=float(row_lat(r)), lon_snap=float(col_lon(c)), cell=int(r * NCOL + c), area_km2=float(a))
    dams.append(rec)
    lines.append(f"{q['id']:16s} {rec['lat_snap']:8.4f} {rec['lon_snap']:8.4f} {a:9.1f} {q['capacity_hm3']:8.1f}")

with open(os.path.join(SCR, "work_points.json"), "w", encoding="utf-8") as f:
    json.dump(dict(points=pts, dams=dams), f, indent=1, ensure_ascii=False)
txt = "\n".join(lines)
with open(os.path.join(SCR, "area_table.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print(txt)
