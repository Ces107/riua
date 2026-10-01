"""Product D: the flow network actually used.

 elevation  = Copernicus DEM GLO-30 averaged 1" -> 3" on the HydroSHEDS lattice
 burning    = OSM waterways (river + named stream; named drains/ditches only when the
              name says Barranc/Barranco/Rambla/Riu/Rio/Torrent), -BURN m on the cells
              they cross (+ optional extra lines listed in burn_extra.json)
 sinks      = official closed (endorheic) basins of the CEDEX/DGA 1:25,000 Pfafstetter
              subcatchments of the Jucar demarcation (digit 0 in the code) -> one sink at
              the lowest cell of each closed-basin group; outside that layer the inland
              sinks of HydroSHEDS v1 (dir == 0 away from the coast)
 fill + D8  = priority-flood with epsilon, steepest descent

Outputs in scratch/h1-catchments: dir_D.npy (uint8 ESRI codes), acc_D.npy (km2),
burn_D.npy (uint8: 0 none, 1 stream, 2 river), sinks_D.json
"""
import glob
import json
import os
import re
import time
import numpy as np
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from common import (SCR, ROOT, NROW, NCOL, W_LON0, W_LAT1, RES, cell_area_rows, cell_dx_rows,
                    cell_dy_km, downstream_index, topo_order, accumulate, row_lat, col_lon, rc_of)
from osmutil import load_ways, cells_of
from pflood import fill_eps, d8_from_filled

BURN_STREAM = 10.0
BURN_RIVER = 14.0
NAME_OK = re.compile(r"^(barranc|barranco|rambla|riu|r[ií]o|torrent|torrente|arroyo|ca[ñn]ada)\b", re.I)

dem = np.load(os.path.join(SCR, "dem_cop30_3s.npy"))
hs = np.load(os.path.join(SCR, "hs_dir.npz"))["dir"]
valid = np.isfinite(dem) & ~((hs == 255) & (dem <= 0.5))
z0 = np.nan_to_num(dem, nan=0.0).astype(np.float64)

# ---- burn --------------------------------------------------------------
burn = np.zeros((NROW, NCOL), dtype=np.uint8)
ways = load_ways()
nb = 0
skip_ids = set()
extra_path = os.path.join(ROOT, "geo", "hydro", "catchments", "build", "burn_extra.json")
extra = {"skip_way_ids": [], "lines": []}
if os.path.exists(extra_path):
    with open(extra_path, encoding="utf-8") as f:
        extra = json.load(f)
    skip_ids = set(extra.get("skip_way_ids", []))
for w in ways:
    if w["id"] in skip_ids:
        continue
    k = w["kind"]
    if k in ("river", "stream"):
        lvl = 2 if k == "river" else 1
    elif k in ("drain", "ditch", "canal") and NAME_OK.search(w["name"] or ""):
        lvl = 1
    else:
        continue
    r, c = cells_of(w["xy"])
    burn[r, c] = np.maximum(burn[r, c], lvl)
    nb += 1
for ln in extra.get("lines", []):
    r, c = cells_of(np.array(ln["xy"], dtype=float))
    burn[r, c] = np.maximum(burn[r, c], ln.get("level", 2))
print("burned ways", nb, "cells", int((burn > 0).sum()), "extra lines", len(extra.get("lines", [])))
z = z0 - np.where(burn == 2, BURN_RIVER, np.where(burn == 1, BURN_STREAM, 0.0))
np.save(os.path.join(SCR, "burn_D.npy"), burn)

# ---- sinks ---------------------------------------------------------------
sink = np.zeros((NROW, NCOL), dtype=bool)
sinks = []
shp = glob.glob(os.path.join(ROOT, "scratch", "r5-geo", "chj", "F851*", "*.shp"))[0]
sub = gpd.read_file(shp)
sub["PFAFCUEN"] = sub["PFAFCUEN"].astype(str)
closed = sub[sub.PFAFCUEN.str[3:].str.contains("0")].copy()
closed["grp"] = closed.PFAFCUEN.apply(lambda s: s[:3 + s[3:].index("0") + 1])
grp = closed.dissolve("grp", aggfunc={"CuencaKm2": "sum"}).to_crs(4326)
tr = from_origin(W_LON0, W_LAT1, RES, RES)
gid = features.rasterize(((g, i + 1) for i, g in enumerate(grp.geometry)), out_shape=(NROW, NCOL),
                         transform=tr, fill=0, dtype="int32")
chj = features.rasterize(((g, 1) for g in sub.to_crs(4326).geometry), out_shape=(NROW, NCOL),
                         transform=tr, fill=0, dtype="uint8")
np.save(os.path.join(SCR, "chj_mask.npy"), chj)
np.save(os.path.join(SCR, "closed_gid.npy"), gid)
zz = np.where(valid, z, 1e9)
for i, (code, row) in enumerate(grp.iterrows()):
    m = gid == i + 1
    if m.sum() == 0:
        continue
    # connected parts of a group can be several separate depressions: one sink per part > 3 km2
    lab, n = ndi.label(m)
    for j in range(1, n + 1):
        mj = lab == j
        if mj.sum() * 0.0067 < 3.0:
            continue
        idx = np.argmin(np.where(mj, zz, 1e9))
        r, c = divmod(int(idx), NCOL)
        sink[r, c] = True
        sinks.append(dict(src="CEDEX closed basin", code=code, lat=float(row_lat(r)), lon=float(col_lon(c)),
                          part_km2=float(mj.sum() * 0.0067), group_km2=float(row.CuencaKm2)))
# HydroSHEDS inland sinks outside the CHJ layer
ocean_near = ndi.binary_dilation(hs == 255, iterations=3)
hr, hc = np.nonzero((hs == 0) & ~ocean_near & (chj == 0))
for r, c in zip(hr, hc):
    r0, r1, c0, c1 = max(0, r - 12), min(NROW, r + 13), max(0, c - 12), min(NCOL, c + 13)
    win = zz[r0:r1, c0:c1]
    k = np.argmin(win)
    rr, cc = r0 + k // win.shape[1], c0 + k % win.shape[1]
    sink[rr, cc] = True
    sinks.append(dict(src="HydroSHEDS sink", lat=float(row_lat(rr)), lon=float(col_lon(cc))))
print("sinks", len(sinks))
with open(os.path.join(SCR, "sinks_D.json"), "w") as f:
    json.dump(sinks, f, indent=1)

# ---- fill + D8 -------------------------------------------------------------
t = time.time()
zf = fill_eps(z, valid, sink, 1e-4)
d = d8_from_filled(zf, valid, sink, cell_dx_rows(), cell_dy_km())
np.save(os.path.join(SCR, "dir_D.npy"), d)
print("fill+d8", round(time.time() - t, 1), "s")
area = np.repeat(cell_area_rows()[:, None], NCOL, axis=1).ravel()
ds = downstream_index(d)
order = topo_order(ds)
acc = accumulate(ds, order, np.where(d.ravel() == 255, 0.0, area)).reshape(NROW, NCOL).astype(np.float32)
np.save(os.path.join(SCR, "acc_D.npy"), acc)
print("acc done; order", order.size, "of", ds.size)
for s in sinks:
    r, c = rc_of(s["lat"], s["lon"])
    s["acc_km2"] = float(acc[r, c])
with open(os.path.join(SCR, "sinks_D.json"), "w") as f:
    json.dump(sinks, f, indent=1)
for s in sorted(sinks, key=lambda s: -s["acc_km2"])[:40]:
    print(s)
