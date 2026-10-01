"""Official reference: CEDEX/DGA 1:25,000 Pfafstetter subcatchments of the Jucar
demarcation (CHJ layer F851, downloaded by r5-geo into scratch/r5-geo/chj).

official_upstream(lat, lon) -> dict with the official area draining to the
subcatchment that contains the point:
   lo  = sum of subcatchments strictly upstream of the containing polygon
   hi  = lo + area of the containing polygon
(the true official value at the point lies between lo and hi), closed (endorheic,
digit 0) subcatchments excluded, plus the raster ids for a spatial comparison.

Pfafstetter rule: with p the common prefix of codes a and b, a is upstream of b iff
a[len(p)] > b[len(p)] and every digit of b after the prefix is odd (b lies on the
main stem at every further level) and a has no 0 after the prefix (not a closed basin).
"""
import glob
import os
import numpy as np
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import Point
from common import SCR, ROOT, NROW, NCOL, W_LON0, W_LAT1, RES

_cache = {}


def _load():
    if _cache:
        return _cache
    shp = glob.glob(os.path.join(ROOT, "scratch", "r5-geo", "chj", "F851*", "*.shp"))[0]
    sub = gpd.read_file(shp).to_crs(4326)
    sub["PFAFCUEN"] = sub["PFAFCUEN"].astype(str)
    sub = sub.reset_index(drop=True)
    f = os.path.join(SCR, "cedex_sub_idx.npy")
    if os.path.exists(f):
        idx = np.load(f)
    else:
        idx = features.rasterize(((g, i + 1) for i, g in enumerate(sub.geometry)), out_shape=(NROW, NCOL),
                                 transform=from_origin(W_LON0, W_LAT1, RES, RES), fill=0, dtype="int32")
        np.save(f, idx)
    _cache.update(sub=sub, idx=idx, codes=sub.PFAFCUEN.values, areas=sub.CuencaKm2.values, sindex=sub.sindex)
    return _cache


def is_upstream(a, b):
    if a == b:
        return False
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    if k == n:
        # one is a prefix of the other: a inside basin b
        return len(a) > len(b) and "0" not in a[k:]
    if not (a[k] > b[k]):
        return False
    if any(int(ch) % 2 == 0 for ch in b[k:]):
        return False
    return "0" not in a[k:]


def official_upstream(lat, lon):
    c = _load()
    r = int((W_LAT1 - lat) / RES)
    cc = int((lon - W_LON0) / RES)
    i = int(c["idx"][r, cc]) - 1
    if i < 0:
        return None
    b = c["codes"][i]
    ups = np.array([is_upstream(a, b) for a in c["codes"]])
    lo = float(c["areas"][ups].sum())
    return dict(code=b, lo=lo, hi=lo + float(c["areas"][i]), own=i + 1, ids=np.nonzero(ups)[0] + 1)


def raster_idx():
    return _load()["idx"]
