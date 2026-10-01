"""Keyless 1 m LiDAR DTM windows from the Institut Cartogràfic Valencià (ICV).

Sheets (MTN50, float32 GeoTIFF, EPSG:25830, LZW, 1-row strips) are read by HTTP range
requests through GDAL /vsicurl/, so only the rows of the window are transferred.
Sheet indexes come from the ICV WFS (cached in scratch/h2-sections/misc/hojas_*.json).

    get_window(cx, cy, half) -> (array float32 with NaN, affine transform, source dict)
"""
import json
import os

import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from common import DTM_DIR, MISC, UA

SERIES = {
    "2015PVAL": "ICV MDT LiDAR 1 m, provincia de València, flight 2015",
    "2016PALI": "ICV MDT LiDAR 1 m, provincia d'Alacant, flight 2016",
    "2017PCAS": "ICV MDT LiDAR 1 m, provincia de Castelló, flight 2017",
}
WFS = ("https://terramapas.icv.gva.es/030201_{s}0100_hojas?service=WFS&version=2.0.0&request=GetFeature"
       "&typeNames=ms:distribucion_descargas&outputFormat=application/json; subtype=geojson")

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_USERAGENT", UA)
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "1048576")

_sheets = None


def sheets():
    global _sheets
    if _sheets is not None:
        return _sheets
    out = []
    for s in SERIES:
        path = os.path.join(MISC, f"hojas_{s}.json")
        if not os.path.exists(path):
            r = requests.get(WFS.format(s=s), headers={"User-Agent": UA}, timeout=90)
            r.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for ft in d["features"]:
            ring = np.array(ft["geometry"]["coordinates"][0])
            out.append({"series": s, "hoja": ft["properties"]["hoja"], "url": ft["properties"]["link_tif"],
                        "xmin": ring[:, 0].min(), "xmax": ring[:, 0].max(),
                        "ymin": ring[:, 1].min(), "ymax": ring[:, 1].max()})
    _sheets = out
    return out


def get_window(cx, cy, half, cache_name=None):
    """1 m DTM for [cx-half, cx+half] x [cy-half, cy+half] (integer-metre aligned)."""
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    n = int(2 * half)
    x1, y1 = x0 + n, y0 + n
    cpath = os.path.join(DTM_DIR, cache_name + ".tif") if cache_name else None
    if cpath and os.path.exists(cpath):
        with rasterio.open(cpath) as d:
            if abs(d.transform.c - x0) < 0.5 and abs(d.transform.f - y1) < 0.5 and d.width == n:
                a = d.read(1)
                return a, d.transform, json.loads(d.tags().get("riua_source", "{}"))
    arr = np.full((n, n), np.nan, dtype="float32")
    used = []
    for sh in sheets():
        if sh["xmax"] <= x0 or sh["xmin"] >= x1 or sh["ymax"] <= y0 or sh["ymin"] >= y1:
            continue
        if not np.isnan(arr).any():
            break
        with rasterio.open("/vsicurl/" + sh["url"]) as d:
            bx0, by0 = max(x0, d.bounds.left), max(y0, d.bounds.bottom)
            bx1, by1 = min(x1, d.bounds.right), min(y1, d.bounds.top)
            if bx1 <= bx0 or by1 <= by0:
                continue
            w = from_bounds(bx0, by0, bx1, by1, d.transform).round_offsets().round_lengths()
            a = d.read(1, window=w).astype("float32")
            a[a <= -9000] = np.nan
            c0 = int(round(bx0 - x0))
            r0 = int(round(y1 - by1))
            sub = arr[r0:r0 + a.shape[0], c0:c0 + a.shape[1]]
            m = np.isnan(sub) & ~np.isnan(a[:sub.shape[0], :sub.shape[1]])
            sub[m] = a[:sub.shape[0], :sub.shape[1]][m]
            if m.any():
                used.append({"series": sh["series"], "hoja": sh["hoja"], "url": sh["url"]})
    tr = from_origin(x0, y1, 1.0, 1.0)
    src = {"sheets": used, "resolution_m": 1.0,
           "name": "; ".join(sorted({SERIES[u["series"]] for u in used})) or "NO DATA",
           "licence": "CC BY 4.0, © Institut Cartogràfic Valencià, Generalitat"}
    if cpath:
        with rasterio.open(cpath, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                           crs="EPSG:25830", transform=tr, compress="deflate", predictor=3,
                           tiled=True, nodata=np.nan) as o:
            o.write(arr, 1)
            o.update_tags(riua_source=json.dumps(src))
    return arr, tr, src


if __name__ == "__main__":
    import sys
    import time
    from common import ll2utm
    lon, lat = float(sys.argv[1]), float(sys.argv[2])
    x, y = ll2utm(lon, lat)
    t = time.time()
    a, tr, src = get_window(x, y, 750, cache_name="_test")
    print(a.shape, np.nanmin(a), np.nanmax(a), np.isnan(a).mean(), src["name"], round(time.time() - t, 1), "s")
