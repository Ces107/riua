"""Load the cached Overpass JSON (see 02_fetch_osm.py)."""
import glob
import json
import os
import numpy as np
from common import SCR, W_LON0, W_LAT1, RES, NROW, NCOL


def load_ways():
    """list of dicts: id, name, kind, tunnel(bool), xy (N,2 lon/lat array)"""
    seen = set()
    out = []
    for f in sorted(glob.glob(os.path.join(SCR, "osm", "waterways_*.json"))):
        with open(f, encoding="utf-8") as fh:
            js = json.load(fh)
        for el in js["elements"]:
            if el["type"] != "way" or el["id"] in seen or "geometry" not in el:
                continue
            seen.add(el["id"])
            t = el.get("tags", {})
            xy = np.array([(p["lon"], p["lat"]) for p in el["geometry"]], dtype=float)
            out.append(dict(id=el["id"], name=t.get("name", ""), kind=t.get("waterway", ""),
                            tunnel=t.get("tunnel", "") not in ("", "no"), xy=xy,
                            intermittent=t.get("intermittent", ""), tags=t))
    return out


def densify(xy, step=RES / 2):
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        pts.append(a + (b - a) * t)
    return np.vstack(pts)


def cells_of(xy):
    """unique (row, col) cells crossed by the polyline, in order"""
    p = densify(xy)
    r = np.floor((W_LAT1 - p[:, 1]) / RES).astype(int)
    c = np.floor((p[:, 0] - W_LON0) / RES).astype(int)
    ok = (r >= 0) & (r < NROW) & (c >= 0) & (c < NCOL)
    r, c = r[ok], c[ok]
    if r.size == 0:
        return r, c
    keep = np.ones(r.size, bool)
    keep[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    return r[keep], c[keep]
