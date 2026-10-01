"""Step 1: locate each control point on its channel using OSM waterway geometry.

For every point: OSM ways whose name matches config name_re are chained into polylines (UTM),
the anchor is the nearest position on them to the start coordinate (config "anchor" >
h1-catchments snapped point > seed). Writes scratch/h2-sections/misc/anchors.json with the
anchor, the oriented centreline (+-1200 m of chainage around the anchor) and any
covered/tunnel stretches.
Usage: py -3.11 locate.py [id ...]
"""
import json
import os
import re
import sys

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

from common import MISC, OSM_DIR, SNAPPED, jdump, ll2utm, load_config, load_seed, utm2ll


def snapped_points():
    if not os.path.exists(SNAPPED):
        return {}
    try:
        with open(SNAPPED, encoding="utf-8") as f:
            d = json.load(f)
        pts = d["points"] if isinstance(d, dict) and "points" in d else d
        out = {}
        for p in pts:
            lat = p.get("lat_snapped", p.get("snap_lat", p.get("lat")))
            lon = p.get("lon_snapped", p.get("snap_lon", p.get("lon")))
            if lat is not None and lon is not None:
                out[p["id"]] = (lat, lon)
        return out
    except Exception as e:  # noqa
        print("could not read snapped control points:", e)
        return {}


def locate(p, cfg, snapped):
    with open(os.path.join(OSM_DIR, p["id"] + ".json"), encoding="utf-8") as f:
        els = json.load(f)["elements"]
    rx = re.compile(cfg["name_re"], re.I)
    lines, covered = [], []
    for e in els:
        t = e.get("tags", {})
        nm = " | ".join(t.get(k, "") for k in ("name", "name:es", "name:ca", "alt_name", "official_name"))
        if not rx.search(nm):
            continue
        if t.get("waterway") not in ("river", "stream", "canal", "drain", "wadi"):
            continue
        xy = [ll2utm(g["lon"], g["lat"]) for g in e["geometry"]]
        if len(xy) < 2:
            continue
        ls = LineString(xy)
        lines.append(ls)
        if t.get("tunnel") or t.get("covered") == "yes":
            covered.append(ls)
    if not lines:
        return None
    if "anchor" in cfg:
        lat, lon = cfg["anchor"]
        start_src = "manual anchor (config)"
    elif p["id"] in snapped:
        lat, lon = snapped[p["id"]]
        start_src = "h1-catchments snapped point"
    else:
        lat, lon = p["lat"], p["lon"]
        start_src = "seed (town-level)"
    s = Point(*ll2utm(lon, lat))
    merged = linemerge(lines)
    parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    best = min(parts, key=lambda g: g.distance(s))
    ch = best.project(s)
    # keep 700 m of line on both sides (anchor at a confluence / end of the mapped way is useless)
    if best.length > 1500:
        ch = min(max(ch, 700.0), best.length - 700.0)
    a = best.interpolate(ch)
    # centreline +-1200 m around anchor, 5 m spacing
    c0, c1 = max(0.0, ch - 1200), min(best.length, ch + 1200)
    cs = np.arange(c0, c1 + 1e-6, 5.0)
    xy = np.array([best.interpolate(c).coords[0] for c in cs])
    cov = [bool(any(Point(*q).distance(c) < 3 for c in covered)) for q in xy] if covered else [False] * len(xy)
    lonlat = utm2ll(a.x, a.y)
    far = []
    for dc in (-3000, -2000, -1500, 1500, 2000, 3000):
        if 0 <= ch + dc <= best.length:
            q = best.interpolate(ch + dc)
            far.append({"chain": dc, "x": q.x, "y": q.y})
    return {
        "far_points": far,
        "id": p["id"], "start_source": start_src, "start_latlon": [lat, lon],
        "anchor_xy": [a.x, a.y], "anchor_latlon": [lonlat[1], lonlat[0]],
        "dist_start_to_channel_m": float(best.distance(s)),
        "line_len_m": float(best.length), "anchor_chainage_idx": int(np.argmin(np.abs(cs - ch))),
        "chain": (cs - ch).tolist(), "xy": xy.tolist(), "covered": cov,
        "n_named_ways": len(lines),
    }


def main():
    only = set(sys.argv[1:])
    cfg = load_config()
    snapped = snapped_points()
    path = os.path.join(MISC, "anchors.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for p in load_seed():
        if only and p["id"] not in only:
            continue
        if not os.path.exists(os.path.join(OSM_DIR, p["id"] + ".json")):
            print(p["id"], "no OSM file yet")
            continue
        r = locate(p, cfg[p["id"]], snapped)
        if r is None:
            print(p["id"], "NO MATCHING WATERWAY for", cfg[p["id"]]["name_re"])
            continue
        res[p["id"]] = r
        print(f'{p["id"]:24s} {r["start_source"][:6]} dist {r["dist_start_to_channel_m"]:6.0f} m  line {r["line_len_m"]:6.0f} m  '
              f'chain {r["chain"][0]:.0f}..{r["chain"][-1]:.0f}  covered {sum(r["covered"])}  anchor {r["anchor_latlon"][0]:.5f},{r["anchor_latlon"][1]:.5f}')
    jdump(res, path)


if __name__ == "__main__":
    main()
