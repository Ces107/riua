"""Download OSM waterways around every seed control point (Overpass API, keyless).

ONE combined query for all points (politer than 51 requests; the public Overpass
instances were answering 504/406 intermittently on 2026-10-01, so we retry slowly).
Result is split per point into scratch/h2-sections/osm/<id>.json (cached).
Usage: py -3.11 fetch_osm.py
"""
import json
import math
import os
import time

import requests

from common import OSM_DIR, UA, load_seed

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
RADIUS = 3500  # m


def query(points):
    parts = "\n".join(
        f'  way["waterway"~"^(river|stream|canal|drain|wadi)$"](around:{RADIUS},{p["lat"]},{p["lon"]});'
        for p in points)
    return f"[out:json][timeout:300];\n(\n{parts}\n);\nout tags geom;"


def run(q):
    last = None
    for attempt in range(40):
        ep = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            r = requests.post(ep, data={"data": q}, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=330)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                print("ok from", ep, len(r.content), flush=True)
                return r.json()
            last = f"{ep} -> {r.status_code}"
        except Exception as e:  # noqa
            last = f"{ep} -> {type(e).__name__}"
        print("retry", attempt, last, flush=True)
        time.sleep(20)
    raise RuntimeError(last)


def dist(lat1, lon1, lat2, lon2):
    k = 111320.0
    return math.hypot((lat1 - lat2) * k, (lon1 - lon2) * k * math.cos(math.radians(lat1)))


def main():
    pts = [p for p in load_seed() if not os.path.exists(os.path.join(OSM_DIR, p["id"] + ".json"))]
    if not pts:
        print("all cached")
        return
    # chunks of 13 points keep each response small
    for i in range(0, len(pts), 13):
        chunk = pts[i:i + 13]
        d = run(query(chunk))
        for p in chunk:
            els = [e for e in d["elements"]
                   if any(dist(p["lat"], p["lon"], g["lat"], g["lon"]) < RADIUS * 1.05 for g in e.get("geometry", []))]
            with open(os.path.join(OSM_DIR, p["id"] + ".json"), "w", encoding="utf-8") as f:
                json.dump({"elements": els}, f, ensure_ascii=False)
            names = sorted({e["tags"].get("name", "?") for e in els})
            print(p["id"], len(els), "ways;", "; ".join(names)[:250].encode("ascii", "replace").decode(), flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
