"""Download OSM bridges (highway/railway ways tagged bridge=*) within 1.3 km of each anchor.
Cached in scratch/h2-sections/osm/<id>_bridges.json. Used to keep sections away from bridge decks.
Usage: py -3.11 fetch_bridges.py
"""
import json
import os
import time

from common import MISC, OSM_DIR
from fetch_osm import dist, run

R = 1300


def main():
    with open(os.path.join(MISC, "anchors.json"), encoding="utf-8") as f:
        anchors = json.load(f)
    todo = [(k, v["anchor_latlon"]) for k, v in anchors.items()
            if not os.path.exists(os.path.join(OSM_DIR, k + "_bridges.json"))]
    for i in range(0, len(todo), 17):
        chunk = todo[i:i + 17]
        parts = "\n".join(
            f'  way["bridge"]["bridge"!="no"]["highway"](around:{R},{ll[0]},{ll[1]});\n'
            f'  way["bridge"]["bridge"!="no"]["railway"](around:{R},{ll[0]},{ll[1]});' for _, ll in chunk)
        d = run(f"[out:json][timeout:300];\n(\n{parts}\n);\nout tags geom;")
        for pid, ll in chunk:
            els = [e for e in d["elements"]
                   if any(dist(ll[0], ll[1], g["lat"], g["lon"]) < R * 1.05 for g in e.get("geometry", []))]
            with open(os.path.join(OSM_DIR, pid + "_bridges.json"), "w", encoding="utf-8") as f:
                json.dump({"elements": els}, f)
            print(pid, len(els), "bridge ways", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
