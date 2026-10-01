"""Named OSM waterways (Overpass) -> scratch/r5-geo/osm_ww.json. Cached: one request only.

Used only OUTSIDE the Júcar district (Segura basin, Ebro tributaries of Castellón): unit names in b30 and lines
in b40. Data © OpenStreetMap contributors, ODbL 1.0.
The public Overpass instance is sometimes busy (HTTP 504): the script tries three public endpoints in turn.
"""
from __future__ import annotations

import time

import requests

from common import RAW

QUERY = """[out:json][timeout:240];
(
  way["waterway"="river"]["name"](37.4,-3.2,41.2,1.0);
  way["waterway"="stream"]["name"~"^(Rambla|Río|Riu|Barranc|Arroyo)"](37.4,-3.2,39.0,-0.5);
  way["waterway"="stream"]["name"~"^(Rambla|Río|Riu|Barranc|Arroyo)"](40.3,-0.9,41.0,0.6);
);
out tags geom;
"""
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
UA = "riua-geo-build/0.1 (non-commercial flood-risk research)"


def main():
    dest = RAW / "osm_ww.json"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"cached: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    for url in ENDPOINTS:
        print("POST", url)
        try:
            r = requests.post(url, data={"data": QUERY}, headers={"User-Agent": UA}, timeout=300)
        except requests.RequestException as e:
            print("  ", type(e).__name__)
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            dest.write_bytes(r.content)
            print(f"  ok: {len(r.content) / 1e6:.1f} MB, {r.text.count(chr(34) + 'type' + chr(34) + ': ' + chr(34) + 'way')} ways")
            return
        print("  HTTP", r.status_code)
        time.sleep(10)
    raise SystemExit("Overpass unavailable; b30/b40 still run, but units outside the Júcar district get generic names")


if __name__ == "__main__":
    main()
