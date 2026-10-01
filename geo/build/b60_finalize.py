"""places.json: the 542 municipalities of the Comunitat Valenciana + relevant upstream towns, with basin unit
and AEMET zone. Requires b20, b30, b50 (terrain.npz gives which units drain into the CV).
"""
from __future__ import annotations

import json
import re
import warnings

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from common import LAT0, LAT1, LON0, LON1, OUT, RAW, find_one, write_json

warnings.filterwarnings("ignore")
UPSTREAM_MIN_POP = 1000


def fix_article(part: str) -> str:
    """INE style 'Vall d'Uixó, la' -> 'la Vall d'Uixó'; 'Alqueries, les' -> 'les Alqueries'."""
    m = re.match(r"^(.*), (.+)$", part.strip())
    if not m:
        return part.strip()
    art = m.group(2)
    return art + ("" if art.endswith("'") else " ") + m.group(1)


def main():
    raw = json.loads((RAW / "places_raw.json").read_text(encoding="utf-8"))
    z = np.load(RAW / "basin_raster.npz")
    lab, ids, ext, res = z["lab"], [str(x) for x in z["ids"]], z["ext"], float(z["res"])
    t = np.load(OUT / "terrain.npz")
    drains = dict(zip([str(x) for x in t["basin_ids"]], t["basin_drains_to_cv"]))
    zones = gpd.read_file(find_one(RAW / "zonas", "AEMET-meteoalerta-v6-zonas-32630.shp"), encoding="iso-8859-15").to_crs(4326)
    zones = zones[zones.COD_CCAA.astype(str) == "77"]
    zgeom = list(zip(zones.COD_Z.astype(str), zones.geometry))

    def basin_at(lon, lat):
        c, r = int((lon - ext[0]) / res), int((ext[3] - lat) / res)
        if 0 <= r < lab.shape[0] and 0 <= c < lab.shape[1] and lab[r, c] > 0:
            return ids[lab[r, c] - 1]
        return None

    def zone_at(lon, lat):
        p = Point(lon, lat)
        for code, g in zgeom:
            if g.contains(p):
                return code
        return None

    out, n_up, no_basin, no_zone = [], 0, [], []
    for p in raw:
        b = basin_at(p["lon"], p["lat"])
        if p["in_cv"]:
            zc = zone_at(p["lon"], p["lat"])
            if zc is None:  # centre a few metres outside the simplified outline: nearest zone
                pt = Point(p["lon"], p["lat"])
                zc = min(zgeom, key=lambda cg: cg[1].distance(pt))[0]
                no_zone.append(p["name"])
            if b is None:
                no_basin.append(p["name"])
        else:
            inside = LON0 <= p["lon"] <= LON1 and LAT0 <= p["lat"] <= LAT1
            if not (inside and b and drains.get(b, False) and p["pop"] >= UPSTREAM_MIN_POP):
                continue
            zc = None
            n_up += 1
        parts = [fix_article(x) for x in p["name"].split("/")]
        out.append({
            "name": "/".join(parts), "names": parts, "ine": p["ine"], "lat": p["lat"], "lon": p["lon"],
            "alt_m": p["alt_m"], "pop": p["pop"], "province": p["province"], "province_code": p["province_code"],
            "in_cv": p["in_cv"], "basin": b, "zone": zc,
        })
    out.sort(key=lambda p: (-p["in_cv"], p["ine"]))
    doc = {
        "meta": {
            "n_cv": sum(p["in_cv"] for p in out), "n_upstream": n_up,
            "coords": "centre of the capital nucleus of the municipality (IGN núcleos de población), WGS84/ETRS89",
            "population": f"INE, padrón municipal a 1 de enero de {raw[0]['pop_year']}",
            "upstream_rule": f"non-CV municipalities inside the box, in a basin unit that drains into the CV, pop >= {UPSTREAM_MIN_POP}",
            "basin": "id of the basin unit (basins.geojson) containing the town centre",
            "zone": "AEMET Meteoalerta zone code (zones.geojson); null outside the CV",
            "attribution": "© Instituto Geográfico Nacional (CC-BY 4.0 scne.es); © INE (www.ine.es)",
        },
        "places": out,
    }
    n = write_json(OUT / "places.json", doc)
    print(f"places.json {n / 1e3:.0f} KB: {doc['meta']['n_cv']} CV municipalities + {n_up} upstream towns")
    print("  CV towns without basin unit:", no_basin)
    print("  CV towns assigned to nearest zone:", no_zone)
    assert doc["meta"]["n_cv"] == 542


if __name__ == "__main__":
    main()
