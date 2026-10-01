"""Static geodata of the page: geo/out (+ geo/hydro) -> web/geo.

    py -3.11 web/dev/build_geo.py

Nothing is simplified: polygons and lines keep every vertex of the source at the precision
the source already has (0.001 deg for basins, 0.0001 deg for rivers); they are only written
as integer deltas, which is what makes the files 3-4 times smaller than GeoJSON.

Compact line format ("rings"): each ring / line is a flat list [x0, y0, dx1, dy1, dx2, dy2, ...]
of integers in units of 1/scale degrees (x = lon, y = lat). web/js/geo.js decodes it.

Outputs
  boundary.geojson, zones.geojson   plain copies
  basins.json   {scale, props:[names], units:[[idx,id,name,river,town,area_km2,up_area_km2,next,[label lon,lat], [rings]]]}
  rivers.json   {scale, lines:[[name, kind, length_km, [ring]]]}
  places.json   {places:[[name, alt_name|null, lat, lon, pop, basin, zone|null, province_code]]}
  cells.json    {nx, ny, zone_codes, zone: base64 uint8[ny*nx] (255 = none), basin: base64 uint16le[ny*nx] (65535 = none),
                 cv: base64 uint8[ny*nx] share of the cell inside the Comunitat Valenciana * 200}
  points.json   control points (only when geo/hydro/catchments/out/control_points.json exists)
  catchments.geojson, streams.geojson   copies with 4 decimals (only when they exist)
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "geo" / "out"
HYD = REPO / "geo" / "hydro"
OUT = REPO / "web" / "geo"


def dump(name: str, obj) -> None:
    f = OUT / name
    f.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{name:22s} {f.stat().st_size / 1024:7.1f} KB")


def ring(coords, scale: int) -> list[int]:
    out, px, py = [], 0, 0
    for k, (x, y) in enumerate(coords):
        ix, iy = int(round(x * scale)), int(round(y * scale))
        if k and ix == px and iy == py:
            continue
        out += [ix - px, iy - py] if k else [ix, iy]
        px, py = ix, iy
    return out


def rings_of(geom, scale: int) -> list[list[int]]:
    t, c = geom["type"], geom["coordinates"]
    if t == "Polygon":
        return [ring(r, scale) for r in c]
    if t == "MultiPolygon":
        return [ring(r, scale) for poly in c for r in poly]
    if t == "LineString":
        return [ring(c, scale)]
    if t == "MultiLineString":
        return [ring(r, scale) for r in c]
    raise ValueError(t)


def rounded(geom, nd: int):
    def rr(c):
        return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [rr(x) for x in c]
    return {"type": geom["type"], "coordinates": rr(geom["coordinates"])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("boundary.geojson", "zones.geojson"):
        shutil.copyfile(SRC / name, OUT / name)
        print(f"{name:22s} {(OUT / name).stat().st_size / 1024:7.1f} KB (copy)")

    g = json.loads((SRC / "basins.geojson").read_text(encoding="utf-8"))
    units = []
    for k, f in enumerate(g["features"]):
        p = f["properties"]
        assert p["idx"] == k, "basins.geojson must be in idx order"
        units.append([p["idx"], p["id"], p["name"], p.get("river"), p.get("town"), p["area_km2"], p["up_area_km2"],
                      p.get("next"), [round(p["label"][0], 3), round(p["label"][1], 3)], rings_of(f["geometry"], 1000)])
    dump("basins.json", {"scale": 1000, "props": ["idx", "id", "name", "river", "town", "area_km2", "up_area_km2",
                                                 "next", "label", "rings"], "units": units})

    g = json.loads((SRC / "rivers.geojson").read_text(encoding="utf-8"))
    lines = [[f["properties"]["name"], f["properties"]["kind"], f["properties"]["length_km"],
              rings_of(f["geometry"], 10000)] for f in g["features"]]
    lines.sort(key=lambda r: -r[2])
    dump("rivers.json", {"scale": 10000, "props": ["name", "kind", "length_km", "rings"], "lines": lines})

    pl = json.loads((SRC / "places.json").read_text(encoding="utf-8"))["places"]
    rows = []
    for p in pl:
        alt = [n for n in p.get("names", []) if n != p["name"]]
        rows.append([p["name"], alt[0] if alt else None, round(p["lat"], 4), round(p["lon"], 4), p["pop"],
                     p["basin"], p["zone"], p["province_code"]])
    rows.sort(key=lambda r: -(r[4] or 0))
    dump("places.json", {"props": ["name", "alt", "lat", "lon", "pop", "basin", "zone", "province_code"], "places": rows})

    z = np.load(SRC / "terrain.npz", allow_pickle=False)
    zone = z["zone_idx"].astype(int).ravel()
    basin = z["basin_idx"].astype(int).ravel()
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")
    dump("cells.json", {
        "nx": int(z["zone_idx"].shape[1]), "ny": int(z["zone_idx"].shape[0]),
        "zone_codes": [str(c) for c in z["zone_codes"]], "zone_names": [str(c) for c in z["zone_names"]],
        "zone": b64(np.where(zone < 0, 255, zone).astype(np.uint8)),
        "basin": b64(np.where(basin < 0, 65535, basin).astype("<u2")),
        "cv": b64(np.clip(np.rint(z["cv_frac"].ravel() * 200), 0, 200).astype(np.uint8)),
    })

    cdir, sdir = HYD / "catchments" / "out", HYD / "sections" / "out"
    cp = cdir / "control_points.json"
    if cp.exists():
        pts = json.loads(cp.read_text(encoding="utf-8"))
        pts = pts["points"] if isinstance(pts, dict) else pts
        secs = {}
        if (sdir / "sections.json").exists():
            sj = json.loads((sdir / "sections.json").read_text(encoding="utf-8"))
            sj = sj.get("points", sj) if isinstance(sj, dict) else sj
            secs = sj if isinstance(sj, dict) else {s["id"]: s for s in sj}
        dump("points.json", {"points": [slim_point(p, secs.get(p["id"], {})) for p in pts]})
        for name in ("catchments.geojson", "streams.geojson"):
            if (cdir / name).exists():
                gj = json.loads((cdir / name).read_text(encoding="utf-8"))
                for f in gj["features"]:
                    f["geometry"] = rounded(f["geometry"], 4)
                dump(name, gj)
    else:
        for name in ("points.json", "catchments.geojson", "streams.geojson"):
            (OUT / name).unlink(missing_ok=True)
        print("no control points yet: points.json not written")


def slim_point(p: dict, s: dict) -> dict:
    """What the page needs of a control point. Capacity = the value the backend uses (static.hydro_net)."""
    cap = s.get("q_capacity") or s.get("q_bankfull")
    rng = s.get("q_bankfull_range")
    return {"id": p["id"], "stream": p.get("stream"), "town": p.get("town"),
            "lat": round(float(p["lat"]), 5), "lon": round(float(p["lon"]), 5),
            "area_km2": p.get("area_unregulated_km2") or p.get("area_km2"), "tc_h": p.get("tc_h"),
            "cap": float(cap) if cap else None, "cap_range": rng if rng else None,
            "bank_m": s.get("bankfull_depth_m"), "lining": s.get("lining"),
            "confidence": s.get("confidence"), "gauge": p.get("gauge_id") or p.get("gauge")}


if __name__ == "__main__":
    main()
