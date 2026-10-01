"""Main rivers and ravines as light named lines -> geo/out/rivers.geojson (< 500 KB).

  * Júcar district: official river water bodies of the PHJ 2022-2027 (CHJ), dissolved by river name
    (the part of the official name before ':'), plus named CEDEX/DGA 1:25.000 rivers >= MIN_CEDEX_KM that are not
    already covered (ravines such as Barranc de la Saleta/Pozalet, Horteta, Gallego ...).
  * Outside the Júcar district (Segura, Ebro tributaries...): OpenStreetMap named waterways (ODbL), only inside
    basin units of the domain.
Requires b30_basins.py to have run (stage-A cache + basin raster).
"""
from __future__ import annotations

import json
import pickle
import re
import warnings
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, mapping

from b30_basins import build_vocab, pretty, sample
from common import OUT, RAW, write_geojson

warnings.filterwarnings("ignore")
MIN_CEDEX_KM = 9.0
MIN_OSM_RIVER_KM = 8.0
MIN_OSM_STREAM_KM = 10.0
TOL = 0.0012


def km(line) -> float:
    tot = 0.0
    for g in getattr(line, "geoms", [line]):
        c = np.asarray(g.coords)
        if len(c) < 2:
            continue
        dx = np.diff(c[:, 0]) * 111.195 * np.cos(np.radians(c[:-1, 1]))
        dy = np.diff(c[:, 1]) * 111.195
        tot += float(np.hypot(dx, dy).sum())
    return tot


def kind_of(name: str) -> str:
    w = name.split()[0].lower()
    return {"río": "rio", "riu": "rio", "rio": "rio", "rambla": "rambla", "barranco": "barranco", "barranc": "barranco",
            "arroyo": "arroyo", "canal": "canal", "cañada": "rambla"}.get(w, "otro")


def merged(lines):
    m = shapely.line_merge(shapely.union_all(list(lines)))
    return shapely.simplify(m, TOL)


def main():
    st = pickle.loads((RAW / "basins_stageA.pkl").read_bytes())
    wbl, riv = st["wbl"], st["riv"]
    places = json.loads((RAW / "places_raw.json").read_text(encoding="utf-8"))
    vocab = build_vocab(list(st["wb"].MasaAguSup) + [p["name"] for p in places])
    z = np.load(RAW / "basin_raster.npz")
    lab, ids = z["lab"], list(z["ids"])
    feats = []

    # 1) official river water bodies, dissolved by river name
    groups = defaultdict(list)
    for nm, g in zip(wbl.MasaAguSup, wbl.geometry):
        river = re.sub(r"\s+", " ", nm.split(":")[0]).strip()
        groups[river].append(g)
    wb_union = shapely.union_all(list(wbl.geometry.values))
    wb_buf = wb_union.buffer(0.003)
    for river, gs in sorted(groups.items()):
        m = merged(gs)
        feats.append({"type": "Feature", "geometry": mapping(m), "properties": {
            "name": river, "kind": kind_of(river), "length_km": round(km(m), 1),
            "source": "CHJ PHJ 2022-2027 masas de agua superficial (río)"}})
    n1 = len(feats)

    # 2) named CEDEX rivers not already covered
    for code, nm, L, g in zip(riv.PFAFRIO, riv.NomRio, riv.LongRioKm, riv.geometry):
        if L < MIN_CEDEX_KM or nm.startswith("SIN ") or g is None or g.is_empty:
            continue
        if g.intersection(wb_buf).length > 0.5 * g.length:
            continue
        s = shapely.simplify(g, TOL)
        name = pretty(nm, vocab)
        feats.append({"type": "Feature", "geometry": mapping(s), "properties": {
            "name": name, "kind": kind_of(name), "length_km": round(float(L), 1), "pfafrio": code,
            "source": "CEDEX/DGA ríos 1:25.000"}})
    n2 = len(feats) - n1

    # 3) OSM named waterways inside HydroBASINS-based units
    osm_path = RAW / "osm_ww.json"
    n3 = 0
    if osm_path.exists():
        osm = json.loads(osm_path.read_text(encoding="utf-8"))["elements"]
        g2 = defaultdict(list)
        for el in osm:
            geom = el.get("geometry")
            if not geom or len(geom) < 2:
                continue
            mid = geom[len(geom) // 2]
            k = sample(lab, mid["lon"], mid["lat"])
            if not k or not ids[k - 1].startswith("H"):
                continue
            g2[(el["tags"]["waterway"], el["tags"]["name"])].append(LineString([(p["lon"], p["lat"]) for p in geom]))
        for (cls, name), gs in sorted(g2.items()):
            m = merged(gs)
            L = km(m)
            if L < (MIN_OSM_RIVER_KM if cls == "river" else MIN_OSM_STREAM_KM):
                continue
            feats.append({"type": "Feature", "geometry": mapping(m), "properties": {
                "name": name, "kind": kind_of(name), "length_km": round(L, 1),
                "source": "OpenStreetMap contributors (ODbL), waterway=" + cls}})
            n3 += 1
    n = write_geojson(OUT / "rivers.geojson", feats, meta={
        "name": "Riuà rivers and ravines",
        "attribution": "© Confederación Hidrográfica del Júcar; CEDEX/DGA-MITECO; © OpenStreetMap contributors (ODbL)"})
    print(f"rivers.geojson {n / 1e3:.0f} KB: {n1} official river water bodies, {n2} CEDEX rivers, {n3} OSM waterways")
    for q in ("Poyo", "Chiva", "Saleta", "Pozalet", "Carraixet", "Magro", "Viuda", "Seco", "Sec", "Girona", "Gorgos",
              "Serpis", "Vinalop", "Segura", "Cervol", "Servol", "Sénia", "Senia", "Júcar", "Turia", "Mijares", "Palancia"):
        hits = sorted({(f["properties"]["name"], f["properties"]["length_km"]) for f in feats if q.lower() in f["properties"]["name"].lower()})
        print(f"  {q}: {hits[:6]}")


if __name__ == "__main__":
    main()
