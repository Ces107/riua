"""AEMET Meteoalerta zones (CV), official precipitation thresholds, CV/province boundary.

Outputs: geo/out/zones.geojson, geo/out/thresholds.json, geo/out/boundary.geojson
Sources (AEMET, https://www.aemet.es/es/eltiempo/prediccion/avisos/ayuda):
  - AEMET-meteoalerta-delimitacion-zonas.zip (shapefile, EPSG:32630, derived from IGN BDLJE, CC-BY ign.es)
  - METEOALERTA_ANX1_Umbrales_y_niveles_de_aviso.pdf (parsed with pdfplumber, nothing typed by hand)
"""
from __future__ import annotations

import re
import warnings

import geopandas as gpd
import pdfplumber
import shapely
from shapely.geometry import box, mapping

from common import LAT0, LAT1, LON0, LON1, OUT, RAW, download, find_one, unzip, write_geojson, write_json

warnings.filterwarnings("ignore")

BASE = "https://www.aemet.es/documentos/es/eltiempo/prediccion/avisos/plan_meteoalerta/"
ZIP_URL = BASE + "AEMET-meteoalerta-delimitacion-zonas.zip"
ANX1_URL = BASE + "METEOALERTA_ANX1_Umbrales_y_niveles_de_aviso.pdf"
PLAN_URL = BASE + "PLAN_METEOALERTA_v9_web_externa.pdf"
PROV = {"03": "Alicante", "12": "Castellón", "46": "Valencia"}


def load_zones() -> gpd.GeoDataFrame:
    z = download(ZIP_URL, RAW / "AEMET-meteoalerta-delimitacion-zonas.zip")
    d = unzip(z, RAW / "zonas")
    shp = find_one(d, "AEMET-meteoalerta-v6-zonas-32630.shp")
    g = gpd.read_file(shp, encoding="iso-8859-15")
    g["COD_Z"] = g["COD_Z"].astype(str)
    return g


def parse_thresholds() -> dict:
    pdf_path = download(ANX1_URL, RAW / "anx1_www.pdf")
    header_expected = "umbrales temp. máximas temp. mínimas racha máxima precipitación 12 h precipitación 1 h nieve 24 h"
    sub_expected = "CÓDIGO NOMBRE DE LA ZONA PROVINCIA" + " amllo nanja rojo" * 6
    row_re = re.compile(r"^(77\d{4}) (.+?) (Alicante|Castellón|Valencia)((?: -?\d+){18})$")
    rows, version, fecha, pages = {}, None, None, []
    with pdfplumber.open(pdf_path) as pdf:
        npages = len(pdf.pages)
        for pn, page in enumerate(pdf.pages, start=1):
            lines = [re.sub(r"\s+", " ", l).strip() for l in (page.extract_text() or "").splitlines()]
            for l in lines:
                m = re.search(r"Versión: (\S+)", l)
                if m:
                    version = m.group(1)
                m = re.search(r"Fecha: (\S+)", l)
                if m:
                    fecha = m.group(1)
            hits = [row_re.match(l) for l in lines]
            hits = [h for h in hits if h]
            if not hits:
                continue
            pages.append(pn)
            # the column order is read from the table header printed on the page where the CV table starts
            if not rows:
                assert header_expected in lines, f"unexpected header on page {pn}"
                assert sub_expected in lines, f"unexpected sub-header on page {pn}"
            for h in hits:
                v = [int(x) for x in h.group(4).split()]
                tmax, tmin, gust, p12, p1, snow = (v[i:i + 3] for i in range(0, 18, 3))
                rows[h.group(1)] = {
                    "name": h.group(2),
                    "province": h.group(3),
                    "precip_1h_mm": dict(zip(("yellow", "orange", "red"), p1)),
                    "precip_12h_mm": dict(zip(("yellow", "orange", "red"), p12)),
                    "other": {
                        "tmax_c": tmax, "tmin_c": tmin, "gust_kmh": gust, "snow_24h_cm": snow,
                    },
                }
    assert len(rows) == 11, f"expected 11 CV zones, parsed {len(rows)}"
    for code, r in rows.items():
        for k in ("precip_1h_mm", "precip_12h_mm"):
            y, o, rd = (r[k][c] for c in ("yellow", "orange", "red"))
            assert 0 < y < o < rd, (code, k, r[k])
    return {
        "source": {
            "document": "Plan Meteoalerta — Anexo 1: Umbrales y niveles de aviso (METEOALERTA_ANX1)",
            "version": version,
            "document_date": fecha,
            "url": ANX1_URL,
            "pages": pages,
            "n_pages": npages,
            "parent_plan": "Plan Nacional de Predicción y Vigilancia de Fenómenos Meteorológicos Adversos (Meteoalerta), versión 9, 10-ene-2025",
            "parent_plan_url": PLAN_URL,
            "publisher": "Agencia Estatal de Meteorología (AEMET)",
            "retrieved": "2026-10-01",
            "note": (
                "Parsed from the PDF served by aemet.es on 2026-10-01 (section 3.17 Comunidad Autónoma Valenciana). "
                "It is still version 1 of 31-may-2022: the precipitation thresholds of the Valencian zones have NOT been "
                "changed (AEMET stated so publicly on 21/22-sep-2026). All 11 zones share the same rain thresholds; "
                "litoral/interior zones differ only in temperature, wind and snow (kept under 'other')."
            ),
        },
        "units": "mm accumulated in 1 hour / in 12 hours; a level is reached when the accumulation is >= the value",
        "levels": ["yellow", "orange", "red"],
        "zones": rows,
    }


def feat(geom, props):
    return {"type": "Feature", "properties": props, "geometry": mapping(geom)}


def main():
    g = load_zones()
    cv = g[g.COD_CCAA.astype(str) == "77"].copy().sort_values("COD_Z").reset_index(drop=True)
    assert len(cv) == 11
    thr = parse_thresholds()
    assert set(thr["zones"]) == set(cv.COD_Z), "zone codes in PDF and shapefile differ"
    for _, r in cv.iterrows():  # names must agree between PDF and shapefile
        assert thr["zones"][r.COD_Z]["name"] == r.NOM_Z, (r.COD_Z, r.NOM_Z, thr["zones"][r.COD_Z]["name"])
    n = write_json(OUT / "thresholds.json", thr, indent=1)
    print(f"thresholds.json {n} B")

    # --- zones: topology-preserving simplification (no slivers between neighbours), UTM 30N metres
    cv["geometry"] = shapely.coverage_simplify(cv.geometry.values, 120.0)
    cv["area_km2"] = cv.area / 1e6
    cvll = cv.to_crs(4326)
    feats = []
    for _, r in cvll.iterrows():
        c = r.geometry.representative_point()
        feats.append(feat(r.geometry, {
            "code": r.COD_Z, "name": r.NOM_Z, "province": PROV[r.COD_Z[2:4]], "province_code": r.COD_Z[2:4],
            "area_km2": round(float(r.area_km2), 1), "label_lon": round(c.x, 4), "label_lat": round(c.y, 4),
        }))
    n = write_geojson(OUT / "zones.geojson", feats, meta={
        "name": "AEMET Meteoalerta zones — Comunitat Valenciana",
        "attribution": "© AEMET. Obra derivada de BDLJE CC-BY ign.es © Instituto Geográfico Nacional de España",
    })
    print(f"zones.geojson {n} B, {len(feats)} zones")

    # --- boundary: provinces + CV outline + coastline + land of the whole box (for sea masking)
    feats = []
    for pc, pname in PROV.items():
        geom = shapely.union_all(cv[cv.COD_Z.str[2:4] == pc].geometry.values)
        feats.append(("province", {"kind": "province", "code": pc, "name": pname, "area_km2": round(geom.area / 1e6, 1)}, geom))
    cv_union = shapely.union_all(cv.geometry.values)
    feats.append(("cv", {"kind": "region", "code": "77", "name": "Comunitat Valenciana", "area_km2": round(cv_union.area / 1e6, 1)}, cv_union))
    # coastline = CV outline not shared with neighbouring (non-CV) zones
    others = g[g.COD_CCAA.astype(str) != "77"]
    near = others[others.intersects(cv_union.buffer(2000))]
    neigh = shapely.union_all(near.geometry.values).buffer(400)
    coast = cv_union.boundary.difference(neigh)
    parts = [p for p in getattr(coast, "geoms", [coast]) if p.length > 3000]
    coast = shapely.line_merge(shapely.MultiLineString(parts))
    feats.append(("coast", {"kind": "coastline", "name": "Costa de la Comunitat Valenciana", "length_km": round(coast.length / 1e3, 1)}, coast))
    # all land in the box (mainland + islands), coarser
    bx = gpd.GeoSeries([box(LON0 - 0.3, LAT0 - 0.3, LON1 + 0.3, LAT1 + 0.3)], crs=4326).to_crs(g.crs).iloc[0]
    land = shapely.union_all(g[g.intersects(bx)].geometry.values).intersection(bx)
    land = shapely.simplify(land, 400.0)
    feats.append(("land", {"kind": "land_box", "name": "Land inside the Riuà box (+0.3° margin)"}, land))
    out = []
    for _, props, geom in feats:
        gl = gpd.GeoSeries([geom], crs=g.crs).to_crs(4326).iloc[0]
        out.append(feat(gl, props))
    n = write_geojson(OUT / "boundary.geojson", out, meta={
        "name": "Comunitat Valenciana boundary, provinces, coastline, land mask",
        "attribution": "© AEMET. Obra derivada de BDLJE CC-BY ign.es © Instituto Geográfico Nacional de España",
    })
    print(f"boundary.geojson {n} B, {len(out)} features; coast {coast.length/1e3:.0f} km in {len(parts)} parts")


if __name__ == "__main__":
    main()
