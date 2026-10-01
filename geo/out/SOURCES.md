# Sources, licences and attribution — Riuà geodata

Built on 2026-10-01 with `py -3.11 geo/build/build_all.py`. Every file below was downloaded from the URL shown on that day
(no API keys). Intended use: a public, non-commercial website. **Show the attribution line of every dataset you display.**

## Short attribution line for the website footer

> Zonas y umbrales de aviso: © AEMET. Límites: obra derivada de BDLJE, CC-BY 4.0 ign.es. Cuencas y ríos: Confederación
> Hidrográfica del Júcar (PHJ 2022-2027) y CEDEX/DGA-MITECO; HydroBASINS/HydroRIVERS (HydroSHEDS, WWF); © OpenStreetMap
> contributors (ODbL). Municipios: © Instituto Geográfico Nacional (CC-BY 4.0 scne.es); población: INE.
> Relieve: Copernicus DEM GLO-90 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under
> COPERNICUS by the European Union and ESA; all rights reserved.

## 1. AEMET — Meteoalerta zones and thresholds

| | |
|---|---|
| Used in | `zones.geojson`, `thresholds.json`, `boundary.geojson`, zone/land arrays of `terrain.npz` |
| Zone polygons | https://www.aemet.es/documentos/es/eltiempo/prediccion/avisos/plan_meteoalerta/AEMET-meteoalerta-delimitacion-zonas.zip (shapefile "v6", EPSG:32630; file dated 2025-01-14 on the server; readme dated February 2018) |
| Thresholds | https://www.aemet.es/documentos/es/eltiempo/prediccion/avisos/plan_meteoalerta/METEOALERTA_ANX1_Umbrales_y_niveles_de_aviso.pdf — "Umbrales y niveles de aviso", METEOALERTA_ANX1, **Versión 1, 31-may-2022**, section 3.17 (pages 13-14). Annex of the Plan Meteoalerta **versión 9, 10-ene-2025** (https://www.aemet.es/documentos/es/eltiempo/prediccion/avisos/plan_meteoalerta/PLAN_METEOALERTA_v9_web_externa.pdf). Index page: https://www.aemet.es/es/eltiempo/prediccion/avisos/ayuda |
| Licence | AEMET legal notice (https://www.aemet.es/es/nota_legal): reuse allowed citing AEMET as the source. The zone file states: "Obra derivada de BDLJE 2017-02-28 CC-BY ign.es © Instituto Geográfico Nacional de España". |
| Attribution | "© AEMET" + "Obra derivada de BDLJE CC-BY ign.es" |
| Processing | CV zones only (11); topology-preserving simplification (120 m); provinces/CV/coast derived by dissolving the zones. Thresholds parsed from the PDF with pdfplumber (column order read from the table header; zone names cross-checked against the shapefile). |

**Note on "revised thresholds".** The document served by AEMET on 2026-10-01 is still version 1 of 31-may-2022. All 11 Valencian
zones have the same rain thresholds: 1 h 20 / 40 / 90 mm and 12 h 60 / 100 / 180 mm (yellow / orange / red). Press reports of
21-22 Sep 2026 quote AEMET saying the thresholds have not been changed. Litoral and interior zones differ only in temperature,
wind and snow thresholds (kept in `thresholds.json` under `other`). Re-run `b10_zones.py` to pick up any future revision: the
script fails loudly if the table layout changes.

## 2. Confederación Hidrográfica del Júcar (CHJ) — water-body catchments

| | |
|---|---|
| Used in | `basins.geojson`, `basins_topology.json` (units of kind `wb`), `rivers.geojson` (source "CHJ PHJ ...") |
| URL | https://aps.chj.es/down/SHP/F2333_PHJ_2022_2027_Masas_de_agua_superficial.zip (index: https://aps.chj.es/down/html/descargas.html) — layers "Masas de agua superficial red Cuenca PHJ22" (367 catchments) and "red Río PHJ22" (313 river water bodies), Plan Hidrológico del Júcar 2022-2027, files dated 2025-10-15 |
| Licence | No licence text is published next to the downloads (**UNVERIFIED**). It is Spanish public-sector information (Ley 37/2007 on reuse: free reuse with citation of the source unless stated otherwise). Ask CHJ if a formal statement is needed. |
| Attribution | "Fuente: Confederación Hidrográfica del Júcar, O.A. — Plan Hidrológico 2022-2027" |

## 3. CEDEX / DGA (MITECO) — sub-basins and rivers 1:25.000 with Pfafstetter codes

| | |
|---|---|
| Used in | topology of all Júcar-district units, units of kind `gap_river`, `patch`, `split`, outlets, `rivers.geojson` (source "CEDEX/DGA") |
| URL | https://aps.chj.es/down/SHP/F851_Subcuencas_1_25000_DGA_CEDEX.zip (25 149 sub-basins) and https://aps.chj.es/down/SHP/F850_Rios_1_25000_DGA_CEDEX.zip (12 947 rivers), served by CHJ for its district |
| Licence | MITECO (https://www.miteco.gob.es/es/cartografia-y-sig/ide/descargas/agua/cuencas-y-subcuencas.html): "esta información puede utilizarse de forma libre y gratuita siempre que se mencione al Ministerio para la Transición Ecológica y el Reto Demográfico como autor y propietario de la información". |
| Attribution | "© Ministerio para la Transición Ecológica y el Reto Demográfico (CEDEX / Dirección General del Agua)" |
| Note | The national MITECO download (gis.miteco.gob.es/descargas) is behind a captcha and is NOT used; the CHJ copy is a plain download. River names are upper-case without accents in the source; they are title-cased and accents are restored only for words that appear accented in the official CHJ / INE names. |

## 4. HydroBASINS / HydroRIVERS (HydroSHEDS)

| | |
|---|---|
| Used in | basin units of kind `hybas`, `hybas_group` (outside the Júcar district: Segura basin, Ebro tributaries of Castellón, coastal basins south of the Segura) and their outlets |
| URL | https://data.hydrosheds.org/file/hydrobasins/standard/hybas_eu_lev12_v1c.zip , https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_eu_shp.zip |
| Licence | HydroSHEDS licence agreement (https://www.hydrosheds.org): free for non-commercial and commercial use, attribution required. |
| Attribution / citation | "HydroBASINS / HydroRIVERS — Lehner, B., Grill, G. (2013): Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems. Hydrological Processes 27(15): 2171-2186. www.hydrosheds.org" |

## 5. OpenStreetMap

| | |
|---|---|
| Used in | names of the `hybas*` units and the lines of `rivers.geojson` with source "OpenStreetMap contributors" (only outside the Júcar district) |
| URL | Overpass API, https://overpass-api.de/api/interpreter , query in `geo/build/b25_osm_waterways.py`, data of 2026-10-01 |
| Licence | Open Database License (ODbL) 1.0. The extracted lines/names are a derivative database: keep the attribution and, if you redistribute those lines, do it under ODbL. |
| Attribution | "© OpenStreetMap contributors" with a link to https://www.openstreetmap.org/copyright |

## 6. Instituto Geográfico Nacional (IGN / CNIG) — population nuclei

| | |
|---|---|
| Used in | `places.json` (coordinates and altitude of the capital nucleus of each municipality, official names) |
| URL | OGC API Features https://api-features.ign.es/collections/nuc/items?f=json&limit=1000&cpro=46&skipGeometry=true (one request per province) |
| Licence | CC-BY 4.0 (https://www.ign.es/resources/licencia/Condiciones_licenciaUso_IGN.pdf) |
| Attribution | "© Instituto Geográfico Nacional" / "CC-BY 4.0 scne.es" |

## 7. Instituto Nacional de Estadística (INE) — population

| | |
|---|---|
| Used in | `places.json` (`pop`, municipal register at 1 January 2025; INE municipality codes and official names) |
| URL | https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/<id>?nult=1&tip=AM with table ids 2856 (Alicante), 2865 (Castellón), 2903 (Valencia), 2855 (Albacete), 2869 (Cuenca), 2883 (Murcia), 2899 (Teruel), 2900 (Tarragona) |
| Licence | INE legal notice (https://www.ine.es/aviso_legal): free reuse citing the source. |
| Attribution | "Fuente: Instituto Nacional de Estadística (www.ine.es). Elaboración propia con datos extraídos del sitio web del INE." |

## 8. Copernicus DEM GLO-90

| | |
|---|---|
| Used in | elevation, slope and gradient arrays of `terrain.npz` |
| URL | AWS Open Data, e.g. https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N39_00_W001_00_DEM/Copernicus_DSM_COG_30_N39_00_W001_00_DEM.tif (19 tiles, N37-N41 x W003-E000) |
| Licence | Copernicus DEM licence for GLO-90: free of charge for any use, with the notice below. |
| Attribution (mandatory text) | "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" |

## Known limitations (read before trusting a number)

- Outside the Júcar district the units come from HydroBASINS (500 m global DEM): fine for the Segura main stem and its big
  tributaries, weaker in the flat Vega Baja. HydroBASINS sends the Albatera / Crevillent / Callosa ravines towards the
  Santa Pola salt flats; this link (`H2120682940 -> JT0302`) is plausible but not checked against an official source.
  HydroBASINS was rejected inside the Júcar district because it routes the Rambla del Poyo into the Turia.
- Catchments of short main-stem water bodies were merged when smaller than 25 km2; the merged official names are listed in
  `merged_names`.
- Closed (endorheic) areas are terminal units with `terminal = "endo"` and are not counted in any `up_area_km2`
  (Júcar at its mouth: 20 472 km2 here vs 21 579 km2 official, the difference being those closed areas of La Mancha).
- Unit names of kind `hybas*` come from the longest named OSM waterway inside the unit plus the largest town; units with no
  reliable name are labelled "Cuenca sin nombre (...)" / "Cuencas litorales (...)" instead of guessing.
- The town of Buñol drains to the río Buñol -> Magro -> Júcar (not to the Poyo); Catarroja's centre falls in the official
  catchment of the Barranco de Picassent, right next to the Poyo unit. Both follow the official CHJ polygons.
