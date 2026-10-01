# Riuà — channel capacity and rating curves at the control points (`h2-sections`)

For every control point of `geo/hydro/catchments/out/control_points.json` (60 points, final list of
h1-catchments; the first 51 sections were located from the seed and kept where the snapped point is
within ~900 m; coves-alcala and gorgos-xabia were relocated to the snapped point) this folder gives the hydraulic
side of the forecast: **how much water the channel carries before it overflows** (`q_bankfull`,
with a low/high range), and **how high the water stands for a given discharge** (rating table and
`h_over_bank(Q)`), from a 1 m LiDAR terrain model and Manning's equation, cross-checked against
published capacities and return-period flows.

## Files

| file | content |
|---|---|
| `sections.json` | one record per control point (see "Fields") |
| `profiles/<id>.json` | raw cross-section profiles (station, elevation) of the chosen section and of the ones ±125 m (or `dx_sec`) up/downstream |
| `published_flows.json` | every published capacity / design / peak / return-period figure found, with source URL and verbatim quote |
| `../build/` | the scripts (below) and `points_config.json` (all analyst decisions, one line per point) |
| `scratch/h2-sections/plots/<id>.png`, `_contact_sheet.png` | QA figure per point: DTM, orthophoto, the three profiles with bank crests (not versioned) |

Pipeline (`py -3.11`, in this order): `fetch_osm.py` → `locate.py` → `fetch_dtm.py` → `fetch_bridges.py` →
`longprofile.py` → `sections.py` → `assemble.py`. `scan.py <id>` prints the bankfull geometry every 50 m
(used to choose/verify the section). Everything downloaded is cached in `scratch/h2-sections/`.

## Data sources (all keyless) and attribution

* **Terrain: ICV 1 m LiDAR DTM** (Institut Cartogràfic Valencià, Generalitat Valenciana), float32 GeoTIFF per
  MTN50 sheet, EPSG:25830, orthometric heights. València province: flight **2015**; Alacant: **2016**;
  Castelló: **2017**. Sheet index by WFS
  `https://terramapas.icv.gva.es/030201_{2015PVAL|2016PALI|2017PCAS}0100_hojas` (layer `ms:distribucion_descargas`,
  attribute `link_tif`), files under `https://descargas.icv.gva.es/dcd/03_mde/02_mdt/...`; read by HTTP range
  requests (GDAL `/vsicurl/`), 1.8 × 1.8 km window per point. Licence **CC BY 4.0** — "© Institut Cartogràfic
  Valencià, Generalitat".
* Not usable: the IGN/CNIG MDT05 services (`https://servicios.idee.es/wcs-inspire/mdt`, coverage
  `Elevacion25830_5`, and `https://api-coverages.idee.es`) work without key but return elevations
  **rounded to whole metres** (int16) — useless for cross-sections. No 2 m coverage is served.
* **Channel centrelines and bridges:** OpenStreetMap via Overpass (© OpenStreetMap contributors, ODbL).
* **Orthophotos for visual checks only:** PNOA máxima actualidad, WMS `https://www.ign.es/wms-inspire/pnoa-ma`
  (CC BY 4.0, scne.es — "PNOA cedido por © Instituto Geográfico Nacional").
* **Return-period flows:** CEDEX/MITECO *Mapa de caudales máximos* (CAUMAX v3.0),
  `https://ceh.cedex.es/caumax/caumax_v30.zip` (rasters `q2…q500.tif`, 500 m grid, natural regime, only
  river cells with basin ≥ 50 km²). No WMS with these values was found. Source: "CEDEX – Dirección General
  del Agua (MITECO)".
* **Published capacities / design flows / observed peaks:** see `published_flows.json` (CHJ, CHS, MITECO,
  BOE, Acuamed, university papers; each entry has its URL and quote).

## Method

1. **Locate.** The seed coordinate (town level) is projected on the OSM waterway of that name
   (`locate.py`); if `geo/hydro/catchments/out/control_points.json` exists its coordinates are used instead
   (it did not exist when this was built: all points start from the seed or from a manual anchor in
   `points_config.json`).
2. **Thalweg.** Along ±1.2 km of the OSM line, the lowest DTM cell within ±`snap_w` m (40 m, more for large
   rivers) of the line, median-filtered over 100 m. Chainage is oriented downstream from the DTM.
3. **Slope.** Theil–Sen slope of the thalweg over ±500 m (windows of ±250 and ±750 m and a reach slope over
   ±3 km from six extra DTM windows are stored in `slope_windows`). The range used for the uncertainty is
   the min/max of the ±500, ±750 and reach values, at least ±20 %. Slopes below 0.0002 are floored. For
   channels built as flat treads between drop structures the reach slope is used (`slope_from: reach`).
4. **Section choice.** Sections perpendicular to the smoothed channel direction, 450 m to each side, 1 m
   sampling (bilinear). Bankfull capacity is first computed every 50 m over ±500 m (`reach_scan`); the
   chosen section is the one — clear of bridges (≥ 35 m from any OSM bridge) and of covered reaches —
   whose capacity is closest to the reach median, unless the analyst fixed it (`offset_m`). The sections
   `dx_sec` (125 m) up- and downstream are kept for consistency (`q_bankfull_3_sections`).
5. **Banks.** Walking outward from the thalweg, a bank crest is the first point ≥ 0.6 m above the
   thalweg from which the ground rises less than 8 % over the next D metres (D = max(12 m, 0.4 × distance
   to the thalweg)) — the break from channel wall to floodplain/street, or a levee top. Later breaks
   replace it only if they at least double the depth (bars and benches inside a larger channel are not
   banks). `bankfull` = the **lower** of the two crests. Where this was wrong on the plot, the crest was
   set by hand (`bank_left_off`, `bank_right_off`, in metres from the thalweg).
6. **Rating (1-D normal depth, divided-channel Manning).**

   `Q(h) = Σ_i (1/n_i) · A_i · R_i^(2/3) · S^(1/2)`, `R_i = A_i / P_i`

   with three sub-sections: left overbank | main channel between the crests | right overbank; the water
   interfaces between sub-sections are not counted in `P`. Overbank water is only counted where it is
   connected to the channel over the crest. `h` runs from 0 to bankfull + 4 m in 0.1 m steps.
   Manning `n` (central, low–high): concrete-lined 0.020 (0.015–0.025); masonry/riprap banks with gravel
   bed 0.030 (0.024–0.038); natural gravel ravine 0.040 (0.030–0.055); cane/reed-choked 0.055
   (0.040–0.080); perennial river 0.040 (0.032–0.055); floodplain/urban 0.10 (0.08–0.12).
   In unlined channels the main-channel discharge is limited to Froude 1 (0.8–1.3 in the range):
   `Q_ch ≤ Fr · A · sqrt(g·A/T)` — steep natural channels do not sustain supercritical flow over a reach
   (Jarrett 1984; Grant 1997), and uncapped Manning gave 8–11 m/s in the steep reaches.
7. **`q_bankfull`** = Q at the bankfull elevation; **range** = (n high, S low, Fr cap low) … (n low, S high,
   Fr cap high). **`h_over_bank`** = for flows above capacity, the depth of water over the lower bank crest,
   i.e. roughly the water depth at the nearest street (tables for the central, low and high ratings).
8. **Cross-check and fallback.** `published_capacity` is the published capacity/design flow of the same
   reach when one exists, with `ratio_dtm_to_published`. `capacity_recommended` is what the forecast
   should use: the DTM value when `confidence` is high/medium; when it is `low`, the published capacity
   of the channel as built; else, as a clearly labelled **proxy**, the CAUMAX T5–T10 natural-regime flow;
   else the DTM value flagged low. Gorge-confined points (`valley_confined: true`: set by the analyst, or
   automatically when bankfull depth > 8 m and capacity > 2 × CAUMAX T500) have no overflow threshold
   (`capacity_recommended.value_m3s = null`): use the rating. CAUMAX is not used for basins < 50 km².

## Fields of `sections.json` → `points[id]`

`section_left_end`, `section_right_end` ([lat, lon]) · `thalweg` (lat, lon, elev) · `bank_left`,
`bank_right` (lat, lon, elev, `crest_found`) · `bankfull_elev_m`, `bankfull_side`, `bankfull_depth_m`,
`bankfull_top_width_m`, `bankfull_area_m2` · `slope`, `slope_range`, `slope_windows` ·
`manning_n_channel` (+range), `manning_n_floodplain` (+range), `lining` · `q_bankfull`, `q_bankfull_range`,
`q_bankfull_3_sections`, `reach_scan` (capacity every 50 m; `p25`/`min` = where overflow starts first) ·
`bankfull_velocity_ms`, `bankfull_froude`, `froude_capped_at_bankfull` ·
`rating` (`h_m[]`, `q_m3s[]`, `q_low_m3s[]`, `q_high_m3s[]`, `floodplain_unbounded_in_section[]`) ·
`h_over_bank` (`q_m3s[]`, `h_over_bank_m[]`, low/high) · `caumax` (`T2…T500` in m³/s, distance of the
river cell used) · `published` (all figures found for the reach) · `published_capacity` ·
`capacity_recommended` (`value_m3s`, `range_m3s`, `basis`) · `dtm` · `perennial` · `confidence`
(`high`/`medium`/`low`) and `confidence_reason` · `analyst_note`.

## Limitations — read before using the numbers

* **1-D uniform flow.** No backwater, no hydraulic jumps, no bends, no unsteady attenuation. In flat
  coastal reaches (Júcar, Segura, lower Vaca, marjal outlets) water levels are controlled from
  downstream and normal depth is a poor model.
* **No bridges, no blockage.** Bridge decks and piers are not in the sections (sections were kept ≥ 35 m
  away from them). On 29 October 2024 the **blockage of bridges by debris, cane and cars was decisive**
  in the Poyo/Horteta/Magro towns: overflow started at flows well below the free-channel capacity
  computed here. The published "capacity without bridges" figures (e.g. Paiporta 1,000–1,700 m³/s) carry
  the same caveat. Treat `q_bankfull` as an **upper bound of the flow at which trouble starts**.
* **Terrain date.** LiDAR of 2015–2017. The October 2024 flood widened and scoured the Poyo, Magro and
  other channels, and works since then (and since 2015) are not included.
* **Bare-earth DTM.** Parapet walls, flood walls < 1–2 m wide, and culverts are not represented; covered
  (buried) reaches are invisible. Dense cane can leave a false "bed" above the real one.
* **Perennial rivers** (flag `perennial`): the LiDAR returns the water surface, so the wetted area at
  the survey date is missing and capacity is underestimated.
* **Bankfull is a terrain concept.** Where houses stand inside the valley (old quarters built on the
  valley side, low riverside streets) damage starts below the detected crest; where the "bank" is a
  farm terrace the first overflow harms nobody. The crest positions are in the JSON and on the plots.
* **Floodplain conveyance** uses the channel slope and n ≈ 0.1 on a bare-earth DTM (buildings removed);
  where `floodplain_unbounded_in_section` is true the water leaves the section sideways (alluvial fans,
  perched channels) and the level above the bank is overestimated for large Q.
* **Manning n and slope** dominate the uncertainty (the range is typically −35 % / +50 %).
* CAUMAX flows are **natural regime** (no dams) and only exist for basins ≥ 50 km²; several small
  ravines have none. Published figures are quoted, not endorsed; sources disagree in places.
