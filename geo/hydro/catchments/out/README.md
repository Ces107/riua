# Riuà — control-point catchments, travel times and time–area tables

Static, DEM-derived inputs for the rainfall → discharge step at 60 control points
(ravines and rivers of the Comunitat Valenciana). Built by agent `h1-catchments` on 2026-10-01.
Everything here loads with `json` / `numpy` only.

## Files

| file | content |
|---|---|
| `control_points.json` | `meta` (grid, velocity law, scope definitions, full dam inventory) + `points` (60) |
| `time_area.npz` | time–area tables on the Riuà 0.05° grid, 3 scopes, 2 velocity laws |
| `catchments.geojson` | simplified catchment polygons (74 features: 60 natural + 14 unregulated), 166 KB |
| `streams.geojson` | main channel of each control point, headwater → sea / confluence, 186 KB |

### control_points.json → `points[]`

`id, stream, town, lat, lon` (snapped to the channel cell centre),
`area_km2` (natural), `area_unregulated_km2` (scope 1), `area_unregulated_strict_km2` (scope 2),
`inside_box_fraction` (share of the natural area inside the Riuà box),
`L_km, slope, tc_h` (longest flow path, its mean slope (z_head − z_outlet)/L, Témez tc) and the same three
for the unregulated scope (`L_unregulated_km, slope_unregulated, tc_unregulated_h`),
`elev_outlet_m, elev_head_m, elev_max_m, elev_mean_m`,
`t_longest_h` / `t_longest_unregulated_h` (flood velocity law) and `t_longest_slow_h` (Témez-like law),
`dams[]` (every inventoried dam upstream: `id, name, river, capacity_hm3, catchment_km2, equiv_mm, cuts, nearest, lat, lon, travel_h`),
`downstream_of` = id of the nearest control point **upstream on the same stream** (this point is downstream of it),
`upstream_points` (all control points that drain directly into this one, any stream), `next_downstream`,
`ref_area_km2, ref_source, cedex_bracket_km2` (official area bracket at the point, see Checks),
`confidence` (`high` / `medium` / `low`), `seed_id` (null = point added here), `notes`.

### time_area.npz

Parallel 1-D arrays `point_idx` (index into `point_ids`), `scope`, `cell` (flat Riuà index `j*64+i`),
`lag_h` (integer hour bin, 0 = arrives within the first hour), `area_km2`.

* `scope` 0 = natural, 1 = unregulated, **2 = unregulated_strict (extra, not in the task card)**.
* For every point and scope, `area_km2` sums to the catchment area of that scope **inside the box**
  (verified after writing: 0 mismatches over 60 × 3 × 2 tables).
* A second set with the prefix `slow_` (`slow_point_idx, slow_scope, slow_cell, slow_lag_h, slow_area_km2`)
  holds the same tables for the Témez-like velocity law (see below). Use the unprefixed arrays by default.

```python
z = np.load("time_area.npz"); ids = list(z["point_ids"])
sel = (z["point_idx"] == ids.index("poyo-paiporta")) & (z["scope"] == 1)
# Q(t) [m3/s] = sum over entries of runoff_mm_per_h[cell, t - lag] * area_km2 / 3.6
```

## Method

1. **Working raster**: 3 arc-second lattice (≈ 92 × 72 m), lon −3.5…1.0, lat 37.0…41.2 (5040 × 5400 cells) so that the
   whole Júcar, Turia and Segura basins are inside it even where they leave the Riuà box.
2. **Elevation**: Copernicus DEM GLO-30 (1″), averaged 3 × 3 onto the lattice.
3. **Burn-in**: OSM waterways (`waterway=river` −14 m; named `stream` −10 m; `drain/ditch/canal` only if the name starts
   with Barranc/Barranco/Rambla/Riu/Río/Torrent/Arroyo/Cañada) — 18,747 ways. Irrigation canals are not burned.
4. **Endorheic areas**: one sink at the lowest cell of each *closed basin* (digit 0 in the Pfafstetter code) of the official
   CEDEX/DGA 1:25,000 subcatchment layer of the Júcar demarcation (Almansa 622 km², La Mancha 765 km², Salinas 73 km², …);
   outside that layer the inland sinks of HydroSHEDS v1 are kept (Yecla, Gallocanta, …). 48 sinks.
5. **Routing**: priority-flood fill with ε gradient (Barnes et al. 2014) + steepest-descent D8, own numba code
   (`build/pflood.py`). Accumulated area uses the true cell area per latitude.
6. **Control points** (`build/points_def.json`): target on the named OSM channel nearest to the town, snapped to the highest
   accumulation within ~270 m (capped so it cannot jump to a bigger river). **Dams** (`build/dams_def.json`): OSM dam node →
   nearest main-stem cell; if only the reservoir is mapped, walk downstream while the DEM stays at lake level.
7. **Catchment, L, J, Témez**: upstream cells of the outlet; main channel = longest flow path;
   `tc = 0.3·(L / J^0.25)^0.76` h (L km, J m/m).
8. **Regulation**. Dams are processed upstream → downstream. A dam *cuts* scope 1 when
   `capacity / own unregulated catchment ≥ 20 mm` (and it is not flagged `no_regulation`): a reservoir that stores less than
   20 mm of runoff from its catchment cannot hold a flash flood (Elx 0.3 mm, Tibi 15 mm, Algar 8 mm, Regajo 13 mm,
   María Cristina 13 mm, Alcora 9 mm, Cortes II 13 mm …). Scope 2 (strict) cuts at **every** inventoried dam except Isbert.
   The threshold is a judgement call of this agent, not an official criterion; `meta.dams_inventory` and each point's
   `dams[]` give `equiv_mm` and `cuts` so it can be changed. A cell is regulated for a point only if the first cutting dam
   downstream of the cell is inside the point's catchment.
9. **Travel time** to the outlet = Σ along the D8 path of cell length / velocity:
   * hillslope cells (accumulated area < 0.1 km²): 0.4 m/s;
   * channel cells: `v = k · S^0.5 · A^0.2` (A = accumulated area in km², S = slope of the monotone channel profile over the
     next 12 cells ≈ 1.2 km), bounded;
   * **flood law (default tables)**: k = 15, bounds 1–4 m/s;
   * **slow law (`slow_*` tables)**: k = 5, bounds 0.5–4 m/s.
10. **Time–area table**: every catchment cell goes to its Riuà cell and to the hour bin `floor(t)`.

## Velocity calibration — the two targets disagree

* Observation, 29 Oct 2024: the Poyo wave took roughly 2–3 h from Chiva/Cheste to Paiporta.
  Flood law: Chiva → Paiporta 34.8 km in **3.08 h** (3.1 m/s), Cheste → Paiporta 27.5 km in **2.50 h**,
  A-3 gauge → Paiporta 18.3 km in 1.79 h.
* Témez: for Paiporta tc = 13.8 h. The flood law gives a longest travel time of 5.0 h, i.e. about 0.4–0.5 × Témez for most
  basins. Matching Témez needs k ≈ 5 (slow law: longest travel time to Paiporta 13.8 h, but Chiva → Paiporta 9.1 h, which
  contradicts the observation).
* Decision: default = flood law (the product is for extreme events); the Témez-like tables are shipped alongside. Neither
  law has been calibrated against measured hydrographs. k was tuned on **one** event on **one** stream.

## Choice of flow network (evidence)

Three candidates were routed and compared (`build/04_eval_products.py`, `scratch/h1-catchments/eval_horta.png`):

| | Barranco de Chiva at lon −0.64 | Poyo at Torrent–Paiporta | Saleta at Aldaia |
|---|---|---|---|
| HydroSHEDS v1 3″ DIR | 156 km² | **177 km²**, leaves the channel at Picanya | **233 km²** (the Poyo is sent through Aldaia into the new Turia channel) |
| Copernicus GLO-30 → 3″, no burn | 156 km² | 354–360 km², follows the OSM line to Catarroja | 60 km² |
| Copernicus GLO-90, no burn | 156 km² | 411 km², deviates at Picanya | 1 km² |
| official CEDEX | — | 351 km² (Poyo 259.5 + Horteta 91.3) | Pozalet 51.8 km² |

In the hills all three agree; at river mouths they agree within 1–3 % (Júcar, Turia, Mijares, Palancia). HydroSHEDS is wrong
for the Poyo in l'Horta Sud, and both HydroSHEDS and the unburned Copernicus DEM spill the endorheic Almansa corridor into
the Cànyoles (Albaida at its mouth 1,760 / 2,065 km² vs 1,222 official). Hence the product used: GLO-30 + OSM burn + official
closed basins. Against 439 official CEDEX basins > 40 km² sampled at their mouths it is within 10 % for 72 % of them
(HydroSHEDS 64 %, unburned GLO-30 67 %; many of the misses are sampling artefacts at confluences).

## Checks

* **Official bracket** (`cedex_bracket_km2`): the CEDEX subcatchment containing the point gives the official area strictly
  upstream (`lo`) and including that subcatchment (`hi`), using Pfafstetter topology and excluding closed basins
  (`build/cedexref.py`). 53 points have a bracket; 52 are inside [0.9·lo, 1.1·hi]. The exception:
  `casella-alzira` (43 vs 7.5–21 km²). For `saleta-aldaia`, `coves-alcala`, `chinchilla-orpesa`, `ovejas-alacant` the
  containing polygon belongs to another river or the coast, so the bracket is not reported; `abanilla-benferri`,
  `segura-orihuela` and `seco-pilar` are in the Segura demarcation, for which no official layer was used (unchecked).
* **Published areas (task card) vs this product**

| basin | published | here | note |
|---|---|---|---|
| Poyo at the A-3 gauge | 187 | 183.7 | point at the OSM A-3 crossing; gauge coordinate itself not verified |
| Poyo at the Albufera | 430–480 | 360.5 at Catarroja | the channel section only; lateral plain areas reach the lake separately |
| Carraixet | 310 (CEDEX 315.0) | 291.4 at Alboraia | point is ~3 km above the mouth |
| Magro | 1,540 (CEDEX 1,584.1 incl. 33 closed) | 1,526.5 at Algemesí | |
| Serpis | 750 (CEDEX 770.6) | 745.2 at Gandia | |
| Girona | 120 (CEDEX 112.0) | 107.4 at El Verger | |
| Palancia | 910 (CEDEX 980.1) | 970.8 at Sagunt | the 910 figure is not confirmed by CEDEX |
| Mijares | 4,030 (CEDEX 4,046.0 incl. 74 closed) | 3,957.2 at Vila-real | |
| Rambla de la Viuda | 1,510 (CEDEX 1,516.8 incl. 74 closed) | 1,437.4 | |
| Vinalopó | 1,690 | 1,605.7 at Elx | point above the mouth; Salinas and Almansa closed basins excluded |
| Turia | 6,390 (CEDEX 6,333.1) | 6,153.3 at Quart | point ~10 km above the mouth |
| Júcar | 21,580 (CEDEX 21,044.5) | 19,423.1 at Sueca | official figures include ≈ 1,600 km² of closed basins |

* **Doubtful delineations** (`confidence`): 29 high, 28 medium, 3 low.
  Low: `saleta-aldaia` (flat urban plain, culverted, disperses), `casella-alzira` (junctions differ from the official
  network), `segura-orihuela` (10 % outside the box, heavily regulated, Guadalentín counted although normally diverted at
  El Paretón, no official reference used).
  Medium for plain / artificial reaches (Poyo below Paiporta, Picassent, lower Carraixet, lower Magro, Riu Verd, Júcar,
  Sec de Borriana, Sec de Castelló, Juncaret, Beniopa), karst (Gallinera, Girona, Gorgos, Vaca, Coves/Sant Miquel) and
  partly endorheic basins (Vinalopó, Cànyoles, Magro on the Utiel plateau). Reasons are in each point's `notes`.
* **Seed corrections**: all 51 seed points kept, 9 added (`poyo-cheste`, `horteta-torrent`, `magro-real`, `verd-alzira`,
  `casella-alzira`, `beniopa-gandia`, `chinchilla-orpesa`, `aiguaoliva-vinaros`, `tarafa-aspe`). Moved > 2 km:
  `carraixet-alfara` (3.3 km, to the Alboraia reach), `castellarda-lliria` (2.9 km, seed was on the Turia),
  `viuda-almassora` (4.7 km, seed was on the Mijares), `coves-alcala` (4.7 km, seed was on no channel),
  `gorgos-xabia` (6.0 km, to the Xàbia reach), `girona-verger` (2.1 km), `seco-pilar` (2.3 km).
  Seed dam lists that do not hold: Forata is not upstream of `jucar-alzira` (the Magro joins below it, it is upstream of
  `jucar-sueca`); Escalona drains into the Tous reservoir.

## Known limits

* Channel catchments only: overbank flow between neighbouring ravines in the plain (Poyo ↔ Saleta ↔ Turia channel,
  Magro ↔ Júcar) is not represented.
* Copernicus GLO-30 is a surface model (buildings, embankments); the burn-in is what keeps channels through towns.
* Closed basins are treated as never contributing, even in an extreme event.
* Reservoir capacities are the rounded totals listed by embalses.net; Contreras is listed at 361 hm³ (design 852 hm³);
  Elx 0.4 hm³ is a commonly quoted figure that was not verified.
* Scope 1 treats María Cristina, Regajo, Algar, Tibi, Elx, Alcora, Cortes II, El Naranjero, Molinar as not regulating —
  this differs from the seed list; use scope 2 for the seed's reading.

## Sources and attribution

* Copernicus DEM GLO-30 / GLO-90 — AWS Open Data buckets `copernicus-dem-30m`, `copernicus-dem-90m`.
  "© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and
  ESA; all rights reserved."
* OpenStreetMap waterways, dams and the A-3 geometry via Overpass — © OpenStreetMap contributors, ODbL 1.0.
* HydroSHEDS v1 (Lehner, Verdin & Jarvis 2008), 3″ flow direction — used for the comparison, the ocean mask and the inland
  sinks outside the Júcar demarcation. https://www.hydrosheds.org
* Confederación Hidrográfica del Júcar geodata: "Subcuencas 1:25.000 DGA-CEDEX" (F851) and "Ríos 1:25.000 DGA-CEDEX" (F850),
  as downloaded by `r5-geo` into `scratch/r5-geo/chj` — reference areas and closed basins. Licence terms not checked by this agent.
* embalses.net (`cuenca-7-jucar.html`, `cuenca-1-segura.html`) — reservoir capacities.

## Rebuild

`geo/hydro/catchments/build/`, in order, with `py -3.11`: `01_fetch_copdem.py 30`, `01_fetch_copdem.py 90`, `02_fetch_osm.py`,
HydroSHEDS window (commands in `coord/journal/h1-catchments.md`), `03_prepare_grids.py`, `04_eval_products.py` (optional),
`05_build_flow.py`, `06_validate_cedex.py D` (optional), `07_snap_points.py`, `08_build_outputs.py` (≈ 8–12 min, ≈ 2 GB RAM).
Edit `points_def.json`, `dams_def.json`, `points_notes.json` to change points, dams or notes.
