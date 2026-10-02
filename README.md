# riuà

Probabilistic risk of extreme rain and flash floods for the Comunitat Valenciana (Spain), in five levels and three horizons, with every number of the calculation exposed.

- Web: https://ces107.github.io/riua/
- API: https://riua-api.onrender.com (`/docs`)

**Unofficial tool.** AEMET and 112 Comunitat Valenciana are the authoritative sources.

## What it shows

| Level | Meaning | 1-h rain | 12-h rain |
|---|---|---|---|
| 1 | no risk | | |
| 2 | medium, ≈ AEMET yellow | 20 mm | 60 mm |
| 3 | high, ≈ AEMET orange | 40 mm | 100 mm |
| 4 | very high, ≈ AEMET red | 90 mm | 180 mm |
| 5 | extreme, well beyond red | 135 mm | 300 mm |

Thresholds 2–4 are AEMET's Plan Meteoalerta values (annex 1, version of 31 May 2022), identical in the eleven Valencian warning zones. Level 5 is 1.5 × red (1 h) and 5/3 × red (12 h); in the 2023–2026 event catalogue only 29 October 2024 reaches it.

Horizons: **Ahora** 0–6 h (hourly frames), **48 h** (3-hour frames from +6 h), **Días 2–7** (daily frames). Three supports: 0.05° cells (2,542 inside the domain), 420 basin units with upstream topology, and 60 control points where a ravine or river meets a town.

## Architecture

```
GitHub Actions (every 10 min, self-chained)          GitHub Pages                Render (free)
  backend/riua/product.py  ── snapshot.json ──▶  static site web/  ◀── polls ──  backend/riua/api.py
  radar · gauges · NWP · ENS · hydrology           map, panel, audit              /v1/snapshot /v1/point
```

- `backend/riua/product.py` — one production cycle (about 2 min): observations, scenarios, probabilities, levels, snapshot.
- `backend/riua/core/` — `risk.py` (amounts, probabilities, decision), `basins.py`, `hydro.py`, `emos.py`, `grid.py`.
- `backend/riua/radar/` — `qpe.py` (radar rainfall estimate), `nowcast.py` (pysteps STEPS).
- `backend/riua/sources/` — one module per data source.
- `backend/riua/diagnostics/` — ingredients-based diagnostics (MetPy).
- `geo/` — static geodata and its build scripts. `hindcast/` — validation. `web/` — the page (plain HTML/CSS/ES modules, Leaflet).
- `Dockerfile` — full server (`RIUA_ROLE=all` runs the cycle on its own scheduler). `render/Dockerfile` — slim API image.
- `.github/workflows/cycle.yml` — the scheduled worker. GitHub's cron proved unreliable, so each run dispatches the next.

Run locally: `pip install -r backend/requirements.txt`, then `cd backend && python -m riua.product --state ../state --out ../web/data` and `python web/dev/serve.py`.

## Data sources (all free, keyless)

| Source | Use |
|---|---|
| EUMETNET OPERA composite (CC BY 4.0), AEMET composite as gap filler | radar reflectivity, 1 km, 5–10 min; archive back to 2012 |
| SAIH Júcar / Segura / Ebro, AEMET | ~410 rain gauges, ~120 river gauges |
| AEMET CAP feed | official warnings (shown, never mixed into the level) |
| Open-Meteo open-data mirror (CC BY 4.0): AROME-HD 1.3 km, AROME 2.5 km (Météo-France), ICON-EU (DWD), ARPEGE, IFS 0.25° (ECMWF) | scenarios 0–48 h, lagged runs |
| ECMWF open data: ENS 51 members | days 2–7 |
| CEDEX CAUMAX v3.0 | flood peaks for T = 2…500 years at every control point |
| IGN LiDAR DTM, Copernicus GLO-30, OpenStreetMap, CHJ/CEDEX basins | catchments, channel sections, basin units |

## Method

**Radar rainfall.** Reflectivity → rain rate with two Z–R laws (Marshall–Palmer; Z = 300 R^1.4 in convective cores found with Steiner et al. 1995), advection-corrected accumulation (optical flow, pysteps), merged with gauges (wradlib, Pfaff 2010). On 29 Oct 2024, against 167 gauges withheld from the merge: correlation 0.96, mean absolute error 11.7 mm (radar alone 0.62 and 38.1 mm).

**Scenarios.** Now: 20 pysteps STEPS members blended into AROME (radar weight 1 until +1 h, 0 at +4 h), plus the newest AROME runs; hours already elapsed are replaced by the radar-gauge analysis. 48 h: the last runs of five models (time-lagged ensemble, weights halve every 12 h of age). Days 2–7: ECMWF ENS.

**Probability.** For each scenario and frame: the largest 1-h and 12-h accumulation (for 1–2.5 km models, the maximum within a neighbourhood). Then

```
r = max(s1·rain_1h / T_1h , s12·rain_12h / T_12h)        P(≥ level) = Σ weight · Φ( ln(b·r) / σ )
```

`s` corrects coarse models, `b` and `σ` come from the hindcast. A censored-shifted-gamma EMOS (Scheuerer & Hamill 2015) is implemented in `core/emos.py` and switches on only when a fitted calibration is present.

**Decision rule.** The level is the highest L whose probability reaches τ_L. τ is 0.40–0.50 for the next six hours, 0.20–0.25 a day ahead (warning services issue a severe tier from 10–30 %; at 0.40 only 13 % of red zone-days were detected, at 0.20 46 %) and 0.25–0.40 for days 2–7. Current values are in `backend/riua/params.json` (tuned) over `params.py` (defaults).

**Basins.** Each scenario is also averaged over every basin unit and over everything upstream of it, for 1, 3, 6 and 12 h, against the thresholds reduced by the areal factor of Norma 5.2-IC. Rain already on its way from upstream counts.

**Control points.** Losses on a wetness W with a 72-h memory, in three parts fitted to 200 measured floods at 26 gauged ravine-like catchments (`hindcast/calibrate_hydro.py`): most of the ground runs only after P0 = 150 mm (E = (W − P0)² / (W − P0 + S), S = 150 mm); a per-catchment share α (2–18 %, `geo/hydro/loss_params.json`; from a gauge on the stream where there is one, else from a regression) runs after 10 mm; rain above 45 mm/h on a cell-hour runs off whatever the wetness. The design value P0 = 25 mm overestimated ordinary episodes by two orders of magnitude. Then time–area routing on travel times derived from the terrain model, a linear reservoir, then: level 2/3 at the 2-/5-year flood or when the channel is 25 % / 60 % full, level 4 when it overflows, level 5 at +1 m over the bank. Channel capacity comes from a LiDAR cross-section with Manning's equation when it is plausible against CAUMAX, else from a published figure; without capacity, levels 4/5 are the 25-/100-year floods.

## Validation

See `hindcast/results.json` and the Validación page. Method: forecasts that really existed at the time (Open-Meteo Previous Runs archive, OPERA radar archive, ECMWF ENS archive) are run through the same code, scored against the radar-gauge analysis, and σ, b and τ are tuned with leave-one-case-out cross-validation.

<!-- VALIDATION-TABLE -->
**Now (0–6 h)** — 3 cases, 684 frames, σ = 0.2, bias = 0.9; per warning zone and day:

| Level | Min. P | Hits | Misses | False alarms | Detected | False-alarm ratio |
|---|---|---|---|---|---|---|
| 2 | 40 % | 66 | 9 | 7 | 88 % | 10 % |
| 3 | 40 % | 42 | 9 | 5 | 82 % | 11 % |
| 4 | 40 % | 20 | 3 | 4 | 87 % | 17 % |
| 5 | 50 % | 6 | 0 | 5 | 100 % | 45 % |

**48 h (day-ahead runs)** — 7 cases, 226 frames, σ = 0.5, bias = 1.35; per warning zone and day:

| Level | Min. P | Hits | Misses | False alarms | Detected | False-alarm ratio |
|---|---|---|---|---|---|---|
| 2 | 20 % | 167 | 41 | 68 | 80 % | 29 % |
| 3 | 20 % | 94 | 42 | 40 | 69 % | 30 % |
| 4 | 25 % | 23 | 40 | 21 | 37 % | 48 % |
| 5 | 40 % | 3 | 10 | 0 | 23 % | 0 % |

**Days 2–7 (issued 3 and 5 days before)** — 6 cases, 34 frames, σ = 1.0, bias = 1.6; per warning zone and day:

| Level | Min. P | Hits | Misses | False alarms | Detected | False-alarm ratio |
|---|---|---|---|---|---|---|
| 2 | 25 % | 71 | 51 | 20 | 58 % | 22 % |
| 3 | 40 % | 14 | 72 | 4 | 16 % | 22 % |
| 4 | 40 % | 8 | 35 | 6 | 19 % | 43 % |
| 5 | 40 % | 0 | 12 | 0 | 0 % | — |

**Rambla del Poyo, 29 Oct 2024** (observed rain as input): 2266 m³/s at the A-3 gauge (measured 2283 m³/s when the sensor was lost), 2920 m³/s at Paiporta. Peak timing cannot be verified: the Cullera radar was attenuated during the maximum and no open sub-daily gauge data exist for 2024.
<!-- /VALIDATION-TABLE -->

## Known limitations

- Few extreme events in the sample: level-5 statistics rest on one day.
- No free archive of convection-permitting runs beyond one day of lead, and none of high-resolution ensembles.
- The radar loses signal under very heavy rain; gauges correct it only where they exist.
- Hydrology is natural-regime: no dam releases, no blocked bridges, no sewer network. Channel capacities are estimates.
- The 0.25° ensemble cannot produce storm-scale maxima; days 2–7 are coarse in space.
- Depends on third-party services with no availability guarantee.

## Licences and attribution

Weather data by Open-Meteo.com (CC BY 4.0); Météo-France; DWD; ECMWF (CC BY 4.0); EUMETNET OPERA (CC BY 4.0); © AEMET; Confederaciones Hidrográficas del Júcar, Segura y Ebro (provisional data); © Instituto Geográfico Nacional; CEDEX; Copernicus DEM; © OpenStreetMap contributors. The data are modified (re-gridded, aggregated, turned into risk levels).
