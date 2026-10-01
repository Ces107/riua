# NWP models for Riuà — what to pull, when

Source of every number: commands run on 2026-10-01 against the public bucket `openmeteo` (see
`coord/findings/r1-nwp-live.md`). Delays are "run complete" times (`meta.json` written) observed for the eight
runs of 2026-09-30; the first valid times of a run appear earlier (column "first file"). All times UTC.

Reader: `riua/sources/openmeteo_s3.py`. S3 = `read_field / read_fields / read_series`, API = `read_ensemble_api`.

## Hard facts that shape the schedule

- **AROME France stops at ~37.9–38.0° N.** 10.6 % of the box cells (everything south of a line 37.91° N at
  2.4° W → 38.00° N at 0.8° E: Torrevieja/Pilar de la Horadada, Murcia, Lorca, upper Segura) are NaN in every
  AROME product. Those cells must be filled from ICON-EU (7 km) or IFS 9 km.
- **KNMI and DMI HARMONIE-AROME "Europe" do not cover the box** (southern edge 41.06–41.10° N at 0.7° W).
  There is no AEMET HARMONIE-AROME in Open-Meteo (not in S3, rejected by the API).
- **No ensemble members on S3**, only `precipitation_probability` (P[> 0.1 mm in the step], %). Members come
  from `ensemble-api.open-meteo.com` (free tier: non-commercial, 600/min, 5 000/h, 10 000/day, 300 000/month;
  weight = locations × members/10 for one variable and ≤ 14 days).
- `precipitation`, `showers`, `rain` are sums over the **preceding step** (15 min / 1 h / 3 h / 6 h depending on
  model and lead time), never since-init. The analysis time of a run has no precipitation (all NaN or absent).
- S3 keeps about **8.5 days** of runs in `data_spatial/` (README says 7). Anything older must come from
  `data_run/` (≈ 3 months) or from our own archive.

## Recommended set

| Horizon | Model (S3 id) | Grid in the box | Runs (UTC) | Steps | Complete after run time | First file | Pull |
|---|---|---|---|---|---|---|---|
| Nowcast 0–6 h | `meteofrance_arome_france_hd_15min` | 0.01°, 341×321, NaN south of ~38° N | every hour | 15 min to +6 h (25) | +0.53…+0.68 h | +0.28…+0.39 h | `precipitation` (15-min sum), `cape`, `wind_gusts_10m` |
| Nowcast 0–6 h (gap filler south of 38° N) | `dwd_icon_eu` | 0.0625°, 55×51 | 00,03,…,21 | 1 h | +2.9 h (03/09/15/21), +3.6…+3.7 h (00/06/12/18) | +2.7 h | `precipitation`, `showers`, `cape` |
| 6–48 h, deterministic hi-res | `meteofrance_arome_france_hd` | 0.01°, 341×321 | 00,03,…,21 | 1 h to +51 h (52) | +2.7…+5.1 h (00/03 Z fastest, 06/18 Z slowest) | +2.25…+4.2 h | `precipitation`, `cape`, `wind_u_component_10m`, `wind_v_component_10m`, `wind_gusts_10m` |
| 6–48 h, environment | `meteofrance_arome_france0025` | 0.025°, 137×129 | 00,03,…,21 | 1 h to +51 h | +2.8…+5.3 h | +2.3…+4.3 h | `temperature_/relative_humidity_/wind_u_component_/wind_v_component_/geopotential_height_` at 1000, 925, 850, 700, 500, 300 hPa; `cape`; `pressure_msl` |
| 6–48 h, second opinion + convective diagnostics | `dwd_icon_eu` | 0.0625°, 55×51 | 00,06,12,18 → +120 h; 03,09,15,21 → +30 h | 1 h to +78 h, then 3 h | +3.6…+3.7 h / +2.9 h | +2.7 h | `precipitation`, `showers`, `cape`, `convective_inhibition`, `freezing_level_height`, pressure levels |
| 6–48 h and 2–7 d, global hi-res | `ecmwf_ifs` (O1280, 1237 points in the box) | ~9 km (0.07°) | 00,12 → +360 h; 06,18 → +144 h | 1 h to +90 h, 3 h to +144 h, then 6 h | +6.0…+6.5 h | +5.4…+5.6 h | `precipitation`, `showers`, `cape`, `convective_inhibition`, `total_column_integrated_water_vapour`, 10 m wind |
| 2–7 d, upper air | `ecmwf_ifs025` | 0.25°, 14×13 | 00,12 → +360 h; 06,18 → +144 h | 3 h to +144 h, then 6 h | +7.2…+7.9 h | +7.0…+7.6 h | pressure levels (T, RH, u, v, gh), `total_column_integrated_water_vapour`, `cape` |
| 2–7 d, extra diagnostics | `ncep_gfs025` (+ `ncep_gfs013` for precipitation / PWAT) | 0.25° / 0.117° | 00,06,12,18 | 1 h to +120 h, then 3 h, to +384 h | +5.3…+6.6 h | +3.7…+4.7 h | `lifted_index`, `cape`, `convective_inhibition`, `freezing_level_height` (gfs025); `precipitation`, `showers`, `total_column_integrated_water_vapour` (gfs013) |
| Ensemble 6 h–5 d (API) | `dwd_icon_eu_eps` → API `icon_eu` | 13 km, hourly, 40 members | 00,06,12,18 | hourly | S3 probability complete +3.15…+3.2 h; API latency UNVERIFIED (00 Z run was being served at 07:40) | — | `precipitation` on a 0.25° lattice (182 pts, weight 728/fetch) |
| Ensemble 0–15 d (API) | `ecmwf_ifs_europe_ensemble` | native O1280 ~9 km, **hourly**, 51 members | docs: "only 0z and 6z run" (UNVERIFIED which) | 1 h to +90 h, 3 h to +144 h, 6 h after | 00 Z run complete in the API at 07:47 (< +7.8 h) | — | `precipitation` on a 0.25° lattice (182 pts, weight 928/fetch) |
| Ensemble 2–7 d (API, fallback) | `ecmwf_ifs025_ensemble` | 0.25°, 3-hourly, 51 members | 00,06,12,18 | 3 h | API +11.5 h (18 Z run available 05:33); S3 probability +11.8…+13.0 h | — | `precipitation` (weight 928/fetch) |
| Cheap probability layer (S3) | `ecmwf_ifs025_ensemble`, `dwd_icon_eu_eps`, `ncep_gefs025`, `ecmwf_aifs025_ensemble` | 0.25° / 0.125° | 4/day | 1–6 h | IFS +12…13 h, ICON-EU-EPS +3.2 h, GEFS +5.6…6.8 h, AIFS +8.4…11 h | — | `precipitation_probability` only |

Optional: `meteofrance_arpege_europe` (0.1°, 35×33 in the box, runs 00/06/12/18, hourly to +102 h, complete
+3.6…+4.3 h). Its files hold `precipitation`, `cape` and 24 pressure levels even though `meta.json` lists only 17
surface variables; it is a reasonable third deterministic member but adds nothing ICON-EU + IFS 9 km lack.

Not recommended: `ukmo_global_deterministic_10km` (fine technically, but **CC-BY-SA** share-alike
licence and +6.8…+7.4 h), KNMI/DMI HARMONIE (out of domain).

## Ingestion schedule (cron-style, UTC)

| When | What | Cost measured from Spain (4 threads) |
|---|---|---|
| every hour at :40 | newest `meteofrance_arome_france_hd_15min` run, `precipitation` for 24 steps | ≈ 25 files × 3 requests, < 5 MB, ~5–10 s |
| every 3 h, poll `latest.json` from run+2 h 30 until the run changes (00/03 Z ≈ +2 h 45, others up to +5 h 10) | `meteofrance_arome_france_hd` precipitation + cape, 51 steps | 162 requests, 8.8 MB, 12 s, 0.7 CPU-s for precipitation alone |
| same trigger | `meteofrance_arome_france0025` pressure levels, every 3rd hour is enough | 51 requests, 2.5 MB per valid time for 30 fields |
| every 3 h at run+3 h (+3 h 45 for 00/06/12/18) | `dwd_icon_eu` precipitation, showers, cape, CIN, freezing level | 184 requests, 9.5 MB, 13 s for 31 steps × 4 variables |
| 06:30, 12:30, 18:30, 00:30 | `ecmwf_ifs` precipitation/showers/cape/CIN/PWAT, steps to +168 h | 4 requests, 0.2 MB per valid time and variable group |
| 08:00, 20:00 | `ecmwf_ifs025` pressure levels + PWAT | 7 requests, 0.27 MB per valid time |
| 4×/day ≈ run+4 h | API `icon_eu` 40 members, precipitation, 5 days | weight 728 → 2 912/day |
| 2×/day ≈ 08:00 and 20:00 | API `ecmwf_ifs_europe_ensemble` 51 members, precipitation, 7 days | weight 928 → 1 856/day; 8.7 s, 9 MB JSON, peak RSS 112 MB |

API total ≈ 4 800 weighted calls/day (limit 10 000/day, 5 000/h). Never run two ensemble fetches within the same
minute: a request is admitted only while the minute counter is below 600 and each of these fetches is above it.

Use `in-progress.json` instead of `latest.json` if the first hours of a run are wanted before the run is complete
(files appear one valid time at a time; AROME 00/03 Z first file at +2 h 15).

## Attribution to show on the site

"Weather data by Open-Meteo.com (CC BY 4.0)" with a link to https://open-meteo.com/, plus the upstream producers
actually used: Météo-France (AROME, Licence Ouverte/Etalab), DWD (ICON-EU, ICON-EU-EPS), ECMWF (IFS open data,
CC BY 4.0), NOAA/NCEP (GFS/GEFS). State that the data were modified (re-gridded, aggregated, turned into risk levels).
