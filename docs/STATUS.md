# Riuà — status at the hard stop (2026-10-01, ~11:00 local)

NOT deployed. Nothing pushed to GitHub, nothing created on Render. Local repo only: `C:\Users\cpereiro\IdeaProjects\riua`.

## Done and run on real data
- Data sources, all keyless, each with a working module in `backend/riua/sources/`:
  radar (EUMETNET OPERA + AEMET fill, archive back to 2012), rain/river gauges (SAIH Júcar/Segura/Ebro, AEMET),
  AEMET CAP warnings, NWP from the Open-Meteo S3 mirror (AROME-HD, AROME, ICON-EU, ARPEGE, IFS), ECMWF ENS direct.
- Geodata (`geo/out`): AEMET zones + official thresholds parsed from the PDF, 420 basin units with topology,
  rivers, 629 places, terrain on the 0.05° grid.
- Radar QPE (`backend/riua/radar/qpe.py`): Steiner Z-R, advection-corrected accumulation (pysteps), gauge merge
  (wradlib AdjustMixed). Checked on 29 Oct 2024 against 167 held-out gauges: correlation 0.96, MAE 11.7 mm
  (radar alone: 0.62, 38.1 mm). Merged maximum 663 mm at Turís.
- Ensemble nowcast (`radar/nowcast.py`, pysteps STEPS) ran on live radar.
- Risk core (`core/risk.py`, `emos.py`, `basins.py`, `hydro.py`): censored shifted gamma EMOS (CRPS formula checked
  against Monte Carlo), Gaussian-copula union of the 1-h and 12-h criteria, asymmetric decision rule, catchment
  aggregation, rain→discharge→stage routing.
- Live members + predictors ran on today's runs (`ingest.py`).

## Written but NOT yet run to completion
- `backend/riua/product.py` (full cycle → snapshot.json): first run was still in progress at the stop.
- `backend/riua/api.py`, `card.py`, `Dockerfile`, `render/Dockerfile`, `.github/workflows/cycle.yml`: untested.

## Not done
- Web page (agent `w1-web` was building `web/`; check `coord/status/w1-web.txt`).
- Control points: catchments (`h1`) and channel sections (`h2`) agents unfinished; inundation mapping not started.
- MetPy ingredients (`i1`) and ERA5 climate/GEV/EFI (`c1`) agents unfinished.
- Hindcast: truth being built (`hindcast/truth/`); `hindcast/run.py` (fit calibration, tune tau, hit/miss/false-alarm
  tables) NOT written. Calibration coefficients in use are placeholders (`DEFAULT_CAL` in product.py) and
  `params.py` tau values are untuned defaults. The levels are therefore NOT validated.
- Truth files lack the 12-h cell-maximum field; add it to `build_truth.py` before fitting.
- Open-Meteo Previous-Runs API quota (10,000/day per IP) was ~used up; day-1/day-2 archives exist only for
  2024-10-28..30 and (downloading) 2024-11-01..14. Other cases need later days' quota.
- README, deploy (GitHub repo + Pages + Actions cron, Render API service), memory of the Gaume envelope
  coefficient to verify.

## Resume order
1. Check `scratch/out/snapshot.json` exists; fix product.py if the cycle failed.
2. Finish/inspect `web/`, copy snapshot into `web/data/`, look at it in a browser.
3. `gh repo create Ces107/riua --public`, push, enable Pages (build_type workflow), create the Render service from
   `render/Dockerfile` (see plusUltra `skills/render-api-deploy/SKILL.md`).
4. Hindcast run + tuning, then README with the real numbers.
