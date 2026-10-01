# Riuà — status (2026-10-01, 12:00 local)

## Live
- Web: https://ces107.github.io/riua/ (branch `gh-pages`, published by `deploy/publish.sh`).
- API: https://riua-api.onrender.com (`/healthz`, `/v1/status`, `/v1/snapshot`, `/v1/point?lat=&lon=`, `/docs`);
  Render free service `srv-dav1q28jo6nc73f47k80`, image `render/Dockerfile`, polls the snapshot from Pages.
- Compute: GitHub Actions workflow `.github/workflows/cycle.yml`, cron every 10 min (GitHub may delay
  scheduled runs), ~8 min per run, state kept in the Actions cache, deploys Pages directly. First cloud run
  10:39Z, all sources OK. Manual run: `gh workflow run cycle.yml -R Ces107/riua`.
  The PC scheduled task was removed; `deploy/loop.sh` remains as a local fallback.

## First real cycle (09:42Z)
All sources OK: radar OPERA+AEMET (13 frames), 413 gauges, AROME-HD/AROME/ICON-EU/ARPEGE/IFS lagged runs,
ECMWF ENS 50 members, pysteps STEPS 20 members, 43 AEMET warnings. Levels 4–5 over Castellón coincide with the
AEMET red warnings in force.

## Method in use (2026-10-01 13:30)
- Probabilities from the scenarios (log-normal dressing, sigma 0.25/0.30/0.35); the EMOS gamma path stays in
  the code and switches on only when params.json carries a fitted `calibration`.
- tau never below 40 % (owner's decision): now 60/55/50/45, 48 h 55/50/45/40, days 2-7 50/45/40/40.
- Neighbourhood max 6 km (now) / 12 km (48 h).
- Control points: capacity from geo/hydro/capacity.json (LiDAR section if plausible and not low confidence,
  else published channel capacity); without it, level from unit discharge vs the Gaume envelope.

## Not done / known gaps
- Hindcast + tuning not run: levels are uncalibrated (page says "No oficial y en pruebas").
- Sections agent (h2) resumed to finish the 28 missing channel sections; capacity.json is rebuilt each run.
- MetPy ingredients and ERA5 climate/EFI unfinished (not shown on the page).
- README not written.
