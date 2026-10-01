# Riuà — status (2026-10-01, 12:00 local)

## Live
- Web: https://ces107.github.io/riua/ (branch `gh-pages`, published by `deploy/publish.sh`).
- API: https://riua-api.onrender.com (`/healthz`, `/v1/status`, `/v1/snapshot`, `/v1/point?lat=&lon=`, `/docs`);
  Render free service `srv-dav1q28jo6nc73f47k80`, image `render/Dockerfile`, polls the snapshot from Pages.
- Compute: GitHub Actions `.github/workflows/cycle.yml`. GitHub's cron did NOT fire for 2.5 h, so the
  workflow chains itself: each run computes (~2.5 min), deploys Pages, waits out 9 min and dispatches the
  next run (permission `actions: write`); the cron stays only as a restart. If the chain ever stops:
  `gh workflow run cycle.yml -R Ces107/riua`. Verified 13:09Z -> 13:19Z.
- Control points: all 60 have CAUMAX (CEDEX) return-period flows (`geo/hydro/caumax_points.json`, built by
  `geo/hydro/build_caumax.py` from the local CAUMAX rasters); 42 have a channel capacity. Levels: with
  capacity 2/3 at min(0.25/0.6 Qb, T2/T5), 4 overflow, 5 +1 m over bank; without: T2/T5/T25/T100.

## First real cycle (09:42Z)
All sources OK: radar OPERA+AEMET (13 frames), 413 gauges, AROME-HD/AROME/ICON-EU/ARPEGE/IFS lagged runs,
ECMWF ENS 50 members, pysteps STEPS 20 members, 43 AEMET warnings. Levels 4–5 over Castellón coincide with the
AEMET red warnings in force.

## Calibration (2026-10-01 evening)
- Hindcast run on 7 cases (28 Oct-14 Nov 2024, Mar/Sep/Dec 2025): `hindcast/run.py` -> `hindcast/results.json`;
  blocks cached in `hindcast/cache/blocks_*.pkl`; pages from `hindcast/make_pages.py`.
- Tuned and deployed (`backend/riua/params.json`, policy floor40 chosen by the owner after seeing the trade-off):
  sigma/bias now 0.2/0.9, 48 h 0.5/1.35, long 1.0/1.6; tau now 45/40/40/55, 48 h and long 40 everywhere.
  Other policies: `python hindcast/deploy_params.py tuned|middle|floor40` (needs the WSL venv), then make_pages + push.
- Event = "level reached within 12 km of the cell" (6 km for now), same for forecast and truth.
- Truth: OPERA archive + gauges with a gauge ceiling (1.5 x nearest gauges + 10 mm); scoring only inside the CV.
  The 2026 dates and Apr 2024 have no radar archive. 29 Oct 2024 hourly timing is unreliable (radar attenuation).
- Poyo 29 Oct 2024 with observed rain: 2472 m3/s at the A-3 gauge (measured 2283), 3205 at Paiporta.

## Still open
- More cases would firm up levels 4-5 (Open-Meteo quota: ~1000 calls per case, 10,000/day).
- ERA5 climate / EFI not built (c1 agent unfinished). MetPy drivers run live and show as "Atmósfera".
- Browser extension was disconnected: visual checks were done with headless Chrome screenshots (`scratch/shots`).
