# Riuà — status (2026-10-01, 12:00 local)

## Live
- Web: https://ces107.github.io/riua/ (branch `gh-pages`, published by `deploy/publish.sh`).
- API: https://riua-api.onrender.com (`/healthz`, `/v1/status`, `/v1/snapshot`, `/v1/point?lat=&lon=`, `/docs`);
  Render free service `srv-dav1q28jo6nc73f47k80`, image `render/Dockerfile`, polls the snapshot from Pages.
- Compute: scheduled task "Riua-ciclo" on this PC (at logon, hidden) runs `deploy/loop.sh` in WSL Ubuntu:
  every 10 min `python -m riua.product` (~2.5 min) then publish. Log: WSL `/root/riua-loop.log`.
  Stops when the PC is off. Remove with `Unregister-ScheduledTask Riua-ciclo`.
- Cloud cron instead of the PC: run `gh auth refresh -s workflow`, copy `deploy/github-actions-cycle.yml`
  to `.github/workflows/cycle.yml`, switch Pages to build_type=workflow, push, then disable the scheduled task.

## First real cycle (09:42Z)
All sources OK: radar OPERA+AEMET (13 frames), 413 gauges, AROME-HD/AROME/ICON-EU/ARPEGE/IFS lagged runs,
ECMWF ENS 50 members, pysteps STEPS 20 members, 43 AEMET warnings. Levels 4–5 over Castellón coincide with the
AEMET red warnings in force.

## Not done / known gaps
- Calibration (EMOS coefficients) and tau are untuned defaults: the page says "En pruebas". Hindcast run +
  tuning not done (`hindcast/run.py` missing; truth build stopped for memory).
- Channel sections: 25 of 60 control points have capacity/rating; the rest show discharge only.
- MetPy ingredients module and ERA5 climate/EFI were cut off mid-way (drivers block present but unreviewed).
- README not written.
