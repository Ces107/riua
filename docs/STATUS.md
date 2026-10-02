# Status — 2 October 2026

Live: https://ces107.github.io/riua/ · recomputed every ~10 min by `.github/workflows/cycle.yml` (about 100–200 s per cycle).
The page reads only GitHub Pages. The Render API (https://riua-api.onrender.com) mirrors the snapshot and is not used by the page.

## What the product computes today

- **Rain already fallen**: OPERA radar + AEMET composite, conditional merging with ~400 gauges after quality control
  (`radar/qpe.py`), 1-km state for the last 12 h, measured 12-h amount = largest total of one pixel.
- **Scenarios**: 0–6 h: 20 pysteps STEPS members, each blended into an AROME run and carrying that run's weight, plus
  ICON-EU, ARPEGE, IFS and AROME-PI. 6–48 h: lagged AROME-HD / AROME / ICON-EU / ARPEGE / IFS 9 km + ECMWF ENS (50,
  3-hourly) + ICON-EU-EPS (40) + AROME-IFS. Days 2–7: ECMWF ENS (50) + IFS.
- **Probability**: scenario dressing (`core/risk.py`), separate kernel for the 1-h criterion a day ahead; rain already
  measured counts in a 12-h amount only while the scenario still brings rain (`obs_gate_mm`).
- **Level**: highest L with P ≥ τ. τ: 0–6 h 40/40/40/50 %, 6–48 h 20/20/25/40 %, days 2–7 25/40/40/40 % and capped at 3.
- **Ravines** (60 control points): losses fitted to 200 measured floods at 26 gauged catchments
  (`hindcast/calibrate_hydro.py`, `geo/hydro/loss_params.json`), time–area routing, channel capacity or return periods.

## Known weaknesses (most important first)

1. **The hindcast does not score what production computes** (14 differences, `coord/findings/q2-audit.md` F9) and held
   only rain episodes. The validation page therefore overstates the skill, and σ, bias, τ and the family weights of
   the new ensemble members are not validated. Being rebuilt (`q8-verify`) on 51 cases / 207 days incl. quiet days.
2. Days 2–7: no demonstrable skill above level 2–3 outside the 2024 DANA.
3. A cell can show a level while its own expected rain is small: the level refers to rain within 6 / 12 km.
4. Ravines: peak timing arrives a median 3.5 h early for ordinary floods; transmission losses are not modelled; the
   2024 rain analysis is 3–5 h late over the Poyo, so routing cannot be tuned on it.
5. ICON-EU-EPS, AROME-IFS and AROME-PI have no archive: they cannot be hindcast-validated (archive `state/extra`).
6. The rain-analysis validation rests on one event (1–2 Oct 2026).

## Where things are

- Findings of the quality pass: `coord/findings/q1-qpe.md … q8-verify.md` (not in git).
- Parameters: `backend/riua/params.py` (defaults) + `params.json` (written by `hindcast/deploy_params.py`).
- Pages generated from results: `py -3.11 hindcast/make_pages.py`.
