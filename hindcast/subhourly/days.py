"""UTC days whose sub-hourly radar feature (qpe.burst_ladder) the calibration needs -> hindcast/subhourly/days.json.

* q9's big floods with radar rain (hindcast/floods/cases.json, rain = "radar"): every UTC day of the case.
* q7's ordinary floods and the whole gauge period (Sep 2024 - 2026): every UTC day on which some 5-km cell-hour of the
  rain grid used by calibrate_hydro.py (hindcast/obs/flows/rain_grid.npz) reaches 5 mm/h. The burst term can only
  act in such hours (the ladder starts at 10 mm/h of raw radar), so the false-alarm count over the period is complete.

Run on the workstation (needs the gitignored rain grid): py -3.11 hindcast/subhourly/days.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "hindcast"))
WET_MMH = 5.0


def main():
    sys.path.insert(0, str(HERE))
    import archive
    cases = json.loads((ROOT / "hindcast" / "floods" / "cases.json").read_text(encoding="utf-8"))["cases"]
    big = sorted({d for c in cases if c.get("rain") == "radar" for d in archive.case_utc_hours(c)})
    z = np.load(ROOT / "hindcast" / "obs" / "flows" / "rain_grid.npz")
    t, p = z["t_end"], z["p"]
    mx = np.nanmax(np.where(np.isfinite(p), p, 0.0), axis=1)
    start = (t - np.timedelta64(1, "h")).astype("datetime64[D]")
    ordinary = sorted({str(d) for d, m in zip(start, mx) if m >= WET_MMH})
    days = sorted(set(big) | set(ordinary))
    out = dict(note=__doc__.strip().splitlines()[0], wet_mmh=WET_MMH, n=len(days), big=big, ordinary=ordinary, days=days)
    (HERE / "days.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{len(big)} big-flood days, {len(ordinary)} ordinary days, {len(days)} in all")


if __name__ == "__main__":
    main()
