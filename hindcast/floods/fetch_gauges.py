"""Gauge totals per civil day for the historical flood cases (hindcast/floods/cases.json).

Source: the AVAMET MeteoXarxa daily table (https://www.avamet.org/mx-meteoxarxa.php?data=YYYY-MM-DD), civil day
00-24 local. Checked 2026-10-02: 11 stations on 2005-10-12, 119 on 2007-10-12, 179 on 2012-09-28, 342 on 2016-12-18,
542 on 2019-09-12 (the table also carries the AEMET and CHJ stations that AVAMET relays). Licence CC BY-NC-ND:
private validation only -> everything is written under hindcast/floods/cache/ (gitignored), never published.

Output: hindcast/floods/cache/obs/<day>.csv, same columns as hindcast/obs/<day>.csv (hindcast/fetch_obs.py), so that
build_truth.load_gauges-style readers work unchanged.

    py -3.11 hindcast/floods/fetch_gauges.py            # every day of every case
    py -3.11 hindcast/floods/fetch_gauges.py 2019-09    # cases whose id starts with ...
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "hindcast"))
import fetch_obs as fo  # noqa: E402

CACHE = HERE / "cache"
fo.CACHE = CACHE / "http"
fo.OUT = CACHE / "obs"
fo._AVAMET_COORD_FILE = fo.CACHE / "avamet_coords.json"


def main(argv):
    fo.CACHE.mkdir(parents=True, exist_ok=True)
    fo.OUT.mkdir(parents=True, exist_ok=True)
    seed = ROOT / "scratch" / "r4-obs" / "cache" / "avamet_coords.json"      # coordinates already looked up by r4
    if not fo._AVAMET_COORD_FILE.exists() and seed.exists():
        shutil.copy(seed, fo._AVAMET_COORD_FILE)
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    days = sorted({d for c in cases if not argv or any(c["id"].startswith(a) for a in argv) for d in c["days"]})
    for day in days:
        f = fo.OUT / f"{day}.csv"
        if f.exists():
            continue
        try:
            rows = fo.rows_avamet(day, intensity=False, p0_min_mm=1e9, p0_max_stations=0)
        except Exception as e:  # noqa: BLE001
            print(day, "FAILED", type(e).__name__, e, flush=True)
            continue
        rows = [r for r in rows if r["lat"] != ""]
        fo.write_csv(day, rows)
        mx = max(rows, key=lambda r: r["precip_24h_mm"]) if rows else None
        print(day, len(rows), "stations, max", mx and (mx["precip_24h_mm"], mx["name"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
