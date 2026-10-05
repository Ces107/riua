"""Sub-hourly radar feature for the calibration days: one npz per UTC day with the excess ladder of every hour.

    python hindcast/subhourly/features.py --chunk K/N [--out DIR] [--no-advection]   (the days of hindcast/subhourly/days.json)
    python hindcast/subhourly/features.py 2019-09-12 2019-09-13 ...

Output <out>/<yyyymmdd>.npz: t_end (24,) datetime64[h] UTC hour ending, ladder (24, L, 68, 64) float16 mm (RAW radar,
riua.radar.qpe.burst_ladder, NaN = no radar), cov (24,) share of the hour bracketed by scans (the ladder is already
divided by it; hours under 60 % are NaN), u (L,) the ladder rates. Frames come from the OPERA archive through
hindcast/build_truth.py (1 km / 10 min from July 2024, 2 km / 15 min before), at the scan times, as the hindcast rain.

Runs on GitHub Actions (.github/workflows/subhourly.yml): public data only, nothing is committed.
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "backend"))
import archive  # noqa: E402
from riua.radar import qpe  # noqa: E402

# workstation: reuse the dBZ days q4 and q9 already downloaded (same format)
LOCAL = [ROOT / "hindcast" / "floods" / "cache" / "radar"]

if "--no-advection" in sys.argv:          # no pysteps on this machine: plain interpolation between scans
    qpe.motion_field = lambda rates, coarse=2: np.zeros((2, *rates[-1].shape))
    qpe._advect = lambda field, v, frac: field


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def day_feature(day: datetime, next_too: bool) -> dict:
    t1 = day + (timedelta(hours=24) if next_too else timedelta(hours=23, minutes=59))
    t_dl = time.time()
    frames = archive.day_frames(day, t1, LOCAL)
    t_dl = time.time() - t_dl
    L = len(qpe.LADDER_U)
    lad = np.full((24, L, 68, 64), np.nan, np.float32)
    cov = np.zeros(24)
    for h in range(24):
        a, b = day + timedelta(hours=h), day + timedelta(hours=h + 1)
        sub = [f for f in frames if a - timedelta(minutes=10) <= f[0] <= b]
        if len(sub) < 2:
            continue
        x, c = qpe.burst_ladder(sub, a, b)
        cov[h] = c
        if c >= 0.6:
            lad[h] = x / c
    t_end = np.array([np.datetime64(day.replace(tzinfo=None) + timedelta(hours=h + 1), "h") for h in range(24)])
    return dict(t_end=t_end, ladder=lad.astype(np.float16), cov=cov, u=qpe.LADDER_U, n_frames=len(frames), t_download=t_dl)


def main():
    out = Path(arg("--out", str(ROOT / "hindcast" / "cache" / "subhourly")))
    out.mkdir(parents=True, exist_ok=True)
    if arg("--raw"):            # where downloaded days are kept (default hindcast/cache/radar)
        archive.RAW = Path(arg("--raw"))
    days = [a for a in sys.argv[1:] if len(a) == 10 and a[4] == "-"]
    if "--chunk" in sys.argv:
        k, n = map(int, arg("--chunk").split("/"))
        alld = json.loads((HERE / "days.json").read_text(encoding="utf-8"))["days"]
        # contiguous blocks: a day and the next one usually share a job (the last hour needs the next day's 00:00 scan)
        size = -(-len(alld) // n)
        days = alld[k * size:(k + 1) * size]
    print(f"{len(days)} days -> {out}", flush=True)
    for i, d in enumerate(days):
        f = out / f"{d.replace('-', '')}.npz"
        if f.exists():
            continue
        day = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
        nxt = (day + timedelta(days=1)).strftime("%Y-%m-%d") in days
        t = time.time()
        try:
            r = day_feature(day, nxt)
        except Exception as e:  # noqa: BLE001  one bad day must not lose the chunk
            print(f"  {d}: FAILED {type(e).__name__}: {e}"[:300], flush=True)
            continue
        np.savez_compressed(f, t_end=r["t_end"], ladder=r["ladder"], cov=r["cov"], u=r["u"])
        lad = r["ladder"].astype(np.float32)
        top = np.nanmax(np.nan_to_num(lad[:, 0])) if np.isfinite(lad).any() else 0.0
        print(f"  {d}: {r['n_frames']} frames, hours {int((r['cov'] >= 0.6).sum())}/24, max cell-hour {top:.1f} mm, "
              f"max X(40) {np.nanmax(np.nan_to_num(lad[:, 5])):.2f} mm, download {r['t_download']:.0f} s, total {time.time() - t:.0f} s "
              f"[{i + 1}/{len(days)}]", flush=True)


if __name__ == "__main__":
    main()
