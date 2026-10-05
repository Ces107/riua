"""Hourly rain analysis on the Riuà 0.05 deg grid for the historical flood cases (hindcast/floods/cases.json).

rain = "radar":  exactly the chain of hindcast/build_truth.py (OPERA archive composites -> riua.radar.qpe: despeckle,
                 Steiner Z-R, advection-corrected hourly accumulation -> per civil day merged with the gauge totals of
                 that day, the radar keeps the timing). Its functions are imported and only the folders are redirected
                 to hindcast/floods/cache/ (radar/, acc/, rain/); days already in q4's hindcast/cache are read, never written.
rain = "gauges": no radar in the archive (before 2012-09-04). Daily gauge totals kriged on the 1 km grid
                 (qpe.gauges_only), spread over the hours of the civil day with a time profile given in the case
                 ("profile": {day: [24 weights, local hours]}) or, without one, evenly over "window_local" [h0, h1) of the
                 day (default 0-24). LOW CONFIDENCE: the timing and the intensity are assumptions, not measurements.

Output: hindcast/floods/cache/rain/<case>.npz with t_end (UTC, hour ending), o_mean (T, 68, 64) mm, o_max, meta (json).

Run in the Linux environment (pysteps, wradlib):
    python hindcast/floods/build_rain.py [case_id ...] [--force] [--radar-only]
"""
from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "hindcast"))
sys.path.insert(0, str(ROOT / "backend"))
import build_truth as bt  # noqa: E402
from riua.radar import nowcast, qpe  # noqa: E402

CACHE = HERE / "cache"
Q4_RAW, Q4_ACC = bt.RAW, bt.ACC
bt.RAW, bt.ACC, bt.OUT = CACHE / "radar", CACHE / "acc", CACHE / "rain"
OBS = CACHE / "obs"


def load_gauges(day: str):
    """Gauge totals of a civil day from hindcast/floods/cache/obs/<day>.csv (fetch_gauges.py)."""
    f = OBS / f"{day}.csv"
    lat, lon, mm = [], [], []
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r["lat"] or not r["lon"] or r["precip_24h_mm"] in ("", None) or not r["period_start"].endswith("T00:00"):
                    continue
                v = float(r["precip_24h_mm"])
                if 0 <= v < 1200:
                    lat.append(float(r["lat"])); lon.append(float(r["lon"])); mm.append(v)
    return np.array(lat), np.array(lon), np.array(mm)


bt.load_gauges = load_gauges

if "--no-advection" in sys.argv:
    # pysteps not available (2026-10-05 the WSL environment was gone): plain time interpolation between scans,
    # no optical flow. At 0.05 deg cell means and 15-min scans the difference is small; recorded in the meta.
    def _no_motion(rates, coarse=2):
        return np.zeros((2, *rates[-1].shape))
    qpe.motion_field = _no_motion
    qpe._advect = lambda field, v, frac: field


def link_q4(case):
    """A UTC day that q4 already downloaded / accumulated is copied into our cache (their folders stay untouched)."""
    import shutil
    for day_iso in bt.case_utc_hours(case):
        name = day_iso.replace("-", "") + ".npz"
        for src, dst in ((Q4_RAW, bt.RAW), (Q4_ACC, bt.ACC)):
            dst.mkdir(parents=True, exist_ok=True)
            if (src / name).exists() and not (dst / name).exists():
                shutil.copy(src / name, dst / name)


def era5_weights(case, la_utc, nh):
    """(nh, ny, nx) share of the civil day's rain that falls in each hour, from ERA5 (era5_profile.py), or None.
    Where ERA5 has less than 1 mm that day the box-mean profile is used. The hour ending at t+1 takes ERA5's value
    stamped t+1 (ERA5 precipitation of the Open-Meteo mirror is the sum over the preceding hour)."""
    if not case.get("era5"):
        return None
    f = CACHE / "era5" / f"{case['era5'][0]}_{case['era5'][1]}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    t = z["t"].astype("datetime64[h]")
    first = np.datetime64((la_utc + timedelta(hours=1)).replace(tzinfo=None), "h")
    k0 = int((first - t[0]) / np.timedelta64(1, "h"))
    if k0 < 0 or k0 + nh > len(t):
        return None
    p = np.nan_to_num(z["p"][k0:k0 + nh], nan=0.0)                   # (nh, ny5, nx5)
    tot = p.sum(axis=0)
    mean_prof = p.mean(axis=(1, 2))
    mean_prof = mean_prof / mean_prof.sum() if mean_prof.sum() > 0 else np.full(nh, 1.0 / nh)
    w5 = np.where(tot[None] >= 1.0, p / np.maximum(tot[None], 1e-6), mean_prof[:, None, None])
    lon2, lat2 = qpe.radar_mesh()
    lat1, lon1 = lat2[:, 0], lon2[0]
    jj = np.clip(np.rint((lat1 - z["lat"][0]) / 0.25).astype(int), 0, len(z["lat"]) - 1)
    ii = np.clip(np.rint((lon1 - z["lon"][0]) / 0.25).astype(int), 0, len(z["lon"]) - 1)
    return w5[:, jj][:, :, ii].astype(np.float32)


def build_gauges_only(case):
    days = case["days"]
    u0 = datetime.fromisoformat(days[0]).replace(tzinfo=bt.MAD).astimezone(timezone.utc)
    u1 = (datetime.fromisoformat(days[-1]).replace(tzinfo=bt.MAD) + timedelta(days=1)).astimezone(timezone.utc)
    hours = int((u1 - u0).total_seconds() // 3600)
    adj = np.zeros((hours, qpe.RG_NY, qpe.RG_NX), np.float32)
    meta = {"case": case["id"], "days": days, "rain": "gauges", "gauge": {}}
    for day in days:
        la = datetime.fromisoformat(day).replace(tzinfo=bt.MAD).astimezone(timezone.utc)
        h0 = int((la - u0).total_seconds() // 3600)
        glat, glon, gmm = load_gauges(day)
        info = {"n_gauges": int(len(gmm)), "gauge_max": float(gmm.max()) if gmm.size else None, "method": "none"}
        meta["gauge"][day] = info
        if len(gmm) < 5 or gmm.max() < 1.0:
            continue
        field = qpe.gauges_only(glat, glon, gmm)
        info["method"] = "kriging of daily gauge totals (sqrt)"
        info["merged_max"] = float(np.nanmax(field))
        nh = min(24, hours - h0)
        w = era5_weights(case, la, nh)
        if w is not None:
            info["timing"] = "ERA5 hourly share of the civil day, nearest 0.25 deg cell"
            adj[h0:h0 + nh] = field[None] * w
            continue
        prof = np.zeros(24)
        if case.get("profile", {}).get(day):
            prof[:] = case["profile"][day]
        else:
            a, b = case.get("window_local", {}).get(day, [0, 24])
            prof[a:b] = 1.0
        prof /= prof.sum()
        info["timing"] = "fixed profile"
        info["profile"] = [round(float(x), 3) for x in prof]
        for h in range(nh):
            adj[h0 + h] = field * prof[h]
    t_end = np.array([np.datetime64((u0 + timedelta(hours=h + 1)).replace(tzinfo=None), "h") for h in range(hours)])
    bt.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bt.OUT / f"{case['id']}.npz", t_end=t_end,
                        o_max=nowcast.to_analysis_max(adj).astype(np.float32),
                        o_mean=nowcast.to_analysis_mean(adj).astype(np.float32), days=np.array(days), meta=json.dumps(meta))
    for day in days:
        print(case["id"], day, meta["gauge"][day], flush=True)


if __name__ == "__main__":
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    force = "--force" in sys.argv
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    if want:                                            # in the order given on the command line
        cases = [c for w in want for c in cases if c["id"] == w]
    for c in cases:
        if want and c["id"] not in want:
            continue
        if c["rain"] == "truth":
            continue
        if (bt.OUT / f"{c['id']}.npz").exists() and not force:
            print("skip", c["id"], flush=True)
            continue
        t = time.time()
        try:
            if c["rain"] == "gauges":
                build_gauges_only(c)
            else:
                link_q4(c)
                if "--radar-only" in sys.argv:
                    a = datetime.fromisoformat(c["days"][0]).replace(tzinfo=bt.MAD).astimezone(timezone.utc)
                    b = (datetime.fromisoformat(c["days"][-1]).replace(tzinfo=bt.MAD) + timedelta(days=1)).astimezone(timezone.utc)
                    bt.day_frames(a, b, rates=False)
                else:
                    bt.build(c)
            print("done", c["id"], f"{time.time() - t:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001  one bad case must not stop the batch
            import traceback
            traceback.print_exc()
            print("FAILED", c["id"], type(e).__name__, e, flush=True)
