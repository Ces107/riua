"""Observed rainfall for the hindcast cases: radar QPE merged with gauges.

For every case in cases.json:
  1. OPERA archive reflectivity composites (1 km, every 10 min) over the box,
  2. quality control, Steiner convective/stratiform Z-R, advection-corrected hourly
     accumulation (riua.radar.qpe),
  3. per civil day, the radar total is merged with the gauge totals of that day
     (wradlib AdjustMixed; AVAMET + SAIH stations) and the resulting correction field is
     applied to every hour of the day (the radar keeps the timing, the gauges fix the amount),
  4. hourly fields are reduced to the 0.05 deg analysis grid as cell maximum and cell mean.

Output: hindcast/truth/<case>.npz with
  t_end (T,) datetime64[h] UTC, o_max (T, 68, 64), o_mean (T, 68, 64) mm,
  day_total_1km (D, 340, 320) adjusted civil-day totals, days, meta (json string).

Run inside the Linux environment (pysteps, wradlib):
  python hindcast/build_truth.py [case_id ...]
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

from riua.radar import nowcast, qpe  # noqa: E402
from riua.sources import radar  # noqa: E402

MAD = ZoneInfo("Europe/Madrid")
OUT = ROOT / "hindcast" / "truth"
RAW = ROOT / "hindcast" / "cache" / "radar"


_CL = ROOT / "hindcast" / "cache" / "clutter.npy"
CLUTTER = np.load(_CL) if _CL.exists() else None      # pixels with echo on a dry day (2026-07-20): ground clutter


def gauge_ceiling(glat, glon, gmm, radius_px: int = 15) -> np.ndarray | None:
    """Upper bound for the merged field: 1.5 x the largest gauge within ~15 km, plus 10 mm.
    Where radar has an echo that no gauge within 15 km supports, the excess is clutter or hail."""
    from scipy import ndimage
    if len(gmm) < 50:
        return None
    g = np.zeros((qpe.RG_NY, qpe.RG_NX), np.float32)
    j = np.floor((glat - qpe.RG_LAT0) / qpe.RG_D).astype(int)
    i = np.floor((glon - qpe.RG_LON0) / qpe.RG_D).astype(int)
    ok = (j >= 0) & (j < qpe.RG_NY) & (i >= 0) & (i < qpe.RG_NX)
    np.maximum.at(g, (j[ok], i[ok]), gmm[ok].astype(np.float32))
    has = np.zeros_like(g); has[j[ok], i[ok]] = 1.0
    near = ndimage.maximum_filter(g, size=2 * radius_px + 1)
    cover = ndimage.maximum_filter(has, size=2 * radius_px + 1) > 0
    return np.where(cover, 1.5 * near + 10.0, np.inf).astype(np.float32)


def load_gauges(day: str):
    """Gauge totals for a civil day (rows whose period is 00-24 local). Returns lat, lon, mm."""
    f = ROOT / "hindcast" / "obs" / f"{day}.csv"
    lat, lon, mm = [], [], []
    if not f.exists():
        return np.array(lat), np.array(lon), np.array(mm)
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r["lat"] or not r["lon"] or r["precip_24h_mm"] in ("", None):
                continue
            if not r["period_start"].endswith("T00:00"):
                continue   # 08-08 reports do not match the civil day
            try:
                v = float(r["precip_24h_mm"])
            except ValueError:
                continue
            if 0 <= v < 1200:
                lat.append(float(r["lat"])); lon.append(float(r["lon"])); mm.append(v)
    return np.array(lat), np.array(lon), np.array(mm)


def day_frames(t0: datetime, t1: datetime):
    """Rain-rate frames in [t0 - 10 min, t1], cached on disk as compressed dBZ."""
    RAW.mkdir(parents=True, exist_ok=True)
    out = []
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= t1:
        cache = RAW / f"{day:%Y%m%d}.npz"
        if cache.exists():
            z = np.load(cache)
            times = [datetime.fromtimestamp(int(s), timezone.utc) for s in z["t"]]
            dbz = z["dbz"].astype(np.float32) / 2.0 - 32.0
            dbz[z["dbz"] == 255] = np.nan
        else:
            fr = radar.fetch_opera_archive(day, day + timedelta(hours=23, minutes=59), step_min=10)
            times = [t for t, _ in fr]
            dbz = np.stack([d for _, d in fr]) if fr else np.zeros((0, qpe.RG_NY, qpe.RG_NX), np.float32)
            code = np.clip(np.rint((np.nan_to_num(dbz, nan=0) + 32.0) * 2.0), 0, 254).astype(np.uint8)
            code[~np.isfinite(dbz)] = 255
            np.savez_compressed(cache, t=np.array([int(t.timestamp()) for t in times]), dbz=code)
        for t, d in zip(times, dbz):
            if t0 - timedelta(minutes=10) <= t <= t1:
                if CLUTTER is not None:
                    d = np.where(CLUTTER & np.isfinite(d), qpe.NO_ECHO_DBZ, d)
                out.append((t, qpe.rain_rate(qpe.despeckle(d))))
        day += timedelta(days=1)
    return out


def build(case: dict) -> None:
    days = case["days"]
    d0 = datetime.fromisoformat(days[0]).replace(tzinfo=MAD)
    d1 = datetime.fromisoformat(days[-1]).replace(tzinfo=MAD) + timedelta(days=1)
    u0, u1 = d0.astimezone(timezone.utc), d1.astimezone(timezone.utc)
    frames = day_frames(u0, u1)
    hours = int((u1 - u0).total_seconds() // 3600)
    acc = np.full((hours, qpe.RG_NY, qpe.RG_NX), np.nan, np.float32)
    cov = np.zeros(hours)
    for h in range(hours):
        a, b = u0 + timedelta(hours=h), u0 + timedelta(hours=h + 1)
        sub = [f for f in frames if a - timedelta(minutes=10) <= f[0] <= b]
        if len(sub) >= 2:
            acc[h], cov[h] = qpe.accumulate(sub, a, b)
    meta = {"case": case["id"], "days": days, "hour_coverage_mean": float(cov.mean()), "gauge": {}}
    adj = acc.copy()
    day_tot = []
    for k, day in enumerate(days):
        la = datetime.fromisoformat(day).replace(tzinfo=MAD).astimezone(timezone.utc)
        h0 = int((la - u0).total_seconds() // 3600)
        h1 = min(h0 + int(((la.astimezone(MAD) + timedelta(days=1)).astimezone(timezone.utc) - la).total_seconds() // 3600), hours)
        raw_day = np.nansum(acc[h0:h1], axis=0)
        raw_day[np.isnan(acc[h0:h1]).all(axis=0)] = np.nan
        glat, glon, gmm = load_gauges(day)
        merged, info = qpe.merge_gauges(raw_day, glat, glon, gmm)
        ceil = gauge_ceiling(glat, glon, gmm)
        if ceil is not None:
            info["capped_px"] = int((np.nan_to_num(merged) > ceil).sum())
            merged = np.minimum(merged, ceil)
        info["gauge_max"] = float(gmm.max()) if gmm.size else None
        info["radar_raw_max"] = float(np.nanmax(raw_day)) if np.isfinite(raw_day).any() else None
        info["merged_max"] = float(np.nanmax(merged)) if np.isfinite(merged).any() else None
        meta["gauge"][day] = info
        day_tot.append(merged)
        if info["method"] == "none":
            continue
        base = np.nan_to_num(raw_day, nan=0.0)
        fac = np.where(base >= 1.0, np.clip(np.nan_to_num(merged, nan=0.0) / np.maximum(base, 1e-6), 0.1, 10.0), 1.0)
        # rain the radar missed entirely: spread the gauge-derived amount with the regional hourly profile
        extra = np.where(base < 1.0, np.maximum(np.nan_to_num(merged, nan=0.0) - base, 0.0), 0.0)
        prof = np.nanmean(np.nan_to_num(acc[h0:h1], nan=0.0), axis=(1, 2))
        prof = prof / prof.sum() if prof.sum() > 0 else np.full(h1 - h0, 1.0 / max(h1 - h0, 1))
        for h in range(h0, h1):
            adj[h] = np.nan_to_num(acc[h], nan=0.0) * fac + extra * prof[h - h0]
    # largest 12-h accumulation at 1 km, then the cell maximum (what a gauge in the cell could have read)
    filled = np.nan_to_num(adj, nan=0.0)
    cs = np.concatenate([np.zeros((1, *filled.shape[1:]), np.float32), np.cumsum(filled, axis=0, dtype=np.float32)])
    idx = np.arange(1, hours + 1)
    r12 = cs[idx] - cs[np.maximum(idx - 12, 0)]
    o12_max = nowcast.to_analysis_max(r12).astype(np.float32)
    del cs, r12
    t_end = np.array([np.datetime64((u0 + timedelta(hours=h + 1)).replace(tzinfo=None), "h") for h in range(hours)])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{case['id']}.npz", t_end=t_end,
        o_max=nowcast.to_analysis_max(np.nan_to_num(adj, nan=0.0)).astype(np.float32),
        o_mean=nowcast.to_analysis_mean(adj).astype(np.float32),
        raw_max=nowcast.to_analysis_max(np.nan_to_num(acc, nan=0.0)).astype(np.float32),
        o12_max=o12_max,
        day_total_1km=np.stack(day_tot).astype(np.float32), days=np.array(days), meta=json.dumps(meta))
    g = meta["gauge"]
    for day in days:
        i = g[day]
        print(f"{case['id']} {day}: gauges {i['n_gauges']} max {i['gauge_max']} | radar raw max {i['radar_raw_max']} | "
              f"merged max {i['merged_max']} | {i['method']} mfb {i['mfb']}", flush=True)


if __name__ == "__main__":
    cases = json.loads((ROOT / "hindcast" / "cases.json").read_text(encoding="utf-8"))["cases"]
    force = "--force" in sys.argv
    want = {a for a in sys.argv[1:] if not a.startswith("--")}
    for c in cases:
        if want and c["id"] not in want:
            continue
        if (OUT / f"{c['id']}.npz").exists() and not force:
            print("skip", c["id"]); continue
        try:
            build(c)
        except Exception as e:  # one bad case must not stop the batch
            print("FAILED", c["id"], type(e).__name__, e, flush=True)
