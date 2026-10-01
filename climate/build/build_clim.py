"""Products A and E: day-of-year climatologies from ERA5 1991-2020.

    python climate/build/build_clim.py precip      -> scratch/c1-climate/parts/clim_precip.npz
    python climate/build/build_clim.py tcwv        -> scratch/c1-climate/parts/clim_tcwv.npz

A. Precipitation quantiles for EFI/SOT. Rolling 12/24/48/72-h sums of hourly ERA5 precipitation, sampled at
   window END times 00/06/12/18 UTC (4 per day, overlapping for the longer windows). A sample belongs to the day
   of year of the window CENTRE. For every 5th day of year (nodes 0, 5, ..., 360 on a 366-day calendar, 0 = 1 Jan,
   59 = 29 Feb) all samples within +-15 days over 1991-2020 are pooled (30 yr x 31 d x 4 = 3 720 values) and the
   quantiles at CLIM_P are taken per grid point, plus the pooled maximum.
E. TCWV mean and standard deviation of hourly values for each of 366 days of year (+-15 days, all hours pooled).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from common import CLIM_P, PARTS, doy366, hour_index, precip, rolling_sum, safe_savez, tcwv

REF0, REF1 = 1991, 2020
WINDOWS = (12, 24, 48, 72)
HALF = 15
NODES = np.arange(0, 365, 5)        # 73 nodes: 0, 5, ..., 360
Q_SCALE = 20.0                      # stored uint16 = mm * 20 (0.05 mm)


def circ_dist(a, b, period=366):
    d = np.abs(np.asarray(a) - b) % period
    return np.minimum(d, period - d)


def build_precip():
    pr, lat, lon = precip()
    a, b = hour_index(REF0), hour_index(REF1 + 1)
    spin = max(WINDOWS)
    x = np.asarray(pr[a - spin:b]).astype(np.int32)           # [T, 15, 15] tenths of mm, ~60 MB
    assert (x != 32767).all()
    ends = np.arange(spin, x.shape[0], 6)                     # local index of window ends at 00/06/12/18 UTC
    assert (a - spin + ends[0]) % 24 == 0
    out = {}
    for w in WINDOWS:
        t0 = time.time()
        rs = rolling_sum(x, w)[ends]                          # [N, 15, 15] tenths of mm
        centre_h = (a - spin) + ends - w // 2
        doy = doy366(centre_h)
        q = np.zeros((len(NODES), len(CLIM_P), len(lat), len(lon)), np.float32)
        mx = np.zeros((len(NODES), len(lat), len(lon)), np.float32)
        nsamp = []
        for k, d in enumerate(NODES):
            sel = circ_dist(doy, d) <= HALF
            nsamp.append(int(sel.sum()))
            s = rs[sel] / 10.0
            q[k] = np.quantile(s, CLIM_P, axis=0)
            mx[k] = s.max(axis=0)
        assert q.max() * Q_SCALE < 65535 and mx.max() * Q_SCALE < 65535
        out[f"pr_q{w}"] = np.round(q * Q_SCALE).astype(np.uint16)
        out[f"pr_max{w}"] = np.round(mx * Q_SCALE).astype(np.uint16)
        print(f"window {w} h: samples per node {min(nsamp)}..{max(nsamp)}, max quantile {q.max():.1f} mm, "
              f"pooled max {mx.max():.1f} mm, {time.time() - t0:.0f}s", flush=True)
    safe_savez(PARTS / "clim_precip.npz", lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        clim_p=CLIM_P, clim_doy=NODES.astype(np.int16), pr_q_scale=Q_SCALE, **out)
    print("saved", PARTS / "clim_precip.npz")


def build_tcwv():
    tc, lat, lon = tcwv(REF0, REF1)
    ny = REF1 - REF0 + 1
    s1 = np.zeros((366, len(lat), len(lon)))
    s2 = np.zeros_like(s1)
    cnt = np.zeros(366)
    hs1 = np.zeros((24, 366))           # box-mean diurnal cycle check
    for y in range(REF0, REF1 + 1):
        a, b = hour_index(y, y0=REF0), hour_index(y + 1, y0=REF0)
        x = np.asarray(tc[a:b]).astype(np.float64) / 100.0
        assert (x < 600).all(), "missing TCWV values"
        doy = doy366(np.arange(a, b), y0=REF0)
        for d in np.unique(doy):
            sel = doy == d
            s1[d] += x[sel].sum(axis=0)
            s2[d] += (x[sel] ** 2).sum(axis=0)
            cnt[d] += sel.sum()
        hh = np.arange(a, b) % 24
        np.add.at(hs1, (hh, doy), x.mean(axis=(1, 2)))
    mean = np.zeros_like(s1)
    std = np.zeros_like(s1)
    for d in range(366):
        idx = (d + np.arange(-HALF, HALF + 1)) % 366
        n = cnt[idx].sum()
        m = s1[idx].sum(axis=0) / n
        mean[d] = m
        std[d] = np.sqrt(np.maximum(s2[idx].sum(axis=0) / n - m * m, 0.0))
    # diurnal amplitude of the box mean (to justify pooling the hours)
    diurnal = hs1.sum(axis=1) / cnt.sum() * 24
    print(f"TCWV: {ny} years, hours per doy {cnt.min():.0f}..{cnt.max():.0f}; mean range {mean.min():.1f}..{mean.max():.1f}, "
          f"std range {std.min():.2f}..{std.max():.2f}; box-mean by hour of day min {diurnal.min():.2f} max {diurnal.max():.2f} kg/m2")
    safe_savez(PARTS / "clim_tcwv.npz", tcwv_lat=lat.astype(np.float32), tcwv_lon=lon.astype(np.float32),
                        tcwv_mean=mean.astype(np.float32), tcwv_std=std.astype(np.float32),
                        tcwv_diurnal_boxmean=diurnal.astype(np.float32))
    print("saved", PARTS / "clim_tcwv.npz")


def build_monthly():
    """Mean monthly and annual totals 1991-2020 (sanity values for verify.py) and the ERA5 fields of 29-Oct-2024."""
    import calendar

    import fetch_era5

    pr, lat, lon = precip()
    mon = np.zeros((12, len(lat), len(lon)))
    for y in range(REF0, REF1 + 1):
        for m in range(12):
            a = hour_index(y, m + 1)
            b = a + calendar.monthrange(y, m + 1)[1] * 24
            mon[m] += np.asarray(pr[a:b]).astype(np.int64).sum(axis=0) / 10.0
    mon /= (REF1 - REF0 + 1)
    # demo fields: the ERA5 "analysis" of the 29 October 2024 event in the units of the climatology
    demo = {}
    for name, (d0, h0, w) in {"demo_20241029_pr24": ((2024, 10, 30), 0, 24), "demo_20241029_pr12": ((2024, 10, 30), 0, 12),
                              "demo_20241029_pr48": ((2024, 10, 30), 12, 48), "demo_20241029_pr72": ((2024, 10, 31), 0, 72)}.items():
        e = hour_index(*d0, h0)                       # window END (value at index t = hour ending at t)
        demo[name] = (np.asarray(pr[e - w + 1:e + 1]).astype(np.int64).sum(axis=0) / 10.0).astype(np.float32)
    z = np.load(fetch_era5.CACHE / "total_column_integrated_water_vapour" / "chunk_953.npz")
    t = (hour_index(2024, 10, 29, 12) + (calendar.timegm((1940, 1, 1, 0, 0, 0)) // 3600)) - 953 * fetch_era5.CHUNK_H
    demo["demo_20241029_tcwv_12utc"] = (z["data"][:, :, t] / float(z["scale"])).astype(np.float32)
    safe_savez(PARTS / "clim_monthly.npz", pr_monthly_mean=mon.astype(np.float32), **demo)
    print("monthly box mean:", np.round(mon.mean(axis=(1, 2)), 1), "annual", round(float(mon.sum(axis=0).mean()), 1))
    print({k: (float(v.max()), float(v.mean())) for k, v in demo.items()})


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("monthly", "all"):
        build_monthly()
    if what in ("precip", "all"):
        build_precip()
    if what in ("tcwv", "all"):
        build_tcwv()
