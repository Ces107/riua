"""Hourly ERA5 precipitation over the box for one year window, from Open-Meteo's public S3 mirror
(data/copernicus_era5/precipitation/year_YYYY.om, 0.25 deg, [lat, lon, hour of year], lat from -90, lon from -180).
Used ONLY to place the daily gauge totals in time for the cases that have no radar (before Sept 2012).

    py -3.11 hindcast/floods/era5_profile.py 2007-10-10 2007-10-14   -> cache/era5/2007-10-10_2007-10-14.npz (t_end UTC, lat, lon, p[T, ny, nx])
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))
from riua.sources import openmeteo_s3 as om  # noqa: E402

BOX = (-2.5, 37.5, 1.0, 41.25)      # west, south, east, north


def fetch(d0: str, d1: str):
    a = datetime.fromisoformat(d0).replace(tzinfo=timezone.utc)
    b = datetime.fromisoformat(d1).replace(tzinfo=timezone.utc) + timedelta(days=1)
    if a.year != b.year and not (b.month == 1 and b.day == 1):
        raise SystemExit("window must stay inside one year")
    url = f"{om.S3_HOST}/data/copernicus_era5/precipitation/year_{a.year}.om"
    root = om._open(url)
    try:
        print("shape", root.shape, "chunks", getattr(root, "chunks", None), flush=True)
        j0, j1 = int(round((BOX[1] + 90) / 0.25)), int(round((BOX[3] + 90) / 0.25)) + 1
        i0, i1 = int(round((BOX[0] + 180) / 0.25)), int(round((BOX[2] + 180) / 0.25)) + 1
        y0 = datetime(a.year, 1, 1, tzinfo=timezone.utc)
        h0, h1 = int((a - y0).total_seconds() // 3600), int((b - y0).total_seconds() // 3600)
        arr = np.asarray(root[j0:j1, i0:i1, h0:h1], np.float32)
    finally:
        root.close()
    lat = -90 + 0.25 * np.arange(j0, j1)
    lon = -180 + 0.25 * np.arange(i0, i1)
    t = np.array([np.datetime64((a + timedelta(hours=k)).replace(tzinfo=None), "h") for k in range(h1 - h0)])
    out = HERE / "cache" / "era5"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{d0}_{d1}.npz", t=t, lat=lat, lon=lon, p=np.moveaxis(arr, 2, 0))
    return t, lat, lon, np.moveaxis(arr, 2, 0)


if __name__ == "__main__":
    t, lat, lon, p = fetch(sys.argv[1], sys.argv[2])
    print(p.shape, "max cell total", float(np.nansum(p, axis=0).max()))
    tot = np.nansum(p, axis=(1, 2))
    for k in range(0, len(t), 3):
        print(t[k], " ".join(f"{x:6.1f}" for x in tot[k:k + 3]))
