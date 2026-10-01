"""Shared helpers of the climate build: consolidated ERA5 arrays (memory-light), time axis, rolling sums.

The machine is shared (7 GB RAM): precipitation is kept as int16 tenths of a millimetre in one memory-mapped
file and every heavy computation loops over grid points or years.
"""
from __future__ import annotations

import calendar
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scratch" / "c1-climate"
STATIC = ROOT / "backend" / "riua" / "static"
PARTS = SCRATCH / "parts"          # intermediate products, one npz per build step
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_era5  # noqa: E402

EPOCH = dt.datetime(1970, 1, 1)
Y0, Y1 = 1940, 2026                # consolidated precipitation record (2026 partial)
CLIM_P = np.concatenate([np.arange(1, 100) / 100.0, [0.995, 0.999]])


def hour_index(y: int, m: int = 1, d: int = 1, hh: int = 0, y0: int = Y0) -> int:
    """Index on the hourly axis that starts at 1 Jan ``y0`` 00 UTC."""
    return (calendar.timegm((y, m, d, hh, 0, 0)) - calendar.timegm((y0, 1, 1, 0, 0, 0))) // 3600


def index_time(i: int, y0: int = Y0) -> dt.datetime:
    return dt.datetime(y0, 1, 1) + dt.timedelta(hours=int(i))


def consolidate(variable: str, y0: int, y1: int, scale: float, dtype) -> Path:
    """Write <variable>_<y0>_<y1>.npy = scaled integers [T, lat, lon] (+ .json meta). Missing -> max int."""
    out = SCRATCH / f"{variable}_{y0}_{y1}.npy"
    if out.exists():
        return out
    n = hour_index(y1 + 1, y0=y0)
    _a, _b, _c, _d, lat, lon = fetch_era5.box_index(variable)
    miss = np.iinfo(dtype).max
    arr = np.lib.format.open_memmap(out.with_suffix(".tmp.npy"), mode="w+", dtype=dtype, shape=(n, len(lat), len(lon)))
    arr[:] = miss
    for name in fetch_era5.names_for(variable, y0, y1):
        f = fetch_era5.CACHE / variable / f"{name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        d = z["data"]
        fscale = float(z["scale"])
        if fscale != scale:
            d = np.round(d.astype(np.float64) / fscale * scale)
        d = d.astype(dtype)
        if z["nan"].size:
            d[np.unpackbits(z["nan"])[:d.size].reshape(d.shape).astype(bool)] = miss
        kind, num = name.split("_")
        s = hour_index(int(num), y0=y0) if kind == "year" else int(num) * fetch_era5.CHUNK_H - calendar.timegm((y0, 1, 1, 0, 0, 0)) // 3600
        a, b = max(s, 0), min(s + d.shape[2], n)
        if b <= a:
            continue
        piece = np.moveaxis(d[:, :, a - s:b - s], 2, 0)
        tgt = arr[a:b]
        ok = piece != miss
        tgt[ok] = piece[ok]
        arr[a:b] = tgt
    arr.flush()
    del arr
    out.with_suffix(".tmp.npy").replace(out)
    return out


def precip() -> tuple[np.memmap, np.ndarray, np.ndarray]:
    """(int16 tenths of mm [T, 15, 15] from 1940-01-01 00 UTC, lat, lon). 32767 = missing."""
    f = consolidate("precipitation", Y0, Y1, 10.0, np.int16)
    _a, _b, _c, _d, lat, lon = fetch_era5.box_index("precipitation")
    return np.load(f, mmap_mode="r"), lat, lon


def tcwv(y0: int = 1991, y1: int = 2020):
    """(uint16 hundredths of kg/m2 [T, 19, 25] from 1 Jan y0, lat, lon). 65535 = missing."""
    v = "total_column_integrated_water_vapour"
    f = consolidate(v, y0, y1, 100.0, np.uint16)
    _a, _b, _c, _d, lat, lon = fetch_era5.box_index(v)
    return np.load(f, mmap_mode="r"), lat, lon


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing w-step sum along axis 0 (value at t = sum of x[t-w+1 .. t]); first w-1 entries = -1.
    Integer in, int32 out; the caller must have replaced missing values."""
    c = np.cumsum(x, axis=0, dtype=np.int64)
    out = np.full(x.shape, -1, np.int32)
    out[w - 1] = c[w - 1]
    out[w:] = c[w:] - c[:-w]
    return out


def doy366(times_h: np.ndarray, y0: int = Y0) -> np.ndarray:
    """0-based day of year on a 366-day calendar (29 Feb = 59, 1 Mar = 60 every year) for hour indices."""
    base = np.datetime64(f"{y0}-01-01T00", "h")
    t = base + np.asarray(times_h).astype("timedelta64[h]")
    days = t.astype("datetime64[D]")
    years = days.astype("datetime64[Y]")
    doy = (days - years.astype("datetime64[D]")).astype(int)
    yr = years.astype(int) + 1970
    leap = ((yr % 4 == 0) & (yr % 100 != 0)) | (yr % 400 == 0)
    return np.where(~leap & (doy >= 59), doy + 1, doy)


def cell(lat: np.ndarray, lon: np.ndarray, la: float, lo: float) -> tuple[int, int]:
    return int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))
