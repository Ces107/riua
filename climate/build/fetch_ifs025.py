"""Operational IFS 0.25 deg precipitation (Open-Meteo S3 ``data/ecmwf_ifs025``, 3-hourly, from 2024-01-25) for the
same 15 x 15 box as the ERA5 climate. Purpose: measure how much wetter the operational model (9 km native,
regridded to 0.25 deg; the same grid the ENS open data come on) is than ERA5 (31 km native) in the tail.

    python climate/build/fetch_ifs025.py        -> scratch/c1-climate/ifs025_precip.npz

The S3 series is the stitched "day 0" one: every 3-h value comes from the latest run (lead 0-6 h).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from riua.sources.openmeteo_s3 import RangeHTTP  # noqa: E402
import fetch_era5  # noqa: E402

S3 = "https://openmeteo.s3.amazonaws.com/data/ecmwf_ifs025/precipitation"
CACHE = ROOT / "scratch" / "c1-climate" / "ifs025"
L, DT = 104, 3          # chunk length (steps) and step (h)
FIRST, LAST = 1519, 1596


def fetch(n: int):
    import omfiles

    out = CACHE / f"chunk_{n}.npy"
    if out.exists():
        return n, 0
    y0, y1, x0, x1, _lat, _lon = fetch_era5.box_index("precipitation")
    fs = RangeHTTP(tail_bytes=1 << 19, read_ahead=96 * 1024, timeout=120, retries=5)
    try:
        rd = omfiles.OmFileReader.from_fsspec(fs, f"{S3}/chunk_{n}.om")
        if tuple(rd.shape) == (721 * 1440, L):      # older chunks: flat [cell, time], row-major from the SW corner
            box = np.stack([np.asarray(rd[y * 1440 + x0:y * 1440 + x1 + 1, :], np.float32) for y in range(y0, y1 + 1)])
        else:                                        # newer chunks: [lat, lon, time]
            assert tuple(rd.shape) == (721, 1440, L), rd.shape
            box = np.asarray(rd[y0:y1 + 1, x0:x1 + 1, :], np.float32)
        rd.close()
    except FileNotFoundError:
        return n, -1
    finally:
        fs.close()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(out, box)
    return n, fs.n_bytes


def main():
    total = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, nb in pool.map(fetch, range(FIRST, LAST + 1)):
            if nb > 0:
                total += nb
            print(f"chunk {n}: {nb / 1e6:.2f} MB (total {total / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)
    pieces, hours = [], []
    for n in range(FIRST, LAST + 1):
        f = CACHE / f"chunk_{n}.npy"
        if f.exists():
            pieces.append(np.load(f))
            hours.append(n * L * DT + np.arange(L) * DT)
    data = np.moveaxis(np.concatenate(pieces, axis=2), 2, 0)
    hours = np.concatenate(hours)                      # hours since 1970 = END of each 3-h step
    _a, _b, _c, _d, lat, lon = fetch_era5.box_index("precipitation")
    np.savez_compressed(ROOT / "scratch" / "c1-climate" / "ifs025_precip.npz", data=data, hours=hours, lat=lat, lon=lon)
    ok = np.isfinite(data).all(axis=(1, 2))
    print(f"saved: {data.shape}, valid steps {ok.sum()}, first valid h {hours[ok][0]}, last {hours[ok][-1]}, max 3 h {np.nanmax(data):.1f}")


if __name__ == "__main__":
    main()
