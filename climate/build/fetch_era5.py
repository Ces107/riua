"""Download the ERA5 box series from Open-Meteo's public S3 bucket (no quota, CC-BY-4.0).

    python climate/build/fetch_era5.py precipitation 1940 2026
    python climate/build/fetch_era5.py total_column_integrated_water_vapour 1991 2020

Layout on S3 (verified 2026-10-01, see coord/findings/c1-climate.md):
    data/copernicus_era5/<variable>/year_YYYY.om    [721, 1440, hours of the year]   chunks (1, 6, 1095)
    data/copernicus_era5/<variable>/chunk_N.om      [721, 1440, 504]                 chunk N = [N*504 h, (N+1)*504 h) since 1970
Grid: 0.25 deg, row 0 = 90S, column 0 = 180W. Values are hourly; precipitation = sum of the hour ENDING at the
time stamp (Open-Meteo convention), mm, stored with 0.1 mm resolution.

Only the file tail and the compressed chunks that intersect the box are fetched with HTTP range requests
(reusing riua.sources.openmeteo_s3.RangeHTTP). One compressed .npz per remote file is cached in
scratch/c1-climate/era5/<variable>/ so that reruns download nothing.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from riua.sources.openmeteo_s3 import RangeHTTP  # noqa: E402

S3 = "https://openmeteo.s3.amazonaws.com/data/copernicus_era5"
CACHE = ROOT / "scratch" / "c1-climate" / "era5"
CHUNK_H = 504
#: variable -> (lat_min, lat_max, lon_min, lon_max, storage scale, storage dtype)
BOXES = {
    "precipitation": (37.5, 41.0, -2.5, 1.0, 10.0, np.int16),
    "total_column_integrated_water_vapour": (37.0, 41.5, -3.0, 3.0, 100.0, np.uint16),
}
FIRST_CHUNK = {"precipitation": 904, "total_column_integrated_water_vapour": 947}
LAST_YEAR_FILE = {"precipitation": 2021, "total_column_integrated_water_vapour": 2023}


def box_index(variable: str):
    la0, la1, lo0, lo1, _s, _d = BOXES[variable]
    y0, y1 = round((la0 + 90) / 0.25), round((la1 + 90) / 0.25)
    x0, x1 = round((lo0 + 180) / 0.25), round((lo1 + 180) / 0.25)
    lat = -90 + 0.25 * np.arange(y0, y1 + 1)
    lon = -180 + 0.25 * np.arange(x0, x1 + 1)
    return y0, y1, x0, x1, lat, lon


def fetch_file(variable: str, name: str) -> tuple[str, int, float]:
    """Fetch one remote file's box -> cached npz. Returns (name, bytes downloaded, seconds)."""
    import omfiles

    out = CACHE / variable / f"{name}.npz"
    if out.exists():
        return name, 0, 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1, lat, lon = box_index(variable)
    _la0, _la1, _lo0, _lo1, scale, dtype = BOXES[variable]
    t = time.time()
    for attempt in range(4):
        fs = RangeHTTP(tail_bytes=1 << 20, read_ahead=96 * 1024, timeout=120, retries=5)
        try:
            rd = omfiles.OmFileReader.from_fsspec(fs, f"{S3}/{variable}/{name}.om")
            assert tuple(rd.shape[:2]) == (721, 1440), rd.shape
            box = np.asarray(rd[y0:y1 + 1, x0:x1 + 1, :], np.float32)
            rd.close()
            break
        except FileNotFoundError:  # chunk not published yet (the archive ends ~5 days before today)
            print(f"  {name}: not on S3 (yet), skipped", flush=True)
            return name, 0, 0.0
        except Exception as exc:  # noqa: BLE001 - network hiccup: retry the whole file
            print(f"  {name}: attempt {attempt + 1} failed: {exc!r}", flush=True)
            time.sleep(5 * (attempt + 1))
        finally:
            fs.close()
    else:
        raise OSError(f"{variable}/{name} failed")
    nan = ~np.isfinite(box)
    packed = np.round(np.where(nan, 0, box) * scale).astype(dtype)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, data=packed, nan=np.packbits(nan) if nan.any() else np.zeros(0, np.uint8),
                        scale=scale, lat=lat, lon=lon)
    tmp.replace(out)
    return name, fs.n_bytes, time.time() - t


def names_for(variable: str, y_first: int, y_last: int) -> list[str]:
    names = [f"year_{y}" for y in range(y_first, min(y_last, LAST_YEAR_FILE[variable]) + 1)]
    if y_last > LAST_YEAR_FILE[variable]:
        import calendar

        t_end = min(calendar.timegm((y_last + 1, 1, 1, 0, 0, 0)), int(time.time())) // 3600
        t_start = calendar.timegm((LAST_YEAR_FILE[variable] + 1, 1, 1, 0, 0, 0)) // 3600
        n0 = max(FIRST_CHUNK[variable], t_start // CHUNK_H)
        names += [f"chunk_{n}" for n in range(n0, (t_end - 1) // CHUNK_H + 1)]
    return names


def load(variable: str, y_first: int, y_last: int):
    """Continuous hourly series from the cache: (hours since 1970 [T], lat, lon, data float32 [T, lat, lon])."""
    import calendar

    h0 = calendar.timegm((y_first, 1, 1, 0, 0, 0)) // 3600
    h1 = calendar.timegm((y_last + 1, 1, 1, 0, 0, 0)) // 3600
    _y0, _y1, _x0, _x1, lat, lon = box_index(variable)
    out = np.full((h1 - h0, len(lat), len(lon)), np.nan, np.float32)
    for name in names_for(variable, y_first, y_last):
        f = CACHE / variable / f"{name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        d = z["data"].astype(np.float32) / float(z["scale"])
        if z["nan"].size:
            d[np.unpackbits(z["nan"])[:d.size].reshape(d.shape).astype(bool)] = np.nan
        kind, num = name.split("_")
        s = calendar.timegm((int(num), 1, 1, 0, 0, 0)) // 3600 if kind == "year" else int(num) * CHUNK_H
        a, b = max(s, h0), min(s + d.shape[2], h1)
        if b <= a:
            continue
        piece = np.moveaxis(d[:, :, a - s:b - s], 2, 0)
        tgt = out[a - h0:b - h0]
        ok = np.isfinite(piece)
        tgt[ok] = piece[ok]
    return np.arange(h0, h1), lat, lon, out


def main():
    variable, y_first, y_last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    names = names_for(variable, y_first, y_last)
    todo = [n for n in names if not (CACHE / variable / f"{n}.npz").exists()]
    print(f"{variable}: {len(names)} files, {len(todo)} to download", flush=True)
    total = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, nb, sec in pool.map(lambda n: fetch_file(variable, n), todo):
            total += nb
            print(f"  {name}: {nb / 1e6:.2f} MB in {sec:.1f}s (total {total / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)
    print(f"DONE {variable}: {total / 1e6:.1f} MB downloaded in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
