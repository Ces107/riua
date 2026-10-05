"""OPERA archive reader for the sub-hourly feature, self-contained so that the GitHub runner needs only this folder
and riua.radar / riua.sources.radar (a copy of the archive functions of hindcast/build_truth.py, q4's file, which is
not committed in this form).

The archive bucket changed format twice (listed 2026-10-02, coord/findings/q4-hindcast.md):
  .. 2024-07-01   OPERA@<t>@0@DBZH_QIND.{h5,tiff}   2 km, every 15 min
  2024-07-01 ..   OPERA@<t>@0@DBZH.{h5,tiff}        1 km, every 5 min (10-min steps used, as production)
  2026-01-01 ..   OPERA@<t>@0@DBZH.h5 only
Frames are cached per UTC day as compressed dBZ (uint8, (dBZ + 32) * 2, 255 = no coverage), the same format as
hindcast/cache/radar, so the workstation can reuse q4's and q9's downloads.
"""
from __future__ import annotations

import re
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from riua.radar import qpe  # noqa: E402
from riua.sources import radar  # noqa: E402

MAD = ZoneInfo("Europe/Madrid")
RAW = ROOT / "hindcast" / "cache" / "radar"


class _HttpFile:
    """Read-only file object over HTTP Range requests with a block cache (what h5py needs to open a file)."""

    def __init__(self, url: str, block: int = 16384):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        r = radar._get(url, headers={"Range": f"bytes=0-{block - 1}"})
        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.cache[0] = r.content

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def tell(self): return self.pos
    def close(self): pass
    def flush(self): pass
    closed = False

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def get(self, a: int, size: int) -> bytes:
        return radar._get(self.url, headers={"Range": f"bytes={a}-{a + size - 1}"}).content

    def _blk(self, k: int) -> bytes:
        if k not in self.cache:
            a = k * self.block
            self.cache[k] = self.get(a, min(self.block, self.size - a))
        return self.cache[k]

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        out = bytearray()
        while len(out) < n:
            k, o = divmod(self.pos + len(out), self.block)
            out += self._blk(k)[o:o + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


_H5_INDEX = {}


def opera_h5_list(day: datetime, quantity: str) -> list[datetime]:
    import requests
    url = f"{radar.OPERA_S3}/{radar.OPERA_BUCKET_ARCHIVE}/?list-type=2&max-keys=1000&prefix={day:%Y/%m/%d}/OPERA/COMP/"
    times, token = [], None
    while True:
        xml = radar._get(url + (f"&continuation-token={requests.utils.quote(token, safe='')}" if token else "")).text
        for m in re.finditer(r"<Key>[^<]*OPERA@(\d{8}T\d{4})@0@" + quantity + r"\.h5</Key>", xml):
            times.append(datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc))
        nxt = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not nxt:
            break
        token = nxt.group(1)
    return sorted(times)


def fetch_opera_h5(t: datetime, quantity: str = "DBZH") -> np.ndarray:
    """One archived ODIM-H5 composite on the 1 km radar grid: dBZ, NO_ECHO_DBZ where the radar saw nothing, NaN where
    there is no coverage (bit-identical to radar.fetch_opera on a date with both formats, q4)."""
    import h5py
    url = radar._opera_url(radar.OPERA_BUCKET_ARCHIVE, t, quantity).replace(".tiff", ".h5")
    f = _HttpFile(url)
    with h5py.File(f, "r") as h:
        d = h["dataset1/data1/data"]
        what = dict(h["dataset1/what"].attrs)
        if "dataset1/data1/what" in h:
            what.update(h["dataset1/data1/what"].attrs)
        if what.get("quantity", b"DBZH") not in (b"DBZH", "DBZH"):
            raise RuntimeError(f"dataset1 is {what.get('quantity')}, not DBZH")
        nodata, undetect = float(what.get("nodata", -9999000.0)), float(what.get("undetect", -8888000.0))
        gain, offset = float(what.get("gain", 1.0)), float(what.get("offset", 0.0))
        xs, ys = float(h["where"].attrs["xscale"]), float(h["where"].attrs["yscale"])
        key = (xs, tuple(d.shape))
        if key not in _H5_INDEX:
            lon2d, lat2d = np.meshgrid(radar.GRID_LON, radar.GRID_LAT)
            x, y = radar.laea_forward(lon2d, lat2d)
            _H5_INDEX[key] = (np.floor((ys / 2 - y) / ys).astype(np.int64), np.floor((x + xs / 2) / xs).astype(np.int64))
        row, col = _H5_INDEX[key]
        ch = d.chunks
        r0, r1, c0, c1 = int(row.min() // ch[0]), int(row.max() // ch[0]), int(col.min() // ch[1]), int(col.max() // ch[1])
        sub = np.full(((r1 - r0 + 1) * ch[0], (c1 - c0 + 1) * ch[1]), nodata, np.float64)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                info = d.id.get_chunk_info_by_coord((rr * ch[0], cc * ch[1]))
                if info.byte_offset is None:
                    continue
                tile = np.frombuffer(zlib.decompress(f.get(info.byte_offset, info.size)), dtype=d.dtype).reshape(ch)
                sub[(rr - r0) * ch[0]:(rr - r0 + 1) * ch[0], (cc - c0) * ch[1]:(cc - c0 + 1) * ch[1]] = tile
    raw = sub[row - r0 * ch[0], col - c0 * ch[1]]
    out = (raw * gain + offset).astype(np.float32)
    out[raw == undetect] = radar.NO_ECHO_DBZ
    out[raw == nodata] = np.nan
    return out


def fetch_radar_day(day: datetime, workers: int = 8) -> list[tuple[datetime, np.ndarray]]:
    """Every archived composite of one UTC day: 10-min steps of the 1 km product (tiff, else h5), completed with the
    15-min 2 km product where the 1 km one does not exist."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(t, "tiff") for t in radar.opera_list(day, "DBZH", radar.OPERA_BUCKET_ARCHIVE) if t.minute % 10 == 0]
    if not jobs:
        jobs = [(t, "DBZH") for t in opera_h5_list(day, "DBZH") if t.minute % 10 == 0]
    first = min((t for t, _ in jobs), default=day + timedelta(days=1))
    if first > day + timedelta(minutes=20):
        jobs += [(t, "DBZH_QIND") for t in opera_h5_list(day, "DBZH_QIND") if t < first]

    def one(job):
        t, kind = job
        err = None
        for _ in range(3):
            try:
                if kind == "tiff":
                    return t, radar.fetch_opera(t, "DBZH", radar.OPERA_BUCKET_ARCHIVE)
                return t, fetch_opera_h5(t, kind)
            except Exception as e:  # noqa: BLE001
                err = e
        print(f"  radar frame {t:%Y-%m-%d %H:%M} ({kind}) skipped: {type(err).__name__} {err}"[:200], flush=True)
        return None

    with ThreadPoolExecutor(workers) as pool:
        got = [r for r in pool.map(one, sorted(jobs)) if r is not None]
    return sorted(got, key=lambda x: x[0])


def _load(cache: Path):
    if not cache.exists():
        return None
    z = np.load(cache)
    if len(z["t"]) == 0:
        return None
    times = [datetime.fromtimestamp(int(s), timezone.utc) for s in z["t"]]
    dbz = z["dbz"].astype(np.float32) / 2.0 - 32.0
    dbz[z["dbz"] == 255] = np.nan
    return times, dbz


def day_frames(t0: datetime, t1: datetime, dirs=None) -> list[tuple[datetime, np.ndarray]]:
    """Rain-rate frames (qpe.despeckle + qpe.rain_rate) in [t0 - 10 min, t1], scan times. Days are read from the
    first folder of `dirs` that has them, else downloaded into RAW."""
    dirs = [RAW] + list(dirs or [])
    out = []
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= t1:
        name = f"{day:%Y%m%d}.npz"
        got = next((g for g in (_load(d / name) for d in dirs) if g is not None), None)
        if got is None:
            fr = fetch_radar_day(day)
            times = [t for t, _ in fr]
            dbz = np.stack([d for _, d in fr]) if fr else np.zeros((0, qpe.RG_NY, qpe.RG_NX), np.float32)
            code = np.clip(np.rint((np.nan_to_num(dbz, nan=0) + 32.0) * 2.0), 0, 254).astype(np.uint8)
            code[~np.isfinite(dbz)] = 255
            if len(times):
                RAW.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(RAW / name, t=np.array([int(t.timestamp()) for t in times]), dbz=code)
        else:
            times, dbz = got
        for t, d in zip(times, dbz):
            if t0 - timedelta(minutes=10) <= t <= t1:
                out.append((t, qpe.rain_rate(qpe.despeckle(d))))
        day += timedelta(days=1)
    return out


def case_utc_hours(case: dict) -> dict:
    """UTC day (iso) -> hours of that day covered by the civil (Madrid) days of a hindcast case."""
    u0 = datetime.fromisoformat(case["days"][0]).replace(tzinfo=MAD).astimezone(timezone.utc)
    u1 = (datetime.fromisoformat(case["days"][-1]).replace(tzinfo=MAD) + timedelta(days=1)).astimezone(timezone.utc)
    out = {}
    while u0 < u1:
        out.setdefault(u0.strftime("%Y-%m-%d"), []).append(u0.hour)
        u0 += timedelta(hours=1)
    return out
