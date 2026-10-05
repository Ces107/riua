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
import os
import sys

# One thread per process: the optical flow (OpenCV) and the linear algebra otherwise start one thread per
# core EACH, and several builds side by side then take 15x longer (measured: motion_field 67 s instead of ~1 s).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
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

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:  # noqa: BLE001
    pass

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


# ----------------------------------------------------------------------------- OPERA archive, all formats
# The archive bucket changed format twice (listed 2026-10-02, see coord/findings/q4-hindcast.md):
#   .. 2024-07-01   OPERA@<t>@0@DBZH_QIND.{h5,tiff}   2 km, every 15 min (the tiff merges "no echo" and "no data")
#   2024-07-01 ..   OPERA@<t>@0@DBZH.{h5,tiff}        1 km, every 5 min
#   2026-01-01 ..   OPERA@<t>@0@DBZH.h5 only          1 km, every 5 min (no tiff any more)
# riua.sources.radar reads the DBZH tiff only, so the h5 files are read here: metadata through h5py over
# HTTP Range requests, then only the 2-4 gzip chunks that cover the box.

class _HttpFile:
    """Read-only file object over HTTP Range requests with a block cache (what h5py needs to open a file)."""

    def __init__(self, url: str, block: int = 16384):     # the whole h5 metadata sits in the first ~13 kB
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


_H5_INDEX = {}      # (xscale, shape) -> (row, col) of every Riuà radar-grid cell in the composite


def opera_h5_list(day: datetime, quantity: str) -> list[datetime]:
    """Times (UTC) of the archived .h5 composites of one day for 'DBZH' or 'DBZH_QIND'."""
    import re
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
    """One archived ODIM-H5 composite on the Riuà 1 km grid: dBZ, NO_ECHO_DBZ where the radar saw nothing,
    NaN where there is no coverage. Bit-identical to radar.fetch_opera() on a date that has both formats
    (2025-12-28 12:00Z: 0 of 108 800 cells differ)."""
    import zlib
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
            # the first pixel is centred on the projection origin (same half-pixel shift as the tiff tie point)
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


def fetch_radar_day(day: datetime, workers: int = 4) -> list[tuple[datetime, np.ndarray]]:
    """Every archived composite of one UTC day on the Riuà grid, whatever the archive format of that date:
    10-min steps of the 1 km product (tiff, else h5), completed with the 15-min 2 km product where the 1 km
    one does not exist (before 2024-07-01 and the first hours of that day)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(t, "tiff") for t in radar.opera_list(day, "DBZH", radar.OPERA_BUCKET_ARCHIVE) if t.minute % 10 == 0]
    if not jobs:
        jobs = [(t, "DBZH") for t in opera_h5_list(day, "DBZH") if t.minute % 10 == 0]
    first = min((t for t, _ in jobs), default=day + timedelta(days=1))
    if first > day + timedelta(minutes=20):
        jobs += [(t, "DBZH_QIND") for t in opera_h5_list(day, "DBZH_QIND") if t < first]

    def one(job):
        t, kind = job
        for attempt in range(3):
            try:
                if kind == "tiff":
                    return t, radar.fetch_opera(t, "DBZH", radar.OPERA_BUCKET_ARCHIVE)
                return t, fetch_opera_h5(t, kind)
            except Exception as e:  # noqa: BLE001  one bad frame must not lose the day
                err = e
        print(f"  radar frame {t:%Y-%m-%d %H:%M} ({kind}) skipped: {type(err).__name__} {err}"[:200], flush=True)
        return None

    with ThreadPoolExecutor(workers) as pool:
        got = [r for r in pool.map(one, sorted(jobs)) if r is not None]
    return sorted(got, key=lambda x: x[0])


def _load_radar_cache(cache: Path):
    if not cache.exists():
        return None
    with np.load(cache) as z:
        t, code = z["t"], z["dbz"]
    if len(t) == 0:
        return None            # an empty file is a failed download of an earlier version, not "no radar"
    times = [datetime.fromtimestamp(int(s), timezone.utc) for s in t]
    dbz = code.astype(np.float32) / 2.0 - 32.0
    dbz[code == 255] = np.nan
    return times, dbz


def day_frames(t0: datetime, t1: datetime, rates: bool = True):
    """Rain-rate frames in [t0 - 10 min, t1], cached on disk as compressed dBZ."""
    RAW.mkdir(parents=True, exist_ok=True)
    out = []
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= t1:
        cache = RAW / f"{day:%Y%m%d}.npz"
        got = _load_radar_cache(cache)
        if got is not None:
            times, dbz = got
        else:
            fr = fetch_radar_day(day)
            times = [t for t, _ in fr]
            dbz = np.stack([d for _, d in fr]) if fr else np.zeros((0, qpe.RG_NY, qpe.RG_NX), np.float32)
            code = np.clip(np.rint((np.nan_to_num(dbz, nan=0) + 32.0) * 2.0), 0, 254).astype(np.uint8)
            code[~np.isfinite(dbz)] = 255
            if len(times):
                np.savez_compressed(cache, t=np.array([int(t.timestamp()) for t in times]), dbz=code)
            print(f"  radar {day:%Y-%m-%d}: {len(times)} frames downloaded", flush=True)
        if not rates:
            day += timedelta(days=1)
            continue
        for t, d in zip(times, dbz):
            if t0 - timedelta(minutes=10) <= t <= t1:
                if CLUTTER is not None:
                    d = np.where(CLUTTER & np.isfinite(d), qpe.NO_ECHO_DBZ, d)
                out.append((t, qpe.rain_rate(qpe.despeckle(d))))
        day += timedelta(days=1)
    return out


# ------------------------------------------------------------------ hourly radar accumulation, cached per UTC day
# The advection-corrected accumulation is ~99 % of the cost of a build (optical flow for every pair of scans)
# and does not depend on the gauges: it is kept in hindcast/cache/acc/<yyyymmdd>.npz so that a change in the
# gauge merging rebuilds a case in seconds, and an interrupted batch resumes at the day it stopped.
ACC = ROOT / "hindcast" / "cache" / "acc"
_TAG = None


def qpe_tag() -> str:
    """Fingerprint of the code that turns scans into hourly rain: a cached day is reused only if it matches."""
    global _TAG
    if _TAG is None:
        import hashlib
        import inspect
        src = "".join(inspect.getsource(f) for f in (qpe.despeckle, qpe.steiner_convective, qpe.rain_rate,
                                                     qpe.motion_field, qpe._advect, qpe.accumulate))
        for c in (CLUTTER, getattr(qpe, "CLUTTER", None)):
            src += "none" if c is None else str(int(np.asarray(c).sum()))
        _TAG = hashlib.sha1(src.encode()).hexdigest()[:12]
    return _TAG


def utc_day_acc(day: datetime, need=range(24)) -> tuple[np.ndarray, np.ndarray]:
    """(acc[24, ny, nx] mm, NaN = no radar or hour not computed; coverage[24]) for the UTC day starting at `day`.
    Only the hours in `need` are guaranteed; hours already in the cache are not recomputed."""
    f = ACC / f"{day:%Y%m%d}.npz"
    # the last hour needs the 00:00 scan of the next day: used when that day is on disk, never downloaded for it
    nxt = _load_radar_cache(RAW / f"{day + timedelta(days=1):%Y%m%d}.npz") is not None
    acc = np.full((24, qpe.RG_NY, qpe.RG_NX), np.nan, np.float32)
    cov = np.zeros(24)
    done = np.zeros(24, bool)
    if f.exists():
        with np.load(f) as z:                   # closed at once: Windows cannot replace an open file
            if str(z["tag"]) == qpe_tag():
                code = z["acc"]
                acc = code.astype(np.float32) / 50.0
                acc[code == 65535] = np.nan
                cov, done = z["cov"].copy(), z["done"].copy()
    todo = [h for h in need if not done[h]]
    if not todo:
        return acc, cov
    frames = day_frames(day + timedelta(hours=todo[0]), min(day + timedelta(hours=todo[-1] + 1),
                                                           day + (timedelta(hours=24) if nxt else timedelta(hours=23, minutes=59))))
    for h in todo:
        a, b = day + timedelta(hours=h), day + timedelta(hours=h + 1)
        sub = [fr for fr in frames if a - timedelta(minutes=10) <= fr[0] <= b]
        acc[h], cov[h] = qpe.accumulate(sub, a, b) if len(sub) >= 2 else (np.nan, 0.0)
        done[h] = h < 23 or nxt                 # the last hour stays open until the next day's first scan is there
    code = np.where(np.isfinite(acc), np.clip(np.rint(np.nan_to_num(acc) * 50.0), 0, 65534), 65535).astype(np.uint16)
    ACC.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, acc=code, cov=cov, done=done, tag=qpe_tag())
    tmp.replace(f)
    out = code.astype(np.float32) / 50.0          # same 0.02 mm rounding whether the day was cached or not
    out[code == 65535] = np.nan
    return out, cov


def _acc_job(job) -> str:
    import time
    t = time.time()
    day_iso, need = job
    day = datetime.fromisoformat(day_iso).replace(tzinfo=timezone.utc)
    try:
        a, cov = utc_day_acc(day, need)
        return (f"acc {day_iso} hours {need[0]}-{need[-1]}: coverage {cov[need].mean():.2f}, "
                f"max 1 h {np.nanmax(a) if np.isfinite(a).any() else float('nan'):.0f} mm, {time.time() - t:.0f} s")
    except Exception as e:  # noqa: BLE001
        return f"acc {day_iso} FAILED {type(e).__name__} {e}"[:300]


def case_utc_hours(case: dict) -> dict:
    """UTC day (iso) -> sorted list of the hours of that day the case needs."""
    u0 = datetime.fromisoformat(case["days"][0]).replace(tzinfo=MAD).astimezone(timezone.utc)
    u1 = (datetime.fromisoformat(case["days"][-1]).replace(tzinfo=MAD) + timedelta(days=1)).astimezone(timezone.utc)
    out = {}
    while u0 < u1:
        out.setdefault(u0.strftime("%Y-%m-%d"), []).append(u0.hour)
        u0 += timedelta(hours=1)
    return out


def build(case: dict) -> None:
    days = case["days"]
    d0 = datetime.fromisoformat(days[0]).replace(tzinfo=MAD)
    d1 = datetime.fromisoformat(days[-1]).replace(tzinfo=MAD) + timedelta(days=1)
    u0, u1 = d0.astimezone(timezone.utc), d1.astimezone(timezone.utc)
    hours = int((u1 - u0).total_seconds() // 3600)
    acc = np.full((hours, qpe.RG_NY, qpe.RG_NX), np.nan, np.float32)
    cov = np.zeros(hours)
    day_frames(u0, u1, rates=False)             # every radar day on disk before the accumulation starts
    h = 0
    for day_iso, need in case_utc_hours(case).items():
        a, c = utc_day_acc(datetime.fromisoformat(day_iso).replace(tzinfo=timezone.utc), need)
        acc[h:h + len(need)], cov[h:h + len(need)] = a[need], c[need]
        h += len(need)
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
    if "--radar-only" in sys.argv:          # download and cache the composites, build nothing
        for c in cases:
            if want and c["id"] not in want:
                continue
            a = datetime.fromisoformat(c["days"][0]).replace(tzinfo=MAD).astimezone(timezone.utc)
            b = (datetime.fromisoformat(c["days"][-1]).replace(tzinfo=MAD) + timedelta(days=1)).astimezone(timezone.utc)
            try:
                day_frames(a, b, rates=False)
                print("radar ok", c["id"], flush=True)
            except Exception as e:  # noqa: BLE001
                print("radar FAILED", c["id"], type(e).__name__, e, flush=True)
        sys.exit(0)
    if "--acc-only" in sys.argv:            # the expensive step alone, one UTC day per job: --acc-only --jobs 6 [case ...]
        from multiprocessing import Pool
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
        want.discard(str(jobs))
        todo = {}
        for c in cases:
            if want and c["id"] not in want:
                continue
            a = datetime.fromisoformat(c["days"][0]).replace(tzinfo=MAD).astimezone(timezone.utc)
            b = (datetime.fromisoformat(c["days"][-1]).replace(tzinfo=MAD) + timedelta(days=1)).astimezone(timezone.utc)
            day_frames(a, b, rates=False)       # downloads what is missing, sequentially
            for d, need in case_utc_hours(c).items():
                todo[d] = sorted(set(todo.get(d, [])) | set(need))
        print(f"{len(todo)} UTC days, {jobs} processes, code tag {qpe_tag()}", flush=True)
        with Pool(jobs) as pool:
            for line in pool.imap_unordered(_acc_job, sorted(todo.items())):
                print(line, flush=True)
        sys.exit(0)
    for c in cases:
        if want and c["id"] not in want:
            continue
        if (OUT / f"{c['id']}.npz").exists() and not force:
            print("skip", c["id"]); continue
        try:
            build(c)
        except Exception as e:  # one bad case must not stop the batch
            print("FAILED", c["id"], type(e).__name__, e, flush=True)
