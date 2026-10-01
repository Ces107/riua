#!/usr/bin/env python
"""Archived deterministic forecasts from Open-Meteo (free, keyless, CC-BY-4.0) for hindcasting.

Two sources, two sub-commands
-----------------------------
prev   Previous Runs API  https://previous-runs-api.open-meteo.com/v1/forecast
       `precipitation`               = "day 0": stitched series, lead ~0..update interval
       `precipitation_previous_dayN` = the value the model predicted N*24 h (+0..update interval)
                                       before the valid time. Archive starts 2024-01-20 (IFS025 2024-02).
       Rate limited (free tier 600 calls/min, 5000/h, 10000/day; one location = one call when a
       request has <=10 variables and <=14 days). Responses are cached on disk, a persistent quota
       ledger throttles the calls, so the command is resumable and can simply be re-run.

s3     Open-Meteo open-data bucket https://openmeteo.s3.amazonaws.com/data/<model>/<variable>/chunk_N.om
       The same stitched "day 0" series the API serves (verified identical), on the native model grid,
       WITHOUT any quota. Needs `pip install omfiles`. Use it for the 0-6 h lead hindcast.

Outputs (npz)
  prev: precip float32 [model, lead_day, time, point] mm/h (NaN = not archived), models, lead_days,
        time (ISO UTC, value = accumulation of the preceding hour), lat, lon (requested points),
        grid_lat, grid_lon, grid_elev [model, point] (cell actually used), meta (json)
  s3  : one file per model: data float32 [time, lat, lon] (native grid, lat/lon ascending), time, lat, lon, meta

Examples
  py -3.11 hindcast/fetch_openmeteo_archive.py prev --start 2024-10-28 --end 2024-10-30 --grid 0.1 \
      --models meteofrance_arome_france,icon_eu,meteofrance_arpege_europe,ecmwf_ifs025,gfs_global \
      --out hindcast/cache/openmeteo/prev_20241028_20241030_g010.npz
  py -3.11 hindcast/fetch_openmeteo_archive.py s3 --start 2024-10-28 --end 2024-10-30 \
      --models meteofrance_arome_france,icon_eu --variable precipitation
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

BOX = dict(lon_min=-2.4, lon_max=0.8, lat_min=37.6, lat_max=41.0)
UA = "riua-hindcast/0.1 (flood-risk research; python-requests)"
CACHE = Path(__file__).resolve().parent / "cache" / "openmeteo"
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
S3_URL = "https://openmeteo.s3.amazonaws.com"

# max_lead: deepest previous_dayN that is populated (verified 2024-02 .. 2026-09 at 39.47N 0.72W)
# snap    : regular lat/lon grid step aligned on multiples of the step -> points are snapped to cell
#           centres and de-duplicated before calling the API (saves quota for coarse models)
# s3      : folder name in the S3 bucket
MODELS = {
    "meteofrance_arome_france":       dict(max_lead=1, snap=0.025,  s3="meteofrance_arome_france0025"),
    "meteofrance_arome_france_hd":    dict(max_lead=1, snap=None,   s3="meteofrance_arome_france_hd"),
    "icon_eu":                        dict(max_lead=4, snap=0.0625, s3="dwd_icon_eu"),
    "icon_global":                    dict(max_lead=6, snap=None,   s3="dwd_icon"),
    "ecmwf_ifs025":                   dict(max_lead=7, snap=0.25,   s3="ecmwf_ifs025"),
    "gfs_global":                     dict(max_lead=7, snap=None,   s3="ncep_gfs013"),
    "meteofrance_arpege_europe":      dict(max_lead=3, snap=0.1,    s3="meteofrance_arpege_europe"),
    "meteofrance_arpege_world":       dict(max_lead=3, snap=0.25,   s3="meteofrance_arpege_world025"),
    "ukmo_global_deterministic_10km": dict(max_lead=6, snap=None,   s3="ukmo_global_deterministic_10km"),
    "gem_global":                     dict(max_lead=7, snap=None,   s3="cmc_gem_gdps"),
}

_session = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = UA
    return _session


# --------------------------------------------------------------------------------------------
# quota ledger (shared by every invocation on this machine)
# --------------------------------------------------------------------------------------------
class QuotaExhausted(RuntimeError):
    pass


class Quota:
    """Client-side throttle. Weights follow Open-Meteo's rule:
    weight = n_locations * max(1, n_variables*n_models/10) * max(1, n_days/14)."""

    def __init__(self, per_minute=450, per_hour=4000, per_day=8000, path: Path | None = None):
        self.lim = {60: per_minute, 3600: per_hour, 86400: per_day}
        self.path = path or CACHE / "_quota_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[tuple[float, float]] = []
        if self.path.exists():
            now = time.time()
            for line in self.path.read_text().splitlines():
                try:
                    t, w = json.loads(line)
                    if now - t < 86400:
                        self.events.append((t, w))
                except Exception:
                    pass

    def used(self, window: int) -> float:
        now = time.time()
        return sum(w for t, w in self.events if now - t < window)

    def wait_for(self, weight: float, allow_long_wait: bool = True):
        if weight > self.lim[60]:
            raise ValueError(f"single request weight {weight:.0f} exceeds per-minute budget {self.lim[60]}")
        while True:
            now = time.time()
            waits = []
            for window, limit in self.lim.items():
                if self.used(window) + weight > limit:
                    # wait until enough old events leave the window
                    need = self.used(window) + weight - limit
                    acc = 0.0
                    for t, w in sorted(e for e in self.events if now - e[0] < window):
                        acc += w
                        if acc >= need:
                            waits.append(t + window - now + 0.5)
                            break
            if not waits:
                return
            wait = max(waits)
            if wait > 120 and not allow_long_wait:
                raise QuotaExhausted(f"local quota budget reached, next slot in {wait / 60:.0f} min")
            if wait > 5:
                print(f"    quota: used {self.used(60):.0f}/min {self.used(3600):.0f}/h {self.used(86400):.0f}/day"
                      f" -> sleeping {wait:.0f}s", flush=True)
            time.sleep(max(wait, 0.2))

    def add(self, weight: float):
        ev = (time.time(), float(weight))
        self.events.append(ev)
        with self.path.open("a") as f:
            f.write(json.dumps(ev) + "\n")


def call_weight(n_loc: int, n_var: int, n_models: int, n_days: int) -> float:
    return n_loc * max(1.0, n_var * n_models / 10.0) * max(1.0, n_days / 14.0)


# --------------------------------------------------------------------------------------------
# Previous Runs API
# --------------------------------------------------------------------------------------------
def _api_get(params: dict, weight: float, quota: Quota, cache_file: Path, max_retries: int = 6):
    if cache_file.exists():
        return json.loads(cache_file.read_text()), False
    n429 = 0
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        quota.wait_for(weight)
        try:
            r = session().get(PREV_URL, params=params, timeout=(15, 180))
        except requests.RequestException as e:
            print(f"    network error {e!r:.100}, retry in 20s", file=sys.stderr)
            time.sleep(20)
            continue
        if r.status_code == 200:
            quota.add(weight)
            j = r.json()
            j = j if isinstance(j, list) else [j]
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(j))
            tmp.replace(cache_file)
            return j, True
        reason = ""
        try:
            reason = r.json().get("reason", "")
        except Exception:
            reason = r.text[:200]
        if r.status_code == 429:
            low = reason.lower()
            print(f"    HTTP 429: {reason}", file=sys.stderr, flush=True)
            attempt -= 1  # rate-limit waits do not count as failures
            if "daily" in low:
                raise QuotaExhausted("Open-Meteo daily limit reached; re-run tomorrow (cache keeps what is done)")
            n429 += 1
            if n429 > 12:
                raise QuotaExhausted(f"still rate limited after {n429} waits: {reason}")
            wait = 600 if "hour" in low else 61
            print(f"    sleeping {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {reason}")
    raise RuntimeError("too many retries")


def fetch_previous_runs(lat, lon, start: str, end: str, models: list[str], max_lead: int = 7,
                        variable: str = "precipitation", chunk_points: int = 100, quota: Quota | None = None,
                        pause: float = 1.0, verbose: bool = True) -> dict:
    """Return dict(precip[model, lead, time, point], time, grid_lat, grid_lon, grid_elev, ...)."""
    lat = np.asarray(lat, float).ravel()
    lon = np.asarray(lon, float).ravel()
    quota = quota or Quota()
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    n_days = (d1 - d0).days + 1
    n_time = n_days * 24
    leads = list(range(0, max_lead + 1))
    out = np.full((len(models), len(leads), n_time, len(lat)), np.nan, np.float32)
    glat = np.full((len(models), len(lat)), np.nan, np.float32)
    glon = np.full_like(glat, np.nan)
    gelev = np.full_like(glat, np.nan)
    times = None
    spent = 0.0
    for mi, model in enumerate(models):
        info = MODELS.get(model, dict(max_lead=7, snap=None))
        mleads = [n for n in leads if n <= info["max_lead"]]
        vars_ = [variable if n == 0 else f"{variable}_previous_day{n}" for n in mleads]
        if info.get("snap"):
            s = info["snap"]
            qlat, qlon = np.round(np.round(lat / s) * s, 5), np.round(np.round(lon / s) * s, 5)
        else:
            qlat, qlon = np.round(lat, 4), np.round(lon, 4)
        uniq, inverse = np.unique(np.stack([qlat, qlon], 1), axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        if verbose:
            print(f"{model}: {len(uniq)} unique cells for {len(lat)} points, leads {mleads}", flush=True)
        vals = np.full((len(mleads), n_time, len(uniq)), np.nan, np.float32)
        ug = np.full((3, len(uniq)), np.nan, np.float32)
        for c0 in range(0, len(uniq), chunk_points):
            pts = uniq[c0:c0 + chunk_points]
            params = dict(latitude=",".join(f"{p[0]:g}" for p in pts), longitude=",".join(f"{p[1]:g}" for p in pts),
                          hourly=",".join(vars_), models=model, start_date=start, end_date=end,
                          timezone="GMT", cell_selection="nearest")
            key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
            cf = CACHE / "prev" / model / f"{start}_{end}" / f"{c0:05d}_{key}.json"
            w = call_weight(len(pts), len(vars_), 1, n_days)
            res, fresh = _api_get(params, w, quota, cf)
            if fresh:
                spent += w
                time.sleep(pause)
            if len(res) != len(pts):
                raise RuntimeError(f"expected {len(pts)} locations, got {len(res)}")
            for k, loc in enumerate(res):
                h = loc.get("hourly")
                if h is None:  # point outside this model's domain (e.g. AROME south of ~38.0N)
                    continue
                if times is None:
                    times = h["time"]
                elif h["time"] != times:
                    raise RuntimeError("time axis mismatch between responses")
                ug[:, c0 + k] = (loc["latitude"], loc["longitude"], loc.get("elevation") or np.nan)
                for li, v in enumerate(vars_):
                    a = h.get(v)
                    if a is not None:
                        vals[li, :, c0 + k] = np.array([np.nan if x is None else x for x in a], np.float32)
            if verbose and fresh:
                print(f"    {model} cells {c0}..{c0 + len(pts) - 1} ok (weight {w:.0f}, session total {spent:.0f})",
                      flush=True)
        for li, n in enumerate(mleads):
            out[mi, leads.index(n)] = vals[li][:, inverse]
        glat[mi], glon[mi], gelev[mi] = ug[0][inverse], ug[1][inverse], ug[2][inverse]
    return dict(precip=out, models=np.array(models), lead_days=np.array(leads, np.int16),
                time=np.array(times), lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                grid_lat=glat, grid_lon=glon, grid_elev=gelev, calls_spent=spent)


# --------------------------------------------------------------------------------------------
# S3 .om time-series (stitched day-0 series, no quota)
# --------------------------------------------------------------------------------------------
class _RangeFS:
    """Minimal fsspec-like object for omfiles.OmFileReader.from_fsspec (HTTP Range reads)."""
    async_impl = False

    def __init__(self):
        self.nbytes = 0

    def size(self, path):
        return int(session().head(path, timeout=60).headers["Content-Length"])

    def cat_file(self, path, start=None, end=None, **kw):
        r = session().get(path, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=180)
        r.raise_for_status()
        self.nbytes += len(r.content)
        return r.content


def _s3_meta(s3model: str) -> dict:
    f = CACHE / "s3" / s3model / "grid.json"
    if f.exists():
        return json.loads(f.read_text())
    import omfiles

    meta = session().get(f"{S3_URL}/data/{s3model}/static/meta.json", timeout=60).json()
    bbox = [float(x) for x in re.findall(r"BBOX\[([^\]]+)\]", meta["crs_wkt"])[0].split(",")]
    hs = omfiles.OmFileReader.from_fsspec(_RangeFS(), f"{S3_URL}/data/{s3model}/static/HSURF.om")
    ny, nx = (int(x) for x in hs.shape[-2:])
    lat0, lon0, lat1, lon1 = bbox
    dx = (lon1 - lon0) / (nx - 1)  # BBOX holds first/last cell centres (checked: AROME, ICON-EU, IFS025)
    dy = (lat1 - lat0) / (ny - 1)
    g = dict(chunk_len=meta["chunk_time_length"], dt=meta["temporal_resolution_seconds"],
             update_interval=meta.get("update_interval_seconds"), ny=ny, nx=nx, lat0=lat0, lon0=lon0, dy=dy, dx=dx)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(g))
    return g


def fetch_s3_box(model: str, variable: str, start: str, end: str, whole_file_max_mb: float = 80.0,
                 verbose: bool = True) -> dict:
    """Read the box from the public .om chunks. Returns data[time, lat, lon] on the native grid."""
    import omfiles

    s3model = MODELS[model]["s3"] if model in MODELS else model
    g = _s3_meta(s3model)
    ny, nx, L, dts = g["ny"], g["nx"], g["chunk_len"], g["dt"]
    eps = 1e-6
    y0 = int(np.ceil((BOX["lat_min"] - g["lat0"]) / g["dy"] - eps))
    y1 = int(np.floor((BOX["lat_max"] - g["lat0"]) / g["dy"] + eps))
    x0 = int(np.ceil((BOX["lon_min"] - g["lon0"]) / g["dx"] - eps))
    x1 = int(np.floor((BOX["lon_max"] - g["lon0"]) / g["dx"] + eps))
    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, ny - 1), min(x1, nx - 1)
    t0 = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp())
    t1 = int((dt.datetime.fromisoformat(end) + dt.timedelta(days=1)).replace(tzinfo=dt.timezone.utc).timestamp())
    pieces, tt, nbytes = [], [], 0
    for n in range(t0 // (L * dts), (t1 - 1) // (L * dts) + 1):
        cf = CACHE / "s3" / s3model / variable / f"box_chunk_{n}.npy"
        if cf.exists():
            box = np.load(cf)
        else:
            url = f"{S3_URL}/data/{s3model}/{variable}/chunk_{n}.om"
            head = session().head(url, timeout=60)
            if head.status_code != 200:
                raise IOError(f"{url}: HTTP {head.status_code} (not archived)")
            size = int(head.headers["Content-Length"])
            tmp = None
            if size <= whole_file_max_mb * 1e6:  # small regional file: one GET, read locally
                tmp = CACHE / "s3" / s3model / variable / f"chunk_{n}.om.part"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with session().get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with tmp.open("wb") as fh:
                        for blk in r.iter_content(1 << 20):
                            fh.write(blk)
                nbytes += size
                rd = omfiles.OmFileReader(str(tmp))
            else:  # big global file: HTTP range reads of the needed rows only
                fs = _RangeFS()
                rd = omfiles.OmFileReader.from_fsspec(fs, url)
            shp = tuple(rd.shape)
            if len(shp) == 2 and shp[0] == ny * nx:
                box = np.stack([rd[y * nx + x0:y * nx + x1 + 1, :] for y in range(y0, y1 + 1)])
            elif len(shp) == 3 and shp[0] == ny and shp[1] == nx:
                box = rd[y0:y1 + 1, x0:x1 + 1, :]
            elif len(shp) == 3 and shp[0] == 1 and shp[1] == ny * nx:
                box = np.stack([rd[0, y * nx + x0:y * nx + x1 + 1, :] for y in range(y0, y1 + 1)])
            else:
                raise IOError(f"unexpected .om shape {shp} for grid {ny}x{nx}")
            box = np.asarray(box, np.float32)
            rd.close()
            if tmp is not None:
                tmp.unlink()
            else:
                nbytes += fs.nbytes
            np.save(cf, box)
        pieces.append(box)
        tt.append(n * L * dts + np.arange(L) * dts)
        if verbose:
            print(f"  {s3model}/{variable} chunk {n}: box {box.shape}, downloaded so far {nbytes / 1e6:.1f} MB", flush=True)
    data = np.concatenate(pieces, axis=2)
    tt = np.concatenate(tt)
    sel = (tt >= t0) & (tt < t1)
    data = np.moveaxis(data[:, :, sel], 2, 0)  # time, lat, lon
    times = np.array([dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M") for t in tt[sel]])
    lat = (g["lat0"] + np.arange(y0, y1 + 1) * g["dy"]).astype(np.float32)
    lon = (g["lon0"] + np.arange(x0, x1 + 1) * g["dx"]).astype(np.float32)
    return dict(data=data, time=times, lat=lat, lon=lon, bytes=nbytes,
                meta=dict(model=model, s3model=s3model, variable=variable, dt_seconds=dts,
                          update_interval_seconds=g["update_interval"],
                          note="stitched series: each value comes from the latest run before the valid time"))


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def box_grid(step: float):
    lats = np.round(np.arange(BOX["lat_min"], BOX["lat_max"] + 1e-9, step), 4)
    lons = np.round(np.arange(BOX["lon_min"], BOX["lon_max"] + 1e-9, step), 4)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return la.ravel(), lo.ravel(), (len(lats), len(lons))


def load_points(path: str):
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p)
        return np.asarray(z["lat"], float).ravel(), np.asarray(z["lon"], float).ravel()
    a = np.loadtxt(p, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prev", help="Previous Runs API (lead-specific, rate limited)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--models", default="meteofrance_arome_france,icon_eu,meteofrance_arpege_europe,ecmwf_ifs025,gfs_global")
    p.add_argument("--grid", type=float, default=None, help="regular grid step over the box, e.g. 0.1")
    p.add_argument("--points", default=None, help=".npz with 1-D lat/lon arrays, or csv with header lat,lon")
    p.add_argument("--max-lead", type=int, default=7)
    p.add_argument("--variable", default="precipitation")
    p.add_argument("--chunk-points", type=int, default=100)
    p.add_argument("--per-minute", type=float, default=450)
    p.add_argument("--per-hour", type=float, default=4000)
    p.add_argument("--per-day", type=float, default=8000)
    p.add_argument("--out", default=None)
    s = sub.add_parser("s3", help="stitched day-0 series from the S3 bucket (no quota)")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--models", default="meteofrance_arome_france,icon_eu,meteofrance_arpege_europe,ecmwf_ifs025")
    s.add_argument("--variable", default="precipitation")
    s.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)
    models = a.models.split(",")
    if a.cmd == "prev":
        if a.points:
            lat, lon = load_points(a.points)
            shape = None
        else:
            lat, lon, shape = box_grid(a.grid or 0.1)
        quota = Quota(a.per_minute, a.per_hour, a.per_day)
        print(f"previous runs {a.start}..{a.end}: {len(lat)} points, models {models}; "
              f"ledger: {quota.used(3600):.0f} calls last hour, {quota.used(86400):.0f} last 24 h")
        t0 = time.time()
        r = fetch_previous_runs(lat, lon, a.start, a.end, models, a.max_lead, a.variable, a.chunk_points, quota)
        out = Path(a.out) if a.out else CACHE / f"prev_{a.variable}_{a.start}_{a.end}_{len(lat)}pts.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(source=PREV_URL, variable=a.variable, units="mm (preceding hour)", box=BOX,
                    grid_shape=shape, calls_spent=r["calls_spent"],
                    lead_definition="lead_day N = precipitation_previous_dayN: value forecast N*24 h (+0..update "
                                    "interval) before valid time; N=0 is the stitched latest-run series",
                    created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **{k: v for k, v in r.items() if k != "calls_spent"})
        tmp.replace(out)
        print(f"wrote {out} precip{r['precip'].shape} ({out.stat().st_size / 1e6:.2f} MB); "
              f"API calls spent now {r['calls_spent']:.0f}; {time.time() - t0:.0f}s")
        pr = r["precip"]
        print("24 h box-max / box-mean per model and lead day (mm), by date:")
        for d in range(pr.shape[2] // 24):
            day = r["time"][d * 24][:10]
            for mi, m in enumerate(models):
                acc = pr[mi, :, d * 24:(d + 1) * 24, :].sum(axis=1)  # lead, point (NaN stays NaN)
                row = " ".join(f"d{n}:{np.nanmax(acc[li]):6.1f}/{np.nanmean(acc[li]):5.1f}" if np.isfinite(acc[li]).any()
                               else f"d{n}:   --/   --" for li, n in enumerate(r["lead_days"]))
                print(f"  {day} {m:28s} {row}")
    else:
        for m in models:
            r = fetch_s3_box(m, a.variable, a.start, a.end)
            outdir = Path(a.outdir) if a.outdir else CACHE
            outdir.mkdir(parents=True, exist_ok=True)
            out = outdir / f"s3_{m}_{a.variable}_{a.start}_{a.end}.npz"
            np.savez_compressed(out, data=r["data"], time=r["time"], lat=r["lat"], lon=r["lon"],
                                meta=json.dumps(dict(r["meta"], box=BOX, bytes_downloaded=r["bytes"])))
            d = r["data"]
            steps_per_day = 86400 // r["meta"]["dt_seconds"]
            daily = [float(np.nanmax(np.nansum(d[i:i + steps_per_day], axis=0))) for i in range(0, d.shape[0], steps_per_day)]
            print(f"wrote {out} data{d.shape} grid {r['lat'][0]:.3f}..{r['lat'][-1]:.3f} x {r['lon'][0]:.3f}..{r['lon'][-1]:.3f}"
                  f" | {r['bytes'] / 1e6:.1f} MB downloaded | daily box max {np.round(daily, 1).tolist()} | NaN {np.isnan(d).mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
