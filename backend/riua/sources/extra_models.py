"""Extra forecast sources for Riuà: every one free, keyless and real-time (verified 2026-10-01,
see ``coord/findings/q6-sources.md``). Each loader returns ``Member`` objects on the Riuà grid
(hourly accumulations, mm), exactly like ``ingest.load_run``.

===================  ========================================  =======  ======  ====================
loader               source                                    members  step    ready after run time
===================  ========================================  =======  ======  ====================
ecmwf_ens_members    ECMWF open data, IFS ENS 0.25 deg         50       3 h     +7.1 .. +8.9 h
ecmwf_ens_members    ECMWF open data, AIFS-ENS 0.25 deg        51       6 h     +5.8 .. +6.7 h
icon_eu_eps_members  DWD open data, ICON-EU-EPS (~13 km here)  40       1-3 h   +2.7 .. +3.1 h
arome_ifs_member     Meteo-France open data, AROME-IFS 2.5 km  1        1 h     +6.7 .. +7.3 h
gefs_members         NOAA GEFS 0.25 deg on AWS                 31       6 h     +3.8 .. +6 h
s3_member            Open-Meteo S3: IFS 9 km, GFS 13 km, ...   1        1-6 h   see EXTRA_S3
===================  ========================================  =======  ======  ====================

Conventions shared by all loaders

* a run is a naive UTC ``datetime``; ``cache`` is a folder that survives between cycles (``state/extra``):
  what has been downloaded is never downloaded again, and a fetch that runs out of ``budget_s`` raises
  :class:`Incomplete` and resumes where it stopped at the next call;
* the native precipitation of all the GRIB sources is ACCUMULATED SINCE THE RUN START (GEFS: 6-h buckets);
  it is differenced here and spread evenly over the hours of the step (``Member.native_step_h`` tells);
* ``transfer()`` gives (requests, bytes) since import, to log what a cycle cost.

One call does everything for the production cycle::

    members, report = extra_models.extra_members(state / "extra", now, budget_s=45)

Smoke test (from ``backend/``)::  ``py -3.11 -m riua.sources.extra_models [ens|aifs|icon|aromeifs|gefs|s3|all]``
"""
from __future__ import annotations

import bz2
import json
import logging
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from ..core import grid
from ..core.risk import Member

log = logging.getLogger("riua.extra")

UA = "riua/0.1 (+https://github.com/Ces107/riua; flood-risk research; python-requests)"
BOX = grid.BBOX                      # west, south, east, north
MARGIN = 0.3                         # degrees kept around the box so the regridding has neighbours

__all__ = ["Incomplete", "transfer", "extra_members", "ecmwf_ens_latest_run", "ecmwf_ens_members",
           "icon_eu_eps_latest_run", "icon_eu_eps_members", "arome_ifs_latest_run", "arome_ifs_member",
           "gefs_latest_run", "gefs_members", "EXTRA_S3", "s3_member"]


class Incomplete(RuntimeError):
    """The time budget ran out before the run was complete; what was fetched is cached."""


# ------------------------------------------------------------------------------------------- HTTP

_local = threading.local()
_stats = {"requests": 0, "bytes": 0}
_stats_lock = threading.Lock()


def transfer() -> tuple[int, int]:
    """(HTTP requests, bytes downloaded) since the module was imported."""
    return _stats["requests"], _stats["bytes"]


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.headers["User-Agent"] = UA
    return s


def _get(url: str, rng: tuple[int, int] | None = None, tries: int = 4, timeout: float = 60.0,
         params: dict | None = None) -> bytes | None:
    """GET (optionally a byte range, both ends inclusive). None on 403/404. Retries 429/5xx."""
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    why = ""
    for attempt in range(tries):
        try:
            r = _session().get(url, headers=headers, timeout=(10, timeout), params=params)
            with _stats_lock:
                _stats["requests"] += 1
                _stats["bytes"] += len(r.content)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code in (403, 404, 416):
                return None
            why = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            why = type(e).__name__
        time.sleep(1.5 * (attempt + 1))
    raise OSError(f"{url}: {why}")


def _exists(url: str) -> bool:
    try:
        r = _session().head(url, timeout=(10, 30), allow_redirects=True)
        with _stats_lock:
            _stats["requests"] += 1
        return r.status_code == 200
    except requests.RequestException:
        return False


class _Deadline:
    def __init__(self, budget_s: float | None):
        self.t_end = None if budget_s is None else time.monotonic() + budget_s

    def check(self, what: str) -> None:
        if self.t_end is not None and time.monotonic() > self.t_end:
            raise Incomplete(what)


def _naive(t: datetime) -> datetime:
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def _save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


# ------------------------------------------------------------------------------------------- GRIB

_ec_lock = threading.Lock()     # first use of eccodes from two threads at once kills the process


def _split(raw: bytes) -> list[bytes]:
    """GRIB2 messages inside a buffer."""
    out, i = [], 0
    while True:
        i = raw.find(b"GRIB", i)
        if i < 0 or i + 16 > len(raw):
            return out
        n = int.from_bytes(raw[i + 8:i + 16], "big")
        if raw[i + 7] != 2 or n < 100 or i + n > len(raw):
            i += 4
            continue
        out.append(raw[i:i + n])
        i += n


def _sections(msg: bytes) -> dict[int, tuple[int, int]]:
    """{section number: (offset, length)} of the sections fully or partly present in ``msg``."""
    out, i = {}, 16
    while i + 5 <= len(msg) and msg[i:i + 4] != b"7777":
        ln, num = int.from_bytes(msg[i:i + 4], "big"), msg[i + 4]
        out[num] = (i, ln)
        if num == 7 or ln < 5:
            break
        i += ln
    return out


def _decode(msg: bytes, keys: tuple[str, ...] = ()):
    """values (float64, 1-D, missing = NaN) and the requested keys of one message."""
    import eccodes as ec
    with _ec_lock:
        h = ec.codes_new_from_message(msg)
        try:
            ec.codes_set(h, "missingValue", 1e20)
            v = ec.codes_get_values(h)
            info = {k: ec.codes_get(h, k) for k in keys}
        finally:
            ec.codes_release(h)
    if (v >= 1e19).any():
        v = np.where(v >= 1e19, np.nan, v)
    return v, info


def _latlon_axes(info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Axes of a regular lat/lon message in storage order (cheap: eccodes' distinctLongitudes costs 1 s)."""
    lat = np.linspace(info["latitudeOfFirstGridPointInDegrees"], info["latitudeOfLastGridPointInDegrees"], info["Nj"])
    lo0, lo1 = info["longitudeOfFirstGridPointInDegrees"], info["longitudeOfLastGridPointInDegrees"]
    if lo1 < lo0:
        lo1 += 360.0
    lon = np.linspace(lo0, lo1, info["Ni"])
    return lat, ((lon + 180.0) % 360.0) - 180.0


_LL_KEYS = ("latitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees", "Ni", "Nj")


def _box_window(lat: np.ndarray, lon: np.ndarray, margin: float = MARGIN):
    """Row and column indices (in storage order) of the points inside the box plus a margin."""
    rows = np.nonzero((lat >= BOX[1] - margin) & (lat <= BOX[3] + margin))[0]
    cols = np.nonzero((lon >= BOX[0] - margin) & (lon <= BOX[2] + margin))[0]
    cols = cols[np.argsort(lon[cols])]            # global grids start at 0 or 180 E: put west first
    return rows, cols


_REGRID: dict[tuple, grid.Regridder] = {}


def _regridder(key: str, lat: np.ndarray, lon: np.ndarray, points: bool = False) -> grid.Regridder:
    k = (key, lat.shape, float(lat.ravel()[0]), float(lon.ravel()[0]), points)
    if k not in _REGRID:
        _REGRID[k] = grid.Regridder(lat, lon, points=points)
    return _REGRID[k]


def _hourly(run: datetime, steps: list[int], acc: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulation since run start at ``steps`` (hours), shape (S, ...) -> hourly amounts.
    Returns t_end (H,) datetime64[h], p (H, ...) float32, largest step. The first step is only the baseline."""
    t, p, big = [], [], 1
    for s in range(1, len(steps)):
        dt = int(steps[s] - steps[s - 1])
        big = max(big, dt)
        inc = np.maximum(acc[s] - acc[s - 1], 0.0) / dt
        for h in range(dt):
            t.append(np.datetime64(run + timedelta(hours=int(steps[s - 1]) + h + 1), "h"))
            p.append(inc)
    return np.array(t), np.stack(p).astype(np.float32), big


def _members(acc: np.ndarray, rg: grid.Regridder, run: datetime, steps: list[int], numbers, label: str,
             family: str, model: str, weight: float = 1.0) -> list[Member]:
    """acc (M, S, *src) accumulated since run start -> one Member per ensemble member."""
    out = []
    for k in range(acc.shape[0]):
        t_end, p, big = _hourly(run, steps, rg(acc[k]))
        out.append(Member(f"{label} m{int(numbers[k]):02d} · {run:%d/%m %H}Z", family, model, run, t_end, p, big, weight))
    return out


# ------------------------------------------------------------------- ECMWF open data: IFS ENS, AIFS-ENS

ECMWF_MIRRORS = {"ecmwf": "https://data.ecmwf.int/forecasts",          # newest ~4 days only
                 "gcs": "https://storage.googleapis.com/ecmwf-open-data"}  # same files + archive since 2023-07
# model -> (directory, file suffixes, native step, last step of a 06/18 run)
_ECMWF = {"ifs": ("ifs/0p25/enfo", ("enfo-ef",), 3, 144), "aifs": ("aifs-ens/0p25/enfo", ("enfo-cf", "enfo-pf"), 6, 360)}


def _ecmwf_stem(run: datetime, step: int, model: str, sfx: str, mirror: str) -> str:
    return f"{ECMWF_MIRRORS[mirror]}/{run:%Y%m%d}/{run:%H}z/{_ECMWF[model][0]}/{run:%Y%m%d%H}0000-{step}h-{sfx}"


def ecmwf_ens_latest_run(now: datetime, last_step: int = 66, model: str = "ifs", mirror: str = "gcs") -> datetime | None:
    """Newest run whose ``last_step`` is already published (one HEAD per candidate)."""
    n = _naive(now)
    base = n.replace(hour=(n.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for k in range(6):
        run = base - timedelta(hours=6 * k)
        if n - run < timedelta(hours=5):
            continue
        if _exists(_ecmwf_stem(run, last_step, model, _ECMWF[model][1][-1], mirror) + ".index"):
            return run
    return None


def _rows_message(prefix: bytes, nrows: int, pad: bytes) -> bytes:
    """A valid GRIB2 message holding only the first ``nrows`` latitude rows of a regular lat/lon field
    scanned north -> south, built from the first bytes of the real message. CCSDS decodes sequentially, so
    the rows are exact as long as ``prefix`` reaches that far; ``pad`` stands in for the missing bytes."""
    s = _sections(prefix)
    o3, o5, o7 = s[3][0], s[5][0], s[7][0]
    b = bytearray(prefix + pad)
    ni = int.from_bytes(b[o3 + 30:o3 + 34], "big")
    la1, dj = int.from_bytes(b[o3 + 46:o3 + 50], "big"), int.from_bytes(b[o3 + 67:o3 + 71], "big")
    b[o3 + 6:o3 + 10] = (nrows * ni).to_bytes(4, "big")              # number of data points
    b[o3 + 34:o3 + 38] = nrows.to_bytes(4, "big")                    # Nj
    b[o3 + 55:o3 + 59] = (la1 - (nrows - 1) * dj).to_bytes(4, "big")  # latitude of the last row
    b[o5 + 5:o5 + 9] = (nrows * ni).to_bytes(4, "big")               # number of packed values
    b[o7:o7 + 4] = (len(b) - o7).to_bytes(4, "big")
    b += b"7777"
    b[8:16] = len(b).to_bytes(8, "big")
    return bytes(b)


def _ecmwf_field(url: str, off: int, length: int):
    """One global 0.25 deg message -> (box values [ny, nx] mm, lat, lon, bytes). Only the part of the
    message north of the box's southern edge is downloaded (~30 %); two decodes with different paddings
    must agree, otherwise more bytes are fetched."""
    have = _get(url, (off, off + int(length * 0.30) - 1))            # enough for 37.5 N in 9 cases out of 10
    s = _sections(have)
    o3 = s[3][0]
    if int.from_bytes(have[o3 + 12:o3 + 14], "big") != 0 or have[o3 + 71] & 0xC0:
        raise OSError("unexpected grid: need regular lat/lon scanned west->east, north->south")
    ni, nj = int.from_bytes(have[o3 + 30:o3 + 34], "big"), int.from_bytes(have[o3 + 34:o3 + 38], "big")
    la1, lo1 = int.from_bytes(have[o3 + 46:o3 + 50], "big") / 1e6, int.from_bytes(have[o3 + 50:o3 + 54], "big") / 1e6
    di, dj = int.from_bytes(have[o3 + 63:o3 + 67], "big") / 1e6, int.from_bytes(have[o3 + 67:o3 + 71], "big") / 1e6
    lat = la1 - dj * np.arange(nj)
    lon = ((lo1 + di * np.arange(ni) + 180.0) % 360.0) - 180.0
    rows, cols = _box_window(lat, lon)
    nrows = min(nj, int(rows[-1]) + 1 + 4)                          # 4 spare rows below the window
    while True:
        if len(have) >= length:
            v, info = _decode(have, ("units",))
            v = v.reshape(nj, ni)
            break
        a, info = _decode(_rows_message(have, nrows, b"\xff" * 4096), ("units",))
        b, _ = _decode(_rows_message(have, nrows, b"\x55" * 4096))
        if np.array_equal(a, b):
            v = a.reshape(nrows, ni)
            break
        have += _get(url, (off + len(have), off + min(length, len(have) + int(length * 0.08)) - 1))
    scale = 1000.0 if info["units"] == "m" else 1.0                  # IFS: metres; AIFS: kg m-2
    box = v[np.ix_(rows, cols)] * scale
    return box[::-1].astype(np.float32), lat[rows][::-1].astype(np.float32), lon[cols].astype(np.float32), len(have)


def ecmwf_ens_fetch(run: datetime, steps: list[int], cache: Path, model: str = "ifs", mirror: str = "gcs",
                    budget_s: float | None = None, workers: int = 6) -> dict:
    """tp of every member at ``steps``. Returns {"acc": (M, S, ny, nx) mm since run start, "lat", "lon"
    (ascending), "numbers" (0 = control), "steps"}. One .npy per step under ``cache``; resumable."""
    dl = _Deadline(budget_s)
    d = cache / f"ec_{model}_{run:%Y%m%d%H}"
    d.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for step in steps:
            f = d / f"tp_{step:03d}.npy"
            if f.exists():
                continue
            dl.check(f"ECMWF {model} {run:%d/%m %H}Z: stopped before step {step}")
            todo = []
            for sfx in _ECMWF[model][1]:
                stem = _ecmwf_stem(run, step, model, sfx, mirror)
                raw = _get(stem + ".index")
                if raw is None:
                    raise FileNotFoundError(stem + ".index")
                for line in raw.decode().splitlines():
                    if '"tp"' not in line:
                        continue
                    j = json.loads(line)
                    if j.get("param") == "tp" and j.get("levtype") == "sfc":
                        todo.append((int(j["number"]) if j.get("type") == "pf" else 0, stem + ".grib2", j["_offset"], j["_length"]))
            todo.sort()
            res = list(ex.map(lambda t: _ecmwf_field(t[1], t[2], t[3]), todo))
            if not (d / "grid.npz").exists():
                np.savez(d / "grid.npz", lat=res[0][1], lon=res[0][2], numbers=np.array([t[0] for t in todo], np.int16))
            _save(f, np.stack([r[0] for r in res]))
    g = np.load(d / "grid.npz")
    return {"acc": np.stack([np.load(d / f"tp_{s:03d}.npy") for s in steps], axis=1), "lat": g["lat"], "lon": g["lon"],
            "numbers": g["numbers"], "steps": list(steps)}


def ecmwf_ens_members(run: datetime, cache: Path, first_step: int = 6, last_step: int = 66, model: str = "ifs",
                      mirror: str = "gcs", budget_s: float | None = None, weight: float = 1.0,
                      every_h: int | None = None) -> list[Member]:
    """ECMWF ensemble as Members: IFS ENS (50 perturbed members; the control left the open ``enfo`` stream,
    it is the deterministic ``oper`` run) at 3-hourly steps, or AIFS-ENS (control + 50) at 6-hourly steps.
    ``every_h`` (a multiple of the native step, e.g. 12 for the long range) thins the steps: the cost is
    proportional to their number (12.4 MB per IFS step, 16.7 MB per AIFS step)."""
    dh = every_h or _ECMWF[model][2]
    lim = 360 if (model == "aifs" or run.hour in (0, 12)) else _ECMWF[model][3]
    steps = [s for s in range(first_step - first_step % dh, last_step + 1, dh)
             if s <= lim and (s <= 144 or s % 6 == 0)]
    z = ecmwf_ens_fetch(run, steps, cache, model, mirror, budget_s)
    rg = _regridder("ec025", z["lat"], z["lon"])
    label, mid = ("ENS", "ifs_ens3h") if model == "ifs" else ("AIFS-ENS", "aifs_ens")
    return _members(z["acc"], rg, run, steps, z["numbers"], label, "ens", mid, weight)


# ----------------------------------------------------------------------------- DWD ICON-EU-EPS

DWD = "https://opendata.dwd.de/weather/nwp/icon-eu-eps/grib"


def _icon_url(run: datetime, step: int) -> str:
    return f"{DWD}/{run:%H}/tot_prec/icon-eu-eps_europe_icosahedral_single-level_{run:%Y%m%d%H}_{step:03d}_tot_prec.grib2.bz2"


def icon_eu_eps_steps(first: int = 0, last: int = 54, step_h: int = 3) -> list[int]:
    """Steps the model really has: hourly to 48 h, 3-hourly to 72 h, 6-hourly to 120 h."""
    native = list(range(0, 49)) + list(range(51, 73, 3)) + list(range(78, 121, 6))
    return [s for s in native if first <= s <= last and (s % step_h == 0 or s > 72)]


def icon_eu_eps_latest_run(now: datetime, last_step: int = 54) -> datetime | None:
    """Newest run (00/06/12/18) whose ``last_step`` file is on the server. DWD keeps one day of runs."""
    n = _naive(now)
    base = n.replace(hour=(n.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for k in range(4):
        run = base - timedelta(hours=6 * k)
        if n - run >= timedelta(hours=2) and _exists(_icon_url(run, last_step)):
            return run
    return None


def _icon_grid(run: datetime, cache: Path) -> dict:
    """Cell centres of the icosahedral grid inside the box (+0.5 deg); downloaded once (2 x 0.17 MB)."""
    f = cache / "grid_icon_eu_eps.npz"
    if f.exists():
        return dict(np.load(f))
    out = {}
    for name in ("clat", "clon"):
        raw = _get(f"{DWD}/{run:%H}/{name}/icon-eu-eps_europe_icosahedral_time-invariant_{run:%Y%m%d%H}_{name}.grib2.bz2")
        if raw is None:
            raise FileNotFoundError(f"ICON-EU-EPS {name}")
        out[name] = _decode(_split(bz2.decompress(raw))[0])[0]
    lat, lon = out["clat"], out["clon"]
    idx = np.nonzero((lon >= BOX[0] - 0.5) & (lon <= BOX[2] + 0.5) & (lat >= BOX[1] - 0.5) & (lat <= BOX[3] + 0.5))[0]
    g = {"idx": idx, "lat": lat[idx].astype(np.float32), "lon": lon[idx].astype(np.float32), "n": np.array(lat.size)}
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, **g)
    return g


def icon_eu_eps_fetch(run: datetime, steps: list[int], cache: Path, budget_s: float | None = None, workers: int = 3) -> dict:
    """tot_prec of the 40 members at ``steps``: {"acc": (40, S, ncell) mm since run start, "lat", "lon"}.
    Every step is one bz2 file with the 40 members on the whole European grid (3-7 MB, no range requests)."""
    dl = _Deadline(budget_s)
    g = _icon_grid(run, cache)
    d = cache / f"icon_eu_eps_{run:%Y%m%d%H}"
    d.mkdir(parents=True, exist_ok=True)

    def one(step: int) -> None:
        f = d / f"tp_{step:03d}.npy"
        if f.exists():
            return
        if step == 0:
            return _save(f, np.zeros((40, g["idx"].size), np.float32))     # nothing accumulated yet
        if dl.t_end is not None and time.monotonic() > dl.t_end:
            return
        raw = _get(_icon_url(run, step))
        if raw is None:
            raise FileNotFoundError(_icon_url(run, step))
        rows = {}
        for m in _split(bz2.decompress(raw)):
            v, info = _decode(m, ("perturbationNumber",))
            if v.size != int(g["n"]):
                raise OSError(f"ICON-EU-EPS grid changed: {v.size} points")
            rows[int(info["perturbationNumber"])] = v[g["idx"]]
        _save(f, np.stack([rows[k] for k in sorted(rows)]).astype(np.float32))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, steps))
    missing = [s for s in steps if not (d / f"tp_{s:03d}.npy").exists()]
    if missing:
        raise Incomplete(f"ICON-EU-EPS {run:%d/%m %H}Z: {len(missing)} of {len(steps)} steps left")
    return {"acc": np.stack([np.load(d / f"tp_{s:03d}.npy") for s in steps], axis=1), "lat": g["lat"], "lon": g["lon"],
            "steps": list(steps)}


def icon_eu_eps_members(run: datetime, cache: Path, first_step: int = 0, last_step: int = 54, step_h: int = 3,
                        budget_s: float | None = None, weight: float = 1.0) -> list[Member]:
    """DWD ICON-EU-EPS, 40 members, ~13 km over the box (parameterised convection)."""
    steps = icon_eu_eps_steps(first_step, last_step, step_h)
    z = icon_eu_eps_fetch(run, steps, cache, budget_s)
    rg = _regridder("icon_eu_eps", z["lat"], z["lon"], points=True)
    return _members(z["acc"], rg, run, steps, np.arange(1, z["acc"].shape[0] + 1), "ICON-EU-EPS", "eps", "icon_eu_eps", weight)


# ------------------------------------------------------------------- Meteo-France AROME-IFS 0.025

MF_PNT = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net"
_AROME_IFS_BLOCKS = ("00H06H", "07H12H", "13H18H", "19H24H", "25H30H", "31H36H", "37H42H", "43H48H", "49H51H")


def _arome_ifs_url(run: datetime, block: str) -> str:
    r = f"{run:%Y-%m-%dT%H}:00:00Z"
    return f"{MF_PNT}/pnt/{r}/aromeifs/0025/SP1/aromeifs__0025__SP1__{block}__{r}.grib2"


def arome_ifs_latest_run(now: datetime) -> datetime | None:
    """Newest AROME-IFS run (00/06/12/18) with its last package (+49..51 h) published."""
    n = _naive(now)
    base = n.replace(hour=(n.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for k in range(5):
        run = base - timedelta(hours=6 * k)
        if n - run >= timedelta(hours=4) and _exists(_arome_ifs_url(run, _AROME_IFS_BLOCKS[-1])):
            return run
    return None


def _is_tp(head: bytes) -> bool:
    """Is this message Meteo-France's total precipitation (discipline 0, category 1, number 52, accumulated)?"""
    s = _sections(head)
    if head[6] != 0 or 4 not in s:
        return False
    o4 = s[4][0]
    return (head[o4 + 9], head[o4 + 10]) == (1, 52) and int.from_bytes(head[o4 + 7:o4 + 9], "big") == 8


def _arome_ifs_block(url: str) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """tp messages of one SP1 package: [(step, values [ny, nx] south->north, lat, lon)]. The package has no
    index: message headers are read one by one (variables come in blocks, tp is the 4th) and the tp block
    (6 messages, ~2 MB) is then fetched in a single range request."""
    off, out, buf, buf0 = 0, [], b"", 0
    for _ in range(400):
        head = buf[off - buf0:off - buf0 + 256] if buf0 <= off and off - buf0 + 256 <= len(buf) else _get(url, (off, off + 255))
        if not head or head[:4] != b"GRIB":
            break
        ln = int.from_bytes(head[8:16], "big")
        if _is_tp(head):
            if not (buf0 <= off and off - buf0 + ln <= len(buf)):
                buf0, buf = off, _get(url, (off, off + int(ln * 7.2)))          # the rest of the tp block
                if len(buf) < ln:
                    raise OSError("short read")
            v, info = _decode(buf[off - buf0:off - buf0 + ln], _LL_KEYS + ("endStep", "units"))
            lat, lon = _latlon_axes(info)
            rows, cols = _box_window(lat, lon)
            box = v.reshape(info["Nj"], info["Ni"])[np.ix_(rows, cols)]
            if lat[0] > lat[-1]:
                box, rows = box[::-1], rows[::-1]
            out.append((int(info["endStep"]), box.astype(np.float32), lat[rows].astype(np.float32), lon[cols].astype(np.float32)))
        elif out:
            break                                                               # past the tp block
        off += ln
    return out


def arome_ifs_member(run: datetime, cache: Path, budget_s: float | None = None, weight: float = 1.0,
                     fill_nan: float | None = None) -> Member:
    """Meteo-France AROME-IFS (AROME 1.3 km started from and driven by ECMWF IFS, delivered at 0.025 deg):
    a convection-permitting scenario whose large scale does not come from ARPEGE. Hourly to +51 h.
    Cells south of ~37.9 N are outside the model: NaN unless ``fill_nan`` is given (``ingest.fill_gaps``
    fills them from a coarser run when the member goes through it)."""
    dl = _Deadline(budget_s)
    d = cache / f"arome_ifs_{run:%Y%m%d%H}"
    d.mkdir(parents=True, exist_ok=True)

    def one(block: str) -> None:
        f = d / f"tp_{block}.npz"
        if f.exists() or (dl.t_end is not None and time.monotonic() > dl.t_end):
            return
        msgs = _arome_ifs_block(_arome_ifs_url(run, block))
        if not msgs:
            raise FileNotFoundError(f"no tp in {block}")
        tmp = d / f"tp_{block}.tmp.npz"
        np.savez(tmp, steps=np.array([m[0] for m in msgs], np.int16), acc=np.stack([m[1] for m in msgs]),
                 lat=msgs[0][2], lon=msgs[0][3])
        tmp.replace(f)

    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(one, _AROME_IFS_BLOCKS))
    missing = [b for b in _AROME_IFS_BLOCKS if not (d / f"tp_{b}.npz").exists()]
    if missing:
        raise Incomplete(f"AROME-IFS {run:%d/%m %H}Z: {len(missing)} packages left")
    steps, acc, lat, lon = [0], [], None, None
    for b in _AROME_IFS_BLOCKS:
        z = np.load(d / f"tp_{b}.npz")
        lat, lon = z["lat"], z["lon"]
        steps += [int(s) for s in z["steps"]]
        acc.append(z["acc"])
    acc = np.concatenate(acc)
    acc = np.concatenate([np.where(np.isfinite(acc[:1]), 0.0, np.nan), acc]).astype(np.float32)
    order = np.argsort(steps)
    steps, acc = [steps[i] for i in order], acc[order]
    t_end, p, big = _hourly(run, steps, _regridder("arome_ifs", lat, lon)(acc))
    nan_frac = float(np.isnan(p).mean())
    if fill_nan is not None:
        p = np.nan_to_num(p, nan=fill_nan)
    return Member(f"AROME-IFS 2,5 km · {run:%d/%m %H}Z", "cp", "arome_ifs", run, t_end, p, big, weight, meta={"nan_frac": nan_frac})


# --------------------------------------------------------------------------- NOAA GEFS (AWS)

GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"


def _gefs_url(run: datetime, num: int, step: int) -> str:
    mem = "gec00" if num == 0 else f"gep{num:02d}"
    return f"{GEFS}/gefs.{run:%Y%m%d}/{run:%H}/atmos/pgrb2sp25/{mem}.t{run:%H}z.pgrb2s.0p25.f{step:03d}"


def gefs_latest_run(now: datetime, last_step: int = 66) -> datetime | None:
    n = _naive(now)
    base = n.replace(hour=(n.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for k in range(5):
        run = base - timedelta(hours=6 * k)
        if n - run >= timedelta(hours=3) and _exists(_gefs_url(run, 30, last_step) + ".idx"):
            return run
    return None


def _gefs_field(run: datetime, num: int, step: int):
    """6-h precipitation bucket ending at ``step`` (a multiple of 6) of one member, cropped to the box."""
    url = _gefs_url(run, num, step)
    idx = _get(url + ".idx")
    if idx is None:
        raise FileNotFoundError(url + ".idx")
    lines = idx.decode().strip().splitlines()
    for i, line in enumerate(lines):
        p = line.split(":")
        if p[3] == "APCP":
            if f"{step - 6}-{step} hour acc" not in p[5]:
                raise OSError(f"GEFS bucket is '{p[5]}', expected {step - 6}-{step} h")
            a = int(p[1])
            b = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else a + 2_000_000
            v, info = _decode(_get(url, (a, b)), _LL_KEYS)
            lat, lon = _latlon_axes(info)
            rows, cols = _box_window(lat, lon)
            box = v.reshape(info["Nj"], info["Ni"])[np.ix_(rows, cols)]
            if lat[0] > lat[-1]:
                box, rows = box[::-1], rows[::-1]
            return box.astype(np.float32), lat[rows].astype(np.float32), lon[cols].astype(np.float32)
    raise OSError(f"no APCP in {url}.idx")


def gefs_members(run: datetime, cache: Path, first_step: int = 6, last_step: int = 66, budget_s: float | None = None,
                 weight: float = 1.0) -> list[Member]:
    """NOAA GEFS, control + 30 members, 0.25 deg, 6-h buckets (whole messages, 0.28 MB each)."""
    dl = _Deadline(budget_s)
    d = cache / f"gefs_{run:%Y%m%d%H}"
    d.mkdir(parents=True, exist_ok=True)
    first_step += -first_step % 6
    steps = list(range(first_step, last_step + 1, 6))
    with ThreadPoolExecutor(max_workers=8) as ex:
        for step in steps[1:]:
            f = d / f"tp_{step:03d}.npy"
            if f.exists():
                continue
            dl.check(f"GEFS {run:%d/%m %H}Z: stopped before step {step}")
            res = list(ex.map(lambda k: _gefs_field(run, k, step), range(31)))
            if not (d / "grid.npz").exists():
                np.savez(d / "grid.npz", lat=res[0][1], lon=res[0][2])
            _save(f, np.stack([r[0] for r in res]))
    g = np.load(d / "grid.npz")
    bucket = np.stack([np.load(d / f"tp_{s:03d}.npy") for s in steps[1:]], axis=1)        # (31, S-1, ny, nx)
    acc = np.concatenate([np.zeros_like(bucket[:, :1]), np.cumsum(bucket, axis=1)], axis=1)
    rg = _regridder("gefs025", g["lat"], g["lon"])
    return _members(acc, rg, run, steps, np.arange(31), "GEFS", "ens", "gefs", weight)


# ------------------------------------------------------- deterministic models on the Open-Meteo S3 mirror

# Not in ingest.MODELS today. hours = how far ahead to read; family as in params.py.
EXTRA_S3 = {
    # IFS at its native ~9 km, hourly to +90 h: the same forecast as "ifs" (0.25 deg, 3-hourly) with its detail
    "ifs_hres": dict(s3="ecmwf_ifs", family="regional", label="IFS 9 km", hours=96, lags=2),
    "gfs": dict(s3="ncep_gfs013", family="global", label="GFS 0,12°", hours=96, lags=2),
    "icon": dict(s3="dwd_icon", family="global", label="ICON 13 km", hours=96, lags=1),
    "gem": dict(s3="cmc_gem_gdps_15km", family="global", label="GEM 15 km", hours=96, lags=1),
    "ukmo": dict(s3="ukmo_global_deterministic_10km", family="regional", label="UKMO 10 km", hours=96, lags=1),  # CC BY-SA
    "aifs": dict(s3="ecmwf_aifs025_single", family="global", label="AIFS 0,25°", hours=192, lags=1),
    # AROME-PI, the nowcasting AROME: a new run EVERY HOUR, 15-min amounts to +6 h, complete ~35 min after run time
    "arome_pi": dict(s3="meteofrance_arome_france_hd_15min", family="cp", label="AROME-PI 1,3 km", hours=7, lags=3, weight=0.5),
}


def _quarters_to_hours(times: list[datetime], vals: np.ndarray) -> tuple[list[datetime], np.ndarray]:
    """15-min amounts (T, ...) -> hourly sums ending on the hour; only hours with their four quarters."""
    ends = [(_naive(t) + timedelta(minutes=59)).replace(minute=0, second=0, microsecond=0) for t in times]
    out_t, out_v = [], []
    for h in sorted(set(ends)):
        k = [i for i, e in enumerate(ends) if e == h and np.isfinite(vals[i]).any()]
        if len(k) == 4:
            out_t.append(h)
            out_v.append(vals[k].sum(axis=0))
    if not out_t:
        return [], vals[:0]
    # to_hourly() skips an all-NaN first field and needs a predecessor one hour earlier for the first real one
    return [out_t[0] - timedelta(hours=1)] + out_t, np.stack([np.full_like(out_v[0], np.nan)] + out_v)


def s3_member(key: str, run: datetime, t_from: datetime, t_to: datetime, finest_only: bool = True) -> Member | None:
    """One run of an EXTRA_S3 model as a Member (same reading path as ``ingest.load_run``; also handles the
    reduced Gaussian grid of ``ecmwf_ifs``, which comes as a list of points). ``finest_only`` stops the
    member where the model's output step gets longer (IFS: hourly to +90 h), so that it keeps
    ``native_step_h`` = 1 and its hourly amounts count."""
    from . import openmeteo_s3 as om
    from .. import ingest
    cfg = EXTRA_S3[key]
    lo, hi = om._utc(t_from), om._utc(t_to)
    allt = om.valid_times(cfg["s3"], run)
    if finest_only and len(allt) > 2:
        d0 = allt[1] - allt[0]
        cut = next((i for i in range(1, len(allt)) if allt[i] - allt[i - 1] > d0), len(allt))
        allt = allt[:cut]
    vt = [t for t in allt if lo <= t <= hi]
    if len(vt) < 2:
        return None
    times, lats, lons, out = om.read_series(cfg["s3"], run, "precipitation", times=vt, max_workers=6)
    vals = out["precipitation"]
    rg = _regridder(key, np.asarray(lats), np.asarray(lons), points=vals.ndim == 2)
    vals = rg(vals)
    if len(times) > 1 and (times[1] - times[0]) < timedelta(hours=1):
        times, vals = _quarters_to_hours(times, vals)
        if not times:
            return None
    t_end, p, step = ingest.to_hourly(times, vals)
    if len(t_end) == 0:
        return None
    r = _naive(run)
    return Member(f"{cfg['label']} · {r:%d/%m %H}Z", cfg["family"], key, r, t_end, p, step, float(cfg.get("weight", 1.0)),
                  meta={"nan_frac": float(np.isnan(p).mean())})


# ------------------------------------------------------------------------------- one call per cycle

def _cached(cache: Path, tag: str, run: datetime, build) -> list[Member]:
    """Members of one run, kept on the Riuà grid in a single npz so later cycles only read a file."""
    f = cache / f"{tag}_{run:%Y%m%d%H}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        t_end, p, step = z["t_end"], z["p"].astype(np.float32), int(z["step"])     # each z[...] decompresses: once
        return [Member(str(n), meta["family"], meta["model"], run, t_end, p[k], step,
                       float(meta["weight"]), meta={"nan_frac": meta.get("nan_frac", 0.0)})
                for k, n in enumerate(z["names"])]
    ms = build()
    ms = ms if isinstance(ms, list) else [ms]
    tmp = cache / f"{tag}_{run:%Y%m%d%H}.tmp.npz"
    np.savez_compressed(tmp, names=np.array([m.name for m in ms]), t_end=ms[0].t_end, step=ms[0].native_step_h,
                        p=np.stack([m.p for m in ms]).astype(np.float16),
                        meta=json.dumps({"family": ms[0].family, "model": ms[0].model, "weight": ms[0].weight,
                                         "nan_frac": ms[0].meta.get("nan_frac", 0.0)}))
    tmp.replace(f)
    return ms


def _prune(cache: Path, tag: str, keep: list[datetime]) -> None:
    """Drop the part folders of finished runs and the files of runs that are no longer used."""
    names = {f"{tag}_{r:%Y%m%d%H}.npz" for r in keep}
    for old in cache.glob(f"{tag}_2*"):
        if old.is_dir():
            if old.name + ".npz" in names or old.name < min(names, default=""):
                shutil.rmtree(old, ignore_errors=True)       # kept only while its run is still downloading
        elif old.name not in names:
            old.unlink(missing_ok=True)


def extra_members(cache: Path, now: datetime, which: tuple[str, ...] = ("ens3h", "icon_eu_eps", "arome_ifs", "ifs_hres"),
                  budget_s: float = 45.0) -> tuple[list[Member], list[dict]]:
    """Everything worth adding to the 6-48 h horizon, newest run of each source. A source that is not
    complete within ``budget_s`` (shared by all) keeps what it downloaded and falls back to its previous
    run; it is finished in the following cycles. Returns (members, source report rows)."""
    from . import openmeteo_s3 as om
    cache.mkdir(parents=True, exist_ok=True)
    t_stop = time.monotonic() + budget_s
    left = lambda: max(0.0, t_stop - time.monotonic())
    n = _naive(now)
    specs = {
        "ens3h": ("ec_ifs", "ECMWF ENS 50 miembros, cada 3 h", "ens", lambda: ecmwf_ens_latest_run(n, 66),
                  lambda r: ecmwf_ens_members(r, cache, 6, 66, budget_s=left())),
        "aifs_ens": ("ec_aifs", "ECMWF AIFS-ENS 51 miembros", "ens", lambda: ecmwf_ens_latest_run(n, 192, "aifs"),
                     lambda r: ecmwf_ens_members(r, cache, 24, 192, "aifs", budget_s=left(), every_h=12)),
        "icon_eu_eps": ("icon_eu_eps", "ICON-EU-EPS 40 miembros", "eps", lambda: icon_eu_eps_latest_run(n, 54),
                        lambda r: icon_eu_eps_members(r, cache, budget_s=left())),
        "arome_ifs": ("arome_ifs", "AROME-IFS 2,5 km", "cp", lambda: arome_ifs_latest_run(n),
                      lambda r: arome_ifs_member(r, cache, budget_s=left())),
        "gefs": ("gefs", "GEFS 31 miembros", "ens", lambda: gefs_latest_run(n, 66),
                 lambda r: gefs_members(r, cache, budget_s=left())),
    }
    members, report = [], []
    for key in which:
        if key in EXTRA_S3:
            cfg = EXTRA_S3[key]
            rep = {"id": key, "label": cfg["label"], "ok": False, "family": cfg["family"]}
            try:
                got = []
                for r in om.latest_runs(cfg["s3"], n=cfg["lags"]):
                    r = _naive(r)
                    ms = _cached(cache, "s3_" + key, r, lambda r=r: s3_member(key, r, n - timedelta(hours=14), n + timedelta(hours=cfg["hours"])))
                    members += ms
                    got.append(r)
                _prune(cache, "s3_" + key, got)
                rep.update(ok=bool(got), runs=[f"{r:%Y-%m-%dT%H:%MZ}" for r in got])
            except Exception as e:  # a missing source must not stop the cycle
                rep["error"] = f"{type(e).__name__}: {e}"[:160]
            report.append(rep)
            continue
        tag, label, family, latest, build = specs[key]
        rep = {"id": key, "label": label, "ok": False, "family": family}
        try:
            run = latest()
            done = sorted(cache.glob(f"{tag}_2*[0-9].npz"))
            try:
                if run is None:
                    raise Incomplete("no run published yet")
                ms = _cached(cache, tag, run, lambda: build(run))
            except Incomplete as e:
                rep["pending"] = str(e)[:160]
                if not done:
                    raise
                run = datetime.strptime(done[-1].stem.rsplit("_", 1)[1], "%Y%m%d%H")
                ms = _cached(cache, tag, run, lambda: build(run))
            members += ms
            _prune(cache, tag, [run])
            rep.update(ok=True, runs=[f"{run:%Y-%m-%dT%H:%MZ}"], n=len(ms))
        except Exception as e:
            rep["error"] = f"{type(e).__name__}: {e}"[:160]
        report.append(rep)
    return members, report


# ------------------------------------------------------------------------------------- smoke test

def _all_transfer() -> tuple[int, int]:
    from . import openmeteo_s3 as om
    (r, b), (r2, b2) = transfer(), om.transfer_stats()
    return r + r2, b + b2


def _describe(tag: str, ms: list[Member], t0: float, r0: int, b0: int) -> None:
    r1, b1 = _all_transfer()
    p = np.stack([m.p for m in ms])
    print(f"{tag}: {len(ms)} members, p {p[0].shape}, {ms[0].t_end[0]} .. {ms[0].t_end[-1]}, step {ms[0].native_step_h} h, "
          f"NaN {100 * np.isnan(p).mean():.1f} % | {time.time() - t0:.1f} s, {r1 - r0} requests, {(b1 - b0) / 1e6:.1f} MB")
    tot = np.nansum(p, axis=1)
    print(f"   total over the period: domain mean {np.nanmean(tot):.1f} mm (members {np.nanmean(tot, axis=(1, 2)).min():.1f} .. "
          f"{np.nanmean(tot, axis=(1, 2)).max():.1f}), cell max {np.nanmax(tot):.0f} mm, max 1 h {np.nanmax(p):.1f} mm")


def _smoke(what: str) -> None:
    import tempfile
    cache = Path(tempfile.gettempdir()) / "riua_extra_smoke"
    now = datetime.now(timezone.utc)
    jobs = {
        "ens": lambda: ecmwf_ens_members(ecmwf_ens_latest_run(now, 66), cache, 6, 66),
        "aifs": lambda: ecmwf_ens_members(ecmwf_ens_latest_run(now, 72, "aifs"), cache, 24, 72, "aifs"),
        "icon": lambda: icon_eu_eps_members(icon_eu_eps_latest_run(now), cache),
        "aromeifs": lambda: [arome_ifs_member(arome_ifs_latest_run(now), cache)],
        "gefs": lambda: gefs_members(gefs_latest_run(now), cache),
        "s3": lambda: [s3_member("ifs_hres", _latest_s3("ifs_hres"), now - timedelta(hours=6), now + timedelta(hours=60))],
    }
    for k, fn in jobs.items():
        if what in (k, "all"):
            t0, (r0, b0) = time.time(), _all_transfer()
            _describe(k, fn(), t0, r0, b0)


def _latest_s3(key: str) -> datetime:
    from . import openmeteo_s3 as om
    return _naive(om.latest_runs(EXTRA_S3[key]["s3"], 1)[0])


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    _smoke(sys.argv[1] if len(sys.argv) > 1 else "all")
