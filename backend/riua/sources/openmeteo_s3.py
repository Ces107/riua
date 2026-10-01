"""Live NWP fields for the Riuà box from Open-Meteo's public S3 bucket (``openmeteo``, CC-BY-4.0).

Primary path: ``data_spatial/<model>/YYYY/MM/DD/HHMMZ/<valid YYYY-MM-DDTHHMM>.om`` — one OM file per valid time that
holds every variable of the model as a 2-D ``[ny, nx]`` array (32x32 chunks, int16 + scale factor). Only the tail
of the file (metadata) and the chunks that intersect the box are downloaded, through plain HTTPS range requests.

Dependencies: ``numpy`` and ``omfiles`` (binary wheel, no fsspec / s3fs / aiohttp / pyproj needed).

What is NOT on S3 (verified 2026-10-01): individual ensemble members. Every ``*_ensemble`` / ``*_eps`` model in the
bucket carries one variable only, ``precipitation_probability``. Members are fetched from the Open-Meteo Ensemble
HTTP API instead, see :func:`read_ensemble_api` (free tier, non-commercial, rate limited).

Conventions (all verified against real files, see coord/findings/r1-nwp-live.md):

* arrays are ``[y, x]`` with row 0 = SOUTH and column 0 = WEST;
* regular lat/lon models: ``lat[i] = S + i*(N-S)/(ny-1)``, ``lon[j] = W + j*(E-W)/(nx-1)`` where S, W, N, E come
  from ``BBOX[S,W,N,E]`` inside the file's ``crs_wkt``;
* ``ecmwf_ifs`` (9 km) is an O1280 reduced Gaussian grid stored as ``[1, 6599680]``: results are 1-D point lists;
* ``precipitation`` (and showers, rain, snowfall) is the sum over the PRECEDING model step (mm), not since init;
  the first valid time of a run (the analysis) has no precipitation;
* cells outside a model's native domain are NaN (AROME France is NaN south of ~37.9-38.0N inside the box).

Times are timezone-aware UTC ``datetime`` objects (naive datetimes are taken as UTC).
"""
from __future__ import annotations

import http.client
import json
import math
import re
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_BBOX", "latest_runs", "list_runs", "run_meta", "valid_times", "file_url", "read_field", "read_fields",
    "read_series", "read_ensemble_api", "transfer_stats", "MembersNotOnS3",
]

S3_HOST = "https://openmeteo.s3.amazonaws.com"
#: (west, south, east, north) — Comunitat Valenciana + upstream basins.
DEFAULT_BBOX = (-2.4, 37.6, 0.8, 41.0)
USER_AGENT = "riua/0.1 (+https://github.com/; flood-risk research; python-http.client)"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_EPS = 1e-6

#: S3 model id -> model name understood by ensemble-api.open-meteo.com (members only exist there).
#: All names below were accepted by the API on 2026-10-01 (members incl. control in brackets).
ENSEMBLE_API_MODELS = {
    "ecmwf_ifs025_ensemble": "ecmwf_ifs025_ensemble",          # [51] 0.25 deg, 3-hourly, 15 d
    "ecmwf_ifs_europe_ensemble": "ecmwf_ifs_europe_ensemble",  # [51] native O1280 ~9 km, HOURLY, 15 d (API only)
    "ecmwf_aifs025_ensemble": "ecmwf_aifs025",                 # [51] 0.25 deg, 6-hourly, 15 d
    "dwd_icon_eu_eps": "icon_eu",                              # [40] ~13 km, hourly, 5 d
    "dwd_icon_eps": "icon_global",                             # [40] ~26 km, 7.5 d
    "ncep_gefs025": "gfs025",                                  # [31] 0.25 deg, 3-hourly, 10 d
    "ncep_gefs05": "gfs05",                                    # [31] 0.5 deg, 16 d
    "ukmo_global_ensemble_20km": "ukmo_global_ensemble_20km",  # [18] 20 km
}


class MembersNotOnS3(NotImplementedError):
    """Raised when individual ensemble members are requested from S3 (they are not published there)."""


# --------------------------------------------------------------------------------------------------------------
# HTTP layer: one keep-alive connection per thread, range requests, tail cache
# --------------------------------------------------------------------------------------------------------------
class RangeHTTP:
    """Tiny HTTPS client that is also the "filesystem" object ``omfiles.OmFileReader.from_fsspec`` wants
    (it only calls ``size(path)`` and ``cat_file(path, start=, end=)``).

    The first access to a file downloads its last ``tail_bytes`` in ONE suffix-range request; that gives the file
    size (``Content-Range``) and the whole metadata tree, so opening a file and finding a variable costs a single
    round trip instead of 10-600 tiny ones.
    """

    def __init__(self, tail_bytes: int = 65536, read_ahead: int = 49152, timeout: float = 30.0, retries: int = 3):
        self.tail_bytes = tail_bytes
        self.read_ahead = read_ahead
        self._segments: dict[str, list[tuple[int, bytes]]] = {}  # url -> [(offset, bytes)] read-ahead blocks
        self.timeout = timeout
        self.retries = retries
        self.n_requests = 0
        self.n_bytes = 0
        self._conns: dict[str, http.client.HTTPSConnection] = {}
        self._tails: dict[str, tuple[int, int, bytes]] = {}  # url -> (total size, tail start, tail bytes)

    # -- low level -------------------------------------------------------------------------------------------
    def request(self, method: str, url: str, headers: dict | None = None, body: bytes | None = None):
        u = urllib.parse.urlsplit(url)
        target = u.path + ("?" + u.query if u.query else "")
        hdrs = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        hdrs.update(headers or {})
        last: Exception | None = None
        for attempt in range(self.retries):
            conn = self._conns.get(u.netloc)
            if conn is None:
                conn = self._conns[u.netloc] = http.client.HTTPSConnection(u.netloc, timeout=self.timeout)
            try:
                conn.request(method, target, body=body, headers=hdrs)
                resp = conn.getresponse()
                data = resp.read()
            except (http.client.HTTPException, OSError) as exc:  # dropped keep-alive, timeout, DNS...
                last = exc
                conn.close()
                self._conns.pop(u.netloc, None)
                time.sleep(0.5 * (attempt + 1))
                continue
            self.n_requests += 1
            self.n_bytes += len(data)
            _STATS.add(1, len(data))
            if resp.status in (500, 502, 503, 504) and attempt + 1 < self.retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return resp.status, resp.headers, data
        raise OSError(f"{method} {url} failed after {self.retries} attempts: {last!r}")

    def get(self, url: str, headers: dict | None = None) -> bytes:
        status, _h, data = self.request("GET", url, headers)
        if status in (403, 404):
            raise FileNotFoundError(f"HTTP {status} for {url}")
        if status >= 400:
            raise OSError(f"HTTP {status} for {url}: {data[:200]!r}")
        return data

    def close(self) -> None:
        for c in self._conns.values():
            c.close()
        self._conns.clear()
        self._tails.clear()
        self._segments.clear()

    # -- fsspec-like API used by omfiles ---------------------------------------------------------------------
    def _tail(self, path: str) -> tuple[int, int, bytes]:
        t = self._tails.get(path)
        if t is None:
            status, hdr, data = self.request("GET", path, {"Range": f"bytes=-{self.tail_bytes}"})
            if status in (403, 404):
                raise FileNotFoundError(f"HTTP {status} for {path}")
            if status == 206:
                total = int(hdr["Content-Range"].rsplit("/", 1)[1])
            elif status == 200:
                total = len(data)
            else:
                raise OSError(f"HTTP {status} for {path}")
            if len(self._tails) >= 4:  # keep memory flat: at most 4 tails (256 KB) per thread
                self._tails.pop(next(iter(self._tails)))
            t = self._tails[path] = (total, total - len(data), data)
        return t

    def size(self, path: str, **_kw) -> int:
        return self._tail(path)[0]

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **_kw) -> bytes:
        total, tail_start, tail = self._tail(path)
        start = 0 if start is None else start
        end = total if end is None else end
        if start >= tail_start:
            return tail[start - tail_start:end - tail_start]
        segs = self._segments.setdefault(path, [])
        for s0, data in segs:
            if s0 <= start and end <= s0 + len(data):
                return data[start - s0:end - s0]
        # Forward read-ahead: omfiles asks for the chunks of a box in ascending order, a few hundred bytes to 2 kB
        # at a time with small gaps; one bigger request replaces 10-50 round trips (latency >> bandwidth here).
        stop = min(total, max(end, start + self.read_ahead))
        data = self.get(path, {"Range": f"bytes={start}-{stop - 1}"})
        if len(segs) >= 24:
            segs.pop(0)
        segs.append((start, data))
        return data[:end - start]

    def forget(self, path: str) -> None:
        """Drop everything cached for one file (called when a file has been read)."""
        self._segments.pop(path, None)
        self._tails.pop(path, None)


class _Stats:
    """Process-wide transfer counters (requests / bytes) — handy to log the cost of an ingestion cycle."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = 0
        self.bytes = 0

    def add(self, n: int, b: int) -> None:
        with self._lock:
            self.requests += n
            self.bytes += b


_STATS = _Stats()
_LOCAL = threading.local()


def transfer_stats() -> tuple[int, int]:
    """(HTTP requests, bytes downloaded) since the process started."""
    return _STATS.requests, _STATS.bytes


def _http() -> RangeHTTP:
    h = getattr(_LOCAL, "http", None)
    if h is None:
        h = _LOCAL.http = RangeHTTP()
    return h


def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _parse_time(s: str) -> datetime:
    return datetime.strptime(s.replace("Z", "")[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------------------------------------------
def _run_prefix(model: str, run: datetime) -> str:
    return f"data_spatial/{model}/{_utc(run):%Y/%m/%d/%H%MZ}/"


def file_url(model: str, run: datetime, valid_time: datetime, suffix: str = "") -> str:
    """URL of the spatial file of one valid time. ``suffix="_model-level"`` selects DWD's delayed side files."""
    return f"{S3_HOST}/{_run_prefix(model, run)}{_utc(valid_time):%Y-%m-%dT%H%M}{suffix}.om"


def run_meta(model: str, run: datetime | None = None, in_progress: bool = False) -> dict:
    """``meta.json`` of a run, or ``latest.json`` (newest COMPLETE run) / ``in-progress.json`` when ``run`` is None.

    Keys: completed, reference_time, last_modified_time, valid_times, variables, crs_wkt.
    """
    if run is None:
        url = f"{S3_HOST}/data_spatial/{model}/{'in-progress' if in_progress else 'latest'}.json"
    else:
        url = f"{S3_HOST}/{_run_prefix(model, run)}meta.json"
    return json.loads(_http().get(url))


def _list_prefixes(prefix: str) -> list[str]:
    q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "delimiter": "/"})
    root = ET.fromstring(_http().get(f"{S3_HOST}/?{q}"))
    return sorted(cp.find(_S3_NS + "Prefix").text for cp in root.findall(_S3_NS + "CommonPrefixes"))


def list_runs(model: str, day: datetime) -> list[datetime]:
    """All run folders present for one UTC day (complete or not), oldest first. One S3 LIST request."""
    out = []
    for p in _list_prefixes(f"data_spatial/{model}/{_utc(day):%Y/%m/%d}/"):
        m = re.search(r"(\d{4})/(\d{2})/(\d{2})/(\d{2})(\d{2})Z/$", p)
        if m:
            out.append(datetime(*map(int, m.groups()), tzinfo=timezone.utc))
    return out


def latest_runs(model: str, n: int = 4, max_days_back: int = 9) -> list[datetime]:
    """The ``n`` newest COMPLETE runs of ``model``, newest first.

    ``latest.json`` names the newest complete run; older ones are found by listing the day folders and checking
    ``completed`` in each run's ``meta.json`` (S3 keeps roughly the last 8-9 days).
    """
    newest = _parse_time(run_meta(model)["reference_time"])
    runs = [newest]
    day = newest
    for _ in range(max_days_back + 1):
        if len(runs) >= n:
            break
        for r in reversed(list_runs(model, day)):
            if r >= newest or r in runs:
                continue
            try:
                if run_meta(model, r).get("completed"):
                    runs.append(r)
            except FileNotFoundError:
                pass
            if len(runs) >= n:
                break
        day = day - timedelta(days=1)
    return runs[:n]


def valid_times(model: str, run: datetime) -> list[datetime]:
    """Sorted valid times of a run (some models list them unordered in meta.json)."""
    return sorted(_parse_time(t) for t in run_meta(model, run)["valid_times"])


# --------------------------------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------------------------------
def _wkt_bbox(crs_wkt: str) -> tuple[float, float, float, float]:
    m = re.search(r"BBOX\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]", crs_wkt)
    if not m:
        raise ValueError("crs_wkt has no BBOX[...]")
    s, w, n, e = map(float, m.groups())
    return s, w, n, e


def _regular_window(crs_wkt: str, shape: Sequence[int], bbox: Sequence[float]):
    """Index window + coordinate vectors of the cells whose centre lies inside ``bbox`` on a regular lat/lon grid."""
    s, w, n, e = _wkt_bbox(crs_wkt)
    ny, nx = int(shape[0]), int(shape[1])
    dlat, dlon = (n - s) / (ny - 1), (e - w) / (nx - 1)
    bw, bs, be, bn = bbox
    y0 = max(0, math.ceil((bs - s) / dlat - _EPS))
    y1 = min(ny, math.floor((bn - s) / dlat + _EPS) + 1)
    x0 = max(0, math.ceil((bw - w) / dlon - _EPS))
    x1 = min(nx, math.floor((be - w) / dlon + _EPS) + 1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"bbox {tuple(bbox)} does not intersect the model domain S,W,N,E={(s, w, n, e)}")
    lats = (s + dlat * np.arange(y0, y1)).astype(np.float32)
    lons = (w + dlon * np.arange(x0, x1)).astype(np.float32)
    return (y0, y1, x0, x1), lats, lons


def _gaussian_o1280_window(n_points: int, bbox: Sequence[float]):
    """Flat index range + per-point coordinates and mask for ECMWF's O1280 octahedral reduced Gaussian grid.

    Open-Meteo's own georeferencing (omfiles/grids/gaussian.py, mirrored here): rows run north -> south,
    ``lat(y) = (1280 - y - 1) * dy + dy/2`` with ``dy = 180 / (2*1280 + 0.5)``; row ``y`` of the northern hemisphere
    has ``20 + 4*y`` points starting at lon 0 going east, and starts at flat index ``2*y*y + 18*y``.
    """
    nlat = 1280
    if n_points != 4 * nlat * (nlat + 9):
        raise NotImplementedError(f"unsupported reduced Gaussian grid with {n_points} points")
    bw, bs, be, bn = bbox
    if bs < 0:
        raise NotImplementedError("Gaussian reader only implemented for the northern hemisphere")
    dy = 180.0 / (2 * nlat + 0.5)
    y_first = max(0, math.ceil(nlat - 1 - (bn - dy / 2) / dy - _EPS))  # northernmost row inside the box
    y_last = min(nlat - 1, math.floor(nlat - 1 - (bs - dy / 2) / dy + _EPS))
    if y_last < y_first:
        raise ValueError("bbox too thin for the Gaussian grid")
    a, b = 2 * y_first * y_first + 18 * y_first, 2 * (y_last + 1) ** 2 + 18 * (y_last + 1)
    lat_parts, lon_parts = [], []
    for y in range(y_first, y_last + 1):
        nx = 20 + 4 * y
        lon = np.arange(nx) * (360.0 / nx)
        lon[lon >= 180.0] -= 360.0
        lon_parts.append(lon)
        lat_parts.append(np.full(nx, (nlat - y - 1) * dy + dy / 2))
    lats, lons = np.concatenate(lat_parts), np.concatenate(lon_parts)
    mask = (lons >= bw - _EPS) & (lons <= be + _EPS)
    return (a, b), lats[mask].astype(np.float32), lons[mask].astype(np.float32), mask


# --------------------------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------------------------
def _open(url: str):
    from omfiles import OmFileReader  # imported lazily so that the API fallback works without the wheel

    return OmFileReader.from_fsspec(_http(), url)


def _read_from_root(root, variables: Iterable[str], bbox: Sequence[float]):
    crs = root.get_child_by_name("crs_wkt").read_scalar()
    gaussian = "Gaussian" in crs
    if not gaussian and not crs.lstrip().startswith('GEOGCRS["WGS 84"'):
        raise NotImplementedError(
            "projected / rotated grid (KNMI and DMI HARMONIE Europe): not implemented because their domain does not "
            "reach the Riuà box; see coord/findings/r1-nwp-live.md for the projection formulas")
    out: dict[str, np.ndarray] = {}
    lats = lons = None
    for name in variables:
        try:
            var = root.get_child_by_name(name)
        except Exception as exc:  # omfiles raises a generic error for unknown children
            raise KeyError(f"variable {name!r} not in file") from exc
        if gaussian:
            (a, b), lats, lons, mask = _gaussian_o1280_window(int(var.shape[-1]), bbox)
            vals = np.asarray(var.read_array((slice(0, 1), slice(a, b))), dtype=np.float32).reshape(-1)[mask]
        else:
            (y0, y1, x0, x1), lats, lons = _regular_window(crs, var.shape, bbox)
            vals = np.asarray(var.read_array((slice(y0, y1), slice(x0, x1))), dtype=np.float32)
            vals = vals.reshape(y1 - y0, x1 - x0)
        out[name] = vals
    return lats, lons, out


def read_fields(model: str, run: datetime, valid_time: datetime, variables: Sequence[str],
                bbox: Sequence[float] = DEFAULT_BBOX, suffix: str = ""):
    """Several variables of one valid time with a single file open. Returns ``(lats, lons, {name: values})``.

    Regular grids: 1-D ``lats`` (south -> north), 1-D ``lons`` (west -> east), values ``[len(lats), len(lons)]``.
    ``ecmwf_ifs`` (Gaussian): 1-D ``lats``, ``lons`` and values, one entry per grid point inside the box.
    """
    if model in ENSEMBLE_API_MODELS and any(v != "precipitation_probability" for v in variables):
        raise MembersNotOnS3(f"{model}: S3 only has 'precipitation_probability'; use read_ensemble_api() for members")
    url = file_url(model, run, valid_time, suffix)
    root = _open(url)
    try:
        return _read_from_root(root, variables, bbox)
    finally:
        root.close()
        _http().forget(url)


def read_field(model: str, run: datetime, valid_time: datetime, variable: str,
               bbox: Sequence[float] = DEFAULT_BBOX, member: int | None = None):
    """One variable of one valid time over ``bbox`` = (west, south, east, north).

    Deterministic models (S3): returns ``(lats, lons, values)`` float32, see :func:`read_fields`.

    Ensemble models (``ecmwf_ifs025_ensemble``, ``dwd_icon_eu_eps``, ``ncep_gefs025``...): members are not on S3.
    With ``variable="precipitation_probability"`` the S3 probability field is returned (2-D). For any other
    variable the Ensemble HTTP API is used — it only serves its LATEST run, so ``run`` is checked against it —
    and the result is ``[members, y, x]`` (``member=None``) or ``[y, x]`` (``member=k``, 0 = control).
    """
    if model in ENSEMBLE_API_MODELS and variable != "precipitation_probability":
        ens = read_ensemble_api(model, variable, bbox)
        if ens["run"] is not None and _utc(run) != ens["run"]:
            raise MembersNotOnS3(f"{model}: the Ensemble API only serves its latest run {ens['run']:%Y-%m-%dT%H:%MZ}, "
                                 f"not {_utc(run):%Y-%m-%dT%H:%MZ}")
        try:
            k = ens["times"].index(_utc(valid_time))
        except ValueError as exc:
            raise KeyError(f"{valid_time} not in the API time axis") from exc
        vals = ens["values"][:, k]
        return ens["lats"], ens["lons"], (vals if member is None else vals[member])
    if member is not None:
        raise ValueError(f"{model} is deterministic: member must be None")
    lats, lons, out = read_fields(model, run, valid_time, [variable], bbox)
    return lats, lons, out[variable]


def read_series(model: str, run: datetime, variables: Sequence[str] | str, bbox: Sequence[float] = DEFAULT_BBOX,
                times: Sequence[datetime] | None = None, max_workers: int = 4, skip_missing: bool = True):
    """Read one or more variables for many valid times of a run, a few files in parallel.

    Returns ``(times, lats, lons, {name: array [time, y, x]})``. A valid time whose file lacks a variable (e.g.
    precipitation at the analysis time) or does not exist yet is filled with NaN when ``skip_missing`` is true.
    """
    names = [variables] if isinstance(variables, str) else list(variables)
    tlist = [_utc(t) for t in times] if times is not None else valid_times(model, run)

    def one(t):
        try:
            return read_fields(model, run, t, names, bbox)
        except (KeyError, FileNotFoundError):
            if not skip_missing:
                raise
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(one, tlist))
    good = next((r for r in results if r is not None), None)
    if good is None:
        raise FileNotFoundError(f"no readable file for {model} run {run}")
    lats, lons, sample = good
    out = {}
    for name in names:
        arr = np.full((len(tlist),) + sample[name].shape, np.nan, dtype=np.float32)
        for i, r in enumerate(results):
            if r is not None:
                arr[i] = r[2][name]
        out[name] = arr
    return tlist, lats, lons, out


# --------------------------------------------------------------------------------------------------------------
# Ensemble members: HTTP API fallback
# --------------------------------------------------------------------------------------------------------------
def read_ensemble_api(model: str = "ecmwf_ifs025_ensemble", variable: str = "precipitation",
                      bbox: Sequence[float] = DEFAULT_BBOX, step: float = 0.25, forecast_days: int = 7,
                      chunk_locations: int = 100, wait_on_429: float = 61.0) -> dict:
    """All members of the LATEST run from ``ensemble-api.open-meteo.com`` on a regular ``step``-degree lattice.

    Returns ``{"run", "times", "lats", "lons", "cell_lats", "cell_lons", "values"}`` with ``values`` float32
    ``[member, time, y, x]`` (member 0 = control). ``cell_lats/cell_lons`` ``[y, x]`` are the model grid cells the
    API snapped each lattice point to (nearest cell). ``run`` is None when the API exposes no meta.json for the
    model. Time axis is hourly UTC starting 00:00 today; for 3-/6-hourly models the API spreads each step's sum
    evenly over its hours, so sums over whole steps are exact but hourly peaks are not
    (``ecmwf_ifs_europe_ensemble`` and ``dwd_icon_eu_eps`` are natively hourly).

    Cost on the free tier (non-commercial use only, 600/min, 5 000/h, 10 000/day): every location counts, and each
    member counts as a variable: ``weight = n_locations * n_members / 10`` (IFS 51 members, 0.25 deg over the default
    box = 182 points -> 928 per fetch). A request is admitted while the minute counter is below 600, hence the
    location chunks and the pause after an HTTP 429.
    """
    api_model = ENSEMBLE_API_MODELS.get(model, model)
    bw, bs, be, bn = bbox
    lats = np.arange(math.ceil(bs / step - _EPS), math.floor(bn / step + _EPS) + 1) * step
    lons = np.arange(math.ceil(bw / step - _EPS), math.floor(be / step + _EPS) + 1) * step
    pts = [(float(la), float(lo)) for la in lats for lo in lons]
    http_ = _http()
    times: list[datetime] | None = None
    values: np.ndarray | None = None
    cells = np.full((2, len(pts)), np.nan, dtype=np.float32)  # grid cell the API actually used for each point
    for c0 in range(0, len(pts), chunk_locations):
        chunk = pts[c0:c0 + chunk_locations]
        body = urllib.parse.urlencode({
            "latitude": ",".join(f"{p[0]:.4f}" for p in chunk), "longitude": ",".join(f"{p[1]:.4f}" for p in chunk),
            "hourly": variable, "models": api_model, "forecast_days": forecast_days, "timezone": "UTC",
            "cell_selection": "nearest"}, safe=",").encode()
        for attempt in range(4):
            status, _h, data = http_.request("POST", "https://ensemble-api.open-meteo.com/v1/ensemble",
                                             {"Content-Type": "application/x-www-form-urlencoded"}, body)
            if status != 429:
                break
            time.sleep(wait_on_429)
        if status != 200:
            raise OSError(f"ensemble API HTTP {status}: {data[:200]!r}")
        res = json.loads(data)
        del data
        res = res if isinstance(res, list) else [res]
        for i, loc in enumerate(res):
            hourly = loc["hourly"]
            cells[:, c0 + i] = (loc["latitude"], loc["longitude"])
            keys = [k for k in hourly if k == variable or k.startswith(variable + "_member")]
            keys.sort(key=lambda k: 0 if k == variable else int(k.rsplit("member", 1)[1]))
            if values is None:
                times = [datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc) for t in hourly["time"]]
                values = np.full((len(keys), len(times), len(pts)), np.nan, dtype=np.float32)
            for m, k in enumerate(keys):
                values[m, :, c0 + i] = np.array([np.nan if v is None else v for v in hourly[k]], dtype=np.float32)
        del res
    run = None
    try:
        meta = json.loads(http_.get(f"https://ensemble-api.open-meteo.com/data/{model}/static/meta.json"))
        run = datetime.fromtimestamp(meta["last_run_initialisation_time"], tz=timezone.utc)
    except (OSError, KeyError, ValueError):
        pass
    assert values is not None and times is not None
    return {"run": run, "times": times, "lats": lats.astype(np.float32), "lons": lons.astype(np.float32),
            "cell_lats": cells[0].reshape(len(lats), len(lons)), "cell_lons": cells[1].reshape(len(lats), len(lons)),
            "values": values.reshape(values.shape[0], values.shape[1], len(lats), len(lons))}


# --------------------------------------------------------------------------------------------------------------
# Smoke test:  python -m riua.sources.openmeteo_s3   (or: python openmeteo_s3.py [--no-ens])
# --------------------------------------------------------------------------------------------------------------
def _describe(tag: str, lats, lons, vals) -> None:
    v = np.asarray(vals)
    ok = np.isfinite(v)
    print(f"  {tag}: values{v.shape} {v.dtype} lats{np.shape(lats)} [{float(np.min(lats)):.3f}..{float(np.max(lats)):.3f}] "
          f"lons{np.shape(lons)} [{float(np.min(lons)):.3f}..{float(np.max(lons)):.3f}] "
          f"min={np.nanmin(v) if ok.any() else float('nan'):.2f} max={np.nanmax(v) if ok.any() else float('nan'):.2f} "
          f"NaN={100 * (1 - ok.mean()):.1f}%")


def _peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (Linux/macOS: resource, Windows: psapi)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1e6 if sys.platform == "darwin" else 1e3)
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1e6
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    return float("nan")


def _smoke(with_ensemble: bool = True) -> None:
    import tracemalloc

    tracemalloc.start()
    t0 = time.time()
    print("AROME HD (meteofrance_arome_france_hd)")
    runs = latest_runs("meteofrance_arome_france_hd", 4)
    print("  newest complete runs:", [f"{r:%Y-%m-%d %H}Z" for r in runs], f"({time.time() - t0:.1f}s)")
    run = runs[0]
    vts = valid_times("meteofrance_arome_france_hd", run)
    t1 = time.time()
    r0, b0 = transfer_stats()
    lats, lons, pr = read_field("meteofrance_arome_france_hd", run, vts[3], "precipitation")
    r1, b1 = transfer_stats()
    _describe(f"precipitation run {run:%d %HZ} valid {vts[3]:%d %HZ} [{time.time() - t1:.2f}s, {r1 - r0} req, {b1 - b0} B]", lats, lons, pr)
    t1 = time.time()
    times, lats, lons, out = read_series("meteofrance_arome_france_hd", run, ["precipitation", "cape"], times=vts[1:13])
    r2, b2 = transfer_stats()
    _describe(f"precipitation 12 valid times [{time.time() - t1:.2f}s, {r2 - r1} req, {(b2 - b1) / 1e3:.0f} kB]", lats, lons, out["precipitation"])
    print("  hourly box maximum (mm):", [round(float(x), 1) for x in np.nanmax(out["precipitation"], axis=(1, 2))])

    print("IFS 0.25 (ecmwf_ifs025), pressure level + PWAT")
    run = latest_runs("ecmwf_ifs025", 1)[0]
    vts = valid_times("ecmwf_ifs025", run)
    t1 = time.time()
    lats, lons, out = read_fields("ecmwf_ifs025", run, vts[4], ["temperature_850hPa", "total_column_integrated_water_vapour", "precipitation"])
    r3, b3 = transfer_stats()
    for k, v in out.items():
        _describe(f"{k} run {run:%d %HZ} valid {vts[4]:%d %HZ}", lats, lons, v)
    print(f"  [{time.time() - t1:.2f}s, {r3 - r2} req, {(b3 - b2) / 1e3:.0f} kB]")

    print("IFS ENS (ecmwf_ifs025_ensemble)")
    run = latest_runs("ecmwf_ifs025_ensemble", 1)[0]
    vts = valid_times("ecmwf_ifs025_ensemble", run)
    lats, lons, pp = read_field("ecmwf_ifs025_ensemble", run, vts[4], "precipitation_probability")
    _describe(f"S3 precipitation_probability run {run:%d %HZ} valid {vts[4]:%d %HZ}", lats, lons, pp)
    if with_ensemble:
        t1 = time.time()
        ens = read_ensemble_api("ecmwf_ifs025_ensemble", "precipitation", forecast_days=3)
        print(f"  API members: run {ens['run']}, {len(ens['times'])} hourly steps from {ens['times'][0]:%Y-%m-%d %HZ} [{time.time() - t1:.1f}s]")
        k = min(27, len(ens["times"]) - 1)
        _describe(f"precipitation members x y x at {ens['times'][k]:%d %HZ}", ens["lats"], ens["lons"], ens["values"][:, k])
        acc = np.nansum(ens["values"], axis=1)
        print("  72 h total, box maximum per member (mm):", np.round(np.nanmax(acc, axis=(1, 2)), 0).astype(int).tolist())
    req, nbytes = transfer_stats()
    print(f"TOTAL {time.time() - t0:.1f}s, {req} requests, {nbytes / 1e6:.2f} MB downloaded, "
          f"python peak alloc {tracemalloc.get_traced_memory()[1] / 1e6:.1f} MB, process peak RSS {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    _smoke(with_ensemble="--no-ens" not in sys.argv)
