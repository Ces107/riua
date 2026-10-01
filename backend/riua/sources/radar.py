"""Radar reflectivity for Riuà: fetch + decode to dBZ on a fixed lat/lon grid.

Three keyless sources, all verified from this machine on 2026-10-01 (see
``coord/findings/r3-radar.md``):

``rainviewer``  RainViewer public tiles (global mosaic; over Spain it is fed by
                AEMET).  PNG, colour scheme "Universal Blue", exact 1-dBZ steps
                from -10 to 65 dBZ.  10-min frames, last 2 h only.  Free for
                personal/educational use, attribution required.
``opera``       EUMETNET OPERA "CIRRUS" maximum-reflectivity composite, 1 km,
                5-min, float dBZ, CC BY 4.0.  Public S3 bucket (24 h cache +
                archive back to 2012).  Read with HTTP Range requests straight
                out of the cloud-optimised GeoTIFF (only the 4 tiles we need).
                CAVEAT: in Oct 2026 the Valencia (Cullera) and Murcia radars are
                NOT in the composite -> hole on the Valencia coast.
``aemet``       AEMET national composite from the public web API (tar.gz of
                GeoTIFF).  Official and includes Cullera, but coarse:
                0.026 deg pixels and 6-dBZ classes starting at 12 dBZ.

``blend``       (default) OPERA, with its uncovered cells filled from AEMET.

Output grid (same for every source): 340 x 320 cells of 0.01 deg, arrays
indexed [j, i] with j growing to the NORTH and i to the east, cell centres::

    lon = -2.4 + (i + 0.5) * 0.01   i = 0..319
    lat = 37.6 + (j + 0.5) * 0.01   j = 0..339

This is exactly a 5 x 5 refinement of the analysis grid in ``core/grid.py``
(0.05 deg, 68 x 64, same origin and orientation), so ``to_analysis_grid``
is a plain block reduction.

0.01 deg is ~1.11 km N-S and ~0.86 km E-W at 39.5N, which matches the native
~1 km of OPERA and of the RainViewer mosaic; a finer grid would only invent
detail.  Regridding is nearest-neighbour on purpose: dBZ is logarithmic and the
PNG sources are categorical, so interpolating/averaging would be wrong.

Values: float32 dBZ.  ``NaN`` = no radar coverage.  ``NO_ECHO_DBZ`` (-32) =
covered but nothing detected.

Only numpy + Pillow + requests are needed (pyproj/h5py are NOT required).

Smoke test (from ``backend/``)::

    py -3.11 -m riua.sources.radar                # all sources, 3 frames
    py -3.11 -m riua.sources.radar blend 13
"""
from __future__ import annotations

import os as _os
import sys as _sys

# Run as a plain script, Python puts this folder first on sys.path and the
# sibling ``warnings.py`` would shadow the standard library.  Drop it.
if __name__ == "__main__" and _sys.path and \
        _os.path.abspath(_sys.path[0]) == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)

import io
import math
import re
import struct
import tarfile
import time
import zlib
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
import requests
from PIL import Image

# --------------------------------------------------------------------------- grid
LON_MIN, LON_MAX = -2.4, 0.8
LAT_MIN, LAT_MAX = 37.6, 41.0
GRID_RES = 0.01
NX = int(round((LON_MAX - LON_MIN) / GRID_RES))  # 320
NY = int(round((LAT_MAX - LAT_MIN) / GRID_RES))  # 340
GRID_LON = LON_MIN + GRID_RES * (np.arange(NX) + 0.5)   # west -> east
GRID_LAT = LAT_MIN + GRID_RES * (np.arange(NY) + 0.5)   # south -> north (j grows north)
ANALYSIS_FACTOR = 5                                     # 0.05 deg analysis cell = 5 x 5 radar cells

NO_ECHO_DBZ = np.float32(-32.0)


def to_analysis_grid(field: np.ndarray, how: str = "mean", min_valid: float = 0.5) -> np.ndarray:
    """Reduce a [340, 320] radar-grid field to the [68, 64] analysis grid of
    ``core/grid.py``.  ``how`` = "mean" (use it on rain rate / accumulation,
    never on dBZ) or "max".  A coarse cell is NaN when less than ``min_valid``
    of its 25 fine cells are valid."""
    f = ANALYSIS_FACTOR
    blocks = field.reshape(NY // f, f, NX // f, f).transpose(0, 2, 1, 3).reshape(NY // f, NX // f, f * f)
    valid = np.isfinite(blocks)
    frac = valid.mean(-1)
    if how == "max":
        out = np.where(valid, blocks, -np.inf).max(-1)
    else:
        out = np.where(valid, blocks, 0.0).sum(-1) / np.maximum(valid.sum(-1), 1)
    return np.where(frac >= min_valid, out, np.nan).astype(np.float32)

USER_AGENT = "riua/0.1 (flood-risk research; python-requests)"
TIMEOUT = 30

_session: requests.Session | None = None
_frame_cache: dict[tuple[str, datetime], np.ndarray] = {}
_FRAME_CACHE_MAX = 40


def _http() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


def _get(url: str, *, headers: dict | None = None, tries: int = 3) -> requests.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = _http().get(url, headers=headers, timeout=TIMEOUT)
            if r.status_code in (200, 206):
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
            if r.status_code in (400, 403, 404):
                break
        except requests.RequestException as exc:  # network hiccup -> retry
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def _cache_put(source: str, t: datetime, arr: np.ndarray) -> None:
    _frame_cache[(source, t)] = arr
    if len(_frame_cache) > _FRAME_CACHE_MAX:
        for key in sorted(_frame_cache, key=lambda k: k[1])[: len(_frame_cache) - _FRAME_CACHE_MAX]:
            _frame_cache.pop(key, None)


# ----------------------------------------------------------------- Z-R relations
def dbz_to_rainrate(dbz: np.ndarray, a: float = 200.0, b: float = 1.6,
                    cap_dbz: float = 55.0, min_dbz: float = 5.0) -> np.ndarray:
    """Rain rate in mm/h from reflectivity, Z = a * R**b.

    Default is Marshall-Palmer (a=200, b=1.6), the same relation OPERA uses for
    its RATE product.  For deep Mediterranean convection the WSR-88D convective
    relation (a=300, b=1.4) is the usual alternative; it gives more rain above
    ~35 dBZ (at 50 dBZ: 63 mm/h vs 49 mm/h).

    ``cap_dbz`` is the hail cap: above ~53-55 dBZ the return is dominated by
    hail and Z-R explodes, so dBZ is clipped (55 dBZ -> 100 mm/h with M-P,
    144 mm/h with the convective relation).  Below ``min_dbz`` rain is set to 0
    (drizzle / clutter / noise).  NaN (no coverage) stays NaN.
    """
    d = np.asarray(dbz, dtype=np.float32)
    z = np.power(10.0, np.minimum(d, cap_dbz) / 10.0)
    r = np.power(z / a, 1.0 / b).astype(np.float32)
    r = np.where(d < min_dbz, np.float32(0.0), r)
    return np.where(np.isnan(d), np.float32(np.nan), r).astype(np.float32)


def dbz_to_rainrate_convective(dbz: np.ndarray, cap_dbz: float = 55.0) -> np.ndarray:
    """Z = 300 R^1.4 (WSR-88D convective)."""
    return dbz_to_rainrate(dbz, a=300.0, b=1.4, cap_dbz=cap_dbz)


# ===================================================================== RainViewer
RAINVIEWER_INDEX = "https://api.rainviewer.com/public/weather-maps.json"
RAINVIEWER_ZOOM = 7      # max zoom on the free tier since 2026-01-01 (z>7 returns a placeholder)
RAINVIEWER_SIZE = 512    # 512 px tiles at z7 -> 0.0055 deg/px in longitude
RAINVIEWER_COLOR = 2     # Universal Blue: the only scheme left (the parameter is ignored)
RAINVIEWER_OPTS = "0_0"  # smooth=0 (no blur, exact palette colours), snow=0 (rain palette everywhere)
_RV_PAD_PX = 8           # tolerate sampling this many px outside the fetched mosaic (clamped)


def _universal_blue_lut() -> dict[int, float]:
    """Packed RGBA -> dBZ for RainViewer "Universal Blue" (rain palette).

    Source: https://www.rainviewer.com/files/rainviewer_api_colors_table.csv
    (linked from https://www.rainviewer.com/api/color-schemes.html), column
    "Universal Blue", rows -32..95 dBZ.  The table is piecewise-linear between
    the anchors below (checked against all 128 CSV rows); every dBZ from -10 to
    65 has a unique colour, so the decode is exact to 1 dBZ.
    """
    anchors = [  # (dBZ, R, G, B, A)
        (-10, 0x63, 0x61, 0x59, 0x14), (5, 0x92, 0x88, 0x71, 0x64), (10, 0xCE, 0xC0, 0x87, 0x96),
        (14, 0xDE, 0xD0, 0x97, 0xBE),
        (15, 0x88, 0xDD, 0xEE, 0xFF), (20, 0x00, 0xA3, 0xE0, 0xFF), (25, 0x00, 0x77, 0xAA, 0xFF),
        (30, 0x00, 0x55, 0x88, 0xFF), (34, 0x00, 0x47, 0x68, 0xFF),
        (35, 0xFF, 0xEE, 0x00, 0xFF), (40, 0xFF, 0xAA, 0x00, 0xFF), (44, 0xFF, 0x81, 0x00, 0xFF),
        (45, 0xFF, 0x44, 0x00, 0xFF), (50, 0xC1, 0x00, 0x00, 0xFF), (54, 0x5D, 0x00, 0x00, 0xFF),
        (55, 0xFF, 0xAA, 0xFF, 0xFF), (60, 0xFF, 0x77, 0xFF, 0xFF), (64, 0xFF, 0x4E, 0xFF, 0xFF),
    ]
    lut: dict[int, float] = {}
    for (d0, *c0), (d1, *c1) in zip(anchors[:-1], anchors[1:]):
        for d in range(d0, d1 + 1):
            f = (d - d0) / (d1 - d0)
            rgba = [int(a + (b - a) * f) for a, b in zip(c0, c1)]
            lut.setdefault(_pack(*rgba), float(d))
    lut[_pack(0xFF, 0xFF, 0xFF, 0xFF)] = 65.0   # 65..74 dBZ are all white
    lut[_pack(0x00, 0xFF, 0x00, 0xFF)] = 75.0   # >= 75 dBZ green
    return lut


def _pack(r: int, g: int, b: int, a: int) -> int:
    return (r << 24) | (g << 16) | (b << 8) | a


_RV_LUT: tuple[np.ndarray, np.ndarray] | None = None


def _rv_decode(rgba: np.ndarray) -> np.ndarray:
    """RGBA uint8 (H, W, 4) -> dBZ float32.  Transparent -> NO_ECHO_DBZ.

    Colours not in the palette (should not happen with smooth=0) are matched to
    the nearest palette colour.
    """
    global _RV_LUT
    if _RV_LUT is None:
        lut = _universal_blue_lut()
        keys = np.array(sorted(lut), dtype=np.uint32)
        _RV_LUT = (keys, np.array([lut[int(k)] for k in keys], dtype=np.float32))
    keys, vals = _RV_LUT
    px = rgba.astype(np.uint32)
    packed = (px[..., 0] << 24) | (px[..., 1] << 16) | (px[..., 2] << 8) | px[..., 3]
    out = np.full(packed.shape, NO_ECHO_DBZ, dtype=np.float32)
    echo = rgba[..., 3] > 0
    if not echo.any():
        return out
    p = packed[echo]
    idx = np.clip(np.searchsorted(keys, p), 0, len(keys) - 1)
    hit = keys[idx] == p
    d = vals[idx]
    if not hit.all():  # nearest colour in RGBA space for the stragglers
        pal = np.stack([(keys >> s) & 0xFF for s in (24, 16, 8, 0)], axis=1).astype(np.int32)
        miss = rgba[echo][~hit].astype(np.int32)
        dist = ((miss[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
        d[~hit] = vals[dist.argmin(1)]
    out[echo] = d
    return out


def _mercator_pixel(lon: np.ndarray, lat: np.ndarray, zoom: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Global Web-Mercator (EPSG:3857) pixel coordinates of lon/lat at a zoom."""
    n = (2 ** zoom) * size
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n
    return x, y


class _RainViewerGeometry:
    """Which tiles to fetch and where each grid cell samples them."""

    def __init__(self) -> None:
        size, z = RAINVIEWER_SIZE, RAINVIEWER_ZOOM
        lon2d, lat2d = np.meshgrid(GRID_LON, GRID_LAT)
        px, py = _mercator_pixel(lon2d, lat2d, z, size)
        px, py = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        # drop tile rows/cols that only a sliver (< _RV_PAD_PX) of the grid touches
        x0 = int((px.min() + _RV_PAD_PX) // size)
        x1 = int((px.max() - _RV_PAD_PX) // size)
        y0 = int((py.min() + _RV_PAD_PX) // size)
        y1 = int((py.max() - _RV_PAD_PX) // size)
        self.tiles = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        self.x0, self.y0 = x0, y0
        self.shape = ((y1 - y0 + 1) * size, (x1 - x0 + 1) * size)
        self.col = np.clip(px - x0 * size, 0, self.shape[1] - 1)
        self.row = np.clip(py - y0 * size, 0, self.shape[0] - 1)

    def mosaic(self, fetch_tile: Callable[[int, int], np.ndarray]) -> np.ndarray:
        size = RAINVIEWER_SIZE
        mos = np.zeros(self.shape + (4,), dtype=np.uint8)
        for x, y in self.tiles:
            r, c = (y - self.y0) * size, (x - self.x0) * size
            mos[r:r + size, c:c + size] = fetch_tile(x, y)
        return mos


_rv_geom: _RainViewerGeometry | None = None
_rv_nocover: np.ndarray | None = None


def _rv_tile(url: str) -> np.ndarray:
    im = Image.open(io.BytesIO(_get(url).content))
    if im.size != (RAINVIEWER_SIZE, RAINVIEWER_SIZE):
        raise RuntimeError(f"unexpected tile size {im.size} for {url}")
    return np.asarray(im.convert("RGBA"))


def _rainviewer_nocover(host: str) -> np.ndarray:
    """True where RainViewer has NO radar coverage (static mask, fetched once).

    Coverage tiles: opaque black = not covered, transparent = covered.
    """
    global _rv_geom, _rv_nocover
    if _rv_geom is None:
        _rv_geom = _RainViewerGeometry()
    if _rv_nocover is None:
        g = _rv_geom
        mos = g.mosaic(lambda x, y: _rv_tile(
            f"{host}/v2/coverage/0/{RAINVIEWER_SIZE}/{RAINVIEWER_ZOOM}/{x}/{y}/0/0_0.png"))
        _rv_nocover = mos[g.row, g.col, 3] > 127
    return _rv_nocover


def fetch_rainviewer_frames(n: int = 13) -> list[tuple[datetime, np.ndarray]]:
    """Latest ``n`` RainViewer frames (10-min step; the API keeps 13 = 2 h).

    4 tile requests per new frame (z7, 512 px: x 63-64, y 48-49), cached in
    process, so a steady-state poll costs 1 + 4 requests.  Limit is
    100 requests/IP/min.  The frame ``time`` is RainViewer's mosaic time.
    """
    idx = _get(RAINVIEWER_INDEX).json()
    host = idx["host"]
    frames = sorted(idx["radar"]["past"], key=lambda f: f["time"])[-n:]
    nocover = _rainviewer_nocover(host)
    g = _rv_geom
    assert g is not None
    out: list[tuple[datetime, np.ndarray]] = []
    for fr in frames:
        t = datetime.fromtimestamp(fr["time"], tz=timezone.utc)
        arr = _frame_cache.get(("rainviewer", t))
        if arr is None:
            base = f"{host}{fr['path']}/{RAINVIEWER_SIZE}/{RAINVIEWER_ZOOM}"
            mos = g.mosaic(lambda x, y: _rv_tile(
                f"{base}/{x}/{y}/{RAINVIEWER_COLOR}/{RAINVIEWER_OPTS}.png"))
            arr = _rv_decode(mos[g.row, g.col])
            arr[nocover] = np.nan
            _cache_put("rainviewer", t, arr)
        out.append((t, arr))
    return out


# ========================================================================== OPERA
OPERA_S3 = "https://s3.waw3-1.cloudferro.com"
OPERA_BUCKET_LIVE = "openradar-24h"         # rolling 24 h
OPERA_BUCKET_ARCHIVE = "openradar-archive"  # 2012 -> today
# +proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84 (from the files)
_LAEA_LAT0, _LAEA_LON0, _LAEA_X0, _LAEA_Y0 = 55.0, 10.0, 1950000.0, -2100000.0
_WGS84_A, _WGS84_F = 6378137.0, 1.0 / 298.257223563
_OPERA_NODATA = -9999000.0
_OPERA_HEADER_BYTES = 32768


def laea_forward(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ellipsoidal Lambert azimuthal equal-area (Snyder 1987, eqs 24-17..24-26)
    for the OPERA composite grid.  Agrees with pyproj to < 1 mm on our box."""
    e2 = _WGS84_F * (2.0 - _WGS84_F)
    e = math.sqrt(e2)

    def q(sin_phi):
        return (1.0 - e2) * (sin_phi / (1.0 - e2 * sin_phi ** 2)
                             - (1.0 / (2.0 * e)) * np.log((1.0 - e * sin_phi) / (1.0 + e * sin_phi)))

    qp = float(q(1.0))
    rq = _WGS84_A * math.sqrt(qp / 2.0)
    s1 = math.sin(math.radians(_LAEA_LAT0))
    b1 = math.asin(float(q(s1)) / qp)
    m1 = math.cos(math.radians(_LAEA_LAT0)) / math.sqrt(1.0 - e2 * s1 * s1)
    d = _WGS84_A * m1 / (rq * math.cos(b1))
    beta = np.arcsin(np.clip(q(np.sin(np.radians(lat))) / qp, -1.0, 1.0))
    dlam = np.radians(lon - _LAEA_LON0)
    bb = rq * np.sqrt(2.0 / (1.0 + math.sin(b1) * np.sin(beta) + math.cos(b1) * np.cos(beta) * np.cos(dlam)))
    x = bb * d * np.cos(beta) * np.sin(dlam)
    y = (bb / d) * (math.cos(b1) * np.sin(beta) - math.sin(b1) * np.cos(beta) * np.cos(dlam))
    return x + _LAEA_X0, y + _LAEA_Y0


_TIFF_TYPES = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i",
               10: "ii", 11: "f", 12: "d", 16: "Q"}


def _tiff_first_ifd(head: bytes) -> dict[int, tuple]:
    """Tags of the first IFD (full resolution) of a little-endian classic TIFF."""
    if head[:4] != b"II*\x00":
        raise RuntimeError("not a little-endian classic TIFF")
    off = struct.unpack_from("<I", head, 4)[0]
    count = struct.unpack_from("<H", head, off)[0]
    tags: dict[int, tuple] = {}
    for i in range(count):
        tag, typ, cnt, raw = struct.unpack_from("<HHI4s", head, off + 2 + 12 * i)
        fmt = _TIFF_TYPES.get(typ)
        if fmt is None or typ == 2:
            continue
        nbytes = struct.calcsize("<" + fmt) * cnt
        if nbytes <= 4:
            data = raw[:nbytes]
        else:
            pos = struct.unpack("<I", raw)[0]
            if pos + nbytes > len(head):
                raise RuntimeError("TIFF header larger than the fetched range")
            data = head[pos:pos + nbytes]
        tags[tag] = struct.unpack("<" + fmt * cnt, data)
    return tags


def _read_cog_box(url: str, band: int = 0) -> np.ndarray:
    """Read one band of an OPERA COG on the Riuà grid, fetching only the header
    and the tiles that intersect the box (HTTP Range).  Raw file values."""
    head = _get(url, headers={"Range": f"bytes=0-{_OPERA_HEADER_BYTES - 1}"}).content
    tags = _tiff_first_ifd(head)
    width, height = tags[256][0], tags[257][0]
    tw, th = tags[322][0], tags[323][0]
    spp = tags[277][0]
    if tags[259][0] != 8 or tags.get(317, (1,))[0] != 1 or set(tags[258]) != {32} or set(tags[339]) != {3}:
        raise RuntimeError("unexpected TIFF layout (want deflate, no predictor, float32)")
    sx, sy = tags[33550][0], tags[33550][1]   # pixel size in m
    tie = tags[33922]                         # pixel (0,0) corner -> (x, y)
    lon2d, lat2d = np.meshgrid(GRID_LON, GRID_LAT)
    x, y = laea_forward(lon2d, lat2d)
    col = np.floor((x - tie[3]) / sx - tie[0]).astype(np.int64)
    row = np.floor((tie[4] - y) / sy - tie[1]).astype(np.int64)
    if col.min() < 0 or row.min() < 0 or col.max() >= width or row.max() >= height:
        raise RuntimeError("grid falls outside the OPERA composite")
    tc0, tc1 = int(col.min() // tw), int(col.max() // tw)
    tr0, tr1 = int(row.min() // th), int(row.max() // th)
    across = -(-width // tw)
    offsets, counts = tags[324], tags[325]
    sub = np.empty(((tr1 - tr0 + 1) * th, (tc1 - tc0 + 1) * tw), dtype=np.float32)
    for tr in range(tr0, tr1 + 1):
        # tiles of one row are stored back to back (ROW_MAJOR) -> one Range request per tile row
        first, last = tr * across + tc0, tr * across + tc1
        start, end = offsets[first], offsets[last] + counts[last]
        blob = _get(url, headers={"Range": f"bytes={start}-{end - 1}"}).content
        for tc in range(tc0, tc1 + 1):
            k = tr * across + tc
            if counts[k] == 0:  # sparse tile
                tile = np.full((th, tw), _OPERA_NODATA, dtype=np.float32)
            else:
                raw = zlib.decompress(blob[offsets[k] - start: offsets[k] - start + counts[k]])
                tile = np.frombuffer(raw, dtype="<f4").reshape(th, tw, spp)[..., band]
            sub[(tr - tr0) * th:(tr - tr0 + 1) * th, (tc - tc0) * tw:(tc - tc0 + 1) * tw] = tile
    return sub[row - tr0 * th, col - tc0 * tw]


def _opera_key(t: datetime, quantity: str = "DBZH") -> str:
    return f"{t:%Y/%m/%d}/OPERA/COMP/OPERA@{t:%Y%m%dT%H%M}@0@{quantity}.tiff"


def _opera_url(bucket: str, t: datetime, quantity: str = "DBZH") -> str:
    return f"{OPERA_S3}/{bucket}/{_opera_key(t, quantity).replace('@', '%40')}"


def opera_list(day: datetime, quantity: str = "DBZH", bucket: str = OPERA_BUCKET_LIVE) -> list[datetime]:
    """Times (UTC) of the composites available for one day in a bucket."""
    url = f"{OPERA_S3}/{bucket}/?list-type=2&max-keys=1000&prefix={day:%Y/%m/%d}/OPERA/COMP/"
    times: list[datetime] = []
    token = None
    while True:
        xml = _get(url + (f"&continuation-token={requests.utils.quote(token, safe='')}" if token else "")).text
        for m in re.finditer(r"<Key>[^<]*OPERA@(\d{8}T\d{4})@0@" + quantity + r"\.tiff</Key>", xml):
            times.append(datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc))
        nxt = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not nxt:
            break
        token = nxt.group(1)
    return sorted(times)


def fetch_opera(t: datetime, quantity: str = "DBZH", bucket: str = OPERA_BUCKET_LIVE) -> np.ndarray:
    """One OPERA composite on the Riuà grid.

    quantity: "DBZH" (dBZ, 1 km, every 5 min), "RATE" (mm/h, 2 km, every 15 min)
    or "ACRR" (mm in the hour ENDING at ``t``, 2 km, every 15 min).
    NaN = no coverage.  No-echo is NO_ECHO_DBZ for DBZH and 0 for RATE/ACRR.
    Use ``bucket=OPERA_BUCKET_ARCHIVE`` for anything older than 24 h.
    """
    raw = _read_cog_box(_opera_url(bucket, t, quantity))
    out = raw.astype(np.float32, copy=True)
    undetect = np.isnan(raw)                      # tiff: undetect = NaN
    nodata = raw <= -1.0e6                        # tiff: nodata = -9999000
    out[undetect] = NO_ECHO_DBZ if quantity == "DBZH" else 0.0
    out[nodata] = np.nan
    return out


def fetch_opera_frames(n: int = 13, step_min: int = 10) -> list[tuple[datetime, np.ndarray]]:
    """Latest ``n`` OPERA DBZH frames whose minute is a multiple of ``step_min``
    (5, 10, 15...).  3 requests per new frame (header + 2 tile rows, ~100-300 KB)."""
    now = datetime.now(timezone.utc)
    times = opera_list(now)
    if len([t for t in times if t.minute % step_min == 0]) < n:
        times = opera_list(now - timedelta(days=1)) + times
    times = [t for t in times if t.minute % step_min == 0][-n:]
    out = []
    for t in times:
        arr = _frame_cache.get(("opera", t))
        if arr is None:
            arr = fetch_opera(t)
            _cache_put("opera", t, arr)
        out.append((t, arr))
    return out


def fetch_opera_archive(start: datetime, end: datetime, step_min: int = 10,
                        quantity: str = "DBZH") -> list[tuple[datetime, np.ndarray]]:
    """Hindcast helper: archived composites in [start, end] (UTC), e.g. 2024-10-29."""
    out = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        for t in opera_list(day, quantity, OPERA_BUCKET_ARCHIVE):
            if start <= t <= end and t.minute % step_min == 0:
                out.append((t, fetch_opera(t, quantity, OPERA_BUCKET_ARCHIVE)))
        day += timedelta(days=1)
    return out


# ========================================================================== AEMET
AEMET_COMPO_URL = "https://www.aemet.es/es/api-eltiempo/radar/download/compo"
# Legend from https://www.aemet.es/es/api-eltiempo/radar/leyenda-radar/compo:
# RGB -> (low, high) dBZ.  We return the class midpoint.
_AEMET_CLASSES = {
    (0, 0, 252): (12, 18), (0, 148, 252): (18, 24), (0, 252, 252): (24, 30),
    (67, 131, 35): (30, 36), (0, 192, 0): (36, 42), (0, 255, 0): (42, 48),
    (255, 255, 0): (48, 54), (255, 187, 0): (54, 60), (255, 127, 0): (60, 66),
    (255, 0, 0): (66, 72), (200, 0, 90): (72, 78),
}
_AEMET_COVERED_NO_ECHO = (255, 255, 255)   # palette index 0 (white): inside radar range, < 12 dBZ
# anything else (index 1, RGB 239,242,249) = outside the radar ranges -> NaN


def _aemet_decode(tif_bytes: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(tif_bytes))
    scale, tie = im.tag_v2[33550], im.tag_v2[33922]
    rgb = np.asarray(im.convert("RGB"))
    col = np.floor((GRID_LON - tie[3]) / scale[0]).astype(np.int64)
    row = np.floor((tie[4] - GRID_LAT) / scale[1]).astype(np.int64)
    inside = np.outer((row >= 0) & (row < rgb.shape[0]), (col >= 0) & (col < rgb.shape[1]))
    px = rgb[np.clip(row, 0, rgb.shape[0] - 1)][:, np.clip(col, 0, rgb.shape[1] - 1)]
    out = np.full(px.shape[:2], np.nan, dtype=np.float32)
    out[(px == _AEMET_COVERED_NO_ECHO).all(-1)] = NO_ECHO_DBZ
    for colour, (lo, hi) in _AEMET_CLASSES.items():
        out[(px == colour).all(-1)] = (lo + hi) / 2.0
    out[~inside] = np.nan
    return out


def fetch_aemet_frames(n: int = 13) -> list[tuple[datetime, np.ndarray]]:
    """Latest ``n`` frames of the AEMET peninsular composite (10-min step, the
    tarball holds the last 4 h = 24 frames; ONE request of ~210 KB for all).

    Values are class midpoints (15, 21, 27 ... dBZ); anything under 12 dBZ is
    reported as no echo.  Native pixel is 0.0263 deg (~2.9 km), so each native
    pixel covers ~2.6 x 2.6 cells of our grid.
    """
    blob = _get(AEMET_COMPO_URL).content
    members: list[tuple[datetime, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            hit = re.search(r"radw(\d{12})_4326\.tif$", m.name)
            if hit and m.isfile():
                t = datetime.strptime(hit.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                members.append((t, tar.extractfile(m).read()))
    # AEMET sometimes republishes the previous image under the next time label
    # (seen 2026-10-01: 07:10 byte-identical to 07:00).  Drop such repeats so a
    # nowcast never sees a storm that "stopped moving".
    members.sort(key=lambda m: m[0])
    members = [m for k, m in enumerate(members) if k == 0 or m[1] != members[k - 1][1]]
    out = []
    for t, data in members[-n:]:
        arr = _frame_cache.get(("aemet", t))
        if arr is None:
            arr = _aemet_decode(data)
            _cache_put("aemet", t, arr)
        out.append((t, arr))
    return out


# ===================================================================== public API
SOURCES: dict[str, Callable[..., list[tuple[datetime, np.ndarray]]]] = {
    "rainviewer": fetch_rainviewer_frames,
    "opera": fetch_opera_frames,
    "aemet": fetch_aemet_frames,
}
DEFAULT_ORDER = ("blend", "rainviewer", "aemet")
ATTRIBUTION = {
    "rainviewer": "Radar: RainViewer (https://www.rainviewer.com/)",
    "opera": "Radar: EUMETNET OPERA, CC BY 4.0",
    "aemet": "Radar: © AEMET",
    "blend": "Radar: EUMETNET OPERA (CC BY 4.0) + © AEMET",
}


def fetch_blended_frames(n: int = 13) -> list[tuple[datetime, np.ndarray]]:
    """OPERA frames with the cells OPERA does not cover filled from the AEMET
    composite of the same time label.

    Why: in Oct 2026 the OPERA composite has no Cullera/Murcia radar, so ~19 %
    of the box (the coast from Valencia city to Denia and the sea) is NaN.
    The AEMET composite covers it, but only as 6-dBZ classes >= 12 dBZ and it
    is published ~15-25 min later, so the newest one or two frames usually keep
    their NaN hole (the caller must treat NaN as "unknown", never as "dry").
    When AEMET sends Cullera to OPERA again this function becomes a no-op.
    """
    frames = fetch_opera_frames(n)
    if not any(np.isnan(a).any() for _, a in frames):
        return frames
    try:
        aemet = dict(fetch_aemet_frames(24))
    except Exception:  # noqa: BLE001 - the fill is best effort
        return frames
    out = []
    for t, arr in frames:
        fill = aemet.get(t)
        if fill is not None:
            hole = np.isnan(arr)
            if hole.any():
                arr = arr.copy()
                arr[hole] = fill[hole]
        out.append((t, arr))
    return out


SOURCES["blend"] = fetch_blended_frames


def fetch_radar_frames(n: int = 13, source: str = "auto") -> list[tuple[datetime, np.ndarray]]:
    """Latest ``n`` radar frames as ``[(datetime_utc, dbz[NY, NX] float32), ...]``,
    oldest first, 10 minutes apart (n=13 -> the last 2 hours).

    ``source``: "blend", "opera", "rainviewer", "aemet", or "auto" (try
    DEFAULT_ORDER and return the first that yields at least 2 frames).
    ``last_source`` tells which one answered, for the attribution line.
    """
    global last_source
    if source != "auto":
        frames = SOURCES[source](n)
        last_source = source
        return frames
    errors = []
    for name in DEFAULT_ORDER:
        try:
            frames = SOURCES[name](n)
            if len(frames) >= 2:
                last_source = name
                return frames
            errors.append(f"{name}: only {len(frames)} frame(s)")
        except Exception as exc:  # noqa: BLE001 - any failure -> next source
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no radar source available: " + " | ".join(errors))


last_source: str | None = None


# ====================================================================== smoke test
_CITIES = [
    ("Valencia", -0.376, 39.470), ("Alicante", -0.481, 38.345), ("Castello", -0.037, 39.986),
    ("Cullera radar", -0.251, 39.176), ("Teruel", -1.106, 40.344), ("Albacete", -1.858, 38.995),
    ("Murcia", -1.130, 37.992), ("Requena", -1.100, 39.488), ("Gandia", -0.180, 38.968),
    ("Tortosa", 0.521, 40.812),
]


def _render(dbz: np.ndarray, path: str, title: str) -> None:
    from PIL import ImageDraw

    stops = [(-32, (235, 235, 235)), (5, (200, 215, 235)), (15, (120, 190, 240)), (25, (30, 110, 200)),
             (35, (250, 220, 40)), (45, (240, 90, 20)), (55, (150, 0, 0)), (65, (255, 120, 255))]
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    dbz = dbz[::-1]                      # image row 0 = north
    d = np.nan_to_num(dbz, nan=-32.0)
    img = np.stack([np.interp(d, xs, [s[1][k] for s in stops]) for k in range(3)], axis=-1).astype(np.uint8)
    img[np.isnan(dbz)] = (70, 70, 70)
    scale = 2
    im = Image.fromarray(img).resize((NX * scale, NY * scale), Image.NEAREST)
    draw = ImageDraw.Draw(im)
    for lon in np.arange(-2.0, 0.81, 1.0):   # 1-degree graticule
        c = (lon - LON_MIN) / GRID_RES * scale
        draw.line([(c, 0), (c, NY * scale)], fill=(150, 150, 150))
    for lat in np.arange(38.0, 41.01, 1.0):
        r = (LAT_MAX - lat) / GRID_RES * scale
        draw.line([(0, r), (NX * scale, r)], fill=(150, 150, 150))
    for name, lon, lat in _CITIES:
        c, r = (lon - LON_MIN) / GRID_RES * scale, (LAT_MAX - lat) / GRID_RES * scale
        draw.ellipse([c - 3, r - 3, c + 3, r + 3], outline=(0, 0, 0), fill=(255, 0, 255))
        draw.text((c + 5, r - 5), name, fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(0, 0, 0))
    im.save(path)


def _smoke(sources: list[str], n: int) -> None:
    os = _os
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch", "r3-radar"))
    os.makedirs(outdir, exist_ok=True)
    print(f"grid {NY} x {NX} (j north, i east), cell centres lon {GRID_LON[0]:.3f}..{GRID_LON[-1]:.3f}, "
          f"lat {GRID_LAT[0]:.3f}..{GRID_LAT[-1]:.3f}")
    for name in sources:
        t0 = time.time()
        try:
            frames = fetch_radar_frames(n, source=name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}")
            continue
        now = datetime.now(timezone.utc)
        print(f"[{name}] {len(frames)} frames in {time.time() - t0:.1f} s; "
              f"newest is {(now - frames[-1][0]).total_seconds() / 60:.0f} min old")
        for t, dbz in frames:
            cov = 100.0 * np.isfinite(dbz).mean()
            mx = np.nanmax(dbz) if np.isfinite(dbz).any() else float("nan")
            rr = dbz_to_rainrate(dbz)
            print(f"  {t:%Y-%m-%d %H:%M}Z  coverage {cov:5.1f} %  max {mx:5.1f} dBZ  "
                  f"cells>30dBZ {int(np.nansum(dbz > 30)):5d}  cells>=12dBZ {int(np.nansum(dbz >= 12)):6d}  "
                  f"max rain {np.nanmax(rr) if np.isfinite(rr).any() else float('nan'):5.1f} mm/h")
        t, dbz = frames[-1]
        path = os.path.join(outdir, f"smoke_{name}.png")
        _render(dbz, path, f"{name} {t:%Y-%m-%d %H:%M}Z")
        print(f"  saved {path}")


if __name__ == "__main__":
    _args = _sys.argv[1:]
    _sources = [a for a in _args if a in SOURCES] or list(SOURCES)
    _n = next((int(a) for a in _args if a.isdigit()), 3)
    _smoke(_sources, _n)
