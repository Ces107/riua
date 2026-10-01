"""Satellite + lightning for Riuà from EUMETView (EUMETSAT GeoServer), keyless.

Endpoint ``https://view.eumetsat.int/geoserver`` (WMS 1.3.0 + WCS 2.0.1), no
key, no login.  Verified from this machine on 2026-10-01 (see
``coord/findings/r3-radar.md``).  Licence: EUMETSAT data policy, attribution
"© EUMETSAT <year>"; H SAF layers "© EUMETSAT H SAF".

What is numeric and what is only a picture:

=====================  =========  ========  ======================================
layer                  step       archive   what we can get
=====================  =========  ========  ======================================
mtg_fd:li_afa          5 min      2025-05   lightning (MTG Lightning Imager,
                                            accumulated flash area).  WMS/WCS give
                                            a colour rendering -> decoded here to
                                            approx. flashes / 5 min (legend 1..20+)
mtg_fd:h40b            10 min     2026-07   H SAF rain rate, NUMERIC float mm/h
                                            through WCS (0.0156 deg)
msg_fes:h60b           15 min     2022-12   H SAF rain rate, picture only ->
                                            decoded to 12 classes (mm/h)
mtg_fd:ir105_hrfi      10 min     2024-09   FCI IR10.5, 8-bit index through WCS
                                            (low = cold cloud top; NOT kelvin)
msg_fes:ir108          15 min     2020-09   SEVIRI IR10.8, picture only
msg_fes:rgb_convection 15 min     2020-09   Convection RGB, picture only
msg_fes:rdt            15 min     2023-03   NWC SAF rapidly developing
                                            thunderstorms, picture only
msg_fes:cth            15 min     2020-09   cloud top height, picture only
=====================  =========  ========  ======================================

Everything is returned on the radar grid of ``radar.py`` (340 x 320 cells of
0.01 deg, j growing north).  Only numpy + Pillow + requests.

Smoke test (from ``backend/``):  ``py -3.11 -m riua.sources.eumetview``
"""
from __future__ import annotations

import os as _os
import sys as _sys

# Run as a plain script: keep the sibling ``warnings.py`` from shadowing the
# standard library and make ``riua`` importable.
if __name__ == "__main__" and not __package__:
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _sys.path and _os.path.abspath(_sys.path[0]) == _here:
        _sys.path.pop(0)
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_here, "..", "..")))

import io
import re
import struct
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from PIL import Image

from riua.sources.radar import GRID_LAT, GRID_LON, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, NX, NY, USER_AGENT

BASE = "https://view.eumetsat.int/geoserver"
TIMEOUT = 40
ATTRIBUTION = "© EUMETSAT (EUMETView); precipitation: © EUMETSAT H SAF"

LAYER_LIGHTNING = "mtg_fd:li_afa"
LAYER_H40B = "mtg_fd:h40b"
LAYER_H60B = "msg_fes:h60b"
LAYER_IR105 = "mtg_fd:ir105_hrfi"
LAYER_IR108 = "msg_fes:ir108"
LAYER_CONVECTION = "msg_fes:rgb_convection"
LAYER_RDT = "msg_fes:rdt"
LAYER_CTH = "msg_fes:cth"

# For a web map (Leaflet: L.tileLayer.wms(WMS_ENDPOINT, {layers, format:'image/png',
# transparent:true, time:'2026-10-01T07:00:00Z', version:'1.3.0'})).
WMS_ENDPOINT = BASE + "/ows"
WMS_GETMAP_TEMPLATE = (
    BASE + "/ows?service=WMS&version=1.3.0&request=GetMap&layers={layer}&styles=&crs=EPSG:3857"
    "&bbox={bbox}&width=256&height=256&format=image/png&transparent=true&time={time}"
)

_session: requests.Session | None = None


def _get(url: str, params: dict | None = None) -> requests.Response:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = _session.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"EUMETView request failed: {url}: {last}")


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------- time axis
def layer_times(layer: str) -> tuple[datetime, datetime, timedelta]:
    """(first, latest, step) of a layer, from its own small GetCapabilities
    (~7 KB at ``/geoserver/<workspace>/<name>/ows``)."""
    ws, name = layer.split(":")
    xml = _get(f"{BASE}/{ws}/{name}/ows",
               {"service": "WMS", "version": "1.3.0", "request": "GetCapabilities"}).text
    m = re.search(r'<Dimension name="time"[^>]*>\s*([^/<]+)/([^/<]+)/PT(\d+)M', xml)
    if not m:
        raise RuntimeError(f"no time dimension for {layer}")
    parse = lambda s: datetime.strptime(s.strip()[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return parse(m.group(1)), parse(m.group(2)), timedelta(minutes=int(m.group(3)))


# ----------------------------------------------------------------- WMS -> grid
def wms_grid_rgba(layer: str, t: datetime | None = None, style: str = "") -> np.ndarray:
    """GetMap rendered exactly on the Riuà radar grid: RGBA uint8 [NY, NX, 4],
    one pixel per cell, flipped so that j grows north like every other field.
    WMS 1.3.0 with EPSG:4326 wants bbox as lat_min,lon_min,lat_max,lon_max."""
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap", "layers": layer, "styles": style,
        "crs": "EPSG:4326", "bbox": f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}",
        "width": NX, "height": NY, "format": "image/png", "transparent": "true",
    }
    if t is not None:
        params["time"] = _iso(t)
    r = _get(BASE + "/ows", params)
    if not r.headers.get("content-type", "").startswith("image/"):
        raise RuntimeError(f"WMS error for {layer}: {r.text[:300]}")
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))[::-1]   # row 0 = south


def _nearest_colour(rgba: np.ndarray, table: list[tuple[tuple[int, int, int], float]],
                    max_dist: float = 40.0) -> np.ndarray:
    """Map opaque pixels to the value of the nearest table colour; transparent
    pixels -> 0; opaque pixels far from every table colour -> NaN."""
    pal = np.array([c for c, _ in table], dtype=np.int32)
    val = np.array([v for _, v in table], dtype=np.float32)
    out = np.zeros(rgba.shape[:2], dtype=np.float32)
    opaque = rgba[..., 3] > 127
    if opaque.any():
        px = rgba[opaque][:, :3].astype(np.int32)
        d2 = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
        k = d2.argmin(1)
        v = val[k]
        v[np.sqrt(d2[np.arange(len(k)), k]) > max_dist] = np.nan
        out[opaque] = v
    return out


# ------------------------------------------------------------------- lightning
# Legend of mtg_fd:li_afa ("Accumulated Flash Area, count / 5 minutes", 1 .. 10 .. 20+),
# sampled from GetLegendGraphic (640x80 px: x=149 -> 1, x=405 -> 10, x=635 -> 20+).
# The ramp is continuous, so the decoded count is approximate (about +-1).
_LI_LEGEND = [
    ((255, 255, 201), 1.0), ((255, 249, 189), 1.0), ((255, 244, 178), 2.0), ((255, 237, 162), 2.8),
    ((255, 228, 143), 3.8), ((254, 219, 124), 4.9), ((254, 195, 94), 6.3), ((254, 171, 72), 7.7),
    ((253, 148, 63), 9.1), ((253, 131, 57), 10.0), ((252, 102, 49), 11.1), ((251, 76, 41), 12.4),
    ((237, 48, 33), 13.7), ((225, 25, 28), 15.0), ((208, 13, 32), 16.3), ((188, 0, 37), 17.6),
    ((161, 0, 38), 18.9), ((139, 0, 38), 20.0),
]


def fetch_lightning(t: datetime | None = None) -> tuple[datetime, np.ndarray]:
    """MTG Lightning Imager accumulated flash area on the Riuà grid.

    Returns (time_utc, flashes[NY, NX] float32): approximate number of flashes
    whose optical footprint touched the cell in the 5 minutes ending at ``time``
    (0 = none, capped at 20).  Native resolution is the 2 km FCI grid.  LI is an
    optical sensor: good detection at night, weaker by day; treat a non-zero
    cell as "electrical activity here", not as a strike location.
    """
    if t is None:
        t = layer_times(LAYER_LIGHTNING)[1]
    return t, _nearest_colour(wms_grid_rgba(LAYER_LIGHTNING, t), _LI_LEGEND, max_dist=60.0)


def fetch_lightning_window(minutes: int = 30) -> tuple[datetime, np.ndarray]:
    """Sum of the last ``minutes`` of 5-min lightning frames (one request each)."""
    _, latest, step = layer_times(LAYER_LIGHTNING)
    total = np.zeros((NY, NX), dtype=np.float32)
    for k in range(max(1, minutes // int(step.total_seconds() // 60))):
        total += np.nan_to_num(fetch_lightning(latest - k * step)[1])
    return latest, total


# -------------------------------------------------------------- H SAF rain rate
# msg_fes:h60b legend (mm/h), class edges read off GetLegendGraphic (linear axis,
# ticks 0 2 5 10 15 20 30 50+); value = class midpoint.  Edges are +-0.2 mm/h.
_H60B_LEGEND = [
    ((255, 255, 204), 0.1), ((227, 244, 192), 0.6), ((198, 232, 179), 1.5), ((162, 218, 183), 2.5),
    ((126, 204, 186), 4.0), ((96, 194, 191), 6.0), ((65, 183, 196), 8.5), ((47, 164, 194), 12.5),
    ((29, 145, 191), 17.5), ((32, 120, 180), 25.0), ((34, 94, 168), 40.0), ((12, 44, 133), 50.0),
]


def fetch_h60b(t: datetime | None = None) -> tuple[datetime, np.ndarray]:
    """H SAF H60B (SEVIRI IR blended with microwave) instantaneous rain rate,
    mm/h, as 12 classes (midpoints; top class "50+" returned as 50).  Native
    pixel ~4-5 km over Spain.  Archive from 2022-12-09 -> usable for the
    29 Oct 2024 hindcast as a categorical field.  0 = no rain OR no data."""
    if t is None:
        t = layer_times(LAYER_H60B)[1]
    return t, _nearest_colour(wms_grid_rgba(LAYER_H60B, t), _H60B_LEGEND, max_dist=25.0)


def _read_plain_tiff(buf: bytes) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Minimal reader for the uncompressed single-tile/strip GeoTIFF that
    GeoServer WCS returns.  -> (array[H, W, bands], (lon0, dlon, lat0, dlat))
    where lon0/lat0 is the outer corner of pixel (0, 0)."""
    bo = ">" if buf[:2] == b"MM" else "<"
    off = struct.unpack(bo + "I", buf[4:8])[0]
    n = struct.unpack(bo + "H", buf[off:off + 2])[0]
    kinds = {1: "B", 3: "H", 4: "I", 11: "f", 12: "d"}
    tags: dict[int, tuple] = {}
    for i in range(n):
        tag, typ, cnt = struct.unpack(bo + "HHI", buf[off + 2 + 12 * i: off + 10 + 12 * i])
        raw = buf[off + 10 + 12 * i: off + 14 + 12 * i]
        if typ not in kinds:
            continue
        size = struct.calcsize(bo + kinds[typ]) * cnt
        data = raw[:size] if size <= 4 else buf[struct.unpack(bo + "I", raw)[0]:][:size]
        tags[tag] = struct.unpack(bo + kinds[typ] * cnt, data)
    w, h, spp = tags[256][0], tags[257][0], tags.get(277, (1,))[0]
    if tags.get(259, (1,))[0] != 1:
        raise RuntimeError("compressed WCS TIFF not supported")
    bits, fmt = tags[258][0], tags.get(339, (1,))[0]
    dtype = np.dtype({(8, 1): "u1", (16, 1): "u2", (32, 3): "f4", (64, 3): "f8"}[(bits, fmt)]).newbyteorder(bo)
    if 324 in tags:   # tiles
        tw, th = tags[322][0], tags[323][0]
        across, down = -(-w // tw), -(-h // th)
        full = np.empty((down * th, across * tw, spp), dtype=dtype)
        for k, (o, c) in enumerate(zip(tags[324], tags[325])):
            r, cidx = divmod(k, across)
            full[r * th:(r + 1) * th, cidx * tw:(cidx + 1) * tw] = np.frombuffer(buf, dtype, th * tw * spp, o).reshape(th, tw, spp)
        arr = full[:h, :w]
    else:             # strips
        data = b"".join(buf[o:o + c] for o, c in zip(tags[273], tags[279]))
        arr = np.frombuffer(data, dtype, h * w * spp).reshape(h, w, spp)
    if 34264 in tags:
        m = tags[34264]
        geo = (m[3], m[0], m[7], m[5])
    else:
        geo = (tags[33922][3], tags[33550][0], tags[33922][4], -tags[33550][1])
    return arr, geo


def _wcs_on_grid(coverage: str, t: datetime | None) -> np.ndarray:
    pad = 0.05
    params = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
              ("coverageId", coverage), ("format", "image/tiff"),
              ("subset", f"Lat({LAT_MIN - pad},{LAT_MAX + pad})"),
              ("subset", f"Long({LON_MIN - pad},{LON_MAX + pad})")]
    if t is not None:
        params.append(("subset", f'time("{_iso(t)}")'))
    r = _get(BASE + "/ows", params)
    if "tiff" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"WCS error for {coverage}: {r.text[:300]}")
    arr, (lon0, dlon, lat0, dlat) = _read_plain_tiff(r.content)
    col = np.clip(np.floor((GRID_LON - lon0) / dlon).astype(int), 0, arr.shape[1] - 1)
    row = np.clip(np.floor((GRID_LAT - lat0) / dlat).astype(int), 0, arr.shape[0] - 1)
    return arr[row][:, col]


def fetch_h40b(t: datetime | None = None) -> tuple[datetime, np.ndarray]:
    """H SAF H40B (MTG FCI IR blended with microwave) instantaneous rain rate
    in mm/h, NUMERIC (float32), via WCS.  Native 0.0156 deg.  NaN = no data.
    Archive on EUMETView only from 2026-07-23."""
    if t is None:
        t = layer_times(LAYER_H40B)[1]
    rr = _wcs_on_grid("mtg_fd__h40b", t)[..., 0].astype(np.float32)
    rr[rr < 0] = np.nan      # nodata is -99
    return t, rr


def fetch_ir105_index(t: datetime | None = None) -> tuple[datetime, np.ndarray]:
    """MTG FCI IR10.5 as the 8-bit display index EUMETView serves (uint8,
    0 = no data, LOW = COLD cloud top, high = warm surface).  It is not
    calibrated to kelvin here: use it as a relative cold-top indicator
    (e.g. the coldest few % of the box) or as a picture."""
    if t is None:
        t = layer_times(LAYER_IR105)[1]
    return t, _wcs_on_grid("mtg_fd__ir105_hrfi", t)[..., 0].astype(np.uint8)


# ------------------------------------------------------------------ smoke test
if __name__ == "__main__":
    os = _os
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch", "r3-radar"))
    os.makedirs(outdir, exist_ok=True)
    now = datetime.now(timezone.utc)
    for lay in (LAYER_LIGHTNING, LAYER_H40B, LAYER_H60B, LAYER_IR105, LAYER_CONVECTION, LAYER_RDT):
        first, latest, step = layer_times(lay)
        print(f"{lay:24s} first {first:%Y-%m-%d} latest {latest:%H:%M}Z "
              f"({(now - latest).total_seconds() / 60:.0f} min old) step {int(step.total_seconds() // 60)} min")
    tl, li = fetch_lightning()
    print(f"lightning {tl:%H:%M}Z: cells with flashes {int((li > 0).sum())}, max {np.nanmax(li):.1f}")
    tw, liw = fetch_lightning_window(30)
    print(f"lightning last 30 min: cells with flashes {int((liw > 0).sum())}, max sum {liw.max():.1f}")
    t4, h40 = fetch_h40b()
    print(f"h40b {t4:%H:%M}Z: valid {np.isfinite(h40).mean():.1%}, raining cells {int(np.nansum(h40 > 0.1))}, max {np.nanmax(h40):.1f} mm/h")
    t6, h60 = fetch_h60b()
    print(f"h60b {t6:%H:%M}Z: raining cells {int(np.nansum(h60 > 0))}, max class {np.nanmax(h60):.1f} mm/h")
    ti, ir = fetch_ir105_index()
    print(f"ir105 {ti:%H:%M}Z: index min {ir.min()} p5 {np.percentile(ir, 5):.0f} median {np.median(ir):.0f} max {ir.max()}")
    th, h60h = fetch_h60b(datetime(2024, 10, 29, 18, 0, tzinfo=timezone.utc))
    print(f"h60b 2024-10-29 18:00Z: raining cells {int(np.nansum(h60h > 0))}, cells >= 25 mm/h {int(np.nansum(h60h >= 25))}")
    Image.fromarray(np.clip(liw[::-1] * 12, 0, 255).astype(np.uint8)).resize((NX * 2, NY * 2), Image.NEAREST).save(
        os.path.join(outdir, "smoke_lightning_30min.png"))
    Image.fromarray(ir[::-1]).resize((NX * 2, NY * 2), Image.NEAREST).save(os.path.join(outdir, "smoke_ir105.png"))
    print("saved smoke_lightning_30min.png, smoke_ir105.png in", outdir)
