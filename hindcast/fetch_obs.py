"""Historical observed rainfall downloader for hindcast verification (agent r4-obs).

Writes one CSV per date in ``hindcast/obs/<YYYY-MM-DD>.csv`` with columns

    source, station_id, name, lat, lon, date, precip_24h_mm, max_1h_mm,
    max_intensity, period_start, period_end

* ``precip_24h_mm``  rainfall accumulated in [period_start, period_end) (local time,
  Europe/Madrid). The period differs by source, so always read the two period columns.
* ``max_1h_mm``      maximum rolling 60-min accumulation inside the period (only where a
  5-min series is available: SAIH Júcar API, dates >= 2025-01-01).
* ``max_intensity``  peak 5-minute rain rate expressed in mm/h
  (SAIH: max of the 5-min intensity series; AVAMET: "P0" = max mm in 5 min, x12).

Sources (all keyless, all verified from this machine on 2026-10-01):

* ``avamet``        https://www.avamet.org/mx-meteoxarxa.php?data=YYYY-MM-DD
                    (civil day 00-24 local). Licence CC BY-NC-ND 4.0: attribution
                    "AVAMET", non-commercial, no derivatives. Unofficial, provisional.
* ``saih_chj``      https://saih.chj.es/lluviasIntervalo/D/D  (server daily value) and
                    /admin/variables/valor/<id>/<from>/<to> (5-min series). Only from
                    2025-01-01 (server returns [] before). Provisional, unvalidated.
* ``saih_chj_pdf``  daily rain reports https://saih.chj.es/docs/<YYYYMMDD>PLU*.pdf
                    (period 08:00 -> 08:00 local, sometimes several days). Back to 2016.
                    Coordinates come from fuzzy-matching the report names against the
                    current station list; unmatched rows keep empty lat/lon.

Usage::

    py -3.11 hindcast/fetch_obs.py                 # all default dates
    py -3.11 hindcast/fetch_obs.py 2024-10-29      # one date
    py -3.11 hindcast/fetch_obs.py --no-intensity  # skip the per-station requests

Polite by construction: every response is cached under ``scratch/r4-obs/cache`` and
network requests are spaced (1.5 s AVAMET, 1.0 s CHJ).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import html
import io
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "scratch" / "r4-obs" / "cache"
OUT = ROOT / "hindcast" / "obs"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126 Safari/537.36 riua-hindcast/0.1 (non-commercial research)")

DEFAULT_DATES = [
    "2024-10-29",
    "2023-09-02", "2023-09-03",
    "2024-11-13", "2024-11-14",
    "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06",
    "2025-09-28", "2025-09-29",
    "2025-10-09", "2025-10-10", "2025-10-11", "2025-10-12",
]

BBOX = (-2.4, 0.8, 37.6, 41.0)  # lon_min, lon_max, lat_min, lat_max

COLUMNS = ["source", "station_id", "name", "lat", "lon", "date", "precip_24h_mm",
           "max_1h_mm", "max_intensity", "period_start", "period_end"]

_last_request: dict[str, float] = {}
_session = requests.Session()
_session.headers["User-Agent"] = UA


# --------------------------------------------------------------------------- helpers
def http_get(url: str, key: str | None = None, delay: float = 1.0, refresh: bool = False,
             max_age_h: float | None = None, timeout: int = 60) -> bytes:
    """GET with on-disk cache and per-host throttling. Raises on HTTP errors."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if key is None:
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
    path = CACHE / re.sub(r"[^A-Za-z0-9_.\-]", "_", key)
    if path.exists() and not refresh:
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        if max_age_h is None or age_h <= max_age_h:
            return path.read_bytes()
    host = url.split("/")[2]
    wait = delay - (time.time() - _last_request.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    r = None
    for attempt in (1, 2, 3):  # transient resets happen (seen on saih.chj.es): back off and retry
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code < 500:
                break
        except requests.RequestException:
            if attempt == 3:
                raise
        finally:
            _last_request[host] = time.time()
        time.sleep(5 * attempt)
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content


def utm30_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """ETRS89 / UTM zone 30N (EPSG:25830) -> (lon, lat) in degrees (GRS80)."""
    a, f = 6378137.0, 1 / 298.257222101
    k0, lon0 = 0.9996, math.radians(-3.0)
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    c1 = ep2 * math.cos(phi1) ** 2
    t1 = math.tan(phi1) ** 2
    n1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    d = (x - 500000.0) / (n1 * k0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120
                  ) / math.cos(phi1)
    return math.degrees(lon), math.degrees(lat)


def es_float(s: str | None) -> float | None:
    """'1.010,2' -> 1010.2 ; '' -> None."""
    if s is None:
        return None
    s = s.strip().replace("\xa0", "")
    if not s or s in "-–":
        return None
    try:
        return float(s.replace(".", "").replace(",", "."))
    except ValueError:
        return None


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fix_text(s: str) -> str:
    """AVAMET pages mix ISO-8859-1 and UTF-8; repair UTF-8 read as latin-1."""
    s = html.unescape(s)
    if "Ã" in s or "Â" in s:
        try:
            s = s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"\s+", " ", s).strip()


def in_bbox(lat, lon) -> bool:
    return lat is not None and lon is not None and BBOX[0] <= lon <= BBOX[1] and BBOX[2] <= lat <= BBOX[3]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- AVAMET
AVAMET = "https://www.avamet.org/"
_avamet_coords: dict[str, list] | None = None
_AVAMET_COORD_FILE = CACHE / "avamet_coords.json"
_DMS = re.compile(r"(\d+)&deg;\s*(\d+)'\s*([\d.,]+)&quot;\s*([NS])\s*,\s*"
                  r"(\d+)&deg;\s*(\d+)'\s*([\d.,]+)&quot;\s*([EWO])")


def avamet_coords(refresh: bool = False) -> dict[str, list]:
    """Station id -> [lat, lon]. Seeded from the live map, completed lazily from fitxa pages."""
    global _avamet_coords
    if _avamet_coords is not None:
        return _avamet_coords
    coords: dict[str, list] = {}
    if _AVAMET_COORD_FILE.exists():
        coords = json.loads(_AVAMET_COORD_FILE.read_text(encoding="utf-8"))
    try:
        raw = http_get(AVAMET + "mxo-mxo.php", "avamet_mxo-mxo.html", delay=1.5,
                       refresh=refresh, max_age_h=24 * 30).decode("latin-1")
        m = re.search(r"var data = (\[.*?\]);", raw, re.S)
        for st in json.loads(m.group(1)) if m else []:
            try:
                coords[st["esta"]] = [float(st["lati"]), float(st["logi"])]
            except (KeyError, TypeError, ValueError):
                continue
    except Exception as exc:  # noqa: BLE001
        log("avamet live map failed:", exc)
    _avamet_coords = coords
    return coords


def _save_avamet_coords():
    if _avamet_coords is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        _AVAMET_COORD_FILE.write_text(json.dumps(_avamet_coords), encoding="utf-8")


def avamet_station_coord(sid: str) -> tuple[float | None, float | None]:
    coords = avamet_coords()
    if sid in coords:
        c = coords[sid]
        return (c[0], c[1]) if c else (None, None)
    lat = lon = None
    try:
        raw = http_get(f"{AVAMET}mx-fitxa.php?id={sid}", f"avamet_fitxa_{sid}.html", delay=1.5).decode("latin-1")
        m = _DMS.search(raw)
        if m:
            d1, m1, s1, h1, d2, m2, s2, h2 = m.groups()
            lat = int(d1) + int(m1) / 60 + float(s1.replace(",", ".")) / 3600
            lon = int(d2) + int(m2) / 60 + float(s2.replace(",", ".")) / 3600
            if h1 == "S":
                lat = -lat
            if h2 in "WO":
                lon = -lon
    except Exception as exc:  # noqa: BLE001
        log(f"avamet fitxa {sid} failed:", exc)
    coords[sid] = [round(lat, 6), round(lon, 6)] if lat is not None else []
    return lat, lon


def avamet_day(date: str) -> list[dict]:
    """Parse the MeteoXarxa daily table. Returns rows with id, name, precip (mm)."""
    raw = http_get(f"{AVAMET}mx-meteoxarxa.php?data={date}", f"avamet_day_{date}.html", delay=1.5).decode("latin-1")
    i = raw.find('<table class="tDades"')
    j = raw.find("</table>", i)
    out = []
    if i < 0:
        return out
    for row in re.findall(r"<tr>\s*<td class=\"rEsta\".*?</tr>", raw[i:j], re.S):
        m = re.search(r"id=(c\w+)", row)
        a = re.search(r"<a [^>]*>(.*?)</a>", row, re.S)
        tds = re.findall(r"<td class=\"rVal[^\"]*\">(.*?)</td>", row, re.S)
        if not m or not a or len(tds) < 5:
            continue
        name = fix_text(re.sub(r"<[^>]+>", " ", a.group(1)))
        p = es_float(re.sub(r"<[^>]+>", "", tds[4]))
        if p is None:
            continue
        out.append({"id": m.group(1), "name": name, "p": p})
    return out


def avamet_p0(sid: str, date: str) -> float | None:
    """Max rain in 5 minutes (mm) for the day from the station month page (column P0)."""
    month = date[:7]
    try:
        raw = http_get(f"{AVAMET}mx-mes.php?id={sid}&data={month}-01", f"avamet_mes_{sid}_{month}.html",
                       delay=1.5).decode("latin-1")
    except Exception as exc:  # noqa: BLE001
        log(f"avamet mes {sid} failed:", exc)
        return None
    i = raw.find("Detall diari")
    j = raw.find("</table>", i)
    if i < 0:
        return None
    day = date[8:10]
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", raw[i:j], re.S):
        tds = [re.sub(r"<[^>]+>|\s+", "", x) for x in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(tds) >= 16 and tds[0] == day:
            return es_float(tds[15])
    return None


def rows_avamet(date: str, intensity: bool, p0_min_mm: float, p0_max_stations: int) -> list[dict]:
    day = avamet_day(date)
    d0 = dt.date.fromisoformat(date)
    rows = []
    top = sorted((r for r in day if r["p"] >= p0_min_mm), key=lambda r: -r["p"])[:p0_max_stations]
    top_ids = {r["id"] for r in top} if intensity else set()
    for r in day:
        lat, lon = avamet_station_coord(r["id"])
        if lat is not None and not in_bbox(lat, lon):
            continue  # "zones limítrofes" far outside the domain
        p0 = avamet_p0(r["id"], date) if r["id"] in top_ids else None
        if p0 is not None and p0 * 12 > MAX_PLAUSIBLE_5MIN_RATE:
            p0 = None  # e.g. Barx la Drova 2025-09-29 "116.3 mm in 5 min": logger backlog, not rain rate
        rows.append({
            "source": "avamet", "station_id": r["id"], "name": r["name"],
            "lat": round(lat, 5) if lat is not None else "", "lon": round(lon, 5) if lon is not None else "",
            "date": date, "precip_24h_mm": r["p"], "max_1h_mm": "",
            "max_intensity": round(p0 * 12, 1) if p0 is not None else "",
            "period_start": f"{d0}T00:00", "period_end": f"{d0 + dt.timedelta(days=1)}T00:00",
        })
    _save_avamet_coords()
    return rows


# --------------------------------------------------------------------------- SAIH Júcar (API, >= 2025)
CHJ = "https://saih.chj.es"
CHJ_API_START = dt.date(2025, 1, 1)
MAX_PLAUSIBLE_5MIN_RATE = 300.0  # mm/h (= 25 mm in 5 min); above this a sample is treated as a backlog dump
_chj_stations: list[dict] | None = None


def chj_stations() -> list[dict]:
    """Current pluviometer list with coordinates (from the public map page)."""
    global _chj_stations
    if _chj_stations is None:
        raw = http_get(f"{CHJ}/mapa-lluvias", "chj_mapa-lluvias.html", max_age_h=24 * 30).decode("utf-8", "replace")
        m = re.search(r"let estaciones = (\[.*?\]);", raw, re.S)
        sts = json.loads(m.group(1)) if m else []
        for s in sts:
            # field names are misleading: "Lat" holds UTM X (easting), "Lon" holds UTM Y
            try:
                s["lon"], s["lat"] = utm30_to_lonlat(float(s["fldNCoordGPSLat"]), float(s["fldNCoordGPSLon"]))
            except (TypeError, ValueError):
                s["lon"] = s["lat"] = None
        _chj_stations = sts
    return _chj_stations


def chj_intensity_variable(station_id: str) -> str | None:
    raw = http_get(f"{CHJ}/chart-lluvia/{station_id}", f"chj_chart-lluvia_{station_id}.html").decode("utf-8", "replace")
    m = re.search(r"let varLluvia = (\[.*?\]);", raw, re.S)
    if not m:
        return None
    try:
        return str(json.loads(m.group(1))[0]["idVariable"])
    except (ValueError, KeyError, IndexError):
        return None


def chj_series_stats(var_id: str, date: str) -> tuple[float | None, float | None, float | None]:
    """(total_mm, max_1h_mm, max_intensity_mmh) from the 5-min intensity series of a local day."""
    d0 = dt.date.fromisoformat(date)
    a = f"{d0} 00:00:00".replace(" ", "%20").replace(":", "%3A")
    b = f"{d0 + dt.timedelta(days=1)} 00:00:00".replace(" ", "%20").replace(":", "%3A")
    raw = http_get(f"{CHJ}/admin/variables/valor/{var_id}/{a}/{b}", f"chj_val_{var_id}_{date}.json")
    data = json.loads(raw.decode("utf-8"))
    pts = {}
    for it in data:
        v = it.get("valor")
        if v is None or v < 0:
            continue
        t = dt.datetime.strptime(it["fecha"][:19], "%Y-%m-%dT%H:%M:%S")
        pts[t] = v / 12.0  # mm/h over 5 min -> mm
    if not pts:
        return None, None, None
    times = sorted(pts)
    best = 0.0
    j = 0
    acc = 0.0
    for i, t in enumerate(times):  # rolling 60-min window ending at t
        acc += pts[t]
        while times[j] <= t - dt.timedelta(hours=1):
            acc -= pts[times[j]]
            j += 1
        best = max(best, acc)
    return round(sum(pts.values()), 1), round(best, 1), round(max(pts.values()) * 12, 1)


def rows_chj_api(date: str, intensity: bool, series_min_mm: float) -> list[dict]:
    d0 = dt.date.fromisoformat(date)
    if d0 < CHJ_API_START:
        return []
    raw = http_get(f"{CHJ}/lluviasIntervalo/{date}/{date}", f"chj_intervalo_{date}.json")
    rows = []
    for s in json.loads(raw.decode("utf-8")):
        try:
            lon, lat = utm30_to_lonlat(float(s["fldNCoordGPSLat"]), float(s["fldNCoordGPSLon"]))
        except (TypeError, ValueError):
            lon = lat = None
        p = s.get("lluvia_int")
        if p is None:
            continue
        max1h = maxint = ""
        if intensity and p >= series_min_mm:
            try:
                var = chj_intensity_variable(str(s["idEstacionRemota"]))
                if var:
                    tot, m1, mi = chj_series_stats(var, date)
                    # QC of the provisional 5-min series (both cases seen in real data):
                    #  * telemetry backlog dumped into one sample (Torre Maçanes 2025-09-29:
                    #    1135 mm/h in one 5-min step) -> the timing of the rain is lost;
                    #  * series empty/incomplete while the daily value exists (Alacant 2025-10-09).
                    ok = (m1 is not None and mi is not None and mi <= MAX_PLAUSIBLE_5MIN_RATE
                          and tot is not None and tot >= 0.7 * float(p))
                    if ok:
                        max1h, maxint = m1, mi
            except Exception as exc:  # noqa: BLE001
                log(f"chj series {s.get('fldTNombre')} {date} failed:", exc)
        rows.append({
            "source": "saih_chj", "station_id": s.get("fldTCodigo") or s.get("idEstacionRemota"),
            "name": s.get("fldTNombre", ""), "lat": round(lat, 5) if lat is not None else "",
            "lon": round(lon, 5) if lon is not None else "", "date": date,
            "precip_24h_mm": round(float(p), 1), "max_1h_mm": max1h, "max_intensity": maxint,
            "period_start": f"{d0}T00:00", "period_end": f"{d0 + dt.timedelta(days=1)}T00:00",
        })
    return rows


# --------------------------------------------------------------------------- SAIH Júcar (PDF reports)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"])}
_PREFIX = re.compile(r"^(pluvionivometro|pluviometro|embalse|aforo|marco|camara de carga|canal|azud|deposito|"
                     r"balsa|cabecera|ea \d+|mc|ats|parada \d+)\b( de la| de los| de las| del| de| en)?\s*")
_chj_pdf_index: list[tuple[dt.date, str]] | None = None


def chj_pdf_index() -> list[tuple[dt.date, str]]:
    """[(filename date, href)] of every rain report listed in /informes."""
    global _chj_pdf_index
    if _chj_pdf_index is None:
        raw = http_get(f"{CHJ}/informes", "chj_informes.html", max_age_h=24 * 7).decode("utf-8", "replace")
        idx = []
        for href in sorted(set(re.findall(r'href="(/docs/[^"]+\.pdf)"', raw))):
            m = re.search(r"/docs/(20\d{6})[^/]*PLU", href)
            if m:
                try:
                    idx.append((dt.datetime.strptime(m.group(1), "%Y%m%d").date(), href))
                except ValueError:
                    pass
        _chj_pdf_index = idx
    return _chj_pdf_index


def chj_pdf_parse(href: str) -> dict | None:
    try:
        import pypdf  # optional dependency, only for pre-2025 dates
    except ImportError:
        log("pypdf not installed: py -3.11 -m pip install --user pypdf")
        return None
    raw = http_get(CHJ + href, "chj_pdf_" + href.rsplit("/", 1)[1])
    text = "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(io.BytesIO(raw)).pages)
    m1 = re.search(r"Fecha inicial:\s*(\d+)\s+([A-Z]{3})\s+(\d{4})\s+Hora:\s*(\d+):(\d+)", text)
    m2 = re.search(r"Fecha final\s*:\s*(\d+)\s+([A-Z]{3})\s+(\d{4})\s+Hora:\s*(\d+):(\d+)", text)
    if not m1 or not m2:
        return None

    def mk(m):
        return dt.datetime(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)), int(m.group(4)), int(m.group(5)))

    seen, rows = set(), []
    for name, prov, val in re.findall(r"^(.+?) \((CS|V|A|TE|CU|AB|T)\) (\d+\.\d)\s*$", text, re.M):
        if (name, prov) in seen:  # some gauges are listed under two basins
            continue
        seen.add((name, prov))
        rows.append((name.strip(), prov, float(val)))
    return {"start": mk(m1), "end": mk(m2), "rows": rows, "href": href}


def _match_station(text: str, prov: str) -> dict | None:
    """Match 'Pluviometro de Chiva Chiva' (name + municipality) to a current station."""
    provs = {"CS": "castell", "V": "valencia", "A": "alicante", "TE": "teruel", "CU": "cuenca",
             "AB": "albacete", "T": "tarragona"}
    nt = norm(text)
    best, best_score = None, 0.0
    for s in chj_stations():
        if provs[prov] not in norm(s.get("fldTProvincia") or ""):
            continue
        pob = norm(s.get("fldTPoblacion") or "")
        sname = _PREFIX.sub("", norm(s.get("fldTNombre") or ""))
        head = nt
        bonus = 0.0
        if pob and nt.endswith(pob):
            head = nt[: -len(pob)].strip()
            bonus = 0.25
        head = _PREFIX.sub("", head)
        score = difflib.SequenceMatcher(None, head, sname).ratio() + bonus
        if sname and (sname in head or head in sname):
            score += 0.2
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 0.85 else None


def rows_chj_pdf(date: str) -> list[dict]:
    d0 = dt.date.fromisoformat(date)
    want_a = dt.datetime.combine(d0, dt.time(8, 0))
    want_b = want_a + dt.timedelta(days=1)
    best = None
    for fdate, href in chj_pdf_index():
        if not (d0 < fdate <= d0 + dt.timedelta(days=6)):
            continue
        try:
            rep = chj_pdf_parse(href)
        except Exception as exc:  # noqa: BLE001
            log(f"pdf {href} failed:", exc)
            continue
        if not rep or not rep["rows"]:
            continue
        # tolerate "24:05 h" style periods; must cover D 08:00 -> D+1 08:00
        if rep["start"] <= want_a + dt.timedelta(minutes=10) and rep["end"] >= want_b - dt.timedelta(minutes=10):
            if best is None or (rep["end"] - rep["start"]) < (best["end"] - best["start"]):
                best = rep
    if best is None:
        return []
    hours = (best["end"] - best["start"]).total_seconds() / 3600
    source = "saih_chj_pdf" if hours <= 25 else "saih_chj_pdf_multiday"
    rows = []
    for name, prov, val in best["rows"]:
        st = _match_station(name, prov)
        lat = st["lat"] if st else None
        lon = st["lon"] if st else None
        rows.append({
            "source": source, "station_id": (st or {}).get("fldTCodigo", ""), "name": f"{name} ({prov})",
            "lat": round(lat, 5) if lat is not None else "", "lon": round(lon, 5) if lon is not None else "",
            "date": date, "precip_24h_mm": val, "max_1h_mm": "", "max_intensity": "",
            "period_start": best["start"].strftime("%Y-%m-%dT%H:%M"),
            "period_end": best["end"].strftime("%Y-%m-%dT%H:%M"),
        })
    return rows


# --------------------------------------------------------------------------- driver
def fetch_date(date: str, intensity: bool = True, p0_min_mm: float = 100.0, p0_max_stations: int = 25,
               series_min_mm: float = 40.0) -> list[dict]:
    rows: list[dict] = []
    for label, fn in (
        ("avamet", lambda: rows_avamet(date, intensity, p0_min_mm, p0_max_stations)),
        ("saih_chj", lambda: rows_chj_api(date, intensity, series_min_mm)),
        ("saih_chj_pdf", lambda: rows_chj_pdf(date)),
    ):
        try:
            got = fn()
            log(f"  {date} {label}: {len(got)} rows")
            rows += got
        except Exception as exc:  # noqa: BLE001  (each source isolated)
            log(f"  {date} {label} FAILED: {type(exc).__name__}: {exc}")
    return rows


def write_csv(date: str, rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{date}.csv"
    rows = sorted(rows, key=lambda r: (r["source"], -float(r["precip_24h_mm"])))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def summarize(date: str, rows: list[dict]) -> str:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    lines = [f"== {date}: {len(rows)} rows"]
    for src, rs in sorted(by.items()):
        with_xy = sum(1 for r in rs if r["lat"] != "")
        lines.append(f"  {src}: {len(rs)} stations ({with_xy} with coordinates), period "
                     f"{rs[0]['period_start']} -> {rs[0]['period_end']}")
        for r in sorted(rs, key=lambda r: -float(r["precip_24h_mm"]))[:10]:
            extra = ""
            if r["max_1h_mm"] != "":
                extra += f"  max1h={r['max_1h_mm']}"
            if r["max_intensity"] != "":
                extra += f"  imax={r['max_intensity']} mm/h"
            lines.append(f"      {float(r['precip_24h_mm']):7.1f}  {r['name']}{extra}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dates", nargs="*", default=DEFAULT_DATES, help="YYYY-MM-DD ...")
    ap.add_argument("--no-intensity", action="store_true", help="skip per-station intensity requests")
    ap.add_argument("--p0-min", type=float, default=100.0, help="AVAMET: only stations >= this daily total")
    ap.add_argument("--p0-max-stations", type=int, default=25)
    ap.add_argument("--series-min", type=float, default=40.0, help="SAIH: 5-min series only if daily >= this")
    args = ap.parse_args(argv)
    summary = []
    for date in args.dates:
        dt.date.fromisoformat(date)
        rows = fetch_date(date, not args.no_intensity, args.p0_min, args.p0_max_stations, args.series_min)
        path = write_csv(date, rows)
        s = summarize(date, rows)
        summary.append(s)
        print(s)
        print(f"  -> {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
