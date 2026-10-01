"""Real-time rain gauges and river gauges for the Comunitat Valenciana + upstream basins.

Public API
----------
``fetch_rain_gauges(sources=None, include_restricted=None) -> list[dict]``
    dict keys: id, name, lat, lon, source, t_utc, p_1h, p_3h, p_6h, p_12h, p_24h
    (mm; ``None`` when the network does not publish that window). Some sources add
    extra windows (``p_4h``, ``p_8h``, ``p_today``) and ``intensity_mmh``.

``fetch_river_gauges(sources=None, with_trend=True) -> list[dict]``
    dict keys: id, name, river, lat, lon, source, t_utc, level_m, flow_m3s,
    thr_low, thr_mid, thr_high (m3/s, SAIH Júcar "umbral bajo/medio/alto"),
    alert (0..3 from those thresholds, None if no thresholds), pct_section
    (SAIH Segura: % of the gauging section that is full), trend
    ('rising' | 'steady' | 'falling' | None).

Sources (every endpoint below was exercised from a normal PC on 2026-10-01, no key):

=============  ====================================================================
saih_chj       https://saih.chj.es/mapa-lluvias and /mapa-aforos (JSON embedded in the
               HTML). 182 pluviometers (1/4/12/24 h), 78 flow points with thresholds.
               5-min updates, provisional data. Cite "SAIH - Confederación
               Hidrográfica del Júcar".
saih_segura    ArcGIS REST layer of the CHS public viewer (rain 1/3/6/12/24 h with
               coordinates) + public iVisor endpoint for river level/flow. Cite
               "Confederación Hidrográfica del Segura"; provisional data.
saih_ebro      JSON API behind www.saihebro.com (Bergantes gauges; rain table has no
               coordinates, those rows come with lat/lon = None unless known).
aemet          Keyless "Últimos datos" GeoJSON bundle (hourly PREC of every AEMET
               automatic station, last 24 h). Cite "© AEMET".
avamet         RESTRICTED (CC BY-NC-ND 4.0). Dense volunteer network, 1/4/8/12/24 h.
meteoclimatic  RESTRICTED (CC BY-NC-ND 3.0). Only "rain today".
=============  ====================================================================

Restricted sources are only used when ``include_restricted=True`` or the environment
variable ``RIUA_GAUGES_RESTRICTED=1`` is set, because their licences forbid derivative
works: get written permission before feeding them into published products.

Design: every source is isolated (its failure never breaks the others), short
timeouts, no retries, small in-process TTL cache for static metadata, and nothing is
written to disk unless ``RIUA_RAW_DIR`` points to a directory (debug dumps).

Dependencies: requests + stdlib only.
"""
from __future__ import annotations

import datetime as dt
import html
import io
import json
import logging
import math
import os
import re
import tarfile
import time
from typing import Any, Callable

import requests

log = logging.getLogger("riua.gauges")

BBOX = (-2.4, 0.8, 37.6, 41.0)  # lon_min, lon_max, lat_min, lat_max
TIMEOUT = (8, 30)  # connect, read (s)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126 Safari/537.36 riua/0.1 (non-commercial flood-risk site)")

OPEN_RAIN_SOURCES = ("saih_chj", "saih_segura", "saih_ebro", "aemet")
RESTRICTED_RAIN_SOURCES = ("avamet", "meteoclimatic")
RIVER_SOURCES = ("saih_chj", "saih_segura", "saih_ebro")

# Two Spanish government hosts cannot be verified with the default (certifi) bundle:
#   * www.chsegura.es chains to the new FNMT root "AC RAIZ FNMT-RCM SERVIDORES SEGUROS G2R"
#     (in the Microsoft root programme, SHA-1 1987948DAB7D62009E2D6BBED883DC3AF56799E8, not
#     yet in certifi 2026.07.22);
#   * www.saihebro.com does not send its intermediate "AC Componentes Informáticos"
#     (issued by "AC RAIZ FNMT-RCM", which certifi does trust).
# Instead of disabling TLS verification we append those two public CA certificates to
# the certifi bundle. Both expire (2040 / 2028): if verification starts failing again,
# refresh them with `openssl s_client -showcerts -connect <host>:443`.
_EXTRA_CA_PEM = """
-----BEGIN CERTIFICATE-----
MIIFlTCCA32gAwIBAgIUKg4cpNS2hPioWgPAf0hBdOZpz0AwDQYJKoZIhvcNAQEM
BQAwUjELMAkGA1UEBhMCRVMxETAPBgNVBAoMCEZOTVQtUkNNMTAwLgYDVQQDDCdB
QyBSQUlaIEZOTVQtUkNNIFNFUlZJRE9SRVMgU0VHVVJPUyBHMlIwHhcNMjUxMjE2
MTAzNDIwWhcNNDAxMjEyMTAzNDE5WjBSMQswCQYDVQQGEwJFUzERMA8GA1UECgwI
Rk5NVC1SQ00xMDAuBgNVBAMMJ0FDIFJBSVogRk5NVC1SQ00gU0VSVklET1JFUyBT
RUdVUk9TIEcyUjCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBALQ5a+qw
hJLRhbQIUIwPLu/7VIlvnpwiQdmDtH+x6dYHfGFJq53a8p5U6upv8qCFPjKdeIf5
FdCGlZb6ZQGn+OG2lG1t2hNUipNIAqfYWdqJzDGIkSDnCgubEsR85/CSB10E9bZB
3FkQJpYgqEsfIz84C0DQNYToJz7qdObLPUrp/BUZvFa/L0wnsx2/RirDmau83zlS
6MYdQTZQDbmsy/0pi+sV8Cu+GYfYcr2DALC111t9RIZoQ59evDzcJsa4S/y5t3qE
gMZs9UzNoGiix2fR/N0QP02cuUiW8DVLcNueqG/ow949OxGwSkJc7dkaZDTeezRB
yG/02e/MBhrfgUodp9aoGxO/sbtpr6Ptvp+FC5SwnTJtqAtxZl2LUPhNyN9eCebl
rJ12rrLI7mzpwPTM9UiJa9pYOfegeXieDwoHufFHR2Hg3RQm/lo75BSvaman89NT
va5tS1bx2Rwed2CEKoT8WrWETGBGKupbAp+c9eE7z39Kx7MkooNqX2TbC5k3OIkc
bbeBu7WByP7OSpM1bz4fuj2JgDsmyMtw4J4pD8+w0k65j+Pmpz82+tbxuoEWw5MP
FRQquMjQ+5b63zlguLtdy+2U4uEhGap0v0UOm9ZIS1tTKGc2vJU4NHBM5nF91AUA
wifsxQu2JUBkO9R6FFmL261K+jpLZETs2dpRAgMBAAGjYzBhMA8GA1UdEwEB/wQF
MAMBAf8wHwYDVR0jBBgwFoAUI4JUVDBhTKBOgbmDiPLKBfYZuJswHQYDVR0OBBYE
FCOCVFQwYUygToG5g4jyygX2GbibMA4GA1UdDwEB/wQEAwIBBjANBgkqhkiG9w0B
AQwFAAOCAgEABImYw67waq5tDwafixQWg2GqO4C1n+M1iEn+UTzmt3BmLojZAgFW
J/0mvcTJcCJEo2+d1NtsTRj2DL1eOfRx0JjiM0a2xEPex+iHNJnmY3KrFo0p5Ilq
AAONNvDbjrZblz7jiO6DaQl1RpX6bjxobPzGM3og0r0TQPkM2KcOnjkfaxsF54sz
Rdm+aqs4mPlVSj09tt1Lv/LwuCboY47Sgza/w6qc/Pm0V5w6cLmG6INiBZv462Yq
/zV2ghkF/w4jg4Eny/euSGnyjLxzMjbYT6tF9/4hVW+Y72US04J8/F4rqa1bMw64
usT+i9M1cFU3suPAX/8lDG2xUplzIoNuFrJqx+LgIXA+CkpMTHLanp+7YelZBnB2
1pEi86THFrJdvttbi7DQdHgJLBjAVGadYdSWc17M3YOXQNWb802aLVBPLEts5Ypl
b+s1bzdN7KPutkzByXMQPbgdZvv26ZbPYEIz1Zq11dXElF+k1tR50f3d/SAfQU2c
hgM/F8Io4fsYQ+mz2hPIgwVt5c5mjq4mrPsyQ9bMBzZd7JlnOlR1DOSCrVBvdDd7
so3XBVp/blKClLGY7PlFsn6FjFUOyWBHkCRGn3zj8adZdAHhgTQJeFUTFojXKGXq
b9LCPP72TbTYzAGD9iSfRE4hFklQsFu9GDmLWpsu6OOK5w0NIS3odxc=
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIIG1jCCBL6gAwIBAgIQNMarBE42mRJRyCULbJTWwDANBgkqhkiG9w0BAQsFADA7
MQswCQYDVQQGEwJFUzERMA8GA1UECgwIRk5NVC1SQ00xGTAXBgNVBAsMEEFDIFJB
SVogRk5NVC1SQ00wHhcNMTMwNjI0MTA1MjU5WhcNMjgwNjI0MTA1MjU5WjBHMQsw
CQYDVQQGEwJFUzERMA8GA1UECgwIRk5NVC1SQ00xJTAjBgNVBAsMHEFDIENvbXBv
bmVudGVzIEluZm9ybcOhdGljb3MwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQCXVx8rdbF7/xY44CaSqzzGo5BhvzA8knxC/3KJYVzTf+CkOvMxMUDub8b0
h38MDujm/RKZhBNOWbKhxF3U61ZVhcR9xOCciuS/soT80m3BByxAKcZsNka0jCA4
XRkglDaAFxCHEZ06MOnvXsSOZDfPYahbQ3VFCVycJuhlHdAwSpmceQwcRYkR6YgX
wTiyzCNGivMKAmRS3dItqDOmDW/nxiDFq/Jd8VWY7GFkwbbAeqYId8FjN8zfvafu
nsB9SLFkUjPPMeqfmC7Bdh7HMxLpaOXROwH201cmlebiPkn0xSFxXFqwhhr6yN8U
QYZ3O/+xdHLrS6DS9+CJUF6d09ijAgMBAAGjggLIMIICxDASBgNVHRMBAf8ECDAG
AQH/AgEAMA4GA1UdDwEB/wQEAwIBBjAdBgNVHQ4EFgQUGfhYLxTWpsybBJgIDUzX
qwCng2UwgZgGCCsGAQUFBwEBBIGLMIGIMEkGCCsGAQUFBzABhj1odHRwOi8vb2Nz
cGZubXRyY21jYS5jZXJ0LmZubXQuZXMvb2NzcGZubXRyY21jYS9PY3NwUmVzcG9u
ZGVyMDsGCCsGAQUFBzAChi9odHRwOi8vd3d3LmNlcnQuZm5tdC5lcy9jZXJ0cy9B
Q1JBSVpGTk1UUkNNLmNydDAfBgNVHSMEGDAWgBT3fcX9xOiaG3dkp/UdoMy/h2Ca
bTCB6wYDVR0gBIHjMIHgMIHdBgRVHSAAMIHUMCkGCCsGAQUFBwIBFh1odHRwOi8v
d3d3LmNlcnQuZm5tdC5lcy9kcGNzLzCBpgYIKwYBBQUHAgIwgZkMgZZTdWpldG8g
YSBsYXMgY29uZGljaW9uZXMgZGUgdXNvIGV4cHVlc3RhcyBlbiBsYSBEZWNsYXJh
Y2nDs24gZGUgUHLDoWN0aWNhcyBkZSBDZXJ0aWZpY2FjacOzbiBkZSBsYSBGTk1U
LVJDTSAoIEMvIEpvcmdlIEp1YW4sIDEwNi0yODAwOS1NYWRyaWQtRXNwYcOxYSkw
gdQGA1UdHwSBzDCByTCBxqCBw6CBwIaBkGxkYXA6Ly9sZGFwZm5tdC5jZXJ0LmZu
bXQuZXMvQ049Q1JMLE9VPUFDJTIwUkFJWiUyMEZOTVQtUkNNLE89Rk5NVC1SQ00s
Qz1FUz9hdXRob3JpdHlSZXZvY2F0aW9uTGlzdDtiaW5hcnk/YmFzZT9vYmplY3Rj
bGFzcz1jUkxEaXN0cmlidXRpb25Qb2ludIYraHR0cDovL3d3dy5jZXJ0LmZubXQu
ZXMvY3Jscy9BUkxGTk1UUkNNLmNybDANBgkqhkiG9w0BAQsFAAOCAgEAo2bsQ2xL
Dcyodieqjd+uy/lfxDw/MbrAq/ZaNFkIlcypUYamOM4vrm5rz8oLjPCoLkJ48P+n
P08Gkcl5Q6q6VFcZLia+U3gfHXrkyqToQlrtViGCGH3xA4u56XtMHGXSdk9vQ0yD
nW5f7bUEkp+uvcKewrOvNcpbIAgD4eU7gdOS0w7BagcFRBgTKBw2s3z73fRZtouJ
g/atmWYtXbBsfNjph+pCh+h5sbSyZUVzO5AemyjpYYYNMWDQrTXq+7O8zIPuPaNE
SjEexuzn+VjHG90RlUK1LygARi+Ir0opD2w6erb/hK8Eea7MFdKQ2ASqNBGJggNo
5vfPVvjHiL+Antmh7mQSKL+4YwFU64d4KK9k0C1mbJethDQFKcjTK1vMvnXFiups
IuyTqwKauo7u2zMKzY4r3VYOW9TpMyLPFIY8pII5GyNzXlL0F4nscOvduTEPEYqx
eNJfpDDPY/DO8WfxgdRTy2W3D/UoAulb+Y+nuzGGCtFQrsSMQX487R+aY0nWot/h
ajef6BcPuxhDfQrg5IafrISVmcJAplb3tXhh0sz7RbYz6jf1bke4eU5fnrTMtGlV
teUL2vjrfUPHW07kBJuaQ7sxORNV3bpHisOnHj+AriQzCn5vINpSHW6hTm7IfRkb
ltu/aQrsMuUhP7HE/v+uXe5CuboV5ubZhHU=
-----END CERTIFICATE-----
"""


def _ca_bundle() -> str | bool:
    """certifi bundle + the two FNMT certificates above, written once to a temp file."""
    try:
        import tempfile

        import certifi

        with open(certifi.where(), "r", encoding="utf-8") as fh:
            base = fh.read()
        path = os.path.join(tempfile.gettempdir(), "riua_ca_bundle.pem")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(base + "\n" + _EXTRA_CA_PEM)
        return path
    except Exception as exc:  # noqa: BLE001 - fall back to default verification
        log.warning("could not build CA bundle (%s); Segura/Ebro sources will fail TLS", exc)
        return True


_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Language": "es-ES,es;q=0.9"})
_session.verify = _ca_bundle()
_cache: dict[str, tuple[float, Any]] = {}
last_errors: dict[str, str] = {}  # source -> last error text (for health endpoints)


# --------------------------------------------------------------------------- helpers
def _dump(name: str, content: bytes) -> None:
    raw_dir = os.environ.get("RIUA_RAW_DIR")
    if not raw_dir:
        return
    try:
        os.makedirs(raw_dir, exist_ok=True)
        with open(os.path.join(raw_dir, re.sub(r"[^A-Za-z0-9_.\-]", "_", name)), "wb") as fh:
            fh.write(content)
    except OSError:
        pass


def _get(url: str, name: str, **kw) -> bytes:
    r = _session.get(url, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    _dump(name, r.content)
    return r.content


def _post(url: str, name: str, data: str, **kw) -> bytes:
    headers = {"Content-Type": "application/x-www-form-urlencoded", "X-Requested-With": "XMLHttpRequest"}
    headers.update(kw.pop("headers", {}))
    r = _session.post(url, data=data, timeout=TIMEOUT, headers=headers, **kw)
    r.raise_for_status()
    _dump(name, r.content)
    return r.content


def _cached(key: str, ttl_s: float, fn: Callable[[], Any]) -> Any:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl_s:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def _in_bbox(lat, lon) -> bool:
    return lat is not None and lon is not None and BBOX[0] <= lon <= BBOX[1] and BBOX[2] <= lat <= BBOX[3]


def _num(v) -> float | None:
    """Tolerant float: accepts 12.3, '12,3', '1.234,5', '', None, -1 sentinel stays -1."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and math.isnan(v) else float(v)
    s = str(v).strip().replace("\xa0", "")
    if not s or s in ("-", "--", "null"):
        return None
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _r1(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def utm30_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """ETRS89 / UTM zone 30N (EPSG:25830) -> (lon, lat) degrees."""
    a, f = 6378137.0, 1 / 298.257222101
    k0, lon0 = 0.9996, math.radians(-3.0)
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    mu = (y / k0) / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
          + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    c1 = ep2 * math.cos(p1) ** 2
    t1 = math.tan(p1) ** 2
    n1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5
    d = (x - 500000.0) / (n1 * k0)
    lat = p1 - (n1 * math.tan(p1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(p1)
    return math.degrees(lon), math.degrees(lat)


def _last_sunday(year: int, month: int) -> dt.datetime:
    d = dt.datetime(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def madrid_to_utc(local: dt.datetime) -> dt.datetime:
    """Naive Europe/Madrid wall time -> naive UTC (EU DST rule, no tzdata needed)."""
    start = _last_sunday(local.year, 3) + dt.timedelta(hours=2)   # 02:00 local -> CEST
    end = _last_sunday(local.year, 10) + dt.timedelta(hours=3)    # 03:00 CEST -> CET
    return local - dt.timedelta(hours=2 if start <= local < end else 1)


def _iso(t: dt.datetime | None) -> str | None:
    return None if t is None else t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_local(s: str | None, fmts=("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M")) -> str | None:
    if not s:
        return None
    for fmt in fmts:
        try:
            return _iso(madrid_to_utc(dt.datetime.strptime(s.strip(), fmt)))
        except ValueError:
            continue
    return None


def _iso_utc(s: str | None) -> str | None:
    """'2026-10-01T06:45:00.000Z' -> '2026-10-01T06:45:00Z'."""
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", s)
    return m.group(1) + "Z" if m else None


def _rain(id_, name, lat, lon, source, t_utc, p1=None, p3=None, p6=None, p12=None, p24=None, **extra) -> dict:
    d = {"id": str(id_), "name": name, "lat": None if lat is None else round(lat, 5),
         "lon": None if lon is None else round(lon, 5), "source": source, "t_utc": t_utc,
         "p_1h": _r1(p1), "p_3h": _r1(p3), "p_6h": _r1(p6), "p_12h": _r1(p12), "p_24h": _r1(p24)}
    d.update({k: v for k, v in extra.items() if v is not None})
    return d


def _fix_text(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    if "Ã" in s or "Â" in s:
        try:
            s = s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- SAIH Júcar
CHJ = "https://saih.chj.es"


def _chj_embedded(page: str, var: str) -> list[dict]:
    raw = _get(f"{CHJ}/{page}", f"chj_{page}.html").decode("utf-8", "replace")
    m = re.search(r"let %s = (\[.*?\]);" % var, raw, re.S)
    if not m:
        raise ValueError(f"saih_chj: variable '{var}' not found in /{page} (page layout changed?)")
    return json.loads(m.group(1))


def _chj_lonlat(s: dict) -> tuple[float | None, float | None]:
    # the JSON calls them GPSLat/GPSLon but they are UTM30N easting/northing
    try:
        return utm30_to_lonlat(float(s["fldNCoordGPSLat"]), float(s["fldNCoordGPSLon"]))
    except (KeyError, TypeError, ValueError):
        return None, None


def _rain_saih_chj() -> list[dict]:
    out = []
    for s in _chj_embedded("mapa-lluvias", "estaciones"):
        lon, lat = _chj_lonlat(s)
        if not _in_bbox(lat, lon):
            continue
        out.append(_rain(
            s.get("fldTCodigo") or s.get("idEstacionRemota"), s.get("fldTNombre", ""), lat, lon, "saih_chj",
            _iso_utc(s.get("fecha_1h") or s.get("fecha_24h")),
            p1=_num(s.get("lluvia_1h")), p12=_num(s.get("lluvia_12h")), p24=_num(s.get("lluvia_24h")),
            p_4h=_r1(_num(s.get("lluvia_4h"))), municipality=s.get("fldTPoblacion"),
            station_ref=str(s.get("idEstacionRemota")), online=bool(s.get("fldTEstado", True))))
    return out


def _chj_trend(id_variable: str) -> str | None:
    """Compare the last value with the one ~1 h before (5-min series, public chart endpoint)."""
    now = dt.datetime.utcnow()
    # the endpoint takes local wall time; ask for a generous window around now
    a = (now - dt.timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    b = (now + dt.timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    raw = _get(f"{CHJ}/admin/variables/valor/{id_variable}/{requests.utils.quote(a)}/{requests.utils.quote(b)}",
               f"chj_val_{id_variable}.json")
    pts = [(p["fecha"], p["valor"]) for p in json.loads(raw.decode("utf-8")) if p.get("valor") is not None]
    if len(pts) < 6:
        return None
    pts.sort()
    last = pts[-1][1]
    prev = pts[max(0, len(pts) - 13)][1]
    tol = max(0.05, 0.10 * max(abs(prev), abs(last)))
    return "rising" if last - prev > tol else "falling" if prev - last > tol else "steady"


def _river_saih_chj(with_trend: bool) -> list[dict]:
    out = []
    for s in _chj_embedded("mapa-aforos", "aforos"):
        lon, lat = _chj_lonlat(s)
        if not _in_bbox(lat, lon):
            continue
        flow = _num(s.get("lastValue") if s.get("lastValue") is not None else s.get("fldNUltimoValor"))
        lo, mid, hi = (_num(s.get(k)) for k in ("fldFUmbralBajo", "fldFUmbralMedio", "fldFUmbralAlto"))
        alert = None
        if flow is not None and lo is not None:
            alert = 3 if hi is not None and flow >= hi else 2 if mid is not None and flow >= mid else 1 if flow >= lo else 0
        var_name = s.get("fldTNombreVariable") or ""
        river = re.sub(r"^CAUDAL( SALIDA)?\s+", "", var_name).strip().title()
        out.append({
            "id": str(s.get("idVariable")), "name": s.get("fldTNombre", ""), "river": river,
            "lat": round(lat, 5), "lon": round(lon, 5), "source": "saih_chj",
            "t_utc": _iso_utc(s.get("fldDFechaComunicacion")), "level_m": None,
            "flow_m3s": None if flow is None else round(flow, 2),
            "thr_low": lo, "thr_mid": mid, "thr_high": hi, "alert": alert, "pct_section": None, "trend": None,
            "kind": "reservoir_outflow" if s.get("fldTTipo") == "Em" else "gauge",
            "subbasin": s.get("fldTSubCuenca"), "municipality": s.get("fldTPoblacion"),
            "station_code": s.get("fldTCodigo"),
        })
    if with_trend:
        # only where it matters (keeps the request count tiny): flow above 30 % of the low threshold
        hot = [g for g in out if g["flow_m3s"] is not None and g["thr_low"] and g["flow_m3s"] >= 0.3 * g["thr_low"]]
        for g in sorted(hot, key=lambda g: -(g["flow_m3s"] / g["thr_low"]))[:12]:
            try:
                g["trend"] = _chj_trend(g["id"])
            except Exception as exc:  # noqa: BLE001
                log.debug("chj trend %s failed: %s", g["id"], exc)
    return out


# --------------------------------------------------------------------------- SAIH Segura
SEG_ARCGIS = ("https://www.chsegura.es/server/rest/services/VISOR_CHSIC3/"
              "VISOR_PUBLICO_ETRS89_v5_Web_Capas/MapServer")
SEG_IVISOR = "https://saihweb.chsegura.es/apps/iVisor/obtener_datos.php"


def _seg_layer(layer: int) -> list[dict]:
    url = f"{SEG_ARCGIS}/{layer}/query?where=1%3D1&outFields=*&outSR=4326&f=json&resultRecordCount=2000"
    d = json.loads(_get(url, f"segura_layer{layer}.json").decode("utf-8"))
    if "features" not in d:
        raise ValueError(f"saih_segura layer {layer}: {str(d)[:200]}")
    return d["features"]


def _rain_saih_segura() -> list[dict]:
    feats = _seg_layer(13)
    stamps: dict[str, str] = {}
    try:  # timestamps (the ArcGIS layer has none); failure here is not fatal
        for it in json.loads(_post(SEG_IVISOR, "segura_pluvios_1h.json", "action=consultar_pluvios&tipo=1")):
            stamps[it.get("CodVariableHidrologica", "")] = it.get("FechaUltimoDato")
    except Exception as exc:  # noqa: BLE001
        log.debug("segura iVisor timestamps failed: %s", exc)
    out = []
    for f in feats:
        a, g = f.get("attributes", {}), f.get("geometry") or {}
        lon, lat = g.get("x"), g.get("y")
        if not _in_bbox(lat, lon):
            continue
        code = a.get("CodVariableHidrologica", "")
        name = re.sub(r"^Pluvi[oó]metro\s*\(\s*(.*?)\s*\)$", r"\1", a.get("DenominacionVariable") or "") \
            or a.get("DenominacionPtoMedicion", "")
        out.append(_rain(code, name, lat, lon, "saih_segura", _parse_local(stamps.get(code)),
                         p1=_num(a.get("LluviaUltimaHora")), p3=_num(a.get("LluviaUltimas3Horas")),
                         p6=_num(a.get("LluviaUltimas6Horas")), p12=_num(a.get("LluviaUltimas12Horas")),
                         p24=_num(a.get("LluviaUltimas24Horas")), municipality=a.get("Municipio")))
    return out


def _seg_points() -> dict[str, dict]:
    """CodPuntoMedicion -> {lat, lon, name} from the flow + level layers (static metadata)."""
    pts: dict[str, dict] = {}
    for layer in (10, 11):
        for f in _seg_layer(layer):
            a, g = f.get("attributes", {}), f.get("geometry") or {}
            code = a.get("CodPuntoMedicion")
            if code and code not in pts and g.get("x") is not None:
                pts[code] = {"lat": g["y"], "lon": g["x"], "name": a.get("DenominacionPtoMedicion", ""),
                             "kind": a.get("TipologiaPuntoMedicion", "")}
    return pts


def _river_saih_segura(with_trend: bool) -> list[dict]:
    pts = _cached("segura_points", 24 * 3600, _seg_points)
    data = json.loads(_post(SEG_IVISOR, "segura_cauces.json", "action=consultar_cauces_topo"))
    now = _iso(dt.datetime.utcnow().replace(microsecond=0))
    out = []
    for it in data:
        code = it.get("CodPuntoMedicion", "")
        meta = pts.get(code)
        if not meta or not _in_bbox(meta["lat"], meta["lon"]):
            continue
        name = meta["name"] or it.get("NombreCortoPM", "")
        m = re.search(r"\b(r[ií]o|rambla|canal|azarbe|barranco)\s+([A-ZÁÉÍÓÚÑa-záéíóúñ]+)", name)
        out.append({
            "id": code, "name": name, "river": (m.group(1) + " " + m.group(2)).title() if m else "",
            "lat": round(meta["lat"], 5), "lon": round(meta["lon"], 5), "source": "saih_segura",
            "t_utc": now,  # endpoint gives "latest value" without a timestamp: fetch time used
            "t_is_fetch_time": True,
            "level_m": _num(it.get("UltimoDatoNivel")), "flow_m3s": _num(it.get("UltimoDatoCaudal")),
            "thr_low": None, "thr_mid": None, "thr_high": None, "alert": None,
            "pct_section": _num(it.get("PorcentajeNivel")), "section_top_m": _num(it.get("CotaMaximaSeccion")),
            "trend": None, "kind": "canal" if "canal" in meta.get("kind", "").lower() else "gauge",
        })
    return out


# --------------------------------------------------------------------------- SAIH Ebro
EBRO = "https://www.saihebro.com"
EBRO_MAPS = ("mapa-aforos-H9-guadalope-martin",)  # Bergantes (els Ports, N Castellón)
_EBRO_HDR = {"X-Requested-With": "XMLHttpRequest", "Referer": EBRO + "/"}
_EBRO_TREND = {"arriba": "rising", "abajo": "falling", "derecha": "steady"}
_EBRO_PROVINCES = ("castellon", "teruel", "tarragona")


def _ebro_map_stations() -> list[dict]:
    sts = []
    for slug in EBRO_MAPS:
        d = json.loads(_get(f"{EBRO}/api/mapa/getDatosMapa?slug={slug}", f"ebro_{slug}.json", headers=_EBRO_HDR))
        sts += d.get("DATOS", [])
    return sts


def _river_saih_ebro(with_trend: bool) -> list[dict]:
    out = []
    for s in _ebro_map_stations():
        try:
            lon, lat = utm30_to_lonlat(float(s["LR_UTM_X"]), float(s["LR_UTM_Y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not _in_bbox(lat, lon):
            continue
        level = flow = trend = t = None
        for tag in s.get("TAGS", []):
            kind = tag.get("LS_TIPO_SENAL")
            if kind == "NRIO":
                level = _num(tag.get("VALOR"))
                trend = _EBRO_TREND.get(tag.get("LS_TAG_TENDENCIA"))
                t = tag.get("ULTIMA_FECHA")
            elif kind == "QRIO":
                flow = _num(tag.get("VALOR"))
                t = t or tag.get("ULTIMA_FECHA")
        if level is None and flow is None:
            continue
        short = s.get("LR_NOMBRE_CORTO", "")
        out.append({
            "id": s.get("CW_REMOTA_TXT", ""), "name": short.title(), "river": short.split("-")[0].title(),
            "lat": round(lat, 5), "lon": round(lon, 5), "source": "saih_ebro", "t_utc": _parse_local(t),
            "level_m": level, "flow_m3s": flow, "thr_low": None, "thr_mid": None, "thr_high": None,
            "alert": None, "pct_section": None, "trend": trend, "kind": "gauge",
        })
    return out


def _rain_saih_ebro() -> list[dict]:
    table = json.loads(_get(f"{EBRO}/api/pluviometrias/getTablaPluviometrias", "ebro_tabla_pluv.json",
                            headers=_EBRO_HDR))
    coords: dict[str, tuple[float, float]] = {}
    try:
        for s in _cached("ebro_map", 6 * 3600, _ebro_map_stations):
            lon, lat = utm30_to_lonlat(float(s["LR_UTM_X"]), float(s["LR_UTM_Y"]))
            coords[s.get("CW_REMOTA_TXT", "")] = (lat, lon)
    except Exception as exc:  # noqa: BLE001
        log.debug("ebro coords failed: %s", exc)
    now = _iso(dt.datetime.utcnow().replace(microsecond=0))
    out = []
    for it in table:
        prov = (it.get("provincia") or "").lower().replace("ó", "o")
        if prov not in _EBRO_PROVINCES:
            continue
        lat, lon = coords.get(it.get("codigo", ""), (None, None))
        if lat is not None and not _in_bbox(lat, lon):
            continue
        if lat is None and prov != "castellon":
            continue  # no coordinates in this API: keep only the rows we know are inside the CV
        p1, p24 = _num(it.get("col1")), _num(it.get("col3"))
        out.append(_rain(it.get("codigo"), it.get("nombre", ""), lat, lon, "saih_ebro", now,
                         p1=None if p1 == -1 else p1, p24=None if p24 == -1 else p24,
                         p_today=None if _num(it.get("col2")) == -1 else _r1(_num(it.get("col2"))),
                         province=it.get("provincia"), t_is_fetch_time=True))
    return out


# --------------------------------------------------------------------------- AEMET (keyless bundle)
AEMET_PAGE = "https://www.aemet.es/es/eltiempo/observacion/ultimosdatos"


def _aemet_bundle() -> dict[str, dict]:
    """IDEMA -> {lat, lon, hours: {utc_iso: mm}} from the public GeoJSON tarball (~3 MB)."""
    page = _get(AEMET_PAGE, "aemet_ultimosdatos.html").decode("iso-8859-15", "replace")
    m = re.search(r"/es/geojson/download/udat/udat_descarga_\d+\.tar\.gz", page)
    if not m:
        raise ValueError("aemet: download link not found in ultimosdatos page")
    blob = _get("https://www.aemet.es" + m.group(0), "aemet_udat.tar.gz")
    st: dict[str, dict] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or "/prec/" not in member.name or "_PB_" not in member.name:
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            for f in json.load(fh).get("features", []):
                p, g = f.get("properties", {}), (f.get("geometry") or {}).get("coordinates") or [None, None]
                lon, lat = g[0], g[1]
                if not _in_bbox(lat, lon) or p.get("FINT") is None:
                    continue
                rec = st.setdefault(p.get("IDEMA", ""), {"lat": lat, "lon": lon, "hours": {}})
                val = p.get("PREC")
                if isinstance(val, (int, float)):
                    rec["hours"][p["FINT"][:19]] = float(val)
    return st


def _rain_aemet() -> list[dict]:
    st = _cached("aemet_bundle", 15 * 60, _aemet_bundle)
    out = []
    for idema, rec in st.items():
        hours = sorted(rec["hours"].items())
        if not hours:
            continue
        t_last = dt.datetime.strptime(hours[-1][0], "%Y-%m-%dT%H:%M:%S")

        def acc(n: int) -> float | None:
            vals = [v for t, v in hours if dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") > t_last - dt.timedelta(hours=n)]
            need = n if n <= 3 else math.ceil(0.8 * n)
            return sum(vals) if len(vals) >= need else None

        out.append(_rain(idema, f"AEMET {idema}", rec["lat"], rec["lon"], "aemet", _iso(t_last),
                         p1=acc(1), p3=acc(3), p6=acc(6), p12=acc(12), p24=acc(24)))
    return out


# --------------------------------------------------------------------------- AVAMET (restricted)
AVAMET = "https://www.avamet.org/"


def _avamet_coords() -> dict[str, tuple[float, float]]:
    raw = _get(AVAMET + "mxo-mxo.php", "avamet_mxo-mxo.html").decode("latin-1")
    m = re.search(r"var data = (\[.*?\]);", raw, re.S)
    if not m:
        raise ValueError("avamet: 'var data' not found in mxo-mxo.php")
    out = {}
    for s in json.loads(m.group(1)):
        try:
            out[s["esta"]] = (float(s["lati"]), float(s["logi"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _rain_avamet() -> list[dict]:
    coords = _cached("avamet_coords", 24 * 3600, _avamet_coords)
    raw = _get(AVAMET + "mxo-mxo-prec.php", "avamet_mxo-mxo-prec.html").decode("latin-1")
    # main table "mxoCompletaN_0" (the CV) + "mxoCompleta" (zones limítrofes); stop before
    # the "extremes" tables, which repeat the same stations
    i = raw.find('<table id="mxoCompleta')
    j = raw.find('<table id="mxoExtrems', i)
    if i < 0:
        raise ValueError("avamet: table mxoCompleta not found")
    out = []
    for row in re.findall(r"<tr[^>]*>\s*<td class=\"rEsta.*?</tr>", raw[i:j if j > 0 else None], re.S):
        m = re.search(r"id=(c\w+)", row)
        a = re.search(r"<a [^>]*>(.*?)</a>", row, re.S)
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if not m or not a or len(tds) < 14:
            continue
        sid = m.group(1)
        lat, lon = coords.get(sid, (None, None))
        if not _in_bbox(lat, lon):
            continue
        v = [_num(re.sub(r"<[^>]+>", "", x)) for x in tds[1:13]]
        # v: today, intensity, type, month, year, 1h, 4h, 8h, 12h, 24h, 48h, 72h
        stamp = re.search(r'title="(\d{2}-\d{2}-\d{4} \d{2}:\d{2})"', row)
        out.append(_rain(sid, _fix_text(a.group(1)), lat, lon, "avamet",
                         _parse_local(stamp.group(1)) if stamp else None,
                         p1=v[5], p12=v[8], p24=v[9], p_4h=_r1(v[6]), p_8h=_r1(v[7]), p_today=_r1(v[0]),
                         intensity_mmh=v[1]))
    return out


# --------------------------------------------------------------------------- Meteoclimatic (restricted)
def _rain_meteoclimatic() -> list[dict]:
    raw = _get("https://www.meteoclimatic.net/feed/rss/ESPVA", "meteoclimatic_ESPVA.xml").decode("iso-8859-15")
    out = []
    for item in re.findall(r"<item>(.*?)</item>", raw, re.S):
        data = re.search(r"\[\[<(ESP\w+);\(([^)]*)\);\(([^)]*)\);\(([^)]*)\);\(([^)]*)\);\(([^)]*)\);(.*?)>\]\]", item)
        pt = re.search(r"<georss:point>\s*(-?[\d.]+)\s+(-?[\d.]+)", item)
        when = re.search(r"<pubDate>\w+, (\d+ \w+ \d{4} \d+:\d+:\d+) \+0000", item)
        if not data or not pt:
            continue
        lat, lon = float(pt.group(1)), float(pt.group(2))
        if not _in_bbox(lat, lon):
            continue
        t = None
        if when:
            try:
                t = _iso(dt.datetime.strptime(when.group(1), "%d %b %Y %H:%M:%S"))
            except ValueError:
                pass
        out.append(_rain(data.group(1), _fix_text(data.group(7)), lat, lon, "meteoclimatic", t,
                         p_today=_r1(_num(data.group(6)))))
    return out


# --------------------------------------------------------------------------- public API
_RAIN = {"saih_chj": _rain_saih_chj, "saih_segura": _rain_saih_segura, "saih_ebro": _rain_saih_ebro,
         "aemet": _rain_aemet, "avamet": _rain_avamet, "meteoclimatic": _rain_meteoclimatic}
_RIVER = {"saih_chj": _river_saih_chj, "saih_segura": _river_saih_segura, "saih_ebro": _river_saih_ebro}


def _restricted_enabled(flag: bool | None) -> bool:
    return bool(flag) if flag is not None else os.environ.get("RIUA_GAUGES_RESTRICTED", "") == "1"


def fetch_rain_gauges(sources: tuple[str, ...] | list[str] | None = None,
                      include_restricted: bool | None = None) -> list[dict]:
    """Latest accumulations of every reachable network. Never raises; see ``last_errors``."""
    if sources is None:
        sources = OPEN_RAIN_SOURCES + (RESTRICTED_RAIN_SOURCES if _restricted_enabled(include_restricted) else ())
    out: list[dict] = []
    for src in sources:
        fn = _RAIN.get(src)
        if fn is None:
            last_errors[f"rain:{src}"] = "unknown source"
            continue
        try:
            got = fn()
            out += got
            last_errors.pop(f"rain:{src}", None)
            log.info("rain %s: %d gauges", src, len(got))
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            last_errors[f"rain:{src}"] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("rain %s failed: %s", src, exc)
    return out


def fetch_river_gauges(sources: tuple[str, ...] | list[str] | None = None, with_trend: bool = True) -> list[dict]:
    """Latest level/flow of the river and rambla gauges. Never raises; see ``last_errors``."""
    out: list[dict] = []
    for src in sources or RIVER_SOURCES:
        fn = _RIVER.get(src)
        if fn is None:
            last_errors[f"river:{src}"] = "unknown source"
            continue
        try:
            got = fn(with_trend)
            out += got
            last_errors.pop(f"river:{src}", None)
            log.info("river %s: %d gauges", src, len(got))
        except Exception as exc:  # noqa: BLE001
            last_errors[f"river:{src}"] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("river %s failed: %s", src, exc)
    return out


# --------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import collections
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    restricted = "--restricted" in sys.argv or os.environ.get("RIUA_GAUGES_RESTRICTED") == "1"
    if "--dump" in sys.argv:
        os.environ["RIUA_RAW_DIR"] = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch", "r4-obs", "raw")

    t0 = time.time()
    rain = fetch_rain_gauges(include_restricted=restricted)
    print(f"\nRAIN GAUGES: {len(rain)} in {time.time() - t0:.1f} s  "
          f"{dict(collections.Counter(g['source'] for g in rain))}")
    for src in sorted({g["source"] for g in rain}):
        gs = [g for g in rain if g["source"] == src]
        stamps = sorted(g["t_utc"] for g in gs if g["t_utc"])
        nxy = sum(1 for g in gs if g["lat"] is not None)
        print(f"  {src:14s} n={len(gs):4d} with_xy={nxy:4d} newest={stamps[-1] if stamps else None}")

    def wet(g):
        return max((g.get(k) or 0.0) for k in ("p_24h", "p_12h", "p_1h", "p_today"))

    print("  5 wettest now (max of 24h / 12h / 1h / today):")
    for g in sorted(rain, key=wet, reverse=True)[:5]:
        print(f"    {g['source']:12s} {g['name'][:38]:38s} 1h={g['p_1h']} 3h={g['p_3h']} 6h={g['p_6h']} "
              f"12h={g['p_12h']} 24h={g['p_24h']} ({g['lat']},{g['lon']}) {g['t_utc']}")

    t0 = time.time()
    rivers = fetch_river_gauges()
    print(f"\nRIVER GAUGES: {len(rivers)} in {time.time() - t0:.1f} s  "
          f"{dict(collections.Counter(g['source'] for g in rivers))}")
    for g in sorted(rivers, key=lambda g: -(g["flow_m3s"] or 0.0))[:5]:
        print(f"    {g['source']:12s} {g['name'][:34]:34s} {g['river'][:22]:22s} Q={g['flow_m3s']} m3/s "
              f"H={g['level_m']} thr={g['thr_low']}/{g['thr_mid']}/{g['thr_high']} alert={g['alert']} trend={g['trend']}")
    poyo = [g for g in rivers if "POYO" in g["name"].upper()]
    if poyo:
        print("  Rambla del Poyo:", poyo[0])
    if last_errors:
        print("\nERRORS:", last_errors)
