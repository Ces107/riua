"""Official AEMET Meteoalerta warnings for the Comunitat Valenciana, without API key.

``fetch_aemet_warnings() -> list[dict]`` with keys

    zone_code     AEMET Meteoalerta zone, e.g. '774602' (7703xx Alicante, 7712xx Castellón,
                  7746xx Valencia; a trailing 'C' marks coastal-water zones)
    zone_name     'Litoral norte de Valencia'
    phenomenon    'rain' | 'thunderstorm' | 'wind' | 'snow' | 'coastal' | 'max_temp' | ...
    level         'yellow' | 'orange' | 'red'   (green = "no warning" is dropped by default)
    onset, expires  ISO-8601 UTC ('2026-10-01T12:00:00Z')
    text          AEMET description in Spanish ('Precipitación acumulada en una hora: 50 mm.')
    params        {'code': 'P1', 'label': 'Precipitación acumulada en una hora',
                   'value': 50.0, 'unit': 'mm', 'accum_hours': 1, 'probability': '40%-70%'}
    plus: level_num (1..3), phenomenon_code ('PR'), sent, msg_type, identifier, headline,
    certainty, feed ('aemet_cap' | 'meteoalarm') and, with include_polygons=True, polygon
    [[lat, lon], ...] (only in the AEMET feed).

Feeds (both verified keyless from a normal PC on 2026-10-01):

1. AEMET CAP bundle (primary). RSS index
   https://www.aemet.es/documentos_d/eltiempo/prediccion/avisos/rss/CAP_AFAC77_RSS.xml
   -> first <item> links a tar.gz (~20 KB) with one CAP 1.2 XML per zone/parameter/day,
   including zone polygons, the threshold value and the probability band.
2. MeteoAlarm (fallback) https://feeds.meteoalarm.org/api/v1/warnings/feeds-spain
   (~2.3 MB JSON for the whole of Spain; no polygons, no probability; the AEMET zone code
   is recovered from the CAP identifier).

Attribution required: "Fuente: AEMET" (AEMET legal note: use and reproduction allowed citing
AEMET as author). For the fallback also credit "MeteoAlarm / EUMETNET" and show the retrieval
time; MeteoAlarm's own terms page is JavaScript-rendered and could not be read by script
(UNVERIFIED), so read https://meteoalarm.org terms before relying on that feed in production.
Never raises: returns [] and records the reason in ``last_errors``.

Dependencies: requests + stdlib.
"""
from __future__ import annotations

import os
import sys

# NAMING HAZARD ---------------------------------------------------------------------------
# This file is called warnings.py, like the stdlib module. Imported as
# ``riua.sources.warnings`` that is harmless. But whenever this DIRECTORY is sys.path[0]
# (any sibling file run as a script, e.g. ``python backend/riua/sources/gauges.py``) the
# stdlib's own ``import warnings`` (done by logging, re, ...) finds THIS file first.
# In that case we turn ourselves into the real stdlib module by executing its source in our
# namespace, and skip the Riuà code entirely (that is why the whole module body lives in
# the ``else`` branch). If the lead ever renames this module (e.g. aemet_warnings.py) the
# shim and the extra indentation can go.
if __name__ == "warnings":
    _std = os.path.join(os.path.dirname(os.__file__), "warnings.py")
    with open(_std, "rb") as _fh:
        _src = _fh.read()
    __file__ = _std
    __doc__ = None
    exec(compile(_src, _std, "exec", dont_inherit=True), globals())
else:
    import datetime as dt
    import io
    import logging
    import re
    import tarfile
    import xml.etree.ElementTree as ET

    import requests

    log = logging.getLogger("riua.warnings")

    AEMET_RSS = "https://www.aemet.es/documentos_d/eltiempo/prediccion/avisos/rss/CAP_AFAC77_RSS.xml"
    METEOALARM = "https://feeds.meteoalarm.org/api/v1/warnings/feeds-spain"
    TIMEOUT = (8, 40)
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/126 Safari/537.36 riua/0.1 (non-commercial flood-risk site)")

    CV_ZONE_PREFIXES = ("7703", "7712", "7746")
    # MeteoAlarm EMMA_ID of the CV zones (land ES239-ES249, coastal waters ES861-ES866)
    CV_EMMA = {f"ES{n}" for n in list(range(239, 250)) + list(range(861, 867))}

    LEVELS = {"amarillo": ("yellow", 1), "naranja": ("orange", 2), "rojo": ("red", 3), "verde": ("green", 0),
              "yellow": ("yellow", 1), "orange": ("orange", 2), "red": ("red", 3), "green": ("green", 0)}
    PHENOMENA = {"PR": "rain", "TO": "thunderstorm", "VI": "wind", "NE": "snow", "CO": "coastal",
                 "TA": "max_temp", "TM": "min_temp", "NI": "fog", "AL": "avalanche", "PO": "dust",
                 "DH": "thaw", "OC": "heat_wave", "OF": "cold_wave", "GA": "galerna", "RI": "rissaga"}
    # MeteoAlarm awareness_type number -> phenomenon (used only if the identifier cannot be parsed)
    AWARENESS = {"1": "wind", "2": "snow", "3": "thunderstorm", "4": "fog", "5": "max_temp", "6": "min_temp",
                 "7": "coastal", "8": "forest_fire", "9": "avalanche", "10": "rain", "12": "flood", "13": "rain_flood"}

    _NS = {"c": "urn:oasis:names:tc:emergency:cap:1.2"}
    _ID_RE = re.compile(r"\.ES\.\d+\.(\d{6}C?)([A-Z]{2})([A-Z0-9]{2})")
    _session = requests.Session()
    _session.headers["User-Agent"] = UA
    last_errors: dict[str, str] = {}

    # ----------------------------------------------------------------------- helpers
    def _to_utc(s: str | None) -> str | None:
        """'2026-10-01T14:00:00+02:00' (or '...-00:00', '...Z') -> '2026-10-01T12:00:00Z'."""
        if not s:
            return None
        s = s.strip().replace("Z", "+00:00")
        try:
            t = dt.datetime.fromisoformat(s)
        except ValueError:
            return None
        if t.tzinfo is not None:
            t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _now() -> str:
        return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _params_from(code: str | None, label: str | None, value_txt: str | None, probability: str | None) -> dict:
        p: dict = {}
        if code:
            p["code"] = code
        if label:
            p["label"] = label
            low = label.lower()
            if "una hora" in low or "1 hora" in low:
                p["accum_hours"] = 1
            else:
                m = re.search(r"(\d+)\s*horas", low)
                if m:
                    p["accum_hours"] = int(m.group(1))
        if value_txt:
            m = re.match(r"\s*(-?\d+(?:[.,]\d+)?)\s*(.*)", value_txt)
            if m:
                p["value"] = float(m.group(1).replace(",", "."))
                p["unit"] = m.group(2).strip()
            else:
                p["value_text"] = value_txt.strip()
        if probability:
            p["probability"] = probability
        return p

    def _params_from_text(text: str) -> dict:
        """'Precipitación acumulada en 12 horas: 80 mm. Sobre todo ...' -> params (MeteoAlarm path)."""
        m = re.match(r"\s*([^:]+):\s*(-?\d+(?:[.,]\d+)?)\s*([^\s.]+(?:/h)?)", text or "")
        if not m:
            return {}
        return _params_from(None, m.group(1).strip(), f"{m.group(2)} {m.group(3)}", None)

    # ----------------------------------------------------------------------- AEMET CAP (primary)
    def _txt(node, path: str) -> str | None:
        el = node.find(path, _NS)
        return el.text.strip() if el is not None and el.text else None

    def _parse_cap(xml_bytes: bytes, include_polygons: bool) -> list[dict]:
        root = ET.fromstring(xml_bytes)
        ident = _txt(root, "c:identifier") or ""
        sent = _to_utc(_txt(root, "c:sent"))
        msg_type = _txt(root, "c:msgType")
        out = []
        for info in root.findall("c:info", _NS):
            if not (_txt(info, "c:language") or "es").lower().startswith("es"):
                continue
            params = {(_txt(p, "c:valueName") or ""): (_txt(p, "c:value") or "")
                      for p in info.findall("c:parameter", _NS)}
            level_txt = params.get("AEMET-Meteoalerta nivel", "").lower()
            level, level_num = LEVELS.get(level_txt, (level_txt or None, None))
            code = label = value = None
            if params.get("AEMET-Meteoalerta parametro"):
                parts = params["AEMET-Meteoalerta parametro"].split(";")
                code = parts[0] if parts else None
                label = parts[1] if len(parts) > 1 else None
                value = parts[2] if len(parts) > 2 else None
            phen_code = None
            ev = info.find("c:eventCode", _NS)
            if ev is not None:
                phen_code = (_txt(ev, "c:value") or "").split(";")[0] or None
            if phen_code is None:
                m = _ID_RE.search(ident)
                phen_code = m.group(2) if m else None
            base = {
                "phenomenon": PHENOMENA.get(phen_code or "", (phen_code or "").lower() or None),
                "phenomenon_code": phen_code, "level": level, "level_num": level_num,
                "onset": _to_utc(_txt(info, "c:onset")), "expires": _to_utc(_txt(info, "c:expires")),
                "text": _txt(info, "c:description") or "", "headline": _txt(info, "c:headline") or "",
                "params": _params_from(code, label, value, params.get("AEMET-Meteoalerta probabilidad") or None),
                "certainty": _txt(info, "c:certainty"), "sent": sent, "msg_type": msg_type,
                "identifier": ident, "feed": "aemet_cap",
            }
            for area in info.findall("c:area", _NS):
                zone = None
                for gc in area.findall("c:geocode", _NS):
                    if "zona" in (_txt(gc, "c:valueName") or "").lower():
                        zone = _txt(gc, "c:value")
                w = dict(base)
                w["zone_code"] = zone
                w["zone_name"] = _txt(area, "c:areaDesc") or ""
                if include_polygons:
                    poly = _txt(area, "c:polygon")
                    w["polygon"] = [[float(a), float(b)] for a, b in
                                    (pt.split(",") for pt in poly.split())] if poly else None
                out.append(w)
        return out

    def _fetch_aemet_cap(include_polygons: bool) -> list[dict]:
        rss = _session.get(AEMET_RSS, timeout=TIMEOUT)
        rss.raise_for_status()
        m = re.search(r"<link>\s*(https://[^<\s]+\.tar\.gz)\s*</link>", rss.text)
        if not m:
            raise ValueError("aemet rss: tar.gz link not found")
        blob = _session.get(m.group(1), timeout=TIMEOUT)
        blob.raise_for_status()
        out: list[dict] = []
        with tarfile.open(fileobj=io.BytesIO(blob.content), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".xml"):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                try:
                    out += _parse_cap(fh.read(), include_polygons)
                except ET.ParseError as exc:
                    log.debug("bad CAP %s: %s", member.name, exc)
        return out

    # ----------------------------------------------------------------------- MeteoAlarm (fallback)
    def _fetch_meteoalarm() -> list[dict]:
        r = _session.get(METEOALARM, timeout=TIMEOUT)
        r.raise_for_status()
        out = []
        for w in r.json().get("warnings", []):
            alert = w.get("alert", {})
            ident = alert.get("identifier", "")
            m = _ID_RE.search(ident)
            zone = m.group(1) if m else None
            for info in alert.get("info", []):
                if not info.get("language", "").lower().startswith("es"):
                    continue
                par = {p.get("valueName"): p.get("value", "") for p in info.get("parameter", [])}
                lv = (par.get("awareness_level", "").split(";") + ["", ""])[1].strip().lower()
                level, level_num = LEVELS.get(lv, (lv or None, None))
                atype = par.get("awareness_type", "").split(";")[0].strip()
                phen_code = m.group(2) if m else None
                for area in info.get("area", []):
                    emma = next((g.get("value") for g in area.get("geocode", [])
                                 if g.get("valueName") == "EMMA_ID"), None)
                    if not ((zone and zone.startswith(CV_ZONE_PREFIXES)) or emma in CV_EMMA):
                        continue
                    params = _params_from_text(info.get("description", ""))
                    if m:
                        params.setdefault("code", m.group(3))
                    out.append({
                        "zone_code": zone, "zone_name": area.get("areaDesc", ""),
                        "phenomenon": PHENOMENA.get(phen_code or "", AWARENESS.get(atype)),
                        "phenomenon_code": phen_code, "level": level, "level_num": level_num,
                        "onset": _to_utc(info.get("onset")), "expires": _to_utc(info.get("expires")),
                        "text": info.get("description", ""), "headline": info.get("headline", ""),
                        "params": params, "certainty": info.get("certainty"), "sent": _to_utc(alert.get("sent")),
                        "msg_type": alert.get("msgType"), "identifier": ident, "feed": "meteoalarm",
                        "emma_id": emma,
                    })
        return out

    # ----------------------------------------------------------------------- public API
    def _clean(ws: list[dict], include_green: bool, include_expired: bool, phenomena) -> list[dict]:
        now = _now()
        latest: dict[tuple, dict] = {}
        for w in ws:
            zone = w.get("zone_code") or ""
            if not zone.startswith(CV_ZONE_PREFIXES):
                continue
            if (w.get("msg_type") or "").lower() == "cancel":
                continue
            if not include_green and (w.get("level_num") in (0, None)):
                continue
            if not include_expired and w.get("expires") and w["expires"] < now:
                continue
            if phenomena and w.get("phenomenon") not in phenomena and w.get("phenomenon_code") not in phenomena:
                continue
            key = (zone, w.get("phenomenon_code"), w.get("params", {}).get("code"), w.get("onset"))
            old = latest.get(key)
            if old is None or (w.get("sent") or "") >= (old.get("sent") or ""):
                latest[key] = w
        return sorted(latest.values(),
                      key=lambda w: (w.get("onset") or "", w["zone_code"], -(w.get("level_num") or 0)))

    def fetch_aemet_warnings(include_green: bool = False, include_expired: bool = False,
                             phenomena: tuple[str, ...] | None = None, include_polygons: bool = False,
                             feed: str = "auto") -> list[dict]:
        """Warnings in force or announced for the CV. ``phenomena=('rain',)`` keeps only rain.

        feed: 'auto' (AEMET CAP, MeteoAlarm on failure) | 'aemet_cap' | 'meteoalarm'.
        """
        order = {"auto": ("aemet_cap", "meteoalarm"), "aemet_cap": ("aemet_cap",),
                 "meteoalarm": ("meteoalarm",)}[feed]
        for name in order:
            try:
                raw = _fetch_aemet_cap(include_polygons) if name == "aemet_cap" else _fetch_meteoalarm()
                last_errors.pop(name, None)
                return _clean(raw, include_green, include_expired, phenomena)
            except Exception as exc:  # noqa: BLE001 - never break the caller
                last_errors[name] = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("warnings feed %s failed: %s", name, exc)
        return []

    def fetch_rain_warnings(**kw) -> list[dict]:
        """Shortcut: only accumulated-rain warnings (1 h and 12 h thresholds)."""
        return fetch_aemet_warnings(phenomena=("rain",), **kw)

    def max_rain_level_by_zone(warnings: list[dict] | None = None) -> dict[str, dict]:
        """zone_code -> {'level_num', 'level', 'p1h_mm', 'p12h_mm'} for the rain warnings given."""
        ws = fetch_rain_warnings() if warnings is None else [w for w in warnings if w.get("phenomenon") == "rain"]
        out: dict[str, dict] = {}
        for w in ws:
            z = out.setdefault(w["zone_code"], {"zone_name": w["zone_name"], "level_num": 0, "level": None,
                                                "p1h_mm": None, "p12h_mm": None})
            if (w.get("level_num") or 0) > z["level_num"]:
                z["level_num"], z["level"] = w["level_num"], w["level"]
            hours, val = w["params"].get("accum_hours"), w["params"].get("value")
            key = {1: "p1h_mm", 12: "p12h_mm"}.get(hours)
            if key and val is not None:
                z[key] = max(z[key] or 0.0, val)
        return out

    # ----------------------------------------------------------------------- smoke test
    if __name__ == "__main__":
        import collections
        import time

        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        for _feed in ("aemet_cap", "meteoalarm"):
            _t0 = time.time()
            _ws = fetch_aemet_warnings(feed=_feed)
            print(f"\n[{_feed}] {len(_ws)} active/announced warnings for the CV in {time.time() - _t0:.1f} s "
                  f"(now {_now()})")
            print("  by phenomenon/level:",
                  dict(collections.Counter((w["phenomenon"], w["level"]) for w in _ws)))
            for _w in [w for w in _ws if w["phenomenon"] == "rain"][:12]:
                print(f"   {_w['zone_code']} {_w['zone_name'][:28]:28s} {_w['level']:6s} {_w['onset']} -> "
                      f"{_w['expires']} {_w['params']} | {_w['text'][:60]}")
        print("\nmax rain level by zone:", max_rain_level_by_zone())
        if last_errors:
            print("ERRORS:", last_errors)
