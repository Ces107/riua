"""Real-time state of the reservoirs: volume, level, inflow, outflow and the recent filling rate.

Public API
----------
``fetch_reservoirs(dams, state=None) -> (list[dict], report)``
    ``dams`` = the entries of geo/hydro/dams/dams.json (needs ``id``, ``saih``, ``vars``). One dict per reservoir
    that answered: id, source, t_utc, volume_hm3, level_m, pct_saih (SAIH Segura: % of capacity as published),
    inflow_m3s, outflow_m3s (everything that leaves the reservoir), outflow_river_m3s (what goes down the river),
    thr_low / thr_mid / thr_high (SAIH Júcar thresholds of the outflow to the river, m3/s), nmn_hm3 and
    spill_level_m as SAIH Júcar publishes them, series = {"t": [UTC ISO hours], "v": [hm3]} (last hours, hourly),
    rate_hm3h (mean change of volume over the last 3 h; None when unknown).
    ``report`` is the row for the snapshot's ``sources``.

Sources (every endpoint exercised from a normal PC on 2026-10-02, no key):

=============  =========================================================================================
saih_chj       https://saih.chj.es/mapa-embalses : ``let embalses = [...]`` embedded in the HTML, 25 reservoirs
               with cota, volumen, caudal recibido, caudal de salida, caudal de salida al río, volumen at the
               maximum normal level and cota de vertido. 5-min updates, provisional data. Recent series:
               /admin/variables/valor/{idVolumenEmbalse}/{from}/{to} (local wall time in, UTC out).
saih_segura    POST https://saihweb.chsegura.es/apps/iVisor/obtener_datos.php  action=consultar_embalses :
               27 reservoirs with level (m above the gauge zero, NOT above sea level), volume and % of
               capacity. No inflow or outflow. Recent series: graficas/graficaVar.php?puntos=<variable>.
saih_ebro      https://www.saihebro.com/api/mapa/getDatosMapa?slug=mapa-embalses-H9-guadalope-martin answers
               (Calanda, Santolea, Gallipuén...), but no Valencian town lies below an Ebro reservoir: unused.
=============  =========================================================================================

Design: both networks are isolated (one failing never hides the other), short timeouts, at most ``WORKERS``
requests at a time, nothing written unless ``state`` is given: then ``state/reservoirs.json`` keeps the last
48 h of volumes per reservoir, which is where the Segura filling rates come from after the first cycles.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from . import gauges as G

log = logging.getLogger("riua.reservoirs")

WORKERS = 3
SERIES_H = 12            # hours of history asked for
RATE_H = 3               # the filling rate is the mean over this many hours
KEEP_H = 48              # hours kept in state/reservoirs.json
SEG_CHART = "https://saihweb.chsegura.es/apps/iVisor/graficas/graficaVar.php"
last_errors: dict[str, str] = {}


def _utc_to_madrid(t: dt.datetime) -> dt.datetime:
    a = G._last_sunday(t.year, 3) + dt.timedelta(hours=1)
    b = G._last_sunday(t.year, 10) + dt.timedelta(hours=1)
    return t + dt.timedelta(hours=2 if a <= t < b else 1)


def _hourly(pairs: list[tuple[dt.datetime, float]]) -> tuple[list[str], list[float]]:
    """(UTC time, value) samples -> the last value of every clock hour, as ISO hour ends."""
    by: dict[dt.datetime, tuple[dt.datetime, float]] = {}
    for t, v in pairs:
        h = (t + dt.timedelta(minutes=59, seconds=59)).replace(minute=0, second=0, microsecond=0)
        if h not in by or t >= by[h][0]:
            by[h] = (t, v)
    hs = sorted(by)
    return [h.strftime("%Y-%m-%dT%H:%MZ") for h in hs], [round(by[h][1], 4) for h in hs]


def _rate(pairs: list[tuple[dt.datetime, float]]) -> float | None:
    """hm3 per hour over the last RATE_H hours (needs at least 1 h of samples)."""
    if len(pairs) < 2:
        return None
    pairs = sorted(pairs)
    t1, v1 = pairs[-1]
    old = [(t, v) for t, v in pairs if t <= t1 - dt.timedelta(hours=RATE_H)]
    t0, v0 = old[-1] if old else pairs[0]
    h = (t1 - t0).total_seconds() / 3600.0
    return None if h < 1.0 else round((v1 - v0) / h, 5)


# ------------------------------------------------------------------------------------ SAIH Júcar
def _chj_series(var: str, now: dt.datetime) -> list[tuple[dt.datetime, float]]:
    a = _utc_to_madrid(now - dt.timedelta(hours=SERIES_H)).strftime("%Y-%m-%d %H:%M:%S")
    b = _utc_to_madrid(now + dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    raw = G._get(f"{G.CHJ}/admin/variables/valor/{var}/{requests.utils.quote(a)}/{requests.utils.quote(b)}", f"chj_val_{var}.json")
    out = []
    for p in json.loads(raw.decode("utf-8")):
        if p.get("valor") is not None and p.get("fecha"):
            out.append((dt.datetime.strptime(p["fecha"][:19], "%Y-%m-%dT%H:%M:%S"), float(p["valor"])))
    return out


def _fetch_chj(dams: list[dict], now: dt.datetime, with_series: bool) -> list[dict]:
    by_code = {d["saih"].split(":", 1)[1]: d for d in dams if d.get("saih", "").startswith("chj:")}
    out = []
    for s in G._chj_embedded("mapa-embalses", "embalses"):
        d = by_code.get(s.get("fldTCodigo"))
        if d is None:
            continue
        vol = G._num(s.get("valorVolumenEmbalse"))
        if vol is None:
            continue
        out.append({
            "id": d["id"], "source": "saih_chj",
            "t_utc": G._iso_utc(s.get("fechaComunicacionVol") or s.get("fechaComunicacionVolumenEmbalse")),
            "volume_hm3": round(vol, 4), "level_m": G._num(s.get("valorCotaEmbalse")), "pct_saih": None,
            "inflow_m3s": G._num(s.get("valorCaudalRecibido")), "outflow_m3s": G._num(s.get("valorCaudalSalida")),
            "outflow_river_m3s": G._num(s.get("valorCaudalSalidaRio")),
            "thr_low": G._num(s.get("umbralBajoCaudalSalidaRio")), "thr_mid": G._num(s.get("umbralMedioCaudalSalidaRio")),
            "thr_high": G._num(s.get("umbralAltoCaudalSalidaRio")),
            "nmn_hm3": G._num(s.get("fldFVolumenNMN")), "spill_level_m": G._num(s.get("fldFCotaVertido")),
            "sensor_state": s.get("estadoVolumenEmbalse"), "series": None, "rate_hm3h": None,
            "_var": s.get("idVolumenEmbalse"),
        })
    if with_series and out:
        def one(r):
            try:
                return _chj_series(str(r["_var"]), now)
            except Exception as exc:  # noqa: BLE001
                log.debug("chj series %s failed: %s", r["id"], exc)
                return []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for r, pairs in zip(out, ex.map(one, out)):
                if pairs:
                    t, v = _hourly(pairs)
                    r["series"], r["rate_hm3h"] = {"t": t, "v": v}, _rate(pairs)
    for r in out:
        r.pop("_var", None)
    return out


# ------------------------------------------------------------------------------------ SAIH Segura
def _seg_series(var: str, now: dt.datetime) -> list[tuple[dt.datetime, float]]:
    a, b = _utc_to_madrid(now - dt.timedelta(hours=SERIES_H)), _utc_to_madrid(now + dt.timedelta(hours=1))
    url = (f"{SEG_CHART}?puntos={var}&dfrom={requests.utils.quote(a.strftime('%d/%m/%Y %H:%M'))}"
           f"&dto={requests.utils.quote(b.strftime('%d/%m/%Y %H:%M'))}&source=I&VerNoFiables=S")
    text = G._get(url, f"segura_chart_{var}.html").decode("utf-8", "replace")
    ser = text[text.find("series: ["):text.find("var ObjChart")]
    out = []
    for x, y in re.findall(r"\{x:(\d+), y:([-\d.eE]+|null),", ser):
        if y != "null":       # x = Madrid wall time written as if it were UTC epoch ms
            out.append((G.madrid_to_utc(dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=int(x))), float(y)))
    return out


def _fetch_segura(dams: list[dict], now: dt.datetime, with_series: bool) -> list[dict]:
    by_code = {d["saih"].split(":", 1)[1]: d for d in dams if d.get("saih", "").startswith("seg:")}
    data = json.loads(G._post(G.SEG_IVISOR, "segura_embalses.json", "action=consultar_embalses"))
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for it in data:
        d = by_code.get(it.get("CodPuntoMedicion"))
        vol = G._num(it.get("UltimoDatoVolumen"))
        if d is None or vol is None:
            continue
        out.append({
            "id": d["id"], "source": "saih_segura", "t_utc": stamp, "t_is_fetch_time": True,
            "volume_hm3": round(vol, 4), "level_m": None, "gauge_m": G._num(it.get("UltimoDatoNivel")),
            "pct_saih": G._num(it.get("PorcentajeCapacidad")),
            "inflow_m3s": None, "outflow_m3s": None, "outflow_river_m3s": None,
            "thr_low": None, "thr_mid": None, "thr_high": None, "nmn_hm3": None, "spill_level_m": None,
            "series": None, "rate_hm3h": None, "_var": it.get("CodVariableHidrologicaVolumen"),
        })
    if with_series and out:
        def one(r):
            try:
                return _seg_series(str(r["_var"]), now)
            except Exception as exc:  # noqa: BLE001
                log.debug("segura series %s failed: %s", r["id"], exc)
                return []
        with ThreadPoolExecutor(max_workers=2) as ex:
            for r, pairs in zip(out, ex.map(one, out)):
                if pairs:
                    t, v = _hourly(pairs)
                    r["series"], r["rate_hm3h"] = {"t": t, "v": v}, _rate(pairs)
    for r in out:
        r.pop("_var", None)
    return out


# ------------------------------------------------------------------------------------ rolling history
def _merge_history(rows: list[dict], state: Path, now: dt.datetime) -> None:
    """Keep the last KEEP_H hours of (time, volume) per reservoir; fill series / rate of the rows that have none."""
    f = state / "reservoirs.json"
    hist: dict[str, list] = {}
    if f.exists():
        try:
            hist = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            hist = {}
    lo = (now - dt.timedelta(hours=KEEP_H)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in rows:
        h = [x for x in hist.get(r["id"], []) if x[0] >= lo]
        if r.get("series"):
            known = {x[0][:13] for x in h}
            h += [[t[:16] + ":00Z", v] for t, v in zip(r["series"]["t"], r["series"]["v"]) if t[:13] not in known and t[:16] + ":00Z" < stamp]
        h.append([stamp, r["volume_hm3"]])
        h.sort()
        hist[r["id"]] = h
        if not r.get("series") and len(h) >= 2:
            pairs = [(dt.datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ"), v) for a, v in h]
            pairs = [p for p in pairs if p[0] >= now - dt.timedelta(hours=SERIES_H)]
            t, v = _hourly(pairs)
            r["series"], r["rate_hm3h"] = {"t": t, "v": v}, _rate(pairs)
    try:
        state.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(hist, separators=(",", ":")), encoding="utf-8")
    except OSError as exc:
        log.debug("reservoir history not saved: %s", exc)


def fetch_reservoirs(dams: list[dict], state: Path | None = None, now: dt.datetime | None = None,
                     with_series: bool = True, segura_series: bool | None = None) -> tuple[list[dict], dict]:
    """Latest state of every reservoir of ``dams`` that its SAIH publishes. Never raises.

    The Segura series cost one chart page per reservoir: by default they are only asked for while the state
    history is still empty (first cycle); afterwards the history kept in ``state`` gives the rates."""
    now = (now or dt.datetime.utcnow()).replace(tzinfo=None, microsecond=0)
    t0 = time.time()
    if segura_series is None:
        segura_series = with_series and not (state is not None and (state / "reservoirs.json").exists())
    rows: list[dict] = []
    nets = []
    for name, fn, ws in (("saih_chj", _fetch_chj, with_series), ("saih_segura", _fetch_segura, segura_series)):
        try:
            got = fn(dams, now, ws)
            rows += got
            nets.append(name)
            last_errors.pop(name, None)
            log.info("reservoirs %s: %d", name, len(got))
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            last_errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("reservoirs %s failed: %s", name, exc)
    if state is not None and rows:
        _merge_history(rows, Path(state), now)
    report = {"id": "reservoirs", "label": "Embalses SAIH (Júcar, Segura)", "ok": bool(rows), "n": len(rows),
              "sources": nets, "errors": dict(last_errors) or None, "seconds": round(time.time() - t0, 1)}
    return rows, report


# ------------------------------------------------------------------------------------ smoke run
if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    here = Path(__file__).resolve().parents[3] / "geo" / "hydro" / "dams"
    f = here / "dams.json" if (here / "dams.json").exists() else here / "dams_sites.json"
    rows, rep = fetch_reservoirs(json.loads(f.read_text(encoding="utf-8"))["dams"],
                                 state=Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(rep)
    for r in rows:
        n = len(r["series"]["t"]) if r.get("series") else 0
        print(f"{r['id']:15s} {r['source']:11s} V={r['volume_hm3']:9.3f} hm3  level={r['level_m']}  in={r['inflow_m3s']}  "
              f"out={r['outflow_m3s']}  river={r['outflow_river_m3s']}  rate={r['rate_hm3h']} hm3/h  series={n} h  {r['t_utc']}")
