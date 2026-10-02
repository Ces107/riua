"""SAIH Júcar history: 5-min series of any variable (flow gauges, dam outflows, rain intensity).

Public chart endpoint (no key):  https://saih.chj.es/admin/variables/valor/{id_variable}/{from}/{to}
  * `from` / `to` are local wall time (Europe/Madrid) "YYYY-MM-DD HH:MM:SS"; `fecha` in the answer is UTC.
  * history starts on 2024-09-02 05:45 UTC for every variable that existed then (verified 2026-10-01).
  * the answer is NOT thinned (a whole year = 99,937 values in 29 s) but long windows get very slow when
    several run at once: ask 15 days at a time (~1 s each).
Rain: every pluviometer has a 5-min intensity variable (mm/h; rain = sum / 12), its id is in the page
/chart-lluvia/{idEstacionRemota} (`let varLluvia = [...]`).

Cache (gitignored): hindcast/obs/flows/chj_{var}.npz  -> t (datetime64[m], UTC), v (float32), estado (uint8)
                    hindcast/obs/flows/chj_rain_vars.json, hindcast/obs/flows/rain_hourly.npz

    py -3.11 geo/hydro/gauges/saih_series.py flows      # every gauge + dam outflow of gauges_decisions.json
    py -3.11 geo/hydro/gauges/saih_series.py rain       # hourly rain of every SAIH pluviometer
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from riua.sources import gauges as G                      # noqa: E402

FLOWS = ROOT / "hindcast" / "obs" / "flows"
T0 = datetime(2024, 9, 1)
CHUNK_D = 15          # 15 days answer in ~1 s; 60-day chunks took minutes when several ran at once
WORKERS = 4


def _get_json(url, tries=3):
    for k in range(tries):
        try:
            r = G._session.get(url, timeout=(10, 60))
            r.raise_for_status()
            return r.json()
        except Exception as e:                            # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(5 + 10 * k)


def fetch(var: str, a: datetime, b: datetime):
    qa, qb = (requests.utils.quote(x.strftime("%Y-%m-%d %H:%M:%S")) for x in (a, b))
    d = _get_json(f"{G.CHJ}/admin/variables/valor/{var}/{qa}/{qb}")
    d = [p for p in d if p.get("valor") is not None]
    t = np.array([p["fecha"][:16] for p in d], dtype="datetime64[m]")
    return t, np.array([p["valor"] for p in d], np.float32), np.array([p.get("estado") or 0 for p in d], np.uint8)


def series(var: str, refresh_tail: bool = False):
    """Whole history of one variable: t (UTC, datetime64[m]), v, estado. Cached; the cache is extended to now on demand."""
    f = FLOWS / f"chj_{var}.npz"
    FLOWS.mkdir(parents=True, exist_ok=True)
    parts, start = [], T0
    if f.exists():
        z = np.load(f)
        if not refresh_tail:
            return z["t"], z["v"], z["estado"]
        parts.append((z["t"], z["v"], z["estado"]))
        start = datetime.fromisoformat(str(z["t_done"]))
    end = datetime.utcnow()
    t = start
    while t < end:
        u = min(t + timedelta(days=CHUNK_D), end)
        parts.append(fetch(var, t, u))
        t = u
        time.sleep(0.25)
    tt = np.concatenate([p[0] for p in parts]); v = np.concatenate([p[1] for p in parts]); e = np.concatenate([p[2] for p in parts])
    tt, i = np.unique(tt, return_index=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=tt, v=v[i], estado=e[i], t_done=np.array((end - timedelta(hours=6)).isoformat()))
    tmp.replace(f)
    return tt, v[i], e[i]


# ---- SAIH Segura: chart page of the public iVisor, 5-min, history back to 2019 and earlier ----------------------
SEG = "https://saihweb.chsegura.es/apps/iVisor"


def _utc_to_madrid(t: datetime) -> datetime:
    a = G._last_sunday(t.year, 3) + timedelta(hours=1)
    b = G._last_sunday(t.year, 10) + timedelta(hours=1)
    return t + timedelta(hours=2 if a <= t < b else 1)


def segura_series(point: str):
    """5-min discharge of a SAIH Segura gauge since T0 (point code such as '01O03A1'): t (UTC), v, estado (zeros).
    GET {SEG}/graficas/graficaVar.php?puntos=<variable>&dfrom=dd/mm/yyyy HH:MM&dto=...&source=I ; at most 7 days per request;
    x = Madrid wall time written as if it were UTC epoch ms."""
    f = FLOWS / f"seg_{point}.npz"
    if f.exists():
        z = np.load(f)
        return z["t"], z["v"], z["estado"]
    data = json.loads(G._post(f"{SEG}/obtener_datos.php", "segura_cauces.json", "action=consultar_cauces_topo"))
    var = next(it["CodVariableHidrologicaCaudal"] for it in data if it.get("CodPuntoMedicion") == point)
    tt, vv = [], []
    a, end = _utc_to_madrid(T0), _utc_to_madrid(datetime.utcnow())
    while a < end:
        b = min(a + timedelta(days=7), end)
        url = (f"{SEG}/graficas/graficaVar.php?puntos={var}&dfrom={requests.utils.quote(a.strftime('%d/%m/%Y %H:%M'))}"
               f"&dto={requests.utils.quote(b.strftime('%d/%m/%Y %H:%M'))}&source=I&VerNoFiables=S")
        r = G._session.get(url, timeout=(10, 60)); r.raise_for_status()
        text = r.text
        ser = text[text.find("series: ["):text.find("var ObjChart")]
        for x, y in re.findall(r"\{x:(\d+), y:([-\d.eE]+|null),", ser):
            if y != "null":
                tt.append(G.madrid_to_utc(datetime(1970, 1, 1) + timedelta(milliseconds=int(x)))); vv.append(float(y))
        a = b
        time.sleep(0.7)
    t = np.array(tt, dtype="datetime64[m]"); v = np.array(vv, np.float32)
    t, i = np.unique(t, return_index=True)
    FLOWS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, t=t, v=v[i], estado=np.zeros(len(t), np.uint8))
    return t, v[i], np.zeros(len(t), np.uint8)


def hourly(t, v, how="max"):
    """5-min series -> hourly (t_end datetime64[h], value, n samples); how = max | mean | sum12 (mm from mm/h)."""
    if len(t) == 0:
        return np.array([], "datetime64[h]"), np.array([]), np.array([], int)
    hh = (t + np.timedelta64(59, "m")).astype("datetime64[h]")       # hour ending
    u, inv, n = np.unique(hh, return_inverse=True, return_counts=True)
    if how == "max":
        out = np.full(len(u), -np.inf); np.maximum.at(out, inv, v)
    else:
        out = np.zeros(len(u)); np.add.at(out, inv, v)
        out = out / n if how == "mean" else out / 12.0
    return u, out, n


def flow_vars():
    dec = json.loads((ROOT / "geo" / "hydro" / "gauges" / "gauges_decisions.json").read_text(encoding="utf-8"))["gauges"]
    out = set()
    for k, d in dec.items():
        if d.get("use") and k.isdigit():                  # SAIH Júcar variable ids are numbers; Segura points go through segura_series
            out.add(k)
            out.update((d.get("release") or {}).values())
    out.update(("16915", "2697", "14099", "13337", "14450", "12905", "13080", "16699", "16687", "16688", "16682", "16684"))
    return sorted(out)


def rain_stations():
    """[{id, name, lat, lon, var}] of the SAIH pluviometers (the variable id needs one page per station; cached)."""
    f = FLOWS / "chj_rain_vars.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    out = []
    for s in G._chj_embedded("mapa-lluvias", "estaciones"):
        lon, lat = G._chj_lonlat(s)
        try:
            raw = G._get(f"{G.CHJ}/chart-lluvia/{s['idEstacionRemota']}", "x").decode("utf-8", "replace")
            m = re.search(r"let varLluvia = (\[.*?\]);", raw, re.S)
            var = json.loads(m.group(1))[0]["idVariable"]
        except Exception as e:                            # noqa: BLE001
            print("  no rain variable for", s.get("fldTNombre"), type(e).__name__)
            continue
        out.append(dict(id=str(s["idEstacionRemota"]), code=s.get("fldTCodigo"), name=s.get("fldTNombre"), lat=lat, lon=lon, var=str(var)))
        time.sleep(0.3)
    FLOWS.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def _one(var):
    try:
        t, v, e = series(var)
        return var, len(t), float(v.max()) if len(v) else None
    except Exception as ex:                               # noqa: BLE001
        return var, -1, f"{type(ex).__name__}: {str(ex)[:80]}"


def download(vars_):
    todo = [v for v in vars_ if not (FLOWS / f"chj_{v}.npz").exists()]
    print(len(vars_), "variables,", len(todo), "to download")
    with ThreadPoolExecutor(WORKERS) as ex:
        for var, n, mx in ex.map(_one, todo):
            print(var, n, mx, flush=True)


def rain_hourly():
    """Hourly rain (mm) of all pluviometers on one axis -> rain_hourly.npz (t_end[h], p (S, T) float32 with NaN where the
    station has no samples in that hour, lat, lon, ids)."""
    st = rain_stations()
    cols = []
    for s in st:
        f = FLOWS / f"chj_{s['var']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        cols.append((s, *hourly(z["t"], z["v"], "sum12")))
    t0 = min(c[1][0] for c in cols if len(c[1])); t1 = max(c[1][-1] for c in cols if len(c[1]))
    axis = np.arange(t0, t1 + np.timedelta64(1, "h"))
    p = np.full((len(cols), len(axis)), np.nan, np.float32)
    for k, (s, u, val, n) in enumerate(cols):
        ok = n >= 9                                        # at least 45 min of data in the hour
        p[k, (u[ok] - t0).astype(int)] = val[ok] * 12.0 / n[ok]
    np.savez_compressed(FLOWS / "rain_hourly.npz", t_end=axis, p=p, lat=np.array([c[0]["lat"] for c in cols]),
                        lon=np.array([c[0]["lon"] for c in cols]), ids=np.array([c[0]["id"] for c in cols]),
                        names=np.array([c[0]["name"] for c in cols]))
    print("rain_hourly", p.shape, "mean annual total (mm/yr)", float(np.nanmean(np.nansum(p, axis=1)) / (len(axis) / 8766)))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "flows"
    if what == "flows":
        download(flow_vars())
    elif what == "segura":
        for pt in sys.argv[2:] or ["01O03A1"]:
            t, v, _ = segura_series(pt)
            print(pt, len(t), float(v.max()) if len(v) else None, t[0] if len(t) else None)
    elif what == "rain":
        download([s["var"] for s in rain_stations()])
        rain_hourly()
