"""HTTP API.

Two roles, chosen with RIUA_ROLE:

  api (default)  serves the latest snapshot produced elsewhere (it polls RIUA_SNAPSHOT_URL).
                 Needs only numpy + fastapi: this is what runs on a small free instance.
  all            also runs the production cycle itself on a schedule aligned with the
                 data (every RIUA_CYCLE_MIN minutes), for a host with real CPU and memory.

Endpoints
  GET /healthz
  GET /v1/status              sources, age of the product, timing
  GET /v1/snapshot            the whole product (see docs/SNAPSHOT.md)
  GET /v1/explain/{horizon}   per-scenario amounts behind every cell (binary)
  GET /v1/point?lat=&lon=     everything about one place, decoded to plain numbers
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

ROLE = os.environ.get("RIUA_ROLE", "api")
SNAPSHOT_URL = os.environ.get("RIUA_SNAPSHOT_URL", "https://ces107.github.io/riua/data/snapshot.json")
POLL_S = int(os.environ.get("RIUA_POLL_S", "120"))
CYCLE_MIN = int(os.environ.get("RIUA_CYCLE_MIN", "10"))
STATE = Path(os.environ.get("RIUA_STATE", "/tmp/riua-state"))
OUT = Path(os.environ.get("RIUA_OUT", "/tmp/riua-out"))
UA = {"User-Agent": "riua-api/1 (+https://ces107.github.io/riua/)"}

app = FastAPI(title="Riuà API", version="1", docs_url="/docs", redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2000)

_store = {"raw": None, "snap": None, "fetched": 0.0, "explain": {}, "error": None, "decoded": None}
_lock = threading.Lock()


def _get(url: str, timeout: float = 30.0) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _set(raw: bytes) -> None:
    snap = json.loads(raw)
    with _lock:
        if _store["snap"] is None or snap.get("generated", "") >= _store["snap"].get("generated", ""):
            if _store["snap"] is None or snap.get("generated") != _store["snap"].get("generated"):
                _store["explain"] = {}
                _store["decoded"] = None
            _store.update(raw=raw, snap=snap, fetched=time.time(), error=None)


def refresh(force: bool = False) -> None:
    if ROLE == "all":
        f = OUT / "snapshot.json"
        if f.exists():
            _set(f.read_bytes())
        return
    if not force and time.time() - _store["fetched"] < POLL_S and _store["snap"] is not None:
        return
    try:
        _set(_get(f"{SNAPSHOT_URL}?t={int(time.time() // 60)}"))
    except Exception as e:  # keep serving the previous product
        _store["error"] = f"{type(e).__name__}: {e}"[:200]
        _store["fetched"] = time.time() - POLL_S + 30


def _loop() -> None:
    while True:
        try:
            if ROLE == "all":
                from . import product
                product.run_cycle(STATE, OUT)
            refresh(force=True)
        except Exception as e:
            _store["error"] = f"{type(e).__name__}: {e}"[:200]
        time.sleep(CYCLE_MIN * 60 if ROLE == "all" else POLL_S)


@app.on_event("startup")
def _start() -> None:
    threading.Thread(target=_loop, daemon=True).start()


def _snap() -> dict:
    refresh()
    if _store["snap"] is None:
        raise HTTPException(503, "No hay producto todavía; vuelve a intentarlo en un minuto.")
    return _store["snap"]


def _age_min(snap: dict) -> float:
    t = datetime.strptime(snap["generated"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - t).total_seconds() / 60.0, 1)


@app.get("/healthz")
def healthz():
    return {"ok": True, "role": ROLE, "has_product": _store["snap"] is not None}


@app.get("/")
def root():
    return {"name": "Riuà API", "web": "https://ces107.github.io/riua/", "docs": "/docs",
            "disclaimer": "Herramienta no oficial. Las fuentes oficiales son AEMET y el 112 Comunitat Valenciana."}


@app.get("/v1/status")
def status():
    s = _snap()
    return {"generated": s["generated"], "age_min": _age_min(s), "params_version": s.get("params_version"),
            "sources": s.get("sources"), "notes": s.get("notes"), "timing_s": s.get("timing_s"),
            "role": ROLE, "last_error": _store["error"]}


@app.get("/v1/snapshot")
def snapshot():
    _snap()
    return Response(_store["raw"], media_type="application/json", headers={"Cache-Control": "public, max-age=60"})


@app.get("/v1/explain/{hz}")
def explain(hz: str):
    s = _snap()
    if hz not in s.get("horizons", {}):
        raise HTTPException(404, "horizonte desconocido")
    if hz not in _store["explain"]:
        try:
            if ROLE == "all":
                _store["explain"][hz] = (OUT / f"explain-{hz}.bin").read_bytes()
            else:
                _store["explain"][hz] = _get(SNAPSHOT_URL.rsplit("/", 1)[0] + f"/explain-{hz}.bin?g={s['generated']}")
        except Exception as e:
            raise HTTPException(502, f"no disponible: {type(e).__name__}")
    return Response(_store["explain"][hz], media_type="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=60", "X-Riua-Generated": s["generated"]})


def _decoded(s: dict) -> dict:
    if _store["decoded"] is None:
        g = s["grid"]
        mask = np.unpackbits(np.frombuffer(base64.b64decode(s["mask"]), np.uint8))[: g["nx"] * g["ny"]].astype(bool)
        index = np.full(mask.size, -1, int)
        index[mask] = np.arange(int(mask.sum()))
        d = {"index": index, "hz": {}}
        u8 = lambda b: np.frombuffer(base64.b64decode(b), np.uint8)
        N = int(mask.sum())
        for hz, blk in s["horizons"].items():
            F = len(blk["frames"])
            c = blk["cells"]
            d["hz"][hz] = {"level": u8(c["level"]).reshape(F, N), "p": u8(c["p"]).reshape(4, F, N),
                           "e1": u8(c["e1"]).reshape(2, F, N), "e12": u8(c["e12"]).reshape(2, F, N),
                           "m1": u8(c["m1"]).reshape(F, N), "q1": u8(c["q1"]).reshape(F, N),
                           "m12": u8(c["m12"]).reshape(F, N), "q12": u8(c["q12"]).reshape(F, N)}
        d["obs"] = {k: u8(s["obs"][k]) for k in ("o1", "o12", "o24")} if s.get("obs") else None
        _store["decoded"] = d
    return _store["decoded"]


@app.get("/v1/point")
def point(lat: float, lon: float):
    s = _snap()
    g = s["grid"]
    i, j = int(np.floor((lon - g["lon0"]) / g["d"])), int(np.floor((lat - g["lat0"]) / g["d"]))
    if not (0 <= i < g["nx"] and 0 <= j < g["ny"]):
        raise HTTPException(404, "fuera del dominio")
    d = _decoded(s)
    n = int(d["index"][j * g["nx"] + i])
    if n < 0:
        raise HTTPException(404, "celda fuera del ámbito (mar o fuera de las cuencas de la Comunitat Valenciana)")
    mm = lambda v: None if v == 255 else round((float(v) / 8.0) ** 2, 1)
    out = {"generated": s["generated"], "age_min": _age_min(s),
           "cell": {"j": j, "i": i, "lat": round(g["lat0"] + (j + 0.5) * g["d"], 3), "lon": round(g["lon0"] + (i + 0.5) * g["d"], 3)},
           "horizons": {}, "disclaimer": "Herramienta no oficial. Fuentes oficiales: AEMET y 112 Comunitat Valenciana."}
    for hz, a in d["hz"].items():
        fr = []
        for f, meta in enumerate(s["horizons"][hz]["frames"]):
            fr.append({"t0": meta["t0"], "t1": meta["t1"], "level": int(a["level"][f, n]),
                       "p": {str(L): round(float(a["p"][k, f, n]) / 200.0, 3) for k, L in enumerate((2, 3, 4, 5))},
                       "rain_1h_mm": {"median": mm(a["e1"][0, f, n]), "high": mm(a["e1"][1, f, n])},
                       "rain_12h_mm": {"median": mm(a["e12"][0, f, n]), "high": mm(a["e12"][1, f, n])},
                       "ensemble": {"m1": mm(a["m1"][f, n]), "q1": mm(a["q1"][f, n]), "m12": mm(a["m12"][f, n]), "q12": mm(a["q12"][f, n])}})
        out["horizons"][hz] = {"tau": s["horizons"][hz]["tau"], "frames": fr,
                               "max_level": max((x["level"] for x in fr), default=0)}
    if d["obs"]:
        out["observed_mm"] = {"last_1h": mm(d["obs"]["o1"][n]), "last_12h": mm(d["obs"]["o12"][n]), "last_24h": mm(d["obs"]["o24"][n])}
    return out
