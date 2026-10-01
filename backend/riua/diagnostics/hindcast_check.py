"""The ingredient diagnostics of :mod:`riua.diagnostics.ingredients` for a PAST date, from the archived forecasts.

Source: Open-Meteo's S3 time-series database ``data/<model>/<variable>/chunk_N.om`` (no quota). Each value there
comes from the latest model run before the valid time ("stitched" series: lead 0-6 h for IFS 0.25, 0-3 h for AROME),
i.e. what the newest forecast available at that moment said — not a reanalysis.

Layout (verified 2026-10-01): array ``[ny*nx, time]`` row-major from the SW corner, chunks of ~29 cells x one whole
time chunk; chunk ``N`` covers ``[N*L*dt, (N+1)*L*dt)`` seconds since the epoch with ``L = chunk_time_length`` and
``dt = temporal_resolution_seconds`` from ``data/<model>/static/meta.json``.

What is computed:

* IFS 0.25 deg on the regional box (same grid as the live product): every column diagnostic, 925-hPa moisture-flux
  convergence, upslope flow on the Riuà terrain, the strip statistics and the score;
* IFS 0.25 deg synoptic lattice (1 deg): closed lows at 500 hPa and the 300-hPa jet;
* AROME 0.025 deg: the latitude band of València and Chiva only (5x5 block means, as in the live product);
* MetPy's reference 1-D diagnostics at the anchor points for both models.

Run:  ``python -m riua.diagnostics.hindcast_check 2024-10-29T12 2024-10-29T18``
      (writes ``scratch/i1-ingredients/hindcast_<first time>.json`` and prints the tables).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from ..sources import openmeteo_s3 as om
from . import ingredients as ing

POINTS = {"valencia": ing.ANCHORS["valencia"], "chiva": ing.ANCHORS["chiva"]}
#: AROME band: rows 39.45..39.55 N, cells -0.80..-0.325 E -> four 5x5 blocks centred -0.75 (Chiva), -0.625, -0.5,
#: -0.375 (València) at 39.5 N, the same blocks as the live 0.125 deg grid.
AROME_BAND = (-0.8001, 39.4499, -0.3249, 39.5501)

_META: dict[str, dict] = {}


def ts_meta(model: str) -> dict:
    """Chunk length, time step and grid of a model in the time-series database."""
    if model not in _META:
        http = om._http()  # noqa: SLF001
        meta = json.loads(http.get(f"{om.S3_HOST}/data/{model}/static/meta.json"))
        s, w, n, e = om._wkt_bbox(meta["crs_wkt"])  # noqa: SLF001
        url = f"{om.S3_HOST}/data/{model}/static/HSURF.om"
        rd = om._open(url)  # noqa: SLF001
        ny, nx = (int(v) for v in rd.shape[-2:])
        rd.close()
        http.forget(url)
        _META[model] = {"L": int(meta["chunk_time_length"]), "dt": int(meta["temporal_resolution_seconds"]),
                        "s": s, "w": w, "ny": ny, "nx": nx, "dy": (n - s) / (ny - 1), "dx": (e - w) / (nx - 1)}
    return _META[model]


def window(model: str, bbox, step: int = 1):
    """(row indices, x0, x1, lats, lons) of the grid cells inside ``bbox`` = (west, south, east, north)."""
    g = ts_meta(model)
    bw, bs, be, bn = bbox
    y0 = max(0, int(np.ceil((bs - g["s"]) / g["dy"] - 1e-6)))
    y1 = min(g["ny"] - 1, int(np.floor((bn - g["s"]) / g["dy"] + 1e-6)))
    x0 = max(0, int(np.ceil((bw - g["w"]) / g["dx"] - 1e-6)))
    x1 = min(g["nx"] - 1, int(np.floor((be - g["w"]) / g["dx"] + 1e-6)))
    ys = list(range(y0, y1 + 1, step))
    lats = g["s"] + g["dy"] * np.array(ys)
    lons = (g["w"] + g["dx"] * np.arange(x0, x1 + 1))[::step]
    return ys, x0, x1, lats, lons


def read_ts(model: str, variable: str, times, bbox, step: int = 1) -> np.ndarray:
    """``[time, lat, lon]`` float32 of one variable at the given valid times (all inside one time chunk or not)."""
    g = ts_meta(model)
    ys, x0, x1, lats, lons = window(model, bbox, step)
    out = np.full((len(times), len(ys), len(lons)), np.nan, np.float32)
    span = g["L"] * g["dt"]
    by_chunk: dict[int, list] = {}
    for i, t in enumerate(times):
        ts = int(t.timestamp())
        by_chunk.setdefault(ts // span, []).append((i, (ts % span) // g["dt"]))
    for nchunk, items in by_chunk.items():
        url = f"{om.S3_HOST}/data/{model}/{variable}/chunk_{nchunk}.om"
        rd = om._open(url)  # noqa: SLF001
        try:
            k0, k1 = min(k for _i, k in items), max(k for _i, k in items)
            for r, y in enumerate(ys):
                a = y * g["nx"] + x0
                b = y * g["nx"] + x1 + 1
                sel = (slice(a, b), slice(k0, k1 + 1)) if len(rd.shape) == 2 else (slice(0, 1), slice(a, b), slice(k0, k1 + 1))
                block = np.asarray(rd.read_array(sel), dtype=np.float32).reshape(b - a, k1 - k0 + 1)[::step]
                for i, k in items:
                    out[i, r] = block[:, k - k0]
        finally:
            rd.close()
            om._http().forget(url)  # noqa: SLF001
    return out


def _load(key, model, variables, times, bbox, step, kind, threads, notes):
    """Store ``{"lat", "lon", "f": {var: [T, ny, nx]}}`` like the live reader, from the time-series database."""
    red, redc = ing._reduce(kind)  # noqa: SLF001
    ts_meta(model)
    _ys, _x0, _x1, lats, lons = window(model, bbox, step)

    def one(v):
        try:
            return v, read_ts(model, v, times, bbox, step), None
        except Exception as exc:  # noqa: BLE001
            return v, None, f"{type(exc).__name__}: {exc}"[:120]

    f, errs = {}, []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for v, arr, err in pool.map(one, variables):
            if err:
                errs.append(f"{v}: {err}")
            else:
                f[v] = np.stack([red(a) for a in arr])
    if errs:
        notes.append(f"{key}: {len(errs)} variable(s) not read, e.g. {errs[0]}")
    return {"lat": redc(np.asarray(lats, float)), "lon": redc(np.asarray(lons, float)), "f": f}


def run(times, threads: int = 10) -> dict:
    """Diagnostics for the given valid times (UTC datetimes)."""
    times = [t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t for t in times]
    notes: list[str] = []
    nt = len(times)
    r0, b0 = om.transfer_stats()
    tic = time.time()
    st_ifs = _load("ifs", ing.IFS, ing.IFS_VARS, times, ing.REGION, 1, "none", threads, notes)
    st_syn = _load("syn", ing.IFS, ing.SYN_VARS, times, ing.SYN_BBOX, ing.SYN_STEP, "none", 4, notes)
    st_aro = _load("arome", ing.AROME, ing.AROME_VARS, times, AROME_BAND, 1, "block", threads, notes)
    extras = {}
    for model, var in (("ncep_gfs025", "freezing_level_height"), ("ncep_gfs025", "cape"), ("ncep_gfs025", "lifted_index"),
                       ("ncep_gfs025", "convective_inhibition"), ("dwd_icon_eu", "freezing_level_height"),
                       ("dwd_icon_eu", "cape")):
        try:
            _ys, _x0, _x1, la, lo = window(model, (-1.0, 39.2, 0.0, 39.8))
            arr = read_ts(model, var, times, (-1.0, 39.2, 0.0, 39.8))
            extras[f"{model}.{var}"] = {"lat": la, "lon": lo, "v": arr}
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{model}.{var} not read: {type(exc).__name__}: {exc}"[:160])
    r1, b1 = om.transfer_stats()
    read_s = time.time() - tic

    out = {"times": [ing._iso(t) for t in times], "notes": notes, "models": {}}  # noqa: SLF001
    spec = {"ifs": (st_ifs, ing.IFS, ing.IFS_T_LEVELS, ing.IFS_W_LEVELS, ing.REGION, "none"),
            "arome": (st_aro, ing.AROME, ing.AROME_T_LEVELS, ing.AROME_W_LEVELS, AROME_BAND, "block")}
    diag = {}
    for key, (st, model, tl, wl, bbox, kind) in spec.items():
        shape = (len(st["lat"]), len(st["lon"]))
        zs = ing._orography(model, bbox, kind, notes)  # noqa: SLF001
        d = {k: np.full((nt,) + shape, np.nan) for k in ing.COLUMN_KEYS}
        pts = {}
        for ti in range(nt):
            c = ing._columns(st, ti, tl, wl, zs)  # noqa: SLF001
            res = ing.diagnose_columns(c)
            for k in ing.COLUMN_KEYS:
                d[k][ti] = res[k].reshape(shape)
            for name, (la, lo) in POINTS.items():
                j, i = int(np.abs(st["lat"] - la).argmin()), int(np.abs(st["lon"] - lo).argmin())
                n = j * shape[1] + i
                s = ing._column_sounding(c, n, float(res["psfc"][n]))  # noqa: SLF001
                ex = ing._exact_job(s) or {}  # noqa: SLF001
                ex.pop("parcel_mu_p", None)
                ex.pop("parcel_mu_t", None)
                pts.setdefault(name, []).append({
                    "grid_lat": float(st["lat"][j]), "grid_lon": float(st["lon"][i]), "zs": float(np.ravel(zs)[n]),
                    "metpy": {k: ing._num(v, 3) if isinstance(v, float) else v for k, v in ex.items()},  # noqa: SLF001
                    "engine": {k: ing._num(res[k][n], 3) for k in ing.COLUMN_KEYS},  # noqa: SLF001
                    "model_cape": ing._num(float(st["f"]["cape"][ti].ravel()[n]), 0) if "cape" in st["f"] else None,  # noqa: SLF001
                    "t850_c": ing._num(float(st["f"]["temperature_850hPa"][ti].ravel()[n]), 1),  # noqa: SLF001
                    "t500_c": ing._num(float(st["f"]["temperature_500hPa"][ti].ravel()[n]), 1),  # noqa: SLF001
                    "wind850": [ing._num(float(st["f"]["wind_u_component_850hPa"][ti].ravel()[n]), 1),  # noqa: SLF001
                                ing._num(float(st["f"]["wind_v_component_850hPa"][ti].ravel()[n]), 1)],  # noqa: SLF001
                })
        d["lat"], d["lon"] = st["lat"], st["lon"]
        diag[key] = d
        out["models"][key] = {"points": pts}

    # ---- IFS regional extras: model PWAT, MFC, upslope, strip statistics, score -------------------------------
    d = diag["ifs"]
    f = st_ifs["f"]
    d["pwat_profile"] = d["pwat"].copy()
    if "total_column_integrated_water_vapour" in f:
        d["pwat"] = f["total_column_integrated_water_vapour"].astype(float)
    d["mfc925"] = np.stack([ing._mfc925(st_ifs, ti, d["psfc"][ti]) for ti in range(nt)])  # noqa: SLF001
    d["llj_dir"] = ing._wdir(d["llj_u"], d["llj_v"])  # noqa: SLF001
    d["ivt_dir"] = ing._wdir(d["ivt_u"], d["ivt_v"])  # noqa: SLF001
    strip = ing._strip_mask(d["lat"], d["lon"])  # noqa: SLF001
    ter = ing._terrain()  # noqa: SLF001
    ups = []
    d["upslope_w"] = np.zeros(d["pwat"].shape)
    for ti in range(nt):
        res = ing._upslope(d["lat"], d["lon"], d["u_low"][ti], d["v_low"][ti], d["q_low"][ti])  # noqa: SLF001
        if res is None:
            ups.append(None)
            continue
        w, qw = res
        dom = ter["in_domain"]
        rec = {"w_p90": ing._num(np.nanpercentile(w[dom], 90), 3), "w_max": ing._num(np.nanmax(w[dom]), 3),  # noqa: SLF001
               "qflux_p90": ing._num(np.nanpercentile(qw[dom], 90), 2)}  # noqa: SLF001
        for name, (la, lo) in POINTS.items():
            j, i = int(np.abs(ter["lat"] - la).argmin()), int(np.abs(ter["lon"] - lo).argmin())
            rec[name] = ing._num(np.nanmax(w[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]), 3)  # noqa: SLF001
        ups.append(rec)
        jr = np.abs(d["lat"][None, :] - ter["lat"][:, None]).argmin(axis=1)
        ir = np.abs(d["lon"][None, :] - ter["lon"][:, None]).argmin(axis=1)
        g = np.zeros(d["lat"].size * d["lon"].size)
        np.maximum.at(g, (jr[:, None] * d["lon"].size + ir[None, :]).ravel(), np.nan_to_num(w).ravel())
        d["upslope_w"][ti] = g.reshape(d["lat"].size, d["lon"].size)
    d.update(ing.ingredients_score(d))
    strip_stats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ti in range(nt):
            rec = {}
            for name, src, stat, nd in ing.SERIES:
                if stat == "vdir":
                    um, vm = np.nanmean(d[src[0]][ti][strip]), np.nanmean(d[src[1]][ti][strip])
                    rec[name] = ing._num(ing._wdir(um, vm), 0)  # noqa: SLF001
                elif isinstance(src, str) and src in d:
                    vals = d[src][ti][strip]
                    rec[name] = ing._num(np.nanmean(vals) if stat == "mean" else np.nanpercentile(vals, int(stat[1:])), nd)  # noqa: SLF001
            rec["ivt_max_box"] = ing._num(np.nanmax(d["ivt"][ti]), 0)  # noqa: SLF001
            rec["pwat_max_box"] = ing._num(np.nanmax(d["pwat"][ti]), 1)  # noqa: SLF001
            rec["mucape_max_box"] = ing._num(np.nanmax(d["mucape"][ti]), 0)  # noqa: SLF001
            rec["model_cape_max_box"] = ing._num(np.nanmax(f["cape"][ti]), 0) if "cape" in f else None  # noqa: SLF001
            rec["llj_max_box"] = ing._num(np.nanmax(d["llj_speed"][ti]), 1)  # noqa: SLF001
            strip_stats.append(rec)
    out["models"]["ifs"]["strip"] = strip_stats
    out["models"]["ifs"]["upslope"] = ups
    for name, (la, lo) in POINTS.items():
        j, i = int(np.abs(d["lat"] - la).argmin()), int(np.abs(d["lon"] - lo).argmin())
        for ti in range(nt):
            out["models"]["ifs"]["points"][name][ti]["score"] = ing._num(d["score"][ti, j, i], 2)  # noqa: SLF001
            out["models"]["ifs"]["points"][name][ti]["pwat_model"] = ing._num(d["pwat"][ti, j, i], 1)  # noqa: SLF001
            out["models"]["ifs"]["points"][name][ti]["mfc925"] = ing._num(d["mfc925"][ti, j, i], 2)  # noqa: SLF001

    # ---- synoptic --------------------------------------------------------------------------------------------
    syn = []
    fs = st_syn["f"]
    for ti in range(nt):
        rec = {"lows": [], "jet": None}
        try:
            lows = ing.find_closed_lows(fs["geopotential_height_500hPa"][ti], st_syn["lat"], st_syn["lon"],
                                        fs.get("temperature_500hPa", [None] * nt)[ti])
            um, vm = np.nanmean(d["llj_u"][ti][strip]), np.nanmean(d["llj_v"][ti][strip])
            fd, fsp = float(ing._wdir(um, vm)), float(np.hypot(um, vm))  # noqa: SLF001
            rec["onshore_flow"] = {"dir": round(fd), "speed": round(fsp, 1), "ok": bool(30 <= fd <= 150 and fsp >= 5)}
            for q in lows:
                q["favourable_position"] = bool(q["dist_km"] <= 1000.0 and 150.0 <= q["bearing"] <= 300.0)
                q["favourable"] = bool(q["favourable_position"] and rec["onshore_flow"]["ok"])
            rec["lows"] = [{k: (ing._num(v, 1) if isinstance(v, float) else v) for k, v in q.items()} for q in lows]  # noqa: SLF001
            rec["dana_flag"] = any(q["favourable"] for q in lows)
            jet = ing.jet_streak(fs["wind_u_component_300hPa"][ti].astype(float), fs["wind_v_component_300hPa"][ti].astype(float),
                                 st_syn["lat"], st_syn["lon"])
            rec["jet"] = {k: (ing._num(v, 1) if isinstance(v, float) else v) for k, v in jet.items()} if jet else None  # noqa: SLF001
            rec["t500_min_box"] = ing._num(float(np.nanmin(fs["temperature_500hPa"][ti])), 1)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            notes.append(f"synoptic {ing._iso(times[ti])}: {type(exc).__name__}: {exc}"[:200])  # noqa: SLF001
        syn.append(rec)
    out["synoptic"] = syn
    out["other_models"] = {}
    for k, e in extras.items():
        j, i = int(np.abs(e["lat"] - 39.47).argmin()), int(np.abs(e["lon"] + 0.5).argmin())
        out["other_models"][k] = {"at": [float(e["lat"][j]), float(e["lon"][i])],
                                  "values": [ing._num(float(e["v"][ti, j, i]), 0) for ti in range(nt)],  # noqa: SLF001
                                  "max_in_box_39.2-39.8N_1W-0E": [ing._num(float(np.nanmax(e["v"][ti])), 0) for ti in range(nt)]}  # noqa: SLF001
    out["transfer"] = {"requests": r1 - r0, "mb": round((b1 - b0) / 1e6, 1), "read_s": round(read_s, 1)}
    return out


def _print(out):
    keys = ["pwat", "ivt", "ivt_dir", "mucape", "mucin", "mlcape", "sbcape", "lcl_agl", "lfc_z", "el_z", "ncape", "li",
            "kindex", "tt", "fzl", "wcd", "rh_700_500", "llj_speed", "llj_dir", "cl_speed", "cl_dir", "corfidi_up",
            "corfidi_dn", "shear06"]
    for ti, ts in enumerate(out["times"]):
        print(f"\n=== {ts} ===")
        print(f"{'':12s}" + " ".join(f"{k[:8]:>8s}" for k in keys))
        for model, m in out["models"].items():
            for name, recs in m["points"].items():
                r = recs[ti]
                e = dict(r["engine"])
                e["ivt_dir"] = ing._num(ing._wdir(e["ivt_u"] or 0, e["ivt_v"] or 0), 0)  # noqa: SLF001
                e["llj_dir"] = ing._num(ing._wdir(e["llj_u"] or 0, e["llj_v"] or 0), 0)  # noqa: SLF001
                e["cl_dir"] = ing._num(ing._wdir(e["cl_u"] or 0, e["cl_v"] or 0), 0)  # noqa: SLF001
                for label, src in (("metpy", r["metpy"]), ("engine", e)):
                    row = " ".join(f"{'-' if src.get(k) is None else format(src[k], '.6g'):>8s}" for k in keys)
                    print(f"{model[:3]}/{name[:3]}/{label[:3]:3s} " + row)
                print(f"   {model}/{name}: grid {r['grid_lat']:.3f}N {r['grid_lon']:.3f}E zs={r['zs']:.0f} m, model cape {r['model_cape']}, "
                      f"T850 {r['t850_c']} C, T500 {r['t500_c']} C, wind850 (u,v) {r['wind850']}, "
                      f"score {r.get('score')}, model PWAT {r.get('pwat_model')}, MFC925 {r.get('mfc925')}, "
                      f"q_low {r['engine'].get('q_low')} g/kg, theta-e850 {r['engine'].get('thetae850')} K, "
                      f"p_sfc {r['engine'].get('psfc')} hPa, MU parcel from {r['engine'].get('mu_p0')} hPa")
        print("IFS strip:", out["models"]["ifs"]["strip"][ti])
        print("IFS upslope:", out["models"]["ifs"]["upslope"][ti])
        print("synoptic:", json.dumps(out["synoptic"][ti]))
        print("other models:", {k: (v["values"][ti], v["max_in_box_39.2-39.8N_1W-0E"][ti]) for k, v in out["other_models"].items()})
    print("\ntransfer:", out["transfer"])
    for n in out["notes"]:
        print(" -", n)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    stamps = argv or ["2024-10-29T12", "2024-10-29T18"]
    times = [datetime.strptime(s, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc) for s in stamps]
    t0 = time.time()
    out = run(times)
    out["elapsed_s"] = round(time.time() - t0, 1)
    path = ing.REPO / "scratch" / "i1-ingredients" / f"hindcast_{times[0]:%Y%m%d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    _print(out)
    print(f"wall {out['elapsed_s']} s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
