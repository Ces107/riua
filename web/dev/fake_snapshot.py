"""A FABRICATED storm snapshot for developing and testing the page. NOT a forecast.

    py -3.11 web/dev/fake_snapshot.py            -> web/dev/data/{snapshot.json, explain-*.bin, points.json}

The rain is invented (a storm over the Poyo / Magro headwaters peaking about 20 h from now, a
weaker second episode on day 4), but everything after the rain is the real chain: the scenarios
are riua.core.risk.Member objects and the snapshot is built by the backend's own
horizon_product() and pack_horizon(), so the structure and the numbers (calibration, copula,
decision, basin products, packing) are exactly what the live pipeline produces.

The control-point block (`points`) cannot come from the real chain while the control-point
network is not built, so it is fabricated here with the documented shape, for a handful of the
points of geo/hydro/sections/out/sections.json. The page opens this data set with `?dev=1`.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from riua import params as P, product as PR, snapshot as S, static  # noqa: E402
from riua.core import grid, risk  # noqa: E402

OUT = REPO / "web" / "dev" / "data"
UTC = timezone.utc
rng = np.random.default_rng(20241029)


def storm(t_rel_h: np.ndarray, lat0: float, lon0: float, t_peak: float, amp: float, dur_h: float, size_km: float):
    """Hourly rain (T, NY, NX): a Gaussian cell in space and time."""
    lat2, lon2 = grid.mesh()
    d2 = ((lat2 - lat0) * 111.0) ** 2 + ((lon2 - lon0) * 86.0) ** 2
    space = np.exp(-d2 / (2 * size_km ** 2))
    time = np.exp(-((t_rel_h - t_peak) ** 2) / (2 * dur_h ** 2))
    return (amp * time[:, None, None] * space[None]).astype(np.float32)


def member(name, family, model, run, t_end, t_rel, step, scale=1.0, jitter=1.0):
    a = float(np.exp(rng.normal(0.0, 0.35 * jitter))) * scale
    dlat, dlon = rng.normal(0, 0.07 * jitter), rng.normal(0, 0.09 * jitter)
    dt = rng.normal(0, 2.0 * jitter)
    p = storm(t_rel, 39.43 + dlat, -0.72 + dlon, 20.0 + dt, 38.0 * a, 3.2, 16.0)
    p += storm(t_rel, 39.05 + dlat, -0.35 + dlon, 24.0 + dt, 14.0 * a, 4.0, 22.0)
    p += storm(t_rel, 38.85 + 2 * dlat, -0.10 + 2 * dlon, 86.0 + 4 * dt, 28.0 * a, 6.0, 30.0)   # day 4, weaker
    p += storm(t_rel, 39.50 + dlat, -0.80 + dlon, 2.5 + 0.3 * dt, 45.0 * a, 1.6, 12.0)            # showers now
    if step > 1:                       # coarse models: uniform within the step
        T = (len(t_end) // step) * step
        q = p[:T].reshape(T // step, step, *p.shape[1:]).mean(axis=1)
        p[:T] = np.repeat(q, step, axis=0)
    return risk.Member(name, family, model, run, t_end, p, step)


def main() -> None:
    now = datetime.now(UTC)
    hnow = PR.top_of_hour(now)
    params = P.load()
    st = static.load()
    thr = static.thresholds(params)
    bs = static.basins()
    h0 = np.datetime64(hnow, "h")
    t_end = h0 + np.arange(-13, 8 * 24 + 1) * np.timedelta64(1, "h")
    t_rel = (t_end - h0) / np.timedelta64(1, "h")
    short = t_rel <= 52

    def run(age_h):
        return hnow - timedelta(hours=age_h)

    nwp = []
    for k, age in enumerate((3, 6, 9)):
        nwp.append(member(f"AROME-HD 1,3 km · {run(age):%d/%m %H}Z", "cp", "arome_hd", run(age), t_end[short], t_rel[short], 1))
    for age in (3, 9):
        nwp.append(member(f"AROME 2,5 km · {run(age):%d/%m %H}Z", "cp", "arome", run(age), t_end[short], t_rel[short], 1, 0.9))
    for age in (4, 10):
        nwp.append(member(f"ICON-EU 6,5 km · {run(age):%d/%m %H}Z", "regional", "icon_eu", run(age), t_end[short], t_rel[short], 1, 0.6))
        nwp.append(member(f"ARPEGE 0,1° · {run(age):%d/%m %H}Z", "regional", "arpege", run(age), t_end[short], t_rel[short], 1, 0.6))
    for age in (6, 18):
        nwp.append(member(f"IFS 0,25° · {run(age):%d/%m %H}Z", "global", "ifs", run(age), t_end, t_rel, 3, 0.45))
    ens = [member(f"ENS m{k:02d} · {run(10):%d/%m %H}Z", "ens", "ifs_ens", run(10), t_end, t_rel, 12, 0.4, 1.6) for k in range(51)]
    hours6 = t_rel[(t_rel >= 1) & (t_rel <= 6)]
    radar = [member(f"Radar STEPS m{k + 1:02d} → {nwp[k % 5].name}", "radar", "steps", hnow,
                    t_end[(t_rel >= 1) & (t_rel <= 6)], hours6, 1, 1.0, 0.6) for k in range(20)]

    jj, ii = np.mgrid[0:grid.NY, 0:grid.NX]
    lattice = (jj % 2 == 0) & (ii % 2 == 0)
    members = {"now": radar + [m for m in nwp if m.family == "cp"], "mid": nwp,
               "long": ens + [m for m in nwp if m.family == "global"]}
    snap = {"v": S.SNAPSHOT_VERSION, "generated": S.iso(now), "params_version": "FABRICATED · " + str(params.get("version")),
            "grid": {"lon0": grid.LON0, "lat0": grid.LAT0, "d": grid.D, "nx": grid.NX, "ny": grid.NY},
            "mask": S.b64(np.packbits(st.mask.ravel())), "n_cells": st.n_cells,
            "thresholds": {"zones": st.zone_thr, "extreme": params["extreme"], "source": st.thresholds_source},
            "horizons": {}, "explain": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    pts = fake_points_meta()
    for hz in P.HORIZONS:
        o = PR.horizon_product(hz, members[hz], now, params, st, thr, bs, None, sample_mask=lattice if hz == "mid" else None)
        blk, blob, head = PR.pack_horizon(hz, o, st.mask, params)
        blk["points"] = fake_points(pts, o["frames"], t_end, t_rel, params["tau"][hz], hz)
        snap["horizons"][hz] = blk
        (OUT / f"explain-{hz}.bin").write_bytes(blob)
        snap["explain"][hz] = head
        print(hz, "levels", np.bincount(o["level"][:, st.mask].ravel(), minlength=6).tolist(), "M", head["M"])

    m = st.mask.ravel()
    past = storm(np.arange(-23, 1.0), 39.50, -0.80, -1.0, 9.0, 4.0, 18.0)
    snap["obs"] = {"hours": 24, "last_hour": S.iso(h0),
                   "o1": S.b64(S.code_mm(past[-1].ravel()[m] * 1.4)), "o12": S.b64(S.code_mm(past[-12:].sum(0).ravel()[m])),
                   "o24": S.b64(S.code_mm(past.sum(0).ravel()[m]))}
    t_obs = (now - timedelta(minutes=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    places = json.loads((REPO / "geo" / "out" / "places.json").read_text(encoding="utf-8"))["places"]
    gauges = []
    for k, p in enumerate(sorted(places, key=lambda p: -p["pop"])[:90:2]):
        c = grid.cell_of(p["lat"], p["lon"])
        v12 = float(past[-12:].sum(0)[c]) if c else 0.0
        src = ("saih_chj", "aemet", "saih_segura")[k % 3]
        gauges.append({"id": f"FAKE{k:03d}", "name": f"{p['name']} (inventado)", "lat": p["lat"] + 0.01, "lon": p["lon"] - 0.01,
                       "source": src, "t_utc": t_obs, "p_1h": round(v12 * 0.18, 1), "p_12h": round(v12, 1),
                       "p_24h": None if src == "aemet" else round(v12 * 1.3, 1)})
    snap["gauges"] = gauges
    snap["rivers"] = [
        {"id": "FAKE-R1", "name": "Rambla del Poyo (A-3) (inventado)", "river": "Rambla del Poyo", "lat": 39.4738, "lon": -0.5833,
         "source": "saih_chj", "t_utc": t_obs, "level_m": 0.42, "flow_m3s": 38.0, "thr_low": 30.0, "thr_mid": 70.0, "thr_high": 150.0},
        {"id": "FAKE-R2", "name": "Magro en Guadassuar (inventado)", "river": "Río Magro", "lat": 39.18, "lon": -0.48,
         "source": "saih_chj", "t_utc": t_obs, "level_m": 0.8, "flow_m3s": 12.5, "thr_low": 50.0, "thr_mid": 120.0, "thr_high": 300.0},
    ]
    on = (now + timedelta(hours=10)).strftime("%Y-%m-%dT%H:00:00Z")
    off = (now + timedelta(hours=34)).strftime("%Y-%m-%dT%H:00:00Z")
    snap["warnings"] = [
        {"zone_code": "774604", "zone_name": "Litoral sur de Valencia", "phenomenon": "rain", "level": "orange", "level_num": 2,
         "onset": on, "expires": off, "text": "Precipitación acumulada en una hora: 50 mm. (AVISO INVENTADO PARA PRUEBAS)",
         "params": {"code": "P1", "label": "Precipitación acumulada en una hora", "value": 50.0, "unit": "mm", "accum_hours": 1,
                    "probability": "40%-70%"}, "feed": "aemet_cap"},
        {"zone_code": "774603", "zone_name": "Interior sur de Valencia", "phenomenon": "rain", "level": "red", "level_num": 3,
         "onset": on, "expires": off, "text": "Precipitación acumulada en 12 horas: 180 mm. (AVISO INVENTADO PARA PRUEBAS)",
         "params": {"code": "PP", "label": "Precipitación acumulada en 12 horas", "value": 180.0, "unit": "mm", "accum_hours": 12,
                    "probability": ">70%"}, "feed": "aemet_cap"},
    ]
    snap["drivers"] = {"text": "DATOS INVENTADOS. Dana al suroeste de la península con flujo húmedo de levante en capas bajas; "
                               "agua precipitable alta y convergencia persistente sobre el interior de Valencia."}
    # reservoirs: INVENTED states (how full each one is), run through the real model with the invented storm
    try:
        from riua.core import reservoirs as RS
        dn = RS.load_dams()
        fill = {"forata": 0.86, "buseo": 0.97, "loriguilla": 0.75, "maria-cristina": 1.01, "bellus": 0.35, "beniarres": 0.72,
                "tous": 0.2, "escalona": 0.03, "regajo": 0.9, "algar": 0.6, "santomera": 0.1, "amadorio": 0.95, "guadalest": 0.99}
        live = []
        for k, i in enumerate(dn.ids):
            if not np.isfinite(dn.v_spill[k]):
                continue
            v = float(fill.get(i, 0.25 + 0.5 * rng.random()) * dn.v_spill[k])
            hrs = [S.iso(h0 - np.timedelta64(n, "h")) for n in range(12, -1, -1)]
            live.append({"id": i, "source": dn.meta[k]["source"], "t_utc": t_obs, "volume_hm3": v, "level_m": None,
                         "inflow_m3s": 2.0, "outflow_m3s": 1.0, "outflow_river_m3s": 1.0, "rate_hm3h": 0.001,
                         "series": {"t": hrs, "v": [v * (1 - 0.002 * n) for n in range(12, -1, -1)]}})
        rblk = RS.static_block(dn, live, [c["id"] for c in static.hydro_net()[1]], h0)
        for hz in P.HORIZONS:
            frs = PR.frames_for(hz, now)
            ta = np.unique(np.concatenate([x.t_end for x in members[hz]]))
            ta = ta[(ta > h0 - np.timedelta64(12, "h")) & (ta <= frs[-1][1])]
            rblk["horizons"][hz] = RS.pack(RS.reservoir_product(members[hz], frs, dn, live, params, hz, hnow, ta), dn)
        snap["reservoirs"] = rblk
    except Exception as e:  # noqa: BLE001 - the page must also work without the block
        import traceback
        traceback.print_exc()
        print("reservoirs skipped:", type(e).__name__, e)
    snap["sources"] = [
        {"id": "reservoirs", "label": "Embalses SAIH (inventado)", "ok": True, "n": 51},
        {"id": "radar", "label": "Radar (OPERA + AEMET)", "ok": True, "source": "opera", "frames": 13, "last": S.iso(now)},
        {"id": "gauges", "label": "Pluviómetros SAIH / AEMET", "ok": True, "n": len(gauges), "sources": ["aemet", "saih_chj", "saih_segura"]},
        {"id": "arome_hd", "label": "AROME-HD 1,3 km", "ok": True, "runs": [run(a).strftime("%Y-%m-%dT%H:%MZ") for a in (3, 6, 9)], "family": "cp"},
        {"id": "arome", "label": "AROME 2,5 km", "ok": True, "runs": [run(a).strftime("%Y-%m-%dT%H:%MZ") for a in (3, 9)], "family": "cp"},
        {"id": "icon_eu", "label": "ICON-EU 6,5 km", "ok": True, "runs": [run(a).strftime("%Y-%m-%dT%H:%MZ") for a in (4, 10)], "family": "regional"},
        {"id": "arpege", "label": "ARPEGE 0,1°", "ok": False, "family": "regional", "error": "TimeoutError: ejemplo de fuente caída (inventado)"},
        {"id": "ifs", "label": "IFS 0,25°", "ok": True, "runs": [run(a).strftime("%Y-%m-%dT%H:%MZ") for a in (6, 18)], "family": "global"},
        {"id": "ifs_ens", "label": "ECMWF ENS 51 miembros", "ok": True, "family": "ens", "runs": [run(10).strftime("%Y-%m-%dT%H:%MZ")], "n": 51},
        {"id": "nowcast", "label": "Nowcast radar pysteps STEPS", "ok": True, "method": "steps", "members": 20},
        {"id": "aemet", "label": "Avisos AEMET (CAP)", "ok": True, "n": 2},
    ]
    snap["notes"] = ["FABRICATED DATA for web development (web/dev/fake_snapshot.py). Not a forecast."]
    snap["timing_s"] = {"total": 0.0}
    (OUT / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    (OUT / "points.json").write_text(json.dumps({"points": pts}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("snapshot.json", round((OUT / "snapshot.json").stat().st_size / 1024), "KB")


FAKE_IDS = ["poyo-chiva", "poyo-ribarroja", "poyo-torrent", "poyo-paiporta", "poyo-catarroja", "saleta-aldaia",
            "picassent-beniparrell", "carraixet-betera", "carraixet-alfara", "magro-requena", "magro-algemesi", "serpis-gandia"]


def fake_points_meta() -> list[dict]:
    sj = json.loads((REPO / "geo" / "hydro" / "sections" / "out" / "sections.json").read_text(encoding="utf-8"))["points"]
    seed = {p["id"]: p for p in json.loads((REPO / "geo" / "hydro" / "control_points_seed.json").read_text(encoding="utf-8"))["points"]}
    out = []
    for i in FAKE_IDS:
        s = sj.get(i)
        if not s:
            continue
        pos = s.get("thalweg") or seed.get(i)
        if not pos:
            continue
        out.append({"id": i, "stream": s["stream"], "town": s["town"], "lat": round(pos["lat"], 5),
                    "lon": round(pos["lon"], 5), "area_km2": None, "tc_h": None,
                    "cap": s.get("q_capacity") or s.get("q_bankfull"), "cap_range": s.get("q_bankfull_range"),
                    "bank_m": s.get("bankfull_depth_m"), "lining": s.get("lining"), "confidence": s.get("confidence"),
                    "gauge": None})
    return out


def fake_points(pts, frames, t_end, t_rel, tau, hz) -> dict:
    """Invented hydrographs with the documented shape."""
    Pn, F = len(pts), len(frames)
    sel = (t_end > np.datetime64(PR.top_of_hour(datetime.now(UTC)), "h") - np.timedelta64(12, "h")) & (t_end <= frames[-1][1])
    tt, tr = t_end[sel], t_rel[sel]
    q = np.zeros((3, len(tt), Pn), np.float32)
    strength = np.array([1.5, 0.9, 1.3, 1.25, 0.8, 0.55, 0.35, 0.12, 0.2, 0.3, 0.7, 0.05])[:Pn]
    lag = np.array([0.0, 1.0, 2.0, 2.5, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0, 5.0, 3.0])[:Pn]
    cap = np.array([p["cap"] or np.nan for p in pts], float)
    for b in range(Pn):
        shape = np.exp(-((tr - 21.0 - lag[b]) ** 2) / (2 * 3.0 ** 2)) + 0.15 * np.exp(-((tr - 88.0) ** 2) / (2 * 8.0 ** 2))
        med = strength[b] * np.nan_to_num(cap[b], nan=200.0) * shape
        q[0, :, b], q[1, :, b], q[2, :, b] = 0.35 * med, med, 2.1 * med
    prob = np.zeros((4, F, Pn), np.float32)
    qpk = np.zeros((2, F, Pn), np.float32)
    hover = np.full((2, F, Pn), np.nan, np.float32)
    f_lv = np.array([0.25, 0.6, 1.0, 2.0])
    from scipy.special import ndtr
    for f, (t0, t1) in enumerate(frames):
        s = (tt > t0) & (tt <= t1)
        if not s.any():
            continue
        qpk[0, f], qpk[1, f] = q[1, s].max(axis=0), q[2, s].max(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = qpk[0, f][None] / (f_lv[:, None] * cap[None])
            prob[:, f] = ndtr(np.log(np.maximum(ratio, 1e-9)) / 0.65)
            for i in range(2):
                bank = np.array([p["bank_m"] or np.nan for p in pts], float)
                hover[i, f] = bank * (np.minimum(qpk[i, f] / cap, 3.0) ** 0.6 - 1.0)
    prob = np.nan_to_num(np.minimum.accumulate(prob, axis=0))
    level = risk.decide(prob, tau)
    return {"level": S.r(level, 0), "p": S.r(prob, 3), "qpeak": S.r(qpk, 0), "hover": S.r(hover, 2),
            "t": [S.iso(t) for t in tt], "q": S.r(q, 0)}


if __name__ == "__main__":
    main()
