"""One production cycle: observations, scenarios, probabilities, levels, snapshot.

    python -m riua.product --state <dir> --out <dir>

writes snapshot.json (everything the page shows), explain-<horizon>.bin (the per-scenario
numbers behind every cell, for the audit view) and keeps a small rolling state
(hourly radar-gauge analyses, downloaded runs) in the state directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import ingest, params as P, snapshot as S, static
from .core import basins as B, grid, hydro as H, risk

log = logging.getLogger("riua.product")
UTC = timezone.utc

DEFAULT_CAL = {"c1": dict(a0=0.05, a1=0.55, a2=0.35, b0=1.0, b1=0.7, delta=-0.5),
               "c1_from12": dict(a0=0.05, a1=0.12, a2=0.10, b0=1.1, b1=0.3, delta=-1.0),
               "c12": dict(a0=0.1, a1=0.6, a2=0.4, b0=1.2, b1=0.7, delta=-1.0), "rho": 0.7}


# ------------------------------------------------------------------------------------ frames

def top_of_hour(now: datetime) -> datetime:
    return ingest.naive(now).replace(minute=0, second=0, microsecond=0)


def frames_for(horizon: str, now: datetime):
    h = np.datetime64(top_of_hour(now), "h")
    one = np.timedelta64(1, "h")
    if horizon == "now":
        return [(h + k * one, h + (k + 1) * one) for k in range(6)]
    if horizon == "mid":
        start = h + 6 * one
        start = start + ((3 - int(str(start)[11:13]) % 3) % 3) * one
        return [(start + 3 * k * one, start + 3 * (k + 1) * one) for k in range(14)]
    day0 = np.datetime64(str(h)[:10] + "T00", "h")
    return [(day0 + 24 * (d) * one, day0 + 24 * (d + 1) * one) for d in range(2, 8)]


def lead_class(horizon: str, now: datetime, frame) -> str:
    lead = float((frame[1] - np.datetime64(top_of_hour(now), "h")) / np.timedelta64(1, "h"))
    if horizon == "now":
        return "h1-2" if lead <= 2 else "h3-4" if lead <= 4 else "h5-6"
    if horizon == "mid":
        return "d1" if lead <= 30 else "d2"
    return "d2-3" if lead <= 96 else "d4-5" if lead <= 144 else "d6-7"


def calibration(params: dict, horizon: str, cls: str) -> risk.Calibration:
    d = params.get("calibration", {}).get(horizon, {}).get(cls)
    return risk.Calibration.from_dict(d or DEFAULT_CAL)


# ------------------------------------------------------------------------------- observations

@dataclass
class Obs:
    t_end: np.ndarray       # (K,) datetime64[h] complete clock hours, oldest first
    o_max: np.ndarray       # (K, NY, NX) mm, largest 1-km value in the cell
    o_mean: np.ndarray      # (K, NY, NX) mm, cell mean
    partial_max: np.ndarray | None = None    # accumulation since the top of the current hour
    partial_mean: np.ndarray | None = None
    partial_min: float = 0.0
    info: dict | None = None


def load_obs_state(state: Path) -> Obs:
    f = state / "obs.npz"
    if f.exists():
        z = np.load(f)
        return Obs(z["t_end"], z["o_max"].astype(np.float32), z["o_mean"].astype(np.float32))
    e = np.zeros((0, grid.NY, grid.NX), np.float32)
    return Obs(np.array([], "datetime64[h]"), e, e.copy())


def save_obs_state(state: Path, obs: Obs, keep_h: int = 96) -> None:
    state.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state / "obs.npz", t_end=obs.t_end[-keep_h:], o_max=obs.o_max[-keep_h:].astype(np.float16),
                        o_mean=obs.o_mean[-keep_h:].astype(np.float16))


def update_obs(state: Path, now: datetime, backfill_h: int = 14) -> tuple[Obs, list, dict]:
    """Radar-gauge analysis of the hours not yet in the state. Returns (obs, rate frames, report)."""
    from .radar import nowcast as NC, qpe
    from .sources import gauges as G, radar as R
    rep = {"radar": {"ok": False}, "gauges": {"ok": False}}
    obs = load_obs_state(state)
    hnow = top_of_hour(now)
    have = set(str(t) for t in obs.t_end)
    need = [hnow - timedelta(hours=k) for k in range(backfill_h - 1, -1, -1)]
    need = [t for t in need if str(np.datetime64(t, "h")) not in have]
    frames = []
    try:
        if need and (hnow - need[0]) > timedelta(hours=2):
            start = need[0] - timedelta(hours=1, minutes=10)
            t = start.replace(minute=(start.minute // 10) * 10)
            while t <= ingest.naive(now):
                try:
                    frames.append((t.replace(tzinfo=UTC), R.fetch_opera(t.replace(tzinfo=UTC))))
                except Exception:
                    pass
                t += timedelta(minutes=10)
            recent = R.fetch_radar_frames(n=13, source="auto")
            seen = {f[0] for f in recent}
            frames = [f for f in frames if f[0] not in seen] + recent
        else:
            frames = R.fetch_radar_frames(n=13, source="auto")
        frames.sort(key=lambda x: x[0])
        rates = [(t, qpe.rain_rate(qpe.despeckle(d))) for t, d in frames]
        rep["radar"] = {"ok": True, "source": getattr(R, "last_source", "?"), "frames": len(rates),
                        "last": S.iso(rates[-1][0]) if rates else None, "attribution": getattr(R, "ATTRIBUTION", None)}
    except Exception as e:
        rates = []
        rep["radar"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    new_t, new_max, new_mean = [], [], []
    for t in need:
        a, b = (t - timedelta(hours=1)).replace(tzinfo=UTC), t.replace(tzinfo=UTC)
        sub = [f for f in rates if a - timedelta(minutes=10) <= f[0] <= b]
        if len(sub) < 2:
            continue
        acc, cov = qpe.accumulate(sub, a, b)
        if cov < 0.6:
            continue
        new_t.append(np.datetime64(t, "h"))
        new_max.append(NC.to_analysis_max(np.nan_to_num(acc, nan=0.0)))
        new_mean.append(np.nan_to_num(NC.to_analysis_mean(acc), nan=0.0))
    if new_t:
        t_all = np.concatenate([obs.t_end, np.array(new_t)])
        order = np.argsort(t_all)
        obs = Obs(t_all[order], np.concatenate([obs.o_max, np.stack(new_max)])[order],
                  np.concatenate([obs.o_mean, np.stack(new_mean)])[order])
    # the hour under way
    if rates:
        a = hnow.replace(tzinfo=UTC)
        sub = [f for f in rates if f[0] >= a - timedelta(minutes=10)]
        if len(sub) >= 2 and sub[-1][0] > a:
            acc, _ = qpe.accumulate(sub, a, sub[-1][0])
            obs.partial_max = NC.to_analysis_max(np.nan_to_num(acc, nan=0.0))
            obs.partial_mean = np.nan_to_num(NC.to_analysis_mean(acc), nan=0.0)
            obs.partial_min = (sub[-1][0] - a).total_seconds() / 60.0
    # gauges: correct the last 12 h with the 12-h gauge totals (radar pattern, gauge amount)
    gl = []
    try:
        gl = G.fetch_rain_gauges()
        rep["gauges"] = {"ok": True, "n": len(gl), "sources": sorted({g["source"] for g in gl})}
        k12 = obs.t_end > np.datetime64(hnow, "h") - np.timedelta64(12, "h")
        if k12.sum() >= 10:
            use = [g for g in gl if g.get("p_12h") is not None and g.get("lat") is not None]
            raw = obs.o_mean[k12].sum(axis=0)
            fac, info = gauge_factor(raw, use)
            rep["gauges"]["merge"] = info
            if fac is not None:
                obs.o_mean[k12] *= fac
                obs.o_max[k12] *= fac
                if obs.partial_mean is not None:
                    obs.partial_mean *= fac; obs.partial_max *= fac
    except Exception as e:
        rep["gauges"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    obs.info = rep
    save_obs_state(state, obs)
    return obs, rates, {"report": rep, "gauges": gl}


def gauge_factor(raw12: np.ndarray, gauges: list[dict]):
    """Correction field (NY, NX) = merged / radar for the last 12 h on the analysis grid."""
    import wradlib as wrl
    if len(gauges) < 8:
        return None, {"method": "none", "n": len(gauges)}
    lat2, lon2 = grid.mesh()
    raw_xy = np.column_stack([lon2.ravel() * grid.KM_PER_DEG_LON, lat2.ravel() * grid.KM_PER_DEG_LAT])
    g_xy = np.array([[g["lon"] * grid.KM_PER_DEG_LON, g["lat"] * grid.KM_PER_DEG_LAT] for g in gauges])
    g_mm = np.array([float(g["p_12h"]) for g in gauges])
    if g_mm.max() < 1.0 and raw12.max() < 1.0:
        return None, {"method": "none (dry)", "n": len(gauges)}
    try:
        adj = wrl.adjust.AdjustMixed(g_xy, raw_xy, nnear_raws=5, mingages=5, minval=0.5, nnearest=8, p=2.0)
        merged = np.asarray(adj(g_mm, raw12.ravel().astype(np.float64))).reshape(raw12.shape)
    except Exception as e:
        return None, {"method": "failed", "error": f"{type(e).__name__}: {e}"[:160]}
    merged = np.where(np.isfinite(merged), np.maximum(merged, 0.0), raw12)
    fac = np.where(raw12 >= 0.5, np.clip(merged / np.maximum(raw12, 1e-6), 0.2, 6.0), 1.0).astype(np.float32)
    both = [(float(g["p_12h"]), float(raw12[grid.cell_of(g["lat"], g["lon"])])) for g in gauges if grid.cell_of(g["lat"], g["lon"])]
    gs, rs = sum(a for a, b in both if a >= 1 and b >= 1), sum(b for a, b in both if a >= 1 and b >= 1)
    return fac, {"method": "wradlib.AdjustMixed", "n": len(gauges), "gauge_max_12h": float(g_mm.max()),
                 "radar_over_gauge": round(rs / gs, 2) if gs > 0 else None}


def with_past(m: risk.Member, obs: Obs, hnow: datetime) -> risk.Member:
    """Replace everything up to the current hour with what was observed (and prepend it)."""
    h = np.datetime64(hnow, "h")
    fut = m.t_end > h
    past = obs.t_end <= h
    if not past.any():
        return m
    t = np.concatenate([obs.t_end[past], m.t_end[fut]])
    p = np.concatenate([obs.o_max[past], m.p[fut]]).astype(np.float32)
    area = np.concatenate([obs.o_mean[past], m.area[fut]]).astype(np.float32)
    return risk.Member(m.name, m.family, m.model, m.run, t, p, m.native_step_h, m.weight, area, m.meta)


# --------------------------------------------------------------------------------- now members

def nowcast_members(rates: list, obs: Obs, nwp: list[risk.Member], now: datetime, n_members: int = 20):
    """STEPS radar ensemble blended into the newest convection-permitting runs."""
    from .radar import nowcast as NC
    hnow = top_of_hour(now)
    rep = {"method": "none", "members": 0}
    cp = sorted([m for m in nwp if m.family == "cp"], key=lambda m: -m.run.timestamp())
    donors = cp or sorted(nwp, key=lambda m: -m.run.timestamp())
    hours = np.array([np.datetime64(hnow, "h") + np.timedelta64(k + 1, "h") for k in range(6)])
    out = []
    steps_h_max = steps_h_mean = None
    lead_off = 0.0
    if len(rates) >= 3:
        try:
            # 2-km grid: 4x cheaper and above the radar's real resolution over most of the box
            def coarse(a):
                s = a.shape
                return np.nanmean(a.reshape(s[0] // 2, 2, s[1] // 2, 2), axis=(1, 3))
            small = [(t, coarse(r)) for t, r in rates[-3:]]
            nc = NC.steps_ensemble(small, n_steps=18, n_members=n_members, seed=int(hnow.timestamp()) % 100000, workers=4)
            rep.update(method=nc["method"], wet_fraction=round(nc["wet_fraction"], 4), t0=S.iso(nc["t0"]))
            if nc["rate"].any():
                rate = np.repeat(np.repeat(nc["rate"], 2, axis=2), 2, axis=3)
                t_ends, acc = NC.hourly_from_steps(rate, nc["t0"])              # future part only
                steps_h_max = {np.datetime64(ingest.naive(t), "h"): NC.to_analysis_max(acc[:, k]) for k, t in enumerate(t_ends)}
                steps_h_mean = {np.datetime64(ingest.naive(t), "h"): NC.to_analysis_mean(acc[:, k]) for k, t in enumerate(t_ends)}
                lead_off = (ingest.naive(nc["t0"]) - hnow).total_seconds() / 3600.0
        except Exception as e:
            rep.update(method=f"failed: {type(e).__name__}: {e}"[:160])
    M = n_members if steps_h_max else 0
    for k in range(M):
        d = donors[k % len(donors)] if donors else None
        p = np.zeros((6, grid.NY, grid.NX), np.float32)
        a = np.zeros_like(p)
        for i, t in enumerate(hours):
            lead = (i + 1) - lead_off                       # hours after the last scan, end of this hour
            w = float(NC.blend_weights(np.array([max(lead - 0.5, 0.0)]))[0])
            if t not in steps_h_max:
                w = 0.0
            r_max = steps_h_max[t][k] if t in steps_h_max else 0.0
            r_mean = steps_h_mean[t][k] if t in steps_h_mean else 0.0
            if i == 0 and obs.partial_max is not None:      # the hour under way: add what already fell
                r_max = r_max + obs.partial_max
                r_mean = r_mean + obs.partial_mean
                frac_left = max(0.0, 1.0 - obs.partial_min / 60.0)
            else:
                frac_left = 1.0
            n_max = n_mean = 0.0
            if d is not None:
                j = np.nonzero(d.t_end == t)[0]
                if j.size:
                    n_max = d.p[j[0]] * frac_left + (obs.partial_max if (i == 0 and obs.partial_max is not None) else 0.0)
                    n_mean = d.area[j[0]] * frac_left + (obs.partial_mean if (i == 0 and obs.partial_mean is not None) else 0.0)
                else:
                    w = 1.0 if t in steps_h_max else 0.0
            else:
                w = 1.0
            p[i] = w * r_max + (1.0 - w) * n_max
            a[i] = w * r_mean + (1.0 - w) * n_mean
        out.append(risk.Member(f"Radar STEPS m{k + 1:02d}" + (f" → {d.name}" if d is not None else ""), "radar", "steps",
                               hnow, hours, p, 1, 1.0, a, {"donor": d.name if d is not None else None}))
    rep["members"] = len(out)
    return out, rep


# ------------------------------------------------------------------------------- ENS members

def ens_members(state: Path, now: datetime) -> tuple[list[risk.Member], dict]:
    """ECMWF ENS (51 members, 0.25 deg) straight from ECMWF open data: 12-h accumulations."""
    from .sources import ecmwf_open as E
    rep = {"id": "ifs_ens", "label": "ECMWF ENS 51 miembros", "ok": False, "family": "ens"}
    n = ingest.naive(now)
    cands = []
    base = n.replace(hour=(n.hour // 12) * 12, minute=0, second=0, microsecond=0)
    for k in range(4):
        cands.append(base - timedelta(hours=12 * k))
    for run in cands:
        if (n - run) < timedelta(hours=7):
            continue                                    # not published yet
        out = state / "ens" / f"ifsens_tp_{run:%Y%m%d%H}.npz"
        try:
            if not out.exists():
                first = int(np.ceil(max((n - run).total_seconds() / 3600.0 + 12, 12) / 12.0) * 12) - 12
                steps = list(range(max(first, 12), 204, 12))
                out.parent.mkdir(parents=True, exist_ok=True)
                E.fetch_run(run, steps, "tp", out, mirrors=["ecmwf", "gcs", "aws"], workers=6, verbose=False)
            z = np.load(out, allow_pickle=True)
        except Exception as e:
            rep["error"] = f"{run:%d/%m %H}Z: {type(e).__name__}: {e}"[:200]
            continue
        data, steps = z["data"], z["steps"].astype(int)             # (M, S, lat, lon) mm since init
        rg = ingest.regridder("ens025", z["lat"], z["lon"])
        members = []
        for k in range(data.shape[0]):
            t_list, p_list = [], []
            for s in range(1, len(steps)):
                dt_h = int(steps[s] - steps[s - 1])
                inc = np.maximum(rg(data[k, s] - data[k, s - 1]), 0.0) / dt_h
                for h in range(dt_h):
                    t_list.append(np.datetime64(run + timedelta(hours=int(steps[s - 1]) + h + 1), "h"))
                    p_list.append(inc)
            members.append(risk.Member(f"ENS m{int(z['members'][k]):02d} · {run:%d/%m %H}Z", "ens", "ifs_ens", run,
                                       np.array(t_list), np.stack(p_list).astype(np.float32), 12))
        rep.update(ok=True, runs=[run.strftime("%Y-%m-%dT%H:%MZ")], n=len(members))
        for old in (state / "ens").glob("ifsens_tp_*.npz"):
            if old != out:
                old.unlink(missing_ok=True)
        return members, rep
    return [], rep


# ------------------------------------------------------------------------------- one horizon

def horizon_product(hz: str, members: list[risk.Member], now: datetime, params: dict, st, thr, bs, net,
                    sample_mask=None) -> dict:
    hnow = top_of_hour(now)
    frames = frames_for(hz, now)
    pred = risk.predictors(members, frames, params, hz, hnow, sample_mask=sample_mask, keep_members=True)
    cals = [calibration(params, hz, lead_class(hz, now, f)) for f in frames]
    prob, p1, p12 = risk.calibrated_probabilities(pred, cals, thr)
    level = risk.decide(prob, params["tau"][hz])
    level[~pred.valid] = 0
    # calibrated expectation of the two amounts (median and 1-in-10 high scenario)
    F = len(frames)
    e1 = np.zeros((2, F, grid.NY, grid.NX), np.float32); e12 = np.zeros_like(e1)
    for f in range(F):
        if not pred.valid[f]:
            continue
        c = cals[f]
        m12, q12 = np.nan_to_num(pred.m12[f]), np.nan_to_num(pred.q12[f])
        for i, q in enumerate((0.5, 0.9)):
            e12[i, f] = c.c12.quantile(m12, q12, q)
            if pred.has_1h[f] and c.c1 is not None:
                e1[i, f] = c.c1.quantile(np.nan_to_num(pred.m1[f]), np.nan_to_num(pred.q1[f]), q)
            elif c.c1_from12 is not None:
                e1[i, f] = c.c1_from12.quantile(m12, q12, q)
    out = dict(frames=frames, pred=pred, prob=prob, p1=p1, p12=p12, level=level, e1=e1, e12=e12, cals=cals)
    out["basins"] = B.basin_product(members, frames, bs, params, hz, hnow)
    if net is not None:
        t_axis = np.unique(np.concatenate([m.t_end for m in members])) if members else np.array([], "datetime64[h]")
        lo = np.datetime64(hnow, "h") - np.timedelta64(12, "h")
        t_axis = t_axis[(t_axis > lo) & (t_axis <= frames[-1][1])]
        out["points"] = H.hydro_product(members, frames, net, params, hz, hnow, t_axis)
    return out


def pack_horizon(hz: str, o: dict, mask: np.ndarray, params: dict) -> tuple[dict, bytes, dict]:
    m = mask.ravel()
    cells = lambda a: a.reshape(*a.shape[:-2], -1)[..., m]
    pred = o["pred"]
    blk = {
        "frames": [{"t0": S.iso(t0), "t1": S.iso(t1), "ok": bool(pred.valid[f]), "has_1h": bool(pred.has_1h[f]),
                    "cal": o["cals"][f].to_dict()} for f, (t0, t1) in enumerate(o["frames"])],
        "tau": params["tau"][hz],
        "cells": {"level": S.b64(cells(o["level"])), "p": S.b64(S.code_prob(cells(o["prob"]))),
                  "e1": S.b64(S.code_mm(cells(o["e1"]))), "e12": S.b64(S.code_mm(cells(o["e12"]))),
                  "m1": S.b64(S.code_mm(cells(pred.m1))), "q1": S.b64(S.code_mm(cells(pred.q1))),
                  "m12": S.b64(S.code_mm(cells(pred.m12))), "q12": S.b64(S.code_mm(cells(pred.q12)))},
        "members": pred.audit,
    }
    bp = o["basins"]
    blk["basins"] = {"level": S.r(bp.level, 0), "p": S.r(bp.prob, 3), "own12": S.r(bp.own12, 0),
                     "up12": S.r(bp.up12, 0), "q": S.r(bp.q, 2), "upstream": S.r(bp.upstream_driven, 2)}
    if "points" in o:
        hp = o["points"]
        blk["points"] = {"level": S.r(hp.level, 0), "p": S.r(hp.prob, 3), "qpeak": S.r(hp.q_peak, 0),
                         "hover": S.r(hp.h_over, 2), "t": [S.iso(t) for t in hp.t_end], "q": S.r(hp.q_series, 0)}
    # audit binary: per-member amounts and the two marginal probabilities
    M = 0 if pred.a1 is None else pred.a1.shape[0]
    if M:
        a1 = S.code_mm(cells(pred.a1)); a1[~np.isfinite(cells(pred.a1))] = 255
        a12 = S.code_mm(cells(pred.a12)); a12[~np.isfinite(cells(pred.a12))] = 255
        blob = a1.tobytes() + a12.tobytes() + S.code_prob(cells(o["p1"])).tobytes() + S.code_prob(cells(o["p12"])).tobytes()
    else:
        blob = b""
    head = {"M": M, "F": len(o["frames"]), "N": int(m.sum()),
            "layout": "uint8 a1[M,F,N] (255 = not available), a12[M,F,N], p1[4,F,N], p12[4,F,N]; mm = (v/8)^2, P = v/200"}
    return blk, blob, head


# ------------------------------------------------------------------------------------- cycle

def run_cycle(state: Path, out: Path, now: datetime | None = None, with_radar: bool = True,
              with_ens: bool = True) -> dict:
    t_start = time.time()
    now = now or datetime.now(UTC)
    hnow = top_of_hour(now)
    params = P.load()
    st = static.load()
    thr = static.thresholds(params)
    bs = static.basins()
    net, cps = static.hydro_net()
    sources, notes = [], []
    timing = {}

    t = time.time()
    gauges_live, river_live, warn = [], [], []
    if with_radar:
        try:
            obs, rates, extra = update_obs(state, now)
            sources.append({"id": "radar", "label": "Radar (OPERA + AEMET)", **extra["report"]["radar"]})
            sources.append({"id": "gauges", "label": "Pluviómetros SAIH / AEMET", **extra["report"]["gauges"]})
            gauges_live = extra["gauges"]
        except Exception as e:
            log.error("obs failed: %s", traceback.format_exc())
            obs, rates = load_obs_state(state), []
            sources.append({"id": "radar", "label": "Radar", "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
    else:
        obs, rates = load_obs_state(state), []
    timing["obs"] = round(time.time() - t, 1)

    t = time.time()
    nwp, rep = ingest_cached(state, now)
    sources.extend(rep)
    timing["nwp"] = round(time.time() - t, 1)

    t = time.time()
    ens, erep = ens_members(state, now) if with_ens else ([], {"id": "ifs_ens", "ok": False, "label": "ECMWF ENS"})
    sources.append(erep)
    timing["ens"] = round(time.time() - t, 1)

    t = time.time()
    radar_m, nrep = nowcast_members(rates, obs, nwp, now) if with_radar else ([], {"method": "off"})
    sources.append({"id": "nowcast", "label": "Nowcast radar pysteps STEPS", "ok": bool(radar_m), **nrep})
    timing["nowcast"] = round(time.time() - t, 1)

    past = lambda ms: [with_past(m, obs, hnow) for m in ms]
    jj, ii = np.mgrid[0:grid.NY, 0:grid.NX]
    lattice = (jj % 2 == 0) & (ii % 2 == 0)
    horizon_members = {
        "now": past(radar_m + [m for m in nwp if m.family == "cp"]),
        "mid": past(nwp),
        "long": past(ens + [m for m in nwp if m.family == "global"]),
    }
    snap = {"v": S.SNAPSHOT_VERSION, "generated": S.iso(now), "params_version": params.get("version"),
            "grid": {"lon0": grid.LON0, "lat0": grid.LAT0, "d": grid.D, "nx": grid.NX, "ny": grid.NY},
            "mask": S.b64(np.packbits(st.mask.ravel())), "n_cells": st.n_cells,
            "thresholds": {"zones": st.zone_thr, "extreme": params["extreme"], "source": st.thresholds_source},
            "horizons": {}, "explain": {}}
    out.mkdir(parents=True, exist_ok=True)
    for hz in P.HORIZONS:
        t = time.time()
        try:
            o = horizon_product(hz, horizon_members[hz], now, params, st, thr, bs, net,
                                sample_mask=lattice if hz == "mid" else None)
            blk, blob, head = pack_horizon(hz, o, st.mask, params)
            snap["horizons"][hz] = blk
            (out / f"explain-{hz}.bin").write_bytes(blob)
            snap["explain"][hz] = head
        except Exception as e:
            log.error("horizon %s failed: %s", hz, traceback.format_exc())
            notes.append(f"{hz}: {type(e).__name__}: {e}"[:300])
        timing[hz] = round(time.time() - t, 1)

    # what has already fallen (analysis), per cell
    def acc(hours):
        k = obs.t_end > np.datetime64(hnow, "h") - np.timedelta64(hours, "h")
        return obs.o_mean[k].sum(axis=0) if k.any() else np.zeros((grid.NY, grid.NX), np.float32)
    m = st.mask.ravel()
    last1 = obs.o_max[-1] if len(obs.t_end) and obs.t_end[-1] == np.datetime64(hnow, "h") else np.zeros((grid.NY, grid.NX))
    snap["obs"] = {"hours": int(len(obs.t_end)), "last_hour": S.iso(obs.t_end[-1]) if len(obs.t_end) else None,
                   "o1": S.b64(S.code_mm(last1.ravel()[m])), "o12": S.b64(S.code_mm(acc(12).ravel()[m])),
                   "o24": S.b64(S.code_mm(acc(24).ravel()[m]))}
    snap["gauges"] = [{k: g.get(k) for k in ("id", "name", "lat", "lon", "source", "t_utc", "p_1h", "p_12h", "p_24h")}
                      for g in gauges_live if g.get("lat") is not None]
    try:
        from .sources import gauges as G
        river_live = G.fetch_river_gauges(with_trend=False)
    except Exception as e:
        notes.append(f"river gauges: {type(e).__name__}: {e}"[:200])
    snap["rivers"] = river_live
    try:
        from .sources import aemet_warnings as W
        warn = W.fetch_rain_warnings()
        sources.append({"id": "aemet", "label": "Avisos AEMET (CAP)", "ok": True, "n": len(warn)})
    except Exception as e:
        sources.append({"id": "aemet", "label": "Avisos AEMET", "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
    snap["warnings"] = warn
    try:
        from .diagnostics import ingredients as I
        snap["drivers"] = I.compute(now)
    except Exception as e:
        notes.append(f"drivers: {type(e).__name__}: {e}"[:200])
    snap["sources"] = sources
    snap["notes"] = notes
    timing["total"] = round(time.time() - t_start, 1)
    snap["timing_s"] = timing
    (out / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    return snap


def ingest_cached(state: Path, now: datetime):
    """Model runs are kept on disk between cycles: only new runs are downloaded."""
    from .sources import openmeteo_s3 as om
    cache = state / "nwp"
    cache.mkdir(parents=True, exist_ok=True)
    members, report, keep = [], [], set()
    t_from = now - timedelta(hours=14)
    for key, cfg in ingest.MODELS.items():
        try:
            runs = om.latest_runs(cfg["s3"], n=cfg["lags"])
        except Exception as e:
            report.append({"id": key, "label": cfg["label"], "ok": False, "family": cfg["family"],
                           "error": f"{type(e).__name__}: {e}"[:160]})
            continue
        got = []
        for r in runs:
            f = cache / f"{key}_{ingest.naive(r):%Y%m%d%H}.npz"
            keep.add(f.name)
            m = None
            if f.exists():
                z = np.load(f)
                m = risk.Member(str(z["name"]), cfg["family"], key, ingest.naive(r), z["t_end"],
                                z["p"].astype(np.float32), int(z["step"]), meta={"nan_frac": float(z["nan_frac"])})
            else:
                ahead = 52 if key != "ifs" else 24 * 8
                m = ingest.load_run(key, r, t_from, now + timedelta(hours=ahead))
                if m is not None:
                    np.savez_compressed(f, name=m.name, t_end=m.t_end, p=m.p.astype(np.float16), step=m.native_step_h,
                                        nan_frac=m.meta.get("nan_frac", 0.0))
            if m is not None:
                members.append(m); got.append(ingest.naive(r).strftime("%Y-%m-%dT%H:%MZ"))
        report.append({"id": key, "label": cfg["label"], "ok": bool(got), "runs": got, "family": cfg["family"]})
    for f in cache.glob("*.npz"):
        if f.name not in keep:
            f.unlink(missing_ok=True)
    ingest.fill_gaps(members)
    return members, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-radar", action="store_true")
    ap.add_argument("--no-ens", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    s = run_cycle(Path(a.state), Path(a.out), with_radar=not a.no_radar, with_ens=not a.no_ens)
    print(json.dumps({"generated": s["generated"], "timing_s": s["timing_s"], "notes": s["notes"],
                      "sources": [{k: v for k, v in x.items() if k in ("id", "ok", "runs", "n", "error", "method", "members")} for x in s["sources"]]},
                     indent=1, ensure_ascii=False))
