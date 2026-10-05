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
        start = start - (int(str(start)[11:13]) % 3) * one     # round down: overlap "Ahora" by 0-2 h, never a hole
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
    # (K, NY, NX) mm, telescoped: the sum of the last n hours of o_tail (n <= 12) is the largest n-hour total of
    # ONE 1-km pixel of the cell, the quantity the 12-h thresholds are about (the sum of o_max adds the maxima
    # of different pixels). Increments, not hourly intensities. Hours older than 12 h carry o_max.
    o_tail: np.ndarray | None = None
    fine_t: np.ndarray | None = None        # (F,) datetime64[h] hours that have a 1-km field
    fine: np.ndarray | None = None          # (F, 340, 320) RAW 1-km accumulations of the last hours, NaN = no coverage
    # sub-hourly bursts (only when params hydro.phi_sub_mmh is set): RAW excess ladder of each hour (radar/qpe.py::burst_ladder),
    # (B, L, NY, NX) mm on its own hour axis burst_t, and the ladder of the hour under way (it adds to the nowcast's)
    burst_t: np.ndarray | None = None
    burst: np.ndarray | None = None
    partial_burst: np.ndarray | None = None


def load_obs_state(state: Path) -> Obs:
    f = state / "obs_raw.npz"        # RAW radar accumulations: the gauge correction is never stored here
    e = np.zeros((0, grid.NY, grid.NX), np.float32)
    obs = Obs(np.array([], "datetime64[h]"), e, e.copy())
    if f.exists():
        z = np.load(f)
        obs = Obs(z["t_end"], z["o_max"].astype(np.float32), z["o_mean"].astype(np.float32))
    f = state / "obs_1km.npz"
    if f.exists():
        z = np.load(f)
        obs.fine_t, obs.fine = z["t_end"], z["acc"].astype(np.float32)
    f = state / "obs_burst.npz"
    if f.exists():
        z = np.load(f)
        obs.burst_t, obs.burst = z["t_end"], z["ladder"].astype(np.float32)
    return obs


def save_obs_state(state: Path, obs: Obs, keep_h: int = 96, keep_fine: int = 13, keep_burst: int = 24) -> None:
    state.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state / "obs_raw.npz", t_end=obs.t_end[-keep_h:], o_max=obs.o_max[-keep_h:].astype(np.float16),
                        o_mean=obs.o_mean[-keep_h:].astype(np.float16))
    if obs.fine is not None:
        np.savez_compressed(state / "obs_1km.npz", t_end=obs.fine_t[-keep_fine:], acc=obs.fine[-keep_fine:].astype(np.float16))
    if obs.burst is not None and len(obs.burst_t):
        np.savez_compressed(state / "obs_burst.npz", t_end=obs.burst_t[-keep_burst:], ladder=obs.burst[-keep_burst:].astype(np.float16))


def update_obs(state: Path, now: datetime, backfill_h: int = 14) -> tuple[Obs, list, dict]:
    """Radar-gauge analysis of the last hours. Returns (obs, rate frames, report).

    The state keeps RAW radar accumulations (cell maximum and mean of every hour, the 1-km fields of the last
    13 h); the gauge correction is redone on every cycle from them (`radar/qpe.py::analyse`), never stored on
    top of itself. Scan times are moved to ground-arrival times (`qpe.FALL_MIN`), also in the frames returned.
    Radar down: the gauges alone are kriged. Gauges down: radar with the climatological bias. Both are said
    in the report, which becomes the `sources` entries of the snapshot."""
    from .radar import nowcast as NC, qpe
    from .sources import gauges as G, radar as R
    rep = {"radar": {"ok": False}, "gauges": {"ok": False}}
    obs = load_obs_state(state)
    hnow = top_of_hour(now)
    key = lambda t: str(np.datetime64(t, "h"))
    fine = {} if obs.fine is None else {str(t): f for t, f in zip(obs.fine_t, obs.fine)}
    need = [hnow - timedelta(hours=k) for k in range(backfill_h - 1, -1, -1)]
    need = [t for t in need if key(t) not in fine]
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
        lag = timedelta(minutes=qpe.FALL_MIN)
        rates = [(t + lag, qpe.rain_rate(qpe.despeckle(d))) for t, d in frames]
        src = getattr(R, "last_source", None)
        rep["radar"] = {"ok": True, "source": src, "frames": len(rates), "last": S.iso(frames[-1][0]) if frames else None,
                        "attribution": getattr(R, "ATTRIBUTION", {}).get(src)}
    except Exception as e:
        rates = []
        rep["radar"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    # hours missing from the state, and the last complete hour again: its newest scans arrive with the
    # OPERA hole on the coast still open and are filled 15-25 min later
    new = []
    want_burst = P.load()["hydro"].get("phi_sub_mmh") is not None        # sub-hourly bursts: only when the option is on
    burst = {} if obs.burst is None else {str(t): x for t, x in zip(obs.burst_t, obs.burst)}
    for t in need + [hnow] * (hnow not in need):
        a, b = (t - timedelta(hours=1)).replace(tzinfo=UTC), t.replace(tzinfo=UTC)
        sub = [f for f in rates if a - timedelta(minutes=10) <= f[0] <= b]
        if len(sub) < 2:
            continue
        acc, cov = qpe.accumulate(sub, a, b)
        if cov < (0.99 if key(t) in fine else 0.6):
            continue
        fine[key(t)] = np.minimum(acc / cov, HARD_CAP_1H)       # a partly covered hour is scaled to the whole hour
        new.append(key(t))
        if want_burst:
            try:
                lad, c = qpe.burst_ladder(sub, a, b)
                burst[key(t)] = lad / max(c, 1e-6)
            except Exception as e:      # the bursts are an extra: never lose the cycle for them
                rep["radar"]["burst_error"] = f"{type(e).__name__}: {e}"[:200]
    raw = {str(t): (m, a) for t, m, a in zip(obs.t_end, obs.o_max, obs.o_mean)}
    for k in new:
        raw[k] = (NC.to_analysis_max(np.nan_to_num(fine[k], nan=0.0)), np.nan_to_num(NC.to_analysis_mean(fine[k]), nan=0.0))
    stack = lambda rows: np.stack(rows).astype(np.float32) if rows else np.zeros((0, grid.NY, grid.NX), np.float32)
    ks, fk = sorted(raw), sorted(fine)[-13:]
    obs = Obs(np.array(ks, "datetime64[h]"), stack([raw[k][0] for k in ks]), stack([raw[k][1] for k in ks]))
    if fk:
        obs.fine_t, obs.fine = np.array(fk, "datetime64[h]"), np.stack([fine[k] for k in fk])
    if want_burst and burst:
        bk = sorted(burst)[-24:]
        obs.burst_t, obs.burst = np.array(bk, "datetime64[h]"), np.stack([burst[k] for k in bk]).astype(np.float32)
    save_obs_state(state, obs)
    # the hour under way
    part = part_end = part_burst = None
    a = hnow.replace(tzinfo=UTC)
    sub = [f for f in rates if f[0] >= a - timedelta(minutes=10)]
    if len(sub) >= 2 and sub[-1][0] > a:
        part, part_end = qpe.accumulate(sub, a, sub[-1][0])[0], ingest.naive(sub[-1][0])
        if want_burst:
            try:
                part_burst = qpe.burst_ladder(sub, a, sub[-1][0])[0]
            except Exception as e:
                rep["radar"]["burst_error"] = f"{type(e).__name__}: {e}"[:200]
    gl = []
    try:
        gl = G.fetch_rain_gauges()
        rep["gauges"] = {"ok": bool(gl), "n": len(gl), "sources": sorted({g["source"] for g in gl}),
                         "errors": dict(G.last_errors) or None}
    except Exception as e:
        rep["gauges"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    # Everything below is a corrected COPY of the raw fields: correcting the stored fields would correct them
    # again on every cycle (it did: 953 mm/h near Millares on 2026-10-01).
    hours = [t for t in (hnow - timedelta(hours=k) for k in range(11, -1, -1)) if key(t) in fine]
    bounds = [(t - timedelta(hours=1), t) for t in hours] + ([(hnow, part_end)] if part is not None else [])
    adj = None
    try:
        if bounds and bounds[-1][1] >= ingest.naive(now) - timedelta(minutes=100):
            adj, info = qpe.analyse(np.stack([fine[key(t)] for t in hours] + ([part] if part is not None else [])), bounds, gl)
            if not rep["gauges"]["ok"] or info["n"] < 8:
                rep["gauges"]["fallback"] = info["method"]
        else:                                       # no radar for the last hour and a half
            hours, part = [hnow - timedelta(hours=k) for k in range(11, -1, -1)], None
            adj, info = qpe.analyse_gauges(gl, hnow)
            rep["radar"].update(ok=False, fallback=info["method"])
        rep["gauges"]["merge"] = info
    except Exception as e:
        rep["gauges"]["merge"] = {"method": "failed", "error": f"{type(e).__name__}: {e}"[:200]}
    # corrected hours: the last 12 from this cycle, older ones as they were last corrected (state/obs_adj.npz)
    cor, tail = {}, {}
    if (state / "obs_adj.npz").exists():
        z = np.load(state / "obs_adj.npz")
        cor = {str(t): (m, a) for t, m, a in zip(z["t_end"], z["o_max"].astype(np.float32), z["o_mean"].astype(np.float32))}
    out = Obs(obs.t_end, obs.o_max, obs.o_mean, info=rep, burst_t=obs.burst_t, burst=obs.burst, partial_burst=part_burst)
    if adj is not None:
        adj = np.minimum(adj, HARD_CAP_1H)
        if part is not None:
            out.partial_max, out.partial_mean = NC.to_analysis_max(adj[-1]), NC.to_analysis_mean(adj[-1])
            out.partial_min = (part_end - hnow).total_seconds() / 60.0
        cum = np.zeros(adj.shape[1:], np.float32)
        top = np.zeros((grid.NY, grid.NX), np.float32)
        for k in range(len(hours) - 1, -1, -1):             # newest hour first: telescoped pixel maxima
            cum = cum + adj[k]
            nxt = NC.to_analysis_max(cum)
            cor[key(hours[k])], tail[key(hours[k])] = (NC.to_analysis_max(adj[k]), NC.to_analysis_mean(adj[k])), nxt - top
            top = nxt
    ks = sorted(set(raw) | set(cor))[-96:]
    pick = lambda k, i: cor[k][i] if k in cor else raw[k][i] * qpe.CLIM_FACTOR
    out.t_end, out.o_max, out.o_mean = np.array(ks, "datetime64[h]"), stack([pick(k, 0) for k in ks]), stack([pick(k, 1) for k in ks])
    out.o_tail = stack([tail.get(k, pick(k, 0)) for k in ks])
    np.savez_compressed(state / "obs_adj.npz", t_end=out.t_end, o_max=out.o_max.astype(np.float16), o_mean=out.o_mean.astype(np.float16))
    return out, rates, {"report": rep, "gauges": gl}


HARD_CAP_1H = 200.0     # mm in one hour: above the Spanish record (184.6 mm, Turís, 2024); beyond it is hail or clutter


def with_past(m: risk.Member, obs: Obs, hnow: datetime, bridge: risk.Member | None = None) -> risk.Member:
    """Replace everything up to the current hour with what was observed (and prepend it)."""
    h = np.datetime64(hnow, "h")
    fut = m.t_end > h
    past = obs.t_end <= h
    if not past.any():
        return m
    t_f, p_f, a_f = m.t_end[fut], m.p[fut], m.area[fut]
    one = np.timedelta64(1, "h")
    if len(t_f) and t_f[0] > h + one:
        # the member starts later than the next hour (ENS). Routing, soil memory and rolling windows work
        # by index: the hours in between must exist. They take the bridge run (newest IFS) or zero.
        gap = np.arange(h + one, t_f[0])
        g = np.zeros((len(gap), grid.NY, grid.NX), np.float32)
        if bridge is not None:
            pos = np.searchsorted(bridge.t_end, gap)
            ok = (pos < len(bridge.t_end)) & (bridge.t_end[np.minimum(pos, len(bridge.t_end) - 1)] == gap)
            g[ok] = np.nan_to_num(bridge.p[pos[ok]])
        t_f, p_f, a_f = np.concatenate([gap, t_f]), np.concatenate([g, p_f]), np.concatenate([g, a_f])
    t = np.concatenate([obs.t_end[past], t_f])
    # o_tail: increments whose sum over the last n hours is the largest n-hour total of ONE pixel
    # (summing hourly cell maxima of different pixels overstates the 12-h amount by 13 %)
    p = np.concatenate([(obs.o_max if getattr(obs, "o_tail", None) is None else obs.o_tail)[past], p_f]).astype(np.float32)
    area = np.concatenate([obs.o_mean[past], a_f]).astype(np.float32)
    meta = {**m.meta, "obs_until": h}
    if getattr(obs, "burst", None) is not None:
        meta["burst_obs"] = (obs.burst_t, obs.burst)        # a reference, shared by all members (core/hydro.py::burst_excess)
    return risk.Member(m.name, m.family, m.model, m.run, t, p, m.native_step_h, m.weight, area, meta)


# --------------------------------------------------------------------------------- now members

def nowcast_members(rates: list, obs: Obs, nwp: list[risk.Member], now: datetime, n_members: int = 20):
    """STEPS radar ensemble blended into the newest convection-permitting runs."""
    from .radar import nowcast as NC, qpe
    hnow = top_of_hour(now)
    rep = {"method": "none", "members": 0}
    steps_burst = None
    # donors: runs that cover the six hours; convection-permitting ones if there are any
    full = [m for m in nwp if all((m.t_end == np.datetime64(hnow, "h") + np.timedelta64(k + 1, "h")).any() for k in range(6))]
    cp = sorted([m for m in full if m.family == "cp"], key=lambda m: -m.run.timestamp())
    donors = cp or sorted(full, key=lambda m: -m.run.timestamp())
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
                if P.load()["hydro"].get("phi_sub_mmh") is not None:
                    steps_burst = _steps_burst(nc["rate"], nc["t0"], steps_h_mean)
        except Exception as e:
            rep.update(method=f"failed: {type(e).__name__}: {e}"[:160])
    M = n_members if steps_h_max else 0
    for k in range(M):
        d = donors[k % len(donors)] if donors else None
        p = np.zeros((6, grid.NY, grid.NX), np.float32)
        a = np.zeros_like(p)
        lad = np.full((6, len(qpe.LADDER_U), grid.NY, grid.NX), np.nan, np.float32) if steps_burst else None
        wb = np.zeros(6)
        for i, t in enumerate(hours):
            lead = (i + 1) - lead_off                       # hours after the last scan, end of this hour
            w = float(NC.blend_weights(np.array([max(lead - 0.5, 0.0)]))[0])
            if t not in steps_h_max:
                w = 0.0
            if steps_burst and t in steps_burst:            # the excess ladder of the STEPS member (+ the hour under way)
                lad[i] = steps_burst[t][k] + (obs.partial_burst if (i == 0 and obs.partial_burst is not None) else 0.0)
                wb[i] = w
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
        # A radar member IS its model run seen through the radar: pure radar at +1 h, the run itself once
        # the radar signal has faded. It takes the run's family, age and weight, shared among the members
        # blended into the same run, and the run is not counted a second time (see run_cycle). The weight of
        # every source is then the same in the six frames: nothing vanishes at +4 h.
        bmeta = {"burst_nc": (hours, lad, wb)} if steps_burst else {}
        if d is None:
            out.append(risk.Member(f"Radar STEPS m{k + 1:02d}", "radar", "steps", hnow, hours, p, 1, 1.0, a, {"donor": None, **bmeta}))
        else:
            copies = sum(1 for j in range(M) if donors[j % len(donors)] is d)
            out.append(risk.Member(f"Radar STEPS m{k + 1:02d} → {d.name}", d.family, d.model, d.run, hours, p, 1,
                                   d.weight / copies, a, {"donor": d.name, "radar": True, **bmeta}))
    rep["members"] = len(out)
    rep["donors"] = sorted({m.meta["donor"] for m in out if m.meta.get("donor")})
    return out, rep


def _steps_burst(rate2: np.ndarray, t0, steps_h_mean: dict) -> dict:
    """Excess ladder of every STEPS member and clock hour: {hour: (M, L, NY, NX)} mm. rate2 (M, n, ny2, nx2) are the
    10-min rates on the 2-km grid; each step counts as constant over its 10 min, as in NC.hourly_from_steps.
    Rung 0 is the hourly accumulation itself (steps_h_mean)."""
    from .radar import nowcast as NC, qpe
    out = {t: [steps_h_mean[t]] for t in steps_h_mean}
    for u in qpe.LADDER_U[1:]:
        t_ends, acc = NC.hourly_from_steps(np.maximum(rate2 - u, 0.0), t0)
        for k, t in enumerate(t_ends):
            h = np.datetime64(ingest.naive(t), "h")
            if h in out:
                out[h].append(NC.to_analysis_mean(np.repeat(np.repeat(acc[:, k], 2, axis=1), 2, axis=2)))
    return {t: np.stack(v, axis=1).astype(np.float32) for t, v in out.items() if len(v) == len(qpe.LADDER_U)}


# ------------------------------------------------------------------------------- ENS members

def _with_timeout(fn, seconds: float):
    """Run fn in a thread; give up (the download keeps its partial cache for the next cycle)."""
    import concurrent.futures as cf
    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=seconds)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def ens_npz_members(z, run: datetime) -> list[risk.Member]:
    """ECMWF ENS file (tp since init at 12-hourly steps, cropped to the box) -> one Member per ensemble member.
    12-h accumulations are spread evenly over their hours (native_step_h = 12: no hourly information)."""
    data, steps = z["data"], z["steps"].astype(int)             # (M, S, lat, lon) mm since init
    rg = ingest.regridder("ens025", z["lat"], z["lon"])
    members = []
    for k in range(data.shape[0]):
        t_list, p_list = [], []
        for s in range(1, len(steps)):
            dt_h = int(steps[s] - steps[s - 1])
            inc = np.nan_to_num(np.maximum(rg(data[k, s] - data[k, s - 1]), 0.0)) / dt_h
            for h in range(dt_h):
                t_list.append(np.datetime64(run + timedelta(hours=int(steps[s - 1]) + h + 1), "h"))
                p_list.append(inc)
        members.append(risk.Member(f"ENS m{int(z['members'][k]):02d} · {run:%d/%m %H}Z", "ens", "ifs_ens", run,
                                   np.array(t_list), np.stack(p_list).astype(np.float32), 12))
    return members


def ens_members(state: Path, now: datetime) -> tuple[list[risk.Member], dict]:
    """ECMWF ENS (50 members, 0.25 deg) straight from ECMWF open data: 12-h accumulations."""
    from .sources import ecmwf_open as E
    rep = {"id": "ifs_ens", "label": "ECMWF ENS 50 miembros", "ok": False, "family": "ens"}
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
                # 12-hourly steps covering the long-range frames (UTC days +2 .. +7) and the 12 h before
                day0 = n.replace(hour=0, minute=0, second=0, microsecond=0)
                lo = int(((day0 + timedelta(days=2) - timedelta(hours=12)) - run).total_seconds() // 3600)
                hi = int(((day0 + timedelta(days=9)) - run).total_seconds() // 3600)   # one day more: the file is used past midnight
                steps = [s for s in range(lo - lo % 12, hi + 1, 12) if 0 < s <= 360]
                out.parent.mkdir(parents=True, exist_ok=True)
                _with_timeout(lambda: E.fetch_run(run, steps, "tp", out, mirrors=["gcs", "ecmwf", "aws"],
                                                  workers=4, verbose=False), 420)
            z = np.load(out, allow_pickle=True)
        except Exception as e:
            rep["error"] = f"{run:%d/%m %H}Z: {type(e).__name__}: {e}"[:200]
            if isinstance(e, TimeoutError):
                # the download resumes in the next cycle; meanwhile keep using the newest complete file
                done = sorted(p for p in (state / "ens").glob("ifsens_tp_*.npz") if ".tmp" not in p.name)
                if done:
                    prev = datetime.strptime(done[-1].stem.split("_")[-1], "%Y%m%d%H")
                    members = ens_npz_members(np.load(done[-1], allow_pickle=True), prev)
                    rep.update(ok=True, runs=[prev.strftime("%Y-%m-%dT%H:%MZ")], n=len(members), stale=True)
                    return members, rep
                break
            continue
        members = ens_npz_members(z, run)
        rep.update(ok=True, runs=[run.strftime("%Y-%m-%dT%H:%MZ")], n=len(members))
        for old in (state / "ens").glob("ifsens_tp_*.npz"):
            if old != out:
                old.unlink(missing_ok=True)
        return members, rep
    return [], rep


# ------------------------------------------------------------------------------- one horizon

def horizon_product(hz: str, members: list[risk.Member], now: datetime, params: dict, st, thr, bs, net,
                    sample_mask=None, dams=None) -> dict:
    """dams: (DamNet, live rows) — reservoirs are simulated first and their outflow feeds the points below."""
    hnow = top_of_hour(now)
    frames = frames_for(hz, now)
    pred = risk.predictors(members, frames, params, hz, hnow, sample_mask=sample_mask, keep_members=True)
    F = len(frames)
    fitted = bool(params.get("calibration", {}).get(hz))
    cals = [calibration(params, hz, lead_class(hz, now, f)) for f in frames] if fitted else [None] * F
    if fitted:
        prob, p1, p12 = risk.calibrated_probabilities(pred, cals, thr)
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
    elif pred.a1 is not None:
        prob, p1, p12, e1, e12 = risk.dressed_probabilities(pred, thr, params, hz)
    else:
        z = np.zeros((4, F, grid.NY, grid.NX), np.float32)
        prob, p1, p12, e1, e12 = z, z.copy(), z.copy(), z[:2].copy(), z[:2].copy()
    level = risk.decide(prob, params["tau"][hz])
    cap = params.get("level_cap", {}).get(hz)
    if cap:
        level = np.minimum(level, cap)
    level[~pred.valid] = 0
    acc, acc_total = frame_accumulation(members, frames, params, hz, hnow)
    out = dict(frames=frames, pred=pred, prob=prob, p1=p1, p12=p12, level=level, e1=e1, e12=e12, cals=cals,
               acc=acc, acc_total=acc_total)
    out["basins"] = B.basin_product(members, frames, bs, params, hz, hnow)
    if net is not None:
        t_axis = np.unique(np.concatenate([m.t_end for m in members])) if members else np.array([], "datetime64[h]")
        lo = np.datetime64(hnow, "h") - np.timedelta64(12, "h")
        t_axis = t_axis[(t_axis > lo) & (t_axis <= frames[-1][1])]
        extra = None
        if dams is not None:
            try:
                from .core import reservoirs as RS
                out["reservoirs"] = RS.reservoir_product(members, frames, dams[0], dams[1], params, hz, hnow, t_axis, keep_sim=True)
                extra = (RS.outflow_to_points(out["reservoirs"].sim, dams[0], net.ids), out["reservoirs"].sim["t"])
            except Exception:
                log.error("reservoirs %s failed: %s", hz, traceback.format_exc())
        out["points"] = H.hydro_product(members, frames, net, params, hz, hnow, t_axis, extra=extra)
    return out


def frame_accumulation(members, frames, params: dict, hz: str, now: datetime):
    """Rain expected AT the cell (no neighbourhood): weighted median and 90th percentile over the
    scenarios of the accumulation inside each frame, (2, F, NY, NX) mm, and over the whole period
    covered by the frames, (2, NY, NX). The plain "how much will it rain here"."""
    F = len(frames)
    out = np.zeros((2, F, grid.NY, grid.NX), np.float32)
    total = np.zeros((2, grid.NY, grid.NX), np.float32)
    spans = list(frames) + [(frames[0][0], frames[-1][1])]
    for f, (t0, t1) in enumerate(spans):
        need = int((t1 - t0) / np.timedelta64(1, "h"))
        vals, ws = [], []
        for m in members:
            sel = (m.t_end > t0) & (m.t_end <= t1)
            if sel.sum() < max(1, int(0.75 * need)):
                continue
            w = risk.member_weight(m, params, hz, now)
            if w <= 0:
                continue
            vals.append(m.area[sel].sum(axis=0)); ws.append(w)
        if vals:
            q = risk.weighted_quantile(np.stack(vals), np.asarray(ws, np.float64).reshape(-1, 1, 1), (0.5, 0.9))
            if f < F:
                out[:, f] = q
            else:
                total = q
    return out, total


def _mm_na(a: np.ndarray) -> np.ndarray:
    """Amounts with a not-available code: 255 = n/a, real amounts stop at 254 (1008 mm)."""
    c = np.minimum(S.code_mm(a), 254)
    c[~np.isfinite(a)] = 255
    return c


def pack_horizon(hz: str, o: dict, mask: np.ndarray, params: dict) -> tuple[dict, bytes, dict]:
    m = mask.ravel()
    cells = lambda a: a.reshape(*a.shape[:-2], -1)[..., m]
    pred = o["pred"]
    # measured mm inside the 12-h amount of each frame (largest over the scenarios)
    if pred.obs12 is None or pred.a12 is None:
        o12 = np.zeros((len(o["frames"]), grid.NY, grid.NX), np.float32)
    else:
        s12 = np.array([params["families"][a["family"]]["s12h"] for a in pred.audit], np.float32)[:, None, None, None]
        o12 = np.max(pred.obs12 * np.nan_to_num(pred.a12) * s12, axis=0)
    blk = {
        "frames": [{"t0": S.iso(t0), "t1": S.iso(t1), "ok": bool(pred.valid[f]), "has_1h": bool(pred.has_1h[f]),
                    "method": "emos" if o["cals"][f] is not None else "dressing",
                    "cal": o["cals"][f].to_dict() if o["cals"][f] is not None else None}
                   for f, (t0, t1) in enumerate(o["frames"])],
        "sigma": params["sigma"][hz], "bias": params["bias"][hz], "sigma_obs": params.get("sigma_obs", 0.15),
        "obs12_factor": params.get("obs12_factor", 1.0),
        "sigma1h": params.get("sigma1h", {}).get(hz, params["sigma"][hz]),
        "bias1h": params.get("bias1h", {}).get(hz, params["bias"][hz]),
        "level_cap": params.get("level_cap", {}).get(hz),
        "fam": {k: {"s1h": v["s1h"], "s12h": v["s12h"]} for k, v in params["families"].items()},
        "tau": params["tau"][hz],
        "cells": {"level": S.b64(cells(o["level"])), "p": S.b64(S.code_prob(cells(o["prob"]))),
                  "e1": S.b64(S.code_mm(cells(o["e1"]))), "e12": S.b64(S.code_mm(cells(o["e12"]))),
                  "acc": S.b64(S.code_mm(cells(o["acc"]))),
                  "acc_total": S.b64(S.code_mm(cells(o["acc_total"]))), "o12": S.b64(S.code_mm(cells(o12))),
                  "m1": S.b64(_mm_na(cells(pred.m1))), "q1": S.b64(_mm_na(cells(pred.q1))),
                  "m12": S.b64(_mm_na(cells(pred.m12))), "q12": S.b64(_mm_na(cells(pred.q12)))},
        "members": pred.audit,
    }
    bp = o["basins"]
    blk["basins"] = {"level": S.r(bp.level, 0), "p": S.r(bp.prob, 3), "own12": S.r(bp.own12, 0),
                     "up12": S.r(bp.up12, 0), "q": S.r(bp.q, 2), "upstream": S.r(bp.upstream_driven, 2)}
    if "points" in o:
        hp = o["points"]
        blk["points"] = {"level": S.r(hp.level, 0), "p": S.r(hp.prob, 3), "qpeak": S.r(hp.q_peak, 0),
                         "hover": S.r(hp.h_over, 2), "t": [S.iso(t) for t in hp.t_end], "q": S.r(hp.q_series, 0),
                         "rp": S.r(hp.rp, 0) if hp.rp is not None else None}
    # audit binary: per-member amounts and the two marginal probabilities
    M = 0 if pred.a1 is None else pred.a1.shape[0]
    has1 = [bool(np.isfinite(pred.a1[k]).any()) for k in range(M)]      # scenarios with hourly information
    if M:
        a1 = _mm_na(cells(pred.a1[np.array(has1)])) if any(has1) else np.zeros((0,), np.uint8)
        a12 = _mm_na(cells(pred.a12))
        blob = a1.tobytes() + a12.tobytes() + S.code_prob(cells(o["p1"])).tobytes() + S.code_prob(cells(o["p12"])).tobytes()
    else:
        blob = b""
    head = {"M": M, "M1": int(sum(has1)), "has1": has1, "F": len(o["frames"]), "N": int(m.sum()),
            "layout": "uint8 a1[M1,F,N] (only the scenarios with has1, in order; 255 = not available), a12[M,F,N], "
                      "p1[4,F,N], p12[4,F,N]; mm = (v/8)^2, P = v/200"}
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

    # a real ensemble for 6-48 h (ECMWF ENS 3-hourly, ICON-EU-EPS), AROME driven by IFS, IFS at 9 km,
    # and the hourly AROME-PI for the first hours; together the 90 ensemble members weigh about a third
    t = time.time()
    extra = []
    try:
        from .sources import extra_models as X
        extra, xrep = X.extra_members(state / "extra", now, ("ens3h", "icon_eu_eps", "arome_ifs", "ifs_hres", "arome_pi"),
                                      budget_s=45)
        sources.extend(xrep)
        for m in extra:
            m.weight *= {"ifs_ens3h": 0.6, "icon_eu_eps": 0.6, "arome_ifs": 0.7}.get(m.model, 1.0)
        ingest.fill_gaps(nwp + extra)                 # AROME-IFS and AROME-PI have no data south of ~38 N
    except Exception as e:
        log.error("extra models failed: %s", traceback.format_exc())
        notes.append(f"extra models: {type(e).__name__}: {e}"[:200])
        extra = [m for m in extra if not np.isnan(m.p).any()]
    pi = [m for m in extra if m.model == "arome_pi"]
    mid_extra = [m for m in extra if m.model != "arome_pi"]
    has_hres = any(m.model == "ifs_hres" for m in mid_extra)
    timing["extra"] = round(time.time() - t, 1)

    t = time.time()
    ens, erep = ens_members(state, now) if with_ens else ([], {"id": "ifs_ens", "ok": False, "label": "ECMWF ENS"})
    sources.append(erep)
    timing["ens"] = round(time.time() - t, 1)

    t = time.time()
    radar_m, nrep = nowcast_members(rates, obs, nwp + pi, now) if with_radar else ([], {"method": "off"})
    sources.append({"id": "nowcast", "label": "Nowcast radar pysteps STEPS", "ok": bool(radar_m), **nrep})
    timing["nowcast"] = round(time.time() - t, 1)

    glob_ = sorted([m for m in nwp if m.family == "global"], key=lambda m: -m.run.timestamp())
    # members that start later than the next hour (the ensembles) take the newest IFS run in between
    past = lambda ms: [with_past(m, obs, hnow, bridge=glob_[0] if glob_ else None) for m in ms]
    jj, ii = np.mgrid[0:grid.NY, 0:grid.NX]
    lattice = (jj % 2 == 0) & (ii % 2 == 0)
    blended = set(nrep.get("donors") or [])        # runs already present as radar-blended members
    horizon_members = {
        "now": past(radar_m + [m for m in nwp + pi if m.name not in blended]),
        "mid": past([m for m in nwp if not (has_hres and m.model == "ifs")] + mid_extra),
        "long": past(ens + glob_),
    }
    snap = {"v": S.SNAPSHOT_VERSION, "generated": S.iso(now), "params_version": params.get("version"),
            "grid": {"lon0": grid.LON0, "lat0": grid.LAT0, "d": grid.D, "nx": grid.NX, "ny": grid.NY},
            "mask": S.b64(np.packbits(st.mask.ravel())), "n_cells": st.n_cells,
            "thresholds": {"zones": st.zone_thr, "extreme": params["extreme"], "source": st.thresholds_source},
            "horizons": {}, "explain": {}}
    out.mkdir(parents=True, exist_ok=True)
    # reservoirs: live state first, so that each horizon can route what the dams let through
    dams, rprod, rrep = None, {}, None
    try:
        from .core import reservoirs as RS
        from .sources import reservoirs as RL
        dn = RS.load_dams(base_alpha=net.alpha if net is not None else None)
        if dn is not None:
            live, rrep = RL.fetch_reservoirs(dn.meta, state=state, now=now)
            dams = (dn, live)
    except Exception as e:
        log.error("reservoirs failed: %s", traceback.format_exc())
        notes.append(f"reservoirs: {type(e).__name__}: {e}"[:200])
    for hz in P.HORIZONS:
        t = time.time()
        if hz == "long" and not ens:
            notes.append("long: ECMWF ENS not available, horizon not computed")
            continue
        try:
            o = horizon_product(hz, horizon_members[hz], now, params, st, thr, bs, net,
                                sample_mask=lattice if hz == "mid" else None, dams=dams)
            if "reservoirs" in o:
                rprod[hz] = o["reservoirs"]
            blk, blob, head = pack_horizon(hz, o, st.mask, params)
            snap["horizons"][hz] = blk
            (out / f"explain-{hz}.bin").write_bytes(blob)
            snap["explain"][hz] = head
        except Exception as e:
            log.error("horizon %s failed: %s", hz, traceback.format_exc())
            notes.append(f"{hz}: {type(e).__name__}: {e}"[:300])
        timing[hz] = round(time.time() - t, 1)

    # ---- reservoirs (q10-dams): live state, filling and spill per horizon; a failure never stops the cycle ----
    t = time.time()
    try:
        if dams is not None:
            from .core import reservoirs as RS
            dn, live = dams
            if rrep:
                sources.append(rrep)
            rblk = RS.static_block(dn, live, [c["id"] for c in cps], hnow)
            for hz in snap["horizons"]:
                try:
                    if hz in rprod:
                        rblk["horizons"][hz] = RS.pack(rprod[hz], dn)
                        continue
                    ms, frs = horizon_members[hz], frames_for(hz, now)
                    ta = np.unique(np.concatenate([x.t_end for x in ms])) if ms else np.array([], "datetime64[h]")
                    ta = ta[(ta > np.datetime64(hnow, "h") - np.timedelta64(12, "h")) & (ta <= frs[-1][1])]
                    rblk["horizons"][hz] = RS.pack(RS.reservoir_product(ms, frs, dn, live, params, hz, hnow, ta), dn)
                except Exception as e:
                    log.error("reservoirs %s failed: %s", hz, traceback.format_exc())
                    notes.append(f"reservoirs {hz}: {type(e).__name__}: {e}"[:200])
            snap["reservoirs"] = rblk
    except Exception as e:
        log.error("reservoirs failed: %s", traceback.format_exc())
        notes.append(f"reservoirs: {type(e).__name__}: {e}"[:200])
    timing["reservoirs"] = round(time.time() - t, 1)
    # ---- end reservoirs ----

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
                                z["p"].astype(np.float32), int(z["step"]), float(cfg.get("weight", 1.0)),
                                meta={"nan_frac": float(z["nan_frac"])})
            else:
                ahead = 66 if key != "ifs" else 24 * 8      # the run is cached: it must still reach the last frame hours later
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
