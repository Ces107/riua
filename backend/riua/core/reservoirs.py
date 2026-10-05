"""Reservoirs: inflow, filling and spill, for every rainfall scenario.

Why: the control points treat everything above a large dam as retained. That is right while the
dam has room and wrong the moment it spills (Forata, 29 Oct 2024: full in a few hours, more than
1000 m3/s over the spillway into a Magro that the model thought was only fed from below the dam).

Chain, for every scenario (and its shifted copies, as at the control points):

1. Inflow. The catchment of each dam, cut from the same terrain model as the control points and
   stopped at the dams above it, has its own travel-time bands. Rain -> net rain with the production
   losses (hydro.net_rain) -> time-area routing (hydro.route) gives the inflow from its own catchment.
   The quick-runoff share alpha of each dam catchment is fitted to the inflow volumes the reservoir
   itself measured (geo/hydro/dams/check_events.py).
2. What is measured now corrects the model: the difference between the measured inflow and the
   simulated one at the current hour is carried forward and fades with the response time of the
   catchment (error-persistence updating). Without a measured inflow nothing is corrected.
3. Model error. Each scenario is run K times with its simulated inflow multiplied by the K
   equal-probability quantiles of a log-normal of spread sigma (the same sigma as the control
   points): the spread of ln(discharge) left by the loss model.
4. Storage, hour by hour in four sub-steps, dams in cascade (what an upper dam lets out reaches the
   next one after the travel time between them):
       V(t + dt) = V(t) + (Q_in - Q_out) dt
       Q_out = release + spill
       release = the outflow measured now, held (nobody can know what the operator will decide);
                 for a flood-control dam with an open bottom outlet: capacity * (V / V_s)^0.2
       spill   = 0 below the spill level; above it the spillway rating
                 Q = C (h - h_sill)^1.5,  C = design capacity / design head^1.5
                 with h(V) from the level-volume curve V = a (h - h0)^b fitted to SAIH pairs.
                 A gated spillway (sill below the top of the gates) passes the whole inflow while it
                 fits through: the level is held at the top of the gates and only rises beyond it
                 when the inflow exceeds the capacity.
5. Levels per frame, same rule as everywhere (highest level whose probability reaches tau):
       2  filling fast (gains 10 % of its capacity, or rises into its flood-control reserve),
          or lets out more than the low outflow threshold
       3  reaches the spill level (it spills), or outflow above the middle threshold
       4  outflow to the river above the high threshold (SAIH Júcar "umbral alto"; elsewhere
          3 A^0.6 m3/s, A = catchment km2)
       5  outflow above the design capacity of the spillway, or water at the crest of the dam

What this cannot know: gate and outlet operations decided by people (preventive releases, turbines,
transfers between reservoirs), sediment that has eaten part of the storage, the real rating of a
spillway under debris, a dam failing. Reservoirs fed by canals (La Pedrera, Crevillente) only get
their small natural catchment here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from . import grid, hydro as H
from .risk import Member, decide, member_weight, weighted_quantile, _bw

HM3_PER_M3S_H = 0.0036          # 1 m3/s during one hour = 3600 m3
N_SUB = 4                       # sub-steps per hour of the storage integration
DEFAULTS = {
    "k_nodes": 7,               # inflow multipliers per scenario (equal-probability log-normal quantiles)
    "gain_l2": 0.10,            # level 2: volume gained since now, as a share of the volume at the spill level
    "rise_l2": 0.02,            # ... or inside the flood reserve after rising at least this share
    "thr_a06": (0.6, 1.5, 3.0), # outflow thresholds low / mid / high as multiples of A^0.6 where SAIH gives none
    "outlet_exp": 0.2,          # open bottom outlet: Q = capacity * (V / V_s)^outlet_exp  (sqrt of head, V ~ h^2.5)
    "min_box": 0.5,             # no inflow forecast when less than this share of the catchment is inside the rain grid
    "max_t": 48,                # points of the time axis kept in the snapshot
}


@dataclass
class DamNet:
    ids: list[str]
    meta: list[dict]                 # the entries of dams.json, same order
    net: H.HydroNet                  # own catchments: lags, area, tc_h (= longest travel time)
    order: list[int]                 # upstream dams first
    up: list[list[tuple[int, int]]]  # per dam: (index of a dam directly above, lag in whole hours >= 1)
    v_spill: np.ndarray              # (D,) hm3 at the spill level (lip of a free spillway / top of the gates)
    v_top: np.ndarray                # (D,) hm3 with the water at the crest of the dam (NaN unknown)
    v_res: np.ndarray                # (12, D) hm3 above which the reservoir is inside its flood reserve (NaN none)
    q_design: np.ndarray             # (D,) design capacity of the spillway(s), m3/s (NaN unknown)
    spill_c: np.ndarray              # (D,) C of Q = C H^1.5
    h_sill: np.ndarray               # (D,) spillway sill, in the level scale of the curve
    lv_a: np.ndarray                 # (D,) level-volume curve V = a (h - h0)^b
    lv_b: np.ndarray
    lv_h0: np.ndarray
    outlet: np.ndarray               # (D,) capacity of an OPEN bottom outlet, m3/s (0 = closed / unknown)
    thr: np.ndarray                  # (3, D) outflow thresholds low, mid, high (m3/s)
    box: np.ndarray                  # (D,) share of the own catchment inside the rain grid
    alpha: np.ndarray | None = None  # (NY*NX,) quick-runoff share per cell

    @property
    def n(self) -> int:
        return len(self.ids)

    def level(self, v: np.ndarray) -> np.ndarray:
        """water level for volumes v (..., D)"""
        return self.lv_h0 + (np.maximum(v, 0.0) / self.lv_a) ** (1.0 / self.lv_b)


def load_dams(geo: Path | None = None, base_alpha: np.ndarray | None = None) -> DamNet | None:
    """Reservoir network from geo/hydro/dams (dams.json + out/time_area.npz), or None while it is not built.
    base_alpha: the per-cell quick-runoff share of the control points (static.hydro_net()[0].alpha); cells of a
    dam catchment that has its own fitted alpha take that one instead."""
    if geo is None:
        from .. import static
        geo = static.GEO
    ddir = Path(geo) / "hydro" / "dams"
    if not (ddir / "dams.json").exists() or not (ddir / "out" / "time_area.npz").exists():
        return None
    dams = json.loads((ddir / "dams.json").read_text(encoding="utf-8"))["dams"]
    ta = np.load(ddir / "out" / "time_area.npz", allow_pickle=False)
    pid = [str(x) for x in ta["point_ids"]]
    by = {d["id"]: d for d in dams}
    ids = [i for i in pid if i in by]
    remap = {pid.index(i): k for k, i in enumerate(ids)}
    sel = np.isin(ta["point_idx"], list(remap))
    pidx = np.array([remap[int(x)] for x in ta["point_idx"][sel]], int)
    cell, lag, akm = ta["cell"][sel].astype(int), ta["lag_h"][sel].astype(int), ta["area_km2"][sel].astype(float)
    D = len(ids)
    meta = [by[i] for i in ids]
    nan = [np.nan] * D
    net = H.HydroNet.build(ids, [m["own_km2"] for m in meta], [max(m.get("t_longest_h") or 1.0, 1.0) for m in meta],
                           pidx, cell, lag, akm, nan, [None] * D, [None] * D, nan, None)
    pos = {i: k for k, i in enumerate(ids)}
    up = [[(pos[u["id"]], max(1, int(round(u["lag_h"])))) for u in m.get("upstream", []) if u["id"] in pos] for m in meta]
    order, seen = [], set()

    def visit(k):
        if k in seen:
            return
        seen.add(k)
        for u, _ in up[k]:
            visit(u)
        order.append(k)
    for k in range(D):
        visit(k)
    f = lambda key, default=np.nan: np.array([m.get(key) if m.get(key) is not None else default for m in meta], float)
    v_spill = f("v_spill_hm3")
    sp = [m.get("spill") or {} for m in meta]
    q_design = np.array([s.get("capacity_m3s") or np.nan for s in sp], float)
    head = np.array([s.get("design_head_m") or np.nan for s in sp], float)
    cv = [m.get("curve") or {} for m in meta]
    lv_a = np.array([c.get("a") or np.nan for c in cv], float)
    lv_b = np.array([c.get("b") or np.nan for c in cv], float)
    lv_h0 = np.array([c.get("h0") if c.get("h0") is not None else np.nan for c in cv], float)
    h_spill = lv_h0 + (v_spill / lv_a) ** (1.0 / lv_b)
    h_sill = np.array([s.get("sill_level_m") if s.get("sill_level_m") is not None else np.nan for s in sp], float)
    h_sill = np.where(np.isfinite(h_sill), np.minimum(h_sill, h_spill), h_spill)
    spill_c = q_design / np.maximum(head, 0.1) ** 1.5
    rc = np.array([s.get("rating_c") or np.nan for s in sp], float)
    spill_c = np.where(np.isfinite(rc), rc, spill_c)                  # a rating measured in a real spill wins
    thr = np.array([[(m.get("thr") or [None] * 3)[k] or np.nan for m in meta] for k in range(3)], float)
    a06 = np.maximum(f("area_km2", 1.0), 1.0) ** 0.6
    for k, c in enumerate(DEFAULTS["thr_a06"]):
        thr[k] = np.where(np.isfinite(thr[k]), thr[k], c * a06)
    thr = np.maximum.accumulate(thr, axis=0)
    v_res = np.full((12, D), np.nan)
    for k, m in enumerate(meta):
        for r in m.get("reserve") or []:
            for mo in r["months"]:
                v_res[mo - 1, k] = r["max_hm3"]
    outlet = np.array([(m.get("outlet_m3s") or 0.0) if m.get("outlet_open") else 0.0 for m in meta], float)
    alpha = None
    if base_alpha is not None or any(m.get("alpha") is not None for m in meta):
        n = grid.NY * grid.NX
        alpha = np.full(n, 0.016, np.float32) if base_alpha is None else np.array(base_alpha, np.float32)
        A = np.zeros((D, n))
        np.add.at(A, (pidx, cell), akm)
        top = A.argmax(axis=0)
        for k, m in enumerate(meta):
            if m.get("alpha") is not None:
                alpha[(top == k) & (A[k] > 0)] = m["alpha"]
    return DamNet(ids, meta, net, order, up, v_spill, f("v_top_hm3"), v_res, q_design, spill_c, h_sill, lv_a, lv_b, lv_h0,
                  outlet, thr, f("inside_box_fraction", 1.0), alpha)


def inflow(p: np.ndarray, dn: DamNet, hp: dict) -> np.ndarray:
    """p (T, ncell) mm/h -> inflow from the own catchment of every dam (T, D) m3/s: production losses and routing."""
    return H.route(H.net_rain(p, hp["p0_mm"], hp["wet_memory_h"], phi=hp.get("phi_mmh"), s=hp.get("s_mm"), alpha=dn.alpha,
                              p0b=hp.get("p0b_mm", 10.0), sb=hp.get("sb_mm", 100.0)), dn.net, hp["clark_k"])


def spill_capacity(v: np.ndarray, dn: DamNet) -> np.ndarray:
    """What the spillway can pass with the reservoir at volume v (..., D), m3/s. Unknown rating: no limit."""
    h = dn.level(v) - dn.h_sill
    q = dn.spill_c * np.maximum(h, 0.0) ** 1.5
    return np.where(np.isfinite(q), q, np.inf)


def integrate(q_own: np.ndarray, v0: np.ndarray, release: np.ndarray, q_before: np.ndarray, dn: DamNet,
              base: np.ndarray | None = None, rp: dict = DEFAULTS, river: np.ndarray | None = None):
    """Storage of every dam along time.

    q_own   (S, T, D) m3/s  inflow from the own catchment in each of S scenarios (already multiplied / corrected)
    v0      (D,) hm3        volume at the start
    release (D,) m3/s       outflow held while below the spill level
    q_before (D,) m3/s      outflow to the river of each dam before the start (feeds the dams below during the first lag hours)
    base    (S, T, D) | None  extra inflow (measured base flow and the fading correction)
    river   (D,) | None     part of the release that goes down the river (the rest leaves by canals and penstocks that
                            do not return above the next dam: Arenós lets out 80 m3/s, 5 reach the Mijares); default all
    Returns v (S, T, D) hm3 at the end of each hour, q_in (S, T, D) total inflow, q_out (S, T, D) outflow to the river
    (river release + spill: what the dams and towns below receive), q_spill (S, T, D); hourly means for the flows.
    """
    S, T, D = q_own.shape
    v = np.empty((S, T, D), np.float32)
    q_in = np.zeros((S, T, D), np.float32)
    q_out = np.zeros((S, T, D), np.float32)
    q_sp = np.zeros((S, T, D), np.float32)
    cur = np.broadcast_to(v0.astype(np.float64), (S, D)).copy()
    dt = 1.0 / N_SUB
    k_out = rp["outlet_exp"]
    vs = dn.v_spill
    for t in range(T):
        qi = q_own[:, t].astype(np.float64)
        if base is not None:
            qi = qi + base[:, t]
        for d in range(D):
            for u, lag in dn.up[d]:
                qi[:, d] += q_out[:, t - lag, u] if t - lag >= 0 else q_before[u]      # q_out = what went down the river
        qi = np.maximum(qi, 0.0)
        acc_o = np.zeros((S, D))
        acc_s = np.zeros((S, D))
        riv = release if river is None else np.minimum(river, release)
        for _ in range(N_SUB):
            rel = np.where(dn.outlet > 0, dn.outlet * np.clip(cur / vs, 0.0, 4.0) ** k_out, release)
            rel = np.minimum(rel, cur / (dt * HM3_PER_M3S_H) + qi)                     # an empty reservoir releases what enters
            over = (cur - vs) / (dt * HM3_PER_M3S_H) + qi - rel                        # what must leave to stay at the spill level
            sp = np.clip(over, 0.0, spill_capacity(cur, dn))
            cur = np.maximum(cur + (qi - rel - sp) * dt * HM3_PER_M3S_H, 0.0)
            acc_o += np.minimum(rel, riv) + sp
            acc_s += sp
        v[:, t] = cur
        q_in[:, t] = qi
        q_out[:, t] = acc_o / N_SUB
        q_sp[:, t] = acc_s / N_SUB
    return v, q_in, q_out, q_sp


@dataclass
class ReservoirProduct:
    t_end: np.ndarray        # (T,) hours of the series below (from the current hour on)
    v_series: np.ndarray     # (3, T, D) p10, p50, p90 of the volume, hm3
    qin_series: np.ndarray   # (2, T, D) p50, p90 of the inflow, m3/s
    qout_series: np.ndarray  # (2, T, D) p50, p90 of the outflow, m3/s
    prob: np.ndarray         # (4, F, D) P(level >= 2..5)
    level: np.ndarray        # (F, D)
    p_reserve: np.ndarray    # (F, D) probability of being inside the flood reserve
    p_spill: np.ndarray      # (F, D) probability of spilling inside the frame
    p_design: np.ndarray     # (F, D) probability of exceeding the design capacity of the spillway
    q_in: np.ndarray         # (2, F, D) median / p90 peak inflow of the frame
    q_out: np.ndarray        # (2, F, D) median / p90 peak outflow
    q_spill: np.ndarray      # (2, F, D) median / p90 peak spill
    v_end: np.ndarray        # (3, F, D) p10 / p50 / p90 volume at the end of the frame, hm3
    p_spill_any: np.ndarray  # (D,) probability of spilling at any time of the horizon
    t_spill: np.ndarray      # (D,) datetime64[h] median time of the first spill among the scenarios that spill (NaT none)
    v0: np.ndarray           # (D,) volume used as the start, hm3 (NaN = no live data: not computed)
    valid: np.ndarray        # (F,)
    ok: np.ndarray           # (D,) bool: dam computed
    sim: dict = field(default_factory=dict)   # per-scenario outflow for the control points (see outflow_to_points)


def _state(dn: DamNet, live: list[dict]):
    """volume, total release, release to the river, measured inflow (NaN unknown) per dam, from the live rows"""
    by = {r["id"]: r for r in live or []}
    D = dn.n
    v0, rel, river, qobs = (np.full(D, np.nan) for _ in range(4))
    for k, i in enumerate(dn.ids):
        r = by.get(i)
        if not r or r.get("volume_hm3") is None:
            continue
        v0[k] = r["volume_hm3"]
        out_t, out_r = r.get("outflow_m3s"), r.get("outflow_river_m3s")
        rel[k] = max(out_t if out_t is not None else (out_r or 0.0), 0.0)
        river[k] = max(out_r if out_r is not None else rel[k], 0.0)
        if r.get("inflow_m3s") is not None:
            qobs[k] = max(r["inflow_m3s"], 0.0)
        elif r.get("rate_hm3h") is not None and out_t is not None:
            qobs[k] = max(r["rate_hm3h"] / HM3_PER_M3S_H + out_t, 0.0)
    return v0, rel, river, qobs


def reservoir_product(members: list[Member], frames, dn: DamNet, live: list[dict], params: dict, horizon: str,
                      now: datetime, t_axis: np.ndarray, keep_sim: bool = False) -> ReservoirProduct:
    """Scenario by scenario: inflow, storage, spill; then probabilities and levels per frame.
    members, frames and t_axis are exactly what hydro.hydro_product receives."""
    hp = params["hydro"]
    rp = {**DEFAULTS, **params.get("reservoirs", {})}
    sigma = rp.get("sigma", hp["sigma"])[horizon]
    shift_km = params["basin_shift_km"][horizon]
    dj, di = int(round(shift_km / grid.DY_KM)), int(round(shift_km / grid.DX_KM))
    D, F = dn.n, len(frames)
    hnow = np.datetime64(now, "h")
    t_axis = t_axis[t_axis > hnow - np.timedelta64(1, "h")]              # the current hour and what follows
    i0 = int(np.searchsorted(t_axis, hnow, side="right"))                # first future hour
    t_fut = t_axis[i0:]
    T = len(t_fut)
    v0, rel, river, qobs = _state(dn, live)
    ok = np.isfinite(v0) & np.isfinite(dn.v_spill) & (dn.box >= rp["min_box"])
    month = int(str(hnow)[5:7])
    v_res = dn.v_res[month - 1]

    QI, W, WF, KEY, QNOW = [], [], [], [], []
    for m in members:
        w = member_weight(m, params, horizon, now)
        if w <= 0:
            continue
        fam = params["families"][m.family]
        variants = [(0, 0, 0.4), (dj, 0, 0.15), (-dj, 0, 0.15), (0, di, 0.15), (0, -di, 0.15)] \
            if m.family in ("cp", "radar") and (dj or di) else [(0, 0, 1.0)]
        valid = np.array([((m.t_end > t0) & (m.t_end <= t1)).any() for t0, t1 in frames])
        if not valid.any():
            continue
        pos = np.searchsorted(t_axis, m.t_end)
        inside = (pos < len(t_axis)) & (t_axis[np.minimum(pos, len(t_axis) - 1)] == m.t_end)
        for sj, si, share in variants:
            p = np.nan_to_num(grid.shift(m.area, sj, si), nan=0.0).reshape(len(m.t_end), -1)
            p = p * np.where(m.observed(), 1.0, fam["s12h"])[:, None]      # measured rain is not rescaled
            q = inflow(p, dn, hp)                                           # (Tm, D)
            qa = np.zeros((len(t_axis), D), np.float32)
            qa[pos[inside]] = q[inside]
            QI.append(qa[i0:]); QNOW.append(qa[i0 - 1] if i0 > 0 else np.zeros(D, np.float32))
            W.append(w * share); WF.append(w * share * valid); KEY.append((m.name, sj, si))
    empty = lambda *s: np.zeros(s, np.float32)
    nat = np.full(D, np.datetime64("NaT"), "datetime64[h]")
    if not QI or T == 0 or not ok.any():
        return ReservoirProduct(t_fut, empty(3, T, D), empty(2, T, D), empty(2, T, D), empty(4, F, D), np.zeros((F, D), np.uint8),
                                empty(F, D), empty(F, D), empty(F, D), empty(2, F, D), empty(2, F, D), empty(2, F, D),
                                empty(3, F, D), empty(D), nat, v0, np.zeros(F, bool), np.zeros(D, bool))
    K = int(rp["k_nodes"])
    mult = np.exp(sigma * ndtri((np.arange(K) + 0.5) / K))                  # (K,) equal-probability multipliers
    q_sim = np.stack(QI)                                                     # (MV, T, D)
    q_now = np.stack(QNOW)                                                   # (MV, D)
    MV = q_sim.shape[0]
    q_own = (q_sim[:, None] * mult[None, :, None, None]).reshape(MV * K, T, D)
    # error-persistence updating: measured minus simulated inflow now, fading with the response time; the part of a
    # positive difference that the reservoir is letting out anyway (base flow in balance with the release) stays
    tau = np.maximum(0.5 * dn.net.tc_h, 3.0)
    diff = np.where(np.isfinite(qobs), qobs, 0.0)[None, None, :] - np.where(np.isfinite(qobs), 1.0, 0.0) \
        * q_now[:, None, :] * mult[None, :, None]                            # (MV, K, D)
    stay = np.where(diff > 0, np.minimum(diff, np.nan_to_num(rel)[None, None, :]), 0.0)
    fade = np.exp(-(np.arange(T) + 0.5)[:, None] / tau[None, :])              # (T, D)
    base = (stay[:, :, None, :] + (diff - stay)[:, :, None, :] * fade[None, None]).reshape(MV * K, T, D).astype(np.float32)
    v, q_in, q_riv, q_sp = integrate(q_own, np.nan_to_num(v0), np.nan_to_num(rel), np.nan_to_num(river), dn, base, rp,
                                     river=np.nan_to_num(river))
    w_s = np.repeat(np.asarray(W), K) / K                                    # (S,)
    wf_s = np.repeat(np.stack(WF), K, axis=0) / K                            # (S, F)
    vs = dn.v_spill[None, :]
    gain = rp["gain_l2"] * dn.v_spill
    rise = rp["rise_l2"] * dn.v_spill
    psum = np.zeros((4, F, D)); pres = np.zeros((F, D)); pspill = np.zeros((F, D)); pdes = np.zeros((F, D))
    qin_f = np.zeros((2, F, D), np.float32); qout_f = np.zeros_like(qin_f); qsp_f = np.zeros_like(qin_f)
    v_f = np.zeros((3, F, D), np.float32)
    wsum = np.zeros(F)
    for f, (t0, t1) in enumerate(frames):
        sel = (t_fut > t0) & (t_fut <= t1)
        use = wf_s[:, f] > 0
        if not sel.any() or not use.any():
            continue
        wv = wf_s[use, f]
        wsum[f] = wv.sum()
        vmax, vend = v[use][:, sel].max(axis=1), v[use][:, sel][:, -1]
        qo, qs, qi = q_riv[use][:, sel].max(axis=1), q_sp[use][:, sel].max(axis=1), q_in[use][:, sel].max(axis=1)
        in_res = np.isfinite(v_res)[None, :] & (vmax >= np.nan_to_num(v_res)[None, :])
        spilling = (qs > 0) | (vmax >= vs * 0.9995)
        design = (np.isfinite(dn.q_design)[None, :] & (qs >= np.nan_to_num(dn.q_design)[None, :])) \
            | (np.isfinite(dn.v_top)[None, :] & (vmax >= np.nan_to_num(dn.v_top, nan=np.inf)[None, :]))
        l2 = (vmax - v0[None, :] >= gain[None, :]) | (in_res & (vmax - v0[None, :] >= rise[None, :])) | (qo >= dn.thr[0][None, :])
        l3 = spilling | (qo >= dn.thr[1][None, :])
        l4 = qo >= dn.thr[2][None, :]
        for k, ev in enumerate((l2 | l3 | l4 | design, l3 | l4 | design, l4 | design, design)):
            psum[k, f] = (wv[:, None] * ev).sum(axis=0)
        pres[f], pspill[f], pdes[f] = ((wv[:, None] * e).sum(axis=0) for e in (in_res, spilling, design))
        wb = _bw(wv, (D,))
        qin_f[:, f] = weighted_quantile(qi, wb, (0.5, 0.9))
        qout_f[:, f] = weighted_quantile(qo, wb, (0.5, 0.9))
        qsp_f[:, f] = weighted_quantile(qs, wb, (0.5, 0.9))
        v_f[:, f] = weighted_quantile(vend, wb, (0.1, 0.5, 0.9))
    valid = wsum > 0
    den = np.where(valid, wsum, 1.0)[None, :, None]
    prob = np.minimum.accumulate((psum / den).astype(np.float32), axis=0)
    level = decide(prob, params["tau"][horizon])
    level[~valid] = 0
    level[:, ~ok] = 0
    prob[:, :, ~ok] = 0
    # series: scenarios that do not reach the end keep their last state (inflow 0 after their last hour)
    wb = _bw(w_s, (T, D))
    v_ser = weighted_quantile(v, wb, (0.1, 0.5, 0.9)).astype(np.float32)
    qin_ser = weighted_quantile(q_in, wb, (0.5, 0.9)).astype(np.float32)
    qout_ser = weighted_quantile(q_riv, wb, (0.5, 0.9)).astype(np.float32)
    # first spill
    sp_any = (q_sp > 0) | (v >= vs[None] * 0.9995)                           # (S, T, D)
    ever = sp_any.any(axis=1)
    first = np.where(ever, sp_any.argmax(axis=1), T).astype(np.float64)       # (S, D) index of the first spilling hour
    p_any = (w_s[:, None] * ever).sum(axis=0) / max(w_s.sum(), 1e-12)
    t_spill = nat.copy()
    for d in range(D):
        e = ever[:, d]
        if ok[d] and e.any():
            k = int(weighted_quantile(first[e, d], w_s[e], (0.5,))[0])
            t_spill[d] = t_fut[min(k, T - 1)]
    for a in (pres, pspill, pdes):
        a /= den[0]
        a[:, ~ok] = 0
    p_any[~ok] = 0
    out = ReservoirProduct(t_fut, v_ser, qin_ser, qout_ser, prob, level, pres.astype(np.float32), pspill.astype(np.float32),
                           pdes.astype(np.float32), qin_f, qout_f, qsp_f, v_f, p_any.astype(np.float32), t_spill, v0, valid, ok)
    if keep_sim:
        mid = K // 2                                                         # the median multiplier of every scenario
        out.sim = {"keys": KEY, "t": t_fut, "q_river": q_riv.reshape(MV, K, T, D)[:, mid],
                   "before": np.nan_to_num(river)}
    return out


def outflow_to_points(sim: dict, dn: DamNet, point_ids: list[str]) -> dict:
    """Discharge that the dams add at each control point, per scenario: {(member name, sj, si): (T, P) m3/s on sim["t"]}.
    Only dams that CUT the catchment of the point (control_points.json: the area above them is not in its
    unregulated catchment) and have no other modelled dam between them and the point; the outflow arrives
    after the travel time dam -> point. Before the first hour the measured outflow is used."""
    if not sim:
        return {}
    pos = {p: k for k, p in enumerate(point_ids)}
    links = []
    for d, m in enumerate(dn.meta):
        if not m.get("cuts"):
            continue
        for b in m.get("points", []):
            if b["id"] in pos and not b.get("through"):
                links.append((d, pos[b["id"]], max(0, int(round(b["lag_h"])))))
    q = sim["q_river"]                                                       # (MV, T, D)
    MV, T, _ = q.shape
    add = np.zeros((MV, T, len(point_ids)), np.float32)
    for d, p, lag in links:
        add[:, lag:, p] += q[:, :T - lag, d] if lag < T else 0.0
        add[:, :min(lag, T), p] += sim["before"][d]
    return {k: add[i] for i, k in enumerate(sim["keys"])}


def pack(rp_: ReservoirProduct, dn: DamNet, max_t: int = DEFAULTS["max_t"]) -> dict:
    """The per-horizon block of snap["reservoirs"] (compact arrays, same conventions as `points`)."""
    r = lambda a, nd: np.round(np.nan_to_num(np.asarray(a, np.float64)), nd).tolist()
    vs = np.where(dn.v_spill > 0, dn.v_spill, np.nan)
    pct = lambda a: 100.0 * a / vs
    T = len(rp_.t_end)
    step = max(1, int(np.ceil(T / max_t)))
    ks = np.arange(step - 1, T, step) if T else np.array([], int)
    iso = lambda t: None if str(t) == "NaT" else str(t)[:13] + ":00Z"
    return {"level": rp_.level.astype(int).tolist(), "p": r(rp_.prob, 3),
            "p_reserve": r(rp_.p_reserve, 3), "p_spill": r(rp_.p_spill, 3), "p_design": r(rp_.p_design, 3),
            "qin": r(rp_.q_in, 0), "qout": r(rp_.q_out, 0), "qspill": r(rp_.q_spill, 0), "pct": r(pct(rp_.v_end), 1),
            "p_spill_any": r(rp_.p_spill_any, 3), "t_spill": [iso(t) for t in rp_.t_spill],
            "t": [iso(t) for t in rp_.t_end[ks]], "v": r(pct(rp_.v_series[:, ks]), 1),
            "qi": r(rp_.qin_series[:, ks], 0), "qo": r(rp_.qout_series[:, ks], 0), "ok": [bool(x) for x in rp_.ok]}


def static_block(dn: DamNet, live: list[dict], point_ids: list[str], now: datetime) -> dict:
    """snap["reservoirs"] without the horizons: what each dam is, what it holds now, and which control points it feeds.

    dams[k]  id, name, river, lat, lon, source, kind (gated | free | operating limit below the spillway | unknown),
             cap (hm3 at the spill level), res (hm3, flood-reserve limit this month, or null), top (hm3 at the crest, or null),
             qd (design spillway capacity m3/s, or null), thr [low, mid, high] outflow thresholds m3/s, area (km2),
             up [ids of the dams directly above], next (id of the next dam below, or null),
             now {t, v (hm3), pct (% of cap), level (m), qin, qout, qriv (m3/s), rate (hm3/h), sv: [[hour ISO, % of cap], ...]} | null
    points   {control point id: [[dam index, share of the point's catchment above the dam, lag h, cuts 0/1], ...]}
             only dams with no other modelled dam between them and the point"""
    by = {r["id"]: r for r in live or []}
    month = int(str(np.datetime64(now, "h"))[5:7])
    pos = {p: k for k, p in enumerate(point_ids)}
    fin = lambda x, nd=2: None if x is None or not np.isfinite(x) else round(float(x), nd)
    dams, pts = [], {}
    for k, m in enumerate(dn.meta):
        r = by.get(m["id"])
        cap = dn.v_spill[k]
        nowb = None
        if r and r.get("volume_hm3") is not None:
            sv = []
            if r.get("series"):
                sv = [[t, fin(100.0 * v / cap, 1) if np.isfinite(cap) and cap > 0 else None] for t, v in zip(r["series"]["t"], r["series"]["v"])]
            nowb = {"t": r.get("t_utc"), "v": fin(r["volume_hm3"], 3), "pct": fin(100.0 * r["volume_hm3"] / cap, 1) if np.isfinite(cap) and cap > 0 else None,
                    "level": fin(r.get("level_m")), "qin": fin(r.get("inflow_m3s"), 1), "qout": fin(r.get("outflow_m3s"), 1),
                    "qriv": fin(r.get("outflow_river_m3s"), 1), "rate": fin(r.get("rate_hm3h"), 4), "sv": sv}
        dams.append({"id": m["id"], "name": m["name"], "river": m["river"], "lat": m["lat"], "lon": m["lon"], "source": m["source"],
                     "kind": m.get("spill_kind", "unknown"), "cap": fin(cap, 2), "res": fin(dn.v_res[month - 1, k], 2), "top": fin(dn.v_top[k], 2),
                     "qd": fin(dn.q_design[k], 0), "thr": [fin(x, 0) for x in dn.thr[:, k]], "area": m.get("area_km2"),
                     "up": [dn.ids[u] for u, _ in dn.up[k]], "next": (m.get("next_dam") or {}).get("id"), "now": nowb})
        for b in m.get("points", []):
            if b["id"] in pos and not b.get("through"):
                pts.setdefault(b["id"], []).append([k, b["share"], b["lag_h"], int(bool(m.get("cuts")))])
    return {"dams": dams, "points": pts, "horizons": {}}
