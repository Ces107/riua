"""Static geodata: grid mask, warning zones and thresholds, basin units, control points.

Everything here is produced once by the build scripts under geo/ and loaded with
numpy + json only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .core import grid
from .core.basins import BasinSet
from .core.risk import Thresholds

REPO = Path(__file__).resolve().parents[2]
GEO = Path(os.environ.get("RIUA_GEO", REPO / "geo"))

DEFAULT_T1H = (20.0, 40.0, 90.0)
DEFAULT_T12H = (60.0, 100.0, 180.0)


@dataclass
class Static:
    mask: np.ndarray            # (NY, NX) bool: cells shown (land, in the CV or draining into it)
    cv_frac: np.ndarray         # (NY, NX)
    land_frac: np.ndarray
    zone_idx: np.ndarray        # (NY, NX) int, -1 = none
    zone_codes: list[str]
    zone_names: list[str]
    zone_thr: dict              # code -> {"1h": [y, o, r], "12h": [y, o, r]}
    basin_idx: np.ndarray       # (NY, NX) int, -1 = none
    terrain: dict               # elev_smooth, dzdx, dzdy
    thresholds_source: dict

    @property
    def n_cells(self) -> int:
        return int(self.mask.sum())


@lru_cache(maxsize=1)
def load() -> Static:
    z = np.load(GEO / "out" / "terrain.npz", allow_pickle=False)
    thr = json.loads((GEO / "out" / "thresholds.json").read_text(encoding="utf-8"))
    codes = [str(c) for c in z["zone_codes"]]
    zone_thr = {}
    for c in codes:
        t = thr["zones"][c]
        zone_thr[c] = {"1h": [t["precip_1h_mm"][k] for k in ("yellow", "orange", "red")],
                       "12h": [t["precip_12h_mm"][k] for k in ("yellow", "orange", "red")]}
    return Static(
        mask=z["in_domain"].astype(bool), cv_frac=z["cv_frac"], land_frac=z["land_frac"],
        zone_idx=z["zone_idx"].astype(int), zone_codes=codes, zone_names=[str(n) for n in z["zone_names"]],
        zone_thr=zone_thr, basin_idx=z["basin_idx"].astype(int),
        terrain={k: z[k] for k in ("elev_mean", "elev_smooth", "dzdx", "dzdy")},
        thresholds_source=thr["source"])


def thresholds(params: dict) -> Thresholds:
    """Per-cell AEMET thresholds; cells outside every zone (upstream, other regions) take the
    values common to the eleven Valencian zones."""
    s = load()
    arr = {k: np.empty((3, grid.NY, grid.NX), np.float32) for k in ("1h", "12h")}
    for k, default in (("1h", DEFAULT_T1H), ("12h", DEFAULT_T12H)):
        for lv in range(3):
            a = np.full((grid.NY, grid.NX), default[lv], np.float32)
            for zi, c in enumerate(s.zone_codes):
                a[s.zone_idx == zi] = s.zone_thr[c][k][lv]
            arr[k][lv] = a
    return Thresholds.from_zone_values(*arr["1h"], *arr["12h"], params)


@lru_cache(maxsize=1)
def basins() -> BasinSet:
    z = np.load(GEO / "out" / "terrain.npz", allow_pickle=False)
    topo = json.loads((GEO / "out" / "basins_topology.json").read_text(encoding="utf-8"))
    ids = [str(i) for i in z["basin_ids"]]
    B, n = len(ids), grid.NY * grid.NX
    area = z["basin_area_km2"].astype(np.float64)
    W = np.zeros((B, n), np.float32)
    ptr, cell, w = z["basin_w_ptr"], z["basin_w_cell"], z["basin_w"]
    for b in range(B):
        s = slice(ptr[b], ptr[b + 1])
        if ptr[b + 1] > ptr[b]:
            ww = w[s] / max(float(w[s].sum()), 1e-12)
            W[b, cell[s]] = ww
    pos = {i: k for k, i in enumerate(ids)}
    U = np.zeros((B, B), np.float32)
    up_area = np.zeros(B)
    inside = z["basin_frac_in_box"].astype(np.float64)
    for b, i in enumerate(ids):
        ups = [pos[u] for u in topo["units"][i]["upstream_all"] if u in pos]
        members = [b] + ups
        a = area[members] * np.clip(inside[members], 0.0, 1.0)     # only the part we have rain for
        if a.sum() <= 0:
            a = np.ones(len(members))
        U[b, members] = a / a.sum()
        up_area[b] = float(topo["units"][i].get("up_area_km2", area[members].sum()))
    s = load()
    t1 = np.zeros((3, B), np.float32); t12 = np.zeros((3, B), np.float32)
    thr_cells = {k: np.stack([np.full(n, d, np.float32) for d in default])
                 for k, default in (("1h", DEFAULT_T1H), ("12h", DEFAULT_T12H))}
    zi = s.zone_idx.ravel()
    for k in ("1h", "12h"):
        for lv in range(3):
            for q, c in enumerate(s.zone_codes):
                thr_cells[k][lv][zi == q] = s.zone_thr[c][k][lv]
    t1[:] = thr_cells["1h"] @ W.T
    t12[:] = thr_cells["12h"] @ W.T
    empty = W.sum(axis=1) == 0
    t1[:, empty] = np.array(DEFAULT_T1H)[:, None]
    t12[:, empty] = np.array(DEFAULT_T12H)[:, None]
    return BasinSet(ids, [str(x) for x in z["basin_names"]], area, up_area, W, U, t1, t12)


@lru_cache(maxsize=1)
def basin_meta() -> dict:
    """Flags per basin unit: shown (in the CV or draining into it) and has cells in the box."""
    z = np.load(GEO / "out" / "terrain.npz", allow_pickle=False)
    shown = (z["basin_in_cv"] | z["basin_drains_to_cv"]) & (z["basin_frac_in_box"] > 0.5)
    return {"shown": shown.astype(bool), "frac_in_box": z["basin_frac_in_box"]}


def hydro_net():
    """Control-point network, or None while the hydro geodata has not been built."""
    from .core.hydro import HydroNet
    cdir, sdir = GEO / "hydro" / "catchments" / "out", GEO / "hydro" / "sections" / "out"
    if not (cdir / "control_points.json").exists() or not (cdir / "time_area.npz").exists():
        return None, []
    cps = json.loads((cdir / "control_points.json").read_text(encoding="utf-8"))
    cps = cps["points"] if isinstance(cps, dict) else cps
    ta = np.load(cdir / "time_area.npz", allow_pickle=False)
    secs = {}
    if (sdir / "sections.json").exists():
        sj = json.loads((sdir / "sections.json").read_text(encoding="utf-8"))
        sj = sj.get("points", sj) if isinstance(sj, dict) else sj
        secs = sj if isinstance(sj, dict) else {s["id"]: s for s in sj}
    pid = [str(x) for x in ta["point_ids"]]
    use = ta["scope"] == 1                                   # unregulated catchment
    order = {p["id"]: p for p in cps}
    ids = [i for i in pid if i in order]
    remap = {pid.index(i): k for k, i in enumerate(ids)}
    sel = use & np.isin(ta["point_idx"], list(remap))
    pidx = np.array([remap[int(x)] for x in ta["point_idx"][sel]], int)
    capf = GEO / "hydro" / "capacity.json"
    caps = json.loads(capf.read_text(encoding="utf-8")) if capf.exists() else {}
    qb, rq, rh, hb = [], [], [], []
    for i in ids:
        c = caps.get(i)
        qb.append(float(c["q"]) if c else np.nan)
        s = secs.get(i, {}) if (c and c.get("rating")) else {}
        rt = s.get("rating") or {}
        q_tab, h_tab = rt.get("q_m3s"), rt.get("h_m")
        if q_tab and h_tab and len(q_tab) > 1:
            rq.append(np.asarray(q_tab, float)); rh.append(np.asarray(h_tab, float))
        else:
            rq.append(None); rh.append(None)
        hb.append(float(s["bankfull_depth_m"]) if s.get("bankfull_depth_m") else np.nan)
    net = HydroNet.build(ids, [order[i].get("area_unregulated_km2") or order[i]["area_km2"] for i in ids],
                         [order[i].get("t_longest_unregulated_h") or order[i]["tc_h"] for i in ids], pidx, ta["cell"][sel].astype(int),
                         ta["lag_h"][sel].astype(int), ta["area_km2"][sel].astype(float), qb, rq, rh, hb)
    return net, [dict(order[i], section=secs.get(i), capacity=caps.get(i)) for i in ids]
