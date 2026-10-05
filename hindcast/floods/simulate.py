"""Run the production flood model over the historical cases and write, per case and point, what it would have shown.

Model = production, nothing refitted here: static.hydro_net() (per-cell alpha of geo/hydro/loss_params.json),
hydro.net_rain with params.load()["hydro"], hydro.route, hydro.level_thresholds. The gauged catchments of q7
(geo/hydro/gauges) are run with their own fitted alpha (fit_catchments.json), uniform over the catchment.

Input : hindcast/floods/cases.json + cache/rain/<case>.npz (build_rain.py) or hindcast/truth/<truth>.npz
Output: hindcast/floods/out/sim.json   one row per case and point (peaks, level, times, rain, lead times)
        hindcast/floods/cache/sim/<case>.npz   hourly series (q of every point, basin-mean rain)

    py -3.11 hindcast/floods/simulate.py [case ...] [--p0 150 --s 150 --phi 45 --k 0.3 --vel 1.0 --alpha-scale 1.0]

--vel scales the travel times (0.5 = water travels at half the speed, as q7 found for ordinary floods).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "geo" / "hydro" / "gauges"))
from riua import params, static  # noqa: E402
from riua.core import hydro  # noqa: E402

OUT = HERE / "out"
SIMC = HERE / "cache" / "sim"


def rain_file(case) -> Path:
    if case["rain"] == "truth":
        return ROOT / "hindcast" / "truth" / f"{case.get('truth', case['id'])}.npz"
    return HERE / "cache" / "rain" / f"{case['id']}.npz"


def cell_area(net) -> sparse.csr_matrix:
    """(P, ncell) km2 of every cell inside each catchment."""
    A = net.lags[0].copy()
    for L in net.lags[1:]:
        A = A + L
    return A.tocsr()


def slow(net, vel: float):
    """Same network with every travel time divided by vel (vel < 1 = slower water)."""
    if vel == 1.0:
        return net
    import copy
    n2 = copy.copy(net)
    kmax = int(np.ceil((len(net.lags) - 1) / vel))
    lags = [sparse.csr_matrix(net.lags[0].shape, dtype=np.float32) for _ in range(kmax + 1)]
    for k, A in enumerate(net.lags):
        k2 = int(round(k / vel))
        lags[k2] = lags[k2] + A
    n2.lags = [m.tocsr() for m in lags]
    n2.tc_h = net.tc_h / vel
    return n2


def first_cross(q, thr):
    """index of the first hour with q >= thr, or -1"""
    over = q >= thr
    return int(np.argmax(over)) if over.any() else -1


def run_case(case, net, meta, thr, gnet, gmeta, galpha, hp, vel=1.0, alpha_scale=1.0, lead=True):
    z = np.load(rain_file(case), allow_pickle=False)
    t_end = z["t_end"].astype("datetime64[h]")
    p = np.nan_to_num(z["o_mean"].reshape(len(t_end), -1), nan=0.0).astype(np.float32)
    T = len(t_end)
    kw = dict(phi=hp.get("phi_mmh"), s=hp.get("s_mm"), p0b=hp.get("p0b_mm", 10.0), sb=hp.get("sb_mm", 100.0))
    alpha = None if net.alpha is None else np.clip(net.alpha * alpha_scale, 0, 1)
    e = hydro.net_rain(p, hp["p0_mm"], hp["wet_memory_h"], alpha=alpha, **kw)
    n1, g1 = slow(net, vel), slow(gnet, vel)
    q = hydro.route(e, n1, hp["clark_k"])                                   # (T, P)
    q0 = hydro.route(p, n1, hp["clark_k"])                                  # no losses at all: the ceiling the rain allows
    # gauged catchments: net = (1 - a) main + a quick, linear in a
    e_main = hydro.net_rain(p, hp["p0_mm"], hp["wet_memory_h"], alpha=None, **kw)
    e_quick = hydro.net_rain(p, hp["p0_mm"], hp["wet_memory_h"], alpha=np.ones(p.shape[1], np.float32), **kw)
    ga = np.clip(galpha * alpha_scale, 0, 1)
    gq = (1 - ga)[None] * hydro.route(e_main, g1, hp["clark_k"]) + ga[None] * hydro.route(e_quick, g1, hp["clark_k"])
    gq0 = hydro.route(p, g1, hp["clark_k"])
    rows = []
    series = {"t_end": t_end.astype("datetime64[h]").astype(str)}
    # what the model could call from the rain already fallen (no forecast): route the net rain cut at hour tau
    qcut_max = None
    if lead:
        # routing is linear: D[k, b, u] = flow that the net rain of hour u sends to point b, k hours later
        from scipy.signal import lfilter
        D = np.stack([0.278 * np.asarray(A @ e.T) for A in n1.lags])        # (K, P, T)
        qcut_max = np.zeros((T, net.n))
        uu = np.arange(T)
        for b in range(net.n):
            C = np.zeros((T, T))                                            # C[u, t]
            for k in range(D.shape[0]):
                ok = uu + k < T
                C[uu[ok], uu[ok] + k] = D[k, b, ok]
            C = np.cumsum(C, axis=0)                                        # rain known up to hour tau = row tau
            K = max(hp["clark_k"] * n1.tc_h[b], 0.25)
            c1 = 1.0 - np.exp(-1.0 / K)
            if hp["clark_k"] > 0:
                C = lfilter([c1], [1.0, -(1.0 - c1)], C, axis=1)
            qcut_max[:, b] = C.max(axis=1)
    for kind, nn, mm, qq, qq0, th in (("control", net, meta, q, q0, thr), ("gauge", gnet, gmeta, gq, gq0, None)):
        A = cell_area(nn)
        area = np.asarray(A.sum(axis=1)).ravel()
        pb = (A @ p.T).T / np.maximum(area, 1e-9)[None]                     # (T, P) basin-mean rain mm/h
        eb = (A @ e.T).T / np.maximum(area, 1e-9)[None] if kind == "control" else None
        series[f"{kind}_ids"] = np.array(nn.ids)
        series[f"{kind}_q"] = qq.astype(np.float32)
        series[f"{kind}_rain"] = pb.astype(np.float32)
        cs = np.concatenate([np.zeros((1, nn.n)), np.cumsum(pb, axis=0)])
        idx = np.arange(1, T + 1)
        r12 = cs[idx] - cs[np.maximum(idx - 12, 0)]
        for b, pid in enumerate(nn.ids):
            k = int(np.argmax(qq[:, b]))
            row = {"case": case["id"], "kind": kind, "point": pid,
                   "area_km2": round(float(area[b]), 1),
                   "rain_mm": round(float(pb[:, b].sum()), 1), "rain_max1h": round(float(pb[:, b].max()), 1),
                   "rain_max12h": round(float(r12[:, b].max()), 1),
                   "t_rain_max1h": str(t_end[int(np.argmax(pb[:, b]))]),
                   "sim_peak": round(float(qq[k, b]), 1), "t_sim_peak": str(t_end[k]),
                   "zero_loss_peak": round(float(qq0[:, b].max()), 1)}
            if eb is not None:
                row["runoff_mm"] = round(float(eb[:, b].sum()), 1)
            if th is not None:
                t4 = float(th[2, b])
                lev = 1 + int((qq[k, b] >= th[:, b]).sum()) if qq[k, b] >= th[0, b] else 1
                row.update(thr2=round(float(th[0, b]), 1), thr3=round(float(th[1, b]), 1), thr4=round(t4, 1),
                           thr5=round(float(th[3, b]), 1), sim_level=lev, sim_overflow=bool(qq[k, b] >= t4),
                           ratio_cap=round(float(qq[k, b] / t4), 3))
                for L, name in ((0, "l2"), (2, "l4")):
                    c = first_cross(qq[:, b], th[L, b])
                    if c < 0:
                        continue
                    row[f"t_{name}"] = str(t_end[c])
                    # hours between the start of the event rain over the basin (first hour >= 2 mm in the 24 h before) and the crossing
                    w0 = max(c - 36, 0)
                    wet = np.nonzero(pb[w0:c + 1, b] >= 2.0)[0]
                    if len(wet):
                        row[f"h_rain_start_to_{name}"] = int(c - (w0 + wet[0]))
                    row[f"h_rain_peak_to_{name}"] = int(c - int(np.argmax(pb[: c + 1, b])))
                    if qcut_max is not None:
                        tau = first_cross(qcut_max[:, b], th[L, b])
                        if tau >= 0:
                            row[f"lead_obs_rain_{name}_h"] = int(c - tau)     # crossing hour minus the hour whose rain made it certain
            rows.append(row)
    SIMC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SIMC / f"{case['id']}.npz", **series)
    return rows


def load_nets():
    from dataset import gauge_net
    net, meta = static.hydro_net()
    hp = params.load()["hydro"]
    thr = hydro.level_thresholds(net, hp)
    gnet, gmeta = gauge_net()
    fit = json.loads((ROOT / "geo" / "hydro" / "gauges" / "out" / "fit_catchments.json").read_text(encoding="utf-8"))
    galpha = np.array([fit.get(i, {}).get("alpha", 0.03) for i in gnet.ids], float)
    return net, meta, thr, gnet, gmeta, galpha, hp


def main(argv):
    opt = {"--p0": None, "--s": None, "--phi": None, "--k": None, "--vel": 1.0, "--alpha-scale": 1.0, "--out": "sim.json"}
    want = []
    it = iter(argv)
    for a in it:
        if a in opt:
            v = next(it)
            opt[a] = v if a == "--out" else float(v)
        else:
            want.append(a)
    net, meta, thr, gnet, gmeta, galpha, hp = load_nets()
    if "--cap-floor-fix" in want:
        # proposed for core/hydro.level_thresholds: the trickle floors never lift the overflow above a known capacity
        want.remove("--cap-floor-fix")
        qb = net.q_bankfull
        ok = np.isfinite(qb)
        thr[1, ok] = np.minimum(thr[1, ok], qb[ok])
        thr[0] = np.minimum(thr[0], thr[1])
        thr[2, ok] = qb[ok]
        thr[3] = np.maximum(thr[3], thr[2])
    for key, name in (("--p0", "p0_mm"), ("--s", "s_mm"), ("--phi", "phi_mmh"), ("--k", "clark_k")):
        if opt[key] is not None:
            hp[name] = opt[key]
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    rows = []
    for c in cases:
        if want and c["id"] not in want:
            continue
        if not rain_file(c).exists():
            print("no rain yet:", c["id"], flush=True)
            continue
        r = run_case(c, net, meta, thr, gnet, gmeta, galpha, hp, opt["--vel"], opt["--alpha-scale"])
        rows += r
        over = [x["point"] for x in r if x.get("sim_overflow")]
        print(c["id"], "overflow simulated at", over, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / opt["--out"]
    if want and out.exists():                       # partial run: replace only those cases
        old = [x for x in json.loads(out.read_text(encoding="utf-8"))["rows"] if x["case"] not in want]
        rows = old + rows
    out.write_text(json.dumps({"params": {k: hp[k] for k in ("p0_mm", "s_mm", "phi_mmh", "p0b_mm", "sb_mm", "wet_memory_h", "clark_k")},
                               "vel": opt["--vel"], "alpha_scale": opt["--alpha-scale"], "rows": rows}, indent=0), encoding="utf-8")
    print("rows", len(rows), "->", out)


if __name__ == "__main__":
    main(sys.argv[1:])
