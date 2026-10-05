"""Final model + the report tables (miss diagnosis, operating points, size classes, 2024 anchors).

    py -3.11 hindcast/ml/train.py           # ~1 min after evaluate.py; writes geo/hydro/flood_ml.json, out/report.json

Model (the recommendation of coord/findings/q11-ml.md): pooled logistic 'hybrid' vote
    P_ml(q >= T) = 1 / (1 + exp(-(b0 + b1 x_sim + b2 x_pot + b3 l_dep)))
fitted on all clean ravine-domain events, thresholds T stacked (c, 2c, 4c, 8c, level 2, level 3), both conceptual variants
(deployed alpha and regression alpha) stacked, so one coefficient set serves gauged and ungauged points.  It is used only
as a vote that can RAISE levels 2 and 3: P = max(P_dressed, P_ml) per scenario.
"""
import sys

import numpy as np
from scipy.special import ndtr

from common import CACHE, MODEL_JSON, OUT, ROOT, counts, read_json, write_json
from evaluate import POOL, SIZES, Table, fit_logit, load, predict_logit

FEATS = ["x_sim", "x_pot", "l_dep"]
UNION = "max(dressed 0.7, logit hybrid ratio + potential + depth)"
HYB = "logit hybrid: ratio + potential + depth"
TAUS = {"now (sigma 0.7, tau 0.40)": (0.7, 0.40), "mid (sigma 0.8, tau 0.20)": (0.8, 0.20), "long (sigma 0.9, tau 0.25)": (0.9, 0.25)}


def fit_final(tb):
    X = np.vstack([tb.X(FEATS, s, k) for s in ("gauged", "ungauged") for k in POOL])
    y = np.concatenate([tb.y(k) for s in ("gauged", "ungauged") for k in POOL])
    beta = fit_logit(X, y, 1.0)
    assert (beta[1:] > 0).all(), f"a coefficient has the wrong physical sign: {beta}"
    return beta


def cause(r, union_hit, T):
    ratio = r["sim_dep"] / T
    if r["regulated"]:
        why = "regulated catchment (dam release subtracted; residual flow poorly known to model and data)"
    elif r["src"] == "gauges" and r["t0"] < "2024-12-01":
        why = "rain analysis: few pluviometers before Dec 2024, no radar truth"
    elif ratio >= 0.5:
        why = "near-miss (simulated 50-100 % of the threshold)"
    elif r["q_obs"] < 1.5 * T:
        why = "marginal exceedance (< 1.5 x threshold)"
    else:
        why = "complete miss: losses too high for this catchment / event"
    return why + (" -> recovered by the vote" if union_hit else "")


def main():
    tb = Table(load())
    oof = np.load(CACHE / "oof.npz")
    beta = fit_final(tb)
    rep = dict(beta=[round(float(b), 4) for b in beta], features=FEATS)
    # ---- operating points of the site's own rule, conceptual alone vs with the vote (out of sample) ----
    ops = {}
    for tkey in ("c", "thr2"):
        y = tb.y(tkey)
        for setting in ("gauged", "ungauged"):
            ph = oof[f"{HYB}|{setting}|{tkey}"]
            for hz, (sg, tau) in TAUS.items():
                pdr = ndtr(np.log(np.maximum(tb.sim[setting] / tb.T[tkey], 1e-9)) / sg)
                a, b = counts(pdr >= tau, y), counts(np.maximum(pdr, ph) >= tau, y)
                ops[f"{tkey}|{setting}|{hz}"] = dict(conceptual=a, with_vote=b)
                print(f"{tkey:5s} {setting:9s} {hz:26s} conceptual {a['hits']:3d}/{a['misses']:2d}/{a['false']:3d} POD {a['pod']:.2f} FAR {a['far']:.2f}"
                      f"  | + vote {b['hits']:3d}/{b['misses']:2d}/{b['false']:3d} POD {b['pod']:.2f} FAR {b['far']:.2f}")
                if hz.startswith("now"):
                    for nm, lo, hi in SIZES:
                        m = (tb.area >= lo) & (tb.area < hi)
                        a2, b2 = counts(pdr[m] >= tau, y[m]), counts(np.maximum(pdr, ph)[m] >= tau, y[m])
                        ops[f"{tkey}|{setting}|{hz}|{nm}"] = dict(conceptual=a2, with_vote=b2)
                        print(f"      {nm:12s} conceptual {a2['hits']}/{a2['misses']}/{a2['false']}  + vote {b2['hits']}/{b2['misses']}/{b2['false']}")
    rep["operating_points"] = ops
    # ---- the 31 misses of q7 (runs, deployed model, in-sample definition) and the 11 level-2 misses ----
    diag = []
    ph_c = oof[f"{HYB}|gauged|c"]; ph_2 = oof[f"{HYB}|gauged|thr2"]
    for i, r in enumerate(tb.rows):
        for tkey, ph in (("c", ph_c), ("thr2", ph_2)):
            T = r["c"] if tkey == "c" else r["thr2"]
            if r["q_obs"] >= T and r["sim_dep"] < T and (tkey == "thr2" or r["usable"]):
                pdr = float(ndtr(np.log(max(r["sim_dep"] / T, 1e-9)) / 0.7))
                hit = max(pdr, float(ph[i])) >= 0.4
                diag.append(dict(target=tkey, pid=r["pid"], name=r["name"], t0=r["t0"][:13], area=r["area"], rain=r["rain"],
                                 observed=r["q_obs"], threshold=round(T, 1), simulated=r["sim_dep"], ratio=round(r["sim_dep"] / T, 2),
                                 p_dressed=round(pdr, 2), p_vote=round(float(ph[i]), 2), cause=cause(r, hit, T)))
    rep["misses"] = diag
    for tkey in ("c", "thr2"):
        L = [d for d in diag if d["target"] == tkey]
        cz = {}
        for d in L:
            k = d["cause"].split(" -> ")[0].split(" (")[0].split(":")[0]
            cz.setdefault(k, [0, 0]); cz[k][0] += 1; cz[k][1] += "recovered" in d["cause"]
        print(f"misses {tkey}: {len(L)}  causes [n, recovered by vote at tau 0.4]: {cz}")
    # ---- 2024 anchors: dressed probability at levels 2..5 with the control points' thresholds (the vote can only add) ----
    sys.path.insert(0, str(ROOT / "backend"))
    from riua import params as PR, static
    from riua.core.hydro import level_thresholds
    net, _ = static.hydro_net()
    thr = level_thresholds(net, PR.load()["hydro"])
    anc = []
    for a in read_json(OUT / "anchors_2024.json"):
        if a["pid"] not in net.ids:
            continue
        b = net.ids.index(a["pid"])
        p = [round(float(ndtr(np.log(a["sim_dep"] / thr[k, b]) / 0.7)), 3) for k in range(4)]
        anc.append(dict(pid=a["pid"], sim=a["sim_dep"], thresholds=[round(float(t)) for t in thr[:, b]], p_dressed_L2_L5=p))
        print("anchor", a["pid"], a["sim_dep"], [round(float(t)) for t in thr[:, b]], p)
    rep["anchors_2024"] = anc
    write_json(OUT / "report.json", rep, indent=1)
    model = dict(
        version="q11-ml 2026-10-05",
        description="Independent vote that may raise flood levels 2 and 3 at the control points: logistic on the conceptual "
                    "peak ratio, the potential flow (12-h mean of the zero-loss routed flow) and the arriving rain depth (24 h). "
                    "Use P = max(P_dressed, P_ml); never lowers a level; not applied to levels 4-5 (no data above level 3).",
        features=["x_sim = ln((q_sim + 0.1 c) / T)", "x_pot = ln((q_pot + 0.1 c) / T)", "l_dep = ln(1 + d24)"],
        c="5 * (A / 184) ** 0.75  [m3/s]",
        beta=[float(b) for b in beta], levels=[2, 3], pot_hours=12, dep_hours=24,
        trained_on=dict(events=tb.n, catchments=len(set(tb.pid)), episodes=len(set(tb.ep)), runs=int(tb.y("c").sum()), level2=int(tb.y("thr2").sum())))
    write_json(MODEL_JSON, model, indent=1)
    print("beta", rep["beta"], "->", MODEL_JSON)


if __name__ == "__main__":
    sys.exit(main())
