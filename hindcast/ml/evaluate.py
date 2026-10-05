"""Candidates for the flood call at the control points, scored out of sample.

    py -3.11 hindcast/ml/evaluate.py            # ~2 min; writes out/evaluation.json and prints the tables

Rows: the clean events of the ravine domain (26 gauged catchments), floods and non-floods (out/dataset.json).
Two targets: 'runs' (peak >= c(A), q7's definition of a flood) and 'level2' (peak >= the site's level-2 threshold).
Two honest settings, never in-sample:
  gauged    the conceptual peak as deployed (alpha fitted to that stream), leave-one-EPISODE-out for anything fitted here;
  ungauged  the conceptual peak with alpha from the attribute regression, leave-one-CATCHMENT-out.
Candidates:
  ratio      alert when the conceptual peak >= k x threshold (k scanned); its probability is the site's lognormal dressing
  logit ...  logistic regression on a few features (with the conceptual ratio = hybrid; without = a rain-threshold rule)
  gbm        gradient boosting, shallow, monotone in rain and in the conceptual ratio
  max(...)   the union: the larger of the dressed conceptual probability and a data-driven probability
Features (all computable inside hydro_product from the routed hydrographs of a scenario):
  x_sim   ln((conceptual peak + 0.1 c) / T)                      T = the threshold in question
  x_pot   ln(potential flow / T), potential = the 12 wettest hours of rain ARRIVING at the point, as a flow (100 % runoff)
  l_dep   ln(1 + rain arriving in 24 h, mm)        l_wv  ln(1 + arriving rain remembered with a 72-h memory)
  l_alpha ln(quick-runoff share of the catchment)  + antecedent (API, 30-day rain, ERA5-Land soil moisture), season, area.
"""
import sys
import warnings

import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from common import CACHE, OUT, auc, c_off, counts, read_json, write_json

warnings.filterwarnings("ignore")
TAU_H = 72.0
POOL = ("c", "2c", "4c", "8c", "thr2", "thr3")          # thresholds stacked when a model is trained 'pooled'
SIZES = (("< 300 km2", 0, 300), ("300-700 km2", 300, 700), (">= 700 km2", 700, 1e9))


# ---------------------------------------------------------------------------------------------------------------- data
def arriving(q0, area, i0, iw):
    """rain arriving at the point (zero-loss routed flow as mm/h over the catchment): largest 6/12/24-h depth and the
    largest value of its 72-h memory inside the event window"""
    v = q0.astype(float) * 3.6 / area
    cs = np.concatenate([[0.0], np.cumsum(v)])
    e = np.arange(i0, iw + 1) + 1
    out = {f"rv{D}": float((cs[e] - cs[np.maximum(e - D, 0)]).max()) for D in (6, 12, 24)}
    lo = max(i0 - 240, 0)
    w, best = 0.0, 0.0
    dec = np.exp(-1.0 / TAU_H)
    for t in range(lo, iw + 1):
        w = w * dec + v[t]
        if t >= i0:
            best = max(best, w)
    out["wv"] = best
    return out


def load():
    rows = [r for r in read_json(OUT / "dataset.json") if r["clean"] and r["domain"]]
    z = np.load(CACHE / "sim_series.npz")
    gl = read_json(OUT / "glofas_rows.json") if (OUT / "glofas_rows.json").exists() else {}
    for r in rows:
        if "rv12" not in r:
            r.update(arriving(z[f"{r['pid']}|q0"], r["area"], r["i0"], r["iw"]))
        f = CACHE / "era5land" / f"{r['pid']}_{r['t0']}.json"
        sm = [np.nan] * 4
        if f.exists():
            h = read_json(f)["hourly"]
            k = int(r["t0"][11:13])
            sm = [h[v][k] if h[v][k] is not None else np.nan for v in
                  ("soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm")]
        r["sm1"], r["sm2"], r["sm3"], r["sm4"] = sm
        g = gl.get(f"{r['pid']}|{r['t0']}")
        r["glofas_rise"] = max(g[0] - g[1], 0.0) if g else np.nan
    return rows


class Table:
    """columns of the event table as arrays + the feature builder"""

    def __init__(self, rows):
        self.rows = rows
        g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in rows], float)      # noqa: E731
        self.g = g
        self.n = len(rows)
        self.pid = np.array([r["pid"] for r in rows]); self.ep = np.array([r["episode"] for r in rows])
        self.obs, self.c, self.area = g("q_obs"), g("c"), g("area")
        self.T = {"c": self.c, "2c": 2 * self.c, "4c": 4 * self.c, "8c": 8 * self.c, "thr2": g("thr2"), "thr3": g("thr3"), "thr4": g("thr4")}
        self.sim = {"gauged": g("sim_dep"), "ungauged": g("sim_reg")}
        self.alpha = {"gauged": g("alpha_dep"), "ungauged": g("alpha_reg")}
        self.groups = {"gauged": self.ep, "ungauged": self.pid}
        med = lambda x: np.where(np.isfinite(x), x, np.nanmedian(x))      # noqa: E731
        self.static = dict(
            l_dep=np.log1p(g("rv24")), l_rv12=np.log1p(g("rv12")), l_wv=np.log1p(g("wv")), l_rain=np.log(g("rain")), l_r1=np.log1p(g("r1")),
            l_cell1=np.log1p(g("cell1")), l_area=np.log(self.area), p0i=g("p0i_mm"), slope=g("slope"),
            api=np.log(g("api") + 5.0), r30=np.log(g("rain30d") + 5.0), r5=np.log(g("rain5d") + 1.0),
            sm1=med(g("sm1")), sm2=med(g("sm2")), sm3=med(g("sm3")), sm4=med(g("sm4")), cosS=g("season_cos"), sinS=g("season_sin"),
            d1=g("d1"), storm_v=g("storm_v"), sp_conc=np.log(g("sp_conc")), t_conc6=g("t_conc6"), l_dur=np.log(g("dur_h")),
            regulated=np.array([float(r["regulated"]) for r in rows]))
        self.qpot = self.area * g("rv12") / 43.2                # 0.278 A rv12 / 12: the 12 wettest arriving hours as a flow
        self.q0 = g("q0")
        self.glofas = g("glofas_rise")

    def X(self, feats, setting, tkey):
        T = self.T[tkey]
        cols = []
        for f in feats:
            if f == "x_sim":
                cols.append(np.log((self.sim[setting] + 0.1 * self.c) / T))
            elif f == "x_pot":
                cols.append(np.log((self.qpot + 0.1 * self.c) / T))
            elif f == "x_q0":
                cols.append(np.log((self.q0 + 0.1 * self.c) / T))
            elif f == "x_glofas":
                cols.append(np.log((np.nan_to_num(self.glofas) + 0.1 * self.c) / T))
            elif f == "l_alpha":
                cols.append(np.log(self.alpha[setting]))
            elif f == "l_spec":                                 # how high the bar is: threshold as a unit discharge
                cols.append(np.log(T / self.area ** 0.75))
            else:
                cols.append(self.static[f])
        return np.column_stack(cols)

    def y(self, tkey):
        return self.obs >= self.T[tkey]


# ---------------------------------------------------------------------------------------------------------------- models
MONO = dict(x_sim=1, x_pot=1, x_q0=1, l_dep=1, l_rv12=1, l_wv=1, l_rain=1, l_r1=1, l_cell1=1, l_alpha=1, api=1, r30=1, r5=1,
            sm1=1, sm2=1, sm3=1, sm4=1, l_spec=-1, x_glofas=1)


def fit_logit(X, y, C=1.0):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    m = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    w = m.coef_[0] / sd
    return np.r_[m.intercept_[0] - float((w * mu).sum()), w]              # raw-feature coefficients, intercept first


def predict_logit(beta, X):
    return 1.0 / (1.0 + np.exp(-(beta[0] + X @ beta[1:])))


def fit_gbm(X, y, feats):
    m = HistGradientBoostingClassifier(max_depth=2, max_iter=120, learning_rate=0.05, min_samples_leaf=20, l2_regularization=2.0,
                                       monotonic_cst=[MONO.get(f, 0) for f in feats], random_state=0)
    return m.fit(X, y)


def cv_predict(tb, cand, setting, tkey):
    """out-of-fold probability of exceeding threshold `tkey` for one candidate in one setting"""
    kind, feats, pooled = cand["kind"], cand.get("feats", []), cand.get("pooled", True)
    groups = tb.groups[setting]
    p = np.zeros(tb.n)
    tkeys = POOL if pooled else (tkey,)
    Xte = tb.X(feats, setting, tkey)
    Xs = {k: tb.X(feats, setting, k) for k in tkeys}
    ys = {k: tb.y(k) for k in tkeys}
    for gval in np.unique(groups):
        te = groups == gval
        Xtr = np.vstack([Xs[k][~te] for k in tkeys]); ytr = np.concatenate([ys[k][~te] for k in tkeys])
        if ytr.sum() == 0:
            continue
        if kind == "logit":
            p[te] = predict_logit(fit_logit(Xtr, ytr, cand.get("C", 1.0)), Xte[te])
        else:
            p[te] = fit_gbm(Xtr, ytr, feats).predict_proba(Xte[te])[:, 1]
    return p


def dressed(tb, setting, tkey, sigma):
    """the site's probability for one scenario: lognormal error around the simulated peak (core/risk.py::dress)"""
    with np.errstate(divide="ignore"):
        return ndtr(np.log(np.maximum(tb.sim[setting] / tb.T[tkey], 1e-9)) / sigma)


# ---------------------------------------------------------------------------------------------------------------- scores
def curve(score, y):
    """operating points along the score: for every distinct cut, (cut, hits, misses, false, POD, FAR)"""
    out = []
    for cut in np.sort(np.unique(score))[::-1]:
        c = counts(score >= cut, y)
        out.append((float(cut), c["hits"], c["misses"], c["false"], c["pod"], c["far"]))
    return out


def at_pod(score, y, target):
    for cut, h, m, f, pod, far in curve(score, y):
        if pod >= target - 1e-9:
            return dict(cut=round(cut, 4), hits=h, misses=m, false=f, pod=round(pod, 2), far=round(far, 2))
    return None


def best_far_limited(score, y, far_max=0.5):
    """the highest POD whose false-alarm ratio stays <= far_max"""
    best = None
    for cut, h, m, f, pod, far in curve(score, y):
        if far <= far_max and (best is None or pod > best["pod"]):
            best = dict(cut=round(cut, 4), hits=h, misses=m, false=f, pod=round(pod, 2), far=round(far, 2))
    return best


def reliability(p, y, edges=(0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.any():
            out.append(dict(bin=[a, min(b, 1.0)], n=int(m.sum()), mean_p=round(float(p[m].mean()), 3), freq=round(float(y[m].mean()), 3)))
    return out


def summary(p, y, prob=True):
    d = dict(auc=round(auc(p, y), 3), pod70=at_pod(p, y, 0.7), pod80=at_pod(p, y, 0.8), pod90=at_pod(p, y, 0.9), far50=best_far_limited(p, y, 0.5))
    if prob:
        d["brier"] = round(float(np.mean((p - y) ** 2)), 4)
    return d


CANDS = {
    "logit: conceptual ratio only (recalibration)": dict(kind="logit", feats=["x_sim"]),
    "logit hybrid: ratio + potential": dict(kind="logit", feats=["x_sim", "x_pot"]),
    "logit hybrid: ratio + potential + depth": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep"]),
    "logit hybrid: + alpha": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep", "l_alpha"]),
    "logit hybrid: + API": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep", "api"]),
    "logit hybrid: + ERA5-Land soil moisture 28-100 cm": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep", "sm3"]),
    "logit hybrid: + season": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep", "cosS", "sinS"]),
    "logit hybrid: + storm position and motion (d1, storm_v)": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep", "d1", "storm_v"]),
    "logit hybrid: + cell intensity": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep", "l_cell1"]),
    "logit hybrid, trained on the target threshold only": dict(kind="logit", feats=["x_sim", "x_pot", "l_dep"], pooled=False),
    "rain rule (no conceptual model): potential + depth": dict(kind="logit", feats=["x_pot", "l_dep"]),
    "rain rule + API": dict(kind="logit", feats=["x_pot", "l_dep", "api"]),
    "rain rule + soil moisture": dict(kind="logit", feats=["x_pot", "l_dep", "sm3"]),
    "rain rule + alpha": dict(kind="logit", feats=["x_pot", "l_dep", "l_alpha"]),
    "rain rule + alpha + P0i + area": dict(kind="logit", feats=["x_pot", "l_dep", "l_alpha", "p0i", "l_area"]),
    "gbm hybrid (monotone): ratio, potential, depth, alpha, API, soil, cell intensity, season":
        dict(kind="gbm", feats=["x_sim", "x_pot", "l_dep", "l_alpha", "api", "sm3", "l_cell1", "cosS", "l_spec"]),
    "gbm rain rule (monotone): potential, depth, alpha, API, soil": dict(kind="gbm", feats=["x_pot", "l_dep", "l_alpha", "api", "sm3", "l_spec"]),
}
UNIONS = {
    "max(dressed 0.7, rain rule)": "rain rule (no conceptual model): potential + depth",
    "max(dressed 0.7, rain rule + alpha)": "rain rule + alpha",
    "max(dressed 0.7, logit hybrid ratio + potential + depth)": "logit hybrid: ratio + potential + depth",
}


def main():
    tb = Table(load())
    res = dict(n=tb.n, n_catchments=len(set(tb.pid)), n_episodes=len(set(tb.ep)), targets={}, settings=dict(
        gauged="conceptual peak as deployed (alpha of the stream), leave-one-episode-out",
        ungauged="conceptual peak with the regression alpha, leave-one-catchment-out"))
    P = {}
    for tkey in ("c", "thr2"):
        y = tb.y(tkey)
        res["targets"][tkey] = dict(n_events=int(y.sum()), episodes_with_events=len(set(tb.ep[y])), catchments_with_events=len(set(tb.pid[y])))
        for setting in ("gauged", "ungauged"):
            out = {}
            ratio = tb.sim[setting] / tb.T[tkey]
            d = summary(ratio, y, prob=False)
            for k in (1.0, 0.84, 0.51, 0.3, 0.2, 0.1):
                d[f"k={k}"] = {a: (round(b, 2) if isinstance(b, float) else b) for a, b in counts(ratio >= k, y).items()}
            out["conceptual ratio (alert when sim >= k x threshold)"] = d
            for sg in (0.7, 0.8, 0.9):
                pd_ = dressed(tb, setting, tkey, sg)
                P[(f"dressed {sg}", setting, tkey)] = pd_
                out[f"conceptual, dressed sigma {sg} (the site's probability)"] = dict(brier=round(float(np.mean((pd_ - y) ** 2)), 4),
                                                                                   reliability=reliability(pd_, y))
            for name, cand in CANDS.items():
                p = cv_predict(tb, cand, setting, tkey)
                P[(name, setting, tkey)] = p
                out[name] = summary(p, y)
                out[name]["reliability"] = reliability(p, y)
            for name, base in UNIONS.items():
                p = np.maximum(P[("dressed 0.7", setting, tkey)], P[(base, setting, tkey)])
                P[(name, setting, tkey)] = p
                out[name] = summary(p, y)
                out[name]["reliability"] = reliability(p, y)
            res[f"{tkey}|{setting}"] = out
            print(f"\n== target {tkey} ({int(y.sum())} events of {tb.n}), {setting}")
            print(f"  {'candidate':88s} {'AUC':>5s} {'Brier':>6s} | POD>=0.7: h/m/f FAR | POD>=0.8: h/m/f FAR | POD>=0.9: h/m/f FAR | best POD with FAR<=0.5")
            for name, d in out.items():
                if "auc" not in d:
                    print(f"  {name:88s} {'':5s} {d['brier']:6.4f}")
                    continue
                f = lambda o: f"{o['hits']:3d}/{o['misses']:2d}/{o['false']:3d} {o['far']:.2f}" if o else "      -      "      # noqa: E731
                b = d["far50"]
                print(f"  {name[:88]:88s} {d['auc']:5.3f} {d.get('brier', float('nan')):6.4f} | {f(d['pod70'])} | {f(d['pod80'])} | {f(d['pod90'])} | "
                      + (f"POD {b['pod']:.2f} ({b['hits']}/{b['misses']}/{b['false']})" if b else "-"))
    np.savez_compressed(CACHE / "oof.npz", **{"|".join(k): v for k, v in P.items()})
    write_json(OUT / "evaluation.json", res, indent=1)


if __name__ == "__main__":
    sys.exit(main())
