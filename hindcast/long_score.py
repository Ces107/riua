"""Scoring of the days 2-7 zone-day units (hindcast/long_zone.py units): every candidate, leave-one-case-out.

    python hindcast/long_zone.py score --units FILE --out FILE.json

Candidates, per level (2 yellow .. 5 extreme), all out of sample (kernel / model chosen without the held-out case):
  today        what the site shows now: production kernel per CELL (ENS + IFS runs), largest cell value of the zone,
               level cap 3 (no level 4-5 from probability)
  zone_prod_*  member-wise zone maximum with the production kernel (sigma 1.0, bias 1.6), member set *
  zone_fit_*   same, (sigma, bias) chosen on the Brier score of the other cases (per lead class with _lead)
  logit_*      ridge logistic regression on zone-day summaries of the member set: log mean ratio, exceedance
               fractions, upper decile, lead; + 'rank' (EFI-like percentile against the model's own climate of the
               same zone and season, built from the other cases' forecasts, days weighted by their real frequency);
               + 'ingr' (column water vapour, onshore 850-hPa moisture flux, CAPE, z500 anomaly)
  agreement    risk.model_agreement (>= 2 models reach red outright in a cell of the zone) on top of a method
Member sets: ens (ECMWF ENS), prod (ENS + IFS runs with production weights = what production uses), all (+ AIFS-ENS
with the ENS member weight), and IFS weight multipliers.
Tau: chosen on the other cases (CSI optimum, largest tau within 0.005 of it): "nested" decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, expit

import long_zone as LZ

LEVELS = LZ.LEVELS
TAUS = (0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)
SIG = (0.3, 0.45, 0.6, 0.8, 1.0, 1.3)
BIA = (0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0)
LEADS = ("d2-3", "d4-5", "d6-7")


class U:
    def __init__(self, f: Path, pop: np.ndarray):
        z = np.load(f, allow_pickle=True)
        self.z = {k: z[k] for k in z.files}
        self.case = z["case"]
        self.cases = sorted(set(self.case.tolist()))
        self.ci = np.array([self.cases.index(c) for c in self.case])
        self.C = len(self.cases)
        s = z["stratum"].astype(int)
        n = np.bincount(s, minlength=len(LZ.EDGES)).astype(float)
        wst = np.where(n > 0, pop / np.maximum(n, 1), 0.0)
        wst *= n.sum() / (wst * n).sum()
        self.w = wst[s]
        self.episode = s >= LZ.EPISODE_FROM
        self.lead = z["lead_h"]
        self.lcls = np.where(self.lead <= 96, 0, np.where(self.lead <= 144, 1, 2))
        self.obs = z["obs"]
        self.R = z["R"]
        self.ok = np.isfinite(self.R[..., 0])
        self.lR = np.log(np.maximum(np.nan_to_num(self.R, nan=1e-6), 1e-6))
        self.W = z["W"]
        self.model = z["model"] if "model" in z else np.where(self.ok, 0, -1)
        self.zone, self.month = z["zone"], z["month"].astype(int)
        self.n = len(self.case)

    def wn(self, mult: dict) -> np.ndarray:
        x = np.zeros_like(self.W, dtype=np.float64)
        for code, m in mult.items():
            x += np.where(self.model == code, self.W * m, 0.0)
        x = np.where(self.ok, x, 0.0)
        return x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)

    def has(self, code: int) -> np.ndarray:
        return ((self.model == code) & self.ok).any(axis=1)


def p_zone(u: U, Wn, k, s, b):
    return (Wn * ndtr((np.log(b) + u.lR[:, :, k]) / s)).sum(axis=1)


def loco_pick(u: U, k: int, P: np.ndarray, groups=None) -> np.ndarray:
    """P (C, n) candidate probabilities -> for each held-out case the candidate with the best weighted Brier score
    on the other cases (within the same group, e.g. lead class, when groups is given)."""
    o = u.obs[:, k].astype(float)
    err = (P - o[None]) ** 2 * u.w[None]
    out = np.zeros(u.n)
    gs = np.zeros(u.n, int) if groups is None else groups
    for g in np.unique(gs):
        sg = gs == g
        tot = err[:, sg].sum(axis=1)
        for c in range(u.C):
            sel = sg & (u.ci == c)
            if sel.any():
                out[sel] = P[int(np.argmin(tot - err[:, sel].sum(axis=1))), sel]
    return out


def ridge(X, y, w, lam):
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    pen = np.full(d + 1, lam); pen[0] = 0.0
    for _ in range(60):
        p = expit(Xb @ beta)
        g = Xb.T @ (w * (p - y)) + pen * beta
        H = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + np.diag(pen + 1e-9)
        st = np.linalg.solve(H, g)
        beta -= st
        if np.abs(st).max() < 1e-7:
            break
    return beta


def loco_logit(u: U, k: int, X: np.ndarray, lam: float) -> tuple[np.ndarray, list]:
    o = u.obs[:, k].astype(float)
    X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = np.zeros(u.n)
    betas = []
    for c in range(u.C):
        tr = u.ci != c
        if o[tr].sum() == 0:
            continue
        b = ridge(Xs[tr], o[tr], u.w[tr], lam)
        betas.append(b)
        out[~tr] = expit(np.hstack([np.ones(((~tr).sum(), 1)), Xs[~tr]]) @ b)
    return out, np.mean(betas, axis=0).round(3).tolist() if betas else []


def summaries(u: U, Wn, k, b=1.6) -> dict:
    lr = u.lR[:, :, k] + np.log(b)
    on = Wn > 0
    mean = np.log(np.maximum((Wn * np.exp(lr)).sum(1), 1e-3))
    fr1 = (Wn * (lr >= 0)).sum(1)
    fr5 = (Wn * (lr >= np.log(0.5))).sum(1)
    order = np.argsort(np.where(on, lr, -99), axis=1)
    lrs = np.take_along_axis(np.where(on, lr, -99), order, 1)
    cw = np.cumsum(np.take_along_axis(Wn, order, 1), 1)
    q90 = np.take_along_axis(lrs, np.argmax(cw >= 0.9, axis=1)[:, None], 1)[:, 0]
    lg = lambda f: np.log((f + 0.02) / (1.02 - f))      # noqa: E731
    return {"mean": mean, "fr1": lg(fr1), "fr5": lg(fr5), "q90": np.maximum(q90, -6), "lead": (u.lead - 48) / 96}


def rank_feature(u: U, stat: np.ndarray) -> np.ndarray:
    """Percentile of `stat` in the model climate of the same zone and season (month +-1), built from the forecasts of
    the OTHER cases, every day weighted by the real frequency of its kind (an EFI/SOT-like relative signal)."""
    out = np.full(u.n, 0.5)
    for zid in np.unique(u.zone):
        iz = np.nonzero(u.zone == zid)[0]
        for i in iz:
            dm = np.minimum((u.month[iz] - u.month[i]) % 12, (u.month[i] - u.month[iz]) % 12)
            cl = iz[(dm <= 1) & (u.ci[iz] != u.ci[i])]
            if len(cl) < 20:
                cl = iz[u.ci[iz] != u.ci[i]]
            ww = u.w[cl]
            out[i] = (ww * (stat[cl] < stat[i])).sum() / max(ww.sum(), 1e-12) + 0.5 * (ww * (stat[cl] == stat[i])).sum() / max(ww.sum(), 1e-12)
    return out


def ingredients(u: U) -> np.ndarray | None:
    ing = u.z.get("ingr")
    if ing is None or not np.isfinite(ing).any():
        return None
    X = ing.astype(np.float64).copy()
    X[:, 2] = np.log1p(np.maximum(X[:, 2], 0))
    z = X[:, 3].copy()
    for m in range(1, 13):                       # z500 anomaly against the month mean of the sample
        sel = (u.month == m) & np.isfinite(z)
        if sel.any():
            X[sel, 3] = z[sel] - np.average(z[sel], weights=u.w[sel])
    return X


# ------------------------------------------------------------------------------------------ verification

def cont(dec, o, w):
    h, m, fa = (w * (dec & o)).sum(), (w * (~dec & o)).sum(), (w * (dec & ~o)).sum()
    return {"POD": round(h / (h + m), 3) if h + m else None, "FAR": round(fa / (h + fa), 3) if h + fa else None,
            "CSI": round(h / (h + m + fa), 3) if h + m + fa else None, "hits": round(float(h), 1),
            "misses": round(float(m), 1), "false_alarms": round(float(fa), 1)}


def base_loco(u: U, k):
    o = u.obs[:, k]
    b = np.zeros(u.n)
    for c in range(u.C):
        tr = u.ci != c
        b[~tr] = (u.w[tr] * o[tr]).sum() / u.w[tr].sum()
    return b


def bss(u: U, p, k, sel=None):
    sel = np.ones(u.n, bool) if sel is None else sel
    o = u.obs[sel, k].astype(float)
    w = u.w[sel]
    ref = (w * (base_loco(u, k)[sel] - o) ** 2).sum()
    return round(float(1 - (w * (p[sel] - o) ** 2).sum() / ref), 4) if ref > 0 else None


def roc_area(u: U, p, k):
    o = u.obs[:, k]
    if o.sum() == 0 or (~o).sum() == 0:
        return None
    order = np.argsort(-p)
    w, oo = u.w[order], o[order]
    tp = np.cumsum(w * oo) / (w * oo).sum()
    fp = np.cumsum(w * ~oo) / (w * ~oo).sum()
    return round(float(np.trapezoid(np.r_[0, tp], np.r_[0, fp])), 3)


def nested(u: U, p, k, extra=None):
    """Decisions with tau chosen on the other cases; `extra` (bool) = raise to this level anyway (agreement rule)."""
    o = u.obs[:, k]
    dec = np.zeros(u.n, bool)
    picks = []
    for c in range(u.C):
        tr = u.ci != c
        best, bt = -1.0, TAUS[0]
        cs = []
        for t in TAUS:
            d = p[tr] >= t
            if extra is not None:
                d |= extra[tr]
            cs.append(cont(d, o[tr], u.w[tr])["CSI"] or 0.0)
        top = max(cs)
        bt = max(t for t, x in zip(TAUS, cs) if x >= top - 0.005)
        picks.append(bt)
        dec[~tr] = p[~tr] >= bt
    if extra is not None:
        dec |= extra
    return dec, picks


def decisions_table(u: U, dec, k):
    o = u.obs[:, k]
    out = {"combined": cont(dec, o, u.w), "episodes": cont(dec[u.episode], o[u.episode], u.w[u.episode]),
           "quiet": cont(dec[~u.episode], o[~u.episode], u.w[~u.episode])}
    for i, n in enumerate(LEADS):
        s = u.lcls == i
        if s.any():
            out[n] = cont(dec[s], o[s], u.w[s])
    return out


def main(units_file: Path, out: Path) -> None:
    import day_climatology as DC
    pop = np.array(DC.load()["frequency"], float)
    u = U(units_file, pop)
    have_ifs, have_aifs = u.has(1), u.has(2)
    res = {"units": u.n, "cases": u.C, "issue_times": int(len(set(u.z["issue"].tolist()))),
           "units_by_lead": {n: int((u.lcls == i).sum()) for i, n in enumerate(LEADS)},
           "units_with_ifs": int(have_ifs.sum()), "units_with_aifs": int(have_aifs.sum()),
           "events_sample": u.obs.sum(0).tolist(), "events_combined": (u.w[:, None] * u.obs).sum(0).round(1).tolist(),
           "levels": {}}
    sets = {"ens": {0: 1.0}, "prod": {0: 1.0, 1: 1.0}, "all": {0: 1.0, 1: 1.0, 2: 1.0},
            "prod_ifs_x0.25": {0: 1.0, 1: 0.25}, "prod_ifs_x4": {0: 1.0, 1: 4.0}, "ens_aifs": {0: 1.0, 2: 1.0}}
    if not have_ifs.any():
        sets = {k: v for k, v in sets.items() if k in ("ens",)}
    elif not have_aifs.any():
        sets = {k: v for k, v in sets.items() if 2 not in v}
    WN = {k: u.wn(v) for k, v in sets.items()}
    X_ing = ingredients(u)
    probs = {}
    for k, L in enumerate(LEVELS):
        o = u.obs[:, k]
        if o.sum() < 5:
            continue
        lv = {}
        cand = {}
        if "pcell" in u.z and len(u.z["pcell"]):
            cand["today_cellmax"] = u.z["pcell"][:, k].astype(float)
        for sn, Wn in WN.items():
            cand[f"zone_prod_{sn}"] = p_zone(u, Wn, k, 1.0, 1.6)
            P = np.stack([p_zone(u, Wn, k, s, b) for s in SIG for b in BIA])
            cand[f"zone_fit_{sn}"] = loco_pick(u, k, P)
            cand[f"zone_fit_lead_{sn}"] = loco_pick(u, k, P, u.lcls)
        best_set = max(WN, key=lambda sn: bss(u, cand[f"zone_fit_{sn}"], k) or -9)
        S = summaries(u, WN[best_set], k)
        Sy = summaries(u, WN[best_set], 0)                    # the yellow ratio feeds every level (more events)
        base = np.stack([S["mean"], S["fr1"], S["fr5"], S["q90"], S["lead"]], 1)
        rk = rank_feature(u, Sy["mean"])
        rk90 = rank_feature(u, Sy["q90"])
        lg = lambda r: np.log((r + 0.01) / (1.01 - r))        # noqa: E731
        Xr = np.hstack([base, np.stack([lg(rk), lg(rk90)], 1)])
        betas = {}
        for lam in (3.0, 30.0, 300.0):
            cand[f"logit_lam{lam:g}"], betas[f"logit_lam{lam:g}"] = loco_logit(u, k, base, lam)
            cand[f"logit_rank_lam{lam:g}"], betas[f"logit_rank_lam{lam:g}"] = loco_logit(u, k, Xr, lam)
            if X_ing is not None:
                Xi = np.hstack([Xr, X_ing])
                cand[f"logit_rank_ingr_lam{lam:g}"], betas[f"logit_rank_ingr_lam{lam:g}"] = loco_logit(u, k, Xi, lam)
        for name, p in cand.items():
            row = {"BSS": bss(u, p, k), "BSS_episodes": bss(u, p, k, u.episode), "BSS_quiet": bss(u, p, k, ~u.episode),
                   "ROC_area": roc_area(u, p, k), "BSS_by_lead": {n: bss(u, p, k, u.lcls == i) for i, n in enumerate(LEADS) if (u.lcls == i).any()},
                   "fixed_tau": {str(t): cont(p >= t, o, u.w) for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)}}
            dec, picks = nested(u, p, k)
            row["nested"] = {"taus_picked": sorted(set(picks)), **decisions_table(u, dec, k)}
            if name in betas:
                row["mean_coefficients"] = betas[name]
            lv[name] = row
            probs[f"{L}_{name}"] = p
        lv["best_member_set"] = best_set
        # the site today: cell max with tau 0.15 / 0.25 and level cap 3
        if "today_cellmax" in cand:
            t_today = {2: 0.15, 3: 0.25}.get(L)
            dec = cand["today_cellmax"] >= t_today if t_today else np.zeros(u.n, bool)
            lv["today_as_published"] = decisions_table(u, dec, k)
        if L in (4, 5) and "agree_cell" in u.z:
            for flag in ("agree_cell", "agree_cell_all"):
                ag = u.z[flag][:, L - 4].astype(bool)
                lv[f"{flag}_alone"] = decisions_table(u, ag, k)
                for name in ("today_cellmax", f"zone_fit_{best_set}"):
                    if name in cand:
                        if name == "today_cellmax":
                            dec = ag.copy()                       # cap 3: today the rule would be the only source of 4/5
                        else:
                            dec, _ = nested(u, cand[name], k, ag)
                        lv[f"{flag}_on_{name}"] = decisions_table(u, dec, k)
                        if name != "today_cellmax":
                            d0, _ = nested(u, cand[name], k)
                            lv[f"{flag}_on_{name}"]["without_rule"] = decisions_table(u, d0, k)["combined"]
        res["levels"][str(L)] = lv
        short = {n: (r["BSS"], r["ROC_area"], r["nested"]["combined"]["POD"], r["nested"]["combined"]["FAR"])
                 for n, r in lv.items() if isinstance(r, dict) and "BSS" in r}
        print(f"L{L} (BSS, ROC, nested POD, FAR):")
        for n, v in sorted(short.items(), key=lambda x: -(x[1][0] or -9)):
            print(f"   {n:32s} {v}")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".probs.npz"), **probs)
    print("wrote", out)


# ------------------------------------------------------------------------------------- the deployed model

def weighted_quantiles(v, w, probs):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(probs, c, v)


def fit(units_file: Path, out_model: Path, out_eval: Path, mult=None, lam: float = 30.0, levels=(2, 3, 4)) -> None:
    """Fit the zone-day model of backend/riua/core/zoneday.py on every case, with its LOCO evaluation.
    mult: member weight multipliers by model code (0 ENS, 1 IFS, 2 AIFS-ENS); default = production (ENS + IFS)."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from riua.core import zoneday as ZD
    import day_climatology as DC
    pop = np.array(DC.load()["frequency"], float)
    u = U(units_file, pop)
    mult = mult or {0: 1.0, 1: 1.0}
    Wn = u.wn(mult)
    lR = u.lR
    Sy = ZD.summaries(lR[:, :, 0], Wn, u.lead)
    # LOCO: ranks against the other cases' climate (exact), model refitted without the held-out case
    rk, rk90 = rank_feature(u, Sy[:, 0]), rank_feature(u, Sy[:, 3])
    probs_tab = np.linspace(0, 1, 101)
    model = {"version": "zoneday-1", "members": {str(k): v for k, v in mult.items()}, "lam": lam,
             "features": ["mean", "fr1", "fr5", "q90", "lead", "rank_mean", "rank_q90"],
             "climate": {"probs": probs_tab.round(3).tolist(), "mean": [], "q90": []}, "levels": {}}
    for z in range(int(u.zone.max()) + 1):
        rows_m, rows_q = [], []
        for m in range(1, 13):
            dm = np.minimum((u.month - m) % 12, (m - u.month) % 12)
            sel = (u.zone == z) & (dm <= 1)
            if sel.sum() < 20:
                sel = u.zone == z
            rows_m.append(weighted_quantiles(Sy[sel, 0], u.w[sel], probs_tab).round(4).tolist())
            rows_q.append(weighted_quantiles(Sy[sel, 3], u.w[sel], probs_tab).round(4).tolist())
        model["climate"]["mean"].append(rows_m)
        model["climate"]["q90"].append(rows_q)
    ev = {"units": u.n, "cases": u.C, "member_set": {str(k): v for k, v in mult.items()}, "levels": {}}
    P_loco = np.zeros((4, u.n))
    for L in levels:
        k = L - 2
        S = ZD.summaries(lR[:, :, k], Wn, u.lead)
        X = np.hstack([S, ZD.rank_lg(rk)[:, None], ZD.rank_lg(rk90)[:, None]])
        p, _ = loco_logit(u, k, X, lam)
        P_loco[k] = p
        # final model on every case, ranks from the shipped tables (what production computes)
        Xt = ZD.features(u.R, Wn, u.lead, u.zone, u.month, model)[k]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        beta = ridge((Xt - mu) / sd, u.obs[:, k].astype(float), u.w, lam)
        model["levels"][str(L)] = {"mu": mu.round(5).tolist(), "sd": sd.round(5).tolist(), "beta": beta.round(5).tolist()}
    P_loco = np.minimum.accumulate(P_loco, axis=0)
    rng = np.random.default_rng(1)
    for L in levels:
        k = L - 2
        p = P_loco[k]
        dec, picks = nested(u, p, k)
        o = u.obs[:, k].astype(float)
        base = base_loco(u, k)
        num = np.array([(u.w * (p - o) ** 2)[u.ci == c].sum() for c in range(u.C)])
        ref = np.array([(u.w * (base - o) ** 2)[u.ci == c].sum() for c in range(u.C)])
        boot = []
        for _ in range(1000):
            ix = rng.integers(0, u.C, u.C)
            boot.append(1 - num[ix].sum() / max(ref[ix].sum(), 1e-12))
        rel = []
        for a, b in zip([0, .05, .1, .15, .2, .3, .4, .5, .7], [.05, .1, .15, .2, .3, .4, .5, .7, 1.01]):
            s = (p >= a) & (p < b)
            if s.any():
                rel.append({"p": round(float(np.average(p[s], weights=u.w[s])), 3), "observed": round(float(np.average(o[s], weights=u.w[s])), 3),
                            "n": round(float(u.w[s].sum()), 1)})
        ev["levels"][str(L)] = {"BSS": bss(u, p, k), "BSS_p05": round(float(np.percentile(boot, 5)), 4),
                                "BSS_p95": round(float(np.percentile(boot, 95)), 4), "BSS_episodes": bss(u, p, k, u.episode),
                                "BSS_quiet": bss(u, p, k, ~u.episode), "ROC_area": roc_area(u, p, k),
                                "BSS_by_lead": {n: bss(u, p, k, u.lcls == i) for i, n in enumerate(LEADS)},
                                "taus_picked": {str(t): picks.count(t) for t in sorted(set(picks))},
                                "nested": decisions_table(u, dec, k),
                                "fixed_tau": {str(t): decisions_table(u, p >= t, k) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)},
                                "reliability": rel}
        print(f"L{L}: LOCO BSS {ev['levels'][str(L)]['BSS']} (p05 {ev['levels'][str(L)]['BSS_p05']}) ROC {ev['levels'][str(L)]['ROC_area']} "
              f"nested {ev['levels'][str(L)]['nested']['combined']} taus {ev['levels'][str(L)]['taus_picked']}", flush=True)
    # in-sample check of the shipped model (table ranks) against the LOCO probabilities
    Pt = ZD.probabilities(u.R, Wn, u.lead, u.zone, u.month, model)
    ev["shipped_in_sample_BSS"] = {str(L): bss(u, Pt[L - 2], L - 2) for L in levels}
    out_model.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    out_eval.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    np.save(out_eval.with_suffix(".loco.npy"), P_loco)
    print("in-sample (shipped tables)", ev["shipped_in_sample_BSS"], "->", out_model, out_eval)


if __name__ == "__main__":
    import sys as _s
    a = _s.argv[1:]
    get = lambda n, d: a[a.index(n) + 1] if n in a else d          # noqa: E731
    if a and a[0] == "fit":
        ms = get("--members", "0,1")
        fit(Path(get("--units", "hindcast/cache/long_units.npz")), Path(get("--model", "backend/riua/zoneday_model.json")),
            Path(get("--eval", "hindcast/long_eval.json")), {int(x): 1.0 for x in ms.split(",")}, float(get("--lam", "30")))


def to_results(units_file: Path, loco_file: Path, eval_file: Path, res_file: Path, tau: dict, cap: int | None) -> None:
    """Write the days 2-7 block of hindcast/results.json (what make_pages.py shows) from the LOCO probabilities of
    the zone-day model, with the deployed thresholds and level cap."""
    import day_climatology as DC
    u = U(units_file, np.array(DC.load()["frequency"], float))
    P = np.load(loco_file)
    ev = json.loads(eval_file.read_text(encoding="utf-8"))

    def rates(dec, o, w):
        h, m, fa, cn = (w * (dec & o)).sum(), (w * (~dec & o)).sum(), (w * (dec & ~o)).sum(), (w * (~dec & ~o)).sum()
        return {"hits": int(round(h)), "misses": int(round(m)), "false_alarms": int(round(fa)), "correct_negatives": int(round(cn)),
                "POD": round(h / (h + m), 3) if h + m else None, "FAR": round(fa / (h + fa), 3) if h + fa else None,
                "CSI": round(h / (h + m + fa), 3) if h + m + fa else None, "freq_bias": round((h + fa) / (h + m), 2) if h + m else None}

    def block(sel):
        out = {}
        for k, L in enumerate(LEVELS):
            dec = (P[k] >= float(tau[str(L)])) & (cap is None or L <= cap)
            out[str(L)] = rates(dec[sel], u.obs[sel, k], u.w[sel])
        return out
    allu = np.ones(u.n, bool)
    rel = {}
    for L in ("2", "3", "4"):
        rel[L] = [{"p_forecast": r["p"], "observed_freq": r["observed"], "n": r["n"]} for r in ev["levels"][L]["reliability"]]
    R = json.loads(res_file.read_text(encoding="utf-8")) if res_file.exists() else {}
    R["long"] = {
        "horizon": "long", "scale": "zone", "cases": u.cases, "n_frames": int(len(set(zip(u.z["issue"].tolist(), u.z["day"].tolist())))),
        "n_issue_times": int(len(set(u.z["issue"].tolist()))), "n_zone_days": u.n,
        "tuned": {"model": "zoneday (core/zoneday.py)", "sigma": None, "bias": None, "tau": tau,
                  "loco_BSS": [ev["levels"][L]["BSS"] for L in ("2", "3", "4")] + [None],
                  "loco_BSS_p05": [ev["levels"][L]["BSS_p05"] for L in ("2", "3", "4")] + [None]},
        "level_cap": cap, "tau": tau, "deployed_tau": tau, "deployed_cap": cap,
        "deployed": {"zone_day": block(allu)},
        "deployed_split": {"combined": {"zone_day": block(allu)}, "episodes": {"zone_day": block(u.episode)},
                           "quiet": {"zone_day": block(~u.episode)}},
        "deployed_by_lead": {n: {"zone_day": block(u.lcls == i)} for i, n in enumerate(LEADS)},
        "reliability": rel,
        "note": "q13-long: issue 08Z with the 00Z ECMWF ENS + the 2 newest IFS runs (production's members), every lead "
                "day 2..7, warning zone x UTC day, leave-one-case-out, days weighted by the real frequency of their kind",
    }
    res_file.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(json.dumps(R["long"]["deployed"], indent=0)[:1500])


if __name__ == "__main__":
    import sys as _s
    a = _s.argv[1:]
    get = lambda n, d: a[a.index(n) + 1] if n in a else d          # noqa: E731
    if a and a[0] == "results":
        to_results(Path(get("--units", "hindcast/cache/long_units.npz")), Path(get("--loco", "hindcast/long_eval.loco.npy")),
                   Path(get("--eval", "hindcast/long_eval.json")), Path(get("--results", "hindcast/results.json")),
                   json.loads(get("--tau", '{"2": 0.3, "3": 0.2, "4": 0.2, "5": 0.5}')), int(get("--cap", "3")))
