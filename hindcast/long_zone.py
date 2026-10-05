"""Days 2-7 at warning-zone scale: P(threshold reached anywhere in the zone that UTC day), scored out of sample.

    python hindcast/long_zone.py units [--blocks DIR] [--strata FILE]   long blocks -> hindcast/cache/long_units.npz
    python hindcast/long_zone.py score [--units FILE]                   every method, leave-one-case-out -> long_zone.json

A unit = (issue time, warning zone, valid UTC day): the unit the site already reports for days 2-7
(score.Tally.units). The event is "some cell of the zone reaches level L" (1-h or 12-h criterion, the truth
of the block, i.e. the radar + SAIH analysis with the neighbourhood of the block's radius).

Methods (all with a kernel / model fitted without the held-out case):
  cellmax   production: largest CELL probability in the zone (what the site shows today)
  zone      member-wise zone maximum: P = sum_m w_m Phi(ln(b * r_m) / s), r_m = max over the zone's cells of
            s12 * a12 / T12_L for member m; (s, b) per level class fitted on the Brier score
  rank      relative signal: the member zone maxima ranked against the model's own climate of zone maxima for that
            zone and season (an EFI/SOT-like index built from the same runs at other times), then logistic
  logit     small logistic model on zone-day ensemble summaries (log mean ratio, exceedance fraction, q90, lead),
            ridge penalty, LOCO
Days are weighted with the real frequency of each kind of day (hindcast/day_climatology.json), as score.py does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr, expit

HC = Path(__file__).resolve().parent
ROOT = HC.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HC))

LEVELS = (2, 3, 4, 5)
EDGES = (0.0, 1.0, 10.0, 30.0, 60.0, 100.0)
EPISODE_FROM = 3


def _arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


# ------------------------------------------------------------------------------------------- units

def build_units(blocks: Path, strata_file: Path, out: Path) -> None:
    from riua import params as P, static
    par = P.load()
    st = static.load()
    thr = static.thresholds(par)
    zone = st.zone_idx.ravel()
    t1 = thr.t1h.reshape(4, -1)
    t12 = thr.t12h.reshape(4, -1)
    strata = json.loads(strata_file.read_text(encoding="utf-8"))
    fam = par["families"]
    rows = {k: [] for k in ("case", "issue", "day", "lead_h", "zone", "stratum", "obs", "R", "w", "month", "model",
                            "agree_cell", "agree_cell_all", "ingr")}
    MMAX = 0
    import long_data as LD
    LD.ENS = Path(_arg("--ens", str(LD.ENS)))
    from datetime import datetime, timedelta
    codes = {"ifs_ens": 0, "ifs": 1, "aifs_ens": 2}
    ingr_cache = {}
    files = sorted(blocks.glob("[0-9]*.npz"))
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        cells = z["cells"]
        zc = zone[cells]
        a12 = z["a12"].astype(np.float32)                        # (M, F, N)
        s12 = np.array([fam[str(x)]["s12h"] for x in z["family"]], np.float32)
        hl = par["age_halflife_h"]["long"]
        w = np.array([fam[str(x)]["weight"] for x in z["family"]]) * z["mw"] * 0.5 ** (z["age_h"] / hl)
        valid = z["valid"]
        o1, o12 = z["o1"].astype(np.float32), z["o12"].astype(np.float32)
        MMAX = max(MMAX, a12.shape[0])
        mcode = np.array([codes.get(str(x), 9) for x in z["model"]], np.int8)
        run = datetime.fromisoformat(meta["issue"][:10])
        for fi in range(a12.shape[1]):
            day = str(z["t0"][fi])[:10]
            s = strata.get(day)
            if s is None:
                continue
            step = int((datetime.fromisoformat(day) - run) / timedelta(hours=1)) + 12
            if (run, step) not in ingr_cache:
                ingr_cache[(run, step)] = LD.ingredient_scalars(run, step) or {}
            ing = ingr_cache[(run, step)]
            # production's model-agreement rule (risk.model_agreement), per cell, red (k=2) and extreme (k=3):
            # ENS group votes where >= 30 % of its weight reaches the threshold, the IFS runs where any does
            x_all = a12[:, fi] * s12[:, None]                            # (M, N)
            vv = valid[:, fi]
            def votes(k, groups):
                hit = (np.nan_to_num(x_all) >= t12[k, cells][None]) & vv[:, None]
                v = np.zeros(len(cells), np.int16)
                for code, is_ens in groups:
                    g = mcode == code
                    if not g.any():
                        continue
                    if is_ens:
                        ww = w[g] * vv[g]
                        v += ((hit[g] * ww[:, None]).sum(0) / max(ww.sum(), 1e-12)) >= 0.3
                    else:
                        v += hit[g].any(0)
                return v
            vprod = [votes(k, ((0, True), (1, False))) for k in (2, 3)]
            vall = [votes(k, ((0, True), (1, False), (2, True))) for k in (2, 3)]
            for zid in np.unique(zc[zc >= 0]):
                cs = np.nonzero(zc == zid)[0]
                ob = [bool(((o1[fi, cs] >= t1[k, cells[cs]]) | (o12[fi, cs] >= t12[k, cells[cs]])).any()) for k in range(4)]
                x = a12[:, fi][:, cs] * s12[:, None]                     # (M, ncs)
                with np.errstate(invalid="ignore"):
                    R = np.stack([np.nanmax(x / t12[k, cells[cs]][None], axis=1) for k in range(4)], axis=1)   # (M, 4)
                R[~valid[:, fi]] = np.nan
                # what the site shows today: production kernel per CELL (ENS + IFS runs), largest cell value of the zone
                pm = np.isin(mcode, (0, 1)) & valid[:, fi]
                wp = w * pm
                pc = []
                for k in range(4):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ph = ndtr(np.log(np.maximum(par["bias"]["long"] * np.nan_to_num(x[pm]) / t12[k, cells[cs]][None], 1e-9)) / par["sigma"]["long"])
                    pc.append(float(((wp[pm][:, None] * ph).sum(0) / max(wp.sum(), 1e-12)).max()))
                rows.setdefault("pcell", []).append(pc)
                rows["case"].append(meta["case"]); rows["issue"].append(meta["issue"]); rows["day"].append(day)
                rows["lead_h"].append(float(z["lead_h"][fi])); rows["zone"].append(int(zid)); rows["stratum"].append(int(s))
                rows["obs"].append(ob); rows["R"].append(R); rows["w"].append(w * valid[:, fi]); rows["month"].append(int(day[5:7]))
                rows["model"].append(mcode)
                rows["agree_cell"].append([bool((v[cs] >= 2).any()) for v in vprod])
                rows["agree_cell_all"].append([bool((v[cs] >= 2).any()) for v in vall])
                rows["ingr"].append([ing.get(k, np.nan) for k in ("tcwv", "flux850", "cape", "z500")])
    U = len(rows["case"])
    R = np.full((U, MMAX, 4), np.nan, np.float32)
    W = np.zeros((U, MMAX), np.float32)
    MC = np.full((U, MMAX), -1, np.int8)
    for u in range(U):
        m = rows["R"][u].shape[0]
        R[u, :m], W[u, :m], MC[u, :m] = rows["R"][u], rows["w"][u], rows["model"][u]
    np.savez_compressed(out, case=np.array(rows["case"]), issue=np.array(rows["issue"]), day=np.array(rows["day"]),
                        lead_h=np.array(rows["lead_h"], np.float32), zone=np.array(rows["zone"], np.int16),
                        stratum=np.array(rows["stratum"], np.int8), obs=np.array(rows["obs"], bool), R=R, W=W,
                        month=np.array(rows["month"], np.int8), model=MC, agree_cell=np.array(rows["agree_cell"], bool),
                        agree_cell_all=np.array(rows["agree_cell_all"], bool), ingr=np.array(rows["ingr"], np.float32),
                        pcell=np.array(rows.get("pcell", []), np.float32))
    print(f"{len(files)} blocks -> {U} zone-day units, {len(set(rows['case']))} cases -> {out}")


# ------------------------------------------------------------------------------------------- scoring

class Units:
    def __init__(self, f: Path, pop: np.ndarray):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            setattr(self, k, z[k])
        self.cases = sorted(set(self.case.tolist()))
        self.ci = np.array([self.cases.index(c) for c in self.case])
        n = np.bincount(self.stratum, minlength=len(EDGES)).astype(float)
        wst = np.where(n > 0, pop / np.maximum(n, 1), 0.0)
        wst *= n.sum() / (wst * n).sum()                         # equivalent counts
        self.wu = wst[self.stratum]                              # weight of each unit ("combined")
        self.episode = self.stratum >= EPISODE_FROM
        self.lcls = np.where(self.lead_h <= 96, 0, np.where(self.lead_h <= 144, 1, 2))
        Wn = self.W / np.maximum(self.W.sum(axis=1, keepdims=True), 1e-12)
        self.Wn = np.where(np.isfinite(self.R[..., 0]), Wn, 0.0)
        self.Wn /= np.maximum(self.Wn.sum(axis=1, keepdims=True), 1e-12)
        self.lR = np.log(np.maximum(np.nan_to_num(self.R, nan=1e-6), 1e-6))     # (U, M, 4)


def p_zone(u: Units, k: int, s: float, b: float) -> np.ndarray:
    return (u.Wn * ndtr((np.log(b) + u.lR[:, :, k]) / s)).sum(axis=1)


def brier(p, o, w):
    return float((w * (p - o) ** 2).sum() / w.sum())


def bss(p, o, w, base):
    ref = (w * (base - o) ** 2).sum() / w.sum()
    return 1.0 - brier(p, o, w) / ref if ref > 0 else float("nan")


def contingency(p, o, w, tau):
    f = p >= tau
    h, m, fa = (w * (f & o)).sum(), (w * (~f & o)).sum(), (w * (f & ~o)).sum()
    return {"POD": round(h / (h + m), 3) if h + m else None, "FAR": round(fa / (h + fa), 3) if h + fa else None,
            "CSI": round(h / (h + m + fa), 3) if h + m + fa else None, "hits": round(float(h), 1),
            "misses": round(float(m), 1), "false_alarms": round(float(fa), 1)}


def loco_grid(u: Units, k: int, cands: list, prob_fn) -> np.ndarray:
    """Out-of-sample probabilities: for each held-out case the candidate with the best Brier score on the others."""
    o = u.obs[:, k].astype(float)
    P = np.stack([prob_fn(c) for c in cands])                 # (C, U)
    err = (P - o[None]) ** 2 * u.wu[None]
    out = np.zeros(len(o))
    tot = err.sum(axis=1)
    for c in range(len(u.cases)):
        sel = u.ci == c
        best = int(np.argmin(tot - err[:, sel].sum(axis=1)))
        out[sel] = P[best, sel]
    return out


def ridge_logit(X, y, w, lam):
    """Weighted logistic regression with an L2 penalty (intercept free), Newton iterations."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    pen = np.full(d + 1, lam); pen[0] = 0.0
    for _ in range(50):
        p = expit(Xb @ beta)
        g = Xb.T @ (w * (p - y)) + pen * beta
        H = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + np.diag(pen + 1e-9)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def features(u: Units, k: int, b: float) -> np.ndarray:
    lr = u.lR[:, :, k] + np.log(b)
    ok = u.Wn > 0
    mean = np.log(np.maximum((u.Wn * np.exp(lr)).sum(axis=1), 1e-3))
    frac = (u.Wn * (lr >= 0)).sum(axis=1)
    frac2 = (u.Wn * (lr >= np.log(0.5))).sum(axis=1)
    srt = np.sort(np.where(ok, lr, -99), axis=1)
    nm = ok.sum(axis=1)
    q90 = srt[np.arange(len(lr)), np.clip(srt.shape[1] - np.maximum((nm * 0.1).astype(int), 1), 0, srt.shape[1] - 1)]
    lead = (u.lead_h - 48.0) / 96.0
    eps = 0.02
    return np.stack([mean, np.log((frac + eps) / (1 - frac + eps)), np.log((frac2 + eps) / (1 - frac2 + eps)),
                     np.maximum(q90, -6), lead], axis=1)


def loco_logit(u: Units, k: int, X: np.ndarray, lam: float, kk: int | None = None) -> np.ndarray:
    o = u.obs[:, k].astype(float)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    out = np.zeros(len(o))
    for c in range(len(u.cases)):
        tr = u.ci != c
        if o[tr].sum() == 0:
            out[~tr] = 0.0
            continue
        beta = ridge_logit(Xs[tr], o[tr], u.wu[tr], lam)
        out[~tr] = expit(np.hstack([np.ones(((~tr).sum(), 1)), Xs[~tr]]) @ beta)
    return out


def nested_tau(u: Units, p: np.ndarray, k: int, taus=(0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)):
    """tau chosen on the other cases (CSI optimum, largest tau within 0.005 of it); decisions on the held-out case."""
    o = u.obs[:, k]
    dec = np.zeros(len(o), bool)
    picks = []
    for c in range(len(u.cases)):
        tr = u.ci != c
        csi = [contingency(p[tr], o[tr], u.wu[tr], t)["CSI"] or 0.0 for t in taus]
        top = max(csi)
        t = max(t for t, x in zip(taus, csi) if x >= top - 0.005)
        picks.append(t)
        dec[~tr] = p[~tr] >= t
    return dec, picks


def table(u: Units, p: np.ndarray, k: int, sel=None, tau=None) -> dict:
    sel = np.ones(len(p), bool) if sel is None else sel
    o = u.obs[:, k]
    w = u.wu
    base_loco = np.zeros(len(p))
    for c in range(len(u.cases)):
        tr = u.ci != c
        base_loco[~tr] = (w[tr] * o[tr]).sum() / w[tr].sum()
    out = {"BSS": round(bss(p[sel], o[sel], w[sel], base_loco[sel]), 4), "events": round(float((w * o)[sel].sum()), 1),
           "events_sample": int(o[sel].sum())}
    if tau is not None:
        out["at_tau"] = {str(t): contingency(p[sel], o[sel], w[sel], t) for t in tau}
    return out


def score(units_file: Path, out: Path) -> None:
    import day_climatology as DC
    pop = np.array(DC.load()["frequency"], float)
    u = Units(units_file, pop)
    res = {"units": int(len(u.case)), "cases": len(u.cases), "issue_times": int(len(set(u.issue.tolist()))),
           "by_lead_class": {n: int((u.lcls == i).sum()) for i, n in enumerate(("d2-3", "d4-5", "d6-7"))}, "levels": {}}
    taus = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
    from riua import params as P
    par = P.load()
    s0, b0 = par["sigma"]["long"], par["bias"]["long"]
    sig = (0.3, 0.45, 0.6, 0.8, 1.0, 1.3)
    bia = (0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0)
    probs = {}
    for k, L in enumerate(LEVELS):
        o = u.obs[:, k]
        if o.sum() < 5:
            continue
        lv = {}
        # cellmax: production kernel on each cell, then the zone maximum. Only the zone max of r is stored per member,
        # and the cell maximum of a monotone per-cell probability is not the probability of the cell max; the exact
        # cell value comes from the blocks (cellmax_exact), this column is the member-wise ZONE value at the
        # production kernel (an upper bound of the cell max)
        p_prod = p_zone(u, k, s0, b0)
        lv["zone_production_kernel"] = table(u, p_prod, k, tau=taus)
        cands = [(s, b) for s in sig for b in bia]
        p_fit = loco_grid(u, k, cands, lambda c: p_zone(u, k, *c))
        lv["zone_fitted_loco"] = table(u, p_fit, k, tau=taus)
        # per lead class kernel
        p_cls = np.zeros(len(o))
        for i in range(3):
            sel = u.lcls == i
            if not sel.any():
                continue
            sub = _subset(u, sel)
            p_cls[sel] = loco_grid(sub, k, cands, lambda c: p_zone(sub, k, *c))
        lv["zone_fitted_by_lead_loco"] = table(u, p_cls, k, tau=taus)
        # logistic on ensemble summaries
        for lam in (1.0, 10.0, 100.0):
            X = features(u, k, 1.0)
            p_lg = loco_logit(u, k, X, lam)
            lv[f"logit_lam{lam:g}"] = table(u, p_lg, k, tau=taus)
            probs[(L, f"logit{lam:g}")] = p_lg
        probs[(L, "zone")] = p_fit
        probs[(L, "zone_prod")] = p_prod
        for name in ("zone",):
            dec, picks = nested_tau(u, probs[(L, name)], k)
            lv[f"{name}_nested_tau"] = {"taus_picked": sorted(set(picks)), **contingency(dec.astype(float), o, u.wu, 0.5),
                                        "episodes": contingency(dec[u.episode].astype(float), o[u.episode], u.wu[u.episode], 0.5),
                                        "quiet": contingency(dec[~u.episode].astype(float), o[~u.episode], u.wu[~u.episode], 0.5)}
        for i, n in enumerate(("d2-3", "d4-5", "d6-7")):
            sel = u.lcls == i
            if sel.any() and o[sel].any():
                lv.setdefault("by_lead", {})[n] = {"zone_fitted_loco": table(u, p_fit, k, sel, taus)}
        res["levels"][str(L)] = lv
        print(L, json.dumps({m: (v.get("BSS"), v.get("at_tau", {}).get("0.15")) for m, v in lv.items() if isinstance(v, dict) and "BSS" in v})[:2000])
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".probs.npz"), **{f"{L}_{n}": v for (L, n), v in probs.items()})
    print("wrote", out)


def _subset(u: Units, sel: np.ndarray) -> Units:
    s = Units.__new__(Units)
    for k, v in u.__dict__.items():
        s.__dict__[k] = v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v
    s.cases = sorted(set(s.case.tolist()))
    s.ci = np.array([s.cases.index(c) for c in s.case])
    return s


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "merge":                       # merge <out> <in1> <in2> ... (per-case unit files of the workflow)
        parts = [dict(np.load(f, allow_pickle=True)) for f in sys.argv[3:]]
        parts = [p for p in parts if len(p["case"])]
        M = max(p["R"].shape[1] for p in parts)
        pad = {"R": np.nan, "W": 0.0, "model": -1}
        merged = {}
        for k in parts[0]:
            arrs = []
            for p in parts:
                a = p[k]
                if k in pad and a.shape[1] < M:
                    shp = list(a.shape); shp[1] = M - a.shape[1]
                    a = np.concatenate([a, np.full(shp, pad[k], a.dtype)], axis=1)
                arrs.append(a)
            merged[k] = np.concatenate(arrs)
        np.savez_compressed(sys.argv[2], **merged)
        print(len(parts), "files ->", len(merged["case"]), "units")
    elif cmd == "units":
        build_units(Path(_arg("--blocks", str(HC / "cache" / "blocks" / "long"))),
                    Path(_arg("--strata", str(HC / "cache" / "day_strata.json"))),
                    Path(_arg("--out", str(HC / "cache" / "long_units.npz"))))
    elif cmd == "score0":
        score(Path(_arg("--units", str(HC / "cache" / "long_units.npz"))), Path(_arg("--out", str(HC / "cache" / "long_zone.json"))))
    else:
        import long_score
        long_score.main(Path(_arg("--units", str(HC / "cache" / "long_units.npz"))), Path(_arg("--out", str(HC / "cache" / "long_zone.json"))))
