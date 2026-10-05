"""Fit and score the hindcast blocks written by run.py.

    python hindcast/run.py fit [--quick] [now|mid|long]     (or: python hindcast/score.py ...)
    python hindcast/score.py check                           the fast dressing against risk.dressed_probabilities

What is computed, per horizon (and per lead class inside it):
  1. Brier skill of the scenario dressing as production computes it (params.json) and of the fitted kernel:
     sigma / bias of the 12-h term, sigma1h / bias1h of the 1-h term, with leave-one-case-out.
  2. Weight experiments (ENS against the deterministic runs, age half-life), the neighbourhood radius, the gate
     on measured rain. Criterion: the Brier score, never the contingency table.
  3. With out-of-sample probabilities (each case scored with the kernel fitted without it): reliability, and
     hits / misses / false alarms per cell-frame (all published cells, warning zones only) and per warning zone,
     valid day and issue time, for every probability threshold tau -> the full curve, the CSI optimum and the
     threshold a cost/loss ratio implies.
Everything is given for rain-episode days, for ordinary / dry days and for the two combined with the real
frequency of each kind of day (hindcast/day_climatology.json): the sample is mostly episodes, a year is not.

Writes hindcast/results.json (what make_pages.py reads + the split), hindcast/fit.json (every table) and
hindcast/cache/score_state.pkl (histograms: deploy_params.py scores any tau table from it in a second).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import day_climatology as DC  # noqa: E402
import run as R  # noqa: E402
from riua import product  # noqa: E402
from riua.core import grid, risk  # noqa: E402

HC = R.HC
LEVELS = (2, 3, 4, 5)
NS = len(DC.EDGES)
NB = 200                                             # probability bins of 0.005
TAUS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
SIG = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3)
BIA = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0, 2.5)
SIG1 = (0.5, 0.65, 0.8, 1.0, 1.3, 1.6)
BIA1 = (0.5, 0.6, 0.75, 0.9, 1.0, 1.2, 1.5, 2.0)
CLASSES = {"now": ("h1-2", "h3-4", "h5-6"), "mid": ("d1", "d2"), "long": ("d2-3", "d4-5", "d6-7")}
JJ, II = np.mgrid[0:grid.NY, 0:grid.NX]
# cells the kernel is fitted on: one in FIT_STRIDE each way (4 = every 0.2 deg, about 20 km: the events are
# neighbourhood maxima within 6-12 km, so neighbouring cells repeat each other). The scores use every cell.
FIT_STRIDE = int(os.environ.get("RIUA_FIT_STRIDE", "4"))
LATTICE = ((JJ % FIT_STRIDE == 0) & (II % FIT_STRIDE == 0)).ravel()
ZONE = R.ZONE.ravel()
T1 = R.THR.t1h.reshape(4, -1)
T12 = R.THR.t12h.reshape(4, -1)
FAMS = list(R.PARAMS["families"])


def lead_class(hz: str, lead_h: float) -> str:
    """Same classes as product.lead_class (lead of the frame end)."""
    if hz == "now":
        return "h1-2" if lead_h <= 2 else "h3-4" if lead_h <= 4 else "h5-6"
    if hz == "mid":
        return "d1" if lead_h <= 30 else "d2"
    return "d2-3" if lead_h <= 96 else "d4-5" if lead_h <= 144 else "d6-7"


def theta_production(hz: str, params: dict | None = None) -> dict:
    p = params or R.PARAMS
    s, b = p["sigma"][hz], p["bias"][hz]
    return {"sigma": s, "bias": b, "sigma1": p.get("sigma1h", {}).get(hz, s), "bias1": p.get("bias1h", {}).get(hz, b),
            "sigma_obs": p.get("sigma_obs", 0.15), "of": p.get("obs12_factor", 1.0), "hl": p["age_halflife_h"][hz],
            "fw": {f: v["weight"] for f, v in p["families"].items()}, "mult": {}}


# -------------------------------------------------------------------------------------------- items

class Item:
    """The part of one block that belongs to one lead class, reduced to the cell-frames where at least one
    scenario has rain ("wet pairs"); everywhere else every probability is zero whatever the kernel."""
    __slots__ = ("case", "issue", "tier", "ens", "cls", "hz", "l12", "ok12", "l1", "i1", "phi", "val", "fam", "model", "age",
                 "mw", "lt1", "lt12", "o", "of", "sidx", "E", "Ef", "N", "dry_o", "dry_of", "fi", "ni", "F", "ncell", "cells",
                 "days", "strata", "dry_mask", "o_full", "of_full")


def block_items(f: Path, var: str = "", cells: str = "fit", truth_var: str | None = None, keep_full: bool = False) -> list[Item]:
    """var: '' (production), '@r<km>' (neighbourhood radius) or '@g<mm>' (gate on measured rain).
    truth_var: radius tag of the truth ('' = the event of the production radius); default: the variant's own."""
    z = np.load(f, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    hz = meta["hz"]
    rtag = var if var.startswith("@r") else ""
    if f"a12{var}" not in z.files:
        return []
    c_all = z["cells"]
    sel = np.nonzero(LATTICE[c_all])[0] if cells == "fit" else np.arange(len(c_all))
    if sel.size == 0:
        return []
    cc = c_all[sel]
    a12 = z["a12" + var][:, :, sel].astype(np.float32)
    has1 = z["has1"]
    a1 = z["a1" + rtag][:, :, sel].astype(np.float32)
    phi = z["phi" + var][:, :, sel].astype(np.float32) / 250.0 if ("phi" + var) in z.files else None
    ttag = rtag if truth_var is None else truth_var
    o1, o12 = z["o1" + ttag][:, sel].astype(np.float32), z["o12" + ttag][:, sel].astype(np.float32)
    gtag = var if var.startswith("@g") else ""
    o12f = z["o12f" + gtag][:, sel].astype(np.float32) if (("o12f" + gtag) in z.files and ttag == "") else o12
    fam = np.array([FAMS.index(x) for x in z["family"]])
    s1 = np.array([R.PARAMS["families"][FAMS[k]]["s1h"] for k in fam], np.float32)
    s12 = np.array([R.PARAMS["families"][FAMS[k]]["s12h"] for k in fam], np.float32)
    t1, t12 = T1[:, cc], T12[:, cc]
    cls = np.array([lead_class(hz, x) for x in z["lead_h"]])
    days = np.array([str(t)[:10] for t in z["t0"]])
    strata = np.array([DC.stratum_of(d) if DC.stratum_of(d) is not None else -1 for d in days])
    out = []
    for c in CLASSES[hz]:
        fs = np.nonzero((cls == c) & (strata >= 0))[0]
        if fs.size == 0:
            continue
        A12, A1 = a12[:, fs], a1[:, fs]
        o = np.stack([(o1[fs] >= t1[k]) | (o12[fs] >= t12[k]) for k in range(4)])
        of = np.stack([(o1[fs] >= t1[k]) | (o12f[fs] >= t12[k]) for k in range(4)])
        with np.errstate(invalid="ignore"):
            wet = (A12 > 0.05).any(axis=0)
            if A1.shape[0]:
                wet |= (A1 > 0.02).any(axis=0)
        fi, ni = np.nonzero(wet)
        it = Item()
        it.case, it.issue, it.tier, it.ens, it.cls, it.hz = meta["case"], meta["issue"], meta["tier"], meta.get("ens"), c, hz
        with np.errstate(divide="ignore", invalid="ignore"):
            x = A12[:, fi, ni] * s12[:, None]
            # not available -> a log amount of -1e9: the kernel gives exactly 0 there, as production does
            it.ok12 = np.isfinite(x)                                  # here: usable on either criterion
            it.l12 = np.where(it.ok12, np.log(np.maximum(x, 1e-12)), -1e9).astype(np.float32)
            x1 = A1[:, fi, ni] * s1[has1][:, None]
            it.l1 = np.where(np.isfinite(x1), np.log(np.maximum(x1, 1e-12)), -1e9).astype(np.float32)
            if len(has1):
                it.ok12[has1] |= np.isfinite(x1)
        it.i1 = has1
        it.phi = phi[:, fs][:, fi, ni] if phi is not None and phi[:, fs].any() else None
        it.val = z["valid"][:, fs][:, fi]
        it.fam, it.model, it.age, it.mw = fam, z["model"], z["age_h"].astype(np.float32), z["mw"].astype(np.float32)
        it.lt1, it.lt12 = np.log(t1[:, ni]).astype(np.float32), np.log(t12[:, ni]).astype(np.float32)
        it.o, it.of = o[:, fi, ni], of[:, fi, ni]
        sf = strata[fs]
        it.sidx = sf[fi]
        it.N = np.bincount(sf, minlength=NS).astype(np.float64) * len(cc)
        dry = ~wet
        it.E = np.stack([np.bincount(sf, weights=o[k].sum(axis=1), minlength=NS) for k in range(4)], axis=1)       # (S, 4)
        it.Ef = np.stack([np.bincount(sf, weights=of[k].sum(axis=1), minlength=NS) for k in range(4)], axis=1)
        it.dry_o = np.stack([np.bincount(sf, weights=(o[k] & dry).sum(axis=1), minlength=NS) for k in range(4)], axis=1)
        it.dry_of = np.stack([np.bincount(sf, weights=(of[k] & dry).sum(axis=1), minlength=NS) for k in range(4)], axis=1)
        it.fi, it.ni, it.F, it.ncell, it.cells = fi, ni, len(fs), len(cc), cc
        it.days, it.strata = days[fs], sf
        it.o_full, it.of_full = (o, of) if keep_full else (None, None)
        it.dry_mask = {}                      # memo: kernel -> item_stats (shared by every FitSet that holds the item)
        out.append(it)
    return out


def item_prob(it: Item, th: dict) -> np.ndarray:
    """(4, K) probabilities at the wet pairs: the formula of risk.dressed_probabilities."""
    K = it.l12.shape[1]
    if K == 0:
        return np.zeros((4, 0), np.float32)
    fw = np.array([th["fw"].get(f, 0.0) for f in FAMS], np.float64)[it.fam]
    w = fw * it.mw * 0.5 ** (it.age / th["hl"])
    for model, x in (th.get("mult") or {}).items():
        w = np.where(it.model == model, w * x, w)
    W = (w[:, None] * it.val).astype(np.float32)                     # (M, K)
    sigma, bias = th["sigma"], th["bias"]
    if it.phi is None:
        b12, sg12 = np.float32(np.log(bias)), np.float32(sigma)
    else:
        b12 = np.log(it.phi * th["of"] + (1.0 - it.phi) * bias).astype(np.float32)
        sg12 = np.maximum(sigma * (1.0 - it.phi), min(sigma, th["sigma_obs"])).astype(np.float32)
    lb1, s1 = np.float32(np.log(th["bias1"])), np.float32(th["sigma1"])
    den = np.maximum((W * it.ok12).sum(axis=0), 1e-12)
    out = np.zeros((4, K), np.float32)
    for k in range(4):
        pb = ndtr((b12 + it.l12 - it.lt12[k]) / sg12)
        if len(it.i1):
            pb[it.i1] = np.maximum(pb[it.i1], ndtr((lb1 + it.l1 - it.lt1[k]) / s1))
        out[k] = (W * pb).sum(axis=0) / den
    return np.minimum.accumulate(out, axis=0)


def item_stats(it: Item, th: dict, fwd: bool = False) -> np.ndarray:
    """(S, 4) sum of (p - o)^2 per day stratum and level (dry pairs included: p = 0)."""
    p = item_prob(it, th)
    o = it.of if fwd else it.o
    out = (it.dry_of if fwd else it.dry_o).astype(np.float64).copy()
    if p.shape[1]:
        d = (p - o) ** 2
        for k in range(4):
            out[:, k] += np.bincount(it.sidx, weights=d[k], minlength=NS)
    return out


# --------------------------------------------------------------------------------------- fit engine

class FitSet:
    def __init__(self, items: list[Item], pop: np.ndarray, fwd: bool = False):
        self.items, self.fwd = items, fwd
        self.cases = sorted({it.case for it in items})
        ci = {c: k for k, c in enumerate(self.cases)}
        self.cidx = np.array([ci[it.case] for it in items])
        C = len(self.cases)
        self.N = np.zeros((C, NS)); self.E = np.zeros((C, NS, 4))
        for it, c in zip(items, self.cidx):
            self.N[c] += it.N
            self.E[c] += it.Ef if fwd else it.E
        tot = self.N.sum(axis=0)
        have = tot > 0
        self.W = np.where(have, pop / np.maximum(tot, 1), 0.0)
        self.W /= max((self.W * tot).sum(), 1e-12)                  # weights of all units add up to 1
        self.cache = {}
        # levels with enough events to be fitted on: 30 cell-frames from at least 2 cases
        self.eligible = [(self.E[:, :, k].sum() >= 30) and ((self.E[:, :, k].sum(axis=1) > 0).sum() >= 2) for k in range(4)]

    def table(self, th: dict) -> np.ndarray:
        key = json.dumps(th, sort_keys=True)
        if key not in self.cache:
            T = np.zeros((len(self.cases), NS, 4))
            for it, c in zip(self.items, self.cidx):
                k2 = (key, self.fwd)
                if k2 not in it.dry_mask:
                    it.dry_mask[k2] = item_stats(it, th, self.fwd)
                T[c] += it.dry_mask[k2]
            self.cache[key] = T
        return self.cache[key]

    def bss(self, T: np.ndarray, use: np.ndarray | None = None, W: np.ndarray | None = None):
        """-> (mean over eligible levels, [per level]) on the cases in `use`, strata weighted with W."""
        use = np.ones(len(self.cases), bool) if use is None else use
        W = self.W if W is None else W
        n = (self.N[use] * W).sum()
        if n <= 0:
            return float("nan"), [None] * 4
        bs = (T[use] * W[None, :, None]).sum(axis=(0, 1)) / n
        ob = (self.E[use] * W[None, :, None]).sum(axis=(0, 1)) / n
        ref = ob * (1 - ob)
        per = [float(1 - bs[k] / ref[k]) if (ref[k] > 0 and self.eligible[k]) else None for k in range(4)]
        ok = [x for x in per if x is not None]
        return (float(np.mean(ok)) if ok else float("nan")), per

    def score(self, th: dict, use=None) -> float:
        return self.bss(self.table(th), use)[0]

    def loco(self, thetas: list[dict]) -> dict:
        """Leave one case out: the kernel is chosen on the other cases, the held-out case is scored with it;
        the reference forecast is the base rate of the other cases."""
        tabs = [self.table(t) for t in thetas]
        C = len(self.cases)
        picks, num, ref, n = [], np.zeros(4), np.zeros(4), 0.0
        per_case = []
        for c in range(C):
            use = np.ones(C, bool); use[c] = False
            sc = [self.bss(T, use)[0] for T in tabs]
            sc = [(-9.0 if not np.isfinite(x) else x) for x in sc]
            b = int(np.argmax(sc))
            picks.append(thetas[b])
            W = self.W
            nt = (self.N[use] * W).sum()
            ob = (self.E[use] * W[None, :, None]).sum(axis=(0, 1)) / max(nt, 1e-12)
            nc = (self.N[c] * W).sum()
            ec = (self.E[c] * W[:, None]).sum(axis=0)
            num_c = (tabs[b][c] * W[:, None]).sum(axis=0)
            ref_c = ob ** 2 * nc - 2 * ob * ec + ec
            num += num_c; ref += ref_c; n += nc
            per_case.append({"case": self.cases[c], "num": num_c.tolist(), "ref": ref_c.tolist()})
        per = [float(1 - num[k] / ref[k]) if (ref[k] > 0 and self.eligible[k]) else None for k in range(4)]
        # bootstrap over cases: is the out-of-sample skill above zero?
        rng = np.random.default_rng(1)
        A, B = np.array([x["num"] for x in per_case]), np.array([x["ref"] for x in per_case])
        boot = []
        for _ in range(400):
            ix = rng.integers(0, C, C)
            a, b_ = A[ix].sum(axis=0), B[ix].sum(axis=0)
            boot.append(np.where(b_ > 0, 1 - a / np.maximum(b_, 1e-30), np.nan))
        boot = np.array(boot)
        lo, hi = np.nanpercentile(boot, 5, axis=0), np.nanpercentile(boot, 95, axis=0)
        ok = [x for x in per if x is not None]
        return {"BSS": per, "mean_BSS": float(np.mean(ok)) if ok else None,
                "BSS_p05": [float(lo[k]) if per[k] is not None else None for k in range(4)],
                "BSS_p95": [float(hi[k]) if per[k] is not None else None for k in range(4)],
                "cases_better_than_base_rate": [int(sum(1 for x in per_case if x["ref"][k] > 0 and x["num"][k] < x["ref"][k])) for k in range(4)],
                "cases_with_events": [int((self.E[:, :, k].sum(axis=1) > 0).sum()) for k in range(4)],
                "picks": picks, "cases": self.cases}


def _near(g, v, d: int = 1) -> list:
    return [g[k] for k in range(len(g)) if abs(k - g.index(v)) <= d] if v in g else [v]


def fit_kernel(fs: FitSet, th0: dict, has_1h: bool, mode: str = "full", log=print) -> tuple[dict, list[dict]]:
    """Coordinate search on the two kernels, then a joint grid around the optimum. Returns (best, every theta
    tried). mode: full = two rounds + joint grid; quick = one round; local = the 12-h kernel only, 2 grid steps
    around the start (used for the variants, where the question is only whether a re-fit changes the answer)."""
    tried, best, best_sc = [dict(th0)], dict(th0), fs.score(th0)

    def take(th):
        nonlocal best, best_sc
        tried.append(th)
        sc = fs.score(th)
        if np.isfinite(sc) and sc > best_sc + 1e-6:
            best, best_sc = th, sc

    if mode == "local":
        for a in _near(SIG, th0["sigma"], 2):
            for b in _near(BIA, th0["bias"], 2):
                take({**th0, "sigma": a, "bias": b})
    else:
        for rnd in range(1 if mode == "quick" else 2):
            for a in SIG:
                for b in BIA:
                    take({**best, "sigma": a, "bias": b})
            if has_1h:
                for a in SIG1:
                    for b in BIA1:
                        take({**best, "sigma1": a, "bias1": b})
            log(f"   round {rnd + 1}: sigma {best['sigma']} bias {best['bias']} sigma1h {best['sigma1']} bias1h {best['bias1']} mean BSS {best_sc:.4f}")
        if mode == "full":
            base = dict(best)
            for a in _near(SIG, base["sigma"]):
                for b in _near(BIA, base["bias"]):
                    for c in (_near(SIG1, base["sigma1"]) if has_1h else [base["sigma1"]]):
                        for d in (_near(BIA1, base["bias1"]) if has_1h else [base["bias1"]]):
                            take({**base, "sigma": a, "bias": b, "sigma1": c, "bias1": d})
            log(f"   joint: sigma {best['sigma']} bias {best['bias']} sigma1h {best['sigma1']} bias1h {best['bias1']} mean BSS {best_sc:.4f}")
    uniq = {json.dumps(t, sort_keys=True): t for t in tried}
    return best, list(uniq.values())


def best_of(fs: FitSet, thetas: list[dict]) -> dict:
    sc = [fs.score(t) for t in thetas]
    sc = [(-9.0 if not np.isfinite(x) else x) for x in sc]
    return thetas[int(np.argmax(sc))]


# ------------------------------------------------------------------------------- full fields, counts

def full_prob(it: Item, th: dict) -> np.ndarray:
    p = np.zeros((4, it.F, it.ncell), np.float32)
    p[:, it.fi, it.ni] = item_prob(it, th)
    return p


class Tally:
    """Histograms of the probability by level, event / no event, day stratum and cell set, and the list of
    warning-zone units (issue time, zone, valid day): everything a tau table needs."""

    def __init__(self):
        self.hist = {}       # (cls, cellset) -> (S, 4, 2, NB + 1) counts
        self.units = {}      # cls -> list of (case, stratum, pmax[4], obs[4])
        self.frames = {}
        self.cases = {}

    @classmethod
    def from_dict(cls, d: dict) -> "Tally":
        t = cls()
        t.__dict__.update(d)
        return t

    def add(self, it: Item, p: np.ndarray, fwd: bool = False):
        o = it.of_full if fwd else it.o_full
        zc = ZONE[it.cells]
        b = np.minimum((p * NB).astype(np.int32), NB)
        for name, csel in (("all", np.ones(it.ncell, bool)), ("zones", zc >= 0)):
            h = self.hist.setdefault((it.cls, name), np.zeros((NS, 4, 2, NB + 1)))
            for f in range(it.F):
                s = it.strata[f]
                for k in range(4):
                    ev = o[k, f][csel]
                    bb = b[k, f][csel]
                    h[s, k, 1] += np.bincount(bb[ev], minlength=NB + 1)
                    h[s, k, 0] += np.bincount(bb[~ev], minlength=NB + 1)
        self.frames[it.cls] = self.frames.get(it.cls, 0) + it.F
        self.cases.setdefault(it.cls, set()).add(it.case)
        u = self.units.setdefault(it.cls, [])
        for day in np.unique(it.days):
            fs = np.nonzero(it.days == day)[0]
            s = int(it.strata[fs[0]])
            for zid in np.unique(zc[zc >= 0]):
                cs = zc == zid
                pm = p[:, fs][:, :, cs].max(axis=(1, 2))
                ob = o[:, fs][:, :, cs].any(axis=(1, 2))
                u.append((it.case, s, pm, ob))


def strata_weights(n_by_stratum: np.ndarray, pop: np.ndarray, which: str) -> np.ndarray:
    """Weight of one unit of each stratum. combined: every stratum counts with its real frequency;
    episodes / quiet: the same inside the wet (>= 30 mm) or the other strata; sample: every unit counts 1."""
    if which == "sample":
        return np.ones(NS)
    grp = np.arange(NS) >= DC.EPISODE_FROM
    use = np.ones(NS, bool) if which == "combined" else grp if which == "episodes" else ~grp
    w = np.where(use & (n_by_stratum > 0), pop / np.maximum(n_by_stratum, 1), 0.0)
    tot = (w * n_by_stratum).sum()
    return w * (n_by_stratum[use].sum() / tot) if tot > 0 else w        # equivalent counts: they add up to the units used


def rates(h, m, fa, cn=None) -> dict:
    return {"hits": int(round(h)), "misses": int(round(m)), "false_alarms": int(round(fa)),
            "correct_negatives": None if cn is None else int(round(cn)),
            "POD": round(h / (h + m), 3) if h + m > 0 else None, "FAR": round(fa / (h + fa), 3) if h + fa > 0 else None,
            "CSI": round(h / (h + m + fa), 3) if h + m + fa > 0 else None,
            "freq_bias": round((h + fa) / (h + m), 2) if h + m > 0 else None}


def cell_counts(h: np.ndarray, k: int, tau: float, w: np.ndarray) -> dict:
    """h (S, 4, 2, NB+1). Forecast yes = P >= tau."""
    b = int(np.ceil(tau * NB - 1e-9))
    ev = (h[:, k, 1] * w[:, None]).sum(axis=0)
    ne = (h[:, k, 0] * w[:, None]).sum(axis=0)
    return rates(ev[b:].sum(), ev[:b].sum(), ne[b:].sum(), ne[:b].sum())


def unit_counts(units: list, k: int, tau: float, which: str, pop: np.ndarray, cap_ok: bool = True) -> dict:
    if not units:
        return rates(0, 0, 0, 0)
    s = np.array([u[1] for u in units])
    w = strata_weights(np.bincount(s, minlength=NS).astype(float), pop, which)[s]
    f = np.array([u[2][k] >= tau for u in units]) & cap_ok
    o = np.array([u[3][k] for u in units])
    return rates((w * (f & o)).sum(), (w * (~f & o)).sum(), (w * (f & ~o)).sum(), (w * (~f & ~o)).sum())


def reliability(h: np.ndarray, k: int, w: np.ndarray) -> list:
    edges = [0, .02, .05, .10, .15, .2, .3, .4, .5, .7, .9, 1.0001]
    ev = (h[:, k, 1] * w[:, None]).sum(axis=0)
    ne = (h[:, k, 0] * w[:, None]).sum(axis=0)
    mid = (np.arange(NB + 1) + 0.5) / NB
    mid[0] = 0.0
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ia, ib = int(np.ceil(a * NB - 1e-9)), int(np.ceil(b * NB - 1e-9))
        n = ev[ia:ib].sum() + ne[ia:ib].sum()
        if n > 0:
            out.append({"p_forecast": round(float(((ev + ne)[ia:ib] * mid[ia:ib]).sum() / n), 3),
                        "observed_freq": round(float(ev[ia:ib].sum() / n), 3), "n": int(round(n)), "events": int(round(ev[ia:ib].sum()))})
    return out


def costloss_tau(h: np.ndarray, k: int, w: np.ndarray, ratio: float) -> float | None:
    """Smallest probability from which the event frequency actually observed (isotonic fit of the reliability
    curve) reaches the cost/loss ratio: acting pays from there on. None: it never does."""
    ev = (h[:, k, 1] * w[:, None]).sum(axis=0)
    n = ev + (h[:, k, 0] * w[:, None]).sum(axis=0)
    # pool adjacent violators
    blocks = [[ev[i], n[i], i] for i in range(NB + 1) if n[i] > 0]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] * blocks[i + 1][1] > blocks[i + 1][0] * blocks[i][1]:
            blocks[i][0] += blocks[i + 1][0]; blocks[i][1] += blocks[i + 1][1]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    for e, m, start in blocks:
        if m > 0 and e / m >= ratio:
            return round(max(start / NB, 0.005), 3)
    return None


def curves(tally: Tally, cls_list, pop: np.ndarray) -> dict:
    """For each level: detection, false-alarm ratio and CSI against tau, on cells (all / warning zones) and on
    zone-days, for episodes, quiet days and combined; the CSI optimum and the cost/loss thresholds."""
    out = {}
    hs = {name: sum(tally.hist[(c, name)] for c in cls_list if (c, name) in tally.hist) for name in ("all", "zones")}
    units = [u for c in cls_list for u in tally.units.get(c, [])]
    if isinstance(hs["all"], int):
        return out
    nstr = {name: hs[name][:, 0].sum(axis=(1, 2)) for name in hs}
    for which in ("combined", "episodes", "quiet", "sample"):
        o = out.setdefault(which, {})
        for k, L in enumerate(LEVELS):
            row = {"cell": [], "cell_zones": [], "zone_day": []}
            for tau in TAUS:
                row["cell"].append({"tau": tau, **cell_counts(hs["all"], k, tau, strata_weights(nstr["all"], pop, which))})
                row["cell_zones"].append({"tau": tau, **cell_counts(hs["zones"], k, tau, strata_weights(nstr["zones"], pop, which))})
                row["zone_day"].append({"tau": tau, **unit_counts(units, k, tau, which, pop)})
            best = {}
            for scope in row:
                ok = [r for r in row[scope] if r["CSI"] is not None]
                top = max((r["CSI"] for r in ok), default=None)
                # the largest tau within 0.005 of the best CSI: no more false alarms than needed
                best[scope] = max((r["tau"] for r in ok if r["CSI"] >= top - 0.005), default=None) if top else None
            w = strata_weights(nstr["all"], pop, which)
            row["csi_optimal_tau"] = best
            row["costloss_tau"] = {str(r): costloss_tau(hs["all"], k, w, r) for r in (0.05, 0.1, 0.2, 0.3)}
            row["reliability"] = reliability(hs["all"], k, w)
            ev = (hs["all"][:, k, 1] * w[:, None]).sum()
            row["base_rate"] = float(ev / max((hs["all"][:, k].sum(axis=(1, 2)) * w).sum(), 1e-12))
            row["events_cell"] = int(hs["all"][:, k, 1].sum()) if which == "sample" else int(round(ev))
            o[str(L)] = row
    return out


def decide_counts(tally: Tally, cls_list, tau: dict, cap: int | None, pop: np.ndarray, which: str) -> dict:
    """Hits / misses / false alarms of `risk.decide` with a tau table and a level cap, for the page."""
    hs = {name: sum(tally.hist[(c, name)] for c in cls_list if (c, name) in tally.hist) for name in ("all", "zones")}
    units = [u for c in cls_list for u in tally.units.get(c, [])]
    out = {"zone_day": {}, "cell": {}, "cell_zones": {}}
    if isinstance(hs["all"], int):
        return out
    for k, L in enumerate(LEVELS):
        capped = cap is not None and L > cap
        # level >= L is issued when P(>= L') reaches tau[L'] for some L' >= L; P is nested, so with a tau table
        # that does not decrease with the level (every table in use) this is exactly P(>= L) >= tau[L]
        t = 2.0 if capped else float(tau[str(L)])
        for name, key in (("all", "cell"), ("zones", "cell_zones")):
            out[key][str(L)] = cell_counts(hs[name], k, t, strata_weights(hs[name][:, 0].sum(axis=(1, 2)), pop, which))
        out["zone_day"][str(L)] = unit_counts(units, k, t, which, pop)
    return out


# ----------------------------------------------------------------------------------------- horizon

def block_files(hz: str) -> list[Path]:
    return sorted((R.BLOCKS / hz).glob("[0-9]*.npz"))


def load_items(hz: str, var: str = "", cells: str = "fit", truth_var=None, keep_full=False, files=None) -> list[Item]:
    out = []
    for f in (files if files is not None else block_files(hz)):
        if ".tmp" in f.name:
            continue
        try:
            out += block_items(f, var, cells, truth_var, keep_full)
        except Exception as e:  # noqa: BLE001   a block being written
            print("   skipped", f.name, type(e).__name__, e)
    return out


def describe(th: dict) -> dict:
    return {"sigma": th["sigma"], "bias": th["bias"], "sigma1h": th["sigma1"], "bias1h": th["bias1"]}


def run_horizon(hz: str, pop: np.ndarray, quick: bool = False, log=print) -> tuple[dict, dict, dict]:
    t_start = time.time()
    files = block_files(hz)
    items = load_items(hz, files=files)
    if not items:
        return {}, {}, {}
    th_prod = theta_production(hz)
    has_1h = any(len(it.i1) for it in items)
    fit = {"horizon": hz, "production": describe(th_prod)}
    log(f"[{hz}] {len(files)} blocks, {len(items)} class items, {len({it.case for it in items})} cases, {time.time() - t_start:.0f} s to load")
    FS = FitSet(items, pop)
    tiers = {}
    for it in items:
        tiers[it.tier] = tiers.get(it.tier, 0) + 1
    fit["sample"] = {"blocks": len(files), "cases": FS.cases, "class_items_by_tier": tiers,
                     "blocks_with_ens": len({it.issue for it in items if it.ens}),
                     "cell_frames_fit": float(FS.N.sum()), "events_fit": FS.E.sum(axis=(0, 1)).tolist(),
                     "cell_frames_by_stratum": FS.N.sum(axis=0).tolist(), "events_by_stratum": FS.E.sum(axis=0).tolist(),
                     "eligible_levels": [bool(x) for x in FS.eligible]}
    grp = np.arange(NS) >= DC.EPISODE_FROM

    def views(fs: FitSet, th: dict) -> dict:
        T = fs.table(th)
        tot = fs.N.sum(axis=0)
        out = {}
        for name, use in (("combined", np.ones(NS, bool)), ("episodes", grp), ("quiet", ~grp)):
            W = np.where(use & (tot > 0), pop / np.maximum(tot, 1), 0.0)
            m, per = fs.bss(T, None, W)
            out[name] = {"mean_BSS": None if not np.isfinite(m) else round(m, 4), "BSS": [None if x is None else round(x, 4) for x in per]}
        m, per = fs.bss(T, None, np.ones(NS))
        out["sample"] = {"mean_BSS": None if not np.isfinite(m) else round(m, 4), "BSS": [None if x is None else round(x, 4) for x in per]}
        return out

    fit["production_skill"] = views(FS, th_prod)
    log(f"[{hz}] production kernel {describe(th_prod)}: {fit['production_skill']['combined']}")
    # ---- kernel, whole horizon and per lead class
    best, tried = fit_kernel(FS, th_prod, has_1h, "quick" if quick else "full", log)
    lo = FS.loco(tried)
    fit["kernel"] = {"fitted": describe(best), "in_sample": views(FS, best), "n_tried": len(tried),
                     "loco": {k: v for k, v in lo.items() if k != "picks"},
                     "loco_picks": [{"held_out": c, **describe(t)} for c, t in zip(lo["cases"], lo["picks"])],
                     "loco_production": {k: v for k, v in FS.loco([th_prod]).items() if k != "picks"}}
    log(f"[{hz}] fitted {describe(best)} in-sample {fit['kernel']['in_sample']['combined']} LOCO {lo['BSS']} (production LOCO {fit['kernel']['loco_production']['BSS']})")
    fit["by_class"] = {}
    for c in CLASSES[hz]:
        sub = [it for it in items if it.cls == c]
        if not sub:
            continue
        fs = FitSet(sub, pop)
        b, tr = best_of(fs, tried), tried          # the class optimum among the kernels tried for the horizon
        lc = fs.loco(tr)
        fit["by_class"][c] = {"n_items": len(sub), "cases": len(fs.cases), "events_fit": fs.E.sum(axis=(0, 1)).tolist(),
                              "production": views(fs, th_prod), "horizon_kernel": views(fs, best), "own_kernel": describe(b),
                              "own_kernel_in_sample": views(fs, b),
                              "loco_own": {k: v for k, v in lc.items() if k not in ("picks", "cases")},
                              "loco_horizon_kernel": {k: v for k, v in fs.loco([best]).items() if k not in ("picks", "cases")},
                              "loco_production": {k: v for k, v in fs.loco([th_prod]).items() if k not in ("picks", "cases")},
                              "loco_picks": [describe(t) for t in lc["picks"]]}
        log(f"[{hz}] class {c}: own kernel {describe(b)} LOCO {lc['BSS']}; horizon kernel LOCO {fit['by_class'][c]['loco_horizon_kernel']['BSS']}")
    # ---- by archive tier
    fit["by_tier"] = {}
    for tier in sorted(tiers):
        sub = [it for it in items if it.tier == tier]
        fs = FitSet(sub, pop)
        fit["by_tier"][tier] = {"n_items": len(sub), "cases": len(fs.cases), "production": views(fs, th_prod), "fitted": views(fs, best)}
    # ---- experiments on weights
    exp = {}
    ens_items = [it for it in items if it.ens]
    if hz == "mid" and ens_items:
        fs = FitSet(ens_items, pop)
        rows = []
        for x in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 1e4):
            th = {**best, "mult": {"ifs_ens3h": x}}
            share = []
            for it in ens_items:
                fw = np.array([th["fw"].get(f, 0.0) for f in FAMS])[it.fam] * it.mw * 0.5 ** (it.age / th["hl"])
                e = it.model == "ifs_ens3h"
                share.append(fw[e].sum() * x / max(fw[e].sum() * x + fw[~e].sum(), 1e-12))
            T = fs.table(th)
            per_case = [fs.bss(T, np.arange(len(fs.cases)) == c)[0] for c in range(len(fs.cases))]
            rows.append({"ens_weight_x": x, "ens_share_of_weight": round(float(np.mean(share)), 3), **views(fs, th)["sample"],
                         "combined": views(fs, th)["combined"], "per_case_mean_BSS": [None if not np.isfinite(v) else round(v, 4) for v in per_case]})
        exp["ens_weight_mid"] = {"blocks": len({it.issue for it in ens_items}), "cases": fs.cases, "rows": rows,
                                 "note": "x multiplies production's ENS member weight (0.04 x 0.6); 1e4 = ENS alone"}
        log(f"[{hz}] ENS weight: " + " | ".join(f"x{r['ens_weight_x']:g} share {r['ens_share_of_weight']} BSS {r['mean_BSS']}" for r in rows))
    if hz in ("mid", "now"):
        sub = [it for it in items if it.tier == "A"]
        if sub:
            fs = FitSet(sub, pop)
            rows = [{"halflife_h": x, **views(fs, {**best, "hl": x})["sample"], "combined": views(fs, {**best, "hl": x})["combined"]}
                    for x in ((1.5, 3.0, 6.0, 12.0, 1e6) if hz == "now" else (3.0, 6.0, 12.0, 24.0, 1e6))]
            exp["age_halflife"] = {"items": len(sub), "cases": fs.cases, "rows": rows, "note": "tier A only: real run ages"}
            log(f"[{hz}] half-life: " + " | ".join(f"{r['halflife_h']:g} h {r['mean_BSS']}" for r in rows))
            rows = []
            for name, fwm in (("production", {}), ("cp x2", {"cp": 2.0}), ("cp x0.5", {"cp": 0.5}), ("equal runs", None)):
                th = dict(best)
                if fwm is None:
                    th = {**best, "fw": {f: 1.0 for f in FAMS}, "hl": 1e6}
                else:
                    th = {**best, "fw": {f: v * fwm.get(f, 1.0) for f, v in best["fw"].items()}}
                rows.append({"weights": name, **views(fs, th)["sample"], "combined": views(fs, th)["combined"]})
            exp["family_weights"] = {"rows": rows}
    if hz == "now":
        rows = []
        for so in (0.05, 0.1, 0.15, 0.25, 0.4):
            for of_ in (0.88, 1.0, 1.1):
                th = {**best, "sigma_obs": so, "of": of_}
                rows.append({"sigma_obs": so, "obs12_factor": of_, **views(FS, th)["sample"], "combined": views(FS, th)["combined"]})
        exp["measured_share_kernel"] = {"rows": rows}
    fit["experiments"] = exp
    # ---- variants: neighbourhood radius and the gate on measured rain (kernel re-fitted on each, coarse)
    var = {}
    for r in R.RADII[hz]:
        for label, tv in (("matched_event", None), ("production_event", "")):
            its = load_items(hz, R.tag_r(r), truth_var=tv, files=files)
            if not its:
                continue
            fs = FitSet(its, pop)
            b, _ = fit_kernel(fs, best, has_1h, "local", lambda *a: None)
            var[f"radius_{r:g}km_{label}"] = {"kernel": describe(b), "refit": views(fs, b), "same_kernel": views(fs, best), "by_class": {
                c: views(FitSet([i for i in its if i.cls == c], pop), b) for c in CLASSES[hz] if any(i.cls == c for i in its)}}
            log(f"[{hz}] radius {r:g} km ({label}): refit {describe(b)} {var[f'radius_{r:g}km_{label}']['refit']['combined']}")
    gated = [it for it in items if it.phi is not None]
    if gated and hz == "now":
        base_files = files
        for g in (None,) + tuple(R.GATES[hz]):
            its = load_items(hz, "" if g is None else R.tag_g(g), files=base_files)
            if not its:
                continue
            name = f"gate_{R.PARAMS.get('obs_gate_mm', 20.0):g}mm_production" if g is None else f"gate_{g:g}mm"
            row = {}
            for label, fwd in (("event_as_published", False), ("event_forward_same_gate", True)):
                fs = FitSet(its, pop, fwd)
                b, _ = fit_kernel(fs, best, has_1h, "local", lambda *a: None)
                row[label] = {"kernel": describe(b), "refit": views(fs, b), "production_kernel": views(fs, th_prod), "events": fs.E.sum(axis=(0, 1)).tolist()}
            var[name] = row
            log(f"[{hz}] {name}: as published {row['event_as_published']['refit']['combined']} forward {row['event_forward_same_gate']['refit']['combined']}")
    fit["variants"] = var
    del items, FS
    # ---- out-of-sample probabilities on every published cell: tallies for production and for the fitted kernel
    pick = {c: t for c, t in zip(lo["cases"], lo["picks"])}
    tallies = {"production": Tally(), "fitted_loco": Tally(), "fitted": Tally()}
    obs_cf = np.zeros(4); n_cf = 0
    for f in files:
        try:
            its = block_items(f, "", "all", None, True)
        except Exception:  # noqa: BLE001
            continue
        for it in its:
            tallies["production"].add(it, full_prob(it, th_prod))
            tallies["fitted_loco"].add(it, full_prob(it, pick.get(it.case, best)))
            tallies["fitted"].add(it, full_prob(it, best))
            obs_cf += it.o_full.sum(axis=(1, 2)); n_cf += it.F * it.ncell
    fit["n_cell_frames"] = int(n_cf)
    fit["observed_cell_frames"] = {str(L): int(obs_cf[k]) for k, L in enumerate(LEVELS)}
    fit["n_frames"] = int(sum(tallies["fitted"].frames.values()))
    fit["curves"] = {}
    present = [c for c in CLASSES[hz] if c in tallies["fitted"].units]
    for name in ("production", "fitted_loco"):
        fit["curves"][name] = {"all": curves(tallies[name], present, pop), **{c: curves(tallies[name], [c], pop) for c in present}}
    # plain dicts: the pickle must load whether this file ran as a script or was imported
    state = {"tallies": {k: dict(vars(v)) for k, v in tallies.items()}, "classes": present, "best": best, "prod": th_prod,
             "cases": lo["cases"]}
    log(f"[{hz}] done in {time.time() - t_start:.0f} s")
    return fit, state, {"best": best}


def summarise(hz: str, fit: dict, state: dict, pop: np.ndarray, tau: dict, cap: int | None) -> dict:
    """The block of results.json that make_pages.py reads, plus the episode / quiet split."""
    b = fit["kernel"]["fitted"]
    t = {k: Tally.from_dict(v) for k, v in state["tallies"].items()}
    cls = state["classes"]
    res = {"horizon": hz, "cases": state["cases"], "n_frames": fit["n_frames"], "n_issue_times": fit["sample"]["blocks"],
           "n_cell_frames": fit["n_cell_frames"], "observed_cell_frames": fit["observed_cell_frames"],
           "tuned": {"sigma": b["sigma"], "bias": b["bias"], "sigma1h": b["sigma1h"], "bias1h": b["bias1h"], "tau": tau,
                     "mean_BSS": fit["kernel"]["in_sample"]["combined"]["mean_BSS"], "BSS": fit["kernel"]["in_sample"]["combined"]["BSS"],
                     "loco_BSS": fit["kernel"]["loco"]["BSS"], "loco_BSS_p05": fit["kernel"]["loco"]["BSS_p05"]},
           "level_cap": cap, "tau": tau,
           "cross_validated": decide_counts(t["fitted_loco"], cls, tau, cap, pop, "combined"),
           "in_sample": decide_counts(t["fitted"], cls, tau, cap, pop, "combined"),
           "split": {w: decide_counts(t["fitted_loco"], cls, tau, cap, pop, w) for w in ("episodes", "quiet", "combined", "sample")},
           "split_note": "episodes = days with >= 30 mm at the wettest SAIH gauge, quiet = the other days, combined = both weighted "
                         "with the real frequency of each kind of day (hindcast/day_climatology.json), sample = unweighted",
           "by_lead": {c: {w: decide_counts(t["fitted_loco"], [c], tau, cap, pop, w) for w in ("combined", "episodes", "quiet")} for c in cls},
           "reliability": {str(L): fit["curves"]["fitted_loco"]["all"]["combined"][str(L)]["reliability"] for L in LEVELS}
           if fit["curves"]["fitted_loco"]["all"] else {}}
    return res


def load_pop() -> np.ndarray:
    return np.array(DC.load()["frequency"], float)


def current_policy() -> tuple[dict, dict]:
    p = R.PARAMS
    return {hz: {str(k): float(v) for k, v in p["tau"][hz].items()} for hz in ("now", "mid", "long")}, dict(p.get("level_cap", {}))


def main(quick: bool = False, hzs=None) -> None:
    pop = load_pop()
    hzs = hzs or ["mid", "long", "now"]
    fit_file, res_file, state_file = HC / "fit.json", HC / "results.json", HC / "cache" / "score_state.pkl"
    fit_all = json.loads(fit_file.read_text(encoding="utf-8")) if fit_file.exists() else {}
    results = json.loads(res_file.read_text(encoding="utf-8")) if res_file.exists() else {}
    states = pickle.loads(state_file.read_bytes()) if state_file.exists() else {}
    tau, cap = current_policy()
    truth = R.Truth()
    for hz in hzs:
        fit, state, _ = run_horizon(hz, pop, quick)
        if not fit:
            print(f"[{hz}] no blocks")
            continue
        fit_all[hz], states[hz] = fit, state
        results[hz] = summarise(hz, fit, state, pop, tau[hz], cap.get(hz))
        results[hz]["tau_source"] = "params.json at fit time; hindcast/deploy_params.py rewrites these tables for the deployed policy"
        fit_all["generated"] = results["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        fit_all["day_climatology"] = DC.load()
        results["truth_cases"] = sorted(truth.cases)
        results["thresholds_mm"] = {"1h": [20, 40, 90, 135], "12h": [60, 100, 180, 300]}
        fit_file.write_text(json.dumps(fit_all, indent=1, default=str), encoding="utf-8")
        state_file.write_bytes(pickle.dumps(states, protocol=4))
        res_file.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    try:
        results["hydrology_poyo_2024"] = R.poyo_check(truth)
    except Exception as e:  # noqa: BLE001
        print("poyo check failed:", type(e).__name__, e)
    for k in [k for k in results if k not in ("now", "mid", "long", "generated", "truth_cases", "thresholds_mm", "hydrology_poyo_2024", "policy")]:
        results.pop(k)
    res_file.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("wrote", res_file, fit_file)


def check(n: int = 6) -> None:
    """The fast dressing of this file against production's risk.dressed_probabilities on real blocks."""
    worst = 0.0
    for hz in ("now", "mid", "long"):
        files = block_files(hz)
        for f in files[:: max(1, len(files) // n)][:n]:
            z = np.load(f, allow_pickle=True)
            cells, M, F = z["cells"], z["a12"].shape[0], z["a12"].shape[1]
            def full(a, fill=np.nan):
                g = np.full((a.shape[0], F, grid.NY * grid.NX), fill, np.float32)
                g[:, :, cells] = a
                return g.reshape(a.shape[0], F, grid.NY, grid.NX)
            a1 = np.full((M, F, len(cells)), np.nan, np.float32)
            a1[z["has1"]] = z["a1"]
            th = theta_production(hz)
            w = np.array([th["fw"][x] for x in z["family"]]) * z["mw"] * 0.5 ** (z["age_h"] / th["hl"])
            ws = w[:, None] * z["valid"]
            pred = risk.Predictors(None, None, None, None, ws.sum(axis=0) > 0, None, [{"family": str(x)} for x in z["family"]],
                                   full(a1), full(z["a12"].astype(np.float32)), ws / np.maximum(ws.sum(axis=0, keepdims=True), 1e-12),
                                   full(z["phi"].astype(np.float32) / 250.0, 0.0) if "phi" in z.files else full(np.zeros((M, F, len(cells)), np.float32), 0.0))
            prob = risk.dressed_probabilities(pred, R.THR, R.PARAMS, hz)[0].reshape(4, F, -1)[:, :, cells]
            mine = np.zeros_like(prob)
            lead = np.array([lead_class(hz, x) for x in z["lead_h"]])
            for it in block_items(f, "", "all", None, True):
                fs = np.nonzero(lead == it.cls)[0]
                if len(fs) == it.F:
                    mine[:, fs] = full_prob(it, th)
            d = float(np.abs(prob - mine).max())
            worst = max(worst, d)
            print(f"{hz} {f.name}: M={M} F={F} N={len(cells)} max P {prob.max():.3f} max |production - fast| = {d:.2e}")
    print("worst difference", worst)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] == "check":
        check()
    else:
        main(quick="--quick" in sys.argv, hzs=[a for a in args if a in ("now", "mid", "long")] or None)
