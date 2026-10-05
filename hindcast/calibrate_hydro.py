"""Calibrate the rain -> discharge losses of core/hydro.py against measured flows.

    py -3.11 hindcast/calibrate_hydro.py            # everything: downloads (cached), event table, simulations, fits, validation,
                                                    # hindcast/hydro_fit.json, geo/hydro/loss_params.json  (~40 min; ~2 h the first time)
    py -3.11 hindcast/calibrate_hydro.py fetch      # only the downloads (SAIH flows + pluviometers)
    py -3.11 hindcast/calibrate_hydro.py nofetch    # no network: use what is cached

Data (built by geo/hydro/gauges/*.py, cached under hindcast/obs/flows/, gitignored):
  * 48 gauged catchments delineated like the control points (geo/hydro/gauges/out/), SAIH 5-min flows since Sep 2024,
    sensor faults removed, dam releases subtracted where a dam cuts the catchment;
  * continuous hourly rain: SAIH pluviometers interpolated, replaced by the hindcast truth analysis inside the episodes;
  * anchors: the official peaks of 29 Oct 2024 at control points (coord/findings/q5-science.md A.3).

Model family (losses per grid cell, then time-area routing + Clark reservoir: the arithmetic of core/hydro.py):
    W(t)   = W(t-1) exp(-1/tau) + p(t)                               wetness with memory tau
    c(P0,S)= x (x + 2S) / (x + S)^2,  x = W - P0                     marginal runoff coefficient of E = (W-P0)^2 / (W-P0+S)
    main   = p c(P0 m, S) + (1 - c) excess(p)                        excess = max(p - phi, 0) | p exp(-phi/p) | 0
    net    = (1 - alpha) main + alpha p c(P0b, Sb)                   alpha = share of the catchment that responds readily
with m(t) an optional antecedent-state multiplier of P0 (API), P0 one number or a per-cell pattern from the CEDEX raster,
and alpha one number or one per catchment (partial pooling; regression on catchment attributes for ungauged points).

Error measure: r = ln((sim + c) / (obs + c)) of the event peak, c = 5 m3/s * (A / 184 km2)^0.75 (5 m3/s at the Poyo gauge).
"""
import hashlib
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "geo" / "hydro" / "gauges"))
from riua import static                                   # noqa: E402
from riua.core import grid                                # noqa: E402

GAUGES = ROOT / "geo" / "hydro" / "gauges"
FLOWS = ROOT / "hindcast" / "obs" / "flows"
A_REF, C_REF = 184.0, 5.0
API_REF = 150.0                  # mm of antecedent index at which the ground counts as wet
API_K_DAY = 0.98
THREADS = 3
ANCHOR_CAP = 0.88                # the 2024 targets are capped at this share of the zero-loss peak of the rain analysis
ELEV_MAX, LON_MIN = 1150.0, -1.45   # "ravine domain": gauged catchments that resemble the control points (below 1150 m mean
#                                     elevation, east of 1.45 W); the interior headwaters and La Mancha are reported apart

# 29 Oct 2024, official figures (q5-science A.3: MITECO recovery plan, CEDEX, DGA-INPROES-HGM Forata report).
# (point, peak m3/s, weight, note, lower bound?)
ANCHORS = [
    ("poyo-ribarroja", 3200.0, 9.0, "CEDEX liquid peak at the A-3 (2282 measured when the sensor was lost)", False),
    ("poyo-chiva", 1100.0, 3.0, "CEDEX estimate above Chiva (43 km2)", False),
    ("horteta-torrent", 1500.0, 3.0, "estimate at the A-7", False),
    ("poyo-paiporta", 3400.0, 4.5, "CEDEX provisional at Picanya", False),
    ("bunol-bunol", 1000.0, 3.0, "estimate, río Buñol at Buñol", False),
    ("magro-real", 2800.0, 1.5, "DERIVED: 3600 modelled at Montroi minus ~800 of Forata spill (1092-1200 measured later, at 21:00)", False),
    ("magro-carlet", 3200.0, 1.5, "DERIVED: ~4000 modelled at Carlet minus ~800 of Forata spill", False),
    ("forata-in", 2016.0, 3.0, "measured inflow to Forata when the sensor range ended (2400 quoted)", True),
]
FORATA_RUNOFF_MM = 93.0          # 98 hm3 entered Forata (1054 km2) of 219 hm3 of rain: runoff coefficient 0.45
FORATA_VOL_WEIGHT = 2.0

FORM_NAME = {0: "none", 1: "max(p-phi,0)", 2: "p*exp(-phi/p)"}
DEF = dict(p0=120.0, s=150.0, tau=72.0, form=1, phi=45.0, ratio=1.0, pat="uniform", kf=0.3, vs=1.0, alpha=0.0, p0b=25.0, sb=100.0)
CURRENT = dict(DEF)              # what params.py holds today
DESIGN = dict(DEF, p0=25.0, s=125.0, form=0, phi=0.0)   # the original design value (SCS, P0 = 25 mm)

# ---- grid of regional simulations (stage 1) --------------------------------------------------------------------------
P0S = [60.0, 80.0, 100.0, 120.0, 150.0, 190.0, 240.0]
SS_ = [75.0, 150.0, 300.0]
EXC = [(0, 0.0), (1, 30.0), (1, 45.0), (1, 60.0), (2, 70.0), (2, 100.0), (2, 150.0)]
LOW = [(a, pb, sb) for a in (0.02, 0.05, 0.10) for pb in (10.0, 25.0) for sb in (30.0, 100.0)]
# ---- joint search of the main component with one alpha per catchment (stage 2) ---------------------------------------
P0S2 = [100.0, 150.0, 190.0, 240.0]
SS2 = [75.0, 150.0]
EXC2 = [(0, 0.0), (1, 30.0), (1, 45.0), (1, 60.0), (2, 100.0)]
LOW2 = [(10.0, 100.0), (25.0, 100.0), (25.0, 30.0)]
ALPHA_REF = 0.03
FACTORS = [0.1, 0.18, 0.3, 0.5, 0.75, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0]      # per-catchment multiplier of alpha
FINE = np.linspace(np.log(FACTORS[0]), np.log(FACTORS[-1]), 89)
DF = FINE[1] - FINE[0]
TAUS_SHRINK = [0.2, 0.35, 0.5, 0.8, 1.5, np.inf]
ATTRS = ["carb", "carb_high", "marl_clay", "quaternary", "perm_rank", "forest", "crops", "urban", "map_mm", "slope", "ln_area",
         "ln_q2_spec", "p0i_mm"]


def c_off(area):
    """error offset: 5 m3/s at the Poyo gauge (184 km2), scaled with area like a flood peak"""
    return C_REF * (np.asarray(area, float) / A_REF) ** 0.75


# ====================================================================================================== kernel
@njit(cache=True, nogil=True)
def sim(p, mult, p0, s, decay, phi, form, alpha, p0b, sb, lag, loc, a, K):
    """One catchment, continuous. p (T, n) mm/h; mult (T,) multiplier of P0 (antecedent state, 1 = dry); p0, s, phi, alpha (n,);
    form 0 none | 1 max(p - phi, 0) | 2 p exp(-phi/p); p0b, sb: thresholds of the responsive share alpha;
    lag/loc/a = time-area table SORTED by loc.  With alpha = 0 this is core/hydro.py net_rain + route (checked: identical
    peaks on the 2024 episode); dry cell-hours are skipped (W decays as decay ** gap).  Returns q (T,) m3/s."""
    T, n = p.shape
    w = np.zeros(n)
    last = np.zeros(n, np.int64)
    ptr = np.zeros(n + 1, np.int64)
    for j in range(lag.size):
        ptr[loc[j] + 1] += 1
    for c in range(n):
        ptr[c + 1] += ptr[c]
    raw = np.zeros(T + lag.max() + 2)
    for t in range(T):
        m = mult[t]
        for c in range(n):
            pt = p[t, c]
            if pt <= 0.0:
                continue
            wd = w[c] * decay ** (t - last[c] + 1)
            wm = wd + 0.5 * pt
            x = wm - p0[c] * m
            cc = 0.0
            if x > 0.0:
                cc = x * (x + 2.0 * s[c]) / ((x + s[c]) * (x + s[c]))
            e = pt * cc
            if form == 1:
                if pt > phi[c]:
                    e += (1.0 - cc) * (pt - phi[c])
            elif form == 2:
                e += (1.0 - cc) * pt * np.exp(-phi[c] / pt)
            al = alpha[c]
            if al > 0.0:
                xb = wm - p0b
                cb = 0.0
                if xb > 0.0:
                    cb = xb * (xb + 2.0 * sb) / ((xb + sb) * (xb + sb))
                e = (1.0 - al) * e + al * pt * cb
            w[c] = wd + pt
            last[c] = t + 1
            if e > 0.0:
                for j in range(ptr[c], ptr[c + 1]):
                    raw[t + lag[j]] += a[j] * e
    c1 = 1.0 - np.exp(-1.0 / K)
    q = np.zeros(T)
    prev = 0.0
    for t in range(T):
        prev = c1 * 0.278 * raw[t] + (1.0 - c1) * prev
        q[t] = prev
    return q


def wet_mult(R, ratio, lag_h=48, api_ref=API_REF):
    """P0 multiplier from the antecedent precipitation index of the basin (API, decay 0.98 per day, read lag_h hours
    earlier so the event's own rain, which is already in W, does not count): ratio ** clip(API / api_ref, 0, 1)."""
    if ratio >= 1.0:
        return np.ones(len(R))
    api = lfilter([1.0], [1.0, -API_K_DAY ** (1.0 / 24.0)], np.nan_to_num(R))
    api = np.concatenate([np.zeros(lag_h), api[:-lag_h]])
    return ratio ** np.clip(api / api_ref, 0.0, 1.0)


# ====================================================================================================== data
def beta_cells():
    """Norma 5.2-IC regional corrector of P0 per grid cell. The regions are only printed as a small map (figure 2.9), so the
    limits below are approximate: 822 coastal Valencia-Alicante 2.40, 821 coastal Castellón 1.30, 81 interior 1.30, 72 Segura 2.10."""
    lat = (grid.LAT0 + (np.arange(grid.NY) + 0.5) * grid.D)[:, None] * np.ones((1, grid.NX))
    lon = (grid.LON0 + (np.arange(grid.NX) + 0.5) * grid.D)[None, :] * np.ones((grid.NY, 1))
    b = np.full((grid.NY, grid.NX), 2.40)
    b[(lat >= 39.72)] = 1.30                                   # Castellón coast (north of the Palancia)
    b[(lon < -1.05) & (lat >= 38.75)] = 1.30                   # interior: upper Júcar, Cabriel, Turia
    b[(lon < -0.85) & (lat < 38.30)] = 2.10                    # Segura
    b[(lon < -1.25) & (lat < 38.75)] = 2.10
    return b.ravel()


def p0i_cells():
    f = ROOT / "scratch" / "q5-science" / "p0_cells.npy"       # CEDEX P0i raster as cell means (q5-science/p0_basins.py)
    a = np.load(f).ravel().astype(float)
    a[~np.isfinite(a)] = np.nanmean(a)
    return a


def p0_fields():
    """per-cell patterns of P0, each with mean 1 over the grid"""
    p0i, beta = p0i_cells(), beta_cells()
    return {"uniform": np.ones(p0i.size), "p0i": p0i / p0i.mean(), "p0i_beta": p0i * beta / (p0i * beta).mean()}


def _prep(c):
    """time-area table sorted by cell (the kernel needs it), contiguous rain"""
    o = np.argsort(c["loc"], kind="stable")
    c["lag"], c["loc"], c["a"] = c["lag"][o], c["loc"][o], c["a"][o]
    c["p"] = np.ascontiguousarray(c["p"], dtype=np.float32)
    return c


def collect(offline=False):
    """catchments {pid: dict(p, R, lag, loc, a, area, cells, tl)} for gauges and control points, event rows with observed
    flow, anchors."""
    from dataset import Data
    D = Data(offline=offline)
    rows = [r for r in D.events() if "q_obs" in r]
    cats = {}
    for k, pt in enumerate(D.pts):
        cats[pt["id"]] = _prep(D.cat(k))
    cnet, cps = static.hydro_net()
    for k, pid in enumerate(cnet.ids):
        if pid not in cats:
            cats[pid] = _prep(D.cat(k, cnet))
    dana = (np.datetime64("2024-10-29T00"), np.datetime64("2024-10-30T12"))
    anchors = []
    for pid, q, wgt, note, lower in ANCHORS:
        c = cats[pid]
        ev = [e for e in D.rain_events(c) if D.t[e["i0"]] <= dana[1] and D.t[e["i1"]] >= dana[0]]
        if not ev:
            print("  anchor without rain event:", pid); continue
        e = max(ev, key=lambda e: e["rain"])
        # what this rain field gives with no losses at all: the official figure cannot be asked for beyond that
        # (Poyo at the A-3: 2723 m3/s with zero losses against 3200 official; the hourly analysis is late and flat there)
        n = len(c["cells"])
        q0 = sim(c["p"], np.ones(c["p"].shape[0]), np.zeros(n), np.full(n, 1e-3), float(np.exp(-1.0 / 72.0)), np.zeros(n), 0, np.zeros(n),
                 0.0, 1.0, c["lag"], c["loc"], c["a"], max(DEF["kf"] * c["tl"], 0.25))
        ceil = float(q0[e["i0"]:e["iw"] + 1].max())
        e.update(pid=pid, name=pid, q_obs=min(q, ANCHOR_CAP * ceil), official=q, zero_loss=round(ceil), weight=wgt, anchor=True, lower=lower,
                 note=note, regulated=False, censored=False)
        anchors.append(e)
    return D, cats, rows, anchors, cnet, cps


def rain_doubt(r):
    """the truth analysis and the SAIH pluviometers disagree by more than a factor 2 on the event rain (e.g. truth episode
    2024-12-11 gives 105 mm over the Vernissa where the pluviometers measured 5-37 mm and the river did not move)"""
    g = r.get("rain_gauges")
    if r["t0"] < "2024-12-01":         # before that most SAIH pluviometer series do not exist yet: no second opinion
        return False
    return g is not None and max(g, r["rain"]) >= 40.0 and not (0.5 <= (g + 5.0) / (r["rain"] + 5.0) <= 2.0)


def clean(rows):
    """events whose observed flow can be trusted as this catchment's own runoff"""
    return [r for r in rows if "q_obs" in r and not r.get("flat") and not r.get("glitch") and not r.get("excluded") and r["cover"] >= 0.7
            and not r["censored"] and not rain_doubt(r)
            and not (r["regulated"] and r["release_peak"] > 0.3 * max(r["q_obs"], 1.0))]      # the dam release dominates


def usable(r):
    """event rows that can be compared with the model: clean, and with enough rain or flow to say something"""
    if r.get("anchor"):
        return True
    return bool(clean([r])) and (r["rain"] >= 30.0 or r["q_obs"] >= c_off(r["area"]))


def episodes(rows):
    """episode id per row: rows whose rain periods overlap (any catchment) belong together"""
    iv = sorted((r["i0"], r["i1"] + 24, k) for k, r in enumerate(rows))
    out, cur, end = {}, -1, -10**9
    for a, b, k in iv:
        if a > end:
            cur += 1
        end = max(end, b)
        out[k] = cur
    return np.array([out[k] for k in range(len(rows))])


def attributes():
    """attribute table of gauged catchments and control points (geo/hydro/gauges/out/attributes.json), regression columns"""
    a = json.loads((GAUGES / "out" / "attributes.json").read_text(encoding="utf-8"))
    out = {}
    for pid, r in a.items():
        out[pid] = dict(carb=(r["carb_high"] or 0) + (r["carb_med"] or 0), carb_high=r["carb_high"] or 0, marl_clay=r["marl_clay"] or 0,
                        quaternary=r["quaternary"] or 0, perm_rank=r["perm_rank"] or 3.0, forest=r["forest"] or 0, crops=r["crops"] or 0,
                        urban=r["urban"] or 0, map_mm=r["map_mm"], slope=r["slope"], ln_area=float(np.log(max(r["area_km2"], 1.0))),
                        ln_q2_spec=float(np.log(max(r["q2_spec"] or 0.3, 0.01))), p0i_mm=r["p0i_mm"] or 20.0, kind=r["kind"],
                        elev=r["elev_mean_m"] or 0.0, lon=r["lon"])
    return out


# ====================================================================================================== simulations
def run_grid(cats, rows, combos, fields, label=""):
    """peak (C, N) m3/s and runoff (C, N) mm of every row under every combo (dicts with the keys of DEF)."""
    by = {}
    for k, r in enumerate(rows):
        by.setdefault(r["pid"], []).append(k)
    C, N = len(combos), len(rows)
    peak, vol = np.zeros((C, N), np.float32), np.zeros((C, N), np.float32)
    cbs = [dict(DEF, **cb) for cb in combos]

    def one(pid):
        c = cats[pid]
        ks = by[pid]
        n = len(c["cells"])
        i0 = np.array([rows[k]["i0"] for k in ks]); iw = np.array([rows[k]["iw"] for k in ks])
        mults, lags = {}, {}
        for ci, cb in enumerate(cbs):
            if cb["ratio"] not in mults:
                mults[cb["ratio"]] = wet_mult(c["R"], cb["ratio"])
            vs = cb["vs"]
            if vs not in lags:
                lags[vs] = c["lag"] if vs == 1.0 else np.floor((c["lag"] + 0.5) / vs).astype(np.int64)
            q = sim(c["p"], mults[cb["ratio"]], cb["p0"] * fields[cb["pat"]][c["cells"]], np.full(n, cb["s"]), float(np.exp(-1.0 / cb["tau"])),
                    np.full(n, cb["phi"]), int(cb["form"]), np.full(n, cb["alpha"]), float(cb["p0b"]), float(cb["sb"]),
                    lags[vs], c["loc"], c["a"], max(cb["kf"] * c["tl"] / vs, 0.25))
            for j, k in enumerate(ks):
                seg = q[i0[j]:iw[j] + 1]
                peak[ci, k] = seg.max()
                vol[ci, k] = seg.sum() * 3.6 / c["area"]
        return pid

    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(one, list(by)))
    print(f"  simulations {label}: {C} parameter sets x {len(by)} catchments in {time.time() - t0:.0f} s", flush=True)
    return peak, vol


def cached_grid(cats, rows, combos, fields, label):
    key = hashlib.sha1(json.dumps([combos, [(r["pid"], r["i0"], r["iw"]) for r in rows],
                                   (FLOWS / "rain_grid.npz").stat().st_mtime], default=str, sort_keys=True).encode()).hexdigest()[:12]
    f = FLOWS / f"grid_{label}_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["peak"], z["vol"]
    peak, vol = run_grid(cats, rows, combos, fields, label)
    np.savez_compressed(f, peak=peak, vol=vol)
    return peak, vol


def run_cells(cats, rows, par, timing=False):
    """Simulation with per-grid-cell parameters, as the pipeline will run it: par = dict(p0, s, phi, alpha: (NY*NX,) arrays,
    form, tau, p0b, sb, kf).  Returns the peak of every row (and the hour index of the peak)."""
    by = {}
    for k, r in enumerate(rows):
        by.setdefault(r["pid"], []).append(k)
    peak, tpk = np.zeros(len(rows)), np.zeros(len(rows), int)
    for pid, ks in by.items():
        c = cats[pid]
        cl = c["cells"]
        q = sim(c["p"], np.ones(c["p"].shape[0]), par["p0"][cl].astype(float), par["s"][cl].astype(float), float(np.exp(-1.0 / par["tau"])),
                par["phi"][cl].astype(float), int(par["form"]), par["alpha"][cl].astype(float), float(par["p0b"]), float(par["sb"]),
                c["lag"], c["loc"], c["a"], max(par["kf"] * c["tl"], 0.25))
        for k in ks:
            seg = q[rows[k]["i0"]:rows[k]["iw"] + 1]
            peak[k] = seg.max(); tpk[k] = rows[k]["i0"] + int(np.argmax(seg))
    return (peak, tpk) if timing else peak


def resid(peak, rows):
    """(..., N) residuals ln((sim + c)/(obs + c)); for lower-bound observations only under-prediction counts"""
    obs = np.array([r["q_obs"] for r in rows]); c = c_off([r["area"] for r in rows])
    res = np.log((peak + c) / (obs + c))
    low = np.array([bool(r.get("lower")) for r in rows])
    res[..., low] = np.minimum(res[..., low], 0.0)
    return res


def score(res, w):
    """weighted mean absolute residual"""
    return (np.abs(res) * w).sum(axis=-1) / w.sum()


def stats(r, w):
    m = float((r * w).sum() / w.sum())
    return dict(mae=round(float((np.abs(r) * w).sum() / w.sum()), 3), sd=round(float(np.sqrt((w * (r - m) ** 2).sum() / w.sum())), 3),
                bias=round(m, 3), n=int(len(r)))


def loo(res, w, groups, sel=None):
    """leave-one-group-out residuals for a family of parameter sets (rows of `res`, optionally the subset `sel`):
    for each group the set with the best score on the other groups predicts the group."""
    idx = np.arange(res.shape[0]) if sel is None else np.asarray(sel)
    out = np.zeros(res.shape[1])
    a = np.abs(res[idx]) * w
    tot = a.sum(axis=1)
    for g in np.unique(groups):
        m = groups == g
        ci = idx[np.argmin((tot - a[:, m].sum(axis=1)) / (w.sum() - w[m].sum()))]
        out[m] = res[ci, m]
    return out


# ====================================================================================================== empirical picture
def _auc(x, y):
    from scipy.stats import rankdata
    r = rankdata(x); n1 = y.sum(); n0 = (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def empirical(rows, attrs, out=None):
    """What the measurements say before any model: does a catchment respond, and how much, as a function of the event depth,
    the intensity and the antecedent rain."""
    from scipy.stats import spearmanr
    res = {}
    for dom_name, sel in (("ravine domain", lambda r: in_domain(r["pid"], attrs)), ("interior headwaters and La Mancha", lambda r: not in_domain(r["pid"], attrs))):
        ok = [r for r in clean(rows) if sel(r)]
        if len(ok) < 20:
            continue
        A = np.array([r["area"] for r in ok]); c = c_off(A)
        q = np.array([r["q_obs"] for r in ok]); y = q >= c
        tot = np.array([r["rain"] for r in ok]); i1 = np.array([r["rain1_cell"] for r in ok])
        rc = np.array([r["rc"] for r in ok]); ro = np.array([r["runoff_mm"] for r in ok])
        d = dict(n_events=len(ok), n_catchments=len({r["pid"] for r in ok}), n_respond=int(y.sum()), auc={}, spearman_runoff={})
        print(f"\n== empirical picture, {dom_name}: {len(ok)} clean events in {d['n_catchments']} catchments; {int(y.sum())} with a peak above c(A)")

        def col(k):
            return np.array([r[k] if r.get(k) is not None else np.nan for r in ok], float)
        for k in ("rain", "rain12", "rain1", "rain1_cell", "rain5d", "rain30d", "api"):
            x = col(k); m = np.isfinite(x)
            if m.sum() > 20 and y[m].any() and (~y[m]).any():
                d["auc"][k] = round(_auc(x[m], y[m]), 3)
        print("  does it respond? area under the ROC curve of each predictor:", d["auc"])
        big = tot >= 40
        for k in ("rain", "rain12", "rain1", "rain1_cell", "rain5d", "rain30d", "api"):
            x = col(k); m = np.isfinite(x) & big
            if m.sum() > 20:
                d["spearman_runoff"][k] = [round(float(spearmanr(x[m], ro[m])[0]), 2), round(float(spearmanr(x[m], rc[m])[0]), 2), int(m.sum())]
        print("  events >= 40 mm, Spearman of [runoff mm, runoff coefficient] with:", d["spearman_runoff"])
        eb = [15, 30, 50, 80, 120, 200, 1000]; ib = [0, 10, 20, 35, 60, 500]
        tab = []
        print("  response frequency; rows = event rain (mm), columns = largest 1-h rain on a 5-km cell (mm)")
        print("            " + "".join(f"{ib[j]:>4d}-{ib[j + 1]:<5d}" for j in range(len(ib) - 1)) + "   median rc   p90 rc")
        for a, b in zip(eb[:-1], eb[1:]):
            line, rowt = f"  {a:4d}-{b:<5d}", []
            for j in range(len(ib) - 1):
                m = (tot >= a) & (tot < b) & (i1 >= ib[j]) & (i1 < ib[j + 1])
                rowt.append([int(y[m].sum()), int(m.sum())])
                line += f"{(str(int(y[m].sum())) + '/' + str(int(m.sum()))):>10s}" if m.any() else f"{'-':>10s}"
            m = (tot >= a) & (tot < b)
            med, p90 = (float(np.median(rc[m])), float(np.quantile(rc[m], 0.9))) if m.any() else (np.nan, np.nan)
            print(line + f"   {med:9.4f} {p90:8.3f}")
            tab.append(dict(rain=[a, b], by_intensity=rowt, rc_median=round(med, 4), rc_p90=round(p90, 3)))
        d["response_table"] = dict(intensity_bins=ib, rows=tab)
        m = big & np.isfinite(col("api"))
        if m.sum() > 30:
            X = np.column_stack([np.ones(m.sum()), np.log(tot[m]), np.log(np.maximum(i1[m], 1.0)), col("api")[m] / 100.0])
            Y = np.log(ro[m] + 0.05)
            d["loglinear"] = {}
            for cols, name in (([0, 1], "ln rain"), ([0, 2], "ln i1"), ([0, 1, 2], "ln rain + ln i1"), ([0, 1, 3], "ln rain + API/100")):
                bb, *_ = np.linalg.lstsq(X[:, cols], Y, rcond=None)
                r2 = 1 - ((Y - X[:, cols] @ bb) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
                d["loglinear"][name] = dict(r2=round(float(r2), 2), coef=[round(float(v), 2) for v in bb])
                print(f"  ln(runoff + 0.05 mm) ~ {name:20s} R2 {r2:.2f} coef {np.round(bb, 2)}")
        by = {}
        for r, yy in zip(ok, y):
            by.setdefault(r["pid"], []).append((r, bool(yy)))
        d["catchments"] = {}
        print("  per catchment (events >= 30 mm or with response): n, responded, largest rain without response, smallest with, max rc, max peak")
        for pid, L in sorted(by.items()):
            L = [(r, yy) for r, yy in L if r["rain"] >= 30 or yy]
            if not L:
                continue
            no = [r["rain"] for r, yy in L if not yy]; yes = [r["rain"] for r, yy in L if yy]
            e = dict(name=L[0][0]["name"], area=round(L[0][0]["area"]), n=len(L), responded=len(yes), max_dry_mm=max(no) if no else None,
                     min_wet_mm=min(yes) if yes else None, max_rc=round(max(r["rc"] for r, _ in L), 3), max_q=round(max(r["q_obs"] for r, _ in L), 1))
            d["catchments"][pid] = e
            print(f"    {pid:12s} {e['name'][:24]:24s} A {e['area']:5d} n {e['n']:2d} resp {e['responded']:2d}  max dry {e['max_dry_mm'] or float('nan'):6.1f}"
                  f"  min wet {e['min_wet_mm'] or float('nan'):6.1f}  max rc {e['max_rc']:.3f}  max q {e['max_q']:7.1f}")
        res[dom_name] = d
    if out:
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def in_domain(pid, attrs):
    a = attrs.get(pid)
    return a is not None and (a["kind"] != "gauge" or (a["elev"] <= ELEV_MAX and a["lon"] >= LON_MIN)) or pid == "forata-in"


# ====================================================================================================== regional families
def families(combos, res, peak, w, ep, cat, anc, obs):
    """In-sample, leave-one-episode-out and leave-one-catchment-out error of each regional model family (one parameter set
    for every catchment); returns the table and the index of the best set of every family."""
    cb = [dict(DEF, **c) for c in combos]
    g = lambda k: np.array([c[k] for c in cb])                # noqa: E731
    p0, s, tau, form, phi, ratio, kf, alpha, pat = g("p0"), g("s"), g("tau"), g("form"), g("phi"), g("ratio"), g("kf"), g("alpha"), g("pat")
    sc = score(res, w)
    base = (tau == 72) & (ratio == 1) & (kf == 0.3) & (alpha == 0)
    uni = pat == "uniform"
    fam = [
        ("in production: P0 120, S 150, hard phi 45", uni & base & (p0 == 120) & (s == 150) & (form == 1) & (phi == 45), 0),
        ("regional P0, S (no intensity term)", uni & base & (form == 0), 2),
        ("+ memory tau (72 or 144 h)", uni & (ratio == 1) & (kf == 0.3) & (alpha == 0) & (form == 0), 3),
        ("+ hard excess max(p - phi, 0)", uni & base & (form <= 1), 3),
        ("+ soft excess p exp(-phi/p)", uni & base & (form != 1), 3),
        ("hard excess 45 + antecedent state (API on P0)", uni & (tau == 72) & (kf == 0.3) & (alpha == 0) & (form == 1) & (phi == 45), 3),
        ("q5 rule: P0 dry 100-120, wet/dry 0.19, hard 45", uni & (tau == 72) & (kf == 0.3) & (alpha == 0) & (form == 1) & (phi == 45) & (ratio == 0.19) & ((p0 == 100) | (p0 == 120)), 2),
        ("+ Clark K 0.15 or 0.3", uni & (tau == 72) & (ratio == 1) & (alpha == 0), 4),
        ("P0 per cell = k * P0i (CEDEX raster)", (pat == "p0i") & base, 3),
        ("P0 per cell = k * P0i * beta", (pat == "p0i_beta") & base, 3),
        ("P0 per cell = k * P0i * beta + state", (pat == "p0i_beta") & (tau == 72) & (kf == 0.3) & (alpha == 0), 4),
        ("two components: responsive share alpha", uni & (tau == 72) & (ratio == 1) & (kf == 0.3) & (alpha > 0), 6),
        ("anything in the grid", np.ones(len(cb), bool), 8),
    ]
    table, best = [], {}
    print(f"\n== one parameter set for all catchments ({res.shape[1]} rows, {len(np.unique(ep))} episodes, {len(np.unique(cat))} catchments)")
    print(f"  {'family':45s} {'P0   S  tau form phi ratio  kf alpha p0b  sb':47s} {'in-sample':>15s} {'leave-episode':>15s} {'leave-catchm.':>15s}  2024")
    for name, m, npar in fam:
        idx = np.nonzero(m)[0]
        if idx.size == 0:
            continue
        b = int(idx[np.argmin(sc[idx])])
        le, lc = loo(res, w, ep, idx), loo(res, w, cat, idx)
        s_in, s_e, s_c = stats(res[b], w), stats(le, w), stats(lc, w)
        ar = float(np.exp(np.mean(np.log((peak[b, anc] + 1.0) / obs[anc])))) if anc.any() else np.nan
        c_ = cb[b]
        table.append(dict(family=name, n_par=npar, **{k: c_[k] for k in ("p0", "s", "tau", "phi", "ratio", "pat", "kf", "alpha", "p0b", "sb")},
                          form=FORM_NAME[int(c_["form"])], in_sample=s_in, leave_episode_out=s_e, leave_catchment_out=s_c, anchors_ratio=round(ar, 2)))
        best[name] = b
        print(f"  {name:45s} {c_['p0']:4.0f} {c_['s']:4.0f} {c_['tau']:4.0f} {int(c_['form']):2d} {c_['phi']:4.0f}  {c_['ratio']:.2f} {c_['kf']:.2f} {c_['alpha']:.2f} {c_['p0b']:4.0f} {c_['sb']:4.0f} {c_['pat'][:8]:8s}"
              f" {s_in['mae']:.3f} sd {s_in['sd']:.2f} | {s_e['mae']:.3f} sd {s_e['sd']:.2f} | {s_c['mae']:.3f} sd {s_c['sd']:.2f} | x{ar:.2f}")
    return table, best


# ====================================================================================================== per-catchment alpha
class Profile:
    """Residuals of every row as a function of the catchment factor f on alpha (the regional parameters fixed): ln(sim + c)
    is interpolated between the simulated factors onto the fine grid FINE of ln f.  R (F, N) residuals, A = w |R|."""

    def __init__(self, peak_f, rows, w, factors=FACTORS):
        self.c = c_off([r["area"] for r in rows])
        lq = np.log(peak_f + self.c)
        self.lo = np.log(np.array([r["q_obs"] for r in rows]) + self.c)
        low = np.array([bool(r.get("lower")) for r in rows])
        lnf = np.log(np.asarray(factors))
        self.LQ = np.stack([np.interp(FINE, lnf, lq[:, k]) for k in range(len(rows))], axis=1)
        self.R = self.LQ - self.lo
        self.R[:, low] = np.minimum(self.R[:, low], 0.0)
        self.w = w
        self.A = np.abs(self.R) * w

    def _interp(self, M, lnf):
        pos = (np.clip(lnf, FINE[0], FINE[-1]) - FINE[0]) / DF
        i = np.minimum(pos.astype(int), len(FINE) - 2)
        fr = pos - i
        k = np.arange(M.shape[1])
        return M[i, k] * (1 - fr) + M[i + 1, k] * fr

    def at(self, lnf):
        """residuals of all rows, row k evaluated at lnf[k]"""
        return self._interp(self.R, np.asarray(lnf, float))

    def peak_at(self, lnf):
        return np.exp(self._interp(self.LQ, np.asarray(lnf, float))) - self.c

    def curve(self, ks):
        """loss of a set of rows as a function of ln f (on FINE)"""
        return self.A[:, ks].sum(axis=1)


def map_estimate(curve, mu, tau, b):
    """MAP of ln f for one catchment: loss / b + (ln f - mu)^2 / (2 tau^2); a flat minimum (no flood seen) gives the value
    nearest to the prior mean."""
    obj = curve / b + ((FINE - mu) ** 2 / (2.0 * tau ** 2) if np.isfinite(tau) else 0.0)
    j = np.flatnonzero(obj <= obj.min() + 1e-9)
    return float(FINE[j[np.argmin(np.abs(FINE[j] - mu))]])


def catchment_factors(prof, groups, fit, w, ep, mu=None, taus=TAUS_SHRINK, label=""):
    """Partial pooling of the per-catchment factor (ln f ~ N(mu_c, tau^2)); tau chosen by leave-one-episode-out.
    Only the rows flagged in `fit` inform the factor of their catchment (the 2024 anchors do not: the rain analysis caps
    them); all rows are evaluated.  Returns dict(tau, lnf {group: value}, loeo and in-sample residuals)."""
    N = len(groups)
    by, byfit = {}, {}
    for k, g in enumerate(groups):
        by.setdefault(g, []).append(k)
        if fit[k]:
            byfit.setdefault(g, []).append(k)
    mu = mu or {}
    mu_row = np.array([mu.get(g, 0.0) for g in groups])
    b = max(float((np.abs(prof.at(mu_row)) * w).sum() / w.sum()), 0.05)
    best = None
    for tau in taus:
        lf = mu_row.copy()
        for g, ks in byfit.items():
            ks = np.array(ks); allk = np.array(by[g])
            for e in set(ep[allk]):
                tr = ks[ep[ks] != e]
                if tr.size:
                    lf[allk[ep[allk] == e]] = map_estimate(prof.curve(tr), mu.get(g, 0.0), tau, b)
        held = prof.at(lf)
        s = stats(held, w)
        if label:
            print(f"    {label}: shrinkage tau {tau:4.2f} -> leave-episode-out MAE {s['mae']:.3f} sd {s['sd']:.3f}")
        if best is None or s["mae"] < best[0] - 1e-4:
            best = (s["mae"], tau, held, s)
    tau = best[1]
    lnf = {g: (map_estimate(prof.curve(byfit[g]), mu.get(g, 0.0), tau, b) if g in byfit else mu.get(g, 0.0)) for g in by}
    ins = prof.at(np.array([lnf[g] for g in groups]))
    return dict(tau=tau, lnf=lnf, loeo=best[2], loeo_stats=best[3], ins=ins, ins_stats=stats(ins, w), b=b,
                n_fit={g: len(v) for g, v in byfit.items()})


def fit_regression(L, X, b, ridge=0.02, x0=None):
    """ln f_c = X_c . beta minimising the summed event loss; L (G, F) loss curves of the catchments, X (G, n)."""
    G = np.arange(L.shape[0])

    def loss(beta):
        pos = (np.clip(X @ beta, FINE[0], FINE[-1]) - FINE[0]) / DF
        i = np.minimum(pos.astype(int), len(FINE) - 2)
        fr = pos - i
        return float((L[G, i] * (1 - fr) + L[G, i + 1] * fr).sum()) / b + ridge * float((beta[1:] ** 2).sum()) * len(G)

    n = X.shape[1]
    x0 = np.zeros(n) if x0 is None else np.asarray(x0, float)
    best = None
    for start in (x0, x0 + np.r_[0.5, np.zeros(n - 1)], x0 - np.r_[0.5, np.zeros(n - 1)]):
        r = minimize(loss, start, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-5, maxiter=3000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def regression(prof, groups, fit, w, attrs, names):
    """ln f_c = b0 + sum b_k z_k(c) fitted on the event loss of the gauged catchments (rows flagged in `fit`);
    z = attribute standardised over those catchments.  Leave-one-catchment-out residuals for the same rows."""
    gs = sorted({g for g, f in zip(groups, fit) if f})
    gi = np.array([gs.index(g) if g in gs else -1 for g in groups])
    use = np.array(fit, bool) & (gi >= 0)
    mat = np.array([[attrs[g][n] for n in names] for g in gs], float) if names else np.zeros((len(gs), 0))
    mean, sd = (mat.mean(axis=0), mat.std(axis=0) + 1e-9) if names else (np.zeros(0), np.ones(0))
    X = np.column_stack([np.ones(len(gs)), (mat - mean) / sd])
    L = np.stack([prof.curve(np.nonzero(use & (gi == i))[0]) for i in range(len(gs))])
    b = max(float((np.abs(prof.at(np.zeros(len(groups))))[use] * w[use]).sum() / w[use].sum()), 0.05)
    beta = fit_regression(L, X, b)
    lf = np.zeros(len(groups))
    for i in range(len(gs)):                                  # leave-one-catchment-out
        keep = np.arange(len(gs)) != i
        bt = fit_regression(L[keep], X[keep], b, x0=beta)
        lf[gi == i] = float(X[i] @ bt)
    held = prof.at(lf)
    pred = {g: float(np.clip(X[i] @ beta, FINE[0], FINE[-1])) for i, g in enumerate(gs)}
    ins = prof.at(np.array([pred.get(g, 0.0) for g in groups]))
    return dict(names=list(names), beta=[float(x) for x in beta], mean=[float(x) for x in mean], sd=[float(x) for x in sd], pred=pred,
                loco_stats=stats(held[use], w[use]), ins_stats=stats(ins[use], w[use]))


# expected sign of the effect of an attribute on the responsive share (0 = no expectation)
SIGN = dict(p0i_mm=-1, urban=1, crops=1, ln_q2_spec=1, marl_clay=1, carb_high=-1, carb=-1, perm_rank=1, forest=-1)
# an ungauged point never gets less than 0.2 x or more than 2 x the regional share: the regression rests on 26 catchments
# and a share that is too high gives false floods with ordinary rain (the only gauge of the dry south, Sax, never ran)
LNF_UNGAUGED = (np.log(0.2), np.log(2.0))


def predict_lnf(reg, attr_row):
    """regression value for an ungauged catchment; attributes outside the gauged range count as 2 sd at most"""
    z = [float(np.clip((attr_row[n] - m) / s, -2.0, 2.0)) for n, m, s in zip(reg["names"], reg["mean"], reg["sd"])]
    return float(np.clip(np.r_[1.0, z] @ np.array(reg["beta"]), *LNF_UNGAUGED))


def select_attributes(prof, groups, fit, w, attrs, max_n=2):
    """forward selection of the regression attributes by leave-one-catchment-out error"""
    chosen, cur = [], regression(prof, groups, fit, w, attrs, [])
    log = [dict(names=[], loco=cur["loco_stats"], in_sample=cur["ins_stats"])]
    print(f"    regression of ln(alpha factor): constant only -> leave-catchment-out MAE {cur['loco_stats']['mae']:.3f} sd {cur['loco_stats']['sd']:.3f}")
    while len(chosen) < max_n:
        trials = []
        for a in ATTRS:
            if a in chosen or (a == "carb" and "carb_high" in chosen) or (a == "carb_high" and "carb" in chosen):
                continue
            r = regression(prof, groups, fit, w, attrs, chosen + [a])
            trials.append((r["loco_stats"]["mae"], a, r))
        trials.sort(key=lambda x: x[0])
        print("      + " + "  ".join(f"{a} {m:.3f} ({r['beta'][-1]:+.2f})" for m, a, r in trials[:8]))
        for m, a, r in trials:
            log.append(dict(names=chosen + [a], loco=r["loco_stats"], in_sample=r["ins_stats"], beta=[round(x, 3) for x in r["beta"]],
                            sign_ok=bool(SIGN.get(a, 0) * r["beta"][-1] >= 0)))
        # only attributes whose effect has the physically expected sign (26 catchments can produce any correlation), and
        # only if they improve the leave-one-catchment-out error by 0.02
        ok = [(m, a, r) for m, a, r in trials if SIGN.get(a, 0) * r["beta"][-1] >= 0]
        if ok and ok[0][0] < cur["loco_stats"]["mae"] - 0.02:
            chosen.append(ok[0][1]); cur = ok[0][2]
        else:
            break
    return cur, log


# ====================================================================================================== routing check
def timing(D, cats, rows, par, vscales=(0.5, 0.75, 1.0, 1.5), kfs=(0.15, 0.3, 0.6)):
    """Is the wave on time?  For every clear flood (peak >= 3 c(A)) the lag between the simulated and the observed
    hydrograph (hours, + = the real river is later than the model) for several velocity scales (travel times divided by
    vs) and Clark constants; the lag is the shift that maximises the correlation of the two hourly series.  Losses are set
    low on purpose (responsive share 30 %) so that the model produces a wave to compare in every event."""
    sel = [r for r in rows if not r.get("anchor") and r["q_obs"] >= 3.0 * c_off(r["area"])]
    out = []
    for vs in vscales:
        for kf in kfs:
            lags, who = [], []
            for pid in sorted({r["pid"] for r in sel}):
                c = cats[pid]
                o = D.obs(pid)
                n = len(c["cells"])
                lag = c["lag"] if vs == 1.0 else np.floor((c["lag"] + 0.5) / vs).astype(np.int64)
                q = sim(c["p"], np.ones(D.T), np.full(n, par["p0"]), np.full(n, par["s"]), float(np.exp(-1.0 / par["tau"])), np.full(n, par["phi"]),
                        int(par["form"]), np.full(n, 0.3), float(par["p0b"]), float(par["sb"]), lag, c["loc"], c["a"], max(kf * c["tl"] / vs, 0.25))
                for r in sel:
                    if r["pid"] != pid:
                        continue
                    a, b = r["i0"], r["iw"]
                    qo = np.nan_to_num(o["qmax"][a:b + 1] - r["base"])
                    qs = q[a:b + 1]
                    if qs.max() <= 0 or len(qo) < 8:
                        continue
                    best = (-2.0, 0)
                    for L in range(-12, 37):
                        x, y = (qs[:len(qs) - L], qo[L:]) if L >= 0 else (qs[-L:], qo[:len(qo) + L])
                        if len(x) > 6 and x.std() > 0 and y.std() > 0:
                            cc = float(np.corrcoef(x, y)[0, 1])
                            if cc > best[0]:
                                best = (cc, L)
                    if best[0] > 0.5:
                        lags.append(best[1]); who.append(c["tl"])
            if lags:
                lags, who = np.array(lags), np.array(who)
                out.append(dict(vscale=vs, clark_k=kf, n=int(len(lags)), lag_median_h=float(np.median(lags)),
                                lag_q25_h=float(np.quantile(lags, 0.25)), lag_q75_h=float(np.quantile(lags, 0.75)),
                                lag_over_travel_median=round(float(np.median(lags / np.maximum(who, 1.0))), 2)))
    print("\n== routing: lag of the observed hydrograph behind the simulated one (hours), floods with peak >= 3 c(A)")
    for d in out:
        print(f"  velocity x{d['vscale']:.2f} Clark K {d['clark_k']:.2f} t_long: n {d['n']:3d}  lag median {d['lag_median_h']:+5.1f} "
              f"(q25 {d['lag_q25_h']:+5.1f}, q75 {d['lag_q75_h']:+5.1f})  lag / travel time {d['lag_over_travel_median']:+.2f}")
    return out


# ====================================================================================================== transmission losses
TL_PAIRS = [("13722", "13102", "Cànyoles: Moixent (227 km2) -> Canals (501 km2)", 3),
            ("13707", "13917", "Monleón at Atzeneta (440 km2) -> Rambla de la Viuda at Vall d'Alba (827 km2)", 3),
            ("2737", "2688", "Guadalaviar: Tramacastilla (96 km2) -> Gea (736 km2)", 6),
            ("13683", "12924", "Alfambra: Villalba Alta (487 km2) -> Teruel (1319 km2)", 8),
            ("13526", "2725", "Mijares: El Terde (661 km2) -> Los Cantos (1408 km2)", 5),
            ("13403", "16915", "Magro: Requena (673 km2) -> Macastre (1055 km2; Forata dam in between)", 8),
            ("12827", "2443", "Albaida: Montaverner (320 km2) -> Manuel (1188 km2; Bellús dam in between)", 6)]


def transmission(D):
    """Flood volume at an upstream gauge against the next gauge downstream on the same stream (flows only, no rain):
    a ratio below 1 although the catchment grows is water lost in the bed."""
    from dataset import despike, _median3
    import saih_series as SS

    def hourly(var):
        f = FLOWS / f"chj_{var}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        v = despike(z["v"].astype(float)); k = np.isfinite(v)
        out = np.full(D.T, np.nan)
        u, val, n = SS.hourly(z["t"][k], _median3(v[k]), "mean")
        pos = (u - D.t[0]).astype(int); ok = (pos >= 0) & (pos < D.T) & (n >= 6)
        out[pos[ok]] = val[ok]
        return out
    res = []
    print("\n== transmission losses: flood volume downstream / upstream")
    for up, dn, name, lag in TL_PAIRS:
        qu, qd = hourly(up), hourly(dn)
        if qu is None or qd is None:
            continue
        base = np.nanmedian(qu)
        hi = np.nonzero(np.nan_to_num(qu) > base + max(2.0, 3 * base))[0]
        if hi.size == 0:
            continue
        ev = []
        for g in np.split(hi, np.nonzero(np.diff(hi) > 48)[0] + 1):
            a, b = max(g[0] - 12, 0), min(g[-1] + 72, D.T - 1)
            bu, bd = np.nanmedian(qu[max(a - 12, 0):a + 1]), np.nanmedian(qd[max(a - 12, 0):a + 1])
            su, sd = qu[a:b + 1], qd[a:min(b + lag, D.T - 1) + 1]
            if np.isnan(su).mean() > 0.2 or np.isnan(sd).mean() > 0.2 or not np.isfinite(bd) or not np.isfinite(bu):
                continue
            vu, vd = np.nansum(np.maximum(su - bu, 0)) * 3600 / 1e6, np.nansum(np.maximum(sd - bd, 0)) * 3600 / 1e6
            if vu >= 0.05:
                ev.append(dict(t=str(D.t[g[0]]), v_up_hm3=round(float(vu), 3), v_down_hm3=round(float(vd), 3), ratio=round(float(vd / vu), 2),
                               peak_up=round(float(np.nanmax(su) - bu), 1), base_down=round(float(bd), 2)))
        if ev:
            ratios = np.array([e["ratio"] for e in ev])
            print(f"  {name}: {len(ev)} flood waves, volume ratio median {np.median(ratios):.2f} (min {ratios.min():.2f}, max {ratios.max():.2f})")
            res.append(dict(pair=name, n=len(ev), ratio_median=float(np.median(ratios)), events=ev))
    return res


# ====================================================================================================== main
def cell_mosaic(cnet, cats, value, default):
    """Per-grid-cell field from per-control-point values: a cell takes the value of the smallest control-point catchment
    that covers it substantially (at least half of the largest area any catchment has in that cell); cells in no
    catchment take `default`.  The same rule is proposed for static.py (coord/findings/q7-hydro.md)."""
    n = grid.NY * grid.NX
    A = np.zeros((len(cnet.ids), n))
    for k, pid in enumerate(cnet.ids):
        A[k, cats[pid]["cells"]] = cats[pid]["w"]
    amax = A.max(axis=0)
    out = np.full(n, float(default))
    done = np.zeros(n, bool)
    for k in np.argsort(cnet.area):                            # smallest catchment first
        m = (A[k] >= 0.5 * amax) & (A[k] > 0) & ~done
        out[m] = value[cnet.ids[k]]
        done |= m
    return out


def main(what="all"):
    import saih_series as SS
    import rain_grid as RG
    offline = what == "nofetch"
    gp = json.loads((GAUGES / "out" / "gauge_points.json").read_text(encoding="utf-8"))["points"]
    if not offline:
        SS.download(SS.flow_vars())
        for p in gp:
            if p["source"] == "saih_segura":
                SS.segura_series(p["var"])
        SS.download([s["var"] for s in SS.rain_stations()])
        if not (FLOWS / "rain_hourly.npz").exists() or what == "fetch":
            SS.rain_hourly()
    truth = list((ROOT / "hindcast" / "truth").glob("*.npz"))
    rg = FLOWS / "rain_grid.npz"
    if not rg.exists() or what == "fetch" or any(f.stat().st_mtime > rg.stat().st_mtime for f in truth):
        RG.build()
    if what == "fetch":
        return
    t_start = time.time()
    D, cats, rows, anchors, cnet, cps = collect(offline=offline)
    attrs = attributes()
    dec = json.loads((GAUGES / "gauges_decisions.json").read_text(encoding="utf-8"))["gauges"]
    gid = {p["var"]: p["id"] for p in gp}
    cp_gauges = {}
    for var, d in dec.items():
        if d.get("use") and d.get("control_point") and var in gid:
            cp_gauges.setdefault(d["control_point"], []).append(gid[var])
    nearest = lambda pid, gs: min(gs, key=lambda g: abs(attrs[g]["ln_area"] - attrs[pid]["ln_area"]))      # noqa: E731
    use = [r for r in rows if usable(r)] + anchors
    anc = np.array([bool(r.get("anchor")) for r in use])
    w = np.array([r.get("weight", 1.0) for r in use])
    obs = np.array([r["q_obs"] for r in use])
    dom = np.array([in_domain(r["pid"], attrs) for r in use])
    ep = episodes(use)
    groups = np.array([nearest(r["pid"], cp_gauges[r["pid"]]) if r.get("anchor") and r["pid"] in cp_gauges else r["pid"] for r in use])
    print(f"{len(rows)} catchment-events with flow, {int((~anc).sum())} usable ({int((dom & ~anc).sum())} in the ravine domain), {int(anc.sum())} anchors; "
          f"{len(np.unique(ep))} episodes; {time.time() - t_start:.0f} s")
    for a in anchors:
        print(f"  anchor {a['pid']:16s} official {a['official']:5.0f}  zero-loss peak of the rain analysis {a['zero_loss']:5d}  target used {a['q_obs']:5.0f}")
    empirical(rows, attrs, out=GAUGES / "out" / "empirical.json")
    fields = p0_fields()

    # ---- stage 1: one parameter set for all (the families the coordinator asked about) ----
    base = [dict(p0=p0, s=s, form=f, phi=phi) for p0 in P0S for s in SS_ for f, phi in EXC]
    combos = [dict(b) for b in base]
    combos += [dict(b, tau=144.0) for b in base if b["form"] == 0]
    combos += [dict(b, ratio=r) for b in base for r in (0.45, 0.19) if b["form"] != 2 and b["phi"] in (0.0, 45.0)]
    combos += [dict(b, kf=0.15) for b in base if b["phi"] in (0.0, 45.0, 100.0)]
    combos += [dict(b, pat=pt) for b in base for pt in ("p0i", "p0i_beta") if b["phi"] in (0.0, 45.0, 100.0)]
    combos += [dict(b, pat="p0i_beta", ratio=0.45) for b in base if b["phi"] in (0.0, 45.0)]
    combos += [dict(b, alpha=a, p0b=pb, sb=sb) for b in base for a, pb, sb in LOW if b["p0"] >= 100 and b["phi"] in (0.0, 45.0, 100.0)]
    if CURRENT not in [dict(DEF, **c) for c in combos]:
        combos.append(dict(CURRENT))
    combos.append(dict(DESIGN))
    full = [dict(DEF, **c) for c in combos]
    peak1, vol1 = cached_grid(cats, use, combos, fields, "regional")
    res1 = resid(peak1, use)
    i_cur, i_des = full.index(CURRENT), len(combos) - 1
    fam_table, fam_best = families(combos[:-1], res1[:-1][:, dom], peak1[:-1][:, dom], w[dom], ep[dom], groups[dom], anc[dom], obs[dom])
    if (~dom).any():
        print(f"  outside the ravine domain ({int((~dom).sum())} rows: interior headwaters, La Mancha): parameters in production -> {stats(res1[i_cur, ~dom], w[~dom])}")

    # ---- stage 2: main component chosen together with one responsive share per catchment ----
    rows2 = [r for r, d in zip(use, dom) if d]
    w2, ep2, g2, anc2 = w[dom], episodes(rows2), groups[dom], anc[dom]
    obs2 = obs[dom]
    cand = [dict(p0=p0, s=s, form=f, phi=phi, p0b=pb, sb=sb) for p0 in P0S2 for s in SS2 for f, phi in EXC2 for pb, sb in LOW2]
    combos2 = [dict(c, alpha=min(ALPHA_REF * f, 0.9)) for c in cand for f in FACTORS]
    peak2, vol2 = cached_grid(cats, rows2, combos2, fields, "joint")
    nF = len(FACTORS)
    k_for = [k for k, r in enumerate(rows2) if r["pid"] == "forata-in"]
    joint = []
    for ci, c in enumerate(cand):
        prof = Profile(peak2[ci * nF:(ci + 1) * nF], rows2, w2)
        cf = catchment_factors(prof, g2, ~anc2, w2, ep2, taus=[1.5])
        tot, wt = float((np.abs(cf["loeo"]) * w2).sum()), float(w2.sum())
        v = None
        if k_for:
            v = float(vol2[ci * nF + FACTORS.index(1.0), k_for[0]])
            tot += FORATA_VOL_WEIGHT * abs(float(np.log((v + 5.0) / (FORATA_RUNOFF_MM + 5.0)))); wt += FORATA_VOL_WEIGHT
        pk = prof.peak_at(np.array([cf["lnf"][g] for g in g2]))
        joint.append(dict(c, score=round(tot / wt, 4), gauges=stats(cf["loeo"][~anc2], w2[~anc2]), anchors=stats(cf["loeo"][anc2], w2[anc2]),
                          anchors_ratio=round(float(np.exp(np.mean(np.log((pk[anc2] + 1.0) / obs2[anc2])))), 2),
                          forata_runoff_mm=None if v is None else round(v, 1)))
    order = np.argsort([j["score"] for j in joint])
    print("\n== main component + one responsive share per catchment (leave-one-episode-out; score = gauges + 2024 anchors + Forata volume)")
    for i in order[:12]:
        j = joint[i]
        print(f"  P0 {j['p0']:4.0f} S {j['s']:4.0f} {FORM_NAME[j['form']]:14s} phi {j['phi']:4.0f} | low P0b {j['p0b']:3.0f} Sb {j['sb']:4.0f} | score {j['score']:.3f} "
              f"gauges MAE {j['gauges']['mae']:.3f} sd {j['gauges']['sd']:.2f} | anchors MAE {j['anchors']['mae']:.2f} x{j['anchors_ratio']:.2f} | Forata runoff {j['forata_runoff_mm']} mm (93 measured)")
    by_form = {f"{FORM_NAME[f]} {phi:g}": min(j["score"] for j in joint if j["form"] == f and j["phi"] == phi) for f, phi in EXC2}
    print("  best score of each excess form:", by_form)
    # the hard threshold at 45 mm/h is what runs today: keep it unless another form is clearly better (0.01 in the score)
    ib = int(order[0])
    hard = [i for i in order if joint[i]["form"] == 1 and joint[i]["phi"] == 45.0]
    if hard and joint[hard[0]]["score"] <= joint[ib]["score"] + 0.01:
        ib = int(hard[0])
    ref = dict(DEF, **cand[ib], alpha=ALPHA_REF)
    prof = Profile(peak2[ib * nF:(ib + 1) * nF], rows2, w2)
    fit2 = ~anc2
    print(f"\n== per-catchment responsive share, reference: {cand[ib]}")
    reg0 = stats(prof.at(np.zeros(len(rows2))), w2)
    print(f"  the same alpha = {ALPHA_REF} everywhere: {reg0}")
    cf_pool = catchment_factors(prof, g2, fit2, w2, ep2, label="one alpha per catchment, pooled to the regional value")
    reg, reg_log = select_attributes(prof, g2, fit2, w2, attrs)
    mu = {g: (reg["pred"][g] if g in reg["pred"] else predict_lnf(reg, attrs[g])) for g in set(g2)}
    cf = catchment_factors(prof, g2, fit2, w2, ep2, mu=mu, label="one alpha per catchment, pooled to the regression")
    print(f"  regression chosen: attributes {reg['names']} beta {np.round(reg['beta'], 3)} -> leave-catchment-out {reg['loco_stats']}")
    print(f"  final (gauged: fitted and shrunk, tau {cf['tau']}): in-sample {cf['ins_stats']}, leave-episode-out {cf['loeo_stats']}")

    # ---- per-control-point parameters ----
    lnf_cp, how = {}, {}
    gauged = {g: k for g, k in cf["n_fit"].items() if k > 0}
    for pid in cnet.ids:
        c = cats[pid]
        mine = [g for g in cp_gauges.get(pid, []) if g in gauged]
        inside = []
        for g in gauged:
            cg = cats[g]
            share = float(c["w"][np.isin(c["cells"], cg["cells"])].sum() / c["w"].sum())
            if share >= 0.8 and 0.8 * c["area"] <= cg["area"] <= 5.0 * c["area"]:      # not a small ravine inside a big gauged basin
                inside.append((cg["area"], g))
        if mine:
            g = nearest(pid, mine)
            lnf_cp[pid] = cf["lnf"][g]; how[pid] = f"gauge {g} on the same stream ({gauged[g]} events), shrunk to the regression"
        elif inside:
            g = min(inside)[1]
            lnf_cp[pid] = cf["lnf"][g]; how[pid] = f"inside the catchment of gauge {g} ({gauged[g]} events)"
        else:
            lnf_cp[pid] = predict_lnf(reg, attrs[pid]); how[pid] = "regression on " + (", ".join(reg["names"]) or "nothing (regional value)")
    alpha_cp = {pid: float(np.clip(ALPHA_REF * np.exp(v), 0.0, 0.5)) for pid, v in lnf_cp.items()}
    alpha_default = float(ALPHA_REF * np.exp(np.clip(reg["beta"][0], *LNF_UNGAUGED)))
    n = grid.NY * grid.NX
    par = dict(p0=np.full(n, ref["p0"]), s=np.full(n, ref["s"]), phi=np.full(n, ref["phi"]), alpha=cell_mosaic(cnet, cats, alpha_cp, alpha_default),
               form=ref["form"], tau=ref["tau"], p0b=ref["p0b"], sb=ref["sb"], kf=ref["kf"])
    np.save(FLOWS / "alpha_cells.npy", par["alpha"])
    pk_dep = run_cells(cats, use, par)
    r_dep = resid(pk_dep, use)
    dep = dict(ravine_domain=stats(r_dep[dom], w[dom]), gauges=stats(r_dep[dom & ~anc], w[dom & ~anc]),
               outside_domain=stats(r_dep[~dom], w[~dom]) if (~dom).any() else None)
    cA = c_off([r["area"] for r in use])
    gdom = dom & ~anc

    def counts(pk):
        return dict(hits=int(((pk >= cA) & (obs >= cA) & gdom).sum()), misses=int(((pk < cA) & (obs >= cA) & gdom).sum()),
                    false_alarms=int(((pk >= cA) & (obs < cA) & gdom).sum()))
    prod = dict(ravine_domain=stats(res1[i_cur, dom], w[dom]), gauges=stats(res1[i_cur, gdom], w[gdom]),
                outside_domain=stats(res1[i_cur, ~dom], w[~dom]) if (~dom).any() else None)
    print(f"\n== as it would run in the pipeline (per-cell alpha mosaic): ravine domain {dep['ravine_domain']} | gauges only {dep['gauges']}")
    print(f"   in production today on the same rows:                     ravine domain {prod['ravine_domain']} | gauges only {prod['gauges']}")
    print(f"   peak above c(A), gauges: new {counts(pk_dep)}  | in production {counts(peak1[i_cur])}")
    print("   2024 anchors (official, target used, new, in production):",
          [(r["pid"], int(r["official"]), int(r["q_obs"]), int(pk_dep[k]), int(peak1[i_cur, k])) for k, r in enumerate(use) if r.get("anchor")])

    tm = timing(D, cats, [r for r, d in zip(use, dom) if d], ref)
    tl = transmission(D)

    # ---- outputs ----
    names = {p["id"]: p["name"] for p in gp}
    g2cp = {g: cp for cp, gs in cp_gauges.items() for g in gs}
    events = []
    for k, r in enumerate(use):
        if not dom[k] or not (r.get("anchor") or obs[k] >= cA[k] or pk_dep[k] >= cA[k] or r["rain"] >= 80.0):
            continue
        events.append(dict(case=r["t0"][:10], point=r["pid"] if r.get("anchor") else (g2cp.get(r["pid"]) or names.get(r["pid"], r["pid"])),
                           gauge=None if r.get("anchor") else r["pid"], rain_mm=round(r["rain"]),
                           observed=round(float(r["official"] if r.get("anchor") else obs[k]), 1),
                           before=round(float(peak1[i_des, k]), 1), production=round(float(peak1[i_cur, k]), 1), after=round(float(pk_dep[k]), 1),
                           rain_source=r.get("src"), anchor=bool(r.get("anchor"))))
    events.sort(key=lambda e: (e["point"], e["case"]))
    c5 = lambda pk: float(np.log((pk[dom] + 5.0) / (obs[dom] + 5.0)).std())                # noqa: E731  the measure of the first fit
    fit = dict(
        p0_mm=ref["p0"], s_mm=ref["s"], phi_mmh=ref["phi"] if ref["form"] else None, wet_memory_h=ref["tau"],
        phi_form=FORM_NAME[ref["form"]], alpha_ref=ALPHA_REF, alpha_default=round(alpha_default, 4), p0b_mm=ref["p0b"], sb_mm=ref["sb"], clark_k=ref["kf"],
        score=dep["ravine_domain"]["mae"], n=int(dom.sum()),
        residuals=dict(mean_ln=dep["ravine_domain"]["bias"], sd_ln=dep["ravine_domain"]["sd"], mae_ln=dep["ravine_domain"]["mae"],
                       offset="c = 5 m3/s * (A / 184 km2)^0.75", sd_ln_plus5=round(c5(pk_dep), 3),
                       production_sd_ln=prod["ravine_domain"]["sd"], production_mae_ln=prod["ravine_domain"]["mae"],
                       production_sd_ln_plus5=round(c5(peak1[i_cur]), 3)),
        events=events,
        generated=time.strftime("%Y-%m-%d %H:%M"), n_catchments=int(len({r["pid"] for r, d in zip(use, gdom) if d})),
        n_episodes=int(len(np.unique(ep[dom]))),
        model="net = (1 - alpha) [p c(W; P0, S) + (1 - c) excess(p, phi)] + alpha p c(W; P0b, Sb);  W(t) = W(t-1) exp(-1/tau) + p(t)",
        families=fam_table, joint_search=[joint[i] for i in order[:15]], joint_best_by_form=by_form,
        validation=dict(same_alpha_everywhere=reg0,
                        per_catchment_pooled_to_regional=dict(tau=cf_pool["tau"], leave_episode_out=cf_pool["loeo_stats"], in_sample=cf_pool["ins_stats"]),
                        regression=dict(attributes=reg["names"], beta=reg["beta"], mean=reg["mean"], sd=reg["sd"],
                                        leave_catchment_out=reg["loco_stats"], in_sample=reg["ins_stats"], search=reg_log),
                        per_catchment_pooled_to_regression=dict(tau=cf["tau"], leave_episode_out=cf["loeo_stats"], in_sample=cf["ins_stats"]),
                        deployed=dep, deployed_counts=counts(pk_dep), production=prod, production_counts=counts(peak1[i_cur])),
        anchors=[dict(point=r["pid"], official=r["official"], zero_loss_peak=r["zero_loss"], target_used=round(float(r["q_obs"])), new=round(float(pk_dep[k])),
                      production=round(float(peak1[i_cur, k])), note=r["note"], lower_bound=bool(r.get("lower"))) for k, r in enumerate(use) if r.get("anchor")],
        catchments={g: dict(name=names.get(g, g), alpha=round(float(np.clip(ALPHA_REF * np.exp(v), 0, 0.5)), 4), n_events=cf["n_fit"].get(g, 0),
                            alpha_regression=round(float(ALPHA_REF * np.exp(mu[g])), 4)) for g, v in sorted(cf["lnf"].items())},
        timing=tm, transmission=tl)
    (ROOT / "hindcast" / "hydro_fit.json").write_text(json.dumps(fit, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    lp = {pid: dict(p0_mm=ref["p0"], s_mm=ref["s"], phi_mmh=ref["phi"] if ref["form"] else None, phi_form=FORM_NAME[ref["form"]],
                    alpha=round(alpha_cp[pid], 4), p0b_mm=ref["p0b"], sb_mm=ref["sb"], how=how[pid]) for pid in cnet.ids}
    (ROOT / "geo" / "hydro" / "loss_params.json").write_text(json.dumps(lp, indent=1, ensure_ascii=False), encoding="utf-8")
    (GAUGES / "out" / "fit_catchments.json").write_text(json.dumps(fit["catchments"], indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: hindcast/hydro_fit.json, geo/hydro/loss_params.json, geo/hydro/gauges/out/fit_catchments.json ({time.time() - t_start:.0f} s)")
    for pid in cnet.ids:
        print(f"  {pid:24s} alpha {lp[pid]['alpha']:.4f}  {lp[pid]['how']}")


# ====================================================================================================== peak-intensity excess
# The intensity-excess term of production sees the hourly MEAN of a 5-km cell; the bursts that make the big floods live inside
# the cell (Moixent 12 Sep 2019: wettest cell-hour mean 43 mm/h, 1-km maximum 76 mm/h).  The rain fields carry the cell maximum
# (radar analysis: 1-km max o_max; Member.p in the pipeline).  Sub-cell model without new parameters ("cone"): the intensities
# of the cell, sorted, fall as i(a) = P (1 - a)^k over the area share a, with k = P/m - 1 so that their mean is m.  The volume
# above phi is then, for P > phi (r = phi / P):
#     excess = m (1 - r^((k+1)/k)) - phi (1 - r^(1/k))         (k -> 0: max(m - phi, 0), the production term)
# form 3 = that; form 4 = the simpler (m / P) * max(P - phi, 0); form 1 = production (mean only).

@njit(cache=True)
def _excess(pt, pk, phi, form):
    if form == 1:
        return pt - phi if pt > phi else 0.0
    if form == 5:               # pk IS the excess, precomputed (sub-hourly radar ladder, see sub_pk)
        return pk if pk < pt else pt
    P = pk if pk > pt else pt
    if P <= phi:
        return 0.0
    if form == 4:
        return pt / P * (P - phi)
    k = P / pt - 1.0
    if k < 1e-3:
        return pt - phi if pt > phi else 0.0
    r = phi / P
    return pt * (1.0 - r ** ((k + 1.0) / k)) - phi * (1.0 - r ** (1.0 / k))


@njit(cache=True, nogil=True)
def sim_pk(p, pk, p0, s, decay, phi, form, alpha, p0b, sb, lag, loc, a, K):
    """sim() with the excess term computed from the cell mean p AND the cell maximum pk (T, n); forms 1 / 3 / 4 above."""
    T, n = p.shape
    w = np.zeros(n)
    last = np.zeros(n, np.int64)
    ptr = np.zeros(n + 1, np.int64)
    for j in range(lag.size):
        ptr[loc[j] + 1] += 1
    for c in range(n):
        ptr[c + 1] += ptr[c]
    raw = np.zeros(T + lag.max() + 2)
    for t in range(T):
        for c in range(n):
            pt = p[t, c]
            if pt <= 0.0:
                continue
            wd = w[c] * decay ** (t - last[c] + 1)
            wm = wd + 0.5 * pt
            x = wm - p0[c]
            cc = 0.0
            if x > 0.0:
                cc = x * (x + 2.0 * s[c]) / ((x + s[c]) * (x + s[c]))
            e = pt * cc + (1.0 - cc) * _excess(pt, pk[t, c], phi, form)
            al = alpha[c]
            if al > 0.0:
                xb = wm - p0b
                cb = 0.0
                if xb > 0.0:
                    cb = xb * (xb + 2.0 * sb) / ((xb + sb) * (xb + sb))
                e = (1.0 - al) * e + al * pt * cb
            w[c] = wd + pt
            last[c] = t + 1
            if e > 0.0:
                for j in range(ptr[c], ptr[c + 1]):
                    raw[t + lag[j]] += a[j] * e
    c1 = 1.0 - np.exp(-1.0 / K)
    q = np.zeros(T)
    prev = 0.0
    for t in range(T):
        prev = c1 * 0.278 * raw[t] + (1.0 - c1) * prev
        q[t] = prev
    return q


PK_VARIANTS = [("production: hard 45 on the cell mean", 1, 45.0)] + \
              [(f"cone on the cell max, phi {f:g}", 3, f) for f in (30.0, 45.0, 60.0, 80.0, 100.0, 130.0)] + \
              [(f"(mean/max) x (max - phi), phi {f:g}", 4, f) for f in (30.0, 45.0, 60.0, 80.0, 100.0, 130.0)]


def _tables(net):
    """per point: cells, lag/loc/a sorted by loc"""
    out = []
    for k in range(net.n):
        lag = np.concatenate([np.full(A.getrow(k).nnz, L) for L, A in enumerate(net.lags)]).astype(np.int64)
        cl = np.concatenate([A.getrow(k).indices for A in net.lags])
        ar = np.concatenate([A.getrow(k).data for A in net.lags]).astype(float)
        cells, loc = np.unique(cl, return_inverse=True)
        o = np.argsort(loc, kind="stable")
        out.append((cells, lag[o], loc[o].astype(np.int64), ar[o]))
    return out


# ---- sub-hourly radar excess (q12): the rain above phi mm/h inside the hour, from the excess ladder of the radar rate frames
# (riua.radar.qpe.burst_ladder, built by hindcast/subhourly/features.py on GitHub Actions into hindcast/cache/subhourly/).
# form 5 of _excess: the excess is precomputed per cell-hour; hours without a ladder keep the production term max(p - 45, 0).
SUB = ROOT / "hindcast" / "cache" / "subhourly"
SUB_VARIANTS = [("production: hard 45 on the cell mean", 1, 45.0)] + \
               [(f"sub-hourly excess, phi {f:g}", 5, f) for f in (30.0, 45.0, 60.0, 70.0, 80.0, 90.0, 100.0, 130.0, 160.0)]


def sub_ladder(t_end):
    """RAW excess ladders of the hours ending at t_end: (pos (T,) index into lad or -1, lad (K, L, NY*NX) float16)."""
    from riua.radar import qpe
    t_end = np.asarray(t_end).astype("datetime64[h]")
    days = (t_end - np.timedelta64(1, "h")).astype("datetime64[D]")
    pos = np.full(len(t_end), -1, np.int64)
    rows = []
    for d in np.unique(days):
        f = SUB / (str(d).replace("-", "") + ".npz")
        if not f.exists():
            continue
        z = np.load(f)
        te = z["t_end"].astype("datetime64[h]")
        lad = z["ladder"].reshape(len(te), len(qpe.LADDER_U), -1)
        idx = np.nonzero(days == d)[0]
        j = np.searchsorted(te, t_end[idx])
        ok = (j < len(te)) & (te[np.minimum(j, len(te) - 1)] == t_end[idx])
        ok &= np.isfinite(lad[np.minimum(j, len(te) - 1), 0]).any(axis=1)
        for i, jj in zip(idx[ok], j[ok]):
            pos[i] = len(rows)
            rows.append(lad[jj])
    return pos, (np.stack(rows) if rows else np.zeros((0, len(qpe.LADDER_U), grid.NY * grid.NX), np.float16))


def sub_pk(pos, lad, cells, P, phi, phi_h=45.0):
    """(T, n) excess above phi (mm in the hour) of the cells: from the ladder where it exists (gauge correction = P / radar),
    the production hourly term max(P - phi_h, 0) elsewhere."""
    from riua.radar import qpe
    PK = np.maximum(P - phi_h, 0.0).astype(np.float32)
    idx = np.nonzero(pos >= 0)[0]
    if idx.size:
        l = lad[:, :, cells][pos[idx]].astype(np.float32)               # (k, L, n)
        x = qpe.ladder_excess(np.moveaxis(l, 1, 0), P[idx], phi)
        PK[idx] = np.where(np.isfinite(x), x, PK[idx])
    return PK


def peak_floods(variants=None, out_name="peak_test.json"):
    """Joint test of the peak-intensity excess on the ordinary floods (gauges, Sep 2024 - 2026) and on q9's big floods
    (hindcast/floods, 2007-2023), leave-one-event-out.  Writes hindcast/obs/flows/peak_test.json and prints the tables.
    variants: (name, form, phi) list (default PK_VARIANTS; SUB_VARIANTS = the sub-hourly radar excess, form 5)."""
    from dataset import Data, gauge_net
    from riua import params as RP
    from riua.core import hydro as H
    PKV = variants or PK_VARIANTS
    hp = RP.load()["hydro"]
    phi_h = 45.0                      # the production hourly term, kept where no sub-hourly ladder exists
    base = dict(p0=hp["p0_mm"], s=hp["s_mm"], tau=hp["wet_memory_h"], p0b=hp.get("p0b_mm", 10.0), sb=hp.get("sb_mm", 100.0), kf=hp["clark_k"])
    decay = float(np.exp(-1.0 / base["tau"]))
    cnet, cps = static.hydro_net()
    thr = H.level_thresholds(cnet, hp)
    gnet, gpts = gauge_net()
    fitc = json.loads((GAUGES / "out" / "fit_catchments.json").read_text(encoding="utf-8"))
    galpha = np.array([fitc.get(i, {}).get("alpha", 0.03) for i in gnet.ids])
    ctab, gtab = _tables(cnet), _tables(gnet)
    calpha = cnet.alpha if cnet.alpha is not None else np.full(grid.NY * grid.NX, 0.03)

    def run(p, pk, tab, net, alpha_of, form, phi):
        q = np.zeros((p.shape[0], net.n))
        for k, (cells, lag, loc, ar) in enumerate(tab):
            n = len(cells)
            q[:, k] = sim_pk(np.ascontiguousarray(p[:, cells]), np.ascontiguousarray(pk[:, cells]), np.full(n, base["p0"]), np.full(n, base["s"]),
                             decay, phi, form, alpha_of(k, cells), base["p0b"], base["sb"], lag, loc, ar, max(base["kf"] * net.tc_h[k], 0.25))
        return q

    c_al = lambda k, cells: calpha[cells].astype(float)                     # noqa: E731
    g_al = lambda k, cells: np.full(len(cells), galpha[k])                   # noqa: E731
    # ---------------- big floods (q9): cases before the SAIH history (the 2024-25 truth episodes are in the ordinary set)
    FL = ROOT / "hindcast" / "floods"
    cases = [c for c in json.loads((FL / "cases.json").read_text(encoding="utf-8"))["cases"] if c["id"] < "2024"]
    facts = json.loads((FL / "out" / "results.json").read_text(encoding="utf-8"))["rows"]
    big_rows, unver = [], {}
    for name, form, phi in PKV:
        unver[name] = 0
    sims = {}
    use_sub = any(form == 5 for _, form, _ in PKV)
    allc = np.arange(grid.NY * grid.NX)
    sub_cover = {}
    for c in cases:
        f = FL / "cache" / "rain" / f"{c['id']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        T = len(z["t_end"])
        p = np.nan_to_num(z["o_mean"].reshape(T, -1)).astype(np.float32)
        pk = np.maximum(np.nan_to_num(z["o_max"].reshape(T, -1)).astype(np.float32), p)
        if use_sub:
            spos, slad = sub_ladder(z["t_end"])
            sub_cover[c["id"]] = [int((spos >= 0).sum()), T]
        for vi, (name, form, phi) in enumerate(PKV):
            pkv = sub_pk(spos, slad, allc, p, phi, phi_h) if form == 5 else pk
            sims[(c["id"], vi, "control")] = run(p, pkv, ctab, cnet, c_al, form, phi).max(axis=0)
            sims[(c["id"], vi, "gauge")] = run(p, pkv, gtab, gnet, g_al, form, phi).max(axis=0)
        print("  big-flood case", c["id"], "sub-hourly hours", sub_cover.get(c["id"]), flush=True)
    case_ids = {c["id"] for c in cases}
    for r in facts:
        if r["event"] not in case_ids:
            continue
        net, kind = (cnet, "control") if r["kind"] == "control" else (gnet, "gauge")
        if r["point"] not in net.ids:
            continue
        k = net.ids.index(r["point"])
        sims_r = [float(sims[(r["event"], vi, kind)][k]) for vi in range(len(PKV))]
        big_rows.append(dict(r, sims=sims_r))
        if kind == "control" and variants is not None:      # q12: the overflow discharge as production has it today
            big_rows[-1]["capacity_q9"], big_rows[-1]["capacity_used"] = r.get("capacity_used"), float(thr[2, k])
    # unverified simulated overflows: control points without any fact in that case
    has_fact = {(r["event"], r["point"]) for r in facts}
    unver_list = {name: [] for name, _, _ in PKV}
    for c in cases:
        for vi, (name, form, phi) in enumerate(PKV):
            if (c["id"], vi, "control") not in sims:
                continue
            pkv = sims[(c["id"], vi, "control")]
            lst = [f"{c['id']} {pid}" for k, pid in enumerate(cnet.ids) if pkv[k] >= thr[2, k] and (c["id"], pid) not in has_fact]
            unver[name] += len(lst)
            unver_list[name] += lst
    # ---------------- ordinary floods: gauge events of the ravine domain, peak field = truth o_max inside the episodes
    D = Data(offline=True)
    attrs = attributes()
    rows = [r for r in D.events(write=False) if "q_obs" in r and usable(r) and in_domain(r["pid"], attrs)]
    pk_all = D.p_all.copy()
    for f in sorted((ROOT / "hindcast" / "truth").glob("*.npz")):
        zt = np.load(f, allow_pickle=True)
        te = zt["t_end"].astype("datetime64[h]")
        pos = (te - D.t[0]).astype(int)
        ok = (pos >= 0) & (pos < D.T) & (D.src[np.clip(pos, 0, D.T - 1)] == 1)
        pk_all[pos[ok]] = np.nan_to_num(zt["o_max"].reshape(len(te), -1)[ok])
    if use_sub:
        opos, olad = sub_ladder(D.t)
        print(f"  ordinary period: {int((opos >= 0).sum())} of {D.T} hours have a sub-hourly ladder", flush=True)
    ord_sim = np.zeros((len(PKV), len(rows)))
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["pid"], []).append(i)
    for pid, ii in by.items():
        k = gnet.ids.index(pid)
        c = D.cat(k)
        cells, lag, loc, ar = gtab[k]
        assert np.array_equal(cells, c["cells"])
        P = c["p"].astype(np.float32)
        PK = pk_all[:, cells]
        PK = np.maximum(np.where(np.isfinite(PK), PK, P), P).astype(np.float32)
        for vi, (name, form, phi) in enumerate(PKV):
            pkv = sub_pk(opos, olad, cells, P, phi, phi_h) if form == 5 else PK
            q = sim_pk(np.ascontiguousarray(P), np.ascontiguousarray(pkv), np.full(len(cells), base["p0"]), np.full(len(cells), base["s"]), decay, phi, form,
                       np.full(len(cells), galpha[k]), base["p0b"], base["sb"], lag, loc, ar, max(base["kf"] * gnet.tc_h[k], 0.25))
            for i in ii:
                ord_sim[vi, i] = q[rows[i]["i0"]:rows[i]["iw"] + 1].max()
    # ordinary period, control points: simulated overflows outside 28 Oct - 5 Nov 2024 (none documented in 2025-26)
    ord_over = np.zeros(len(PKV), int)
    ord_over_2526 = np.zeros(len(PKV), int)
    ord_over_list = {name: [] for name, _, _ in PKV}
    quiet = (D.t < np.datetime64("2024-10-28")) | (D.t > np.datetime64("2024-11-06"))
    y2526 = D.t >= np.datetime64("2025-01-01")
    for k, pid in enumerate(cnet.ids):
        c = D.cat(k, cnet)
        cells, lag, loc, ar = ctab[k]
        P = c["p"].astype(np.float32)
        PK = pk_all[:, cells]
        PK = np.maximum(np.where(np.isfinite(PK), PK, P), P).astype(np.float32)
        for vi, (name, form, phi) in enumerate(PKV):
            pkv = sub_pk(opos, olad, cells, P, phi, phi_h) if form == 5 else PK
            q = sim_pk(np.ascontiguousarray(P), np.ascontiguousarray(pkv), np.full(len(cells), base["p0"]), np.full(len(cells), base["s"]), decay, phi, form,
                       calpha[cells].astype(float), base["p0b"], base["sb"], lag, loc, ar, max(base["kf"] * cnet.tc_h[k], 0.25))
            over = (q >= thr[2, k]) & quiet
            starts = np.nonzero(np.diff(np.r_[0, over.astype(int)]) == 1)[0]          # separate overflow episodes
            ord_over[vi] += len(starts)
            ord_over_2526[vi] += int(y2526[starts].sum())
            ord_over_list[name] += [f"{str(D.t[s])[:13]} {pid} {q[s:s + 48].max():.0f}/{thr[2, k]:.0f}" for s in starts]
    # ---------------- scores
    ob = np.array([r["q_obs"] for r in rows]); cA = c_off([r["area"] for r in rows])
    r_ord = np.log((ord_sim + cA) / (ob + cA))
    nat = [r for r in big_rows if r["documented_peak"] and not r["regulated"] and r["peak_kind"] in ("measured", "estimated", "modelled")]
    dp = np.array([r["documented_peak"] for r in nat]); cB = c_off([r["area_km2"] for r in nat])
    sb_ = np.array([r["sims"] for r in nat]).T
    r_big = np.log((sb_ + cB) / (dp + cB))
    isbig = dp >= 100
    ovr = [r for r in big_rows if r["kind"] == "control" and r["documented_overflow"] is not None]
    table = []
    print("\n== peak-intensity excess (P0 %g, S %g, per-cell alpha as deployed)" % (base["p0"], base["s"]))
    print(f"  {'variant':38s} | ordinary {len(rows)} events: MAE  bias  hit/miss/false | big floods: all {len(nat)} MAE bias | >=100 ({int(isbig.sum())}) MAE bias median ratio "
          f"| overflow H/M/F/N | unverified | ordinary-period overflows")
    for vi, (name, form, phi) in enumerate(PKV):
        ro, rb = r_ord[vi], r_big[vi]
        hits = int(((ord_sim[vi] >= cA) & (ob >= cA)).sum()); miss = int(((ord_sim[vi] < cA) & (ob >= cA)).sum()); fa = int(((ord_sim[vi] >= cA) & (ob < cA)).sum())
        H_ = sum(1 for r in ovr if r["documented_overflow"] and r["sims"][vi] >= r["capacity_used"])
        M_ = sum(1 for r in ovr if r["documented_overflow"] and r["sims"][vi] < r["capacity_used"])
        F_ = sum(1 for r in ovr if not r["documented_overflow"] and r["sims"][vi] >= r["capacity_used"])
        N_ = sum(1 for r in ovr if not r["documented_overflow"] and r["sims"][vi] < r["capacity_used"])
        d = dict(variant=name, form=form, phi=phi, ord_mae=round(float(np.abs(ro).mean()), 3), ord_bias=round(float(ro.mean()), 3), ord_hmf=[hits, miss, fa],
                 big_mae=round(float(np.abs(rb).mean()), 3), big_bias=round(float(rb.mean()), 3),
                 big100_mae=round(float(np.abs(rb[isbig]).mean()), 3), big100_bias=round(float(rb[isbig].mean()), 3),
                 big100_median_ratio=round(float(np.median(sb_[vi][isbig] / dp[isbig])), 2), overflow=[H_, M_, F_, N_],
                 unverified=unver[name], ordinary_overflows=int(ord_over[vi]), overflows_2025_26=int(ord_over_2526[vi]),
                 unverified_list=unver_list[name], ordinary_overflow_list=ord_over_list[name])
        table.append(d)
        print(f"  {name:38s} | {d['ord_mae']:.3f} {d['ord_bias']:+.2f}  {hits}/{miss}/{fa} | {d['big_mae']:.3f} {d['big_bias']:+.2f} | {d['big100_mae']:.3f} {d['big100_bias']:+.2f} "
              f"x{d['big100_median_ratio']:.2f} | {H_}/{M_}/{F_}/{N_} | {unver[name]} | {int(ord_over[vi])} ({int(ord_over_2526[vi])} in 2025-26)")
    # leave-one-event-out over both sets (ordinary episodes + big-flood cases), choice by pooled MAE
    ev_o = np.array(["o%03d" % e for e in episodes(rows)])
    ev_b = np.array([r["event"] for r in nat])
    allr = np.concatenate([r_ord, r_big], axis=1)
    allev = np.concatenate([ev_o, ev_b])
    held = np.zeros(allr.shape[1]); picks = {}; pick_of = {}
    for e in np.unique(allev):
        m = allev == e
        vi = int(np.argmin(np.abs(allr[:, ~m]).mean(axis=1)))
        held[m] = allr[vi, m]; picks[PKV[vi][0]] = picks.get(PKV[vi][0], 0) + 1
        pick_of[e] = vi
    # the overflow call of each big-flood case with the variant picked without that case
    lo = [0, 0, 0, 0]
    for r in ovr:
        vi = pick_of.get(r["event"], int(np.argmin(np.abs(allr).mean(axis=1))))
        sim_over = r["sims"][vi] >= r["capacity_used"]
        lo[(0 if sim_over else 1) if r["documented_overflow"] else (2 if sim_over else 3)] += 1
    no, nb = len(rows), len(nat)
    loo = dict(ordinary=stats(held[:no], np.ones(no)), big=stats(held[no:], np.ones(nb)), big100=stats(held[no:][isbig], np.ones(int(isbig.sum()))),
               overflow=lo, picks=picks)
    print("  leave-one-event-out (variant picked on the other events, pooled MAE):", loo)
    # q12: the same, but the ordinary floods and the big floods weigh alike in the choice, and a variant that adds
    # simulated overflows in the ordinary period (the false-alarm guard) cannot be picked
    guard = ord_over <= ord_over[0]
    held2 = np.zeros(allr.shape[1]); picks2 = {}; pick2 = {}
    for e in np.unique(allev):
        m = allev == e
        sc = 0.5 * np.abs(r_ord[:, ~m[:no]]).mean(axis=1) + 0.5 * np.abs(r_big[:, ~m[no:]]).mean(axis=1)
        sc = np.where(guard, sc, np.inf)
        vi = int(np.argmin(sc))
        held2[m] = allr[vi, m]; picks2[PKV[vi][0]] = picks2.get(PKV[vi][0], 0) + 1; pick2[e] = vi
    lo2 = [0, 0, 0, 0]
    for r in ovr:
        vi = pick2.get(r["event"], 0)
        sim_over = r["sims"][vi] >= r["capacity_used"]
        lo2[(0 if sim_over else 1) if r["documented_overflow"] else (2 if sim_over else 3)] += 1
    hb = held2[no:][isbig]
    loo["balanced_guarded"] = dict(ordinary=stats(held2[:no], np.ones(no)), big=stats(held2[no:], np.ones(nb)),
                                   big100=stats(hb, np.ones(int(isbig.sum()))), overflow=lo2, picks=picks2)
    print("  leave-one-event-out (balanced choice, guard: no added ordinary-period overflow):", loo["balanced_guarded"])
    print("  floods >= 100 m3/s (measured, then each variant):")
    for r in sorted([r for r in nat if r["documented_peak"] >= 100], key=lambda r: r["event"]):
        print(f"    {r['event']} {r['point']:16s} {r['rain_input']:6s} obs {r['documented_peak']:6.0f} | " + " ".join(f"{v:6.0f}" for v in r["sims"]))
    print("  overflow facts (documented, capacity, then each variant):")
    for r in sorted(ovr, key=lambda r: (r["event"], r["point"])):
        print(f"    {r['event']} {r['point']:18s} {'yes' if r['documented_overflow'] else 'no ':3s} cap {r['capacity_used']:6.0f} | " + " ".join(f"{v:6.0f}" for v in r["sims"]))
    out = dict(params=base, table=table, leave_one_event_out=loo, n_ordinary=no, n_big=nb, sub_hourly_cover=sub_cover,
               resid=dict(ordinary=np.round(r_ord, 4).tolist(), big=np.round(r_big, 4).tolist(), ev_ordinary=ev_o.tolist(),
                          ev_big=ev_b.tolist(), big100=isbig.tolist(), points_big=[r["point"] for r in nat]),
               big_floods=[dict(event=r["event"], point=r["point"], rain=r["rain_input"], obs=r["documented_peak"],
                                sims={PKV[i][0]: round(v, 1) for i, v in enumerate(r["sims"])}) for r in nat if r["documented_peak"] >= 100],
               overflow_facts=[dict(event=r["event"], point=r["point"], documented=r["documented_overflow"], capacity=r["capacity_used"],
                                    sims={PKV[i][0]: round(v, 1) for i, v in enumerate(r["sims"])}) for r in ovr],
               check_production_vs_q9=[(r["event"], r["point"], r["simulated_peak"], round(r["sims"][0], 1)) for r in big_rows[:12]])
    (FLOWS / out_name).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "peak":
        peak_floods()
    elif len(sys.argv) > 1 and sys.argv[1] == "subhourly":       # q12: py -3.11 hindcast/calibrate_hydro.py subhourly [feature dir] [tag]
        if len(sys.argv) > 2:
            SUB = Path(sys.argv[2])
        peak_floods(SUB_VARIANTS, f"subhourly_test{'_' + sys.argv[3] if len(sys.argv) > 3 else ''}.json")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "all")
