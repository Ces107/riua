"""Hindcast: re-run the risk model on forecasts that existed at the time, score it, tune it.

    python hindcast/run.py            (inside the Linux environment: needs pysteps for the nowcast part)

Truth      hindcast/truth/*.npz   radar-gauge analysis, hourly, on the Riuà grid (build_truth.py)
Forecasts  48 h   day-ahead runs of AROME 2.5 km, ICON-EU, ARPEGE (+ IFS) from the Open-Meteo
                  Previous Runs archive ("previous_day1": issued 24-30 h before each hour)
           now    radar STEPS nowcast from the OPERA archive blended with the short-lead AROME series
           long   ECMWF ENS 51 members, runs issued 3 and 5 days before each target day
For every horizon:
  1. the same code as the live pipeline turns the scenarios into per-scenario amounts,
  2. sigma and bias of the scenario dressing are chosen to maximise the mean Brier skill score
     over the levels (grid search), with leave-one-case-out cross-validation,
  3. tau per level maximises the critical success index, never below 0.40,
  4. hits / misses / false alarms are counted per cell and frame, and per warning zone and day.
Writes hindcast/results.json and the tuned backend/riua/params.json.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

from riua import ingest, params as P, product, static  # noqa: E402
from riua.core import grid, hydro as H, risk  # noqa: E402
from scipy.special import ndtr  # noqa: E402

H1 = np.timedelta64(1, "h")
PARAMS = P.load()
ST = static.load()
THR = static.thresholds(PARAMS)
MASK = ST.mask & (ST.cv_frac > 0.3)      # score only where the gauge network makes the truth trustworthy
ZONE = ST.zone_idx
TAU_GRID = np.round(np.arange(0.10, 0.81, 0.05), 2)
SIGMAS = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)
BIASES = (0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6)


# ------------------------------------------------------------------------------------------ truth

class Truth:
    def __init__(self):
        ts, om, on, o12 = [], [], [], []
        self.cases = {}
        for f in sorted(glob.glob(str(ROOT / "hindcast" / "truth" / "*.npz"))):
            z = np.load(f, allow_pickle=True)
            if "o12_max" not in z.files or len(z["t_end"]) == 0:
                continue
            ts.append(z["t_end"]); om.append(z["o_max"]); on.append(z["o_mean"]); o12.append(z["o12_max"])
            self.cases[Path(f).stem] = (z["t_end"][0], z["t_end"][-1])
        t = np.concatenate(ts)
        _, first = np.unique(t, return_index=True)
        self.t = t[first]
        self.o_max = np.concatenate(om)[first]
        self.o_mean = np.nan_to_num(np.concatenate(on)[first])
        self.o12 = np.concatenate(o12)[first]
        self.pos = {str(x): k for k, x in enumerate(self.t)}

    def has(self, t0, t1) -> bool:
        return all(str(t0 + (k + 1) * H1) in self.pos for k in range(int((t1 - t0) / H1)))

    def frame(self, t0, t1):
        idx = [self.pos[str(t0 + (k + 1) * H1)] for k in range(int((t1 - t0) / H1))]
        return self.o_max[idx].max(axis=0), self.o12[idx].max(axis=0)

    def case_of(self, t) -> str | None:
        for c, (a, b) in self.cases.items():
            if a <= t <= b:
                return c
        return None


def obs_level(o1, o12):
    L = np.ones(o1.shape, np.uint8)
    for k in range(4):
        L = np.where((o1 >= THR.t1h[k]) | (o12 >= THR.t12h[k]), k + 2, L)
    return L


# ------------------------------------------------------------------------------- forecast blocks
# A block = everything needed to re-score a set of frames with any (sigma, bias):
#   a1, a12 (M, F, N) scaled per-scenario amounts at the N evaluated cells (NaN = not available)
#   w (M, F) weights, o1, o12 (F, N) truth, cells (N,) flat cell index, frames, case

def make_block(members, frames, hz, now, truth: Truth, case, sample_mask=None, radius=None) -> dict | None:
    frames = [f for f in frames if truth.has(*f)]
    if not frames or not members:
        return None
    par = PARAMS if radius is None else {**PARAMS, "radius_km": {**PARAMS["radius_km"], hz: radius}}
    pred = risk.predictors(members, frames, par, hz, now, sample_mask=sample_mask, keep_members=True)
    if pred.a1 is None:
        return None
    s1 = np.array([PARAMS["families"][a["family"]]["s1h"] for a in pred.audit], np.float32)
    s12 = np.array([PARAMS["families"][a["family"]]["s12h"] for a in pred.audit], np.float32)
    ok = MASK & np.isfinite(pred.a12).any(axis=0).all(axis=0)
    cells = np.nonzero(ok.ravel())[0]
    if cells.size == 0:
        return None
    flat = lambda a: a.reshape(*a.shape[:-2], -1)[..., cells]
    r_ev = par["radius_km"][hz]
    o = [tuple(grid.neighbourhood_max(x, r_ev) for x in truth.frame(*f)) for f in frames]
    return dict(a1=(flat(pred.a1) * s1[:, None, None]).astype(np.float32), a12=(flat(pred.a12) * s12[:, None, None]).astype(np.float32),
                w=pred.w.astype(np.float32), o1=np.stack([flat(x[0]) for x in o]), o12=np.stack([flat(x[1]) for x in o]),
                cells=cells, frames=frames, case=case, valid=pred.valid)


def block_prob(b: dict, sigma: float, bias: float) -> np.ndarray:
    """(4, F, N) probabilities with the scenario dressing."""
    t1 = THR.t1h.reshape(4, -1)[:, b["cells"]]
    t12 = THR.t12h.reshape(4, -1)[:, b["cells"]]
    M, F, N = b["a12"].shape
    out = np.zeros((4, F, N), np.float32)
    den = np.zeros((F, N), np.float32)
    for m in range(M):
        ok12, ok1 = np.isfinite(b["a12"][m]), np.isfinite(b["a1"][m])
        w = b["w"][m][:, None] * (ok12 | ok1)
        den += w
        for k in range(4):
            r = np.maximum(np.where(ok1, b["a1"][m] / t1[k], 0.0), np.where(ok12, b["a12"][m] / t12[k], 0.0))
            with np.errstate(divide="ignore"):
                out[k] += w * ndtr(np.log(np.maximum(bias * r, 1e-9)) / sigma)
    out /= np.maximum(den, 1e-9)
    return np.minimum.accumulate(out, axis=0)


def block_obs(b: dict) -> np.ndarray:
    t1 = THR.t1h.reshape(4, -1)[:, b["cells"]]
    t12 = THR.t12h.reshape(4, -1)[:, b["cells"]]
    return np.stack([(b["o1"] >= t1[k]) | (b["o12"] >= t12[k]) for k in range(4)])     # (4, F, N) bool


# ------------------------------------------------------------------------------------- scoring

def brier_skill(blocks, sigma, bias) -> tuple[float, list]:
    num = np.zeros(4); n = 0; ev = np.zeros(4)
    for b in blocks:
        p, o = block_prob(b, sigma, bias), block_obs(b)
        num += ((p - o) ** 2).sum(axis=(1, 2)); ev += o.sum(axis=(1, 2)); n += o[0].size
    bs = num / max(n, 1)
    base = ev / max(n, 1)
    ref = base * (1 - base)
    bss = [float(1 - bs[k] / ref[k]) if ev[k] >= 30 else None for k in range(4)]
    use = [x for x in bss if x is not None]
    return (float(np.mean(use)) if use else -9.0), bss


def tune_dressing(blocks):
    best = None
    for s in SIGMAS:
        for bi in BIASES:
            sc, bss = brier_skill(blocks, s, bi)
            if best is None or sc > best[0]:
                best = (sc, s, bi, bss)
    return best


def contingency(blocks, sigma, bias, tau: dict) -> dict:
    """Counts per level (forecast level >= L vs observed level >= L), per cell-frame and per zone-day."""
    cell = {L: [0, 0, 0, 0] for L in (2, 3, 4, 5)}        # hits, misses, false alarms, correct negatives
    zone = {L: [0, 0, 0, 0] for L in (2, 3, 4, 5)}
    zday = {}
    for b in blocks:
        p, o = block_prob(b, sigma, bias), block_obs(b)
        fl = np.ones(p.shape[1:], np.uint8)
        for k, L in enumerate((2, 3, 4, 5)):
            fl = np.where(p[k] >= tau[L], L, fl)
        ol = 1 + o.sum(axis=0)
        zi = ZONE.ravel()[b["cells"]]
        for k, L in enumerate((2, 3, 4, 5)):
            f, ob = fl >= L, ol >= L
            cell[L][0] += int((f & ob).sum()); cell[L][1] += int((~f & ob).sum())
            cell[L][2] += int((f & ~ob).sum()); cell[L][3] += int((~f & ~ob).sum())
        for fi, (t0, _) in enumerate(b["frames"]):
            day = str(t0)[:10]
            for z in np.unique(zi[zi >= 0]):
                sel = zi == z
                key = (b["case"], day, int(z))
                a = zday.setdefault(key, [1, 1])
                a[0] = max(a[0], int(fl[fi, sel].max())); a[1] = max(a[1], int(ol[fi, sel].max()))
    for (c, d, z), (f, ob) in zday.items():
        for L in (2, 3, 4, 5):
            zone[L][0 if (f >= L and ob >= L) else 1 if (ob >= L) else 2 if (f >= L) else 3] += 1
    return {"cell": cell, "zone_day": zone}


def rates(c):
    h, m, fa, cn = c
    return {"hits": h, "misses": m, "false_alarms": fa, "correct_negatives": cn,
            "POD": round(h / (h + m), 2) if h + m else None, "FAR": round(fa / (h + fa), 2) if h + fa else None,
            "CSI": round(h / (h + m + fa), 2) if h + m + fa else None}


def tune_tau(blocks, sigma, bias, lo: float = 0.10) -> dict:
    po = [(block_prob(b, sigma, bias), block_obs(b)) for b in blocks]      # once per setting
    tau = {}
    for k, L in enumerate((2, 3, 4, 5)):
        best = (-1.0, 0.5)
        for t in TAU_GRID[TAU_GRID >= lo - 1e-9]:
            h = m = fa = 0
            for p, o in po:
                f = p[k] >= t
                h += int((f & o[k]).sum()); m += int((~f & o[k]).sum()); fa += int((f & ~o[k]).sum())
            csi = h / (h + m + fa) if h + m + fa else -1.0
            if csi > best[0] + 0.005:
                best = (csi, float(t))
        tau[L] = best[1] if best[0] >= 0 else None
    # levels with no observed event: one step below the previous level, within the allowed range
    prev = 0.5
    for L in (2, 3, 4, 5):
        if tau[L] is None:
            tau[L] = max(lo, 0.40 if lo >= 0.4 else 0.15, prev - 0.05)
        prev = tau[L]
    return tau


def reliability(blocks, sigma, bias) -> dict:
    edges = np.array([0, .05, .15, .3, .5, .7, .9, 1.0001])
    out = {}
    for k, L in enumerate((2, 3, 4, 5)):
        n = np.zeros(len(edges) - 1); s = np.zeros_like(n); pm = np.zeros_like(n)
        for b in blocks:
            p, o = block_prob(b, sigma, bias)[k].ravel(), block_obs(b)[k].ravel()
            idx = np.digitize(p, edges) - 1
            np.add.at(n, idx, 1); np.add.at(s, idx, o); np.add.at(pm, idx, p)
        out[L] = [{"p_forecast": round(float(pm[i] / n[i]), 3), "observed_freq": round(float(s[i] / n[i]), 3), "n": int(n[i])}
                  for i in range(len(n)) if n[i] > 0]
    return out


def evaluate(blocks, name: str) -> dict:
    cases = sorted({b["case"] for b in blocks})
    sc, sigma, bias, bss = tune_dressing(blocks)
    tau = tune_tau(blocks, sigma, bias)
    tau40 = tune_tau(blocks, sigma, bias, 0.40)
    res = {"horizon": name, "cases": cases, "n_frames": int(sum(len(b["frames"]) for b in blocks)),
           "n_cell_frames": int(sum(b["o1"].size for b in blocks)),
           "observed_cell_frames": {L: int(sum(block_obs(b)[k].sum() for b in blocks)) for k, L in enumerate((2, 3, 4, 5))},
           "tuned": {"sigma": sigma, "bias": bias, "tau": tau, "mean_BSS": round(sc, 3), "BSS": bss},
           "tau_min40": tau40,
           "in_sample": {k: {L: rates(v) for L, v in d.items()} for k, d in contingency(blocks, sigma, bias, tau).items()},
           "in_sample_min40": {k: {L: rates(v) for L, v in d.items()} for k, d in contingency(blocks, sigma, bias, tau40).items()},
           "reliability": reliability(blocks, sigma, bias)}
    # leave one case out: tune on the others, score the held-out case, add the counts up
    if len(cases) >= 3:
        tot = {"cell": {L: [0, 0, 0, 0] for L in (2, 3, 4, 5)}, "zone_day": {L: [0, 0, 0, 0] for L in (2, 3, 4, 5)}}
        tot40 = {"cell": {L: [0, 0, 0, 0] for L in (2, 3, 4, 5)}, "zone_day": {L: [0, 0, 0, 0] for L in (2, 3, 4, 5)}}
        picks = []
        for c in cases:
            train = [b for b in blocks if b["case"] != c]
            test = [b for b in blocks if b["case"] == c]
            _, s, bi, _ = tune_dressing(train)
            t = tune_tau(train, s, bi)
            picks.append({"held_out": c, "sigma": s, "bias": bi, "tau": t})
            cont = contingency(test, s, bi, t)
            cont40 = contingency(test, s, bi, tune_tau(train, s, bi, 0.40))
            for k in tot:
                for L in tot[k]:
                    tot[k][L] = [a + b for a, b in zip(tot[k][L], cont[k][L])]
                    tot40[k][L] = [a + b for a, b in zip(tot40[k][L], cont40[k][L])]
        res["cross_validated"] = {k: {L: rates(v) for L, v in d.items()} for k, d in tot.items()}
        res["cross_validated_min40"] = {k: {L: rates(v) for L, v in d.items()} for k, d in tot40.items()}
        res["cv_picks"] = picks
    return res


# ------------------------------------------------------------------------------------ 48 h blocks

FAM = {"meteofrance_arome_france": ("arome", "cp"), "meteofrance_arome_france_hd": ("arome_hd", "cp"),
       "icon_eu": ("icon_eu", "regional"), "meteofrance_arpege_europe": ("arpege", "regional"), "ecmwf_ifs025": ("ifs", "global")}


def mid_blocks(truth: Truth, lead: int = 1, radius=None) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "hindcast" / "cache" / "openmeteo" / "prev_*.npz"))):
        z = np.load(f, allow_pickle=True)
        t = np.array([np.datetime64(x, "h") for x in z["time"]])
        j = np.floor((z["lat"] - grid.LAT0) / grid.D + 1e-6).astype(int)
        i = np.floor((z["lon"] - grid.LON0) / grid.D + 1e-6).astype(int)
        ok = (j >= 0) & (j < grid.NY) & (i >= 0) & (i < grid.NX)
        sample = np.zeros((grid.NY, grid.NX), bool); sample[j[ok], i[ok]] = True
        leads = z["lead_days"].tolist()
        if lead not in leads:
            continue
        li = leads.index(lead)
        run0 = datetime.fromisoformat(str(t[0]))
        arrs = {}
        for m, model in enumerate(z["models"]):
            a = z["precip"][m, li][:, ok]
            if np.isfinite(a).mean() < 0.3:
                continue
            g = np.full((len(t), grid.NY, grid.NX), np.nan, np.float32)
            g[:, j[ok], i[ok]] = a
            arrs[str(model)] = g
        if not arrs:
            continue
        donor = arrs.get("icon_eu")
        members = []
        for model, g in arrs.items():
            if donor is not None:
                g = np.where(np.isnan(g), donor, g)           # AROME has no data south of 38 N
            key, fam = FAM[model]
            members.append(risk.Member(f"{key} d{lead}", fam, key, run0, t, np.nan_to_num(g, nan=0.0), 1))
        day = np.datetime64(str(t[0])[:10] + "T00", "h")
        frames = []
        while day + 3 * H1 <= t[-1]:
            if day - 12 * H1 >= t[0] - H1:                     # the 12-h window must be inside the series
                frames.append((day, day + 3 * H1))
            day = day + 3 * H1
        # one block per verification case (the file may span several)
        by_case = {}
        for fr in frames:
            c = truth.case_of(fr[1])
            if c:
                by_case.setdefault(c, []).append(fr)
        for c, frs in by_case.items():
            b = make_block(members, frs, "mid", run0, truth, c, sample_mask=sample, radius=radius)
            if b:
                out.append(b)
    return out


# ------------------------------------------------------------------------------------ long blocks

def long_blocks(truth: Truth, lead_days: int) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "hindcast" / "cache" / "ens" / "ifsens_tp_*.npz"))):
        z = np.load(f, allow_pickle=True)
        run = datetime.strptime(Path(f).stem.split("_")[-1], "%Y%m%d%H")
        steps = set(z["steps"].astype(int).tolist())
        base = lead_days * 24
        if not {base - 12, base, base + 12, base + 24} <= steps:
            continue
        t0 = np.datetime64(run + timedelta(days=lead_days), "h")
        fr = (t0, t0 + 24 * H1)
        c = truth.case_of(fr[1])
        if not c or not truth.has(*fr):
            continue
        b = make_block(product.ens_npz_members(z, run), [fr], "long", run, truth, c)
        if b:
            out.append(b)
    return out


# ------------------------------------------------------------------------------------- now blocks

def _radar_rates(t_issue: datetime):
    from riua.radar import qpe
    out = []
    for day in {(t_issue - timedelta(minutes=30)).strftime("%Y%m%d"), t_issue.strftime("%Y%m%d")}:
        f = ROOT / "hindcast" / "cache" / "radar" / f"{day}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for s, code in zip(z["t"], z["dbz"]):
            t = datetime.fromtimestamp(int(s), timezone.utc)
            if timedelta(0) <= (t_issue.replace(tzinfo=timezone.utc) - t) <= timedelta(minutes=25):
                d = code.astype(np.float32) / 2.0 - 32.0
                d[code == 255] = np.nan
                out.append((t, qpe.rain_rate(qpe.despeckle(d))))
    return sorted(out, key=lambda x: x[0])


_DAY0 = {}


def _day0(model: str):
    if model not in _DAY0:
        fs = sorted(glob.glob(str(ROOT / "hindcast" / "cache" / "openmeteo" / "s3" / f"s3_{model}_precipitation_2024-10-27_*.npz")))
        if not fs:
            _DAY0[model] = None
        else:
            z = np.load(fs[0], allow_pickle=True)
            rg = grid.Regridder(z["lat"], z["lon"])
            data = z["data"]
            if model != "icon_eu":
                ic = _day0("icon_eu")
                p = rg(data)
                if ic is not None:
                    p = np.where(np.isnan(p), ic[1], p)
            else:
                p = rg(data)
            _DAY0[model] = (np.array([np.datetime64(x, "h") for x in z["time"]]), np.nan_to_num(p, nan=0.0).astype(np.float32))
    return _DAY0[model]


def now_block(args):
    t_issue, case = args
    truth = Truth()
    h = np.datetime64(t_issue, "h")
    past = truth.t <= h
    keep = past & (truth.t > h - 24 * H1)
    obs = product.Obs(truth.t[keep], truth.o_max[keep], truth.o_mean[keep])
    nwp = []
    for model, key in (("meteofrance_arome_france_hd", "arome_hd"), ("meteofrance_arome_france", "arome")):
        d = _day0(model)
        if d is not None:
            nwp.append(risk.Member(f"{key} corto plazo", "cp", key, t_issue - timedelta(hours=3), d[0], d[1], 1))
    radar_m, _ = product.nowcast_members(_radar_rates(t_issue), obs, nwp, t_issue.replace(tzinfo=timezone.utc))
    members = [product.with_past(m, obs, t_issue) for m in radar_m + nwp]
    frames = product.frames_for("now", t_issue)
    return make_block(members, frames, "now", t_issue, truth, case)


def now_blocks(truth: Truth) -> list[dict]:
    from multiprocessing import Pool
    d = _day0("meteofrance_arome_france_hd")
    if d is None:
        return []
    lo, hi = d[0][0], d[0][-1]
    jobs = []
    for c, (a, b) in truth.cases.items():
        t = max(a, lo) + 12 * H1
        t = np.datetime64(str(t)[:10] + "T00", "h") + 24 * H1 if str(t)[11:13] != "00" else t
        while t + 6 * H1 <= min(b, hi):
            jobs.append((datetime.fromisoformat(str(t)), c))
            t = t + 3 * H1
    print(f"nowcast hindcast: {len(jobs)} issue times", flush=True)
    with Pool(4) as pool:
        out = pool.map(now_block, jobs, chunksize=2)
    return [b for b in out if b]


# -------------------------------------------------------------------------------- hydrology check

def poyo_check(truth: Truth) -> dict | None:
    net, cps = static.hydro_net()
    if net is None or "2024-10-dana" not in truth.cases:
        return None
    sel = (truth.t >= np.datetime64("2024-10-28T00", "h")) & (truth.t <= np.datetime64("2024-10-31T00", "h"))
    t, p = truth.t[sel], truth.o_mean[sel]
    hp = PARAMS["hydro"]
    out = {"observed": "SAIH Rambla del Poyo (A-3): 2283 m3/s at 18:55 local (17:55 UTC) on 29 Oct 2024, then the sensor was lost",
           "rain_input": "radar-gauge analysis, cell means", "runs": []}
    flat = p.reshape(len(t), -1)
    for p0 in (15.0, 25.0, 40.0):
        q = H.route(H.net_rain(flat, p0, hp["wet_memory_h"]), net, hp["clark_k"])
        row = {"p0_mm": p0}
        for pid in ("poyo-chiva", "poyo-ribarroja", "poyo-paiporta", "magro-algemesi"):
            if pid in net.ids:
                k = net.ids.index(pid)
                row[pid] = {"peak_m3s": round(float(q[:, k].max())), "peak_utc": str(t[int(q[:, k].argmax())])}
        out["runs"].append(row)
    return out


def cached(name: str, build):
    """Blocks are expensive (the nowcast ones re-run STEPS): keep them on disk; --rebuild-<name> recomputes."""
    import pickle
    f = ROOT / "hindcast" / "cache" / f"blocks_{name}.pkl"
    if f.exists() and f"--rebuild-{name}" not in sys.argv and "--rebuild" not in sys.argv:
        return pickle.loads(f.read_bytes())
    blocks = build()
    f.write_bytes(pickle.dumps(blocks, protocol=4))
    return blocks


def dump(results: dict) -> None:
    (ROOT / "hindcast" / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    truth = Truth()
    print("truth cases:", {c: (str(a), str(b)) for c, (a, b) in truth.cases.items()}, flush=True)
    results = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "truth_cases": sorted(truth.cases),
               "thresholds_mm": {"1h": [20, 40, 90, 135], "12h": [60, 100, 180, 300]}}
    tuned = {"sigma": {}, "bias": {}, "tau": {}}

    # 48 h: also choose the neighbourhood radius
    best = None
    for radius in (12.0,):          # fixed: the event is "within 12 km", so other radii would score a different event
        blocks = cached("mid", lambda: mid_blocks(truth, 1, radius))
        if not blocks:
            continue
        sc = tune_dressing(blocks)[0]
        print(f"48 h, radius {radius:.0f} km: {len(blocks)} blocks, mean BSS {sc:.3f}", flush=True)
        if best is None or sc > best[0] + 0.003:
            best = (sc, radius, blocks)
    if best:
        res = evaluate(best[2], "mid")
        res["radius_km"] = best[1]
        b2 = cached("mid_d2", lambda: mid_blocks(truth, 2, best[1]))
        if b2:
            t = res["tuned"]
            res["lead_day2_same_settings"] = {k: {L: rates(v) for L, v in d.items()}
                                              for k, d in contingency(b2, t["sigma"], t["bias"], t["tau"]).items()}
        results["mid"] = res
        for k in ("sigma", "bias", "tau"):
            tuned[k]["mid"] = res["tuned"][k]
        tuned["radius_km_mid"] = best[1]
        print(json.dumps(res["tuned"]), flush=True)

    dump(results)
    blocks = cached("long", lambda: long_blocks(truth, 3) + long_blocks(truth, 5))
    if blocks:
        res = evaluate(blocks, "long")
        results["long"] = res
        for k in ("sigma", "bias", "tau"):
            tuned[k]["long"] = res["tuned"][k]
        print("long:", len(blocks), "blocks", json.dumps(res["tuned"]), flush=True)

    if "--no-now" not in sys.argv:
        dump(results)
        blocks = cached("now", lambda: now_blocks(truth))
        if blocks:
            res = evaluate(blocks, "now")
            results["now"] = res
            for k in ("sigma", "bias", "tau"):
                tuned[k]["now"] = res["tuned"][k]
            print("now:", len(blocks), "blocks", json.dumps(res["tuned"]), flush=True)

    results["hydrology_poyo_2024"] = poyo_check(truth)
    (ROOT / "hindcast" / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")

    if "--write-params" in sys.argv:
        out = {"version": f"hindcast-{results['generated'][:10]}", "sigma": tuned["sigma"], "bias": tuned["bias"],
               "tau": {hz: {str(L): v for L, v in t.items()} for hz, t in tuned["tau"].items()}}
        if "radius_km_mid" in tuned:
            out["radius_km"] = {"mid": tuned["radius_km_mid"]}
        P.save(out)
        print("wrote", P.PARAMS_FILE)
    print("done")
