"""Hindcast: re-run what PRODUCTION computes on forecasts that existed at the time, then score and fit it.

    python hindcast/run.py build [now] [mid] [long] [--jobs N] [--cases a,b] [--force]    (WSL: `now` needs pysteps)
    python hindcast/run.py status
    python hindcast/run.py fit [--quick]         -> hindcast/results.json, hindcast/fit.json   (see score.py)
    python hindcast/run.py all [--jobs N]        build what is missing, then fit

A block = one issue time of one horizon. The member list is built the way `product.run_cycle` builds it
(lagged runs with their real age and model weight, `ingest.fill_gaps`, ECMWF ENS with production's member
weight, `product.nowcast_members`, `product.with_past` with the measured hours of the truth analysis as the
observations up to the issue time), and `risk.predictors` - the production function - turns it into
per-scenario amounts, weights and measured shares. Blocks are cached in hindcast/cache/blocks/<horizon>/,
one small file per issue time; re-running `build` only adds the issue times that are missing, so the
command is the same when more truth files or archives appear.

What the archives allow (see coord/findings/q8-verify.md):
  tier A  complete runs kept by q4 from the Open-Meteo S3 `data_run` (2026-06-30 on): AROME 2.5 km, ICON-EU,
          ARPEGE with real run times -> production's lags, ages and leads, whole grid.
  tier B  older cases: only the stitched Previous Runs series exist (day 1 / day 2). A leak-free pseudo-run
          is assembled per model: day 1 where that run was already out at the issue time, day 2 elsewhere.
          Lattice points only; the runs are 5-24 h older than the ones production would have had.
  ENS     3-hourly (6-48 h) when the 00Z run of the day is on disk and is the run production would use
          (issues 08Z and 14Z); 12-hourly (days 2-7) for the frames the archive covers (day +3, +4, +5).
  never   AROME-HD, IFS 0.25 / 9 km (not fetched), ICON-EU-EPS, AROME-IFS, AROME-PI (no archive).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # the machine is shared: one thread per process

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

from riua import ingest, params as P, product, static  # noqa: E402
from riua.core import grid, hydro as H, risk  # noqa: E402

H1 = np.timedelta64(1, "h")
HC = ROOT / "hindcast"
BLOCKS = HC / "cache" / "blocks"
RUNS = HC / "cache" / "openmeteo" / "runs"
ENS3H = HC / "obs" / "ens3h"
ENS12 = HC / "cache" / "ens"
BLOCK_VERSION = 4                               # 4: long blocks carry the IFS runs and AIFS-ENS when archived

PARAMS = P.load()
ST = static.load()
THR = static.thresholds(PARAMS)
MASK = ST.mask                                  # every published cell
MASK_B = ST.mask & (ST.cv_frac > 0.3)           # where the lattice archive of the older cases has data
ZONE = ST.zone_idx

ISSUE_HOURS = (2, 8, 14, 20)
# hours after the run time at which production sees a run as complete (r1-nwp-live, runs of 2026-09-30)
DELAY_H = {"arome_hd": 5.0, "arome": 5.0, "icon_eu": 4.0, "arpege": 4.5, "ifs": 8.0, "ens3h": 8.0, "ens": 8.0}
RUNS_S3 = {k: v["s3"] for k, v in ingest.MODELS.items()}
# variants stored next to the production setting: neighbourhood radius (km) and the gate on measured rain (mm)
RADII = {"now": (12.0,), "mid": (6.0, 20.0), "long": (25.0,)}
GATES = {"now": (0.01, 5.0, 60.0, 1e9), "mid": (), "long": ()}      # 0.01 = measured rain always counts, 1e9 = never
ENS3H_WEIGHT = 0.6                              # product.run_cycle: m.weight *= 0.6 for ifs_ens3h


def tag_r(r: float) -> str:
    return f"@r{r:g}"


def tag_g(g: float) -> str:
    return f"@g{g:g}"


# ------------------------------------------------------------------------------------------ truth

class Truth:
    """Hourly radar-gauge analysis of every case on disk. Arrays are read per case, on demand."""

    def __init__(self):
        self.files, self.cases, self.kind = {}, {}, {}
        hour = {}
        for f in sorted(glob.glob(str(HC / "truth" / "*.npz"))):
            try:
                z = np.load(f, allow_pickle=True)
                if "o12_max" not in z.files or len(z["t_end"]) == 0:
                    continue
                t = z["t_end"]
            except Exception:
                continue                        # a file q4 is writing right now
            c = Path(f).stem
            self.files[c], self.cases[c] = f, (t[0], t[-1])
            for k, x in enumerate(t):
                hour.setdefault(str(x), (c, k))     # a duplicated hour keeps the first file
        self.hour = hour
        try:
            for c in json.loads((HC / "cases.json").read_text(encoding="utf-8"))["cases"]:
                self.kind[c["id"]] = c.get("kind")
        except Exception:
            pass
        self._data = {}

    def _case(self, c):
        if c not in self._data:
            if len(self._data) >= 4:
                self._data.pop(next(iter(self._data)))
            z = np.load(self.files[c], allow_pickle=True)
            self._data[c] = (z["o_max"].astype(np.float32), np.nan_to_num(z["o_mean"]).astype(np.float32),
                             z["o12_max"].astype(np.float32))
        return self._data[c]

    def has(self, t0, t1) -> bool:
        return all(str(t0 + (k + 1) * H1) in self.hour for k in range(int((t1 - t0) / H1)))

    def hours(self, ts):
        """-> o_max, o_mean, o12_max (len(ts), NY, NX) for hours that all exist."""
        om, on, o12 = [], [], []
        for t in ts:
            c, k = self.hour[str(t)]
            d = self._case(c)
            om.append(d[0][k]); on.append(d[1][k]); o12.append(d[2][k])
        return np.stack(om), np.stack(on), np.stack(o12)

    def case_of(self, t) -> str | None:
        x = self.hour.get(str(t))
        return x[0] if x else None

    def has_obs(self, hnow: np.datetime64) -> bool:
        return all(str(hnow - k * H1) in self.hour for k in range(12))

    def obs(self, hnow: np.datetime64, back_h: int = 24):
        """What production would hold at the issue time: `product.Obs` of the last hours, or None when the
        12 h before the issue time are not all in the truth. `o_tail` is the hourly cell maximum scaled so
        that the last 12 h add up to the largest 12-h total of one pixel (production telescopes the 1-km
        fields to the same total; the truth files keep the 12-h pixel maximum, not the 1-km hours)."""
        ts = [hnow - k * H1 for k in range(back_h - 1, -1, -1)]
        ts = [t for t in ts if str(t) in self.hour]
        if not self.has_obs(hnow):
            return None
        om, on, o12 = self.hours(ts)
        tail = om.copy()
        s = om[-12:].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(s > 0, o12[-1] / s, 1.0)
        tail[-12:] = om[-12:] * f
        return product.Obs(np.array(ts, "datetime64[h]"), om, on, o_tail=tail.astype(np.float32))

    def frame_truth(self, frames, hnow: np.datetime64, gates):
        """Per frame: o1 = largest hourly amount, o12 = largest 12-h pixel total ending inside the frame
        (the event scored until now), and for every gate G the forward-looking 12-h amount: production's
        own rule applied to the rain that really fell (new + clip(new / G, 0, 1) x already measured)."""
        o1, o12, og = [], [], {g: [] for g in gates}
        for t0, t1 in frames:
            hs = [t0 + (k + 1) * H1 for k in range(int((t1 - t0) / H1))]
            om, _, full = self.hours(hs)
            o1.append(om.max(axis=0)); o12.append(full.max(axis=0))
            if not gates:
                continue
            best = {g: np.zeros_like(full[0]) for g in gates}
            for k, h in enumerate(hs):
                lo = max(h - 12 * H1, hnow)
                n_new = int((h - lo) / H1)
                if n_new >= 12:
                    new = full[k]
                else:
                    back = [h - j * H1 for j in range(n_new)]
                    new = np.minimum(self.hours(back)[0].sum(axis=0), full[k]) if (back and all(str(b) in self.hour for b in back)) else np.zeros_like(full[k])
                for g in gates:
                    best[g] = np.maximum(best[g], new + np.clip(new / g, 0.0, 1.0) * (full[k] - new))
            for g in gates:
                og[g].append(best[g])
        return np.stack(o1), np.stack(o12), {g: np.stack(v) for g, v in og.items()}


# ---------------------------------------------------------------------------------------- archives

_RUN_INDEX: dict[str, list[datetime]] = {}


def run_index(key: str) -> list[datetime]:
    if key not in _RUN_INDEX:
        d = RUNS / RUNS_S3[key]
        _RUN_INDEX[key] = sorted(datetime.strptime(f.stem.split("_")[-1], "%Y%m%d%H") for f in d.glob("precipitation_*.npz")
                                 if ".tmp" not in f.name) if d.exists() else []
    return _RUN_INDEX[key]


def run_member(key: str, run: datetime, t_from: datetime, t_to: datetime) -> risk.Member | None:
    """One complete archived run as production's `ingest.load_run` would deliver it."""
    cfg = ingest.MODELS[key]
    z = np.load(RUNS / RUNS_S3[key] / f"precipitation_{run:%Y%m%d%H}.npz")
    valid = [datetime.fromisoformat(str(v)) for v in z["valid"]]
    sel = [k for k, v in enumerate(valid) if t_from <= v <= t_to]
    if len(sel) < 2:
        return None
    code = z["data"][sel]
    d = code.astype(np.float32) / 10.0
    d[code < 0] = np.nan
    vals = ingest.regridder(key, z["lat"], z["lon"])(d)
    t_end, p, step = ingest.to_hourly([valid[k] for k in sel], vals)
    if len(t_end) == 0:
        return None
    return risk.Member(f"{cfg['label']} · {run:%d/%m %H}Z", cfg["family"], key, run, t_end, p, step,
                       float(cfg.get("weight", 1.0)), meta={"nan_frac": float(np.isnan(p).mean())})


def lagged_runs(T: datetime, hours_ahead: int) -> list[risk.Member]:
    """Tier A: the runs `ingest_cached` would hold at T (newest `lags` complete runs of every model)."""
    out = []
    for key, cfg in ingest.MODELS.items():
        runs = [r for r in run_index(key) if r + timedelta(hours=DELAY_H[key]) <= T and T - r <= timedelta(hours=36)]
        for r in sorted(runs, reverse=True)[:cfg["lags"]]:
            m = run_member(key, r, T - timedelta(hours=14), T + timedelta(hours=hours_ahead))
            if m is not None:
                out.append(m)
    return out


_STITCH = None


def stitched_index():
    """Tier B: files of the Previous Runs archive -> [(t0, t1, path)]."""
    global _STITCH
    if _STITCH is None:
        _STITCH = []
        for f in sorted(glob.glob(str(HC / "cache" / "openmeteo" / "prev_*.npz"))):
            z = np.load(f, allow_pickle=True)
            t = z["time"]
            _STITCH.append((np.datetime64(str(t[0]), "h"), np.datetime64(str(t[-1]), "h"), f))
    return _STITCH


_STITCH_DATA = {}


def stitched_load(f: str) -> dict:
    if f not in _STITCH_DATA:
        if len(_STITCH_DATA) >= 4:
            _STITCH_DATA.pop(next(iter(_STITCH_DATA)))
        z = np.load(f, allow_pickle=True)
        j = np.floor((z["lat"] - grid.LAT0) / grid.D + 1e-6).astype(int)
        i = np.floor((z["lon"] - grid.LON0) / grid.D + 1e-6).astype(int)
        ok = (j >= 0) & (j < grid.NY) & (i >= 0) & (i < grid.NX)
        _STITCH_DATA[f] = dict(t=np.array([np.datetime64(x, "h") for x in z["time"]]), models=[str(m) for m in z["models"]],
                               leads=z["lead_days"].tolist(), j=j[ok], i=i[ok], precip=z["precip"][..., ok])
    return _STITCH_DATA[f]


def stitched_runs(T: datetime, hours_ahead: int, full_grid: bool):
    """Tier B pseudo-runs at issue time T. Returns (members, sample mask or None).
    For valid hour v the day-1 series (run issued <= v - 24 h) is used while that run was already complete at
    T, the day-2 series afterwards: nothing issued after T is ever used. `full_grid`: lattice values are
    copied to their 2x2 block of cells (the nowcast works on whole fields, without the sampling lattice)."""
    hT = np.datetime64(T, "h")
    lo, hi = hT + H1, hT + hours_ahead * H1
    # archive files that touch the window; an issue time may precede the first archived hour of a case, and a
    # window may run from one file into the next: every hour takes the first file that has it
    files = sorted([x for x in stitched_index() if x[0] <= hi and x[1] >= lo], key=lambda x: (not (x[0] <= lo <= x[1]), x[0]))
    if not files:
        return [], None
    src, v = [], max(lo, min(x[0] for x in files))
    while v <= hi:
        f = next((x[2] for x in files if x[0] <= v <= x[1]), None)
        if f is None:
            break                       # the series must be continuous
        src.append((v, f))
        v = v + H1
    if len(src) < 2:
        return [], None
    hours = np.array([s[0] for s in src])
    sample = np.zeros((grid.NY, grid.NX), bool)
    fields = {}
    for key in ("arome", "icon_eu", "arpege"):      # IFS 0.25: 3 days of one case; production uses the 9-km run instead
        model = ingest.ARCHIVE_IDS[key]
        delay = np.timedelta64(int(round(DELAY_H[key] * 60)), "m")
        g = np.full((len(hours), grid.NY, grid.NX), np.nan, np.float32)
        good = np.zeros(len(hours), bool)
        for k, (v, f) in enumerate(src):
            d = stitched_load(f)
            lead = 1 if (v - 24 * H1) <= (np.datetime64(T, "m") - delay) else 2
            if model in d["models"] and lead in d["leads"]:
                row = d["precip"][d["models"].index(model), d["leads"].index(lead), int(np.searchsorted(d["t"], v))]
                g[k, d["j"], d["i"]] = row
                good[k] = np.isfinite(row).mean() > 0.3
        n = int(np.argmin(good)) if not good.all() else len(hours)      # the series ends where the archive ends
        if n < 2:
            continue
        fields[key] = g[:n]
        sample |= np.isfinite(g[:n]).any(axis=0)
    members = []
    fam = PARAMS["families"]
    donor = fields.get("icon_eu", fields.get("arpege"))
    for key, g in fields.items():
        cfg = ingest.MODELS[key]
        if donor is not None and donor is not g:                 # AROME has no data south of 38 N: as ingest.fill_gaps
            n = min(len(g), len(donor))
            hole = np.isnan(g[:n]) & np.isfinite(donor[:n])
            g[:n][hole] = donor[:n][hole] * (fam["regional"]["s12h"] / fam[cfg["family"]]["s12h"])
        p = np.nan_to_num(g, nan=0.0)
        if full_grid:
            p = np.maximum.reduce([p, np.roll(p, 1, axis=1), np.roll(p, 1, axis=2), np.roll(np.roll(p, 1, axis=1), 1, axis=2)])
        members.append(risk.Member(f"{cfg['label']} · archivo día 1-2", cfg["family"], key, T - timedelta(hours=6),
                                   hours[:len(g)], p, 1, float(cfg.get("weight", 1.0)), meta={"stitched": True}))
    return members, sample


def ens3h_run(T: datetime) -> datetime | None:
    """The 00Z ENS run on disk, when it is the run production would use at T (complete, 8-14 h old)."""
    run = T.replace(hour=0, minute=0, second=0, microsecond=0)
    age = (T - run).total_seconds() / 3600.0
    d = ENS3H / f"ec_ifs_{run:%Y%m%d%H}"
    if not (DELAY_H["ens3h"] <= age <= 14.0) or not (d / "grid.npz").exists():
        return None
    return run if all((d / f"tp_{s:03d}.npy").exists() for s in range(6, 67, 3)) else None


def ens3h_members(run: datetime) -> list[risk.Member]:
    from riua.sources import extra_models as X
    ms = X.ecmwf_ens_members(run, ENS3H, 6, 66)          # every step is on disk: no download
    for m in ms:
        m.weight *= ENS3H_WEIGHT
    return ms


def ens12_file(T: datetime) -> Path | None:
    run = T.replace(hour=0, minute=0, second=0, microsecond=0)
    f = ENS12 / f"ifsens_tp_{run:%Y%m%d%H}.npz"
    return f if f.exists() and (T - run) >= timedelta(hours=DELAY_H["ens"]) else None


def long_extra(T: datetime, run: datetime) -> list[risk.Member]:
    """Days 2-7, besides the ENS (hindcast/long_data.py fetches them): the IFS 0.25 deg runs production adds as
    "global" members (its 2 newest complete runs: 00Z of the issue day and 12Z of the day before), and AIFS-ENS
    (not in production; kept in the block under its own model name so the scoring can test it)."""
    out = []
    for r in (run, run - timedelta(hours=12)):
        f = ENS12 / f"ifsoper_tp_{r:%Y%m%d%H}.npz"
        if f.exists() and T - r >= timedelta(hours=DELAY_H["ifs"]):
            for m in product.ens_npz_members(np.load(f, allow_pickle=True), r):
                m.family, m.model, m.name, m.native_step_h = "global", "ifs", f"IFS 0,25° · {r:%d/%m %H}Z", 6
                out.append(m)
    f = ENS12 / f"aifsens_tp_{run:%Y%m%d%H}.npz"
    if f.exists() and T - run >= timedelta(hours=7):
        for m in product.ens_npz_members(np.load(f, allow_pickle=True), run):
            m.model, m.name = "aifs_ens", m.name.replace("ENS", "AIFS-ENS")
            out.append(m)
    return out


def radar_rates(T: datetime) -> list:
    """The last scans production would have at T, as (ground-arrival time, rain rate) like `update_obs`."""
    from riua.radar import qpe
    out = []
    lag = timedelta(minutes=qpe.FALL_MIN)
    Tz = T.replace(tzinfo=timezone.utc)
    for day in {(T - timedelta(minutes=50)).strftime("%Y%m%d"), T.strftime("%Y%m%d")}:
        f = HC / "cache" / "radar" / f"{day}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ts = z["t"]
        for k in np.nonzero((ts > Tz.timestamp() - 45 * 60) & (ts <= Tz.timestamp() - lag.total_seconds()))[0]:
            code = z["dbz"][k]
            d = code.astype(np.float32) / 2.0 - 32.0
            d[code == 255] = np.nan
            out.append((datetime.fromtimestamp(int(ts[k]), timezone.utc) + lag, qpe.rain_rate(qpe.despeckle(d))))
    return sorted(out, key=lambda x: x[0])[-3:]


# ----------------------------------------------------------------------------- members of one cycle

def members_at(hz: str, T: datetime, truth: Truth):
    """The member list `run_cycle` would hand to `horizon_product`, from the archives. -> members, sample, info"""
    hnow = product.top_of_hour(T)
    h = np.datetime64(hnow, "h")
    obs = truth.obs(h)
    info = {"tier": None, "ens": None, "obs": obs is not None, "radar": None, "fc_first": None}

    def past(ms):
        # first forecast hour that every member has: earlier hours are measured (with obs) or unknown (without)
        first = [m.t_end[m.t_end > h].min() for m in ms if (m.t_end > h).any()]
        info["fc_first"] = str(max(first)) if first else None
        return [product.with_past(m, obs, hnow, bridge=None) for m in ms] if obs is not None else ms
    jj, ii = np.mgrid[0:grid.NY, 0:grid.NX]
    lattice = (jj % 2 == 0) & (ii % 2 == 0)
    if hz == "long":
        f = ens12_file(T)
        if f is None:
            return [], None, info
        run = T.replace(hour=0, minute=0, second=0, microsecond=0)
        ens = product.ens_npz_members(np.load(f, allow_pickle=True), run)
        info.update(tier="ens", ens=f"{run:%Y-%m-%dT%HZ}")
        return past(ens + long_extra(T, run)), None, info
    nwp = lagged_runs(T, 66 if hz == "mid" else 8)
    sample = lattice if hz == "mid" else None
    if nwp:
        info["tier"] = "A"
        ingest.fill_gaps(nwp)
    else:
        nwp, smp = stitched_runs(T, 52 if hz == "mid" else 7, full_grid=(hz == "now"))
        if not nwp:
            return [], None, info
        info["tier"] = "B"
        sample = smp if hz == "mid" else None
    if hz == "mid":
        extra = []
        run = ens3h_run(T)
        if run is not None:
            extra = ens3h_members(run)
            info["ens"] = f"{run:%Y-%m-%dT%HZ}"
        return past(nwp + extra), sample, info
    # now: radar members blended into the newest convection-permitting runs, the other runs beside them
    if obs is None:
        return [], None, info
    rates = radar_rates(T)
    from riua.radar import nowcast as NC
    if not getattr(NC, "_q8_two_workers", False):       # shared machine: 2 threads instead of production's 4
        orig = NC.steps_ensemble
        NC.steps_ensemble = lambda *a, **k: orig(*a, **{**k, "workers": 2})
        NC._q8_two_workers = True
    radar_m, nrep = product.nowcast_members(rates, obs, nwp, T.replace(tzinfo=timezone.utc))
    info["radar"] = {k: nrep.get(k) for k in ("method", "members", "wet_fraction")}
    info["scans"] = len(rates)
    blended = set(nrep.get("donors") or [])
    return past(radar_m + [m for m in nwp if m.name not in blended]), None, info


# -------------------------------------------------------------------------------------- one block

def block_path(hz: str, T: datetime) -> Path:
    return BLOCKS / hz / f"{T:%Y%m%d%H}.npz"


def build_block(hz: str, T: datetime, truth: Truth) -> str:
    out = block_path(hz, T)
    out.parent.mkdir(parents=True, exist_ok=True)
    hnow = product.top_of_hour(T)
    h = np.datetime64(hnow, "h")
    frames = product.frames_for(hz, T)
    have_obs = truth.has_obs(h)
    if not any(truth.has(t0, t1) and (have_obs or t0 >= h + 11 * H1) for t0, t1 in frames):
        return "no truth"
    members, sample, info = members_at(hz, T, truth)
    if not members or info["fc_first"] is None:
        return "no members"
    # A frame is scored when every hour of its 12-h windows is known to every member: measured hours up to the
    # issue time followed at once by the forecast, or forecast hours only (windows starting after the first one).
    first = np.datetime64(info["fc_first"], "h")
    whole = have_obs and first <= h + H1
    keep = [k for k, (t0, t1) in enumerate(frames) if truth.has(t0, t1) and (whole or t0 - 10 * H1 >= first)]
    if not keep:
        return "no frame with complete 12-h windows"
    r0 = float(PARAMS["radius_km"][hz])
    g0 = float(PARAMS.get("obs_gate_mm", 20.0))
    variants = [("", PARAMS)]
    variants += [(tag_r(r), {**PARAMS, "radius_km": {**PARAMS["radius_km"], hz: r}}) for r in RADII[hz]]
    gates = GATES[hz] if info["obs"] else ()
    variants += [(tag_g(g), {**PARAMS, "obs_gate_mm": g}) for g in gates]
    arrays, cells, base = {}, None, None
    for tag, par in variants:
        pred = risk.predictors(members, frames, par, hz, hnow, sample_mask=sample, keep_members=True)
        if pred.a1 is None:
            return "no predictors"
        if tag == "":
            kf = [k for k in keep if pred.valid[k]]
            if not kf:
                return "no valid frame"
            ok = (MASK if info["tier"] in ("A", "ens") else MASK_B) & np.isfinite(pred.a12[:, kf]).any(axis=0).all(axis=0)
            cells = np.nonzero(ok.ravel())[0]
            if cells.size == 0:
                return "no cells"
            base = pred
        flat = lambda a: a[:, kf].reshape(a.shape[0], len(kf), -1)[:, :, cells]      # noqa: E731
        a1 = flat(pred.a1)
        has1 = np.nonzero(np.isfinite(a1).any(axis=(1, 2)))[0]
        if not tag.startswith("@g"):                     # the gate does not touch the 1-h amounts
            arrays["a1" + tag] = a1[has1].astype(np.float16)
        arrays["a12" + tag] = flat(pred.a12).astype(np.float16)
        if info["obs"]:
            arrays["phi" + tag] = np.round(flat(pred.obs12) * 250.0).astype(np.uint8)
        if tag == "":
            arrays["has1"] = has1.astype(np.int16)
    fr = [frames[k] for k in kf]
    all_gates = tuple(gates) + ((g0,) if info["obs"] else ())
    o1, o12, og = truth.frame_truth(fr, h, all_gates)
    cut = lambda a: a.reshape(a.shape[0], -1)[:, cells].astype(np.float16)           # noqa: E731
    for r in (r0,) + tuple(RADII[hz]):
        tag = "" if r == r0 else tag_r(r)
        arrays["o1" + tag] = cut(grid.neighbourhood_max(o1, r))
        arrays["o12" + tag] = cut(grid.neighbourhood_max(o12, r))
    for g, v in og.items():
        arrays["o12f" + ("" if g == g0 else tag_g(g))] = cut(grid.neighbourhood_max(v, r0))
    names = [a["name"] for a in base.audit]
    by_name = {m.name: m for m in members}
    age = [max(0.0, (hnow - by_name[n].run).total_seconds() / 3600.0) for n in names]
    case = truth.case_of(fr[0][1]) or "?"
    meta = {"v": BLOCK_VERSION, "hz": hz, "issue": f"{T:%Y-%m-%dT%H:%MZ}", "case": case, "kind": truth.kind.get(case),
            "radius": r0, "gate": g0, "radii": list(RADII[hz]), "gates": list(gates), "params_version": PARAMS.get("version"),
            "names": names, "n_truth": sum(1 for t0, t1 in frames if truth.has(t0, t1)), **info}
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, meta=json.dumps(meta, ensure_ascii=False), cells=cells.astype(np.int32),
        t0=np.array([f[0] for f in fr]), t1=np.array([f[1] for f in fr]),
        lead_h=np.array([float((f[1] - h) / H1) for f in fr], np.float32),
        valid=(base.w[:, kf] > 0), family=np.array([a["family"] for a in base.audit]),
        model=np.array([a["model"] for a in base.audit]), age_h=np.array(age, np.float32),
        mw=np.array([by_name[n].weight for n in names], np.float32),
        radar=np.array([bool(by_name[n].meta.get("radar")) or by_name[n].family == "radar" for n in names]), **arrays)
    tmp.replace(out)
    return f"ok tier {info['tier']} M={len(names)} F={len(kf)} N={cells.size}" + (f" ens {info['ens']}" if info["ens"] else "")


# ------------------------------------------------------------------------------------ issue times

def issue_times(hz: str, truth: Truth, cases: list[str] | None = None) -> list[datetime]:
    """Every issue time whose frames can be verified, for the cases on disk."""
    out = set()
    lead_max = {"now": 6, "mid": 54, "long": 24 * 8}[hz]
    lead_min = {"now": 0, "mid": 4, "long": 48}[hz]
    for c, (a, b) in truth.cases.items():
        if cases and c not in cases:
            continue
        t = a - lead_max * H1
        t = np.datetime64(str(t)[:10] + "T00", "h")
        while t <= b - lead_min * H1:
            hh = int(str(t)[11:13])
            ok = hh in ISSUE_HOURS if hz != "long" else hh == 8
            if ok and (hz != "now" or truth.has_obs(t)):
                out.add(datetime.fromisoformat(str(t)))
            t = t + H1
    return sorted(out)


_TRUTH = None


def _job(args):
    global _TRUTH
    hz, T, force = args
    if _TRUTH is None:
        _TRUTH = Truth()
    if not force and not stale(hz, T, _TRUTH):
        return hz, T, "cached"
    try:
        msg = build_block(hz, T, _TRUTH)
    except Exception as e:  # noqa: BLE001  one bad issue time must not stop the batch
        msg = f"FAILED {type(e).__name__}: {e}"[:300]
    return hz, T, msg


def block_meta(f: Path) -> dict:
    with np.load(f, allow_pickle=True) as z:
        m = json.loads(str(z["meta"]))
        m["n_frames"] = int(len(z["t0"]))
    return m


def stale(hz: str, T: datetime, truth: Truth) -> bool:
    """True when the block is missing or was built from less than what is on disk now: an older block
    format, an ENS run that arrived later, more verifiable frames (a neighbouring truth file appeared)."""
    f = block_path(hz, T)
    if not f.exists():
        return True
    m = block_meta(f)
    if m.get("v") != BLOCK_VERSION:
        return True
    if hz == "mid" and m.get("ens") is None and ens3h_run(T) is not None:
        return True
    h = np.datetime64(product.top_of_hour(T), "h")
    n = sum(1 for t0, t1 in product.frames_for(hz, T) if truth.has(t0, t1))
    return n > m.get("n_truth", n) or (hz != "long" and truth.has_obs(h) and not m.get("obs"))


def build(hzs, jobs: int = 1, cases=None, force: bool = False) -> None:
    """Build every block that is missing or stale. Issue times with no archived forecast are tried again on
    every call (cheap), so the command is the same after q4 adds truth files, runs or ENS steps."""
    truth = Truth()
    print(f"truth: {len(truth.cases)} cases, {len(truth.hour)} hours", flush=True)
    todo = []
    for hz in hzs:
        ts = issue_times(hz, truth, cases)
        new = [(hz, T, force) for T in ts if force or stale(hz, T, truth)]
        print(f"{hz}: {len(ts)} issue times, {len(new)} missing or stale", flush=True)
        todo += new
    if not todo:
        return
    if "--spread" in sys.argv:
        # slow machine: one issue time of every case first (14Z, then 02Z, 20Z, 08Z), the severe cases first, so that
        # whatever is built when the job is stopped is a sample of all the cases and not the first cases only
        rank = {"major": 0, "moderate": 1, "null": 2, "ordinary": 3, "dry": 4}
        kind = lambda T: rank.get(truth.kind.get(truth.case_of(np.datetime64(T, "h")) or ""), 5)      # noqa: E731
        todo.sort(key=lambda x: ((14, 2, 20, 8).index(x[1].hour) if x[1].hour in (14, 2, 20, 8) else 9, kind(x[1]), x[1]))
    if "--reverse" in sys.argv:          # a second process working from the other end (each job re-checks the cache)
        todo.reverse()
    n_ok = 0
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for hz, T, msg in pool.imap_unordered(_job, todo, chunksize=1):
                n_ok += msg.startswith("ok")
                print(f"{hz} {T:%Y-%m-%d %H}Z {msg}", flush=True)
    else:
        for a in todo:
            hz, T, msg = _job(a)
            n_ok += msg.startswith("ok")
            print(f"{hz} {T:%Y-%m-%d %H}Z {msg}", flush=True)
    print(f"build done: {n_ok} blocks written of {len(todo)} tried", flush=True)


def status() -> None:
    truth = Truth()
    print(f"truth cases on disk: {len(truth.cases)}")
    for hz in P.HORIZONS:
        ts = issue_times(hz, truth)
        done = [T for T in ts if block_path(hz, T).exists()]
        size = sum(f.stat().st_size for f in (BLOCKS / hz).glob("*.npz")) / 1e6 if (BLOCKS / hz).exists() else 0.0
        print(f"{hz}: {len(ts)} issue times, {len(done)} blocks ({size:.0f} MB), {len(ts) - len(done)} without a block "
              f"(no archived forecast, or not built yet)")


# -------------------------------------------------------------------------------- hydrology check

def poyo_check(truth: Truth) -> dict | None:
    net, cps = static.hydro_net()
    if net is None or "2024-10-dana" not in truth.cases:
        return None
    ts = [t for t in (np.datetime64("2024-10-28T00", "h") + k * H1 for k in range(73)) if str(t) in truth.hour]
    t, p = np.array(ts), truth.hours(ts)[1]
    hp = PARAMS["hydro"]
    out = {"observed": "SAIH Rambla del Poyo (A-3): 2283 m3/s at 18:55 local (17:55 UTC) on 29 Oct 2024, then the sensor was lost",
           "rain_input": "radar-gauge analysis, cell means", "runs": []}
    flat = p.reshape(len(t), -1)
    q = H.route(H.net_rain(flat, hp["p0_mm"], hp["wet_memory_h"], phi=hp.get("phi_mmh"), s=hp.get("s_mm"), alpha=net.alpha,
                           p0b=hp.get("p0b_mm", 10.0), sb=hp.get("sb_mm", 100.0)), net, hp["clark_k"])
    row = {"p0_mm": hp["p0_mm"]}
    for pid in ("poyo-chiva", "poyo-ribarroja", "poyo-paiporta", "magro-algemesi"):
        if pid in net.ids:
            k = net.ids.index(pid)
            row[pid] = {"peak_m3s": round(float(q[:, k].max())), "peak_utc": str(t[int(q[:, k].argmax())])}
    out["runs"].append(row)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
    for k in ("--jobs", "--cases"):                      # also accept "--jobs 2"
        if k in sys.argv and sys.argv.index(k) + 1 < len(sys.argv) and opt.get(k) is True:
            opt[k] = sys.argv[sys.argv.index(k) + 1]
            args = [a for a in args if a != opt[k]]
    cmd = args[0] if args else "status"
    jobs = int(opt.get("--jobs", 1))
    cases = opt["--cases"].split(",") if isinstance(opt.get("--cases"), str) else None
    hzs = [a for a in args[1:] if a in P.HORIZONS] or list(P.HORIZONS)
    if cmd == "status":
        status()
    elif cmd in ("build", "all"):
        build(hzs, jobs, cases, force="--force" in opt)
    if cmd in ("fit", "all"):
        import score
        score.main(quick="--quick" in opt, hzs=[a for a in args[1:] if a in P.HORIZONS] or None)
