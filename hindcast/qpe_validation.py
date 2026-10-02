"""Cross-validation of the "rain already fallen" analysis (radar + gauges) against withheld gauges.

    python hindcast/qpe_validation.py capture [dir]     freeze the live inputs (radar scans, gauges)
    python hindcast/qpe_validation.py live [dir]        leave-gauges-out validation on a capture (default: newest)
    python hindcast/qpe_validation.py event <case_id> [day ...]   the same on civil-day totals of a hindcast case

`live` scores the production analysis (riua.radar.qpe.analyse), the chain it replaced and the alternatives
(mean-field bias, wradlib AdjustAdd / AdjustMultiply / AdjustMixed, kriging with external drift, gauges only):
  A. clock hours with rain: 25 % of the gauges with an hourly record (SAIH Jucar 5-min series, AEMET) are
     withheld in 4 folds x 2 repetitions and the 12-h and 1-h analyses are read at them;
  B. the moment of the capture, exactly as production sees it (snapshot gauges with their own stamps, hour
     under way), scored at the AVAMET network, which production never uses.
Run inside the Linux environment (pysteps, wradlib). Captures go to scratch/q1-qpe/live/<stamp>/
(gitignored: they hold AVAMET data, CC BY-NC-ND, private validation only).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "hindcast"))
warnings.filterwarnings("ignore")
UTC = timezone.utc
LIVE = ROOT / "scratch" / "q1-qpe" / "live"
T12, T1 = (60.0, 100.0, 180.0), (20.0, 40.0, 90.0)       # warning classes, mm in 12 h and in 1 h
OPEN = ("saih_chj", "saih_segura", "aemet")              # the networks production uses


# ------------------------------------------------------------------------------------ capture

def _code(dbz: np.ndarray) -> np.ndarray:
    c = np.clip(np.rint((np.nan_to_num(dbz, nan=0) + 32.0) * 2.0), 0, 254).astype(np.uint8)
    c[~np.isfinite(dbz)] = 255
    return c


def _decode(code: np.ndarray) -> np.ndarray:
    d = code.astype(np.float32) / 2.0 - 32.0
    d[code == 255] = np.nan
    return d


def capture(out: Path, hours: int = 27) -> None:
    """Radar scans of the last `hours` (OPERA, AEMET 4 h, the production blend 2 h), every gauge network
    (restricted ones included) and the hourly/5-min gauge series that are public (AEMET, SAIH Jucar)."""
    import re
    import requests
    from riua.sources import gauges as G, radar as R
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    gl = G.fetch_rain_gauges(include_restricted=True)
    (out / "gauges.json").write_text(json.dumps({"fetched": now.isoformat(), "gauges": gl, "errors": G.last_errors}), encoding="utf-8")
    print("gauges", len(gl), G.last_errors, flush=True)
    try:
        b = G._aemet_bundle()
        (out / "aemet_hours.json").write_text(json.dumps(b), encoding="utf-8")
        print("aemet hourly stations", len(b), flush=True)
    except Exception as e:
        print("aemet hourly failed", e, flush=True)
    for name, fn in (("blend", lambda: R.fetch_blended_frames(13)), ("aemet", lambda: R.fetch_aemet_frames(24))):
        try:
            fr = fn()
            np.savez_compressed(out / f"radar_{name}.npz", t=np.array([int(t.timestamp()) for t, _ in fr]),
                                dbz=np.stack([_code(d) for _, d in fr]))
            print(name, len(fr), "frames", fr[0][0], fr[-1][0], flush=True)
        except Exception as e:
            print(name, "failed", e, flush=True)
    t = (now - timedelta(hours=hours)).replace(minute=0, second=0, microsecond=0)       # OPERA, 10-min scans
    ts, ds = [], []
    while t <= now:
        try:
            ds.append(_code(R.fetch_opera(t))); ts.append(int(t.timestamp()))
        except Exception as e:
            print("opera", t, str(e)[:80], flush=True)
        t += timedelta(minutes=10)
    np.savez_compressed(out / "radar_opera.npz", t=np.array(ts), dbz=np.stack(ds))
    print("opera", len(ts), "frames", flush=True)
    # SAIH Jucar 5-min intensity series (mm/h), public chart route; local wall time in the URL, UTC in the answer
    ses = requests.Session(); ses.headers["User-Agent"] = G.UA
    series = {}
    loc = lambda d: (d + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    for g in [g for g in gl if g["source"] == "saih_chj"]:
        try:
            page = ses.get(f"{G.CHJ}/chart-lluvia/{g['station_ref']}", timeout=G.TIMEOUT).text
            var = str(json.loads(re.search(r"let varLluvia = (\[.*?\]);", page, re.S).group(1))[0]["idVariable"])
            a, b = requests.utils.quote(loc(now - timedelta(hours=hours + 3))), requests.utils.quote(loc(now + timedelta(hours=3)))
            pts = ses.get(f"{G.CHJ}/admin/variables/valor/{var}/{a}/{b}", timeout=G.TIMEOUT).json()
            series[g["id"]] = {"lat": g["lat"], "lon": g["lon"], "name": g["name"], "var": var,
                               "pts": [[p["fecha"][:19], p["valor"], p.get("estado")] for p in pts]}
        except Exception as e:
            print("chj series", g["id"], str(e)[:80], flush=True)
        time.sleep(0.25)
    (out / "chj_series.json").write_text(json.dumps(series), encoding="utf-8")
    print("chj series", len(series), flush=True)


def capture_aemet_png(out: Path, hours: int = 27) -> None:
    """AEMET composite pictures of the last 24 h (the numeric tarball only holds 4 h): they fill the hole
    of the OPERA composite on the Valencia coast for the older hours, as the production blend did live."""
    import io
    import requests
    from PIL import Image
    from riua.sources import radar as R
    base = "https://www.aemet.es/es/api-eltiempo/radar"
    ses = requests.Session(); ses.headers["User-Agent"] = R.USER_AGENT
    names = sorted(e["Nombre fichero"] for e in ses.get(f"{base}/timeline/compo/PB", timeout=30).json()[0]["Elementos"])
    lon0, lon1, lat0, lat1 = -16.08, 12.14, 27.22, 51.30          # /bounds-radar/compo/<file>, same for every file
    my = lambda lat: np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    t_min = datetime.now(UTC) - timedelta(hours=hours)
    ts, ds = [], []
    for n in names:
        t = datetime.strptime(n[4:16], "%Y%m%d%H%M").replace(tzinfo=UTC)
        if t < t_min:
            continue
        raw = out / "aemet_png" / n
        try:
            if not raw.exists():
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(ses.get(f"{base}/imagen-radar/compo/{n}", timeout=30).content)
                time.sleep(0.15)
            im = np.asarray(Image.open(io.BytesIO(raw.read_bytes())).convert("RGBA"))
        except Exception as e:
            print("aemet png", n, str(e)[:80], flush=True); continue
        H, W = im.shape[:2]
        col = np.floor((R.GRID_LON - 0.01 - lon0) / (lon1 - lon0) * W).astype(int)   # -0.01: best match with the GeoTIFF
        row = np.floor((my(lat1) - my(R.GRID_LAT)) / (my(lat1) - my(lat0)) * H).astype(int)
        px = im[row][:, col]
        d = np.full(px.shape[:2], R.NO_ECHO_DBZ, np.float32)
        for c, (lo, hi) in R._AEMET_CLASSES.items():
            d[(px[..., :3] == c).all(-1) & (px[..., 3] > 0)] = (lo + hi) / 2.0
        ts.append(int(t.timestamp())); ds.append(_code(d))
    np.savez_compressed(out / "radar_aemet_png.npz", t=np.array(ts), dbz=np.stack(ds))
    print("aemet png", len(ts), "frames", flush=True)


# ------------------------------------------------------------------------------- a capture

def _npz_frames(f: Path) -> dict:
    if not f.exists():
        return {}
    z = np.load(f)
    return {datetime.fromtimestamp(int(t), UTC): _decode(c) for t, c in zip(z["t"], z["dbz"])}


def load_frames(d: Path, fill: bool = True) -> list:
    """[(t, dBZ)] every 10 min: OPERA, its hole filled with the AEMET composite of the same label."""
    opera, tif, png = (_npz_frames(d / f"radar_{n}.npz") for n in ("opera", "aemet", "aemet_png"))
    out = []
    for t in sorted(opera):
        a = opera[t]
        f = tif.get(t, png.get(t)) if fill else None
        if f is not None:
            a = np.where(np.isnan(a), f, a)
        out.append((t, a))
    return out


def rates_of(frames: list, lag_min: float) -> list:
    from riua.radar import qpe
    return [(t + timedelta(minutes=lag_min), qpe.rain_rate(qpe.despeckle(x))) for t, x in frames]


def hourly_radar(d: Path, tag: str, frames: list | None = None, lag_min: float = 0.0, **acc_kw):
    """(t_end (K,), acc (K, 340, 320) raw mm with NaN = no coverage, coverage (K,)); cached as radar_hours_<tag>.npz.
    tag "nolag": scan labels as they come (the chain that was in production); "lag": ground-arrival times
    (lag_min = qpe.FALL_MIN, the production chain); "lagadv": the same plus drift_min (tested, rejected)."""
    from riua.radar import qpe
    f = d / f"radar_hours_{tag}.npz"
    if f.exists():
        z = np.load(f)
        return z["t_end"], z["acc"].astype(np.float32), z["cov"]
    rates = rates_of(frames if frames is not None else load_frames(d), lag_min)
    t0 = rates[0][0].replace(minute=0) + timedelta(hours=1)
    t_end, acc, cov = [], [], []
    while t0 + timedelta(hours=1) <= rates[-1][0]:
        a, b = t0, t0 + timedelta(hours=1)
        sub = [x for x in rates if a - timedelta(minutes=10) <= x[0] <= b]
        if len(sub) >= 2:
            x, c = qpe.accumulate(sub, a, b, **acc_kw)
            t_end.append(np.datetime64(b.replace(tzinfo=None), "h")); acc.append(x); cov.append(c)
        t0 = b
    t_end, acc, cov = np.array(t_end), np.stack(acc), np.array(cov)
    np.savez_compressed(f, t_end=t_end, acc=acc.astype(np.float16), cov=cov)
    return t_end, acc, cov


def gauge_hours(d: Path) -> list[dict]:
    """Gauges with a complete hourly record: [{id, source, lat, lon, h: {hour_end (datetime64[h]): mm}}].
    SAIH Jucar 5-min intensities (mm/h, stamp = end of the 5 minutes) and the AEMET hourly bundle."""
    out = []
    f = d / "chj_series.json"
    for sid, s in (json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}).items():
        h, n = {}, {}
        for stamp, v, _ in s["pts"]:
            if v is None or v < 0 or v > 300:            # 300 mm/h in 5 min = a telemetry backlog dump
                continue
            t = np.datetime64(stamp, "s")
            k = (t - np.timedelta64(1, "s")).astype("datetime64[h]") + np.timedelta64(1, "h")
            h[k] = h.get(k, 0.0) + v / 12.0; n[k] = n.get(k, 0) + 1
        out.append({"id": sid, "source": "saih_chj", "lat": s["lat"], "lon": s["lon"], "name": s["name"],
                    "h": {k: v for k, v in h.items() if n[k] >= 11}})
    f = d / "aemet_hours.json"
    for sid, s in (json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}).items():
        out.append({"id": sid, "source": "aemet", "lat": s["lat"], "lon": s["lon"], "name": f"AEMET {sid}",
                    "h": {np.datetime64(k, "h"): float(v) for k, v in s["hours"].items()}})
    return out


# ------------------------------------------------------------------------- merging methods
# Alternatives to the production merge. Every method: f(radar (340, 320) mm, NaN = no coverage; g_lat, g_lon,
# g_mm) -> merged (340, 320) mm.

KM_LON, KM_LAT = 111.2 * float(np.cos(np.deg2rad(39.3))), 111.2
LON0, LAT0, DD, NY, NX = -2.4, 37.6, 0.01, 340, 320


def px_of(lat, lon):
    j = np.floor((np.asarray(lat, float) - LAT0) / DD).astype(int)
    i = np.floor((np.asarray(lon, float) - LON0) / DD).astype(int)
    return j, i, (j >= 0) & (j < NY) & (i >= 0) & (i < NX)


def at_gauges(field: np.ndarray, lat, lon, nb: int = 3) -> np.ndarray:
    from riua.radar import qpe
    return qpe.at_gauges(field, lat, lon, nb)


def cell_at(field5: np.ndarray, lat, lon) -> np.ndarray:
    """Value of a (68, 64) analysis-grid field at the cell of each gauge."""
    j, i, ok = px_of(lat, lon)
    out = np.full(len(j), np.nan)
    out[ok] = field5[j[ok] // 5, i[ok] // 5]
    return out


def block(field, how="mean"):
    b = np.nan_to_num(field, nan=0.0).reshape(*field.shape[:-2], NY // 5, 5, NX // 5, 5)
    return b.max(axis=(-3, -1)) if how == "max" else b.mean(axis=(-3, -1))


def m_raw(radar, lat, lon, mm):
    return np.nan_to_num(radar, nan=0.0)


def m_mfb(radar, lat, lon, mm):
    r = at_gauges(radar, lat, lon)
    both = np.isfinite(r) & (np.asarray(mm) >= 1.0) & (r >= 1.0)
    f = float(np.clip(np.asarray(mm)[both].sum() / r[both].sum(), 0.2, 5.0)) if both.sum() >= 3 else 1.0
    return np.nan_to_num(radar, nan=0.0) * f


def m_gauges(radar, lat, lon, mm):
    from riua.radar import qpe
    return qpe.gauges_only(lat, lon, mm)


def m_wradlib(kind: str, nnear_raws: int = 9):
    def f(radar, lat, lon, mm):
        import wradlib as wrl
        lon2 = (LON0 + (np.arange(NX) + 0.5) * DD) * KM_LON
        lat2 = (LAT0 + (np.arange(NY) + 0.5) * DD) * KM_LAT
        x, y = np.meshgrid(lon2, lat2)
        raw = np.nan_to_num(radar, nan=0.0).ravel().astype(np.float64)
        adj = getattr(wrl.adjust, kind)(np.column_stack([np.asarray(lon) * KM_LON, np.asarray(lat) * KM_LAT]),
                                        np.column_stack([x.ravel(), y.ravel()]), nnear_raws=nnear_raws, mingages=5,
                                        minval=0.5, nnearest=8, p=2.0)
        res = np.asarray(adj(np.asarray(mm, float), raw))
        return np.where(np.isfinite(res), np.maximum(res, 0.0), raw).reshape(radar.shape)
    return f


def m_ked(radar, lat, lon, mm, rng=40.0, nug=0.1):
    """Kriging with external drift in square-root space (dual form): the gauges interpolated with the radar as trend."""
    r = at_gauges(radar, lat, lon)
    ok = np.isfinite(r)
    xy = np.column_stack([np.asarray(lon)[ok] * KM_LON, np.asarray(lat)[ok] * KM_LAT])
    z, dr = np.sqrt(np.asarray(mm, float)[ok]), np.sqrt(r[ok])
    base = np.nan_to_num(radar, nan=0.0)
    if len(z) < 8 or dr.std() < 1e-3:
        return base
    n = len(z)
    gam = lambda h: nug + (1.0 - nug) * (1.0 - np.exp(-3.0 * h / rng))
    A = np.zeros((n + 2, n + 2))
    A[:n, :n] = gam(np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]))
    A[np.arange(n), np.arange(n)] = 0.0
    A[:n, n] = A[n, :n] = 1.0
    A[:n, n + 1] = A[n + 1, :n] = dr
    c = np.linalg.lstsq(A, np.concatenate([z, [0.0, 0.0]]), rcond=None)[0]
    lon2 = (LON0 + (np.arange(NX) + 0.5) * DD) * KM_LON
    lat2 = (LAT0 + (np.arange(NY) + 0.5) * DD) * KM_LAT
    x, y = np.meshgrid(lon2, lat2)
    t, drt = np.column_stack([x.ravel(), y.ravel()]), np.sqrt(base).ravel()
    out = np.empty(len(t))
    for a in range(0, len(t), 20000):
        g = gam(np.hypot(t[a:a + 20000, None, 0] - xy[None, :, 0], t[a:a + 20000, None, 1] - xy[None, :, 1]))
        out[a:a + 20000] = g @ c[:n] + c[n] + c[n + 1] * drt[a:a + 20000]
    return np.maximum(out, 0.0).reshape(radar.shape) ** 2


def m_cm(register=True, qc=True, ceil=None):
    """The production merge of a single accumulation (qpe.merge_gauges), with switches."""
    def f(radar, lat, lon, mm):
        from riua.radar import qpe
        out, _ = qpe.merge_gauges(radar, lat, lon, mm, do_register=register, qc=qc)
        if ceil is not None:
            c, _ = qpe.ceiling(lat, lon, mm, qpe.at_gauges(radar, lat, lon), **ceil)
            out = np.minimum(out, c)
        return np.nan_to_num(out, nan=0.0)
    return f


def previous_production(acc: np.ndarray, lat, lon, mm, n: int = 12):
    """The chain that was in production until 2026-10-01 (product.py at commit 89b4657), on the hourly raw fields
    acc (K, 340, 320): wradlib AdjustMixed on the 5-km cell means of the 12-h total, the factor applied to every
    hour, then the ceiling 1.5 x largest gauge within ~20 km + 10 mm. Returns hourly (o_mean, o_max) (K, 68, 64).
    Fields after the first `n` (the hour under way) take the factor and the ceiling but do not define them."""
    import wradlib as wrl
    from scipy import ndimage
    from riua.core import grid
    o_max = np.minimum(block(acc, "max"), 200.0).astype(np.float32)
    o_mean = np.minimum(np.stack([np.nan_to_num(np.nanmean(a.reshape(NY // 5, 5, NX // 5, 5), axis=(1, 3)), nan=0.0) for a in acc]), 200.0).astype(np.float32)
    raw12 = o_mean[:n].sum(axis=0)
    mm = np.asarray(mm, float)
    if len(mm) >= 8 and (mm.max() >= 1.0 or raw12.max() >= 1.0):
        lat2, lon2 = grid.mesh()
        adj = wrl.adjust.AdjustMixed(np.column_stack([np.asarray(lon) * grid.KM_PER_DEG_LON, np.asarray(lat) * grid.KM_PER_DEG_LAT]),
                                     np.column_stack([lon2.ravel() * grid.KM_PER_DEG_LON, lat2.ravel() * grid.KM_PER_DEG_LAT]),
                                     nnear_raws=5, mingages=5, minval=0.5, nnearest=8, p=2.0)
        merged = np.asarray(adj(mm, raw12.ravel().astype(np.float64))).reshape(raw12.shape)
        merged = np.where(np.isfinite(merged), np.maximum(merged, 0.0), raw12)
        fac = np.where(raw12 >= 0.5, np.clip(merged / np.maximum(raw12, 1e-6), 0.2, 6.0), 1.0).astype(np.float32)
        o_mean *= fac; o_max *= fac
    g = np.zeros((grid.NY, grid.NX), np.float32); has = np.zeros_like(g)
    for a, b, c in zip(lat, lon, mm):
        cell = grid.cell_of(a, b)
        if cell:
            g[cell] = max(g[cell], c); has[cell] = 1.0
    ceil = np.full(g.shape, 2.0 * float(g.max()) + 10.0, np.float32)
    for rc in (10, 4):
        size = 2 * rc + 1
        ceil = np.where(ndimage.maximum_filter(has, size=size) > 0, 1.5 * ndimage.maximum_filter(g, size=size) + 10.0, ceil)
    for arr in (o_mean, o_max):
        tot = arr[:n].sum(axis=0)
        arr *= np.where(tot > ceil, ceil / np.maximum(tot, 1e-6), 1.0).astype(np.float32)
    return o_mean, o_max


# ------------------------------------------------------------------------------ scores

def scores(est, obs, thr) -> dict:
    est, obs = np.asarray(est, float), np.asarray(obs, float)
    ok = np.isfinite(est) & np.isfinite(obs)
    est, obs = est[ok], obs[ok]
    wet = (obs >= 1.0) | (est >= 1.0)
    ce, co = np.searchsorted(thr, est, side="right"), np.searchsorted(thr, obs, side="right")
    return {"n": int(ok.sum()), "bias": float((est - obs).mean()), "mae": float(np.abs(est - obs).mean()),
            "mae_wet": float(np.abs(est - obs)[wet].mean()) if wet.any() else np.nan,
            "rmse": float(np.sqrt(((est - obs) ** 2).mean())),
            "r": float(np.corrcoef(est, obs)[0, 1]) if est.std() > 0 and obs.std() > 0 else np.nan,
            "wrong": float((ce != co).mean()), "under": int((ce < co).sum()), "over": int((ce > co).sum()),
            "n_hi": int((co >= 1).sum()), "hit_hi": int((ce[co >= 1] == co[co >= 1]).sum())}


class Table:
    """Collects (estimate, gauge) pairs per row name and prints one score line per row."""

    def __init__(self):
        self.rows = {}

    def add(self, name, est, obs):
        r = self.rows.setdefault(name, [[], []])
        r[0] += list(np.asarray(est, float)); r[1] += list(np.asarray(obs, float))

    def show(self, title, thr, key=""):
        print(f"\n{title}\n{'':46s}     n   bias    MAE MAE_wet   RMSE      r  wrong class  under  over  hit>={thr[0]:.0f}")
        for name, (e, o) in self.rows.items():
            if key in name:
                s = scores(e, o, thr)
                print(f"{name.replace(key, ''):46s} {s['n']:5d} {s['bias']:+6.2f} {s['mae']:6.2f} {s['mae_wet']:7.2f} {s['rmse']:6.2f} "
                      f"{s['r']:6.3f} {100 * s['wrong']:10.1f}% {s['under']:6d} {s['over']:5d}  {s['hit_hi']}/{s['n_hi']}", flush=True)


# ----------------------------------------------------------------------------- live validation

def _window(te, acc, T, n=12):
    k = np.nonzero((te > T - np.timedelta64(n, "h")) & (te <= T))[0]
    return acc[k], [te[i] for i in k]


def _tails(adj: np.ndarray) -> np.ndarray:
    """Largest total of one pixel of each cell over the whole of `adj` (S, 340, 320)."""
    return block(adj.sum(axis=0), "max")


def live(d: Path, folds: int = 4, reps: int = 2, seed: int = 7) -> None:
    from riua.radar import qpe
    print(f"capture {d.name}")
    te0, acc0, _ = hourly_radar(d, "nolag")
    te1, acc1, _ = hourly_radar(d, "lag", lag_min=qpe.FALL_MIN)
    gh = gauge_hours(d)
    as_dt = lambda t: t.astype("datetime64[s]").astype(datetime)
    # ---- A. clock hours
    A12, A1, alt = Table(), Table(), Table()
    rs = np.random.RandomState(seed)
    alts = {"raw radar": m_raw, "mean-field bias": m_mfb, "wradlib AdjustAdd": m_wradlib("AdjustAdd"),
            "wradlib AdjustMultiply": m_wradlib("AdjustMultiply"), "wradlib AdjustMixed": m_wradlib("AdjustMixed"),
            "kriging with external drift (sqrt)": m_ked, "gauges only (kriging, sqrt + variance)": m_gauges,
            "cond. merging sqrt, no QC, no registration": m_cm(False, False),
            "cond. merging + QC": m_cm(False, True), "cond. merging + QC + registration (qpe.merge_gauges)": m_cm(True, True),
            "  + ceiling 1.5 x G(20 km) + 10": m_cm(True, True, dict(factor=1.5, plus=10.0, min_gauges=1)),
            "  + ceiling 3 x G(20 km) + 20, >= 3 gauges (production)": m_cm(True, True, {})}
    for T in te1[12:]:
        s0, _ = _window(te0, acc0, T)
        s1, hrs = _window(te1, acc1, T)
        if len(s0) < 12 or len(s1) < 12:
            continue
        G = [g for g in gh if all(h in g["h"] for h in hrs) and px_of(g["lat"], g["lon"])[2]]
        lat, lon = (np.array([g[k] for g in G]) for k in ("lat", "lon"))
        g12, g1 = np.array([sum(g["h"][h] for h in hrs) for g in G]), np.array([g["h"][hrs[-1]] for g in G])
        if (g12 >= 5.0).sum() < 10:
            continue
        r12 = np.where(np.isfinite(s1).any(0), np.nansum(s1, 0), np.nan)
        bounds = [(as_dt(h) - timedelta(hours=1), as_dt(h)) for h in hrs]
        stamp = as_dt(T).strftime("%Y-%m-%dT%H:%M:%SZ")
        for rep in range(reps):
            perm = rs.permutation(len(G))
            for f in range(folds):
                te_ix = perm[f::folds]; tr = np.setdiff1d(perm, te_ix)
                la, lo, o12, o1 = lat[te_ix], lon[te_ix], g12[te_ix], g1[te_ix]
                om, ox = previous_production(s0, lat[tr], lon[tr], g12[tr])
                A12.add("previous production | mean", cell_at(om.sum(0), la, lo), o12)
                A12.add("previous production | level", cell_at(ox.sum(0), la, lo), o12)
                A1.add("previous production | mean", cell_at(om[-1], la, lo), o1)
                A1.add("previous production | max", cell_at(ox[-1], la, lo), o1)
                A1.add("raw radar | mean", cell_at(block(s1[-1]), la, lo), o1)
                A1.add("raw radar | max", cell_at(block(s1[-1], "max"), la, lo), o1)
                train = [{"lat": lat[k], "lon": lon[k], "t_utc": stamp, "p_12h": g12[k], "p_1h": g1[k]} for k in tr]
                name = "new production (qpe.analyse)"
                adj, info = qpe.analyse(s1, bounds, train)
                A1.add(name + " | mean", cell_at(block(adj[-1]), la, lo), o1)
                A1.add(name + " | max", cell_at(block(adj[-1], "max"), la, lo), o1)
                A12.add(name + " | mean", cell_at(block(adj.sum(0)), la, lo), o12)
                A12.add(name + " | level", cell_at(_tails(adj), la, lo), o12)
                A12.add(name + " | px", at_gauges(adj.sum(0), la, lo), o12)
                for name, fn in alts.items():
                    mg = fn(r12, lat[tr], lon[tr], g12[tr])
                    alt.add(name + " | px", at_gauges(mg, la, lo), o12)
                    alt.add(name + " | level", cell_at(block(mg, "max"), la, lo), o12)
        print(f"  {T}: {len(G)} gauges with a 12-h record, max {g12.max():.1f} mm, {int((g12 >= 60).sum())} >= 60 mm; last analyse: {info}", flush=True)
    A12.show("A. 12 h, cell MEAN at the withheld gauge (the amount the map shows)", T12, " | mean")
    A12.show("A. 12 h, amount the LEVEL uses at the withheld gauge (previous: sum of hourly cell maxima; new: largest total of one pixel = o_tail)."
             "\n   'under' = the gauge is in a higher class than the analysis of its cell: always an error. 'over' need not be one.", T12, " | level")
    A12.show("A. 12 h, 1-km value (3 x 3 px) at the withheld gauge", T12, " | px")
    A1.show("A. last hour, cell MEAN at the withheld gauge", T1, " | mean")
    A1.show("A. last hour, cell MAXIMUM at the withheld gauge", T1, " | max")
    alt.show("A. alternatives on the 12-h total (same folds), 1-km value at the withheld gauge", T12, " | px")
    alt.show("A. alternatives on the 12-h total, cell maximum at the withheld gauge", T12, " | level")
    # ---- B. the moment of the capture, as production runs, against AVAMET
    snap = json.loads((d / "gauges.json").read_text(encoding="utf-8"))
    now = datetime.fromisoformat(snap["fetched"]).replace(tzinfo=None)
    hnow = now.replace(minute=0, second=0, microsecond=0)
    gl = [g for g in snap["gauges"] if g["source"] in OPEN]
    av = [g for g in snap["gauges"] if g["source"] == "avamet" and g.get("p_12h") is not None and g.get("lat") is not None]
    la, lo = (np.array([g[k] for g in av]) for k in ("lat", "lon"))
    a12, a1 = np.array([g["p_12h"] for g in av]), np.array([np.nan if g.get("p_1h") is None else g["p_1h"] for g in av])
    B12, B1 = Table(), Table()
    frames = [f for f in load_frames(d) if f[0] >= (hnow - timedelta(minutes=30)).replace(tzinfo=UTC)]
    for name, te, acc, lag in (("previous production", te0, acc0, 0.0), ("new production (qpe.analyse)", te1, acc1, qpe.FALL_MIN)):
        s, hrs = _window(te, acc, np.datetime64(hnow, "h"))
        rates = rates_of(frames, lag)
        sub = [x for x in rates if x[0] >= hnow.replace(tzinfo=UTC) - timedelta(minutes=10)]
        part, part_end = qpe.accumulate(sub, hnow.replace(tzinfo=UTC), sub[-1][0])[0], sub[-1][0].replace(tzinfo=None)
        f = (part_end - hnow).total_seconds() / 3600.0              # the AVAMET periods end 5-20 min after the hour
        segs = np.concatenate([s, part[None]])
        if lag == 0.0:
            use = [g for g in gl if g.get("p_12h") is not None and g.get("lat") is not None]
            om, ox = previous_production(segs, [g["lat"] for g in use], [g["lon"] for g in use], [g["p_12h"] for g in use])
            mean12, lev12, mean1, max1 = om.sum(0), ox.sum(0), (1 - f) * om[-2] + om[-1], (1 - f) * ox[-2] + ox[-1]
        else:
            bounds = [(as_dt(h) - timedelta(hours=1), as_dt(h)) for h in hrs] + [(hnow, part_end)]
            adj, info = qpe.analyse(segs, bounds, gl)
            print(f"\nB. capture at {now:%Y-%m-%d %H:%M} UTC, hour under way until {part_end:%H:%M}; analyse: {info}")
            h1 = (1 - f) * adj[-2] + adj[-1]
            mean12, lev12, mean1, max1 = block(adj.sum(0)), _tails(adj), block(h1), block(h1, "max")
        B12.add(name + " | mean", cell_at(mean12, la, lo), a12)
        B12.add(name + " | level", cell_at(lev12, la, lo), a12)
        B1.add(name + " | mean", cell_at(mean1, la, lo), a1)
        B1.add(name + " | max", cell_at(max1, la, lo), a1)
    B12.show(f"B. 12 h + the hour under way, cell MEAN at {len(av)} AVAMET gauges (never used by production)", T12, " | mean")
    B12.show("B. 12 h + the hour under way, amount the LEVEL uses at the AVAMET gauges", T12, " | level")
    B1.show("B. last 60 min, cell MEAN at the AVAMET gauges", T1, " | mean")
    B1.show("B. last 60 min, cell MAXIMUM at the AVAMET gauges", T1, " | max")


# ---------------------------------------------------------------------------- hindcast event

def event(case_id: str, days: list[str] | None = None, folds: int = 4) -> None:
    """Leave-gauges-out validation of the merge on the civil-day totals of a hindcast case (the inputs of
    build_truth.py: archived OPERA scans, hindcast/obs/<day>.csv). The classes are the 12-h ones, applied to 24 h.
    The hourly radar comes from build_truth.utc_day_acc (computed and cached there when missing)."""
    import build_truth as BT
    from riua.radar import qpe
    case = next(c for c in json.loads((ROOT / "hindcast" / "cases.json").read_text(encoding="utf-8"))["cases"] if c["id"] == case_id)
    tab = Table()
    rs = np.random.RandomState(7)
    alts = {"raw radar": m_raw, "mean-field bias": m_mfb, "wradlib AdjustMixed (previous build_truth merge)": m_wradlib("AdjustMixed"),
            "  + its ceiling 1.5 x G(15 km) + 10": None, "kriging with external drift (sqrt)": m_ked,
            "gauges only (kriging, sqrt + variance)": m_gauges, "cond. merging sqrt, no QC, no registration": m_cm(False, False),
            "cond. merging + QC + registration (qpe.merge_gauges)": m_cm(True, True),
            "  + ceiling 3 x G(20 km) + 20, >= 3 gauges": m_cm(True, True, {})}
    for day in days or case["days"]:
        lat, lon, mm = BT.load_gauges(day)
        ok = px_of(lat, lon)[2] if len(mm) else np.zeros(0, bool)
        lat, lon, mm = lat[ok], lon[ok], mm[ok]
        if len(mm) < 100 or mm.max() < 20.0:
            print(f"  {day}: {len(mm)} gauges, max {mm.max() if len(mm) else 0:.0f} mm: skipped"); continue
        u0 = datetime.fromisoformat(day).replace(tzinfo=BT.MAD).astimezone(UTC)
        hours = [u0 + timedelta(hours=h) for h in range(24)]
        acc = []
        for dd in sorted({h.replace(hour=0) for h in hours}):
            need = [h.hour for h in hours if h.replace(hour=0) == dd]
            a, _ = BT.utc_day_acc(dd, need)
            acc.append(a[need])
        acc = np.concatenate(acc)
        raw = np.where(np.isfinite(acc).any(0), np.nansum(acc, 0), np.nan)
        dj, di, gain = qpe.register(raw, lat, lon, mm)
        print(f"  {day}: {len(mm)} gauges, max {mm.max():.0f} mm, {int((mm >= 60).sum())} >= 60; raw radar max {np.nanmax(raw):.0f} mm, "
              f"radar/gauge {np.nansum(at_gauges(raw, lat, lon)) / mm.sum():.2f}; registration with all gauges {dj, di} (+{gain:.3f})", flush=True)
        perm = rs.permutation(len(mm))
        for f in range(folds):
            te_ix = perm[f::folds]; tr = np.setdiff1d(perm, te_ix)
            for name, fn in alts.items():
                if fn is None:
                    mg = np.minimum(m_wradlib("AdjustMixed")(raw, lat[tr], lon[tr], mm[tr]), BT.gauge_ceiling(lat[tr], lon[tr], mm[tr]))
                else:
                    mg = fn(raw, lat[tr], lon[tr], mm[tr])
                tab.add(name + " | px", at_gauges(mg, lat[te_ix], lon[te_ix]), mm[te_ix])
                tab.add(name + " | level", cell_at(block(mg, "max"), lat[te_ix], lon[te_ix]), mm[te_ix])
    tab.show(f"EVENT {case_id}: civil-day totals, 1-km value (3 x 3 px) at the withheld gauge", T12, " | px")
    tab.show(f"EVENT {case_id}: civil-day totals, cell maximum at the withheld gauge ('under' is always an error)", T12, " | level")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "live"
    if cmd == "capture":
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else LIVE / datetime.now(UTC).strftime("%Y%m%dT%H%M")
        capture(out)
        capture_aemet_png(out)
    elif cmd == "event":
        event(sys.argv[2], sys.argv[3:])
    else:
        live(Path(sys.argv[2]) if len(sys.argv) > 2 else sorted(p for p in LIVE.iterdir() if p.is_dir())[-1])
