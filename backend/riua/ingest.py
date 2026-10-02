"""Turn model runs into Members on the Riuà grid (live path, Open-Meteo S3 mirror).

One member = one run of one model, hourly accumulations, no NaN. Runs of the same model
initialised at different times form a time-lagged ensemble (Hoffman and Kalnay 1983):
cheap, and it samples the initial-condition uncertainty the single run hides.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from .core import grid
from .core.risk import Member
from .sources import openmeteo_s3 as om

log = logging.getLogger("riua.ingest")

# label, S3 id, family, how many lagged runs, native grid is fine enough to be "sampled"
MODELS = {
    # AROME 1.3 km and 2.5 km of the same run are almost the same forecast: together they weigh 1.2, not 2
    "arome_hd": dict(s3="meteofrance_arome_france_hd", family="cp", label="AROME-HD 1,3 km", lags=3, weight=0.7),
    "arome": dict(s3="meteofrance_arome_france0025", family="cp", label="AROME 2,5 km", lags=2, weight=0.5),
    "icon_eu": dict(s3="dwd_icon_eu", family="regional", label="ICON-EU 6,5 km", lags=2),
    "arpege": dict(s3="meteofrance_arpege_europe", family="regional", label="ARPEGE 0,1°", lags=2),
    "ifs": dict(s3="ecmwf_ifs025", family="global", label="IFS 0,25°", lags=2),
}
# ids used by the forecast archive (Open-Meteo Previous Runs API) for the same models
ARCHIVE_IDS = {"arome_hd": "meteofrance_arome_france_hd", "arome": "meteofrance_arome_france",
               "icon_eu": "icon_eu", "arpege": "meteofrance_arpege_europe", "ifs": "ecmwf_ifs025"}

_REGRID: dict[tuple, grid.Regridder] = {}


def naive(t: datetime) -> datetime:
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def h64(t: datetime) -> np.datetime64:
    return np.datetime64(naive(t).replace(minute=0, second=0, microsecond=0), "h")


def regridder(key: str, lats: np.ndarray, lons: np.ndarray) -> grid.Regridder:
    k = (key, lats.shape, lons.shape, float(np.asarray(lats).ravel()[0]), float(np.asarray(lons).ravel()[0]))
    if k not in _REGRID:
        _REGRID[k] = grid.Regridder(lats, lons)
    return _REGRID[k]


def to_hourly(times: list[datetime], vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulations over the preceding step (1, 3 or 6 h, possibly changing with lead time)
    -> hourly series. Longer steps are spread evenly (and flagged through the returned max step).
    vals: (T, NY, NX). Returns t_end (H,) datetime64[h], p (H, NY, NX), max native step.
    """
    tt = np.array([h64(t) for t in times])
    order = np.argsort(tt)
    tt, vals = tt[order], vals[order]
    out_t, out_p, max_step = [], [], 1
    for k in range(len(tt)):
        if not np.isfinite(vals[k]).any():
            continue                      # analysis time: no accumulation yet
        step = int((tt[k] - tt[k - 1]) / np.timedelta64(1, "h")) if k > 0 else 1
        step = max(1, min(step, 6))
        max_step = max(max_step, step)
        for s in range(step):
            out_t.append(tt[k] - np.timedelta64(step - 1 - s, "h"))
            out_p.append(vals[k] / step)
    if not out_t:
        return np.array([], "datetime64[h]"), np.zeros((0, grid.NY, grid.NX), np.float32), 1
    return np.array(out_t), np.stack(out_p).astype(np.float32), max_step


def load_run(key: str, run: datetime, t_from: datetime, t_to: datetime) -> Member | None:
    cfg = MODELS[key]
    try:
        vt = [t for t in om.valid_times(cfg["s3"], run) if t_from <= t <= t_to]
        if len(vt) < 2:
            return None
        times, lats, lons, out = om.read_series(cfg["s3"], run, "precipitation", times=vt, max_workers=6)
    except Exception as e:  # a missing run must not stop the cycle
        log.warning("%s %s: %s: %s", key, run, type(e).__name__, e)
        return None
    vals = regridder(key, lats, lons)(out["precipitation"])
    t_end, p, step = to_hourly(times, vals)
    if len(t_end) == 0:
        return None
    return Member(name=f"{cfg['label']} · {naive(run):%d/%m %H}Z", family=cfg["family"], model=key,
                  run=naive(run), t_end=t_end, p=p, native_step_h=step, weight=float(cfg.get("weight", 1.0)),
                  meta={"nan_frac": float(np.isnan(p).mean())})


def fill_gaps(members: list[Member]) -> None:
    """AROME has no data south of ~38 N. Fill each member's NaN cells with the best
    available coarser run at the same hour (regional first, then global), else 0.
    The donor's values carry the donor's representativeness factor (a 25-km amount put
    into a 1.3-km member would otherwise lose it)."""
    from . import params as P
    fams = P.load()["families"]
    donors = sorted([m for m in members if m.family in ("regional", "global") and not np.isnan(m.p).any()],
                    key=lambda m: (m.family != "regional", -m.run.timestamp()))
    for m in members:
        if not np.isnan(m.p).any():
            continue
        for d in donors:
            pos = np.searchsorted(d.t_end, m.t_end)
            ok = (pos < len(d.t_end)) & (d.t_end[np.minimum(pos, len(d.t_end) - 1)] == m.t_end)
            if ok.any():
                hole = np.isnan(m.p[ok])
                sub = m.p[ok]
                sub[hole] = d.p[pos[ok]][hole] * (fams[d.family]["s12h"] / fams[m.family]["s12h"])
                m.p[ok] = sub
                m.meta["filled_from"] = d.name
            if not np.isnan(m.p).any():
                break
        m.p = np.nan_to_num(m.p, nan=0.0)


def live_members(now: datetime, hours_back: int = 14, hours_ahead: int = 52,
                 keys: tuple[str, ...] = ("arome_hd", "arome", "icon_eu", "arpege", "ifs")) -> tuple[list[Member], list[dict]]:
    """Newest complete runs of every model, as members. Returns (members, source report)."""
    t_from = now - timedelta(hours=hours_back)
    members, report = [], []
    for key in keys:
        cfg = MODELS[key]
        try:
            runs = om.latest_runs(cfg["s3"], n=cfg["lags"])
        except Exception as e:
            report.append({"id": key, "label": cfg["label"], "ok": False, "error": f"{type(e).__name__}: {e}"[:160]})
            continue
        got = []
        ahead = hours_ahead if key != "ifs" else 24 * 8
        for r in runs:
            m = load_run(key, r, t_from, now + timedelta(hours=ahead))
            if m is not None:
                members.append(m); got.append(naive(r).strftime("%Y-%m-%dT%H:%MZ"))
        report.append({"id": key, "label": cfg["label"], "ok": bool(got), "runs": got, "family": cfg["family"]})
    fill_gaps(members)
    return members, report
