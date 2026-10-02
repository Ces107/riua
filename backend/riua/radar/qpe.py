"""Radar quantitative precipitation estimation (QPE) on the 0.01 deg radar grid.

Chain
-----
1. Quality: isolated strong pixels (clutter) removed, plus a static clutter mask when the
   file `clutter.npy` sits next to this module; NaN stays NaN (no coverage is not the same
   as no rain).
2. Reflectivity to rain rate with two Z-R laws: Marshall-Palmer Z = 200 R^1.6 for
   stratiform rain and Z = 300 R^1.4 (the WSR-88D convective law) where the echo is
   convective. Convective cells are found with the Steiner et al. (1995) peakedness
   criterion: >= 40 dBZ, or exceeding the 11-km background mean by a margin that
   shrinks as the background grows. Reflectivity is capped at 57 dBZ (hail).
3. Accumulation with advection correction (Anagnostou and Krajewski 1999): storms
   move between scans, so adding 10-minute snapshots prints stripes and misses the
   peak between two positions. The field is advected with the optical-flow motion
   (pysteps, Lucas-Kanade) in sub-steps, forwards from the earlier scan and
   backwards from the later one, and the two are cross-faded.
4. Merging with rain gauges: radar gives the pattern, gauges give the amount.
   a) gauge quality control against the neighbours and the radar,
   b) registration: rain measured aloft lands downwind, so the accumulation is moved by
      the few km that best match the gauges,
   c) conditional merging (Sinclair and Pegram 2005) in square-root space: the radar field
      plus the kriged gauge-minus-radar residual. Additive where it rains little,
      multiplicative where it rains a lot, exact at the gauges up to the nugget.
   Where the radar has no coverage the gauges alone are kriged. Chosen by leave-gauges-out
   cross-validation against the wradlib adjustments, mean-field bias and kriging with
   external drift (hindcast/qpe_validation.py). The radar alone is not trustworthy in
   amount: C-band attenuation behind a strong core under-reads, hail and clutter over-read.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage

RG_LON0, RG_LAT0, RG_D = -2.4, 37.6, 0.01
RG_NX, RG_NY = 320, 340
KM_LAT = 111.2
KM_LON = 111.2 * float(np.cos(np.deg2rad(39.3)))
NO_ECHO_DBZ = -32.0
HAIL_CAP_DBZ = 57.0
FALL_MIN = 10.0         # minutes between the scan label and the rain in the gauges (2026-10-01: hourly r 0.63 -> 0.68)
CLIM_FACTOR = 0.75      # gauge / radar over 29 wet hindcast days (median 0.71) and 2026-10-01 (0.82): used without gauges

_CL = Path(__file__).with_name("clutter.npy")
CLUTTER = np.load(_CL) if _CL.exists() else None        # (340, 320) bool: echo there is ground clutter


def radar_mesh():
    lon = RG_LON0 + (np.arange(RG_NX) + 0.5) * RG_D
    lat = RG_LAT0 + (np.arange(RG_NY) + 0.5) * RG_D
    return np.meshgrid(lon, lat)  # lon2, lat2


def despeckle(dbz: np.ndarray, min_size: int = 4, strong: float = 30.0) -> np.ndarray:
    """Remove echoes >= `strong` dBZ that form objects smaller than `min_size` pixels; static clutter
    pixels become NaN (unknown: the neighbours and the gauges fill them, they are not dry)."""
    out = dbz.copy()
    if CLUTTER is not None:
        out[CLUTTER] = np.nan
    echo = np.nan_to_num(out, nan=NO_ECHO_DBZ) >= 12.0
    lab, n = ndimage.label(echo)
    if n == 0:
        return out
    sizes = ndimage.sum_labels(echo, lab, index=np.arange(1, n + 1))
    peak = ndimage.maximum(np.nan_to_num(out, nan=NO_ECHO_DBZ), lab, index=np.arange(1, n + 1))
    bad = np.nonzero((sizes < min_size) & (peak >= strong))[0] + 1
    if bad.size:
        out[np.isin(lab, bad)] = NO_ECHO_DBZ
    return out


def steiner_convective(dbz: np.ndarray, bg_radius_km: float = 11.0, intense: float = 40.0) -> np.ndarray:
    """Convective mask after Steiner, Houze and Yuter (1995), on a ~1 km grid."""
    z = np.nan_to_num(dbz, nan=NO_ECHO_DBZ)
    lin = 10.0 ** (np.maximum(z, 0.0) / 10.0)
    ry, rx = int(round(bg_radius_km / (RG_D * KM_LAT))), int(round(bg_radius_km / (RG_D * KM_LON)))
    if z.max() < 25.0:
        return np.zeros(z.shape, bool)
    # box of the same area as the 11-km disc (separable, ~50x faster than a disc convolution)
    k = np.sqrt(np.pi) / 2.0
    bg_lin = ndimage.uniform_filter(lin, size=(max(int(2 * ry * k) | 1, 3), max(int(2 * rx * k) | 1, 3)), mode="nearest")
    bg = 10.0 * np.log10(np.maximum(bg_lin, 1.0))
    # peakedness margin: 10 dB at weak background, falling to 0 at 42.43 dBZ
    margin = np.where(bg < 0, 10.0, np.where(bg < 42.43, 10.0 - bg ** 2 / 180.0, 0.0))
    core = (z >= intense) | ((z - bg >= margin) & (z >= 25.0))
    # convective radius grows with background intensity (1 to 5 km)
    rad_km = np.clip(1.0 + (bg - 25.0) / 5.0, 1.0, 5.0)
    out = core.copy()
    for r in (2, 3, 4, 5):
        sel = core & (rad_km >= r)
        if sel.any():
            out |= ndimage.binary_dilation(sel, iterations=r - 1)
    return out & (z >= 15.0)


def rain_rate(dbz: np.ndarray) -> np.ndarray:
    """dBZ -> mm/h with stratiform / convective Z-R. NaN preserved."""
    z = np.minimum(np.where(np.isfinite(dbz), dbz, NO_ECHO_DBZ), HAIL_CAP_DBZ)
    lin = 10.0 ** (z / 10.0)
    r = np.where(steiner_convective(dbz), (lin / 300.0) ** (1.0 / 1.4), (lin / 200.0) ** (1.0 / 1.6))
    r = np.where(z < 7.0, 0.0, r)   # below ~0.1 mm/h: drizzle / noise
    return np.where(np.isfinite(dbz), r, np.nan).astype(np.float32)


def motion_field(rates: list[np.ndarray], coarse: int = 2) -> np.ndarray:
    """Lucas-Kanade optical flow (pysteps) from the last 2-4 rain-rate fields. (2, ny, nx) px/step.
    Computed on `coarse`-px block means: storm motion is a smooth field and this is ~4x cheaper."""
    from pysteps import motion
    from pysteps.utils import transformation
    stack = np.stack([np.nan_to_num(r, nan=0.0) for r in rates[-4:]]).astype(np.float64)
    ny, nx = stack.shape[1:]
    if (stack[-1] > 0.1).mean() < 0.002 or len(rates) < 2:
        return np.zeros((2, ny, nx))
    c = coarse if ny % coarse == 0 and nx % coarse == 0 else 1
    small = stack.reshape(len(stack), ny // c, c, nx // c, c).mean(axis=(2, 4))
    db, _ = transformation.dB_transform(small, threshold=0.1, zerovalue=-15.0)
    v = np.nan_to_num(motion.get_method("LK")(db), nan=0.0)
    return np.repeat(np.repeat(v, c, axis=1), c, axis=2) * c


def _advect(field: np.ndarray, v: np.ndarray, frac: float) -> np.ndarray:
    from pysteps import extrapolation
    if frac == 0 or not np.any(v):
        return field
    out = extrapolation.get_method("semilagrangian")(field, v * frac, 1, outval=0.0)[0]
    return np.nan_to_num(out, nan=0.0)


def accumulate(frames: list[tuple[datetime, np.ndarray]], t0: datetime, t1: datetime,
               substeps: int = 5, max_gap_min: float = 25.0, drift_min: float = 0.0) -> tuple[np.ndarray, float]:
    """Advection-corrected accumulation (mm) between t0 and t1 from (time, rain-rate) frames.

    Returns (accumulation, coverage) where coverage is the fraction of the period
    actually bracketed by scans: the accumulation covers only that fraction. Cells without
    radar coverage come back NaN; a cell seen by one of the two scans of an interval takes
    that scan for the whole interval (the newest scans of the blend still have the OPERA hole).
    `drift_min`: the frame times are ground-arrival times, FALL_MIN after the scan (the caller
    adds it); the echo is carried that many minutes further along the storm motion.
    """
    frames = sorted(frames, key=lambda x: x[0])
    acc = np.zeros((RG_NY, RG_NX), np.float64)
    seen = np.zeros((RG_NY, RG_NX), np.float64)
    covered = 0.0
    total = (t1 - t0).total_seconds() / 60.0
    for k in range(len(frames) - 1):
        (ta, ra), (tb, rb) = frames[k], frames[k + 1]
        lo, hi = max(ta, t0), min(tb, t1)
        if hi <= lo:
            continue
        gap = (tb - ta).total_seconds() / 60.0
        if gap > max_gap_min:
            continue
        a0 = np.nan_to_num(np.where(np.isnan(ra), rb, ra), nan=0.0)
        b0 = np.nan_to_num(np.where(np.isnan(rb), ra, rb), nan=0.0)
        ok = (np.isfinite(ra) | np.isfinite(rb)).astype(np.float64)
        covered += (hi - lo).total_seconds() / 60.0
        if max((a0 >= 0.5).mean(), (b0 >= 0.5).mean()) < 0.001:
            # (almost) nothing to move: plain trapezoid, no optical flow
            w = (hi - lo).total_seconds() / 3600.0
            acc += 0.5 * (a0 + b0) * w
            seen += ok * w
            continue
        v = motion_field([frames[max(k - 1, 0)][1], ra, rb])
        dt_h = gap / substeps / 60.0
        for s in range(substeps):
            tau = (s + 0.5) / substeps
            ts = ta + timedelta(minutes=gap * tau)
            if ts < lo or ts > hi:
                continue
            fa, fb = tau + drift_min / gap, 1.0 - tau - drift_min / gap
            acc += ((1.0 - tau) * _advect(a0, v, fa) + tau * _advect(b0, -v, fb)) * dt_h
            seen += ok * dt_h
    out = np.where(seen > 0, acc, np.nan).astype(np.float32)
    return out, (covered / total if total > 0 else 0.0)


# ------------------------------------------------------------------------------ gauges

def gauge_px(g_lat, g_lon):
    """Radar pixel (j, i) of each gauge and whether it is inside the grid."""
    j = np.floor((np.asarray(g_lat, np.float64) - RG_LAT0) / RG_D).astype(int)
    i = np.floor((np.asarray(g_lon, np.float64) - RG_LON0) / RG_D).astype(int)
    return j, i, (j >= 0) & (j < RG_NY) & (i >= 0) & (i < RG_NX)


def at_gauges(field: np.ndarray, g_lat, g_lon, nb: int = 3) -> np.ndarray:
    """Mean of the nb x nb pixels around each gauge (3 km: gauge-pixel mismatch); NaN outside or uncovered."""
    v = np.isfinite(field).astype(np.float32)
    w = ndimage.uniform_filter(v, nb, mode="nearest")
    f = np.where(w > 0, ndimage.uniform_filter(np.nan_to_num(field, nan=0.0), nb, mode="nearest") / np.maximum(w, 1e-6), np.nan)
    j, i, ok = gauge_px(g_lat, g_lon)
    out = np.full(len(j), np.nan)
    out[ok] = f[j[ok], i[ok]]
    return out


def shift(field: np.ndarray, dj: int, di: int) -> np.ndarray:
    """Move the last two axes by (dj north, di east) pixels, repeating the edge."""
    if dj == 0 and di == 0:
        return field
    js = np.clip(np.arange(field.shape[-2]) - dj, 0, field.shape[-2] - 1)
    is_ = np.clip(np.arange(field.shape[-1]) - di, 0, field.shape[-1] - 1)
    return field[..., js[:, None], is_[None, :]]


def register(radar_mm: np.ndarray, g_lat, g_lon, g_mm, max_px: int = 6, min_wet: int = 30,
             min_gain: float = 0.03) -> tuple[int, int, float]:
    """Displacement (dj, di, correlation gain) that best aligns a radar accumulation with the gauges.

    The radar sees the rain 1-3 km up; it lands minutes later and km downwind, a different
    way every day. All shifts within `max_px` are tried and the best is kept only when the
    correlation of the square roots improves by `min_gain` and enough gauges are wet."""
    j, i, ok = gauge_px(g_lat, g_lon)
    g = np.sqrt(np.asarray(g_mm, np.float64))
    f = np.sqrt(ndimage.uniform_filter(np.nan_to_num(radar_mm, nan=0.0), 3, mode="nearest"))
    ok &= np.isfinite(g)
    j, i, g = j[ok], i[ok], g[ok]
    if ((g >= 1.0) & (f[j, i] >= 1.0)).sum() < min_wet:
        return 0, 0, 0.0
    best = (-2.0, 0, 0)
    c0 = 0.0
    for dj in range(-max_px, max_px + 1):
        for di in range(-max_px, max_px + 1):
            r = f[np.clip(j - dj, 0, RG_NY - 1), np.clip(i - di, 0, RG_NX - 1)]
            c = float(np.corrcoef(r, g)[0, 1]) if r.std() > 0 else -1.0
            if dj == 0 and di == 0:
                c0 = c
            if c > best[0]:
                best = (c, dj, di)
    return (best[1], best[2], best[0] - c0) if best[0] - c0 >= min_gain else (0, 0, 0.0)


def gauge_qc(g_lat, g_lon, g_mm, r_at, radius_km: float = 15.0) -> np.ndarray:
    """Keep-mask of the gauges. Out: impossible values; a dry gauge where the radar and the
    neighbours agree that it rained (blocked funnel, dead telemetry); a gauge far above both the
    radar and every neighbour (maintenance tip, backlog dump)."""
    g = np.asarray(g_mm, np.float64)
    keep = np.isfinite(g) & (g >= 0.0) & (g < 1500.0)
    x, y = np.asarray(g_lon, np.float64) * KM_LON, np.asarray(g_lat, np.float64) * KM_LAT
    r = np.nan_to_num(np.asarray(r_at, np.float64), nan=-1.0)
    for k in np.nonzero(keep & ((g < 0.5) | (g > 30.0)))[0]:
        near = keep & (np.hypot(x - x[k], y - y[k]) <= radius_km)
        near[k] = False
        if near.sum() < 2:
            continue
        nb = g[near]
        if g[k] < 0.5 and r[k] >= 15.0 and np.median(nb) >= 10.0:
            keep[k] = False
        elif g[k] > 30.0 and r[k] >= 0.0 and g[k] > 3.0 * max(nb.max(), r[k]) + 20.0:
            keep[k] = False
    return keep


def krige(xy: np.ndarray, z: np.ndarray, xy_t: np.ndarray, rng_km: float = 40.0, nugget: float = 0.1):
    """Ordinary kriging, exponential variogram (practical range `rng_km`, relative nugget).
    Returns (estimate, variance as a fraction of the sill) at the targets."""
    n = len(z)
    gam = lambda h: nugget + (1.0 - nugget) * (1.0 - np.exp(-3.0 * h / rng_km))
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = gam(np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]))
    A[np.arange(n + 1), np.arange(n + 1)] = 0.0
    B = np.ones((n + 1, len(xy_t)))
    B[:n] = gam(np.hypot(xy[:, None, 0] - xy_t[None, :, 0], xy[:, None, 1] - xy_t[None, :, 1]))
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:           # two gauges on the same spot
        W = np.linalg.lstsq(A, B, rcond=None)[0]
    return W[:n].T @ z, np.maximum((W * B).sum(axis=0), 0.0)


def krige_field(g_lat, g_lon, z, rng_km: float = 40.0, nugget: float = 0.1, step: int = 5):
    """Kriged field of a smooth quantity on the radar grid: solved every `step` px, then bilinear.
    Returns (field, variance fraction)."""
    lon = (RG_LON0 + (np.arange(0, RG_NX, step) + step / 2.0) * RG_D) * KM_LON
    lat = (RG_LAT0 + (np.arange(0, RG_NY, step) + step / 2.0) * RG_D) * KM_LAT
    x, y = np.meshgrid(lon, lat)
    xy = np.column_stack([np.asarray(g_lon, np.float64) * KM_LON, np.asarray(g_lat, np.float64) * KM_LAT])
    est, var = krige(xy, np.asarray(z, np.float64), np.column_stack([x.ravel(), y.ravel()]), rng_km, nugget)
    up = lambda a: ndimage.zoom(a.reshape(x.shape), step, order=1, mode="nearest", grid_mode=True)
    return up(est), up(var)


def gauges_only(g_lat, g_lon, g_mm, rng_km: float = 40.0, nugget: float = 0.1) -> np.ndarray:
    """Rain field from the gauges alone (radar down, or no coverage): kriging of the square root,
    squared back with the kriging variance so that the mean is not lost between gauges."""
    s = np.sqrt(np.maximum(np.asarray(g_mm, np.float64), 0.0))
    est, var = krige_field(g_lat, g_lon, s, rng_km, nugget)
    est = np.maximum(est, 0.0)
    return (est ** 2 + np.minimum(var * float(np.var(s)), est ** 2)).astype(np.float32)


def merge_gauges(radar_mm: np.ndarray, g_lat: np.ndarray, g_lon: np.ndarray, g_mm: np.ndarray,
                 min_gauges: int = 5, r_at: np.ndarray | None = None, rng_km: float = 40.0,
                 nugget: float = 0.1, do_register: bool = True, qc: bool = True) -> tuple[np.ndarray, dict]:
    """Adjust a radar accumulation with gauge totals over the same period (chain in the module
    docstring, step 4). `r_at`: radar value at each gauge when the caller has a better one (the
    radar total over the gauge's own time window). Without enough gauges: mean-field bias, then
    the climatological factor. Where the radar has no coverage: the gauges alone."""
    info = {"n_gauges": 0, "method": "none", "mfb": None}
    g_lat, g_lon, g_mm = map(lambda a: np.asarray(a, np.float64), (g_lat, g_lon, g_mm))
    inside = np.isfinite(g_mm) & gauge_px(g_lat, g_lon)[2]
    if r_at is None and do_register and inside.sum() >= min_gauges:
        dj, di, gain = register(radar_mm, g_lat[inside], g_lon[inside], g_mm[inside])
        if dj or di:
            radar_mm = shift(radar_mm, dj, di)
            info["shift_px"] = [dj, di, round(gain, 3)]
    r_at = at_gauges(radar_mm, g_lat, g_lon) if r_at is None else np.asarray(r_at, np.float64)
    keep = inside & (gauge_qc(g_lat, g_lon, g_mm, r_at) if qc else True)
    info["n_gauges"], info["n_rejected"] = int(keep.sum()), int((inside & ~keep).sum())
    g_lat, g_lon, g_mm, r_at = g_lat[keep], g_lon[keep], g_mm[keep], r_at[keep]
    have = np.isfinite(radar_mm)
    out = np.nan_to_num(radar_mm, nan=0.0).astype(np.float64)
    pair = np.isfinite(r_at)
    both = pair & (g_mm >= 1.0) & (r_at >= 1.0)
    if both.sum() >= 3:
        info["mfb"] = float(np.clip(g_mm[both].sum() / r_at[both].sum(), 0.2, 5.0))
    if pair.sum() >= min_gauges:
        try:
            d, _ = krige_field(g_lat[pair], g_lon[pair], np.sqrt(g_mm[pair]) - np.sqrt(r_at[pair]), rng_km, nugget)
            out = np.maximum(np.sqrt(out) + d, 0.0) ** 2
            info["method"] = "conditional merging (sqrt)"
        except Exception as e:  # keep the pipeline alive, record why
            info["error"] = f"{type(e).__name__}: {e}"[:200]
    if info["method"] == "none":
        f = info["mfb"] if info["mfb"] is not None else CLIM_FACTOR
        out *= f
        info["method"] = "mean-field-bias" if info["mfb"] is not None else f"radar x {CLIM_FACTOR} (climatological bias)"
    # radar holes: gauges only
    if (~have).any() and g_mm.size >= min_gauges:
        try:
            out[~have] = gauges_only(g_lat, g_lon, g_mm, rng_km, nugget)[~have]
            info["hole_fill"] = "kriging of gauges (sqrt-transformed)"
        except Exception as e:
            info["hole_fill_error"] = f"{type(e).__name__}: {e}"[:200]
    if "hole_fill" not in info:
        out[~have] = np.nan
    return out.astype(np.float32), info


FAC_MIN, FAC_MAX = 0.2, 5.0       # bounds of the correction factor (upper: Park et al. 2019)


def ceiling(g_lat, g_lon, g_mm, r_at=None, factor: float = 3.0, plus: float = 20.0, radius_cells: int = 4,
            min_gauges: int = 3) -> tuple[np.ndarray, int]:
    """Upper bound (ny, nx) of an accumulation: `factor` x the largest gauge within ~20 km plus `plus` mm.
    It exists to stop hail and clutter, so it is loose (a convective core between gauges is real far more
    often than not) and it only applies where at least `min_gauges` gauges lie within that distance. It is
    lifted (inf) around every gauge that reads far more than the radar above it: there the radar is
    attenuated behind a strong core and is a lower bound. Decided per 5-km analysis cell, so that the
    maximum and the mean of a cell are bounded alike. Returns (bound, number of such gauges)."""
    j, i, ok = gauge_px(g_lat, g_lon)
    g_mm = np.asarray(g_mm, np.float64)
    g = np.zeros((RG_NY // 5, RG_NX // 5), np.float32)
    n, flag = np.zeros_like(g), np.zeros(g.shape, bool)
    np.maximum.at(g, (j[ok] // 5, i[ok] // 5), g_mm[ok].astype(np.float32))
    np.add.at(n, (j[ok] // 5, i[ok] // 5), 1.0)
    size = 2 * radius_cells + 1
    count = ndimage.uniform_filter(n, size, mode="constant") * size * size
    out = np.where(count >= min_gauges - 0.5, factor * ndimage.maximum_filter(g, size=size) + plus, np.inf)
    att = np.zeros(len(g_mm), bool)
    if r_at is not None:
        r = np.asarray(r_at, np.float64)
        att = ok & np.isfinite(r) & (g_mm >= 20.0) & (g_mm > 2.5 * np.maximum(r, 1.0))
        flag[j[att] // 5, i[att] // 5] = True
        out[ndimage.maximum_filter(flag, size=size)] = np.inf
    return np.repeat(np.repeat(out, 5, axis=0), 5, axis=1).astype(np.float32), int(att.sum())


def _minutes(t_utc, ref: datetime) -> float:
    """Minutes from `ref` to a gauge stamp ('2026-10-01T21:10:00Z'); 0 when the network gives none."""
    try:
        return (datetime.strptime(str(t_utc)[:19], "%Y-%m-%dT%H:%M:%S") - ref).total_seconds() / 60.0
    except ValueError:
        return 0.0


def analyse(segs: np.ndarray, bounds: list[tuple[datetime, datetime]], gauges: list[dict],
            max_age_h: float = 2.5) -> tuple[np.ndarray, dict]:
    """The radar accumulations of the last 12 h, corrected with the gauges.

    segs (S, ny, nx): raw mm of consecutive periods, oldest first (clock hours, then the hour under
    way), NaN = no coverage; bounds: their (start, end), naive UTC ground-arrival times. gauges: dicts
    with lat, lon, t_utc and the total p_12h of the 12 h ENDING at t_utc. Every gauge is compared with
    the radar over its own period (the networks stamp between now and 2 h ago; older stamps are
    dropped) and the correction of the 12-h total is applied to every hour. A separate correction of
    the last hour with the 1-h totals was tried and did not improve the 1-h scores (findings q1-qpe).
    Returns (corrected segs (S, ny, nx) float32 without NaN, info)."""
    t_ref = bounds[-1][1]
    ss = np.array([(a - t_ref).total_seconds() / 60.0 for a, _ in bounds])
    se = np.array([(b - t_ref).total_seconds() / 60.0 for _, b in bounds])
    gs = [g for g in gauges if g.get("lat") is not None and g.get("lon") is not None]
    lat, lon = (np.array([g[k] for g in gs], np.float64) for k in ("lat", "lon"))
    p12 = np.array([np.nan if g.get("p_12h") is None else float(g["p_12h"]) for g in gs], np.float64)
    tg =np.array([_minutes(g.get("t_utc"), t_ref) for g in gs], np.float64)
    inside = gauge_px(lat, lon)[2] if len(gs) else np.zeros(0, bool)
    fresh = inside & (tg >= -60.0 * max_age_h)
    tg = np.minimum(tg, 0.0)
    info = {"method": "none", "n": int(fresh.sum()), "n_stale": int((inside & ~fresh).sum())}
    u = fresh & np.isfinite(p12)
    if u.sum() >= 5:                      # where the rain landed, not where the radar saw it
        dj, di, gain = register(np.nan_to_num(segs, nan=0.0).sum(axis=0), lat[u], lon[u], p12[u])
        if dj or di:
            segs = shift(segs, dj, di)
            info["shift_px"] = [dj, di, round(gain, 3)]
    base = np.nan_to_num(segs, nan=0.0)
    rg = np.stack([at_gauges(s, lat, lon) for s in segs]) if len(gs) else np.zeros((len(segs), 0))

    def window(w):
        """Radar over w minutes: at every gauge (period ending at its stamp), as a field (ending at t_ref)."""
        ov = np.clip(np.minimum(se[:, None], tg[None, :]) - np.maximum(ss[:, None], tg[None, :] - w), 0.0, None)
        r = (ov / (se - ss)[:, None] * np.where(ov > 0, rg, 0.0)).sum(axis=0)
        r[ov.sum(axis=0) < 0.9 * w] = np.nan
        wf = np.clip(np.minimum(se, 0.0) - np.maximum(ss, -w), 0.0, None) / (se - ss)
        field = np.tensordot(wf, base, axes=1)
        field[~np.isfinite(segs[wf > 0]).any(axis=0)] = np.nan
        return r, field, wf, float((wf * (se - ss)).sum() / w)

    out = base * CLIM_FACTOR
    r12, R12, w12, cov12 = window(720.0)
    merged12 = None
    if cov12 >= 0.8:
        u[u] = gauge_qc(lat[u], lon[u], p12[u], r12[u])
        merged12, m = merge_gauges(R12, lat[u], lon[u], p12[u], r_at=r12[u], qc=False)
        R12z = np.nan_to_num(R12, nan=0.0)
        fac = np.where(R12z >= 0.5, np.clip(merged12 / np.maximum(R12z, 1e-6), FAC_MIN, FAC_MAX), 1.0)
        out = base * fac
        info.update(method=m["method"], n=int(u.sum()), n_rejected=int((fresh & np.isfinite(p12) & ~u).sum()),
                    gauge_max_12h=float(p12[u].max()) if u.any() else None,
                    radar_over_gauge=round(1.0 / m["mfb"], 2) if m.get("mfb") else None)
    if merged12 is not None:
        # rain the radar missed or could not see: the gauge-derived amount, spread with the regional hourly profile
        extra = np.where(R12z < 0.5, np.maximum(np.nan_to_num(merged12, nan=0.0) - R12z, 0.0), 0.0)
        prof = w12 * base.mean(axis=(1, 2))
        prof = prof / prof.sum() if prof.sum() > 0 else w12 * (se - ss) / (w12 * (se - ss)).sum()
        out += extra[None] * prof[:, None, None]
        ceil, info["attenuation_flags"] = ceiling(lat[u], lon[u], p12[u], r12[u])
        tot = np.tensordot(w12, out, axes=1)
        scale = np.where(tot > ceil, ceil / np.maximum(tot, 1e-6), 1.0)
        out *= scale[None]
        info["ceiling_px"] = int((scale < 0.999).sum())
    return out.astype(np.float32), info


def analyse_gauges(gauges: list[dict], t_ref: datetime, hours: int = 12, max_age_h: float = 2.5):
    """Radar down: `hours` hourly fields ending at t_ref from the gauges alone. The 12-h and 1-h totals are
    kriged; the last hour takes the 1-h field and the earlier ones share the rest evenly (the gauges
    give no timing). Returns ((hours, ny, nx) float32 or None, info)."""
    gs = [g for g in gauges if g.get("lat") is not None and g.get("p_12h") is not None
          and _minutes(g.get("t_utc"), t_ref) >= -60.0 * max_age_h]
    if len(gs) < 8:
        return None, {"method": "none (radar down, fewer than 8 gauges)", "n": len(gs)}
    lat, lon = (np.array([g[k] for g in gs], np.float64) for k in ("lat", "lon"))
    g12 = gauges_only(lat, lon, [g["p_12h"] for g in gs])
    h1 = [g for g in gs if g.get("p_1h") is not None]
    g1 = np.minimum(gauges_only([g["lat"] for g in h1], [g["lon"] for g in h1], [g["p_1h"] for g in h1]), g12) \
        if len(h1) >= 8 else g12 / hours
    out = np.repeat(((g12 - g1) / (hours - 1))[None], hours, axis=0)
    out[-1] = g1
    return out.astype(np.float32), {"method": "gauges only (kriging), radar down", "n": len(gs),
                                    "gauge_max_12h": float(max(g["p_12h"] for g in gs))}
