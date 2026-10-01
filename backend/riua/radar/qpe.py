"""Radar quantitative precipitation estimation (QPE) on the 0.01 deg radar grid.

Chain
-----
1. Quality: isolated strong pixels (clutter) removed; NaN stays NaN (no coverage is
   not the same as no rain).
2. Reflectivity to rain rate with two Z-R laws: Marshall-Palmer Z = 200 R^1.6 for
   stratiform rain and Z = 300 R^1.4 (the WSR-88D convective law) where the echo is
   convective. Convective cells are found with the Steiner et al. (1995) peakedness
   criterion: >= 40 dBZ, or exceeding the 11-km background mean by a margin that
   shrinks as the background grows. Reflectivity is capped at 57 dBZ (hail).
3. Accumulation with advection correction (Anagnostou and Krajewski 1999): storms
   move between scans, so adding 10-minute snapshots prints stripes and misses the
   peak between two positions. The field is advected with the optical-flow motion
   (pysteps, Lucas-Kanade) in 1-minute sub-steps, forwards from the earlier scan and
   backwards from the later one, and the two are cross-faded.
4. Merging with rain gauges (wradlib, Pfaff 2010 "mixed" additive-multiplicative
   error model): radar gives the pattern, gauges give the amount. In October 2024
   the Valencia radar lost much of its signal to attenuation by the storm itself;
   without gauges the radar alone would have halved the rain.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from scipy import ndimage

RG_LON0, RG_LAT0, RG_D = -2.4, 37.6, 0.01
RG_NX, RG_NY = 320, 340
KM_LAT = 111.2
KM_LON = 111.2 * float(np.cos(np.deg2rad(39.3)))
NO_ECHO_DBZ = -32.0
HAIL_CAP_DBZ = 57.0


def radar_mesh():
    lon = RG_LON0 + (np.arange(RG_NX) + 0.5) * RG_D
    lat = RG_LAT0 + (np.arange(RG_NY) + 0.5) * RG_D
    return np.meshgrid(lon, lat)  # lon2, lat2


def despeckle(dbz: np.ndarray, min_size: int = 4, strong: float = 30.0) -> np.ndarray:
    """Remove echoes >= `strong` dBZ that form objects smaller than `min_size` pixels."""
    out = dbz.copy()
    echo = np.nan_to_num(dbz, nan=NO_ECHO_DBZ) >= 12.0
    lab, n = ndimage.label(echo)
    if n == 0:
        return out
    sizes = ndimage.sum_labels(echo, lab, index=np.arange(1, n + 1))
    peak = ndimage.maximum(np.nan_to_num(dbz, nan=NO_ECHO_DBZ), lab, index=np.arange(1, n + 1))
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
    import wradlib as wrl
    z = np.minimum(np.where(np.isfinite(dbz), dbz, NO_ECHO_DBZ), HAIL_CAP_DBZ)
    lin = wrl.trafo.idecibel(z)
    conv = steiner_convective(dbz)
    r = np.where(conv, wrl.zr.z_to_r(lin, a=300.0, b=1.4), wrl.zr.z_to_r(lin, a=200.0, b=1.6))
    r = np.where(z < 7.0, 0.0, r)   # below ~0.1 mm/h: drizzle / noise
    return np.where(np.isfinite(dbz), r, np.nan).astype(np.float32)


def motion_field(rates: list[np.ndarray]) -> np.ndarray:
    """Lucas-Kanade optical flow (pysteps) from the last 2-4 rain-rate fields. (2, ny, nx) px/step."""
    from pysteps import motion
    from pysteps.utils import transformation
    stack = np.stack([np.nan_to_num(r, nan=0.0) for r in rates[-4:]]).astype(np.float64)
    if (stack[-1] > 0.1).mean() < 0.002 or len(rates) < 2:
        return np.zeros((2, *stack.shape[1:]))
    db, _ = transformation.dB_transform(stack, threshold=0.1, zerovalue=-15.0)
    v = motion.get_method("LK")(db)
    return np.nan_to_num(v, nan=0.0)


def _advect(field: np.ndarray, v: np.ndarray, frac: float) -> np.ndarray:
    from pysteps import extrapolation
    if frac == 0 or not np.any(v):
        return field
    out = extrapolation.get_method("semilagrangian")(field, v * frac, 1, outval=0.0)[0]
    return np.nan_to_num(out, nan=0.0)


def accumulate(frames: list[tuple[datetime, np.ndarray]], t0: datetime, t1: datetime,
               substeps: int = 5, max_gap_min: float = 25.0) -> tuple[np.ndarray, float]:
    """Advection-corrected accumulation (mm) between t0 and t1 from (time, rain-rate) frames.

    Returns (accumulation, coverage) where coverage is the fraction of the period
    actually bracketed by scans. Cells without radar coverage come back NaN.
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
        a0, b0 = np.nan_to_num(ra, nan=0.0), np.nan_to_num(rb, nan=0.0)
        if max((a0 >= 0.5).mean(), (b0 >= 0.5).mean()) < 0.001:
            # (almost) nothing to move: plain trapezoid, no optical flow
            ok = (np.isfinite(ra) | np.isfinite(rb)).astype(np.float64)
            w = (hi - lo).total_seconds() / 3600.0
            acc += 0.5 * (a0 + b0) * w
            seen += ok * w
            covered += (hi - lo).total_seconds() / 60.0
            continue
        v = motion_field([frames[max(k - 1, 0)][1], ra, rb])
        ok = (np.isfinite(ra) | np.isfinite(rb)).astype(np.float64)
        for s in range(substeps):
            tau = (s + 0.5) / substeps
            ts = ta + timedelta(minutes=gap * tau)
            if ts < lo or ts > hi:
                continue
            fwd = _advect(a0, v, tau)
            bwd = _advect(b0, -v, 1.0 - tau)
            rate = (1.0 - tau) * fwd + tau * bwd
            dt_h = gap / substeps / 60.0
            acc += rate * dt_h
            seen += ok * dt_h
        covered += (hi - lo).total_seconds() / 60.0
    out = np.where(seen > 0, acc, np.nan).astype(np.float32)
    return out, (covered / total if total > 0 else 0.0)


def merge_gauges(radar_mm: np.ndarray, g_lat: np.ndarray, g_lon: np.ndarray, g_mm: np.ndarray,
                 min_gauges: int = 5) -> tuple[np.ndarray, dict]:
    """Adjust a radar accumulation with gauge totals over the same period.

    wradlib AdjustMixed (Pfaff 2010): the radar-gauge error is modelled as a spatially
    variable multiplicative factor plus an additive term, both interpolated from the
    gauges. Where the radar has no coverage the gauges alone are interpolated
    (ordinary kriging, exponential variogram). Falls back to the mean-field bias when
    there are few usable gauges.
    """
    import wradlib as wrl
    info = {"n_gauges": 0, "method": "none", "mfb": None}
    lon2, lat2 = radar_mesh()
    g_lat, g_lon, g_mm = map(lambda a: np.asarray(a, np.float64), (g_lat, g_lon, g_mm))
    inside = (np.isfinite(g_mm) & (g_lon > RG_LON0) & (g_lon < RG_LON0 + RG_NX * RG_D)
              & (g_lat > RG_LAT0) & (g_lat < RG_LAT0 + RG_NY * RG_D))
    g_lat, g_lon, g_mm = g_lat[inside], g_lon[inside], g_mm[inside]
    info["n_gauges"] = int(g_mm.size)
    if g_mm.size < min_gauges:
        return radar_mm, info
    raw_xy = np.column_stack([lon2.ravel() * KM_LON, lat2.ravel() * KM_LAT])
    obs_xy = np.column_stack([g_lon * KM_LON, g_lat * KM_LAT])
    raw = radar_mm.ravel().astype(np.float64)
    have = np.isfinite(raw)
    out = raw.copy()
    # mean-field bias from gauge / radar pairs with rain on both
    ji = (np.floor((g_lat - RG_LAT0) / RG_D).astype(int), np.floor((g_lon - RG_LON0) / RG_D).astype(int))
    r_at = ndimage.maximum_filter(np.nan_to_num(radar_mm, nan=0.0), size=3)[ji]   # 3x3 km: gauge-pixel mismatch
    both = (g_mm >= 1.0) & (r_at >= 1.0)
    if both.sum() >= 3:
        info["mfb"] = float(np.clip(g_mm[both].sum() / r_at[both].sum(), 0.2, 5.0))
    try:
        adj = wrl.adjust.AdjustMixed(obs_xy, raw_xy[have], nnear_raws=9, mingages=min_gauges, minval=0.5,
                                     nnearest=8, p=2.0)
        res = adj(g_mm, raw[have])
        if np.isfinite(res).mean() > 0.9:
            out[have] = np.where(np.isfinite(res), np.maximum(res, 0.0), raw[have])
            info["method"] = "wradlib.AdjustMixed"
    except Exception as e:  # keep the pipeline alive, record why
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    if info["method"] == "none" and info["mfb"] is not None:
        out[have] = raw[have] * info["mfb"]
        info["method"] = "mean-field-bias"
    # radar holes: gauges only
    if (~have).any() and g_mm.size >= min_gauges:
        try:
            from pykrige.ok import OrdinaryKriging
            ok = OrdinaryKriging(obs_xy[:, 0], obs_xy[:, 1], np.sqrt(g_mm), variogram_model="exponential",
                                 variogram_parameters={"sill": float(max(np.var(np.sqrt(g_mm)), 0.05)), "range": 25.0, "nugget": 0.02},
                                 enable_plotting=False)
            z, _ = ok.execute("points", raw_xy[~have, 0], raw_xy[~have, 1])
            out[~have] = np.maximum(np.asarray(z), 0.0) ** 2
            info["hole_fill"] = "kriging of gauges (sqrt-transformed)"
        except Exception as e:
            info["hole_fill_error"] = f"{type(e).__name__}: {e}"[:200]
    return out.reshape(radar_mm.shape).astype(np.float32), info
