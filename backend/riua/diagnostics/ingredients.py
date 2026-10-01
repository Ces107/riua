"""Ingredients-based heavy-rain diagnostics for the Valencia region — the "drivers" block of the Riuà product.

Heavy rain = high rain rate x long duration (Doswell, Brooks & Maddox 1996, Wea. Forecasting 11, 560-581):
``P = E * w * q * D``. Each family below measures one ingredient from real model fields:

1. moisture       PWAT (and its standardised anomaly, Hart & Grumm 2001), IVT, 925-hPa moisture-flux convergence,
                  850-hPa theta-e
2. instability    MUCAPE/MUCIN, MLCAPE (lowest 100 hPa), SBCAPE, LCL, LFC, EL, normalised CAPE, LI, K, TT
3. efficiency     freezing level, warm-cloud depth (freezing level - LCL), mean RH 700-500 hPa, cloud-layer RH
4. stationarity   Corfidi (2003) upwind/downwind vectors, cloud-layer wind, low-level jet, 0-6 km shear,
                  back-building flag
5. forcing        upslope flow w = V_low . grad(h) on the 0.05 deg Riuà terrain, upslope moisture flux
6. synoptic       closed/cut-off lows at 500 hPa (position, depth, track, "DANA favourable" flag), 300-hPa jet
7. a heuristic 0-1 "ingredients score" (display only — it never sets a risk level)

Data (Open-Meteo public S3 mirror, ``riua.sources.openmeteo_s3``; no API quota):

* ``meteofrance_arome_france0025`` (AROME 2.5 km, 24 pressure levels, hourly to +51 h): thermodynamics, moisture and
  wind diagnostics for the first 48 h. The 0.025 deg fields are block-averaged 5x5 to 0.125 deg, so that a column is
  the storm ENVIRONMENT and not the inside of one convective updraft of a convection-permitting model.
* ``ecmwf_ifs025`` (IFS 0.25 deg, 12 levels to 100 hPa, 3-hourly to +144 h, 6-hourly after): the same diagnostics for
  every time step (it is the only source after +48 h and fills AROME's hole south of ~38 deg N), the model's own
  ``total_column_integrated_water_vapour``, and the synoptic fields (Z500, T500, 300-hPa wind) on a 1 deg lattice.

How the numbers are computed:

* Gridded fields use a vectorised column engine written with MetPy's array-capable functions (``lcl``,
  ``moist_lapse`` -> pseudo-adiabat table, ``saturation_mixing_ratio``, ``virtual_temperature``,
  ``equivalent_potential_temperature``, ``dewpoint_from_relative_humidity``, ``k_index``, ``total_totals_index``,
  ``divergence``...). It follows MetPy's definitions (virtual-temperature CAPE integrated in ln p between the LFC and
  the EL, most-unstable parcel = highest theta-e in the lowest 300 hPa, mixed-layer parcel = lowest 100 hPa,
  Corfidi vectors with a pressure-weighted 850-300 hPa mean wind) on a 10-hPa grid that starts at the model surface.
* The anchor points (València, Alacant, Castelló, Gandia, Chiva) and :func:`point_profile` use MetPy's reference
  1-D functions (``most_unstable_cape_cin``, ``mixed_layer_cape_cin``, ``surface_based_cape_cin``, ``lcl``, ``lfc``,
  ``el``, ``parcel_profile``, ``corfidi_storm_motion``, ``bulk_shear``, ``precipitable_water``...). The scatter
  between the two is reported in ``checks`` at every run, together with MUCAPE against the models' own ``cape``.

Below-ground pressure levels are removed with a surface pressure obtained from the model's geopotential heights and
its orography (``data/<model>/static/HSURF.om``); the surface parcel is the 2 m temperature / humidity and 10 m wind.

Run:  ``python -m riua.diagnostics.ingredients``  (writes ``scratch/i1-ingredients/drivers_latest.json``).
"""
from __future__ import annotations

import heapq
import json
import math
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import metpy.calc as mpcalc
from metpy.units import units

from ..sources import openmeteo_s3 as om

__all__ = ["compute", "point_profile", "diagnose_columns", "metpy_sounding", "find_closed_lows", "jet_streak",
           "ingredients_score", "ANCHORS", "UNITS", "SCORE_DOC"]

# ----------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
GEO = Path(os.environ.get("RIUA_GEO", REPO / "geo"))
CLIMATE_FILE = Path(os.environ.get("RIUA_CLIMATE", REPO / "backend" / "riua" / "static" / "climate_era5.npz"))
CACHE_DIR = Path(os.environ.get("RIUA_DIAG_CACHE", Path(__file__).resolve().parent / "_static_cache"))

AROME = "meteofrance_arome_france0025"
IFS = "ecmwf_ifs025"
IFS_HRES = "ecmwf_ifs"

#: (west, south, east, north). Regional box = Riuà box + sea buffer to 2.5 E.
REGION = (-2.4, 37.6, 2.5, 41.0)
#: AROME window whose 5x5 block centres fall on multiples of 0.125 deg (every second one is an IFS 0.25 deg node).
AROME_BBOX = (-2.4251, 37.5749, 2.5501, 41.0501)
AROME_BLOCK = 5
#: Synoptic window read at 0.25 deg and subsampled to 1 deg (a little wider than the map so that lows near its
#: edge still have closed contours inside the grid).
SYN_BBOX = (-30.0, 24.0, 20.0, 56.0)
SYN_STEP = 4
SYN_MAP = (-24.0, 28.0, 14.0, 52.0, 2.0)  # west, south, east, north, step of the z500 map sent to the client

VLC = (39.47, -0.38)
ANCHORS = {
    "valencia": (39.47, -0.38), "alacant": (38.35, -0.49), "castello": (39.99, -0.04),
    "gandia": (38.97, -0.18), "chiva": (39.47, -0.72),
}

AROME_T_LEVELS = [1000, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]
AROME_W_LEVELS = [1000, 950, 925, 900, 850, 800, 750, 700, 600, 500, 400, 300]
IFS_T_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
IFS_W_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]
SFC_VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_component_10m", "wind_v_component_10m",
            "pressure_msl", "cape"]
IFS_EXTRA = ["total_column_integrated_water_vapour", "vertical_velocity_700hPa"]
SYN_VARS = ["geopotential_height_500hPa", "temperature_500hPa", "wind_u_component_300hPa",
            "wind_v_component_300hPa", "pressure_msl"]

G = 9.80665
RD = 287.047
KAPPA = 0.2857
EPSILON = 0.62196
DP = 10.0      # hPa, vertical step of the column engine
NK = 96        # levels of the column engine: p_k = p_sfc - 10 k
T0C = 273.15

HORIZON_FINE_H = 48
HORIZON_H = 168

UNITS = {
    "pwat": "mm", "pwat_anom": "sigma", "pwat_ifs": "mm", "ivt": "kg m-1 s-1", "ivt_dir": "deg (from)",
    "mfc925": "g kg-1 h-1 (positive = convergence)", "thetae850": "K",
    "mucape": "J kg-1", "mucin": "J kg-1", "mlcape": "J kg-1", "mlcin": "J kg-1", "sbcape": "J kg-1",
    "cape_model": "J kg-1", "lcl_agl": "m above ground (mixed-layer parcel)", "lfc_z": "m MSL (MU parcel)",
    "el_z": "m MSL (MU parcel)", "ncape": "m s-2", "li": "K", "kindex": "degC", "tt": "degC",
    "fzl": "m MSL", "wcd": "m", "rh_700_500": "%", "rh_cloud": "%",
    "corfidi_up": "m s-1", "corfidi_up_dir": "deg (from)", "corfidi_dn": "m s-1", "cl_speed": "m s-1",
    "cl_dir": "deg (from)", "llj_speed": "m s-1", "llj_dir": "deg (from)", "shear06": "m s-1",
    "backbuild": "fraction of strip points (anchors: 0/1)", "upslope_w": "m s-1",
    "upslope_qflux": "g kg-1 m s-1", "omega700": "Pa s-1 (IFS, negative = ascent)", "score": "0-1",
    "score_moisture": "0-1", "score_instability": "0-1", "score_efficiency": "0-1", "score_stationarity": "0-1",
    "score_forcing": "0-1", "z500": "dam", "sst": "degC",
}

SCORE_DOC = (
    "Heuristic summary, display only: geometric mean of five sub-scores in [0.05, 1]. "
    "moisture = mean(ramp(PWAT anomaly, +1..+3 sigma) [Hart & Grumm 2001] or ramp(PWAT, 25..40 mm) when no "
    "climatology is loaded, ramp(IVT, 250..600 kg/m/s) [Rutz et al. 2014 AR threshold; > 500 on 29-Oct-2024, "
    "Campos et al. 2025]); instability = ramp(MUCAPE, 50..800 J/kg) x ramp(MUCIN, -150..-50) [Doswell et al. 1996: "
    "instability only has to be sufficient; Mediterranean torrential rain often has modest, tall-skinny CAPE]; "
    "efficiency = mean(ramp(warm-cloud depth, 1500..4000 m) [Davis 2001], ramp(RH 700-500, 40..80 %)); "
    "stationarity = ramp-down(|Corfidi upwind vector|, 4..12 m/s) [Corfidi 2003]; forcing = mean(ramp(low-level jet "
    "from 045-135 deg, 6..18 m/s) [Homar et al. 2002; Pastor et al. 2010; Romero et al. 2000: persistent easterly "
    "low-level jet onto the Valencian ranges], ramp(upslope w, 0.03..0.25 m/s), ramp(925-hPa moisture-flux "
    "convergence, 0.5..3 g/kg/h)). Thresholds are round numbers chosen by the author from those sources, not fitted."
)


# ----------------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------------
def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _iso(t: datetime) -> str:
    return _utc(t).strftime("%Y-%m-%dT%H:%MZ")


def _ramp(x, lo, hi):
    """0 at ``lo``, 1 at ``hi`` (works for hi < lo: ramp-down)."""
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)


def _lst(a, nd=0, scale=1.0):
    """JSON list: rounded, NaN -> None, integers when nd == 0."""
    arr = np.asarray(a, float) / scale
    flat = arr.ravel()
    out = []
    for x in flat:
        if not np.isfinite(x):
            out.append(None)
        elif nd == 0:
            out.append(int(round(float(x))))
        else:
            out.append(round(float(x), nd))
    return out


def _num(x, nd=1):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return int(round(x)) if nd == 0 else round(x, nd)


def _wdir(u, v):
    """Meteorological direction (deg the wind comes FROM), MetPy."""
    return mpcalc.wind_direction(np.asarray(u, float) * units("m/s"), np.asarray(v, float) * units("m/s")).m_as("degree")


def _gc(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) and initial bearing (deg) from point 1 to point 2 (sphere, vectorised)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    brg = np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))
    return dist, (brg + 360.0) % 360.0


def _gc_dest(lat, lon, bearing_deg, dist_km):
    """Destination point on the sphere."""
    d = dist_km / 6371.0
    p1, b = math.radians(lat), np.radians(bearing_deg)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b))
    l2 = math.radians(lon) + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), np.degrees(l2)


_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _compass(deg):
    return _COMPASS[int(((deg % 360) + 11.25) // 22.5) % 16]


# ----------------------------------------------------------------------------------------------------------------
# Pseudo-adiabat table (built once with MetPy's moist_lapse)
# ----------------------------------------------------------------------------------------------------------------
class _MoistAdiabats:
    """T(p) along pseudo-adiabats labelled by their temperature at 1000 hPa, every 0.5 K, on a 5-hPa grid."""

    P_FIRST, P_STEP = 1060.0, 5.0

    def __init__(self):
        self.p = np.arange(self.P_FIRST, 84.0, -self.P_STEP)
        self.t1000 = np.arange(-45.0, 50.01, 0.5) + T0C
        self.tab = mpcalc.moist_lapse(self.p * units.hPa, self.t1000 * units.K, 1000.0 * units.hPa).m_as("K")

    def _at(self, i, p):
        x = np.clip((self.P_FIRST - p) / self.P_STEP, 0.0, self.p.size - 1.001)
        j = np.floor(x).astype(int)
        f = x - j
        return self.tab[i, j] * (1.0 - f) + self.tab[i, j + 1] * f

    def lift(self, p_lcl, t_lcl, p):
        """Temperature (K) at pressures ``p`` [N, K] of the pseudo-adiabat through (p_lcl, t_lcl) [N]."""
        nc = self.t1000.size
        col = self._at(np.arange(nc)[None, :], p_lcl[:, None])            # [N, nc] T of each curve at the LCL
        i = np.clip((col <= t_lcl[:, None]).sum(axis=1) - 1, 0, nc - 2)
        rows = np.arange(p_lcl.size)
        c0, c1 = col[rows, i], col[rows, i + 1]
        g = np.clip((t_lcl - c0) / (c1 - c0), 0.0, 1.0)[:, None]
        return self._at(i[:, None], p) * (1.0 - g) + self._at(i[:, None] + 1, p) * g


_TABLE: _MoistAdiabats | None = None


def _table() -> _MoistAdiabats:
    global _TABLE
    if _TABLE is None:
        _TABLE = _MoistAdiabats()
    return _TABLE


# ----------------------------------------------------------------------------------------------------------------
# Vectorised column engine
# ----------------------------------------------------------------------------------------------------------------
COLUMN_KEYS = [
    "psfc", "pwat", "ivt", "ivt_u", "ivt_v", "mucape", "mucin", "mlcape", "mlcin", "sbcape", "sbcin",
    "lcl_agl", "mu_lcl_z", "mu_p0", "lfc_z", "el_z", "lfc_p", "el_p", "ncape", "li", "kindex", "tt", "fzl", "wcd",
    "rh_700_500", "rh_cloud", "thetae850", "cl_u", "cl_v", "llj_u", "llj_v", "llj_speed", "cor_up_u", "cor_up_v",
    "corfidi_up", "corfidi_dn", "cl_speed", "shear06", "u_low", "v_low", "q_low", "backbuild",
]


def _interp_lnp(p_src, vals, ps, v_sfc, pf):
    """Linear interpolation in ln p of ``vals`` [N, L] (levels ``p_src`` hPa, descending) onto ``pf`` [N, K].

    The surface value ``v_sfc`` [N] at ``ps`` [N] is prepended and every level at or below the ground
    (``p_src >= ps - 0.5``) is discarded. Targets above the highest source level are NaN.
    """
    n, nl = vals.shape
    above = p_src[None, :] < (ps[:, None] - 0.5)
    lns = np.log(ps)[:, None]
    a = -np.concatenate([lns, np.where(above, np.log(p_src)[None, :], lns)], axis=1)       # ascending
    y = np.concatenate([v_sfc[:, None], np.where(above, vals, v_sfc[:, None])], axis=1)
    t = -np.log(pf)
    j = np.clip((a[:, None, :] <= t[:, :, None] + 1e-12).sum(axis=-1) - 1, 0, nl - 1)
    a0, a1 = np.take_along_axis(a, j, 1), np.take_along_axis(a, j + 1, 1)
    y0, y1 = np.take_along_axis(y, j, 1), np.take_along_axis(y, j + 1, 1)
    w = np.clip((t - a0) / np.where(a1 > a0, a1 - a0, 1.0), 0.0, 1.0)
    out = y0 + (y1 - y0) * w
    out[t > a[:, -1:] + 1e-9] = np.nan
    return out


def _frac(arr, x):
    """Sample ``arr`` [N, K] at fractional level index ``x`` [N] (linear); NaN outside 0..K-1 or for NaN x."""
    n, nk = arr.shape
    ok = np.isfinite(x) & (x >= 0) & (x <= nk - 1)
    xs = np.where(ok, x, 0.0)
    k = np.clip(np.floor(xs).astype(int), 0, nk - 2)
    f = xs - k
    rows = np.arange(n)
    out = arr[rows, k] * (1.0 - f) + arr[rows, k + 1] * f
    # exact hits on a level must not be poisoned by a NaN neighbour
    out = np.where(f < 1e-9, arr[rows, k], out)
    return np.where(ok, out, np.nan)


def surface_pressure(tl, z, t, zs, t2=None):
    """Surface pressure (hPa) from geopotential heights ``z`` [N, L] (levels ``tl`` descending in pressure) and the
    model orography ``zs`` [N]: ln p interpolated linearly in z between the bracketing levels; hypsometric
    extrapolation with the mean of the lowest-level and 2 m temperatures when the ground is below the lowest level.
    """
    tl = np.asarray(tl, float)
    n, nl = z.shape
    rows = np.arange(n)
    lnp = np.log(tl)
    nb = (z <= zs[:, None]).sum(axis=1)
    i1 = np.clip(nb, 1, nl - 1)
    i0 = i1 - 1
    z0, z1 = z[rows, i0], z[rows, i1]
    w = (zs - z0) / np.where(z1 > z0, z1 - z0, 1.0)
    ps_int = np.exp(lnp[i0] + w * (lnp[i1] - lnp[i0]))
    t_low = t[:, 0]
    t_sfc = t_low + 0.0065 * (z[:, 0] - zs)
    if t2 is not None:
        t_sfc = np.where(np.isfinite(t2), t2, t_sfc)
    tm = 0.5 * (t_low + t_sfc) * 1.006          # ~virtual temperature of a moist boundary layer
    ps_ext = tl[0] * np.exp(G * (z[:, 0] - zs) / (RD * tm))
    return np.where(nb == 0, ps_ext, ps_int)


def _lift(pf, lnp, valid, tve, ze, ps, p0, t0, td0, k0):
    """Lift parcels (p0, t0, td0) [N] starting at level index k0 [N]; MetPy's cape_cin definitions."""
    n, nk = pf.shape
    rows = np.arange(n)
    karr = np.arange(nk)[None, :]
    p_lcl_q, t_lcl_q = mpcalc.lcl(p0 * units.hPa, t0 * units.K, td0 * units.K)
    p_lcl = np.minimum(p_lcl_q.m_as("hPa"), p0)
    t_lcl = t_lcl_q.m_as("K")
    r0 = mpcalc.saturation_mixing_ratio(p0 * units.hPa, td0 * units.K).m_as("")
    dry = t0[:, None] * (pf / p0[:, None]) ** KAPPA
    moist = _table().lift(p_lcl, t_lcl, pf)
    below = pf >= p_lcl[:, None]
    tp = np.where(below, dry, moist)
    rp = np.where(below, r0[:, None], mpcalc.saturation_mixing_ratio(pf * units.hPa, tp * units.K).m_as(""))
    tvp = tp * (1.0 + rp / EPSILON) / (1.0 + rp)
    act = valid & (karr >= k0[:, None])
    b = np.where(act, tvp - tve, np.nan)
    bz = np.nan_to_num(b)
    k_top = np.where(act, karr, -1).max(axis=1)                 # last active level

    x_lcl = np.clip((ps - p_lcl) / DP, k0, None)
    b_lcl = _frac(b, np.minimum(x_lcl, k_top))

    b0, b1 = b[:, :-1], b[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        xc = karr[:, :-1] + b0 / (b0 - b1)                      # zero-crossing position inside each layer
        up = (b0 <= 0) & (b1 > 0) & (xc >= x_lcl[:, None])
        x_up = np.where(up, xc, np.inf).min(axis=1)
        x_lfc = np.where(b_lcl > 0, x_lcl, x_up)
        dn = (b0 > 0) & (b1 <= 0) & (xc > x_lfc[:, None])
        x_dn = np.where(dn, xc, -np.inf).max(axis=1)
        top_pos = b[rows, np.maximum(k_top, 0)] > 0
        x_el = np.where(top_pos, k_top.astype(float), x_dn)
    has = np.isfinite(x_lfc) & np.isfinite(x_el) & (x_el > x_lfc)

    dln = lnp[:, :-1] - lnp[:, 1:]
    seg = RD * 0.5 * (bz[:, :-1] + bz[:, 1:]) * np.where(np.isfinite(b0) & np.isfinite(b1), dln, 0.0)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, axis=1)], axis=1)

    def integral(x):
        xs = np.where(np.isfinite(x), x, 0.0)
        k = np.clip(np.floor(xs).astype(int), 0, nk - 2)
        f = xs - k
        bk, bk1 = bz[rows, k], bz[rows, k + 1]
        return cum[rows, k] + RD * (bk * f + 0.5 * (bk1 - bk) * f * f) * dln[rows, k]

    i_lfc, i_el, i_start = integral(x_lfc), integral(x_el), integral(k0.astype(float))
    cape = np.where(has, np.maximum(i_el - i_lfc, 0.0), 0.0)
    cin = np.where(has, np.minimum(i_lfc - i_start, 0.0), np.nan)
    x_lfc = np.where(has, x_lfc, np.nan)
    x_el = np.where(has, x_el, np.nan)
    return {
        "cape": cape, "cin": cin, "p_lcl": p_lcl, "z_lcl": _frac(ze, x_lcl), "x_lcl": x_lcl,
        "p_lfc": ps - DP * x_lfc, "z_lfc": _frac(ze, x_lfc), "p_el": ps - DP * x_el, "z_el": _frac(ze, x_el),
        "tp": tp,
    }


def diagnose_columns(c: dict) -> dict:
    """Ingredient diagnostics of N columns at once.

    ``c`` holds: ``tl`` [Lt] pressure levels (hPa, descending) of ``t`` (K), ``rh`` (%), ``z`` (gpm), each [N, Lt];
    ``wl`` [Lw] levels of ``u``, ``v`` (m/s) [N, Lw]; surface ``t2`` (K), ``rh2`` (%), ``u10``, ``v10`` (m/s) and
    model orography ``zs`` (m) [N]. Optional ``ps`` (hPa) overrides the surface pressure.
    Returns ``{name: array [N]}`` for the names in ``COLUMN_KEYS`` (NaN where the column is unusable).
    """
    tl = np.asarray(c["tl"], float)
    t, rh, z = (np.asarray(c[k], float) for k in ("t", "rh", "z"))
    n = t.shape[0]
    out = {k: np.full(n, np.nan) for k in COLUMN_KEYS}
    keep = np.isfinite(t).any(0) & np.isfinite(rh).any(0) & np.isfinite(z).any(0)
    tl, t, rh, z = tl[keep], t[:, keep], rh[:, keep], z[:, keep]
    wl = np.asarray(c.get("wl", []), float)
    u = np.asarray(c.get("u", np.empty((n, 0))), float)
    v = np.asarray(c.get("v", np.empty((n, 0))), float)
    if wl.size:
        keepw = np.isfinite(u).any(0) & np.isfinite(v).any(0)
        wl, u, v = wl[keepw], u[:, keepw], v[:, keepw]
    have_wind = wl.size >= 4
    zs = np.asarray(c["zs"], float)
    good = np.isfinite(t).all(1) & np.isfinite(rh).all(1) & np.isfinite(z).all(1) & np.isfinite(zs)
    if have_wind:
        good &= np.isfinite(u).all(1) & np.isfinite(v).all(1)
    if tl.size < 6 or not good.any():
        return out
    sub = {"tl": tl, "t": t[good], "rh": rh[good], "z": z[good], "zs": zs[good]}
    for k in ("t2", "rh2", "u10", "v10", "ps"):
        a = c.get(k)
        sub[k] = np.full(int(good.sum()), np.nan) if a is None else np.asarray(a, float)[good]
    if have_wind:
        sub.update(wl=wl, u=u[good], v=v[good])
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        res = _engine(sub, have_wind)
    for k, val in res.items():
        out[k][good] = val
    return out


def _engine(c, have_wind):
    tl, t, rh, z, zs = c["tl"], c["t"], np.clip(c["rh"], 1.0, 100.0), c["z"], c["zs"]
    n = t.shape[0]
    rows = np.arange(n)
    ps = np.where(np.isfinite(c["ps"]), c["ps"], surface_pressure(tl, z, t, zs, c["t2"]))
    ps = np.clip(ps, 500.0, 1080.0)
    above = tl[None, :] < ps[:, None] - 0.5
    first = above.argmax(axis=1)
    t_s = np.where(np.isfinite(c["t2"]), c["t2"], t[rows, first])
    rh_s = np.clip(np.where(np.isfinite(c["rh2"]), c["rh2"], rh[rows, first]), 1.0, 100.0)
    td_l = mpcalc.dewpoint_from_relative_humidity(t * units.K, rh * units.percent).m_as("K")
    td_s = mpcalc.dewpoint_from_relative_humidity(t_s * units.K, rh_s * units.percent).m_as("K")

    p_top = float(tl.min())
    pf_raw = ps[:, None] - DP * np.arange(NK)[None, :]
    valid = pf_raw >= p_top - 1e-6
    pf = np.maximum(pf_raw, p_top)
    lnp = np.log(pf)
    te = _interp_lnp(tl, t, ps, t_s, pf)
    tde = np.minimum(_interp_lnp(tl, td_l, ps, td_s, pf), te)
    ze = _interp_lnp(tl, z, ps, zs, pf)
    r = mpcalc.saturation_mixing_ratio(pf * units.hPa, tde * units.K).m_as("")
    q = r / (1.0 + r)
    tve = mpcalc.virtual_temperature(te * units.K, r * units("")).m_as("K")
    rhe = 100.0 * mpcalc.relative_humidity_from_dewpoint(te * units.K, tde * units.K).m_as("")
    o: dict[str, np.ndarray] = {"psfc": ps}

    # ---- moisture ------------------------------------------------------------------------------------------
    both = valid[:, :-1] & valid[:, 1:]
    o["pwat"] = (np.where(both, 0.5 * (q[:, :-1] + q[:, 1:]), 0.0).sum(axis=1)) * DP * 100.0 / G

    # ---- parcels -------------------------------------------------------------------------------------------
    nmu = 31  # lowest 300 hPa
    thetae = mpcalc.equivalent_potential_temperature(pf[:, :nmu] * units.hPa, te[:, :nmu] * units.K,
                                                     tde[:, :nmu] * units.K).m_as("K")
    k_mu = np.where(valid[:, :nmu], thetae, -np.inf).argmax(axis=1)
    mu = _lift(pf, lnp, valid, tve, ze, ps, pf[rows, k_mu], te[rows, k_mu], tde[rows, k_mu], k_mu)
    zero = np.zeros(n, dtype=int)
    sb = _lift(pf, lnp, valid, tve, ze, ps, ps, te[:, 0], tde[:, 0], zero)
    nml = 11  # lowest 100 hPa, trapezoid mean (equal spacing)
    wts = np.ones(nml)
    wts[[0, -1]] = 0.5
    wts /= wts.sum()
    theta_ml = ((te[:, :nml] * (1000.0 / pf[:, :nml]) ** KAPPA) * wts).sum(axis=1)
    r_ml = (r[:, :nml] * wts).sum(axis=1)
    t_ml = theta_ml * (ps / 1000.0) ** KAPPA
    e_ml = mpcalc.vapor_pressure(ps * units.hPa, r_ml * units(""))
    td_ml = np.minimum(mpcalc.dewpoint(e_ml).m_as("K"), t_ml)
    ml = _lift(pf, lnp, valid, tve, ze, ps, ps, t_ml, td_ml, zero)

    o["mucape"], o["mucin"], o["mlcape"], o["mlcin"] = mu["cape"], mu["cin"], ml["cape"], ml["cin"]
    o["sbcape"], o["sbcin"] = sb["cape"], sb["cin"]
    o["lcl_agl"] = ml["z_lcl"] - zs
    o["mu_lcl_z"], o["mu_p0"] = mu["z_lcl"], pf[rows, k_mu]
    o["lfc_z"], o["el_z"], o["lfc_p"], o["el_p"] = mu["z_lfc"], mu["z_el"], mu["p_lfc"], mu["p_el"]
    depth = mu["z_el"] - mu["z_lfc"]
    o["ncape"] = np.where((mu["cape"] >= 50.0) & (depth > 500.0), mu["cape"] / depth, np.nan)
    x500 = (ps - 500.0) / DP
    o["li"] = _frac(te, x500) - _frac(ml["tp"], x500)

    def at_p(arr, p):
        return np.where(ps >= p, _frac(arr, (ps - p) / DP), np.nan)

    t3 = np.stack([at_p(te, 850.0), at_p(te, 700.0), at_p(te, 500.0)])
    td3 = np.stack([at_p(tde, 850.0), at_p(tde, 700.0), at_p(tde, 500.0)])
    p3 = np.repeat(np.array([850.0, 700.0, 500.0])[:, None], n, axis=1) * units.hPa
    o["kindex"] = mpcalc.k_index(p3, t3 * units.K, td3 * units.K, vertical_dim=0).m_as("degC")
    o["tt"] = mpcalc.total_totals_index(p3, t3 * units.K, td3 * units.K, vertical_dim=0).m_as("delta_degC")
    o["thetae850"] = mpcalc.equivalent_potential_temperature(850.0 * units.hPa, t3[0] * units.K,
                                                             td3[0] * units.K).m_as("K")

    # ---- warm-rain efficiency ------------------------------------------------------------------------------
    cross = (te[:, :-1] > T0C) & (te[:, 1:] <= T0C) & valid[:, 1:]
    kf = cross.argmax(axis=1)
    xf = kf + (te[rows, kf] - T0C) / (te[rows, kf] - te[rows, kf + 1])
    fzl = np.where(cross.any(axis=1), _frac(ze, xf), np.nan)
    o["fzl"] = np.where(te[:, 0] <= T0C, zs, fzl)
    o["wcd"] = np.maximum(o["fzl"] - mu["z_lcl"], 0.0)
    m75 = valid & (pf <= 700.0) & (pf >= 500.0)
    o["rh_700_500"] = np.where(m75.sum(1) >= 5, (rhe * m75).sum(1) / np.maximum(m75.sum(1), 1), np.nan)
    mcl = valid & (pf <= mu["p_lcl"][:, None]) & (pf >= 500.0)
    o["rh_cloud"] = np.where(mcl.sum(1) >= 5, (rhe * mcl).sum(1) / np.maximum(mcl.sum(1), 1), np.nan)

    # ---- winds ---------------------------------------------------------------------------------------------
    if have_wind:
        wl, u, v = c["wl"], c["u"], c["v"]
        abw = wl[None, :] < ps[:, None] - 0.5
        firstw = abw.argmax(axis=1)
        u_s = np.where(np.isfinite(c["u10"]), c["u10"], u[rows, firstw])
        v_s = np.where(np.isfinite(c["v10"]), c["v10"], v[rows, firstw])
        ue = _interp_lnp(wl, u, ps, u_s, pf)
        ve = _interp_lnp(wl, v, ps, v_s, pf)
        okw = np.isfinite(ue) & valid
        m300 = okw & (pf >= 300.0)
        b3 = m300[:, :-1] & m300[:, 1:]
        qu, qv = q * np.nan_to_num(ue), q * np.nan_to_num(ve)
        o["ivt_u"] = np.where(b3, 0.5 * (qu[:, :-1] + qu[:, 1:]), 0.0).sum(axis=1) * DP * 100.0 / G
        o["ivt_v"] = np.where(b3, 0.5 * (qv[:, :-1] + qv[:, 1:]), 0.0).sum(axis=1) * DP * 100.0 / G
        o["ivt"] = np.hypot(o["ivt_u"], o["ivt_v"])
        # cloud-layer wind: pressure-weighted mean 850-300 hPa (MetPy mean_pressure_weighted)
        mc = okw & (pf <= np.minimum(850.0, ps)[:, None] + 1e-6) & (pf >= 300.0 - 1e-6)
        wsum = np.maximum((pf * mc).sum(1), 1e-9)
        o["cl_u"] = (np.nan_to_num(ue) * pf * mc).sum(1) / wsum
        o["cl_v"] = (np.nan_to_num(ve) * pf * mc).sum(1) / wsum
        o["cl_speed"] = np.hypot(o["cl_u"], o["cl_v"])
        # low-level jet: strongest wind from the surface to 850 hPa (lowest 100 hPa over high ground)
        p_llj = np.where(ps >= 900.0, 850.0, ps - 100.0)
        ml_ = okw & (pf >= p_llj[:, None] - 1e-6)
        spd = np.where(ml_, np.hypot(ue, ve), -1.0)
        kj = spd.argmax(axis=1)
        o["llj_u"], o["llj_v"] = ue[rows, kj], ve[rows, kj]
        o["llj_speed"] = np.hypot(o["llj_u"], o["llj_v"])
        o["cor_up_u"], o["cor_up_v"] = o["cl_u"] - o["llj_u"], o["cl_v"] - o["llj_v"]
        o["corfidi_up"] = np.hypot(o["cor_up_u"], o["cor_up_v"])
        o["corfidi_dn"] = np.hypot(2 * o["cl_u"] - o["llj_u"], 2 * o["cl_v"] - o["llj_v"])
        # 0-6 km bulk shear
        zt = zs + 6000.0
        k6 = np.clip((np.where(valid, ze, np.inf) < zt[:, None]).sum(axis=1) - 1, 0, NK - 2)
        f6 = np.clip((zt - ze[rows, k6]) / (ze[rows, k6 + 1] - ze[rows, k6]), 0.0, 1.0)
        u6, v6 = _frac(ue, k6 + f6), _frac(ve, k6 + f6)
        o["shear06"] = np.hypot(u6 - ue[:, 0], v6 - ve[:, 0])
        # low-level flow that meets the terrain: 925 hPa, or 30 hPa above the model ground where it is higher
        x_low = (ps - np.minimum(925.0, ps - 30.0)) / DP
        o["u_low"], o["v_low"], o["q_low"] = _frac(ue, x_low), _frac(ve, x_low), 1000.0 * _frac(q, x_low)
        llj_dir = _wdir(o["llj_u"], o["llj_v"])
        o["backbuild"] = ((o["corfidi_up"] < 5.0) & (o["llj_speed"] >= 10.0)
                          & (llj_dir >= 45.0) & (llj_dir <= 135.0)).astype(float)
    return o


# ----------------------------------------------------------------------------------------------------------------
# Reference 1-D diagnostics with MetPy's own functions
# ----------------------------------------------------------------------------------------------------------------
def metpy_sounding(p, t, td, z, pw=None, u=None, v=None, zw=None) -> dict:
    """Sounding diagnostics of ONE profile with MetPy's reference functions.

    ``p`` (hPa, surface first, descending), ``t``/``td`` (K), ``z`` (m MSL); optional wind profile ``pw`` (hPa),
    ``u``, ``v`` (m/s), ``zw`` (m MSL) on its own levels (surface first). Returns plain floats (NaN if a function
    could not be evaluated) plus the MU parcel temperature profile under ``"parcel_mu_t"`` / ``"parcel_mu_p"``.
    """
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    td = np.minimum(np.asarray(td, float), t)
    z = np.asarray(z, float)
    P, T, TD = p * units.hPa, t * units.K, td * units.K
    o: dict = {}

    def zat(pq):
        pv = pq.m_as("hPa") if hasattr(pq, "m_as") else float(pq)
        return float(np.interp(-math.log(pv), -np.log(p), z)) if np.isfinite(pv) else float("nan")

    def safe(name, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - MetPy raises on degenerate profiles; record and go on
            o.setdefault("errors", []).append(f"{name}: {type(exc).__name__}: {exc}"[:120])
            return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = safe("most_unstable_cape_cin", lambda: mpcalc.most_unstable_cape_cin(P, T, TD))
        o["mucape"], o["mucin"] = (r[0].m_as("J/kg"), r[1].m_as("J/kg")) if r else (np.nan, np.nan)
        r = safe("mixed_layer_cape_cin", lambda: mpcalc.mixed_layer_cape_cin(P, T, TD, depth=100 * units.hPa))
        o["mlcape"], o["mlcin"] = (r[0].m_as("J/kg"), r[1].m_as("J/kg")) if r else (np.nan, np.nan)
        r = safe("surface_based_cape_cin", lambda: mpcalc.surface_based_cape_cin(P, T, TD))
        o["sbcape"], o["sbcin"] = (r[0].m_as("J/kg"), r[1].m_as("J/kg")) if r else (np.nan, np.nan)

        # most-unstable parcel: LCL, LFC, EL and its profile
        for k in ("mu_p0", "mu_lcl_p", "mu_lcl_z", "lfc_p", "lfc_z", "el_p", "el_z", "ncape"):
            o[k] = np.nan
        mup = safe("most_unstable_parcel", lambda: mpcalc.most_unstable_parcel(P, T, TD))
        if mup:
            i0 = int(mup[3])
            o["mu_p0"] = mup[0].m_as("hPa")
            lc = safe("lcl", lambda: mpcalc.lcl(mup[0], mup[1], mup[2]))
            if lc:
                o["mu_lcl_p"], o["mu_lcl_z"] = lc[0].m_as("hPa"), zat(lc[0])
            prof = safe("parcel_profile", lambda: mpcalc.parcel_profile(P[i0:], mup[1], mup[2]))
            if prof is not None:
                o["parcel_mu_p"], o["parcel_mu_t"] = p[i0:].tolist(), prof.m_as("K").tolist()
                lf = safe("lfc", lambda: mpcalc.lfc(P[i0:], T[i0:], TD[i0:], prof, which="bottom"))
                if lf and np.isfinite(lf[0].m):
                    o["lfc_p"], o["lfc_z"] = lf[0].m_as("hPa"), zat(lf[0])
                e = safe("el", lambda: mpcalc.el(P[i0:], T[i0:], TD[i0:], prof, which="top"))
                if e and np.isfinite(e[0].m):
                    o["el_p"], o["el_z"] = e[0].m_as("hPa"), zat(e[0])
                dz = o["el_z"] - o["lfc_z"]
                if np.isfinite(dz) and dz > 500 and o["mucape"] >= 50:
                    o["ncape"] = o["mucape"] / dz
        # mixed-layer parcel: cloud base and lifted index
        o["lcl_agl"] = o["li"] = np.nan
        mlp = safe("mixed_parcel", lambda: mpcalc.mixed_parcel(P, T, TD, depth=100 * units.hPa))
        if mlp:
            lc = safe("lcl_ml", lambda: mpcalc.lcl(mlp[0], mlp[1], mlp[2]))
            if lc:
                o["lcl_agl"] = zat(lc[0]) - float(z[0])
            if p.min() <= 500.0:
                mlprof = safe("parcel_profile_ml", lambda: mpcalc.parcel_profile(P, mlp[1], mlp[2]))
                if mlprof is not None:
                    li = safe("lifted_index", lambda: mpcalc.lifted_index(P, T, mlprof))
                    if li is not None:
                        o["li"] = float(np.atleast_1d(li.m_as("delta_degC"))[0])
        has_std = all(np.any(np.isclose(p, lv)) for lv in (850.0, 700.0, 500.0))
        ki = safe("k_index", lambda: mpcalc.k_index(P, T, TD)) if has_std else None
        o["kindex"] = ki.m_as("degC") if ki is not None else np.nan
        tti = safe("total_totals_index", lambda: mpcalc.total_totals_index(P, T, TD)) if has_std else None
        o["tt"] = tti.m_as("delta_degC") if tti is not None else np.nan
        pw_ = safe("precipitable_water", lambda: mpcalc.precipitable_water(P, TD))
        o["pwat"] = pw_.m_as("mm") if pw_ is not None else np.nan
        # freezing level and warm-cloud depth
        o["fzl"] = np.nan
        if t[0] <= T0C:
            o["fzl"] = float(z[0])
        else:
            idx = np.nonzero((t[:-1] > T0C) & (t[1:] <= T0C))[0]
            if idx.size:
                k = int(idx[0])
                o["fzl"] = float(z[k] + (t[k] - T0C) / (t[k] - t[k + 1]) * (z[k + 1] - z[k]))
        o["wcd"] = max(o["fzl"] - o["mu_lcl_z"], 0.0) if np.isfinite(o["fzl"] - o["mu_lcl_z"]) else np.nan
        rh = mpcalc.relative_humidity_from_dewpoint(T, TD).m_as("percent")
        lay = safe("rh_700_500", lambda: mpcalc.mean_pressure_weighted(P, rh * units.percent, bottom=700 * units.hPa,
                                                                       depth=200 * units.hPa))
        o["rh_700_500"] = lay[0].m_as("percent") if lay and p[0] >= 700 else np.nan

        for k in ("corfidi_up", "corfidi_up_dir", "corfidi_dn", "cl_speed", "cl_dir", "llj_speed", "llj_dir",
                  "shear06"):
            o[k] = np.nan
        if pw is not None and len(pw) >= 4:
            pwq = np.asarray(pw, float) * units.hPa
            uq, vq = np.asarray(u, float) * units("m/s"), np.asarray(v, float) * units("m/s")
            cf = safe("corfidi_storm_motion", lambda: mpcalc.corfidi_storm_motion(pwq, uq, vq))
            if cf:
                (uu, vu), (ud, vd) = cf[0].m_as("m/s"), cf[1].m_as("m/s")
                o["corfidi_up"], o["corfidi_dn"] = float(np.hypot(uu, vu)), float(np.hypot(ud, vd))
                o["corfidi_up_dir"] = float(_wdir(uu, vu))
                ucl, vcl = ud - uu, vd - vu
                o["cl_speed"], o["cl_dir"] = float(np.hypot(ucl, vcl)), float(_wdir(ucl, vcl))
                ul, vl = ucl - uu, vcl - vu
                o["llj_speed"], o["llj_dir"] = float(np.hypot(ul, vl)), float(_wdir(ul, vl))
            if zw is not None:
                hq = (np.asarray(zw, float) - float(zw[0])) * units.m
                sh = safe("bulk_shear", lambda: mpcalc.bulk_shear(pwq, uq, vq, height=hq, depth=6000 * units.m))
                if sh:
                    o["shear06"] = float(np.hypot(sh[0].m_as("m/s"), sh[1].m_as("m/s")))
    return {k: (float(val) if isinstance(val, (float, np.floating, int)) else val) for k, val in o.items()}


def _column_sounding(c, n, ps):
    """1-D sounding (surface first) of column ``n`` of a column dict; None if unusable."""
    tl, wl = np.asarray(c["tl"], float), np.asarray(c["wl"], float)
    t, rh, z = c["t"][n], np.clip(c["rh"][n], 1, 100), c["z"][n]
    lev_ok = np.isfinite(t) & np.isfinite(rh) & np.isfinite(z)          # a level missing in the files is dropped
    tl, t, rh, z = tl[lev_ok], t[lev_ok], rh[lev_ok], z[lev_ok]
    if not np.isfinite(ps) or tl.size < 6:
        return None
    m = tl < ps - 0.5
    td = mpcalc.dewpoint_from_relative_humidity(t * units.K, rh * units.percent).m_as("K")
    t2, rh2 = c["t2"][n], c["rh2"][n]
    if np.isfinite(t2) and np.isfinite(rh2):
        td2 = float(mpcalc.dewpoint_from_relative_humidity(t2 * units.K, np.clip(rh2, 1, 100) * units.percent).m_as("K"))
        p = np.r_[ps, tl[m]]
        tt, tdd, zz = np.r_[t2, t[m]], np.r_[td2, td[m]], np.r_[c["zs"][n], z[m]]
    else:
        p, tt, tdd, zz = tl[m], t[m], td[m], z[m]
    s = {"p": p, "t": tt, "td": tdd, "z": zz}
    u, v = c["u"][n], c["v"][n]
    w_ok = np.isfinite(u) & np.isfinite(v)
    wl, u, v = wl[w_ok], u[w_ok], v[w_ok]
    if wl.size >= 4:
        mw = wl < ps - 0.5
        zw = np.interp(-np.log(wl[mw]), -np.log(tl), z)
        if np.isfinite(c["u10"][n]) and np.isfinite(c["v10"][n]):
            s.update(pw=np.r_[ps, wl[mw]], u=np.r_[c["u10"][n], u[mw]], v=np.r_[c["v10"][n], v[mw]],
                     zw=np.r_[c["zs"][n], zw])
        else:
            s.update(pw=wl[mw], u=u[mw], v=v[mw], zw=zw)
    return s


def _exact_job(s):
    if s is None:
        return None
    return metpy_sounding(s["p"], s["t"], s["td"], s["z"], s.get("pw"), s.get("u"), s.get("v"), s.get("zw"))


# ----------------------------------------------------------------------------------------------------------------
# Synoptic diagnostics
# ----------------------------------------------------------------------------------------------------------------
def _sample(field, lats, lons, lat, lon):
    """Bilinear sample of a regular field at points; NaN outside."""
    fy = (np.asarray(lat, float) - lats[0]) / (lats[1] - lats[0])
    fx = (np.asarray(lon, float) - lons[0]) / (lons[1] - lons[0])
    ok = (fy >= 0) & (fy <= len(lats) - 1) & (fx >= 0) & (fx <= len(lons) - 1)
    j = np.clip(np.floor(fy).astype(int), 0, len(lats) - 2)
    i = np.clip(np.floor(fx).astype(int), 0, len(lons) - 2)
    wy, wx = np.clip(fy - j, 0, 1), np.clip(fx - i, 0, 1)
    val = (field[j, i] * (1 - wy) * (1 - wx) + field[j, i + 1] * (1 - wy) * wx
           + field[j + 1, i] * wy * (1 - wx) + field[j + 1, i + 1] * wy * wx)
    return np.where(ok, val, np.nan)


def _closed_depth(z, j0, i0, dist_km, max_km):
    """Depth (gpm) of the deepest closed contour around the minimum (j0, i0): priority flood from the minimum
    until the water reaches the edge of the grid, a point farther than ``max_km`` or a deeper basin."""
    ny, nx = z.shape
    z0 = float(z[j0, i0])
    level = z0
    seen = np.zeros(z.shape, bool)
    seen[j0, i0] = True
    heap: list = []

    def push(j, i):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and not seen[jj, ii]:
                    seen[jj, ii] = True
                    heapq.heappush(heap, (float(z[jj, ii]), jj, ii))

    push(j0, i0)
    area = 1
    while heap:
        zc, j, i = heapq.heappop(heap)
        if zc < z0:
            break
        level = max(level, zc)
        if j in (0, ny - 1) or i in (0, nx - 1) or dist_km[j, i] > max_km:
            break
        area += 1
        push(j, i)
    return level - z0, area


def find_closed_lows(z500, lats, lons, t500=None, min_depth=30.0, ring_deg=6.0, max_radius_km=1500.0,
                     ref=VLC) -> list[dict]:
    """Closed lows at 500 hPa on a regular (about 1 deg) grid.

    A low is a grid minimum (5x5 neighbourhood) surrounded by at least one closed height contour ``min_depth`` gpm
    above it that stays within ``max_radius_km`` of the centre (closed-contour depth by priority flood = the
    "prominence" of the minimum). Also returned: the ring depth (lowest and mean height on a great-circle ring of
    radius ``ring_deg`` minus the centre; lowest >= min_depth means every point of the ring is higher — the strict
    cut-off criterion), the 500-hPa temperature anomaly of the core against the ring mean (cold core < -1 K), and
    distance / bearing from ``ref`` (València).
    """
    from scipy import ndimage

    z = np.asarray(z500, float)
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    if not np.isfinite(z).all():
        return []
    zmin = ndimage.minimum_filter(z, size=5, mode="nearest")
    cand = np.argwhere((z == zmin))
    out = []
    lon2, lat2 = np.meshgrid(lons, lats)
    az = np.arange(0.0, 360.0, 15.0)
    for j0, i0 in cand:
        if j0 < 2 or i0 < 2 or j0 > len(lats) - 3 or i0 > len(lons) - 3:
            continue
        dist, _ = _gc(lats[j0], lons[i0], lat2, lon2)
        depth, area = _closed_depth(z, int(j0), int(i0), dist, max_radius_km)
        if depth < min_depth:
            continue
        rl, ro = _gc_dest(lats[j0], lons[i0], az, ring_deg * 111.195)
        ring = _sample(z, lats, lons, rl, ro)
        inside = np.isfinite(ring)
        d, b = _gc(ref[0], ref[1], lats[j0], lons[i0])
        low = {"lat": float(lats[j0]), "lon": float(lons[i0]), "z500": float(z[j0, i0]), "depth": float(depth),
               "area_cells": int(area),
               "ring_min_depth": float(ring[inside].min() - z[j0, i0]) if inside.sum() >= 18 else None,
               "ring_mean_depth": float(ring[inside].mean() - z[j0, i0]) if inside.sum() >= 18 else None,
               "dist_km": float(d), "bearing": float(b), "sector": _compass(float(b))}
        if t500 is not None and np.isfinite(t500).all():
            tr = _sample(np.asarray(t500, float), lats, lons, rl, ro)
            low["t500"] = float(t500[j0, i0])
            low["t500_anom"] = float(t500[j0, i0] - np.nanmean(tr)) if np.isfinite(tr).sum() >= 18 else None
            low["cold_core"] = bool(low["t500_anom"] is not None and low["t500_anom"] < -1.0)
        low["cutoff"] = bool(low["ring_min_depth"] is not None and low["ring_min_depth"] >= min_depth)
        out.append(low)
    out.sort(key=lambda q: q["dist_km"])
    # one centre per system (two minima of the same closed low closer than 500 km: keep the deeper)
    kept: list[dict] = []
    for low in sorted(out, key=lambda q: -q["depth"]):
        if all(_gc(low["lat"], low["lon"], k["lat"], k["lon"])[0] > 500.0 for k in kept):
            kept.append(low)
    kept.sort(key=lambda q: q["dist_km"])
    return kept


def jet_streak(u300, v300, lats, lons, ref=VLC, min_speed=30.0, max_km=2500.0):
    """Strongest 300-hPa wind within ``max_km`` of ``ref`` and the quadrant of the streak in which ``ref`` lies.

    Left-exit and right-entrance are the quadrants of upper-level divergence (ascent) of a straight jet streak.
    """
    spd = np.hypot(u300, v300)
    lon2, lat2 = np.meshgrid(lons, lats)
    dist, _ = _gc(ref[0], ref[1], lat2, lon2)
    s = np.where(dist <= max_km, spd, -1.0)
    if not np.isfinite(s).any():
        return None
    j, i = np.unravel_index(int(np.nanargmax(s)), s.shape)
    smax = float(spd[j, i])
    d, b_to_core = _gc(ref[0], ref[1], lats[j], lons[i])
    res = {"speed": smax, "lat": float(lats[j]), "lon": float(lons[i]), "dist_km": float(d), "bearing": float(b_to_core),
           "dir": float(_wdir(u300[j, i], v300[j, i])), "quadrant": None, "favourable": False}
    if smax >= min_speed:
        d2, b2 = _gc(lats[j], lons[i], ref[0], ref[1])           # from the core to the reference point
        dx, dy = d2 * math.sin(math.radians(b2)), d2 * math.cos(math.radians(b2))
        ux, uy = u300[j, i] / smax, v300[j, i] / smax
        along, left = dx * ux + dy * uy, ux * dy - uy * dx
        res["quadrant"] = ("left" if left > 0 else "right") + "-" + ("exit" if along > 0 else "entrance")
        res["favourable"] = bool(res["quadrant"] in ("left-exit", "right-entrance") and d <= 1200.0)
    return res


def _link_tracks(lows_per_time, times, max_km=800.0, max_gap_h=12.0):
    """Give every low an id shared along its track (nearest previous position within ``max_km``)."""
    tracks: dict[int, dict] = {}
    next_id = 1
    for ti, lows in enumerate(lows_per_time):
        used = set()
        for low in sorted(lows, key=lambda q: -q["depth"]):
            best, best_d = None, max_km
            for tid, tr in tracks.items():
                if tid in used or (times[ti] - tr["time"]).total_seconds() / 3600.0 > max_gap_h:
                    continue
                d = float(_gc(tr["lat"], tr["lon"], low["lat"], low["lon"])[0])
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = next_id
                next_id += 1
            used.add(best)
            tracks[best] = {"time": times[ti], "lat": low["lat"], "lon": low["lon"]}
            low["id"] = best


# ----------------------------------------------------------------------------------------------------------------
# Score
# ----------------------------------------------------------------------------------------------------------------
def ingredients_score(d: dict) -> dict:
    """Composite 0-1 score and its five parts from arrays of diagnostics (see ``SCORE_DOC``). Display only."""
    def g(k):
        a = d.get(k)
        return None if a is None else np.asarray(a, float)

    pw_s = _ramp(g("pwat"), 25.0, 40.0)
    anom = g("pwat_anom")
    if anom is not None and np.isfinite(anom).any():
        pw_s = np.where(np.isfinite(anom), _ramp(anom, 1.0, 3.0), pw_s)
    moist = 0.5 * (pw_s + _ramp(g("ivt"), 250.0, 600.0))
    cin = g("mucin")
    inst = _ramp(g("mucape"), 50.0, 800.0) * np.where(np.isfinite(cin), _ramp(cin, -150.0, -50.0), 1.0)
    eff = 0.5 * (_ramp(g("wcd"), 1500.0, 4000.0) + _ramp(g("rh_700_500"), 40.0, 80.0))
    stat = _ramp(g("corfidi_up"), 12.0, 4.0)
    ldir = g("llj_dir")
    onshore = np.where((ldir >= 45.0) & (ldir <= 135.0), g("llj_speed"), 0.0)
    parts = [_ramp(onshore, 6.0, 18.0)]
    for k, lo, hi in (("upslope_w", 0.03, 0.25), ("mfc925", 0.5, 3.0)):
        a = g(k)
        if a is not None:
            parts.append(np.nan_to_num(_ramp(a, lo, hi)))
    forc = np.mean(np.stack(parts), axis=0)
    subs = {"score_moisture": moist, "score_instability": inst, "score_efficiency": eff,
            "score_stationarity": stat, "score_forcing": forc}
    prod = np.ones_like(moist)
    for a in subs.values():
        prod = prod * np.clip(a, 0.05, 1.0)
    score = prod ** (1.0 / len(subs))
    bad = ~(np.isfinite(g("pwat")) & np.isfinite(g("mucape")) & np.isfinite(g("corfidi_up")))
    subs["score"] = np.where(bad, np.nan, score)
    return subs


# ----------------------------------------------------------------------------------------------------------------
# Static data: orography, terrain, climatology
# ----------------------------------------------------------------------------------------------------------------
def _block_mean(a, n):
    ny, nx = a.shape
    a = a[: ny // n * n, : nx // n * n].reshape(ny // n, n, nx // n, n)
    cnt = np.isfinite(a).sum(axis=(1, 3))
    s = np.nansum(a, axis=(1, 3))
    return np.where(cnt >= n * n / 2.0, s / np.maximum(cnt, 1), np.nan).astype(np.float32)


def _reduce(kind):
    if kind == "block":
        return (lambda a: _block_mean(a, AROME_BLOCK)), (lambda x: x[: x.size // AROME_BLOCK * AROME_BLOCK].reshape(-1, AROME_BLOCK).mean(1))
    if kind == "sub":
        return (lambda a: a[::SYN_STEP, ::SYN_STEP]), (lambda x: x[::SYN_STEP])
    return (lambda a: a), (lambda x: x)


def _orography(model, bbox, kind, notes):
    """Model orography (m, sea = 0) on the reduced grid; cached on disk (static)."""
    name = f"orog_{model}_{'_'.join(f'{b:.4f}' for b in bbox)}_{kind}.npy"
    path = CACHE_DIR / name
    try:
        if path.exists():
            return np.load(path)
    except Exception:  # noqa: BLE001
        pass
    red, _ = _reduce(kind)
    http = om._http()  # noqa: SLF001 - static files have no public reader in openmeteo_s3
    meta = json.loads(http.get(f"{om.S3_HOST}/data/{model}/static/meta.json"))
    url = f"{om.S3_HOST}/data/{model}/static/HSURF.om"
    rd = om._open(url)  # noqa: SLF001
    try:
        (y0, y1, x0, x1), _la, _lo = om._regular_window(meta["crs_wkt"], rd.shape, bbox)  # noqa: SLF001
        h = np.asarray(rd.read_array((slice(y0, y1), slice(x0, x1))), dtype=np.float32).reshape(y1 - y0, x1 - x0)
    finally:
        rd.close()
        http.forget(url)
    h = np.where(h < -100.0, 0.0, h)       # Open-Meteo marks sea points with -999
    h = red(h)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, h)
    except Exception as exc:  # noqa: BLE001 - read-only file system: just do not cache
        notes.append(f"orography cache not written: {type(exc).__name__}")
    return h


_TERRAIN = None


def _terrain():
    """geo/out/terrain.npz arrays needed here (None if the file is missing)."""
    global _TERRAIN
    if _TERRAIN is None:
        try:
            zf = np.load(GEO / "out" / "terrain.npz", allow_pickle=False)
            _TERRAIN = {k: np.asarray(zf[k]) for k in ("lon", "lat", "dzdx", "dzdy", "cv_frac", "land_frac", "in_domain")}
        except Exception:  # noqa: BLE001
            _TERRAIN = {}
    return _TERRAIN or None


def _strip_mask(lats, lons, buffer_km=40.0):
    """Points of a grid within ``buffer_km`` of the Comunitat Valenciana (territory + adjacent sea = inflow side)."""
    ter = _terrain()
    lon2, lat2 = np.meshgrid(lons, lats)
    if ter is None:
        return (lon2 >= -1.6) & (lon2 <= 0.9) & (lat2 >= 37.8) & (lat2 <= 40.8)
    cv = ter["cv_frac"] > 0.2
    clat = np.repeat(ter["lat"][:, None], ter["lon"].size, 1)[cv]
    clon = np.repeat(ter["lon"][None, :], ter["lat"].size, 0)[cv]
    dy = (lat2[..., None] - clat) * 111.2
    dx = (lon2[..., None] - clon) * 111.2 * np.cos(np.radians(39.3))
    return np.sqrt(dx * dx + dy * dy).min(axis=-1) <= buffer_km


def load_pwat_climatology():
    """Hook for the PWAT standardised anomaly (Hart & Grumm 2001).

    Reads ``backend/riua/static/climate_era5.npz`` (or ``$RIUA_CLIMATE``) if it exists. Expected content: 1-D
    ``lat`` and ``lon`` (ERA5 0.25 deg), and TCWV mean and standard deviation by day of year with the day axis
    first, shape ``[365|366, nlat, nlon]``; accepted key names: ``tcwv_mean``/``tcwv_std`` (preferred),
    ``pwat_mean``/``pwat_std``, ``tcwv_clim_mean``/``tcwv_clim_std``. Returns ``None`` when the file is missing or
    does not look like that, and a function ``f(time, lats, lons) -> (mean, std)`` [ny, nx] otherwise.
    """
    if not CLIMATE_FILE.exists():
        return None
    zf = np.load(CLIMATE_FILE, allow_pickle=False)
    keys = set(zf.files)
    pair = next(((m, s) for m, s in (("tcwv_mean", "tcwv_std"), ("pwat_mean", "pwat_std"),
                                     ("tcwv_clim_mean", "tcwv_clim_std"), ("tcwv_doy_mean", "tcwv_doy_std"))
                 if m in keys and s in keys), None)
    lat_k = next((k for k in ("lat", "lats", "latitude", "tcwv_lat") if k in keys), None)
    lon_k = next((k for k in ("lon", "lons", "longitude", "tcwv_lon") if k in keys), None)
    if pair is None or lat_k is None or lon_k is None:
        raise ValueError(f"{CLIMATE_FILE.name}: unrecognised keys {sorted(keys)[:20]}")
    mean, std = np.asarray(zf[pair[0]], float), np.asarray(zf[pair[1]], float)
    clat, clon = np.asarray(zf[lat_k], float), np.asarray(zf[lon_k], float)
    if mean.ndim != 3 or mean.shape[1:] != (clat.size, clon.size):
        raise ValueError(f"{CLIMATE_FILE.name}: {pair[0]} has shape {mean.shape}, expected [day, {clat.size}, {clon.size}]")

    def clim(t, lats, lons):
        nd = mean.shape[0]
        doy = _utc(t).timetuple().tm_yday
        k = min(int((doy - 1) / 366.0 * nd), nd - 1) if nd not in (365, 366) else min(doy - 1, nd - 1)
        j = np.abs(clat[None, :] - np.asarray(lats)[:, None]).argmin(axis=1)
        i = np.abs(clon[None, :] - np.asarray(lons)[:, None]).argmin(axis=1)
        return mean[k][np.ix_(j, i)], std[k][np.ix_(j, i)]

    return clim


# ----------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------
def _level_vars(tl, wl):
    out = []
    for lv in tl:
        out += [f"temperature_{lv}hPa", f"relative_humidity_{lv}hPa", f"geopotential_height_{lv}hPa"]
    for lv in wl:
        out += [f"wind_u_component_{lv}hPa", f"wind_v_component_{lv}hPa"]
    return out


AROME_VARS = _level_vars(AROME_T_LEVELS, AROME_W_LEVELS) + SFC_VARS
IFS_VARS = _level_vars(IFS_T_LEVELS, IFS_W_LEVELS) + SFC_VARS + IFS_EXTRA


def _read_job(job):
    key, ti, model, run, t, names, bbox, kind = job
    red, redc = _reduce(kind)
    missing, out, lats, lons = [], {}, None, None
    try:
        lats, lons, raw = om.read_fields(model, run, t, names, bbox)
        out = {k: red(val) for k, val in raw.items()}
    except KeyError:
        for nm in names:                      # one variable is absent: read the others one by one
            try:
                lats, lons, raw = om.read_fields(model, run, t, [nm], bbox)
                out[nm] = red(raw[nm])
            except Exception:  # noqa: BLE001
                missing.append(nm)
    except Exception as exc:  # noqa: BLE001 - network, missing file...
        return key, ti, None, None, {}, names, f"{type(exc).__name__}: {exc}"[:160]
    if lats is not None:
        lats, lons = redc(np.asarray(lats, float)), redc(np.asarray(lons, float))
    return key, ti, lats, lons, out, missing, None


def _fetch(jobs, threads, notes):
    """Run the read jobs on a thread pool. Returns ``{key: {"lat", "lon", "f": {var: [T, ny, nx]}}}``."""
    stores: dict[str, dict] = {}
    nt = {}
    for j in jobs:
        nt[j[0]] = max(nt.get(j[0], 0), j[1] + 1)
    fails: dict[str, list] = {}
    miss: dict[str, set] = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for key, ti, lats, lons, out, missing, err in pool.map(_read_job, jobs):
            if err:
                fails.setdefault(key, []).append(err)
                continue
            if missing:
                miss.setdefault(key, set()).update(missing)
            st = stores.setdefault(key, {"lat": lats, "lon": lons, "f": {}})
            for name, arr in out.items():
                if name not in st["f"]:
                    st["f"][name] = np.full((nt[key],) + arr.shape, np.nan, np.float32)
                if st["f"][name].shape[1:] == arr.shape:
                    st["f"][name][ti] = arr
    for key, errs in fails.items():
        notes.append(f"{key}: {len(errs)} read job(s) failed, e.g. {errs[0]}")
    for key, names in miss.items():
        notes.append(f"{key}: variables missing in some files: {sorted(names)[:6]}")
    return stores


def _columns(store, ti, tl, wl, zs):
    """Column dict [N = ny*nx] of one time step from a store (temperatures to K)."""
    f = store["f"]
    shape = (len(store["lat"]), len(store["lon"]))
    nanf = np.full(shape, np.nan, np.float32)

    def g(name):
        a = f.get(name)
        return nanf if a is None else a[ti]

    def lev(name, levels, off=0.0):
        return np.stack([g(f"{name}_{lv}hPa").ravel() + off for lv in levels], axis=-1).astype(float)

    return {
        "tl": np.array(tl, float), "wl": np.array(wl, float),
        "t": lev("temperature", tl, T0C), "rh": lev("relative_humidity", tl), "z": lev("geopotential_height", tl),
        "u": lev("wind_u_component", wl), "v": lev("wind_v_component", wl),
        "t2": g("temperature_2m").ravel().astype(float) + T0C, "rh2": g("relative_humidity_2m").ravel().astype(float),
        "u10": g("wind_u_component_10m").ravel().astype(float), "v10": g("wind_v_component_10m").ravel().astype(float),
        "zs": np.asarray(zs, float).ravel(),
    }


def _mfc925(store, ti, ps2d, coarsen=1):
    """925-hPa moisture-flux convergence -div(q V) in g/kg/h on the store's grid (NaN where 925 hPa is underground).

    The magnitude of a divergence depends on the grid length, so the finer grid is first averaged ``coarsen`` x
    ``coarsen`` (AROME 0.125 deg -> 0.25 deg, the IFS grid length) and the result repeated back.
    """
    f = store["f"]
    try:
        t, rh = f["temperature_925hPa"][ti].astype(float) + T0C, np.clip(f["relative_humidity_925hPa"][ti].astype(float), 1, 100)
        u, v = f["wind_u_component_925hPa"][ti].astype(float), f["wind_v_component_925hPa"][ti].astype(float)
    except KeyError:
        return np.full(ps2d.shape, np.nan)
    lat, lon = np.asarray(store["lat"], float), np.asarray(store["lon"], float)
    ny, nx = t.shape
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore")
        td = mpcalc.dewpoint_from_relative_humidity(t * units.K, rh * units.percent)
        q = mpcalc.specific_humidity_from_dewpoint(925.0 * units.hPa, td).m_as("g/kg")
        qu, qv = q * u, q * v
        if coarsen > 1 and ny >= 2 * coarsen and nx >= 2 * coarsen:
            qu, qv = _block_mean(qu, coarsen).astype(float), _block_mean(qv, coarsen).astype(float)
            lat = lat[: ny // coarsen * coarsen].reshape(-1, coarsen).mean(1)
            lon = lon[: nx // coarsen * coarsen].reshape(-1, coarsen).mean(1)
        if min(qu.shape) < 3:
            return np.full(ps2d.shape, np.nan)
        dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
        mfc = -mpcalc.divergence(qu * units("m/s"), qv * units("m/s"), dx=dx, dy=dy).m_as("1/s") * 3600.0
        if mfc.shape != (ny, nx):
            full = np.full((ny, nx), np.nan)
            rep = np.repeat(np.repeat(mfc, coarsen, axis=0), coarsen, axis=1)
            full[: rep.shape[0], : rep.shape[1]] = rep
            mfc = full
    return np.where(ps2d >= 930.0, mfc, np.nan)


# ----------------------------------------------------------------------------------------------------------------
# The driver block
# ----------------------------------------------------------------------------------------------------------------
#: (series name, key in the diagnostics dict, statistic over the strip, decimals)
SERIES = [
    ("pwat", "pwat", "p90", 1), ("pwat_anom", "pwat_anom", "p90", 2), ("ivt", "ivt", "p90", 0),
    ("ivt_dir", ("ivt_u", "ivt_v"), "vdir", 0), ("mfc925", "mfc925", "p90", 2), ("thetae850", "thetae850", "p90", 1),
    ("mucape", "mucape", "p90", 0), ("mucin", "mucin", "p50", 0), ("mlcape", "mlcape", "p90", 0),
    ("sbcape", "sbcape", "p90", 0), ("cape_model", "cape_model", "p90", 0), ("lcl_agl", "lcl_agl", "p50", 0),
    ("lfc_z", "lfc_z", "p50", 0), ("el_z", "el_z", "p90", 0), ("ncape", "ncape", "p50", 3), ("li", "li", "p10", 1),
    ("kindex", "kindex", "p90", 1), ("tt", "tt", "p90", 1), ("fzl", "fzl", "p50", 0), ("wcd", "wcd", "p50", 0),
    ("rh_700_500", "rh_700_500", "p50", 0), ("rh_cloud", "rh_cloud", "p50", 0),
    ("corfidi_up", "corfidi_up", "p10", 1), ("corfidi_up_dir", ("cor_up_u", "cor_up_v"), "vdir", 0),
    ("corfidi_dn", "corfidi_dn", "p50", 1), ("cl_speed", "cl_speed", "p50", 1), ("cl_dir", ("cl_u", "cl_v"), "vdir", 0),
    ("llj_speed", "llj_speed", "p90", 1), ("llj_dir", ("llj_u", "llj_v"), "vdir", 0), ("shear06", "shear06", "p50", 1),
    ("backbuild", "backbuild", "mean", 2), ("score", "score", "p90", 2), ("score_moisture", "score_moisture", "p90", 2),
    ("score_instability", "score_instability", "p90", 2), ("score_efficiency", "score_efficiency", "p90", 2),
    ("score_stationarity", "score_stationarity", "p90", 2), ("score_forcing", "score_forcing", "p90", 2),
]
#: series that also carry the value at every anchor point (the others only have the strip statistic: JSON size)
SERIES_WITH_ANCHORS = {"pwat", "pwat_anom", "ivt", "ivt_dir", "mfc925", "mucape", "mucin", "mlcape", "cape_model",
                       "lcl_agl", "el_z", "ncape", "li", "kindex", "fzl", "wcd", "rh_700_500", "corfidi_up",
                       "cl_speed", "cl_dir", "llj_speed", "llj_dir", "shear06", "backbuild", "score"}
SERIES_STAT_DOC = {"p90": "90th percentile over the strip", "p50": "median over the strip",
                   "p10": "10th percentile over the strip", "mean": "mean over the strip",
                   "vdir": "direction of the strip-mean vector"}
EXACT_KEYS = ["mucape", "mucin", "mlcape", "sbcape", "lcl_agl", "lfc_z", "el_z", "ncape", "li", "kindex", "tt",
              "pwat", "fzl", "wcd", "corfidi_up", "corfidi_dn", "llj_speed", "llj_dir", "cl_speed", "shear06"]
EXACT_ND = {"ncape": 3, "li": 1, "kindex": 1, "tt": 1, "pwat": 1, "corfidi_up": 1, "corfidi_dn": 1, "llj_speed": 1,
            "cl_speed": 1, "shear06": 1}


def _time_axis(now):
    t0 = now.replace(minute=0, second=0, microsecond=0)
    t0 -= timedelta(hours=t0.hour % 3)
    times = [t0 + timedelta(hours=h) for h in range(0, HORIZON_FINE_H + 1, 3)]
    t = times[-1] + timedelta(hours=1)
    while t <= t0 + timedelta(hours=HORIZON_H):
        if t.hour % 6 == 0:
            times.append(t)
        t += timedelta(hours=1)
    return times


def _plan(times, notes):
    """Which run feeds which time index: ``{"arome": {ti: run}, "ifs": {ti: run}}`` and the run labels."""
    plan = {"arome": {}, "ifs": {}}
    labels = {}
    try:
        ra = om.latest_runs(AROME, 1)[0]
        va = set(om.valid_times(AROME, ra))
        for ti, t in enumerate(times):
            if t in va and t - times[0] <= timedelta(hours=HORIZON_FINE_H):
                plan["arome"][ti] = ra
        labels[AROME] = _iso(ra)
        if not plan["arome"]:
            notes.append(f"AROME run {_iso(ra)} has none of the requested times")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"AROME run discovery failed: {type(exc).__name__}: {exc}"[:200])
    try:
        runs = om.latest_runs(IFS, 3)
        cands = [runs[0]] + [r for r in runs[1:] if r.hour in (0, 12)][:1]
        used = []
        for r in cands:
            vt = set(om.valid_times(IFS, r))
            n0 = len(plan["ifs"])
            for ti, t in enumerate(times):
                if ti not in plan["ifs"] and t in vt:
                    plan["ifs"][ti] = r
            if len(plan["ifs"]) > n0:
                used.append(r)
            if len(plan["ifs"]) == len(times):
                break
        if used:
            labels[IFS] = _iso(used[0])
        if len(used) > 1:
            labels[IFS + "_extended"] = _iso(used[1])
            notes.append(f"IFS 0.25: run {_iso(used[0])} ends before +{HORIZON_H} h; later steps come from {_iso(used[1])}")
        if len(plan["ifs"]) < len(times):
            notes.append(f"IFS 0.25: {len(times) - len(plan['ifs'])} of {len(times)} time steps not available")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"IFS run discovery failed: {type(exc).__name__}: {exc}"[:200])
    return plan, labels


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _upsample_idx(lat_f, lon_f, lat_c, lon_c):
    j = np.abs(np.asarray(lat_c)[None, :] - np.asarray(lat_f)[:, None]).argmin(axis=1)
    i = np.abs(np.asarray(lon_c)[None, :] - np.asarray(lon_f)[:, None]).argmin(axis=1)
    return j, i


def _upslope(lats, lons, u, v, q):
    """Upslope flow on the Riuà 0.05 deg grid: w = V_low . grad(h) (m/s) and q*w (g/kg m/s); None without terrain."""
    ter = _terrain()
    if ter is None:
        return None
    from scipy.interpolate import RegularGridInterpolator

    lon2, lat2 = np.meshgrid(ter["lon"], ter["lat"])
    pts = np.stack([lat2.ravel(), lon2.ravel()], axis=-1)

    def interp(a):
        a = np.asarray(a, float)
        if not np.isfinite(a).all():
            if not np.isfinite(a).any():
                return np.full(lat2.shape, np.nan)
            from scipy import ndimage
            idx = ndimage.distance_transform_edt(~np.isfinite(a), return_distances=False, return_indices=True)
            a = a[tuple(idx)]
        return RegularGridInterpolator((lats, lons), a, bounds_error=False, fill_value=None)(pts).reshape(lat2.shape)

    ui, vi, qi = interp(u), interp(v), interp(q)
    w = (ui * ter["dzdx"] + vi * ter["dzdy"]) / 1000.0
    return w, qi * w


def compute(now_utc: datetime | None = None, *, threads: int | None = None, procs: int | None = None,
            exact_anchors: bool = True) -> dict:
    """The JSON-serialisable "drivers" block. Never raises: problems are recorded in ``notes``.

    ``threads``: parallel S3 readers (default ``$RIUA_DIAG_THREADS`` or 24). ``procs``: worker processes for the
    MetPy reference soundings at the anchor points (default ``$RIUA_DIAG_PROCS`` or min(4, cpu count); 1 = inline).
    """
    notes: list[str] = []
    t_wall = time.time()
    try:
        out = _compute(now_utc, threads, procs, exact_anchors, notes)
    except Exception as exc:  # noqa: BLE001 - the product must go out even if the drivers fail
        import traceback

        notes.append(f"ingredients.compute failed: {type(exc).__name__}: {exc}"[:300])
        notes.append(traceback.format_exc()[-600:])
        out = {"model_runs": {}, "times": [], "regional": {"grid": {"lat": [], "lon": []}, "fields": {}},
               "series": {}, "synoptic": {"lows": [], "z500": None, "dana_flag": []}, "score": [], "units": UNITS}
    out["notes"] = notes
    out["elapsed_s"] = round(time.time() - t_wall, 1)
    return out


def _compute(now_utc, threads, procs, exact_anchors, notes):
    timing = {}
    tic = time.time()
    cpu0 = time.process_time()
    now = _utc(now_utc) if now_utc is not None else datetime.now(timezone.utc)
    threads = threads or int(os.environ.get("RIUA_DIAG_THREADS", "24"))
    procs = procs or int(os.environ.get("RIUA_DIAG_PROCS", str(min(4, os.cpu_count() or 1))))
    times = _time_axis(now)
    nt = len(times)
    plan, labels = _plan(times, notes)

    # ---- read ----------------------------------------------------------------------------------------------
    jobs = []
    for ti in range(nt):                       # interleave models so that every time step completes early
        if ti in plan["arome"]:
            for grp in _chunks(AROME_VARS, 23):
                jobs.append(("arome", ti, AROME, plan["arome"][ti], times[ti], grp, AROME_BBOX, "block"))
        if ti in plan["ifs"]:
            for grp in _chunks(IFS_VARS, 21):
                jobs.append(("ifs", ti, IFS, plan["ifs"][ti], times[ti], grp, REGION, "none"))
            jobs.append(("syn", ti, IFS, plan["ifs"][ti], times[ti], SYN_VARS, SYN_BBOX, "sub"))
    r0, b0 = om.transfer_stats()
    stores = _fetch(jobs, threads, notes)
    r1, b1 = om.transfer_stats()
    timing["read_s"] = round(time.time() - tic, 1)
    timing["read_cpu_s"] = round(time.process_time() - cpu0, 1)
    timing["requests"], timing["mb"] = r1 - r0, round((b1 - b0) / 1e6, 1)
    tic = time.time()

    # ---- column diagnostics per model ------------------------------------------------------------------------
    diag: dict[str, dict] = {}
    cols: dict[str, dict] = {}
    spec = {"arome": (AROME, AROME_T_LEVELS, AROME_W_LEVELS, AROME_BBOX, "block"),
            "ifs": (IFS, IFS_T_LEVELS, IFS_W_LEVELS, REGION, "none")}
    for key, (model, tl, wl, bbox, kind) in spec.items():
        st = stores.get(key)
        if st is None:
            continue
        shape = (len(st["lat"]), len(st["lon"]))
        try:
            zs = _orography(model, bbox, kind, notes)
            if zs.shape != shape:
                raise ValueError(f"orography shape {zs.shape} != grid {shape}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{key}: orography unavailable ({type(exc).__name__}: {exc}); surface taken as sea level"[:200])
            zs = np.zeros(shape, np.float32)
        d = {k: np.full((nt,) + shape, np.nan, np.float32) for k in COLUMN_KEYS + ["mfc925", "cape_model"]}
        cols[key] = {}
        for ti in sorted(plan[key]):
            c = _columns(st, ti, tl, wl, zs)
            res = diagnose_columns(c)
            for k in COLUMN_KEYS:
                d[k][ti] = res[k].reshape(shape)
            d["mfc925"][ti] = _mfc925(st, ti, d["psfc"][ti], coarsen=2 if key == "arome" else 1)
            if "cape" in st["f"]:
                d["cape_model"][ti] = st["f"]["cape"][ti]
            cols[key][ti] = c
        d["lat"], d["lon"] = np.asarray(st["lat"], float), np.asarray(st["lon"], float)
        diag[key] = d
    if not diag:
        raise RuntimeError("no model data could be read")
    if "ifs" in diag and "ifs" in stores:
        f = stores["ifs"]["f"]
        nan3 = np.full(diag["ifs"]["pwat"].shape, np.nan, np.float32)
        diag["ifs"]["pwat_profile"] = diag["ifs"]["pwat"]
        tcwv = f.get("total_column_integrated_water_vapour", nan3)
        diag["ifs"]["pwat"] = np.where(np.isfinite(tcwv), tcwv, diag["ifs"]["pwat"])   # model field first
        diag["ifs"]["omega700"] = f.get("vertical_velocity_700hPa", nan3)
    # sanity check (before any filling): our parcels against each model's own CAPE field at the same points
    checks: dict = {"cape_vs_model": {}}
    for key, d in diag.items():
        model_cape = d["cape_model"]
        rec = {}
        for pk in ("mucape", "mlcape", "sbcape"):
            okm = np.isfinite(d[pk]) & np.isfinite(model_cape)
            if okm.sum() >= 10:
                rec[pk] = _scatter(list(zip(d[pk][okm].tolist(), model_cape[okm].tolist())), "ours", "model")
        if rec:
            checks["cape_vs_model"][key] = rec
            best = max(rec, key=lambda k: rec[k]["r"] if rec[k]["r"] is not None else -9)
            m = rec["mucape"]
            notes.append(f"sanity: MUCAPE vs {key.upper()} model 'cape' field: n={m['n']}, r={m['r']}, bias(ours-model)="
                         f"{m['bias']} J/kg, RMSE={m['rmse']} J/kg; best-correlated parcel: {best} (r={rec[best]['r']}, "
                         f"bias={rec[best]['bias']})")
    timing["columns_s"] = round(time.time() - tic, 1)
    tic = time.time()

    # ---- primary model per time step; AROME holes (south of ~38 N) filled with IFS ---------------------------
    primary = []
    for ti in range(nt):
        a = diag.get("arome")
        if a is not None and np.isfinite(a["mucape"][ti]).mean() >= 0.5:
            primary.append("arome")
        elif "ifs" in diag and np.isfinite(diag["ifs"]["mucape"][ti]).any():
            primary.append("ifs")
        else:
            primary.append(None)
    if "arome" in diag and "ifs" in diag:
        a, i = diag["arome"], diag["ifs"]
        jj, ii = _upsample_idx(a["lat"], a["lon"], i["lat"], i["lon"])
        nfill = 0
        for k in list(a.keys()):
            if k in ("lat", "lon") or k not in i:
                continue
            fill = i[k][:, jj][:, :, ii]
            hole = ~np.isfinite(a[k])
            for ti, pm in enumerate(primary):
                if pm == "arome":
                    if k == "mucape":
                        nfill = max(nfill, int(hole[ti].sum()))
                    a[k][ti] = np.where(hole[ti], fill[ti], a[k][ti])
        if nfill:
            notes.append(f"AROME has no data on {nfill} of {a['mucape'][0].size} regional points (south of ~38 N / "
                         "outside its domain): filled with the IFS diagnostics there")
    for key in diag:
        d = diag[key]
        d["strip"] = _strip_mask(d["lat"], d["lon"])
        d["ivt_dir"] = _wdir(d["ivt_u"], d["ivt_v"])
        d["llj_dir"] = _wdir(d["llj_u"], d["llj_v"])

    # ---- PWAT standardised anomaly (hook) ------------------------------------------------------------------
    clim = None
    try:
        clim = load_pwat_climatology()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"PWAT climatology not usable: {type(exc).__name__}: {exc}"[:200])
    if clim is None:
        notes.append("pwat_anom is null: no climatology file (backend/riua/static/climate_era5.npz) — "
                     "see load_pwat_climatology() for the expected content")
    for key, d in diag.items():
        d["pwat_anom"] = np.full(d["pwat"].shape, np.nan, np.float32)
        if clim is not None:
            try:
                for ti in range(nt):
                    mean, std = clim(times[ti], d["lat"], d["lon"])
                    d["pwat_anom"][ti] = (d["pwat"][ti] - mean) / np.maximum(std, 0.5)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"PWAT anomaly failed on {key}: {type(exc).__name__}: {exc}"[:200])
                clim = None

    # ---- orographic forcing on the Riuà grid -----------------------------------------------------------------
    ter = _terrain()
    ups_w = ups_q = None
    if ter is None:
        notes.append("geo/out/terrain.npz not found: no upslope diagnostics, strip = fixed lon/lat box")
    else:
        ups_w = np.full((nt,) + ter["dzdx"].shape, np.nan, np.float32)
        ups_q = np.full_like(ups_w, np.nan)
        for ti, pm in enumerate(primary):
            if pm is None:
                continue
            d = diag[pm]
            res = _upslope(d["lat"], d["lon"], d["u_low"][ti], d["v_low"][ti], d["q_low"][ti])
            if res is not None:
                ups_w[ti], ups_q[ti] = res
        # strongest upslope flow of the terrain cells nearest to each diagnostic grid point (for the score)
        for key, d in diag.items():
            jr = np.abs(d["lat"][None, :] - ter["lat"][:, None]).argmin(axis=1)
            ir = np.abs(d["lon"][None, :] - ter["lon"][:, None]).argmin(axis=1)
            flat = jr[:, None] * d["lon"].size + ir[None, :]
            g = np.full((nt, d["lat"].size * d["lon"].size), 0.0, np.float32)
            for ti in range(nt):
                if np.isfinite(ups_w[ti]).any():
                    np.maximum.at(g[ti], flat.ravel(), np.nan_to_num(ups_w[ti]).ravel())
            d["upslope_w"] = g.reshape((nt, d["lat"].size, d["lon"].size))

    # ---- score ---------------------------------------------------------------------------------------------
    for d in diag.values():
        d.update(ingredients_score(d))

    # ---- series (strip statistic + anchors) on the primary model's grid --------------------------------------
    anchors_idx = {}
    for key, d in diag.items():
        anchors_idx[key] = {a: (int(np.abs(d["lat"] - la).argmin()), int(np.abs(d["lon"] - lo).argmin()))
                            for a, (la, lo) in ANCHORS.items()}
    series: dict[str, dict] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, src, stat, nd in SERIES:
            with_anchors = name in SERIES_WITH_ANCHORS
            rec = {"strip": [], **({a: [] for a in ANCHORS} if with_anchors else {})}
            for ti, pm in enumerate(primary):
                d = diag.get(pm) if pm else None
                if d is None or (isinstance(src, str) and src not in d):
                    for lst in rec.values():
                        lst.append(None)
                    continue
                m = d["strip"]
                if stat == "vdir":
                    uu, vv = d[src[0]][ti], d[src[1]][ti]
                    um, vm = np.nanmean(uu[m]), np.nanmean(vv[m])
                    rec["strip"].append(_num(_wdir(um, vm), 0) if np.isfinite(um) else None)
                    if with_anchors:
                        for a, (j, i) in anchors_idx[pm].items():
                            rec[a].append(_num(_wdir(uu[j, i], vv[j, i]), 0) if np.isfinite(uu[j, i]) else None)
                    continue
                fld = d[src][ti]
                vals = fld[m]
                if stat == "mean":
                    sv = np.nanmean(vals)
                else:
                    sv = np.nanpercentile(vals, int(stat[1:])) if np.isfinite(vals).any() else np.nan
                rec["strip"].append(_num(sv, nd))
                if with_anchors:
                    for a, (j, i) in anchors_idx[pm].items():
                        rec[a].append(_num(fld[j, i], nd))
            if any(x is not None for x in rec["strip"]):
                series[name] = rec
        if "ifs" in diag:                                        # cross-model references, IFS at every step
            d = diag["ifs"]
            series["pwat_ifs"] = {"strip": [_num(np.nanpercentile(d["pwat"][ti][d["strip"]], 90), 1)
                                            if np.isfinite(d["pwat"][ti]).any() else None for ti in range(nt)]}
            series["omega700"] = {"strip": [_num(np.nanpercentile(d["omega700"][ti][d["strip"]], 10), 2)
                                            if np.isfinite(d["omega700"][ti]).any() else None for ti in range(nt)]}
        if ups_w is not None:
            dom = ter["in_domain"]
            rec_w = {"strip": [], "max": [], **{a: [] for a in ANCHORS}}
            rec_q = {"strip": [], "max": [], **{a: [] for a in ANCHORS}}
            for ti in range(nt):
                for rec, arr, nd in ((rec_w, ups_w[ti], 3), (rec_q, ups_q[ti], 2)):
                    ok = np.isfinite(arr).any()
                    rec["strip"].append(_num(np.nanpercentile(arr[dom], 90), nd) if ok else None)
                    rec["max"].append(_num(np.nanmax(arr[dom]), nd) if ok else None)
                    for a, (la, lo) in ANCHORS.items():
                        j, i = int(np.abs(ter["lat"] - la).argmin()), int(np.abs(ter["lon"] - lo).argmin())
                        rec[a].append(_num(np.nanmax(arr[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]), nd) if ok else None)
            series["upslope_w"], series["upslope_qflux"] = rec_w, rec_q
    stat_of = {name: stat for name, _s, stat, _n in SERIES}
    stat_of.update(pwat_ifs="p90", omega700="p10", upslope_w="p90", upslope_qflux="p90")

    # ---- regional maps on the 0.25 deg lattice ---------------------------------------------------------------
    if "ifs" in diag:
        glat, glon = diag["ifs"]["lat"], diag["ifs"]["lon"]
    else:
        glat, glon = diag["arome"]["lat"][1::2], diag["arome"]["lon"][1::2]
    maps = {"pwat": ("pwat", 1.0), "ivt": ("ivt", 10.0), "ivt_dir": ("ivt_dir", 10.0), "mucape": ("mucape", 25.0)}
    fields = {k: [] for k in maps}
    for ti, pm in enumerate(primary):
        for k, (src, scale) in maps.items():
            if pm is None:
                fields[k].append(None)
                continue
            d = diag[pm]
            jj, ii = _upsample_idx(glat, glon, d["lat"], d["lon"])
            fields[k].append(_lst(d[src][ti][np.ix_(jj, ii)], 0, scale))
    regional = {"grid": {"lat": [round(float(x), 3) for x in glat], "lon": [round(float(x), 3) for x in glon]},
                "fields": fields, "scale": {k: sc for k, (_s, sc) in maps.items()},
                "layout": "fields[name][time] = flattened [lat, lon] row-major, south to north; value x scale = physical",
                "model": primary}
    timing["series_s"] = round(time.time() - tic, 1)
    tic = time.time()

    # ---- synoptic ------------------------------------------------------------------------------------------
    syn = {"lows": [[] for _ in range(nt)], "z500": None, "dana_flag": [None] * nt, "jet": [None] * nt,
           "tracks": {}, "reference": {"name": "València", "lat": VLC[0], "lon": VLC[1]}}
    st = stores.get("syn")
    if st is None:
        notes.append("no synoptic fields: lows, jet and dana_flag are empty")
    else:
        slat, slon = np.asarray(st["lat"], float), np.asarray(st["lon"], float)
        f = st["f"]
        z5 = f.get("geopotential_height_500hPa")
        t5 = f.get("temperature_500hPa")
        lows_all = [[] for _ in range(nt)]
        for ti in range(nt):
            if z5 is None or not np.isfinite(z5[ti]).all():
                continue
            lows_all[ti] = find_closed_lows(z5[ti], slat, slon, None if t5 is None else t5[ti])
            try:
                syn["jet"][ti] = jet_streak(f["wind_u_component_300hPa"][ti].astype(float),
                                            f["wind_v_component_300hPa"][ti].astype(float), slat, slon)
            except Exception:  # noqa: BLE001
                pass
        _link_tracks(lows_all, times)
        for ti in range(nt):
            if z5 is None or not np.isfinite(z5[ti]).all():
                continue
            pm = primary[ti]
            onshore = None
            if pm:
                d = diag[pm]
                um, vm = np.nanmean(d["llj_u"][ti][d["strip"]]), np.nanmean(d["llj_v"][ti][d["strip"]])
                if np.isfinite(um):
                    fd, fs = float(_wdir(um, vm)), float(np.hypot(um, vm))
                    onshore = {"dir": round(fd), "speed": round(fs, 1), "ok": bool(30.0 <= fd <= 150.0 and fs >= 5.0)}
            flag = False
            for low in lows_all[ti]:
                geom = bool(low["dist_km"] <= 1000.0 and 150.0 <= low["bearing"] <= 300.0)
                low["favourable"] = bool(geom and onshore is not None and onshore["ok"])
                low["favourable_position"] = geom
                flag = flag or low["favourable"]
            syn["dana_flag"][ti] = flag
            syn["lows"][ti] = [{
                "id": q.get("id"), "lat": q["lat"], "lon": q["lon"], "z500": _num(q["z500"], 0), "depth": _num(q["depth"], 0),
                "ring_min_depth": _num(q["ring_min_depth"], 0), "cutoff": q["cutoff"], "t500": _num(q.get("t500"), 1),
                "t500_anom": _num(q.get("t500_anom"), 1), "cold_core": q.get("cold_core"),
                "dist_km": _num(q["dist_km"], 0), "bearing": _num(q["bearing"], 0), "sector": q["sector"],
                "favourable_position": q["favourable_position"], "favourable": q["favourable"]}
                for q in lows_all[ti] if q["dist_km"] <= 3000.0]
            if syn["jet"][ti]:
                syn["jet"][ti] = {k: (_num(val, 0 if k in ("dist_km", "bearing", "dir") else 1) if isinstance(val, float) else val)
                                  for k, val in syn["jet"][ti].items()}
            syn.setdefault("onshore_flow", [None] * nt)[ti] = onshore
            syn.setdefault("z500_valencia", [None] * nt)[ti] = _num(float(_sample(z5[ti], slat, slon, VLC[0], VLC[1])) / 10.0, 1)
            if t5 is not None and np.isfinite(t5[ti]).all():
                syn.setdefault("t500_valencia", [None] * nt)[ti] = _num(float(_sample(t5[ti], slat, slon, VLC[0], VLC[1])), 1)
        for ti in range(nt):
            if times[ti].hour % 6:
                continue
            for q in syn["lows"][ti]:
                syn["tracks"].setdefault(str(q["id"]), []).append([_iso(times[ti]), q["lat"], q["lon"], q["z500"], q["depth"]])
        w, s, e, n, step = SYN_MAP
        mj = [int(j) for j in np.nonzero((slat >= s) & (slat <= n) & (np.mod(slat - s, step) == 0))[0]]
        mi = [int(i) for i in np.nonzero((slon >= w) & (slon <= e) & (np.mod(slon - w, step) == 0))[0]]
        tsel = [ti for ti in range(nt) if times[ti].hour % 6 == 0 and z5 is not None and np.isfinite(z5[ti]).all()]
        if z5 is not None and tsel:
            syn["z500"] = {"lat": [float(slat[j]) for j in mj], "lon": [float(slon[i]) for i in mi],
                           "time_index": tsel, "units": "dam",
                           "values": [_lst(z5[ti][np.ix_(mj, mi)], 0, 10.0) for ti in tsel]}
    timing["synoptic_s"] = round(time.time() - tic, 1)
    tic = time.time()

    # ---- MetPy reference soundings at the anchors (first 48 h) and self-checks ------------------------------
    anchors_metpy: dict = {}
    if exact_anchors:
        tasks, where = [], []
        for ti, pm in enumerate(primary):
            if pm is None or times[ti] - times[0] > timedelta(hours=HORIZON_FINE_H) or ti not in cols.get(pm, {}):
                continue
            d = diag[pm]
            for a, (j, i) in anchors_idx[pm].items():
                nflat = j * d["lon"].size + i
                tasks.append(_column_sounding(cols[pm][ti], nflat, float(d["psfc"][ti].ravel()[nflat])))
                where.append((a, ti, pm, j, i))
        results = _run_exact(tasks, procs, notes)
        tix = sorted({w[1] for w in where})
        pos = {ti: n for n, ti in enumerate(tix)}
        anchors_metpy = {"time_index": tix,
                         "doc": "MetPy 1-D reference functions on the native model levels + surface, first 48 h",
                         **{a: {k: [None] * len(tix) for k in EXACT_KEYS} for a in ANCHORS}}
        pairs = {k: [] for k in ("mucape", "mlcape", "sbcape", "pwat", "corfidi_up", "lcl_agl", "el_z")}
        for (a, ti, pm, j, i), r in zip(where, results):
            if not r:
                continue
            for k in EXACT_KEYS:
                anchors_metpy[a][k][pos[ti]] = _num(r.get(k), EXACT_ND.get(k, 0))
            d = diag[pm]
            for k in pairs:
                src = "pwat_profile" if (k == "pwat" and pm == "ifs") else k
                fv, ev = float(d[src][ti][j, i]), r.get(k)
                if ev is not None and np.isfinite(fv) and np.isfinite(ev):
                    pairs[k].append((fv, float(ev)))
        checks["engine_vs_metpy"] = {k: _scatter(v, "engine", "metpy") for k, v in pairs.items() if len(v) >= 3}
        if checks["engine_vs_metpy"]:
            m = checks["engine_vs_metpy"].get("mucape")
            if m:
                notes.append(f"self-check: vectorised MUCAPE vs MetPy most_unstable_cape_cin at the anchors: n={m['n']}, "
                             f"r={m['r']}, bias={m['bias']} J/kg, RMSE={m['rmse']} J/kg")
    timing["exact_s"] = round(time.time() - tic, 1)
    timing["cpu_main_process_s"] = round(time.process_time() - cpu0, 1)
    timing["threads"], timing["procs"] = threads, procs

    # ---- sea-surface temperature (IFS 9 km) ------------------------------------------------------------------
    sst = None
    try:
        rh_ = om.latest_runs(IFS_HRES, 1)[0]
        vts = [t for t in om.valid_times(IFS_HRES, rh_) if t >= times[0]]
        _la, _lo, o = om.read_fields(IFS_HRES, rh_, vts[0] if vts else rh_, ["sea_surface_temperature"],
                                     (-0.6, 37.6, 2.5, 41.0))
        v = o["sea_surface_temperature"]
        v = v[np.isfinite(v)]
        if v.size:
            sst = {"mean": _num(v.mean(), 1), "max": _num(v.max(), 1), "min": _num(v.min(), 1), "n_points": int(v.size),
                   "model": IFS_HRES, "run": _iso(rh_), "box": [-0.6, 37.6, 2.5, 41.0]}
            labels[IFS_HRES + "_sst"] = _iso(rh_)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"SST not read: {type(exc).__name__}: {exc}"[:160])

    for ti, pm in enumerate(primary):
        if pm is None:
            notes.append(f"{_iso(times[ti])}: no model data")
    notes.append("series switch from AROME (0.125 deg block means of the 2.5 km fields) to IFS 0.25 deg after the "
                 "last AROME step: expect a small discontinuity there (see regional.model)")
    notes.append("score: " + SCORE_DOC)
    return {
        "generated": _iso(datetime.now(timezone.utc)), "now": _iso(now), "model_runs": labels,
        "times": [_iso(t) for t in times], "regional": regional, "series": series, "series_stat": stat_of,
        "series_stat_doc": SERIES_STAT_DOC, "anchors": {a: {"lat": la, "lon": lo} for a, (la, lo) in ANCHORS.items()},
        "anchors_metpy": anchors_metpy, "synoptic": syn, "score": series.get("score", {}).get("strip", []),
        "sst": sst, "checks": checks, "units": UNITS, "timing": timing,
    }


def _scatter(pairs, na, nb):
    a = np.array([p[0] for p in pairs], float)
    b = np.array([p[1] for p in pairs], float)
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"n": int(a.size), "r": _num(r, 3), "bias": _num((a - b).mean(), 1), "rmse": _num(np.sqrt(((a - b) ** 2).mean()), 1),
            "mean_" + na: _num(a.mean(), 1), "mean_" + nb: _num(b.mean(), 1),
            "p90_" + na: _num(np.percentile(a, 90), 1), "p90_" + nb: _num(np.percentile(b, 90), 1)}


def _run_exact(tasks, procs, notes):
    if not tasks:
        return []
    if procs > 1 and len(tasks) > 8:
        try:
            ctx = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with ProcessPoolExecutor(max_workers=procs, mp_context=ctx) as pool:
                    return list(pool.map(_exact_job, tasks, chunksize=max(1, len(tasks) // (procs * 4))))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"process pool failed ({type(exc).__name__}: {exc}); MetPy soundings computed inline"[:200])
    return [_exact_job(s) for s in tasks]


# ----------------------------------------------------------------------------------------------------------------
# One point, "how was this computed"
# ----------------------------------------------------------------------------------------------------------------
def point_profile(lat: float, lon: float, valid_time: datetime, model: str | None = None) -> dict:
    """Sounding and its diagnostics at one point and one valid time.

    AROME (5x5 block mean of the 0.025 deg cells around the point) when the time is in its latest run and the
    point inside its domain, otherwise IFS 0.25 deg (nearest node). ``model`` = "arome" / "ifs" forces the choice.
    Returns the profile (p, T, Td, z, wind), the MU-parcel path, MetPy's reference diagnostics and the values of the
    vectorised engine for the same column. Never raises: errors are returned under ``"notes"``.
    """
    notes: list[str] = []
    vt = _utc(valid_time)
    out = {"lat": lat, "lon": lon, "valid_time": _iso(vt), "notes": notes}
    order = [m for m in ("arome", "ifs") if model in (None, m)]
    for key in order:
        try:
            mdl, tl, wl = (AROME, AROME_T_LEVELS, AROME_W_LEVELS) if key == "arome" else (IFS, IFS_T_LEVELS, IFS_W_LEVELS)
            names = (AROME_VARS if key == "arome" else IFS_VARS)
            run = next((r for r in om.latest_runs(mdl, 3) if vt in set(om.valid_times(mdl, r))), None)
            if run is None:
                notes.append(f"{key}: valid time not in the latest runs")
                continue
            if key == "arome":
                la0, lo0 = round(lat / 0.025) * 0.025, round(lon / 0.025) * 0.025
                bbox, kind = (lo0 - 0.0501, la0 - 0.0501, lo0 + 0.0501, la0 + 0.0501), "block"
            else:
                la0, lo0 = round(lat / 0.25) * 0.25, round(lon / 0.25) * 0.25
                bbox, kind = (lo0 - 0.01, la0 - 0.01, lo0 + 0.01, la0 + 0.01), "none"
            jobs = [(key, 0, mdl, run, vt, grp, bbox, kind) for grp in _chunks(names, 12)]
            st = _fetch(jobs, 10, notes).get(key)
            if st is None:
                continue
            zs = _orography(mdl, bbox, kind, notes)
            c = _columns(st, 0, tl, wl, zs)
            eng = diagnose_columns(c)
            if not np.isfinite(eng["psfc"][0]):
                notes.append(f"{key}: no data at this point")
                continue
            s = _column_sounding(c, 0, float(eng["psfc"][0]))
            ex = _exact_job(s)
            rnd = lambda a, nd=1: [_num(x, nd) for x in np.asarray(a, float)]  # noqa: E731
            out.update({
                "model": mdl, "run": _iso(run), "grid_lat": round(float(np.mean(st["lat"])), 4),
                "grid_lon": round(float(np.mean(st["lon"])), 4), "surface_height_m": _num(float(np.ravel(zs)[0]), 0),
                "surface_pressure_hpa": _num(eng["psfc"][0], 1),
                "profile": {"p_hpa": rnd(s["p"]), "t_c": rnd(s["t"] - T0C), "td_c": rnd(s["td"] - T0C), "z_m": rnd(s["z"], 0),
                            "wind_p_hpa": rnd(s.get("pw", [])), "u_ms": rnd(s.get("u", [])), "v_ms": rnd(s.get("v", []))},
                "parcel_mu": {"p_hpa": rnd(ex.pop("parcel_mu_p", [])), "t_c": rnd(np.asarray(ex.pop("parcel_mu_t", []), float) - T0C)},
                "metpy": {k: (_num(val, 3) if isinstance(val, float) else val) for k, val in ex.items()},
                "engine": {k: _num(eng[k][0], 3) for k in COLUMN_KEYS},
                "method": "metpy = MetPy 1-D reference functions on the native levels + surface; engine = the vectorised "
                          "10-hPa column engine used for the maps (same definitions)",
            })
            return out
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{key}: {type(exc).__name__}: {exc}"[:200])
    return out


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def _peak_rss_mb():
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return r / 1e3
    except Exception:  # noqa: BLE001
        return float("nan")


def print_table(block, hours=48, who="strip"):
    """Readable table of the regional series."""
    s = block["series"]
    cols = [("pwat", 5), ("pwat_anom", 5), ("ivt", 4), ("ivt_dir", 3), ("mfc925", 5), ("mucape", 5), ("mlcape", 5),
            ("mucin", 5), ("cape_model", 5), ("lcl_agl", 5), ("el_z", 5), ("ncape", 5), ("li", 5), ("kindex", 4),
            ("fzl", 4), ("wcd", 4), ("rh_700_500", 3), ("llj_speed", 4), ("llj_dir", 3), ("cl_speed", 4), ("cl_dir", 3),
            ("corfidi_up", 4), ("shear06", 4), ("backbuild", 4), ("upslope_w", 5), ("score", 4)]
    head = {"pwat": "PWAT", "pwat_anom": "PWsig", "ivt": "IVT", "ivt_dir": "dir", "mfc925": "MFC", "mucape": "MUCAP",
            "mlcape": "MLCAP", "mucin": "MUCIN", "cape_model": "mCAPE", "lcl_agl": "LCL", "el_z": "EL", "ncape": "NCAPE",
            "li": "LI", "kindex": "K", "fzl": "FZL", "wcd": "WCD", "rh_700_500": "RH", "llj_speed": "LLJ", "llj_dir": "dir",
            "cl_speed": "CL", "cl_dir": "dir", "corfidi_up": "Cup", "shear06": "S06", "backbuild": "BB", "upslope_w": "w_up",
            "score": "scor"}
    t0 = datetime.strptime(block["times"][0], "%Y-%m-%dT%H:%MZ")
    print(f"\n[{who}] " + "time(UTC)    mdl " + " ".join(f"{head[k]:>{max(w, len(head[k]))}}" for k, w in cols) + "  DANA low (dist km, bearing)")
    for ti, ts in enumerate(block["times"]):
        t = datetime.strptime(ts, "%Y-%m-%dT%H:%MZ")
        if (t - t0).total_seconds() > hours * 3600:
            break
        row = []
        for k, w in cols:
            val = s.get(k, {}).get(who, [None] * (ti + 1))[ti] if k in s else None
            w = max(w, len(head[k]))
            row.append(f"{'-':>{w}}" if val is None else f"{val:>{w}}")
        lows = block["synoptic"]["lows"][ti]
        near = lows[0] if lows else None
        low_txt = "-" if near is None else (f"{near['lat']:.0f}N {near['lon']:.0f}E z={near['z500']} d={near['depth']} "
                                           f"({near['dist_km']} km, {near['sector']})" + (" *FAV*" if near["favourable"] else ""))
        print(f"{t:%d %HZ}  {str(block['regional']['model'][ti] or '-'):>5} " + " ".join(row) + "  " + low_txt)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.time()
    now = None
    if argv and not argv[0].startswith("-"):
        now = datetime.strptime(argv[0], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    block = compute(now)
    wall = time.time() - t0
    out = REPO / "scratch" / "i1-ingredients" / "drivers_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(block, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    out.write_text(txt, encoding="utf-8")
    print("model runs:", block.get("model_runs"))
    if block.get("times"):
        print_table(block, 48, "strip")
        print_table(block, 168, "valencia")
        print_table(block, 48, "chiva")
        print("\nSST:", block.get("sst"))
        print("checks:", json.dumps(block.get("checks"), indent=1))
        print("jet (first 5):", block["synoptic"]["jet"][:5])
        print("onshore flow (first 5):", block["synoptic"].get("onshore_flow", [])[:5])
        print("dana_flag:", block["synoptic"]["dana_flag"])
        sizes = {k: len(json.dumps(val, separators=(",", ":"))) for k, val in block.items()}
        print("JSON bytes per key:", sizes)
    print("timing:", block.get("timing"))
    print("notes:")
    for nline in block["notes"]:
        print("  -", nline[:400])
    print(f"wall {wall:.1f} s, peak RSS (self+children) {_peak_rss_mb():.0f} MB, JSON {len(txt) / 1e3:.1f} kB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
