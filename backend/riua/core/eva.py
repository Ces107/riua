"""Return periods of rainfall in MODEL SPACE (ERA5 0.25 deg, 1950-2025) and the bridge to warning thresholds.

The 0.25 deg ensemble (IFS ENS open data) cannot produce a 700 mm day; ERA5 gave about 100 mm/24 h at the grid
point of Turis on 29 October 2024, where the gauge measured 772 mm. Forecast amounts are therefore ranked against
the climate of a model of the same kind: "this 24-h total is a 1-in-40-year value for this grid point of the
model", not "this is a 40-year rain at the gauge". See ``backend/riua/static/climate_README.md``.

Data: ``static/climate_era5.npz`` (built by ``climate/build/``). GEV fitted by L-moments to annual maxima of
sliding 1/6/12/24/48/72-h sums, shape pooled over the 5 x 5 neighbouring grid points.

    F(x) = exp(-(1 + xi (x - loc) / scale)^(-1/xi))      xi > 0: heavy upper tail

Public functions (numpy only, vectorised over lat/lon/amount):

    return_period(amount_mm, duration_h, lat, lon)        -> years (annual-maximum sense; >= 1)
    return_level(T, duration_h, lat, lon)                 -> mm
    return_level_ci(T, duration_h, lat, lon)              -> (lo, hi) 90 % bootstrap interval, T in the stored list
    model_equivalent(obs_mm, window_h, lat, lon)          -> ERA5-space amount as frequent as obs_mm at a gauge
    observed_equivalent(model_mm, window_h, lat, lon)     -> the inverse (rough, see README)

Parameters are interpolated bilinearly between grid points (clamped at the edge of the box lon -2.5..1.0,
lat 37.5..41.0) and, for durations between the stored ones, linearly in log(duration).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

CLIMATE_FILE = Path(__file__).resolve().parents[1] / "static" / "climate_era5.npz"
T_MAX = 1.0e5   # return periods are capped here: beyond it the number means nothing


@lru_cache(maxsize=1)
def _data() -> dict:
    with np.load(CLIMATE_FILE, allow_pickle=False) as z:
        keys = ["lat", "lon", "eva_durations_h", "eva_T", "gev_loc", "gev_scale", "gev_shape", "rl_lo", "rl_hi",
                "obs_thresholds_mm", "obs_gev_loc", "obs_gev_scale", "obs_gev_shape"]
        return {k: np.asarray(z[k], np.float64) for k in keys if k in z.files}


def _bilinear(field: np.ndarray, lat, lon) -> np.ndarray:
    """field [..., ny, nx] on the regular grid -> values at (lat, lon), broadcast; leading axes are kept first."""
    d = _data()
    glat, glon = d["lat"], d["lon"]
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    fy = np.clip((lat - glat[0]) / (glat[1] - glat[0]), 0.0, len(glat) - 1.0)
    fx = np.clip((lon - glon[0]) / (glon[1] - glon[0]), 0.0, len(glon) - 1.0)
    y0 = np.minimum(fy.astype(int), len(glat) - 2)
    x0 = np.minimum(fx.astype(int), len(glon) - 2)
    wy, wx = fy - y0, fx - x0
    return (field[..., y0, x0] * (1 - wy) * (1 - wx) + field[..., y0, x0 + 1] * (1 - wy) * wx
            + field[..., y0 + 1, x0] * wy * (1 - wx) + field[..., y0 + 1, x0 + 1] * wy * wx)


def _duration_weights(duration_h: float):
    durs = _data()["eva_durations_h"]
    h = float(duration_h)
    if h < durs[0] - 1e-9 or h > durs[-1] + 1e-9:
        raise ValueError(f"duration {duration_h} h outside the fitted range {durs[0]:g}..{durs[-1]:g} h")
    k = int(np.clip(np.searchsorted(durs, h, side="right") - 1, 0, len(durs) - 2))
    w = (np.log(h) - np.log(durs[k])) / (np.log(durs[k + 1]) - np.log(durs[k]))
    return k, float(np.clip(w, 0.0, 1.0))


def gev_params(duration_h: float, lat, lon, prefix: str = "gev"):
    """(loc, scale, xi) at the point(s): bilinear in space, log-linear in duration (loc and scale in log space)."""
    d = _data()
    k, w = _duration_weights(duration_h)
    loc = np.exp((1 - w) * np.log(_bilinear(d[f"{prefix}_loc"][k], lat, lon)) + w * np.log(_bilinear(d[f"{prefix}_loc"][k + 1], lat, lon)))
    scale = np.exp((1 - w) * np.log(_bilinear(d[f"{prefix}_scale"][k], lat, lon)) + w * np.log(_bilinear(d[f"{prefix}_scale"][k + 1], lat, lon)))
    xi = (1 - w) * _bilinear(d[f"{prefix}_shape"][k], lat, lon) + w * _bilinear(d[f"{prefix}_shape"][k + 1], lat, lon)
    return loc, scale, xi


def _cdf(x, loc, scale, xi):
    xi = np.where(np.abs(xi) < 1e-6, 1e-6, xi)
    z = 1.0 + xi * (np.asarray(x, np.float64) - loc) / scale
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        F = np.exp(-np.maximum(z, 1e-300) ** (-1.0 / xi))
    return np.where(z <= 0, np.where(xi > 0, 0.0, 1.0), F)


def _quantile(F, loc, scale, xi):
    xi = np.where(np.abs(xi) < 1e-6, 1e-6, xi)
    return loc + scale / xi * ((-np.log(F)) ** (-xi) - 1.0)


def return_period(amount_mm, duration_h: float, lat, lon) -> np.ndarray:
    """Return period (years) of ``amount_mm`` accumulated in ``duration_h`` hours, in ERA5 space.

    Annual-maximum sense: T = 1 / P(annual maximum > amount). Amounts below most annual maxima give T -> 1;
    T is capped at 1e5 years. Anything above ~200 years is an extrapolation of a 76-year record.
    """
    loc, scale, xi = gev_params(duration_h, lat, lon)
    F = _cdf(amount_mm, loc, scale, xi)
    with np.errstate(divide="ignore"):
        return np.minimum(1.0 / np.maximum(1.0 - F, 1.0 / T_MAX), T_MAX)


def return_level(T, duration_h: float, lat, lon) -> np.ndarray:
    """ERA5-space amount (mm in ``duration_h`` hours) with return period ``T`` years (T > 1)."""
    loc, scale, xi = gev_params(duration_h, lat, lon)
    return _quantile(1.0 - 1.0 / np.asarray(T, np.float64), loc, scale, xi)


def return_level_ci(T: float, duration_h: float, lat, lon):
    """(lo, hi) 90 % bootstrap interval of the return level; T and duration must be stored values."""
    d = _data()
    kt = int(np.argmin(np.abs(d["eva_T"] - T)))
    kd = int(np.argmin(np.abs(d["eva_durations_h"] - duration_h)))
    if abs(d["eva_T"][kt] - T) > 1e-6 or abs(d["eva_durations_h"][kd] - duration_h) > 1e-6:
        raise ValueError("confidence intervals are stored only for T in eva_T and durations in eva_durations_h")
    return _bilinear(d["rl_lo"][kd, kt], lat, lon), _bilinear(d["rl_hi"][kd, kt], lat, lon)


def _obs_params(lat, lon):
    d = _data()
    return (_bilinear(d["obs_gev_loc"], lat, lon), _bilinear(d["obs_gev_scale"], lat, lon),
            _bilinear(d["obs_gev_shape"], lat, lon))


def model_equivalent(obs_mm, window_h: float, lat, lon) -> np.ndarray:
    """ERA5-space amount in ``window_h`` hours that is exceeded as often (annual-maximum frequency) as ``obs_mm``
    is exceeded at a rain gauge at the same place. Quantile mapping obs GEV -> ERA5 GEV. The gauge-space
    distribution is that of the DAILY maximum (the only one available), used as a stand-in for the observed
    12-h / 24-h amount; see the README for the size of that approximation."""
    F = np.clip(_cdf(obs_mm, *_obs_params(lat, lon)), 0.02, 1.0 - 1.0 / 2000.0)
    return _quantile(F, *gev_params(window_h, lat, lon))


def observed_equivalent(model_mm, window_h: float, lat, lon) -> np.ndarray:
    """Gauge-space daily amount as frequent as ``model_mm`` (ERA5-space, ``window_h`` hours). Inverse mapping."""
    F = np.clip(_cdf(model_mm, *gev_params(window_h, lat, lon)), 0.02, 1.0 - 1.0 / 2000.0)
    return _quantile(F, *_obs_params(lat, lon))


if __name__ == "__main__":  # smoke run on the real file:  python -m riua.core.eva
    pts = {"Turis": (39.39, -0.71), "Valencia": (39.48, -0.37), "Alicante": (38.37, -0.49), "Oliva": (38.92, -0.12)}
    for name, (la, lo) in pts.items():
        print(f"{name}: T100 24 h {float(return_level(100, 24, la, lo)):.1f} mm, "
              f"T(100 mm / 24 h) {float(return_period(100, 24, la, lo)):.1f} yr, "
              f"model equivalent of 180 mm observed (12 h) {float(model_equivalent(180, 12, la, lo)):.1f} mm")
