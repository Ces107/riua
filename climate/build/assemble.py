"""Merge the parts into ``backend/riua/static/climate_era5.npz`` (one compressed file, numpy-only at runtime).

    python climate/build/assemble.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from common import PARTS, STATIC, safe_savez


def main():
    cp = np.load(PARTS / "clim_precip.npz")
    ct = np.load(PARTS / "clim_tcwv.npz")
    cm = np.load(PARTS / "clim_monthly.npz")
    ev = np.load(PARTS / "eva.npz")
    am = np.load(PARTS / "annmax.npz")
    sc = np.load(PARTS / "scaling.npz")
    events = json.loads((PARTS / "events.json").read_text(encoding="utf-8"))

    out = {}
    out["lat"] = cp["lat"].astype(np.float32)
    out["lon"] = cp["lon"].astype(np.float32)
    # A ---------------------------------------------------------------------------------------------------
    out["clim_p"] = cp["clim_p"].astype(np.float64)
    out["clim_doy"] = cp["clim_doy"].astype(np.int16)
    out["pr_q_scale"] = np.float32(cp["pr_q_scale"])
    for w in (12, 24, 48, 72):
        out[f"pr_q{w}"] = cp[f"pr_q{w}"]
        out[f"pr_max{w}"] = cp[f"pr_max{w}"]
    out["pr_monthly_mean"] = cm["pr_monthly_mean"]
    for k in cm.files:
        if k.startswith("demo_"):
            out[k] = cm[k]
    # B ---------------------------------------------------------------------------------------------------
    names = [str(s) for s in ev["eva_durations"]]
    sl = [i for i, s in enumerate(names) if s != "day"]
    kday = names.index("day")
    out["eva_durations_h"] = np.array([float(names[i]) for i in sl], np.float32)
    out["eva_T"] = ev["eva_T"].astype(np.float32)
    for k in ("gev_loc", "gev_scale", "gev_shape", "gev_shape_lo", "gev_shape_hi", "gev_site_shape", "rl", "rl_lo", "rl_hi",
              "rl_site", "rl_site_lo", "rl_site_hi", "hw_H1", "hw_Zgev"):
        out[k] = ev[k][sl].astype(np.float32)
    for k in ("loc", "scale", "shape"):
        out[f"gev_day_{k}"] = ev[f"gev_{k}"][kday].astype(np.float32)
    out["rl_day"] = ev["rl"][kday].astype(np.float32)
    out["annmax_years"] = am["years"].astype(np.int16)
    out["annmax_durations"] = np.array(names)
    out["annmax_mm10"] = np.round(am["am"] * 10).astype(np.uint16)
    out["eva_fit_years"] = ev["fit_years"].astype(np.int16)
    # C, D ------------------------------------------------------------------------------------------------
    for k in sc.files:
        out[k] = sc[k]
    out["events_json"] = np.array(json.dumps(events))
    # E ---------------------------------------------------------------------------------------------------
    for k in ("tcwv_lat", "tcwv_lon", "tcwv_mean", "tcwv_std"):
        out[k] = ct[k].astype(np.float32)
    meta = dict(
        built=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
        source="ERA5 hourly (Copernicus C3S / ECMWF) from the Open-Meteo public S3 bucket s3://openmeteo/data/copernicus_era5 (CC-BY-4.0)",
        grid="0.25 deg regular; precipitation box lat 37.5..41.0, lon -2.5..1.0 (15 x 15); TCWV box lat 37..41.5, lon -3..3 (19 x 25)",
        record="precipitation 1940-01-01 07 UTC .. 2026-09-24 23 UTC (no gaps); TCWV used 1991-2020",
        climate_reference="1991-2020, +-15 days, quantile nodes every 5 days of a 366-day calendar (0 = 1 Jan, 59 = 29 Feb)",
        precip_windows="12/24/48/72 h sums ending 00/06/12/18 UTC; a sample belongs to the day of year of the window centre",
        eva="GEV by L-moments on annual maxima 1950-2025 (76 values), sliding windows 1/6/12/24/48/72 h; shape = regional (5x5 points); "
            "xi > 0 = heavy tail; 90 % CI = parametric bootstrap with inter-site dependence",
        obs_space="GHCN-Daily stations (16) + AEMET regional GPD (Ruiz Garcia & Nunez Mora) anchor on the Safor coast",
        units="mm, kg/m2, years")
    out["meta_json"] = np.array(json.dumps(meta))
    path = STATIC / "climate_era5.npz"
    safe_savez(path, **out)
    size = path.stat().st_size
    raw = sum(np.asarray(v).nbytes for v in out.values())
    print(f"wrote {path}: {size / 1e6:.2f} MB on disk, {raw / 1e6:.1f} MB in memory if every array is loaded, {len(out)} arrays")
    for k, v in out.items():
        v = np.asarray(v)
        print(f"  {k:28s} {str(v.dtype):8s} {v.shape}")


if __name__ == "__main__":
    main()
