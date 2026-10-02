"""Continuous hourly rain on the Riuà grid, Sep 2024 -> now, for the flood calibration.

Base: the 5-min rain of the SAIH Júcar pluviometers (hindcast/obs/flows/rain_hourly.npz, built by saih_series.py),
interpolated with inverse squared distance from the 6 nearest stations within 40 km that report in that hour
(cells farther than 40 km from every reporting station stay NaN: the Segura and Ebro parts of the box).
Overlay: the hours covered by a hindcast truth episode (hindcast/truth/*.npz, radar + ~1000 gauges) replace the
interpolation, they are the better analysis.

Output (gitignored): hindcast/obs/flows/rain_grid.npz  t_end (datetime64[h], UTC), p (T, NY*NX) float32 mm/h,
                     src (T,) uint8: 0 gauges only, 1 truth episode

    py -3.11 geo/hydro/gauges/rain_grid.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from riua.core import grid                                # noqa: E402

FLOWS = ROOT / "hindcast" / "obs" / "flows"
KNN, RMAX_KM = 6, 40.0


def build(truth_only=False):
    if truth_only or not (FLOWS / "rain_hourly.npz").exists():      # only the hindcast episodes (no gauge download yet)
        t = np.arange(np.datetime64("2024-09-01T01", "h"), np.datetime64("now", "h"))
        p, lat, lon = np.full((3, len(t)), np.nan, np.float32), np.array([39.0, 39.1, 39.2]), np.array([-0.5, -0.6, -0.7])
    else:
        z = np.load(FLOWS / "rain_hourly.npz")
        t, p, lat, lon = z["t_end"], z["p"], z["lat"], z["lon"]
    ok_st = np.isfinite(lat) & np.isfinite(lon)
    p, lat, lon = p[ok_st], lat[ok_st], lon[ok_st]
    # a stuck / absurd gauge hour: more than 150 mm in one hour is not rain
    p = np.where(p > 150.0, np.nan, p)
    gy = grid.LAT0 + (np.arange(grid.NY) + 0.5) * grid.D
    gx = grid.LON0 + (np.arange(grid.NX) + 0.5) * grid.D
    glon, glat = np.meshgrid(gx, gy)
    kx = 111.2 * np.cos(np.deg2rad(39.3))
    cells = np.column_stack([glon.ravel() * kx, glat.ravel() * 111.2])
    sxy = np.column_stack([lon * kx, lat * 111.2])
    out = np.full((len(t), cells.shape[0]), np.nan, np.float32)
    # stations change their reporting state rarely: group the hours by the set of reporting stations
    have = np.isfinite(p)                                 # (S, T)
    keys = np.packbits(have, axis=0).T                    # (T, bytes)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    for g, k0 in enumerate(first):
        hours = np.nonzero(inv == g)[0]
        st = np.nonzero(have[:, k0])[0]
        if st.size < 3:
            continue
        tree = cKDTree(sxy[st])
        d, j = tree.query(cells, k=min(KNN, st.size), distance_upper_bound=RMAX_KM)
        if d.ndim == 1:
            d, j = d[:, None], j[:, None]
        w = np.where(np.isfinite(d), 1.0 / np.maximum(d, 1.0) ** 2, 0.0)
        jj = np.where(np.isfinite(d), j, 0)
        ws = w.sum(axis=1)
        good = ws > 0
        wn = (w[good] / ws[good, None]).astype(np.float32)
        ph = p[st][:, hours]                              # (s, H)
        wet = np.nonzero(np.nanmax(ph, axis=0) > 0)[0]
        res = np.zeros((hours.size, good.sum()), np.float32)
        if wet.size:
            res[wet] = np.einsum("ck,ckh->hc", wn, ph[jj[good]][:, :, wet])
        blk = np.full((hours.size, cells.shape[0]), np.nan, np.float32)
        blk[:, good] = res
        out[hours] = blk
    src = np.zeros(len(t), np.uint8)
    gauges_only = out.copy()
    for f in sorted(glob.glob(str(ROOT / "hindcast" / "truth" / "*.npz"))):
        zt = np.load(f, allow_pickle=True)
        te = zt["t_end"].astype("datetime64[h]")
        if len(te) < 12:
            continue
        o = zt["o_mean"].reshape(len(te), -1)
        pos = (te - t[0]).astype(int)
        inside = (pos >= 0) & (pos < len(t))
        okh = inside & np.isfinite(o).any(axis=1)
        out[pos[okh]] = np.nan_to_num(o[okh]).astype(np.float32)
        src[pos[okh]] = 1
    # p_gauges: the interpolation alone in the truth hours (to compare the two rain fields)
    np.savez_compressed(FLOWS / "rain_grid.npz", t_end=t, p=out, src=src, p_gauges=gauges_only[src == 1])
    print("rain_grid", out.shape, "truth hours", int(src.sum()), "cells with gauge coverage",
          int(np.isfinite(out[src == 0]).any(axis=0).sum()), "of", cells.shape[0])


if __name__ == "__main__":
    build(truth_only=len(sys.argv) > 1 and sys.argv[1] == "truth")
