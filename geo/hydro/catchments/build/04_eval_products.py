"""Evaluate candidate flow-direction products on the W grid.

A = HydroSHEDS v1 3s DIR (conditioned SRTM)
B = Copernicus GLO-30 (averaged to 3") + priority-flood fill + D8, no burning
C = Copernicus GLO-90 (bilinear to the lattice) + priority-flood + D8

Writes accumulated area (km2) rasters to scratch for later inspection:
  acc_A.npy, acc_B.npy, acc_C.npy and dir_B.npy, dir_C.npy
"""
import os
import time
import numpy as np
from common import (SCR, NROW, NCOL, cell_area_rows, cell_dx_rows, cell_dy_km,
                    downstream_index, topo_order, accumulate)
from pflood import fill_eps, d8_from_filled

area = np.repeat(cell_area_rows()[:, None], NCOL, axis=1).ravel()
hs = np.load(os.path.join(SCR, "hs_dir.npz"))["dir"]


def acc_of(dirr, tag):
    t = time.time()
    ds = downstream_index(dirr)
    order = topo_order(ds)
    w = np.where(dirr.ravel() == 255, 0.0, area)
    acc = accumulate(ds, order, w).reshape(NROW, NCOL).astype(np.float32)
    np.save(os.path.join(SCR, f"acc_{tag}.npy"), acc)
    print(tag, "acc done", round(time.time() - t, 1), "s; cells in order", order.size, "of", ds.size)
    return acc


acc_of(hs, "A")
for tag, name in (("B", "cop30"), ("C", "cop90")):
    dem = np.load(os.path.join(SCR, f"dem_{name}_3s.npy"))
    # land mask: Copernicus sea is 0 m exactly in its water-body-flattened DSM;
    # use the HydroSHEDS ocean mask (dir==255) where the DEM is <= 0, and NaN.
    valid = np.isfinite(dem) & ~((hs == 255) & (dem <= 0.5))
    sink = np.zeros_like(valid)
    t = time.time()
    z = fill_eps(np.nan_to_num(dem, nan=0.0), valid, sink, 1e-4)
    d = d8_from_filled(z, valid, sink, cell_dx_rows(), cell_dy_km())
    np.save(os.path.join(SCR, f"dir_{tag}.npy"), d)
    print(tag, "fill+d8", round(time.time() - t, 1), "s")
    acc_of(d, tag)
