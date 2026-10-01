"""Bring everything onto the W grid (HydroSHEDS 3" lattice, see common.py):
 - hs_dir.npz / hs_acc.npz were cut from the HydroSHEDS v1 3s Europe GeoTIFFs
   (dir: local zip via /vsizip/, acc: /vsizip//vsicurl/ window read).
 - Copernicus GLO-30 tiles are mosaicked and averaged 1" -> 3" onto the same lattice
   (and GLO-90 bilinear, for comparison).
Output: scratch/h1-catchments/dem_cop30_3s.npy (float32, NaN = sea/no tile), dem_cop90_3s.npy
"""
import glob
import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from common import SCR, W_LON0, W_LAT1, RES, NROW, NCOL

dst_tr = from_origin(W_LON0, W_LAT1, RES, RES)
for tag, rs in (("cop30", Resampling.average), ("cop90", Resampling.bilinear)):
    out = np.full((NROW, NCOL), np.nan, dtype=np.float32)
    for f in sorted(glob.glob(os.path.join(SCR, tag, "*.tif"))):
        with rasterio.open(f) as ds:
            tmp = np.full((NROW, NCOL), np.nan, dtype=np.float32)
            reproject(rasterio.band(ds, 1), tmp, src_transform=ds.transform, src_crs=ds.crs,
                      dst_transform=dst_tr, dst_crs="EPSG:4326", resampling=rs,
                      src_nodata=ds.nodata, dst_nodata=np.nan)
            ok = np.isfinite(tmp)
            out[ok] = tmp[ok]
        print(tag, os.path.basename(f), int(ok.sum()))
    np.save(os.path.join(SCR, f"dem_{tag}_3s.npy"), out)
    print(tag, "finite", int(np.isfinite(out).sum()), "min", np.nanmin(out), "max", np.nanmax(out))
