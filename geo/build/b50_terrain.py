"""Terrain + grid lookups on the Riuà 0.05 deg grid -> geo/out/terrain.npz (convention in geo/out/GRID.md).

DEM: Copernicus DEM GLO-90 (COG tiles on AWS Open Data, keyless):
  https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N39_00_W001_00_DEM/Copernicus_DSM_COG_30_N39_00_W001_00_DEM.tif
Requires b10_zones.py and b30_basins.py to have run.
"""
from __future__ import annotations

import json
import time
import warnings

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import requests
import scipy.ndimage as ndi
import shapely
from shapely.geometry import box, shape

from common import DX, LAT0, LAT1, LON0, LON1, NX, NY, OUT, RAW, UA, download, find_one

warnings.filterwarnings("ignore")

# DEM mosaic: 1 deg tiles, 1200 px per degree
MLON0, MLON1, MLAT0, MLAT1 = -3, 1, 37, 42
PPD = 1200
SMOOTH_KM = 5.0  # Gaussian sigma for the smoothed gradient (FWHM ~ 12 km)


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"


def load_dem() -> np.ndarray:
    cache = RAW / "dem_mosaic.npy"
    if cache.exists():
        return np.load(cache)
    ny, nx = (MLAT1 - MLAT0) * PPD, (MLON1 - MLON0) * PPD
    dem = np.zeros((ny, nx), dtype=np.float32)
    for lat in range(MLAT0, MLAT1):
        for lon in range(MLON0, MLON1):
            name = tile_name(lat, lon)
            url = f"https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
            dest = RAW / "dem" / f"{name}.tif"
            if not dest.exists():
                for attempt in range(4):
                    try:
                        r = requests.head(url, headers={"User-Agent": UA}, timeout=60)
                        break
                    except requests.ConnectionError:
                        if attempt == 3:
                            raise
                        time.sleep(5 * (attempt + 1))
                if r.status_code == 404:
                    print(f"  {name}: no tile (open sea)")
                    continue
                download(url, dest, pause=0.5)
            with rasterio.open(dest) as src:
                a = src.read(1)
                assert a.shape == (PPD, PPD), (name, a.shape)
            r0 = (MLAT1 - (lat + 1)) * PPD
            c0 = (lon - MLON0) * PPD
            dem[r0:r0 + PPD, c0:c0 + PPD] = a
    np.save(cache, dem)
    return dem


def block(a: np.ndarray, f: int, fn) -> np.ndarray:
    h, w = a.shape
    return fn(a.reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, f * f), axis=2)


def main():
    dem = load_dem()
    ny, nx = dem.shape
    tr = rasterio.transform.from_origin(MLON0, MLAT1, 1 / PPD, 1 / PPD)

    # land mask on the DEM grid from the AEMET zones (derived from IGN BDLJE)
    shp = find_one(RAW / "zonas", "AEMET-meteoalerta-v6-zonas-32630.shp")
    g = gpd.read_file(shp, encoding="iso-8859-15").to_crs(4326)
    bb = box(MLON0, MLAT0, MLON1, MLAT1)
    land_geoms = [x for x in g.geometry.values if x.intersects(bb)]
    land = rasterio.features.rasterize(((x, 1) for x in land_geoms), out_shape=dem.shape, transform=tr, fill=0, dtype="uint8").astype(bool)

    # slope at DEM resolution
    lat_rows = MLAT1 - (np.arange(ny) + 0.5) / PPD
    dy_m = 111195.0 / PPD
    dx_m = dy_m * np.cos(np.radians(lat_rows))[:, None]
    gy = np.gradient(dem, axis=0) / dy_m
    gx = np.gradient(dem, axis=1) / dx_m
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    del gx, gy

    # window of the Riuà box inside the mosaic
    f = round(DX * PPD)  # 60 DEM px per cell
    r0 = round((MLAT1 - LAT1) * PPD)
    c0 = round((LON0 - MLON0) * PPD)
    win = (slice(r0, r0 + NY * f), slice(c0, c0 + NX * f))
    z = np.where(land[win], dem[win], np.nan)
    s = np.where(land[win], slope[win], np.nan)
    with np.errstate(all="ignore"):
        elev_mean = block(z, f, np.nanmean)
        elev_std = block(z, f, np.nanstd)
        elev_min = block(z, f, np.nanmin)
        elev_max = block(z, f, np.nanmax)
        slope_mean = block(s, f, np.nanmean)
    land_frac = block(land[win].astype(np.float32), f, np.mean)
    for a in (elev_mean, elev_std, elev_min, elev_max, slope_mean):
        a[~np.isfinite(a)] = 0.0

    # smoothed terrain gradient (m per km), sea = 0 m
    f1 = 12  # 0.01 deg
    zc = block(np.where(land, dem, 0.0).astype(np.float32), f1, np.mean)
    lat_c = MLAT1 - (np.arange(zc.shape[0]) + 0.5) * 0.01
    km_y = 0.01 * 111.195
    km_x = km_y * np.cos(np.radians(lat_c))
    zs = ndi.gaussian_filter(zc, sigma=(SMOOTH_KM / km_y, SMOOTH_KM / km_x.mean()), mode="nearest")
    dzdx = np.gradient(zs, axis=1) / km_x[:, None]          # + = terrain rises towards the east
    dzdy = -np.gradient(zs, axis=0) / km_y                  # + = terrain rises towards the north
    f2 = 5
    rr0, cc0 = round((MLAT1 - LAT1) * 100), round((LON0 - MLON0) * 100)
    w2 = (slice(rr0, rr0 + NY * f2), slice(cc0, cc0 + NX * f2))
    dzdx_c = block(dzdx[w2], f2, np.mean)
    dzdy_c = block(dzdy[w2], f2, np.mean)
    elev_smooth = block(zs[w2], f2, np.mean)

    # ---- basin / zone membership from 0.001 deg rasters
    zr = np.load(RAW / "basin_raster.npz")
    lab, ids, ext, res = zr["lab"], [str(x) for x in zr["ids"]], zr["ext"], float(zr["res"])
    f3 = round(DX / res)  # 50
    R0 = round((ext[3] - LAT1) / res)
    C0 = round((LON0 - ext[0]) / res)
    labw = lab[R0:R0 + NY * f3, C0:C0 + NX * f3].astype(np.int32)
    zones = json.loads((OUT / "zones.geojson").read_text(encoding="utf-8"))["features"]
    zcodes = [zf["properties"]["code"] for zf in zones]
    znames = [zf["properties"]["name"] for zf in zones]
    trb = rasterio.transform.from_origin(LON0, LAT1, res, res)
    zonw = rasterio.features.rasterize(((shape(zf["geometry"]), k + 1) for k, zf in enumerate(zones)),
                                       out_shape=labw.shape, transform=trb, fill=0, dtype="int16").astype(np.int32)
    topo = json.loads((OUT / "basins_topology.json").read_text(encoding="utf-8"))
    assert topo["order"] == ids
    nU, nZ = len(ids), len(zcodes)
    # full-raster CV overlap per unit (units may extend outside the box)
    Hh, Ww = lab.shape
    tr_full = rasterio.transform.from_origin(ext[0], ext[3], res, res)
    cv_full = rasterio.features.rasterize(((shape(zf["geometry"]), 1) for zf in zones), out_shape=lab.shape,
                                          transform=tr_full, fill=0, dtype="uint8").astype(bool)
    lat_f = ext[3] - (np.arange(Hh) + 0.5) * res
    row_km2 = (res * 111.195) ** 2 * np.cos(np.radians(lat_f))
    wfull = np.repeat(row_km2, Ww)
    unit_km2 = np.bincount(lab.ravel(), weights=wfull, minlength=nU + 1)[1:]
    unit_cv_km2 = np.bincount(lab.ravel(), weights=wfull * cv_full.ravel(), minlength=nU + 1)[1:]
    in_cv = unit_cv_km2 >= 1.0
    drains_cv = in_cv.copy()
    for k, uid in enumerate(ids):
        drains_cv[k] = in_cv[k] or any(in_cv[ids.index(d)] for d in topo["units"][uid]["downstream_chain"])
    del cv_full, wfull

    def to_cells(a):  # (NY*f3, NX*f3) north-up -> (ncell, f3*f3) with cells ordered north-up row-major
        return a.reshape(NY, f3, NX, f3).transpose(0, 2, 1, 3).reshape(NY * NX, f3 * f3)

    lat_rows_w = LAT1 - (np.arange(NY) + 0.5) * DX
    cell_km2_row = (DX * 111.195) ** 2 * np.cos(np.radians(lat_rows_w))

    def membership(raster, n):
        cells = to_cells(raster)
        key = (np.arange(NY * NX, dtype=np.int64)[:, None] * (n + 1) + cells).ravel()
        cnt = np.bincount(key, minlength=NY * NX * (n + 1)).reshape(NY * NX, n + 1)
        frac = cnt[:, 1:] / float(f3 * f3)                      # fraction of each cell covered by each unit
        major = np.where(frac.max(axis=1) > 0, frac.argmax(axis=1), -1)
        return frac, major

    bfrac, bmajor = membership(labw, nU)
    zfrac, zmajor = membership(zonw, nZ)
    cv_frac = zfrac.sum(axis=1)
    dom_frac = np.maximum(cv_frac, 0)  # CV ...
    dom_frac = np.minimum(1.0, np.maximum(cv_frac, (bfrac * drains_cv[None, :]).sum(axis=1)))
    # union of (CV) and (unit draining into the CV), computed exactly on the fine raster
    fine_dom = (zonw > 0) | ((labw > 0) & np.concatenate([[False], drains_cv])[labw])
    dom_frac = to_cells(fine_dom).mean(axis=1)

    def flip(a_flat):  # north-up flat -> (NY, NX) south-up
        return a_flat.reshape(NY, NX)[::-1].copy()

    def csr(frac, row_weights):
        """per unit: flat cell index (south-up, j*NX+i), weight (normalised over the box), fraction of the cell."""
        ptr, cell, wgt, cfrac, cover = [0], [], [], [], []
        km2_cell = np.repeat(cell_km2_row, NX)  # north-up
        jj = (NY - 1) - (np.arange(NY * NX) // NX)
        ii = np.arange(NY * NX) % NX
        south_idx = jj * NX + ii
        for k in range(frac.shape[1]):
            nz = np.where(frac[:, k] > 0)[0]
            a = frac[nz, k] * km2_cell[nz]
            order = np.argsort(south_idx[nz])
            cell += list(south_idx[nz][order])
            wgt += list((a / a.sum())[order]) if len(nz) else []
            cfrac += list(frac[nz, k][order])
            ptr.append(len(cell))
            cover.append(float(a.sum()))
        return (np.array(ptr, dtype=np.int32), np.array(cell, dtype=np.int32), np.array(wgt, dtype=np.float32),
                np.array(cfrac, dtype=np.float32), np.array(cover))

    bptr, bcell, bw, bcf, bkm2_box = csr(bfrac, None)
    zptr, zcell, zw, zcf, zkm2_box = csr(zfrac, None)
    names = [topo["units"][u]["name"] for u in ids]

    lon_c = LON0 + (np.arange(NX) + 0.5) * DX
    lat_c2 = LAT0 + (np.arange(NY) + 0.5) * DX
    up = lambda a: a[::-1].astype(np.float32).copy()  # noqa: E731  north-up (NY,NX) -> south-up
    out = dict(
        lon=lon_c, lat=lat_c2,
        elev_mean=up(elev_mean), elev_std=up(elev_std), elev_min=up(elev_min), elev_max=up(elev_max),
        relief=up(elev_max - elev_min), slope_mean_deg=up(slope_mean), land_frac=up(land_frac),
        elev_smooth=up(elev_smooth), dzdx=up(dzdx_c), dzdy=up(dzdy_c),
        basin_idx=flip(bmajor).astype(np.int16), zone_idx=flip(zmajor).astype(np.int16),
        cv_frac=flip(cv_frac).astype(np.float32), domain_frac=flip(dom_frac).astype(np.float32),
        in_domain=(flip(dom_frac) > 0.02) & (up(land_frac) > 0.0),
        basin_ids=np.array(ids), basin_names=np.array(names),
        basin_area_km2=unit_km2.astype(np.float32), basin_frac_in_box=(bkm2_box / np.maximum(unit_km2, 1e-9)).astype(np.float32),
        basin_in_cv=in_cv, basin_drains_to_cv=drains_cv,
        basin_w_ptr=bptr, basin_w_cell=bcell, basin_w=bw, basin_w_cellfrac=bcf,
        zone_codes=np.array(zcodes), zone_names=np.array(znames),
        zone_w_ptr=zptr, zone_w_cell=zcell, zone_w=zw, zone_w_cellfrac=zcf,
        grid=np.array([LON0, LON1, LAT0, LAT1, DX]),
    )
    np.savez_compressed(OUT / "terrain.npz", **out)
    print(f"terrain.npz {(OUT / 'terrain.npz').stat().st_size / 1e3:.0f} KB")
    print(f"  land cells {(out['land_frac'] > 0).sum()}, in_domain {out['in_domain'].sum()} of {NX * NY}; "
          f"units with cells {int((np.diff(bptr) > 0).sum())}/{nU}; draining to CV {int(drains_cv.sum())}")
    print(f"  elev max {out['elev_mean'].max():.0f} m; |grad| max {np.hypot(out['dzdx'], out['dzdy']).max():.1f} m/km")


if __name__ == "__main__":
    main()
