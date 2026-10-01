"""The Riuà analysis grid and the spatial operators that work on it.

The grid is a regular 0.05 deg lat/lon mesh over the Comunitat Valenciana and the
upstream part of its river basins. Arrays are indexed [j, i] with j growing to the
north and i to the east; cell (j, i) is centred on (LAT0 + (j + 0.5) * D, LON0 + (i + 0.5) * D).

Why 0.05 deg (~4.3 x 5.6 km): the convection-permitting inputs are 1.3-2.5 km models
whose individual cells carry no predictable skill; skill only appears once the field is
read through a neighbourhood of tens of km (Roberts and Lean 2008). The radar composite
is ~1-2 km and the ensembles are 25 km. A 5 km mesh is fine enough to separate one
ravine catchment from the next and coarse enough not to pretend street-level precision.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

LON0, LAT0, D = -2.4, 37.6, 0.05
NX, NY = 64, 68
LONS = LON0 + (np.arange(NX) + 0.5) * D
LATS = LAT0 + (np.arange(NY) + 0.5) * D
BBOX = (LON0, LAT0, LON0 + NX * D, LAT0 + NY * D)  # west, south, east, north

KM_PER_DEG_LAT = 111.2
KM_PER_DEG_LON = 111.2 * float(np.cos(np.deg2rad(39.3)))  # 86 km at the domain centre
DX_KM = D * KM_PER_DEG_LON
DY_KM = D * KM_PER_DEG_LAT


def mesh() -> tuple[np.ndarray, np.ndarray]:
    """2-D arrays (lat, lon) of cell centres."""
    lon2, lat2 = np.meshgrid(LONS, LATS)
    return lat2, lon2


def cell_of(lat: float, lon: float) -> tuple[int, int] | None:
    j = int(np.floor((lat - LAT0) / D))
    i = int(np.floor((lon - LON0) / D))
    if 0 <= j < NY and 0 <= i < NX:
        return j, i
    return None


def disk(radius_km: float) -> np.ndarray:
    """Boolean footprint of a circle of the given radius, in grid cells."""
    rj = int(np.floor(radius_km / DY_KM))
    ri = int(np.floor(radius_km / DX_KM))
    jj, ii = np.mgrid[-rj:rj + 1, -ri:ri + 1]
    return (jj * DY_KM) ** 2 + (ii * DX_KM) ** 2 <= radius_km ** 2 + 1e-9


def neighbourhood_max(field: np.ndarray, radius_km: float) -> np.ndarray:
    """Maximum of `field` within `radius_km` of every cell (last two axes are j, i).

    This is the building block of the neighbourhood maximum ensemble probability
    (Schwartz and Sobash 2017): the question asked of each member is not "does it
    rain X exactly here" but "does it rain X within R km of here".
    """
    if radius_km <= 0:
        return field
    fp = disk(radius_km)
    if fp.size == 1:
        return field
    if field.ndim == 2:
        return ndimage.maximum_filter(field, footprint=fp, mode="nearest")
    out = np.empty_like(field)
    flat_in = field.reshape(-1, *field.shape[-2:])
    flat_out = out.reshape(-1, *field.shape[-2:])
    for k in range(flat_in.shape[0]):
        ndimage.maximum_filter(flat_in[k], footprint=fp, mode="nearest", output=flat_out[k])
    return out


def shift(field: np.ndarray, dj: int, di: int) -> np.ndarray:
    """Translate the field by whole cells, repeating the edge values."""
    if dj == 0 and di == 0:
        return field
    ny, nx = field.shape[-2:]
    js = np.clip(np.arange(ny) - dj, 0, ny - 1)
    is_ = np.clip(np.arange(nx) - di, 0, nx - 1)
    return field[..., js[:, None], is_[None, :]]


class Regridder:
    """Maps a source model grid onto the Riuà grid.

    Fine sources (spacing <= ~0.08 deg) are *sampled*: each Riuà cell takes the value of
    the nearest source point. Sampling, not block-averaging, keeps the model's own
    intensity distribution, and it is exactly what the archived-forecast API returns for
    a point, so live products and hindcasts see statistically identical inputs.
    Coarse sources are interpolated bilinearly (through Delaunay-free inverse-distance
    weights of the 4 nearest points) so 25 km boxes do not print on the map.
    """

    def __init__(self, src_lat: np.ndarray, src_lon: np.ndarray, coarse: bool | None = None):
        src_lat = np.asarray(src_lat, dtype=np.float64)
        src_lon = np.asarray(src_lon, dtype=np.float64)
        if src_lat.ndim == 1 and src_lon.ndim == 1 and src_lat.size != src_lon.size:
            src_lon, src_lat = np.meshgrid(src_lon, src_lat)
        elif src_lat.ndim == 1 and src_lon.ndim == 1:
            # ambiguous square case or a plain list of points: treat as point list
            pass
        self.src_shape = src_lat.shape
        pts = np.column_stack([src_lat.ravel() * KM_PER_DEG_LAT, src_lon.ravel() * KM_PER_DEG_LON])
        tree = cKDTree(pts)
        lat2, lon2 = mesh()
        tgt = np.column_stack([lat2.ravel() * KM_PER_DEG_LAT, lon2.ravel() * KM_PER_DEG_LON])
        if coarse is None:
            d2, _ = tree.query(pts[: min(len(pts), 200)], k=2)
            coarse = float(np.median(d2[:, 1])) > 8.0  # km between neighbours
        self.coarse = bool(coarse)
        if self.coarse:
            k = min(4, len(pts))
            dist, idx = tree.query(tgt, k=k)
            dist = np.atleast_2d(dist.reshape(len(tgt), k))
            idx = np.atleast_2d(idx.reshape(len(tgt), k))
            w = 1.0 / np.maximum(dist, 0.05) ** 2
            self.w = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)
            self.idx = idx
            self.maxdist = dist[:, 0].reshape(NY, NX)
        else:
            dist, idx = tree.query(tgt, k=1)
            self.idx = idx
            self.w = None
            self.maxdist = dist.reshape(NY, NX)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """values: (..., *src_shape) -> (..., NY, NX) float32. NaN where the source has no data."""
        v = np.asarray(values, dtype=np.float32)
        lead = v.shape[: v.ndim - len(self.src_shape)]
        flat = v.reshape(*lead, -1)
        if self.coarse:
            g = flat[..., self.idx]  # (..., ncell, k)
            ok = np.isfinite(g)
            ww = np.where(ok, self.w, 0.0)
            s = ww.sum(axis=-1)
            out = np.where(s > 0, (np.where(ok, g, 0.0) * ww).sum(axis=-1) / np.maximum(s, 1e-9), np.nan)
        else:
            out = flat[..., self.idx]
        return out.reshape(*lead, NY, NX).astype(np.float32)
