"""Shared constants and D8 helpers for the h1-catchments build.

Working raster ("W grid"): HydroSHEDS 3 arc-second lattice, window
lon -3.5..1.0, lat 37.0..41.2 -> 5040 rows x 5400 cols, row 0 = north.
Direction encoding = ESRI/HydroSHEDS: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE,
0 = sink (endorheic), 255 = ocean / nodata.
"""
import os
import numpy as np
from numba import njit

ROOT = r"C:\Users\cpereiro\IdeaProjects\riua"
SCR = os.path.join(ROOT, "scratch", "h1-catchments")
OUT = os.path.join(ROOT, "geo", "hydro", "catchments", "out")

W_LON0, W_LAT1 = -3.5, 41.2          # west edge, north edge
W_LON1, W_LAT0 = 1.0, 37.0
RES = 3.0 / 3600.0
NROW, NCOL = 5040, 5400

# Riua grid
G_LON0, G_LAT0, G_D, G_NX, G_NY = -2.4, 37.6, 0.05, 64, 68

R_EARTH = 6371.0088  # km

# ESRI codes -> (drow, dcol)
CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
DROW = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
DCOL = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)


def row_lat(r):
    """latitude of the centre of row r (float or array)"""
    return W_LAT1 - (np.asarray(r) + 0.5) * RES


def col_lon(c):
    return W_LON0 + (np.asarray(c) + 0.5) * RES


def rc_of(lat, lon):
    r = int(np.floor((W_LAT1 - lat) / RES))
    c = int(np.floor((lon - W_LON0) / RES))
    return r, c


def cell_area_rows():
    """km2 of one cell for every row"""
    lat = np.deg2rad(row_lat(np.arange(NROW)))
    d = np.deg2rad(RES)
    return (R_EARTH * d) * (R_EARTH * d * np.cos(lat))


def cell_dy_km():
    return R_EARTH * np.deg2rad(RES)


def cell_dx_rows():
    return R_EARTH * np.deg2rad(RES) * np.cos(np.deg2rad(row_lat(np.arange(NROW))))


@njit(cache=True)
def downstream_index(dirr):
    """flat index of the receiving cell, -1 for sink / ocean / off-grid"""
    nr, nc = dirr.shape
    ds = np.full(nr * nc, -1, dtype=np.int32)
    for r in range(nr):
        for c in range(nc):
            d = dirr[r, c]
            dr = 0
            dc = 0
            if d == 1:
                dc = 1
            elif d == 2:
                dr = 1; dc = 1
            elif d == 4:
                dr = 1
            elif d == 8:
                dr = 1; dc = -1
            elif d == 16:
                dc = -1
            elif d == 32:
                dr = -1; dc = -1
            elif d == 64:
                dr = -1
            elif d == 128:
                dr = -1; dc = 1
            else:
                continue
            r2 = r + dr
            c2 = c + dc
            if r2 < 0 or r2 >= nr or c2 < 0 or c2 >= nc:
                continue
            if dirr[r2, c2] == 255:
                continue
            ds[r * nc + c] = r2 * nc + c2
    return ds


@njit(cache=True)
def topo_order(ds):
    """cells ordered upstream -> downstream (Kahn)"""
    n = ds.size
    indeg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        j = ds[i]
        if j >= 0:
            indeg[j] += 1
    order = np.empty(n, dtype=np.int32)
    k = 0
    for i in range(n):
        if indeg[i] == 0:
            order[k] = i
            k += 1
    h = 0
    while h < k:
        i = order[h]
        h += 1
        j = ds[i]
        if j >= 0:
            indeg[j] -= 1
            if indeg[j] == 0:
                order[k] = j
                k += 1
    return order[:k]


@njit(cache=True)
def accumulate(ds, order, w):
    acc = w.copy()
    for k in range(order.size):
        i = order[k]
        j = ds[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc


@njit(cache=True)
def label_upstream(ds, order, outlet):
    """boolean mask (flat) of all cells draining through `outlet` (inclusive)"""
    n = ds.size
    m = np.zeros(n, dtype=np.bool_)
    m[outlet] = True
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        j = ds[i]
        if j >= 0 and m[j]:
            m[i] = True
    return m


@njit(cache=True)
def trace_down(ds, start, maxn):
    out = np.empty(maxn, dtype=np.int32)
    k = 0
    i = start
    while i >= 0 and k < maxn:
        out[k] = i
        k += 1
        i = ds[i]
    return out[:k]
