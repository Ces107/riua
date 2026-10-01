"""Priority-flood depression filling with epsilon gradient (Barnes et al. 2014,
"Priority-Flood: An Optimal Depression-Filling and Watershed-Labeling Algorithm
for Digital Elevation Models", Alg. 3) + steepest-descent D8 in ESRI encoding.

Outlets (seeds of the flood): every valid cell that touches the sea / nodata or
the raster edge, plus any cell flagged in `sink` (kept endorheic sinks).
"""
import heapq
import numpy as np
from numba import njit


@njit(cache=True)
def fill_eps(dem, valid, sink, eps):
    nr, nc = dem.shape
    z = dem.astype(np.float64).copy()
    closed = np.zeros((nr, nc), dtype=np.bool_)
    heap = [(0.0, 0)]
    heap.pop()
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            seed = sink[r, c]
            if not seed:
                if r == 0 or c == 0 or r == nr - 1 or c == nc - 1:
                    seed = True
                else:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if not valid[r + dr, c + dc]:
                                seed = True
            if seed:
                heapq.heappush(heap, (z[r, c], r * nc + c))
                closed[r, c] = True
    while len(heap) > 0:
        e, i = heapq.heappop(heap)
        r = i // nc
        c = i - r * nc
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                r2 = r + dr
                c2 = c + dc
                if r2 < 0 or r2 >= nr or c2 < 0 or c2 >= nc:
                    continue
                if closed[r2, c2] or not valid[r2, c2]:
                    continue
                closed[r2, c2] = True
                if z[r2, c2] <= e:
                    z[r2, c2] = e + eps
                heapq.heappush(heap, (z[r2, c2], r2 * nc + c2))
    return z


@njit(cache=True)
def d8_from_filled(z, valid, sink, dxrow, dy):
    """steepest descent; cells next to sea drain to the sea (code pointing to it)."""
    nr, nc = z.shape
    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    drow = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dcol = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    out = np.full((nr, nc), 255, dtype=np.uint8)
    for r in range(nr):
        dx = dxrow[r]
        dd = (dx * dx + dy * dy) ** 0.5
        for c in range(nc):
            if not valid[r, c]:
                continue
            if sink[r, c]:
                out[r, c] = 0
                continue
            best = 0.0
            bk = -1
            sea = -1
            for k in range(8):
                r2 = r + drow[k]
                c2 = c + dcol[k]
                if r2 < 0 or r2 >= nr or c2 < 0 or c2 >= nc:
                    if sea < 0:
                        sea = k
                    continue
                if not valid[r2, c2]:
                    if sea < 0 or (k % 2 == 0):
                        sea = k
                    continue
                dist = dd
                if drow[k] == 0:
                    dist = dx
                elif dcol[k] == 0:
                    dist = dy
                s = (z[r, c] - z[r2, c2]) / dist
                if s > best:
                    best = s
                    bk = k
            if sea >= 0:
                out[r, c] = codes[sea]
            elif bk >= 0:
                out[r, c] = codes[bk]
            else:
                out[r, c] = 0
    return out
