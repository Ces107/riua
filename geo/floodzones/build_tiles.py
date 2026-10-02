"""Map tiles of the official 500-year flood zones (SNCZI, MITECO, 2nd cycle, 14 Jan 2026) for the Riuà box.

    py -3.11 geo/floodzones/build_tiles.py <path to Q500_2Ciclo_PB_*.shp> [zmin zmax]

Input: the shapefile inside laminasPB-q500.zip (https://gis.miteco.gob.es/descargas/app/DescargaFichero?f=laminasPB-q500.zip,
16 million vertices in the box: too heavy as vectors for a phone). Output: web/geo/zi500/{z}/{x}/{y}.png, palette PNG,
transparent, only the tiles that contain a zone. Style: the river blue of the map, a flat wash with a darker edge
from zoom 11, so the zones read as water without hiding what is under them.
"""
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "geo" / "zi500"
CACHE = ROOT / "scratch" / "q500" / "zi500_3857.npy"
BOX = (-2.4, 37.6, 0.8, 41.0)
FILL = (52, 104, 150)           # #346896, the river ink
A_FILL, A_EDGE = 105, 235        # alpha 0..255
SS = 2                           # supersampling for smooth edges
BLOCK = 8                        # tiles per side rasterised at once
R = 6378137.0 * math.pi

_G = None


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n


def bounds(x0, y0, x1, y1, z):
    """3857 bounds of the tile range [x0, x1) x [y0, y1)."""
    s = 2 * R / 2 ** z
    return (-R + x0 * s, R - y1 * s, -R + x1 * s, R - y0 * s)


def prepare(shp):
    import geopandas as gpd
    import pyproj
    import shapely
    tr = pyproj.Transformer.from_crs(4326, 25830, always_xy=True)
    xs, ys = zip(*[tr.transform(a, b) for a in (BOX[0], BOX[2]) for b in (BOX[1], BOX[3])])
    g = gpd.read_file(shp, bbox=(min(xs), min(ys), max(xs), max(ys)), columns=[], engine="pyogrio")
    g = g.to_crs(3857).explode(index_parts=False)
    np.save(CACHE, shapely.to_wkb(g.geometry.values), allow_pickle=True)
    print("prepared", len(g), "polygons", flush=True)


def _init(z):
    global _G
    import shapely
    geoms = shapely.from_wkb(np.load(CACHE, allow_pickle=True))
    px = 2 * R / 2 ** z / 256
    if z < 15:
        geoms = shapely.simplify(geoms, px * 0.35)
    geoms = geoms[~shapely.is_empty(geoms)]
    _G = (geoms, shapely.STRtree(geoms))


def _block(args):
    import shapely
    from PIL import Image
    from rasterio import features, transform
    from scipy import ndimage
    z, bx, by, nx, ny = args
    geoms, tree = _G
    b = bounds(bx, by, bx + nx, by + ny, z)
    idx = tree.query(shapely.box(*b))
    if len(idx) == 0:
        return 0
    W, H = nx * 256 * SS, ny * 256 * SS
    m = features.rasterize(((geoms[i], 1) for i in idx), out_shape=(H, W),
                           transform=transform.from_bounds(*b, W, H), all_touched=z <= 9, dtype="uint8")
    if not m.any():
        return 0
    a = m.astype(np.float32) * A_FILL
    if z >= 11:
        edge = m.astype(bool) & ~ndimage.binary_erosion(m, iterations=SS * (2 if z >= 13 else 1))
        a[edge] = A_EDGE
    n = 0
    for j in range(ny):
        for i in range(nx):
            t = a[j * 256 * SS:(j + 1) * 256 * SS, i * 256 * SS:(i + 1) * 256 * SS]
            if not t.any():
                continue
            t = t.reshape(256, SS, 256, SS).mean(axis=(1, 3))
            rgba = np.zeros((256, 256, 4), np.uint8)
            rgba[..., :3] = FILL
            rgba[..., 3] = np.clip(np.rint(t), 0, 255).astype(np.uint8)
            f = OUT / str(z) / str(bx + i) / f"{by + j}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, "RGBA").quantize(colors=16, method=Image.Quantize.FASTOCTREE).save(f, optimize=True)
            n += 1
    return n


def main():
    shp = sys.argv[1]
    zmin, zmax = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (7, 14)
    if not CACHE.exists():
        prepare(shp)
    for z in range(zmin, zmax + 1):
        t = time.time()
        x0, y0 = (int(v) for v in lonlat_to_tile(BOX[0], BOX[3], z))
        x1, y1 = (int(v) + 1 for v in lonlat_to_tile(BOX[2], BOX[1], z))
        jobs = [(z, bx, by, min(BLOCK, x1 - bx), min(BLOCK, y1 - by))
                for bx in range(x0, x1, BLOCK) for by in range(y0, y1, BLOCK)]
        with ProcessPoolExecutor(3, initializer=_init, initargs=(z,)) as ex:
            n = sum(ex.map(_block, jobs))
        print(f"z{z}: {n} tiles in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
