"""Map tiles of the official 500-year flood zones (SNCZI, MITECO, 2nd cycle, 14 Jan 2026) for the Riuà box.

    py -3.11 geo/floodzones/build_tiles.py <path to Q500_2Ciclo_PB_*.shp>

Input: the shapefile inside laminasPB-q500.zip (https://gis.miteco.gob.es/descargas/app/DescargaFichero?f=laminasPB-q500.zip,
33 million vertices in the box: far too heavy as vectors for a phone). Output: web/geo/zi500/{z}/{x}/{y}.png for zooms
7-14, palette PNGs, transparent, only the tiles that contain a zone, and index.json listing them.

Method: the polygons are rasterised once, at zoom 14 with 2x supersampling, one zoom-10 tile (16 x 16 zoom-14 tiles)
at a time; every coarser zoom is the averaged coverage of that raster, so nothing has to be simplified.
Style: the river ink of the map as a light wash, a crisp darker edge from zoom 11.
"""
import json
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
FILL = (52, 104, 150)            # #346896, the river ink
A_FILL, A_EDGE = 100, 230        # alpha 0..255
ZTOP, ZBLK, ZMIN = 14, 10, 7
SS = 2
R = 6378137.0 * math.pi

_G = None


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n


def tile_bounds(x, y, z):
    s = 2 * R / 2 ** z
    return (-R + x * s, R - (y + 1) * s, -R + (x + 1) * s, R - y * s)


def prepare(shp):
    import geopandas as gpd
    import pyproj
    import shapely
    tr = pyproj.Transformer.from_crs(4326, 25830, always_xy=True)
    xs, ys = zip(*[tr.transform(a, b) for a in (BOX[0], BOX[2]) for b in (BOX[1], BOX[3])])
    g = gpd.read_file(shp, bbox=(min(xs), min(ys), max(xs), max(ys)), columns=[], engine="pyogrio")
    g = g.to_crs(3857).explode(index_parts=False)
    np.save(CACHE, shapely.to_wkb(g.geometry.values), allow_pickle=True)


def _init():
    global _G
    import shapely
    geoms = shapely.from_wkb(np.load(CACHE, allow_pickle=True))
    _G = (geoms, shapely.STRtree(geoms))


def _save(alpha, z, x, y):
    from PIL import Image
    rgba = np.zeros((256, 256, 4), np.uint8)
    rgba[..., :3] = FILL
    rgba[..., 3] = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)
    f = OUT / str(z) / str(x) / f"{y}.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").quantize(colors=12, method=Image.Quantize.FASTOCTREE).save(f, optimize=True)


def style(cov, z):
    """Coverage (0..1, 256 x 256 or larger) -> alpha: a wash, plus a crisp edge from zoom 11."""
    from scipy import ndimage
    a = cov * A_FILL
    if z >= 11:
        b = cov >= 0.5
        edge = b & ~ndimage.binary_erosion(b, iterations=2 if z >= 13 else 1, border_value=1)
        a = np.where(edge, A_EDGE, a)
    return a


def _block(args):
    import shapely
    from rasterio import features, transform
    X, Y = args
    geoms, tree = _G
    b = tile_bounds(X, Y, ZBLK)
    idx = tree.query(shapely.box(*b))
    if len(idx) == 0:
        return X, Y, None, 0
    n = 256 * 2 ** (ZTOP - ZBLK) * SS                     # 8192
    m = features.rasterize(((geoms[i], 1) for i in idx), out_shape=(n, n),
                           transform=transform.from_bounds(*b, n, n), dtype="uint8")
    if not m.any():
        return X, Y, None, 0
    m = m.astype(np.float32)
    count = 0
    for z in range(ZTOP, ZBLK - 1, -1):
        k = 2 ** (z - ZBLK)                                # tiles per side of the block at this zoom
        f = n // (256 * k)                                 # pixels averaged into one
        cov = m.reshape(256 * k, f, 256 * k, f).mean(axis=(1, 3))
        a = style(cov, z)
        for j in range(k):
            for i in range(k):
                t = a[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256]
                if t.max() >= 1:
                    _save(t, z, X * k + i, Y * k + j)
                    count += 1
        if z == ZBLK:
            return X, Y, cov.astype(np.float32), count
    return X, Y, None, count


def main():
    shp = sys.argv[1]
    if not CACHE.exists():
        prepare(shp)
    t = time.time()
    x0, y0 = (int(v) for v in lonlat_to_tile(BOX[0], BOX[3], ZBLK))
    x1, y1 = (int(v) for v in lonlat_to_tile(BOX[2], BOX[1], ZBLK))
    jobs = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    covs, total = {}, 0
    with ProcessPoolExecutor(3, initializer=_init) as ex:
        for X, Y, cov, c in ex.map(_block, jobs):
            total += c
            if cov is not None:
                covs[(X, Y)] = cov
            print(f"block {X},{Y}: {c} tiles ({time.time() - t:.0f} s)", flush=True)
    # coarser zooms from the zoom-10 coverage
    for z in range(ZBLK - 1, ZMIN - 1, -1):
        nxt = {}
        for (X, Y), cov in covs.items():
            p = (X // 2, Y // 2)
            big = nxt.setdefault(p, np.zeros((512, 512), np.float32))
            big[(Y % 2) * 256:(Y % 2 + 1) * 256, (X % 2) * 256:(X % 2 + 1) * 256] = cov
        covs = {}
        for (X, Y), big in nxt.items():
            cov = big.reshape(256, 2, 256, 2).mean(axis=(1, 3))
            a = style(np.minimum(cov * 1.6, 1.0), z)       # thin valleys stay visible from far away
            if a.max() >= 1:
                _save(a, z, X, Y)
                total += 1
            covs[(X, Y)] = cov
    idx = {z.name: sorted(f"{x.name}/{f.stem}" for x in z.iterdir() for f in x.glob("*.png"))
           for z in sorted(OUT.iterdir(), key=lambda p: int(p.name)) if z.is_dir()}
    (OUT / "index.json").write_text(json.dumps(idx, separators=(",", ":")), encoding="utf-8")
    print(f"done: {total} tiles in {time.time() - t:.0f} s", {z: len(v) for z, v in idx.items()}, flush=True)


if __name__ == "__main__" and "--restyle" not in sys.argv:
    main()


# ------------------------------------------------------------------------------------------- restyle
# The wash alone turns grey over the orange and red risk cells. Final style, applied to the built tiles
# (python build_tiles.py --restyle): a clear water blue, diagonal hatching inside the zones from zoom 11
# (the cartographic convention for flood zones: it stays readable over any colour) and a dark outline.
WASH, HATCH, EDGE_RGB = (46, 116, 196), (30, 88, 160), (18, 52, 98)


def restyle():
    from PIL import Image
    files = list(OUT.glob("*/*/*.png"))
    for f in files:
        z, x, y = int(f.parts[-3]), int(f.parts[-2]), int(f.stem)
        a = np.asarray(Image.open(f).convert("RGBA"))[..., 3].astype(np.float32)
        edge = a >= (A_EDGE - 10) if z >= 11 else np.zeros(a.shape, bool)
        cov = np.clip(a / A_FILL, 0, 1)
        cov[edge] = 1.0
        rgba = np.zeros((256, 256, 4), np.uint8)
        rgba[..., :3] = WASH
        alpha = cov * 85
        if z >= 11:
            jj, ii = np.mgrid[0:256, 0:256]
            period = 9 if z >= 13 else 7
            hatch = ((ii + x * 256) + (jj + y * 256)) % period < (2 if z >= 13 else 1.5)
            m = hatch & (cov > 0.3) & ~edge
            rgba[m, :3] = HATCH
            alpha = np.where(m, np.maximum(alpha, 200 * cov), alpha)
            rgba[edge, :3] = EDGE_RGB
            alpha = np.where(edge, 245, alpha)
        else:
            alpha = np.minimum(cov * 150, 170)
        rgba[..., 3] = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").quantize(colors=16, method=Image.Quantize.FASTOCTREE).save(f, optimize=True)
    print("restyled", len(files))


if __name__ == "__main__" and "--restyle" in sys.argv:
    restyle()
