"""Rebuild every file in geo/out/ from the public sources (downloads are cached in scratch/r5-geo/).

    py -3.11 geo/build/build_all.py

Needs: geopandas shapely pyproj rasterio pdfplumber requests scipy matplotlib numpy (build machine only;
the outputs are plain JSON/GeoJSON/NPZ, loadable with json + numpy).
First run downloads ~0.6 GB (CHJ shapefiles 280 MB, HydroBASINS 74 MB, HydroRIVERS 68 MB, DEM tiles 90 MB,
OSM 33 MB) and the first b30 run reads the big shapefiles (~4 min); with everything cached a full run takes ~3 min.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "b10_zones.py",          # zones.geojson, thresholds.json, boundary.geojson
    "b20_places_raw.py",     # scratch: places_raw.json (IGN + INE)
    "b25_osm_waterways.py",  # scratch: osm_ww.json (Overpass)
    "b30_basins.py",         # basins.geojson, basins_topology.json (+ scratch basin_raster.npz)
    "b40_rivers.py",         # rivers.geojson
    "b50_terrain.py",        # terrain.npz
    "b60_finalize.py",       # places.json
    "b90_verify.py",         # sanity tables + PNG overviews in scratch/r5-geo/
]

if __name__ == "__main__":
    for s in STEPS:
        t = time.time()
        print(f"\n##### {s}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE)
        if r.returncode:
            sys.exit(f"{s} failed ({r.returncode})")
        print(f"##### {s} done in {time.time() - t:.0f} s", flush=True)
