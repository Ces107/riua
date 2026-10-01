"""Download Copernicus DEM GLO-90 (and optionally GLO-30) tiles from the AWS
open-data buckets (keyless HTTPS) into scratch/h1-catchments/cop90|cop30.

Usage: py -3.11 01_fetch_copdem.py 90   (or 30)
Tiles: lat N37..N40, lon W004..E000 (covers Riua box + Segura/Jucar headwaters).
"""
import os
import sys
import time
import requests

ROOT = r"C:\Users\cpereiro\IdeaProjects\riua"
res = sys.argv[1] if len(sys.argv) > 1 else "90"
code = {"90": "30", "30": "10"}[res]
out = os.path.join(ROOT, "scratch", "h1-catchments", f"cop{res}")
os.makedirs(out, exist_ok=True)
s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0 (riua-research; h1-catchments)"
for lat in range(37, 41):
    for lon in range(-4, 1):
        ew = "W" if lon < 0 else "E"
        name = f"Copernicus_DSM_COG_{code}_N{lat:02d}_00_{ew}{abs(lon):03d}_00_DEM"
        url = f"https://copernicus-dem-{res}m.s3.amazonaws.com/{name}/{name}.tif"
        dst = os.path.join(out, name + ".tif")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            continue
        r = s.get(url, timeout=300)
        if r.status_code != 200:
            print("MISSING", r.status_code, url)
            continue
        with open(dst, "wb") as f:
            f.write(r.content)
        print("ok", name, len(r.content))
        time.sleep(0.3)
