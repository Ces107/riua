"""Fetch OSM waterways (named rivers/streams/canals) and dams for the Riua box
from the Overpass API, cached as JSON in scratch/h1-catchments/osm/.

Split in 4 latitude bands to keep each response small. (c) OpenStreetMap contributors, ODbL.
"""
import json
import os
import time
import requests

ROOT = r"C:\Users\cpereiro\IdeaProjects\riua"
OUT = os.path.join(ROOT, "scratch", "h1-catchments", "osm")
os.makedirs(OUT, exist_ok=True)
EPS = ["https://overpass-api.de/api/interpreter",
       "https://overpass.kumi.systems/api/interpreter",
       "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
UA = {"User-Agent": "Mozilla/5.0 (riua-research; h1-catchments)"}


def run(q, name):
    dst = os.path.join(OUT, name + ".json")
    if os.path.exists(dst) and os.path.getsize(dst) > 200:
        print("cached", name)
        return
    for k in range(6):
        ep = EPS[k % len(EPS)]
        try:
            r = requests.post(ep, data={"data": q}, headers=UA, timeout=400)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(r.text)
                print("ok", name, len(r.text), ep)
                time.sleep(3)
                return
            print("fail", name, r.status_code, ep, r.text[:120].replace("\n", " "))
        except Exception as e:  # noqa
            print("err", name, ep, repr(e)[:120])
        time.sleep(15)
    print("GAVE UP", name)


W, E = -2.6, 0.8
bands = [(37.6, 38.5), (38.5, 39.3), (39.3, 40.1), (40.1, 41.0)]
for i, (s, n) in enumerate(bands):
    bb = f"{s},{W},{n},{E}"
    q = f"""[out:json][timeout:300];
(
  way["waterway"~"^(river|canal)$"]({bb});
  way["waterway"~"^(stream|drain|ditch)$"]["name"]({bb});
);
out tags geom;"""
    run(q, f"waterways_{i}")
bb = f"37.6,-3.2,41.0,0.8"
q = f"""[out:json][timeout:300];
(
  nwr["waterway"="dam"]({bb});
  nwr["water"="reservoir"]["name"]({bb});
);
out tags center;"""
run(q, "dams")
