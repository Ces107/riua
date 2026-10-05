"""Two keyless Open-Meteo downloads for the event table, cached one file per (catchment, event):

    py -3.11 hindcast/ml/fetch_openmeteo.py soil      # ERA5-Land soil moisture on the day the event starts (archive API)
    py -3.11 hindcast/ml/fetch_openmeteo.py glofas    # GloFAS v4 daily river discharge around the event (Flood API)

Quota discipline (10,000 weighted calls / day per IP, shared): one call = one location and at most 14 days, so each
weighs 1; only the rows of the ravine domain are fetched (soil: all clean rows; GloFAS: the rows q7 scored, i.e. rain
>= 30 mm or a measured response); 0.5 s between calls; the run stops at the first HTTP 429.  Every call is logged in
cache/_calls.jsonl.  A cached file is never fetched again, so re-running costs nothing.
"""
import json
import sys
import time
import urllib.request

import numpy as np

from common import CACHE, GAUGES, OUT, UA, read_json

SOIL_VARS = "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def day(t, shift=0):
    return str((np.datetime64(t[:10]) + np.timedelta64(shift, "D")))


def centroids():
    """area-weighted centre (lat, lon) of every gauged catchment, from the time-area tables"""
    f = CACHE / "centroids.json"
    if f.exists():
        return read_json(f)
    import dataset as DS
    from riua.core import grid
    net, pts = DS.gauge_net()
    out = {}
    for k, p in enumerate(pts):
        cells, w, *_ = DS.catchment(net, k)
        iy, ix = cells // grid.NX, cells % grid.NX
        out[p["id"]] = dict(lat=round(float(((grid.LAT0 + (iy + 0.5) * grid.D) * w).sum() / w.sum()), 4),
                            lon=round(float(((grid.LON0 + (ix + 0.5) * grid.D) * w).sum() / w.sum()), 4))
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(what):
    rows = [r for r in read_json(OUT / "dataset.json") if r["clean"] and r["domain"]]
    pts = {p["id"]: p for p in read_json(GAUGES / "out" / "gauge_points.json")["points"]}
    attr = centroids()
    if what == "glofas":
        rows = [r for r in rows if r["usable"]]
    n_new = 0
    for r in rows:
        key = f"{r['pid']}_{r['t0']}"
        if what == "soil":
            f = CACHE / "era5land" / f"{key}.json"
            a = attr[r["pid"]]                      # centre of the catchment
            d0 = day(r["t0"])
            url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={a['lat']:.4f}&longitude={a['lon']:.4f}"
                   f"&start_date={d0}&end_date={d0}&hourly={SOIL_VARS}&models=era5_land")
        else:
            f = CACHE / "glofas" / f"{key}.json"
            p = pts[r["pid"]]
            # the event, one day before and two after its response window; never more than 14 days (weight 1)
            d0 = day(r["t0"], -1)
            d1 = min(np.datetime64(day(r["t1"], 3)), np.datetime64(d0) + np.timedelta64(13, "D"))
            url = (f"https://flood-api.open-meteo.com/v1/flood?latitude={p['lat']:.5f}&longitude={p['lon']:.5f}"
                   f"&daily=river_discharge&start_date={d0}&end_date={d1}")
        if f.exists():
            continue
        f.parent.mkdir(exist_ok=True)
        try:
            js = get(url)
        except urllib.error.HTTPError as e:
            print("HTTP", e.code, url)
            if e.code == 429:
                print("quota reached: stopping; re-run later (cached files are kept)")
                break
            continue
        f.write_text(json.dumps(js), encoding="utf-8")
        with open(CACHE / "_calls.jsonl", "a", encoding="utf-8") as g:
            g.write(json.dumps([time.time(), 1.0, what]) + "\n")
        n_new += 1
        time.sleep(0.5)
    print(f"{what}: {n_new} new calls, {len(rows)} rows wanted")


if __name__ == "__main__":
    main(sys.argv[1])
