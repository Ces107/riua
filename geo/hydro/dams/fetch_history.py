"""Measured history of the SAIH Júcar reservoirs, for the level-volume curves and for checking the filling model.

For every reservoir of dams_sites.json with SAIH Júcar variables: level, volume, outflow (total and to the river) and
inflow, 5-min values, inside (a) every episode that has a rain analysis in hindcast/truth/ (6 h before .. 72 h after)
and (b) the first day of every month since September 2024 (to sample the level-volume relation over its range).

Endpoint: https://saih.chj.es/admin/variables/valor/{id}/{from}/{to}  (see geo/hydro/gauges/saih_series.py).
Cache (gitignored): scratch/q10-dams/series/chj_{var}.npz -> t (datetime64[m], UTC), v (float32), windows done.

    py -3.11 geo/hydro/dams/fetch_history.py            # ~4,600 short requests, two at a time, ~30 min the first time
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "geo" / "hydro" / "gauges"))
import saih_series as SS                                  # noqa: E402

CACHE = ROOT / "scratch" / "q10-dams" / "series"
TRUTH = ROOT / "hindcast" / "truth"
KINDS = ("level", "volume", "outflow", "river", "inflow")


def windows():
    out = []
    for f in sorted(TRUTH.glob("*.npz")):
        t = np.load(f)["t_end"]
        a = t[0].astype("datetime64[s]").astype(datetime) - timedelta(hours=6)
        b = t[-1].astype("datetime64[s]").astype(datetime) + timedelta(hours=72)
        out.append((f.stem, a, min(b, datetime.utcnow())))
    d = datetime(2024, 9, 5)
    while d < datetime.utcnow():
        out.append((f"m{d:%Y%m}", d, d + timedelta(hours=6)))
        d = (d.replace(day=1) + timedelta(days=32)).replace(day=5)
    return out


def load(var):
    f = CACHE / f"chj_{var}.npz"
    if not f.exists():
        return np.array([], "datetime64[m]"), np.array([], np.float32), []
    z = np.load(f, allow_pickle=False)
    return z["t"], z["v"], [str(x) for x in z["done"]]


def fetch_var(var):
    t, v, done = load(var)
    parts_t, parts_v, n = [t], [v], 0
    for name, a, b in windows():
        if name in done:
            continue
        try:
            # the endpoint takes Madrid wall time: ask two hours more on each side instead of converting
            tt, vv, _ = SS.fetch(var, a - timedelta(hours=2), b + timedelta(hours=4))
        except Exception as e:                            # noqa: BLE001
            print("  failed", var, name, type(e).__name__, flush=True)
            continue
        parts_t.append(tt); parts_v.append(vv); done.append(name); n += 1
        time.sleep(0.2)
    if n:
        tt, i = np.unique(np.concatenate(parts_t), return_index=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"chj_{var}.tmp.npz"
        np.savez_compressed(tmp, t=tt, v=np.concatenate(parts_v)[i], done=np.array(done))
        tmp.replace(CACHE / f"chj_{var}.npz")
    return var, n


if __name__ == "__main__":
    sites = json.loads((HERE / "dams_sites.json").read_text(encoding="utf-8"))["dams"]
    only = set(sys.argv[1:])
    todo = []
    for s in sites:
        if s["saih"].startswith("chj:") and (not only or s["id"] in only):
            todo += [s["vars"][k] for k in KINDS if s["vars"].get(k)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        for k, (var, n) in enumerate(ex.map(fetch_var, todo)):
            print(f"{k + 1}/{len(todo)} var {var}: {n} new windows, {time.time() - t0:.0f} s", flush=True)
