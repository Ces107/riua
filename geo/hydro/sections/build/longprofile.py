"""Step 2b: reach-scale long profile. For each point, bed elevation at +-1.5, 2 and 3 km along the
OSM line (robust minimum of a 160 x 160 m DTM window read by range requests), to get a slope over
several km that includes drop structures and is insensitive to local DTM noise.
Writes scratch/h2-sections/misc/longprofile.json (cached per point).
Usage: py -3.11 longprofile.py [id ...]
"""
import json
import os
import sys

import numpy as np

from common import MISC, jdump, load_seed
from dtm import get_window

HALF = 80


def main():
    only = set(sys.argv[1:])
    with open(os.path.join(MISC, "anchors.json"), encoding="utf-8") as f:
        anchors = json.load(f)
    path = os.path.join(MISC, "longprofile.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for p in load_seed():
        pid = p["id"]
        if (only and pid not in only) or pid not in anchors:
            continue
        key = "%.0f_%.0f" % tuple(anchors[pid]["anchor_xy"])
        if pid in res and res[pid].get("key") == key:
            continue
        pts = []
        for fp in anchors[pid].get("far_points", []):
            try:
                a, _, _ = get_window(fp["x"], fp["y"], HALF)
            except Exception as e:  # noqa
                print(pid, fp["chain"], "failed", repr(e)[:100], flush=True)
                continue
            if np.isnan(a).mean() > 0.5:
                continue
            pts.append({"chain": fp["chain"], "z_min": float(np.nanpercentile(a, 1)), "z_med": float(np.nanmedian(a))})
        res[pid] = {"key": key, "points": pts}
        jdump(res, path)
        print(pid, [(q["chain"], round(q["z_min"], 1)) for q in pts], flush=True)


if __name__ == "__main__":
    main()
