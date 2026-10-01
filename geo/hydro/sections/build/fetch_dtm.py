"""Step 2: download a 1 m DTM window (1.8 x 1.8 km) around each anchor. Cached per id.
Usage: py -3.11 fetch_dtm.py [id ...]   (sequential, polite: one sheet stream at a time)
"""
import json
import os
import sys
import time

import numpy as np

from common import MISC, load_seed
from dtm import get_window

HALF = 900


def main():
    only = set(sys.argv[1:])
    with open(os.path.join(MISC, "anchors.json"), encoding="utf-8") as f:
        anchors = json.load(f)
    for p in load_seed():
        pid = p["id"]
        if (only and pid not in only) or pid not in anchors:
            continue
        x, y = anchors[pid]["anchor_xy"]
        t = time.time()
        for attempt in range(3):
            try:
                a, tr, src = get_window(x, y, HALF, cache_name=pid)
                break
            except Exception as e:  # noqa
                print(pid, "retry", attempt, repr(e)[:200], flush=True)
                time.sleep(15)
        else:
            print(pid, "FAILED", flush=True)
            continue
        print(f"{pid:24s} {a.shape} z {np.nanmin(a):.1f}..{np.nanmax(a):.1f} nan {np.isnan(a).mean():.3f} "
              f"{src.get('name', '')[:40].encode('ascii', 'replace').decode()} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
