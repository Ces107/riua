"""Download the ECMWF ENS runs needed to hindcast the long range for a list of target days.

For target UTC day D the 00Z runs of D-3 and D-5 are used (lead classes d2-3 and d4-5),
with total precipitation at the 12-hourly steps that bracket D and the 12 h before it.

    py -3.11 hindcast/fetch_ens_cases.py 2024-10-28:2024-10-30 2024-11-01:2024-11-14
"""
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from riua.sources import ecmwf_open as E  # noqa: E402

LEADS = (3, 5)


def days(spec):
    for s in spec:
        a, _, b = s.partition(":")
        d0, d1 = datetime.fromisoformat(a), datetime.fromisoformat(b or a)
        while d0 <= d1:
            yield d0
            d0 += timedelta(days=1)


def plan(target_days):
    need = defaultdict(set)
    for d in target_days:
        for L in LEADS:
            run = d - timedelta(days=L)
            base = L * 24
            need[run].update({base - 12, base, base + 12, base + 24})
    return need


if __name__ == "__main__":
    need = plan(list(days(sys.argv[1:])))
    for run in sorted(need):
        out = E.CACHE / f"ifsens_tp_{run:%Y%m%d%H}.npz"
        steps = sorted(need[run])
        if out.exists():
            import numpy as np
            if set(steps) <= set(np.load(out)["steps"].astype(int).tolist()):
                print("have", run, steps, flush=True)
                continue
        try:
            E.fetch_run(run, steps, "tp", out, mirrors=["gcs", "aws"], workers=6, verbose=False)
            print("ok", run, steps, flush=True)
        except Exception as e:
            print("FAILED", run, type(e).__name__, e, flush=True)
