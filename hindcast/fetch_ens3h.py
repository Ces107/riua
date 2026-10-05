"""ECMWF ENS at 3-hourly steps for the 6-48 h horizon, as production uses it (extra_models.ecmwf_ens_members).

For every case-day D the 00Z run of D-1 is fetched at the production steps 6..66 h (21 steps, 50 members,
about 270 MB of GRIB per run, kept as one small .npy per step) into hindcast/obs/ens3h/ec_ifs_<run>/, so
    extra_models.ecmwf_ens_members(run, ROOT / "hindcast/obs/ens3h")
works offline in the hindcast. Resumable: steps on disk are never downloaded again.

    py -3.11 hindcast/fetch_ens3h.py                      # every case of cases.json, in file order
    py -3.11 hindcast/fetch_ens3h.py 2025-10-alice ...    # these cases, in this order
    py -3.11 hindcast/fetch_ens3h.py --list               # what is on disk
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from riua.sources import extra_models as X  # noqa: E402

OUT = ROOT / "hindcast" / "obs" / "ens3h"
STEPS = list(range(6, 67, 3))


def runs_for(case: dict) -> list[datetime]:
    return [datetime.fromisoformat(d) - timedelta(days=1) for d in case["days"]]


def complete(run: datetime) -> bool:
    d = OUT / f"ec_ifs_{run:%Y%m%d%H}"
    return (d / "grid.npz").exists() and all((d / f"tp_{s:03d}.npy").exists() for s in STEPS)


if __name__ == "__main__":
    cases = json.loads((ROOT / "hindcast" / "cases.json").read_text(encoding="utf-8"))["cases"]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    by_id = {c["id"]: c for c in cases}
    order = [by_id[a] for a in args] if args else cases
    runs = []
    for c in order:
        for r in runs_for(c):
            if r not in runs:
                runs.append(r)
    if "--list" in sys.argv:
        ok = [r for r in runs if complete(r)]
        print(f"{len(ok)} of {len(runs)} runs complete")
        print("missing:", " ".join(f"{r:%Y-%m-%d}" for r in runs if not complete(r)))
        sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    for r in runs:
        if complete(r):
            print("have", f"{r:%Y-%m-%d}", flush=True)
            continue
        t0, b0 = time.time(), X.transfer()[1]
        try:
            z = X.ecmwf_ens_fetch(r, STEPS, OUT)
            print(f"ok {r:%Y-%m-%d} acc{z['acc'].shape} {(X.transfer()[1] - b0) / 1e6:.0f} MB {time.time() - t0:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001  one missing run must not stop the batch
            print(f"FAILED {r:%Y-%m-%d} {type(e).__name__} {e}"[:300], flush=True)
