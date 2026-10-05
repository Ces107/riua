"""One command for the historical-flood validation (q9-floods). Every step reuses its cache in hindcast/floods/cache/.

    py -3.11 hindcast/floods/run_all.py            # gauges -> ERA5 timing -> rain -> simulate -> evaluate -> sensitivity
    py -3.11 hindcast/floods/run_all.py --fast     # simulate + evaluate only (rain already built)

Steps
  1. fetch_gauges.py     AVAMET daily totals per civil day                      cache/obs/<day>.csv
  2. era5_profile.py     ERA5 hourly rain (S3 mirror) for the pre-radar cases   cache/era5/<d0>_<d1>.npz
  3. build_rain.py       radar + gauges (or gauges + ERA5 timing)               cache/rain/<case>.npz
                         (--no-advection where pysteps is missing: see the note in build_rain.py)
  4. simulate.py         production model at the 60 control points + q7 gauges  out/sim.json, cache/sim/<case>.npz
  5. evaluate.py         join with facts.json + anuario_events.csv              out/results.json (+ tables on stdout)
  6. sensitivity.py      global parameters, leave-one-event-out                 out/sens_*.json
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=HERE.parents[1])


def main(argv):
    if "--fast" not in argv:
        run(str(HERE / "fetch_gauges.py"))
        for c in json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]:
            if c.get("era5") and not (HERE / "cache" / "era5" / f"{c['era5'][0]}_{c['era5'][1]}.npz").exists():
                run(str(HERE / "era5_profile.py"), *c["era5"])
        adv = [] if importlib.util.find_spec("pysteps") else ["--no-advection"]
        run(str(HERE / "build_rain.py"), *adv)
    run(str(HERE / "simulate.py"))
    run(str(HERE / "evaluate.py"))
    if "--fast" not in argv:
        run(str(HERE / "sensitivity.py"))


if __name__ == "__main__":
    main(sys.argv[1:])
