"""Global-parameter sensitivity on the historical floods, with leave-one-event-out choice.

For each variant: simulate.py -> out/sens_<name>.json, evaluate.py -> scores on
  (a) natural gauged/documented peaks: MAE and bias of r = ln((sim + c)/(obs + c)), all and >= 100 m3/s
  (b) overflow call at control points with a documented yes/no.
Leave-one-event-out: for each event, pick the variant with the best MAE on the other events and score it on the
left-out one.

    py -3.11 hindcast/floods/sensitivity.py            (uses the rain already built; ~30 s per variant and case set)
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402
import simulate  # noqa: E402

VARIANTS = {
    "production": [],
    "phi30": ["--phi", "30"],
    "phi20": ["--phi", "20"],
    "p0_100": ["--p0", "100"],
    "alpha_x2": ["--alpha-scale", "2"],
    "vel_0.5": ["--vel", "0.5"],
    "phi20_alpha_x2": ["--phi", "20", "--alpha-scale", "2"],
}


def scores(rows, events=None):
    sel = [r for r in rows if (events is None or r["event"] in events)]
    pk = [r for r in sel if r["documented_peak"] and not r["regulated"] and r["peak_kind"] in ("measured", "estimated", "modelled")]
    big = [r for r in pk if r["documented_peak"] >= 100]
    ov = [r for r in sel if r["kind"] == "control" and r["documented_overflow"] is not None]
    e = np.array([r["ln_err"] for r in pk]) if pk else np.array([np.nan])
    eb = np.array([r["ln_err"] for r in big]) if big else np.array([np.nan])
    H = sum(1 for r in ov if r["documented_overflow"] and r["simulated_overflow"])
    M = sum(1 for r in ov if r["documented_overflow"] and not r["simulated_overflow"])
    F = sum(1 for r in ov if not r["documented_overflow"] and r["simulated_overflow"])
    N = sum(1 for r in ov if not r["documented_overflow"] and not r["simulated_overflow"])
    return {"n": len(pk), "mae": float(np.nanmean(np.abs(e))), "bias": float(np.nanmean(e)),
            "n_big": len(big), "mae_big": float(np.nanmean(np.abs(eb))), "bias_big": float(np.nanmean(eb)),
            "H": H, "M": M, "F": F, "N": N}


def main():
    res = {}
    for name, args in VARIANTS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            if not (name == "production" and (HERE / "out" / "sens_production.json").exists() and "--resim" not in sys.argv):
                simulate.main(args + ["--out", f"sens_{name}.json"])
            evaluate.main([f"sens_{name}.json"])
        rows = json.loads((HERE / "out" / "results.json").read_text(encoding="utf-8"))["rows"]
        res[name] = rows
        s = scores(rows)
        print(f"{name:16s} peaks n {s['n']:3d} MAE {s['mae']:.2f} bias {s['bias']:+.2f} | >=100: n {s['n_big']} MAE {s['mae_big']:.2f} "
              f"bias {s['bias_big']:+.2f} | overflow H {s['H']} M {s['M']} F {s['F']} N {s['N']}", flush=True)
    events = sorted({r["event"] for rows in res.values() for r in rows})
    print("\nleave-one-event-out (variant chosen on the other events by MAE of all natural peaks):")
    errs, picks = [], []
    for ev in events:
        others = [e for e in events if e != ev]
        best = min(res, key=lambda k: scores(res[k], others)["mae"])
        s = scores(res[best], [ev])
        if s["n"]:
            errs += [r["ln_err"] for r in res[best] if r["event"] == ev and r["documented_peak"] and not r["regulated"]
                     and r["peak_kind"] in ("measured", "estimated", "modelled")]
        picks.append((ev, best))
    print("  picks:", ", ".join(f"{e}:{b}" for e, b in picks))
    print(f"  out-of-sample MAE {np.mean(np.abs(errs)):.2f} bias {np.mean(errs):+.2f} (n {len(errs)}); production in-sample "
          f"{scores(res['production'])['mae']:.2f}")
    # evaluate.py last wrote the last variant: rewrite the production results for the page
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate.main(["sens_production.json"])


if __name__ == "__main__":
    main()
