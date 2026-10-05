"""Join the simulated floods (out/sim.json) with the documented ones (facts.json + anuario_events.csv).

Writes out/results.json (one row per documented fact, what the validation page renders) and prints the tables:
peaks (ratio, ln error) and the overflow call (hits / misses / false alarms / correct negatives).

    py -3.11 hindcast/floods/evaluate.py [sim.json]
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# CEDEX Anuario station -> q7 gauge point (same gauge; geo/hydro/gauges/out/gauge_points.json "EA nn" = ROEA 80nn)
ANUARIO_GAUGE = {"8029": "chj-12827", "8060": "chj-13403", "8148": "chj-2720", "7169": "seg-01O03A1",
                 "8074": "chj-12810", "8028": "chj-13683", "8134": "chj-2725", "8030": "chj-13526",
                 "8025": "chj-1523", "8089": "chj-13070"}
ANUARIO_CONTROL = {"7628": "segura-orihuela"}     # Azud de los Huertos, 3 km upstream of the control point
REGULATED = {"chj-12810", "chj-1523", "chj-13070", "chj-14551", "chj-2443", "segura-orihuela", "chj-13917"}


def c_off(area):
    return 5.0 * (max(area, 1.0) / 184.0) ** 0.75


def rain_meta(case):
    f = HERE / "cache" / "rain" / f"{case['id']}.npz"
    if case["rain"] == "truth":
        return {"rain": "truth (q4 build_truth: radar + gauges)"}
    if not f.exists():
        return None
    m = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
    g = m.get("gauge", {})
    days = [d for d in g if (g[d].get("gauge_max") or 0) >= 20]
    return {"rain": case["rain"], "advection": (case["id"] in ADVECTED) if case["rain"] == "radar" else None,
            "days": {d: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in g[d].items()
                         if k in ("n_gauges", "gauge_max", "radar_raw_max", "merged_max", "method", "mfb")} for d in days}}


ADVECTED = {"2012-11-11", "2015-03-22", "2015-11-02"}     # built in WSL with pysteps (optical flow); the rest without


def anuario_facts(cases):
    out = []
    day_case = {d: c["id"] for c in cases for d in c["days"][1:]}
    with open(HERE / "anuario_events.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            pid = ANUARIO_GAUGE.get(r["code"]) or ANUARIO_CONTROL.get(r["code"])
            if not pid:
                continue
            inst, dmax = r["inst_max_m3s"], r["window_max_daily_m3s"]
            date = r["inst_max_date"] if inst else r["window_max_date"]
            case = day_case.get(date)
            if not case or not (inst or dmax):
                continue
            out.append({"case": case, "kind": "gauge" if r["code"] in ANUARIO_GAUGE else "control", "point": pid,
                        "peak": float(inst) if inst else float(dmax),
                        "peak_kind": "measured" if inst else "measured daily mean (lower bound of the peak)",
                        "overflow": None, "t_peak": date,
                        "detail": f"CEDEX Anuario de aforos, station {r['code']} {r['name']} ({r['area_km2']} km2)",
                        "source": "https://ceh.cedex.es/anuarioaforos/ (Anuario 2021-22 tables, afliqi/afliq)",
                        "confidence": "high"})
    return out


def main(argv):
    sim = json.loads((OUT / (argv[0] if argv else "sim.json")).read_text(encoding="utf-8"))
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    cmap = {c["id"]: c for c in cases}
    facts = json.loads((HERE / "facts.json").read_text(encoding="utf-8"))["facts"]
    have = {(f["case"], f["point"]) for f in facts if f["peak"]}
    facts += [f for f in anuario_facts(cases) if (f["case"], f["point"]) not in have]
    idx = {(r["case"], r["kind"], r["point"]): r for r in sim["rows"]}
    rows = []
    for f in facts:
        s = idx.get((f["case"], f["kind"], f["point"]))
        if s is None:
            continue
        row = {"event": f["case"], "label": cmap[f["case"]].get("label", ""), "rain_input": cmap[f["case"]]["rain"],
               "point": f["point"], "kind": f["kind"], "area_km2": s["area_km2"],
               "documented_peak": f["peak"], "peak_kind": f["peak_kind"], "documented_t_peak": f.get("t_peak"),
               "simulated_peak": s["sim_peak"], "t_sim_peak": s["t_sim_peak"], "zero_loss_peak": s["zero_loss_peak"],
               "rain_mm": s["rain_mm"], "rain_max1h": s["rain_max1h"], "runoff_mm": s.get("runoff_mm"),
               "documented_overflow": f["overflow"], "simulated_overflow": s.get("sim_overflow"),
               "simulated_level": s.get("sim_level"), "capacity_used": s.get("thr4"),
               "regulated": f["point"] in REGULATED, "detail": f["detail"], "source": f["source"],
               "confidence": f["confidence"], "lead_obs_rain_l4_h": s.get("lead_obs_rain_l4_h"),
               "h_rain_start_to_l4": s.get("h_rain_start_to_l4"), "h_rain_peak_to_l4": s.get("h_rain_peak_to_l4")}
        if f["peak"]:
            c = c_off(s["area_km2"])
            row["ratio"] = round(s["sim_peak"] / f["peak"], 3)
            row["ln_err"] = round(math.log((s["sim_peak"] + c) / (f["peak"] + c)), 3)
        rows.append(row)
    rows.sort(key=lambda r: (r["event"], r["kind"], r["point"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"params": sim["params"], "vel": sim.get("vel"), "rows": rows,
                                                  "rain": {c["id"]: rain_meta(c) for c in cases}}, indent=1),
                                      encoding="utf-8")
    # ---- tables
    print("| event | point | rain mm | documented | kind | simulated | ratio | ln err | zero-loss |")
    print("|---|---|---|---|---|---|---|---|---|")
    pk = [r for r in rows if r["documented_peak"]]
    for r in pk:
        print(f"| {r['event']} | {r['point']}{' (reg.)' if r['regulated'] else ''} | {r['rain_mm']} | {r['documented_peak']} | "
              f"{r['peak_kind'][:8]} | {r['simulated_peak']} | {r['ratio']} | {r['ln_err']} | {r['zero_loss_peak']} |")
    for name, sel in (("all peaks", pk), ("natural, measured instantaneous", [r for r in pk if not r["regulated"] and r["peak_kind"] == "measured"]),
                      ("natural, documented >= 100 m3/s", [r for r in pk if not r["regulated"] and r["documented_peak"] >= 100])):
        if sel:
            e = np.array([r["ln_err"] for r in sel])
            print(f"\n{name}: n {len(sel)}, MAE ln {np.abs(e).mean():.2f}, bias {e.mean():+.2f}, median ratio "
                  f"{np.median([r['ratio'] for r in sel]):.2f}")
    print("\nOverflow call (control points with a documented overflow yes/no):")
    print("| event | point | documented | simulated | sim peak | zero-loss peak | capacity used | reading |")
    print("|---|---|---|---|---|---|---|---|")
    H = M = F = N = 0
    for r in rows:
        if r["kind"] != "control" or r["documented_overflow"] is None:
            continue
        d, s = r["documented_overflow"], bool(r["simulated_overflow"])
        H += d and s; M += d and not s; F += (not d) and s; N += (not d) and not s
        reading = "" if d == s else ("no loss setting reaches the capacity: rain or capacity" if d and r["zero_loss_peak"] < r["capacity_used"]
                                     else "losses too high" if d else "losses too low or capacity too small")
        r["miss_reading"] = reading or None
        print(f"| {r['event']} | {r['point']} | {'yes' if d else 'no'} | {'yes' if s else 'no'} | {r['simulated_peak']} | "
              f"{r['zero_loss_peak']} | {r['capacity_used']} | {reading} |")
    print(f"\nhits {H}, misses {M}, false alarms {F}, correct negatives {N}")
    told = {(r["event"], r["point"]) for r in rows if r["documented_overflow"] is not None}
    extra = [s for s in sim["rows"] if s["kind"] == "control" and s.get("sim_overflow") and (s["case"], s["point"]) not in told]
    print("\nSimulated overflows with no documented fact (unverified, possible false alarms):",
          ", ".join(f"{s['case']} {s['point']} ({s['sim_peak']:.0f} vs {s['thr4']:.0f})" for s in extra))
    (OUT / "results.json").write_text(json.dumps({"params": sim["params"], "vel": sim.get("vel"), "rows": rows,
                                                  "overflow_table": {"hits": H, "misses": M, "false_alarms": F, "correct_negatives": N},
                                                  "rain": {c["id"]: rain_meta(c) for c in cases}}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
