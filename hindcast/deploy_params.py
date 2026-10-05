"""Write backend/riua/params.json from the hindcast fit, with a threshold policy, and store the scores of
exactly that policy in hindcast/results.json (<horizon>.deployed, what web/validacion.html shows).

    py -3.11 hindcast/deploy_params.py                  # = recommended
    py -3.11 hindcast/deploy_params.py recommended      # the table of coord/findings/q8-verify.md
    py -3.11 hindcast/deploy_params.py current          # keep kernel and thresholds as they are, only refresh the scores
    py -3.11 hindcast/deploy_params.py csi              # fitted kernel, CSI-optimal thresholds (zone-days, combined sample)
    py -3.11 hindcast/deploy_params.py floor40          # fitted kernel, nothing below 40 %
    add --dry-run to print without writing

Needs hindcast/fit.json and hindcast/cache/score_state.pkl (`python hindcast/run.py fit`). The scores are computed
from out-of-sample probabilities (each case with the kernel fitted without it) when the fitted kernel is deployed,
and from production's own kernel when it is kept; counts are "equivalent counts" of the combined sample (rain
episodes and quiet days weighted with their real frequency), the split is stored next to them.
Keys of params.json that this script does not own (anything but version, sigma, bias, sigma1h, bias1h, tau,
level_cap, radius_km) are left as they are.
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hindcast"))
import score as S  # noqa: E402

R = S.R
LV = ("2", "3", "4", "5")

# ---- the recommended policy (reasoning: coord/findings/q8-verify.md, section "Thresholds") ---------------------
# kernel: "fitted" = sigma / bias / sigma1h / bias1h of the fit; "keep" = leave production's values
# tau: P(>= level) from which the level is issued; cap: highest level the horizon may show
# Level 5 has no demonstrable out-of-sample skill in 0-6 h (events in 2 cases) nor in 6-48 h (4 cases, bootstrap
# BSS interval below 0): capped at 4. Days 2-7: no dry day can be verified (no ENS archive for them): cap 3 kept,
# and production's kernel kept (the fitted one is worse leave-one-case-out).
RECOMMENDED = {
    "now":  {"kernel": "fitted", "tau": {"2": 0.15, "3": 0.30, "4": 0.30, "5": 0.40}, "cap": 4},
    "mid":  {"kernel": "fitted", "tau": {"2": 0.15, "3": 0.15, "4": 0.15, "5": 0.40}, "cap": 4},
    "long": {"kernel": "keep", "tau": {"2": 0.15, "3": 0.25, "4": 0.40, "5": 0.40}, "cap": 3},
}


def policy_table(name: str, fit: dict, params: dict) -> dict:
    cur = {hz: {"kernel": "keep", "tau": {k: float(v) for k, v in params["tau"][hz].items()},
                "cap": params.get("level_cap", {}).get(hz)} for hz in ("now", "mid", "long")}
    if name == "current":
        return cur
    if name == "recommended":
        return RECOMMENDED
    out = {}
    for hz in ("now", "mid", "long"):
        if hz not in fit:
            continue
        cv = fit[hz]["curves"]["fitted_loco"]["all"].get("combined", {})
        tau = {}
        for L in LV:
            t = (cv.get(L, {}).get("csi_optimal_tau") or {}).get("zone_day")
            tau[L] = float(t) if t is not None else cur[hz]["tau"][L]
        if name == "floor40":
            tau = {L: max(0.40, v) for L, v in tau.items()}
        for a, b in zip(LV[:-1], LV[1:]):            # never decreasing with the level (see score.decide_counts)
            tau[b] = max(tau[b], tau[a]) if name == "floor40" else tau[b]
        out[hz] = {"kernel": "fitted", "tau": tau, "cap": cur[hz]["cap"]}
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "recommended"
    dry = "--dry-run" in sys.argv
    fit = json.loads((R.HC / "fit.json").read_text(encoding="utf-8"))
    res = json.loads((R.HC / "results.json").read_text(encoding="utf-8"))
    states = pickle.loads((R.HC / "cache" / "score_state.pkl").read_bytes())
    pop = S.load_pop()
    params = R.P.load()
    table = policy_table(name, fit, params)
    out = json.loads(R.P.PARAMS_FILE.read_text(encoding="utf-8")) if R.P.PARAMS_FILE.exists() else {}
    for key in ("sigma", "bias", "sigma1h", "bias1h", "tau", "level_cap"):
        out[key] = dict(out.get(key) or {})
    # defaults the live code falls back on, written out so that params.json says what is in force
    for hz in ("now", "mid", "long"):
        out["sigma"].setdefault(hz, params["sigma"][hz]); out["bias"].setdefault(hz, params["bias"][hz])
    for k, v in params.get("sigma1h", {}).items():
        out["sigma1h"].setdefault(k, v)
    for k, v in params.get("bias1h", {}).items():
        out["bias1h"].setdefault(k, v)
    for k, v in params.get("level_cap", {}).items():
        out["level_cap"].setdefault(k, v)
    for hz, pol in table.items():
        if hz not in fit or hz not in states:
            print(f"{hz}: no fit on disk, left as it is")
            continue
        st = states[hz]
        fitted = pol["kernel"] == "fitted"
        if fitted:
            k = fit[hz]["kernel"]["fitted"]
            out["sigma"][hz], out["bias"][hz] = k["sigma"], k["bias"]
            if hz != "long":                           # days 2-7: 12-hourly ensemble amounts, no 1-h term
                out["sigma1h"][hz], out["bias1h"][hz] = k["sigma1h"], k["bias1h"]
        out["tau"][hz] = {L: float(pol["tau"][L]) for L in LV}
        if pol["cap"]:
            out["level_cap"][hz] = int(pol["cap"])
        else:
            out["level_cap"].pop(hz, None)
        tally = S.Tally.from_dict(st["tallies"]["fitted_loco" if fitted else "production"])
        cls = st["classes"]
        dep = S.decide_counts(tally, cls, out["tau"][hz], pol["cap"], pop, "combined")
        res[hz]["deployed"] = dep
        res[hz]["deployed_split"] = {w: S.decide_counts(tally, cls, out["tau"][hz], pol["cap"], pop, w) for w in ("episodes", "quiet", "combined", "sample")}
        res[hz]["deployed_by_lead"] = {c: S.decide_counts(tally, [c], out["tau"][hz], pol["cap"], pop, "combined") for c in cls}
        res[hz]["deployed_tau"], res[hz]["deployed_cap"] = out["tau"][hz], pol["cap"]
        res[hz]["deployed_kernel"] = {"source": "fit, scored leave-one-case-out" if fitted else "production values kept",
                                      "sigma": out["sigma"][hz], "bias": out["bias"][hz],
                                      "sigma1h": out["sigma1h"].get(hz, out["sigma"][hz]), "bias1h": out["bias1h"].get(hz, out["bias"][hz])}
        res[hz]["fitted_kernel"] = fit[hz]["kernel"]["fitted"]
        res[hz]["tuned"] = {**res[hz].get("tuned", {}), "sigma": out["sigma"][hz], "bias": out["bias"][hz], "tau": out["tau"][hz]}
        z = dep["zone_day"]
        print(f"{hz}: sigma {out['sigma'][hz]} bias {out['bias'][hz]} sigma1h {out['sigma1h'].get(hz)} bias1h {out['bias1h'].get(hz)} "
              f"tau {out['tau'][hz]} cap {pol['cap']}")
        print("   zone-days, combined: " + " | ".join(f"L{L} POD {v['POD']} FAR {v['FAR']} ({v['hits']}/{v['misses']}/{v['false_alarms']})" for L, v in z.items()))
        q = res[hz]["deployed_split"]["quiet"]["zone_day"]
        print("   zone-days, quiet days: " + " | ".join(f"L{L} false alarms {v['false_alarms']} of {v['false_alarms'] + (v['correct_negatives'] or 0) + v['hits'] + v['misses']}" for L, v in q.items()))
    for key in ("sigma1h", "bias1h", "level_cap"):
        if not out[key]:
            out.pop(key)
    out["version"] = f"hindcast-{fit.get('generated', '')[:10]}-{name}"
    res["policy"] = name
    if dry:
        print(json.dumps(out, indent=1))
        print("dry run: nothing written")
    else:
        # days 2-7 are scored per zone and day by long_score.py (q13): the cell scores of this script must not replace them
        prev = json.loads((R.HC / "results.json").read_text(encoding="utf-8")) if (R.HC / "results.json").exists() else {}
        if (prev.get("long") or {}).get("scale") == "zone":
            res["long"] = prev["long"]
        (R.HC / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        R.P.save(out)
        print("wrote", R.P.PARAMS_FILE, out["version"], "and", R.HC / "results.json")
