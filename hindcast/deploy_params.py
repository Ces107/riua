"""Write backend/riua/params.json from the hindcast, with a chosen threshold policy.

    python hindcast/deploy_params.py floor40     # nothing below 40 % (owner's rule)
    python hindcast/deploy_params.py tuned       # CSI-optimal thresholds from the hindcast
    python hindcast/deploy_params.py middle      # 40 / 30 / 25 / 30 % at 48 h and beyond
    python hindcast/deploy_params.py service     # what warning services do: a severe level from about 20 %

sigma and bias always come from the hindcast. The scores of the deployed thresholds are
stored in results.json under <horizon>.deployed (computed on the cached hindcast blocks).
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hindcast"))
import run as R  # noqa: E402

policy = sys.argv[1] if len(sys.argv) > 1 else "floor40"
res = json.loads((ROOT / "hindcast" / "results.json").read_text(encoding="utf-8"))
MIDDLE = {"2": 0.40, "3": 0.30, "4": 0.25, "5": 0.30}
# AEMET issues any colour from 10 % (bands 10-40 / 40-70 / > 70 %); no service asks 40 % for its
# severe tier. Day-ahead red: 0.40 detects 13 % of the zone-days, 0.20 detects 43 % for the same
# false-alarm ratio (50 -> 53 %). Days 2-7 stay stricter: those probabilities are not reliable yet.
SERVICE = {"mid": {"2": 0.20, "3": 0.20, "4": 0.25, "5": 0.40}, "long": {"2": 0.25, "3": 0.40, "4": 0.40, "5": 0.40}}
out = {"version": f"hindcast-{res['generated'][:10]}-{policy}", "sigma": {}, "bias": {}, "tau": {}}
for hz in ("now", "mid", "long"):
    if hz not in res:
        continue
    t = res[hz]["tuned"]
    tuned = {str(k): float(v) for k, v in t["tau"].items()}
    floor = {str(k): float(v) for k, v in res[hz]["tau_min40"].items()}
    if policy == "tuned":
        tau = tuned
    elif policy == "service":
        tau = SERVICE.get(hz) or {k: max(0.40, tuned[k]) for k in tuned}
    elif policy == "middle":
        tau = floor if hz == "now" else MIDDLE
    else:
        tau = {k: max(0.40, tuned[k]) for k in tuned}
    out["sigma"][hz], out["bias"][hz], out["tau"][hz] = t["sigma"], t["bias"], tau
    f = ROOT / "hindcast" / "cache" / f"blocks_{hz}.pkl"
    if f.exists():
        blocks = pickle.loads(f.read_bytes())
        cont = R.contingency(blocks, t["sigma"], t["bias"], {int(k): v for k, v in tau.items()})
        res[hz]["deployed"] = {k: {L: R.rates(v) for L, v in d.items()} for k, d in cont.items()}
        res[hz]["deployed_tau"] = tau
        z = res[hz]["deployed"]["zone_day"]
        print(hz, tau, " | ".join(f"L{L} POD {v['POD']} FAR {v['FAR']}" for L, v in z.items()))
if "radius_km" in res.get("mid", {}):
    out["radius_km"] = {"mid": res["mid"]["radius_km"]}
res["policy"] = policy
(ROOT / "hindcast" / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
R.P.save(out)
print("wrote", R.P.PARAMS_FILE, out["version"])
