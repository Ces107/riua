"""Channel capacity per control point, with its source, in one table: geo/hydro/capacity.json

Order of preference
  1. capacity computed from the LiDAR cross-section (sections.json, capacity_recommended / q_bankfull)
  2. a published channel capacity (works design, CHJ/CHS documents)
A computed capacity is rejected when its unit value exceeds the envelope of the largest
Mediterranean flash floods, q = 100 A^-0.4 m3/s/km2 (Gaume et al. 2009): such a section
spans a whole gorge, not a channel. The MCO (ordinary flood) is NOT used: channelised
urban reaches carry several times more, so it would cry wolf.
Points with no usable capacity get their level from the unit discharge instead (see
riua/core/hydro.py, level_thresholds).
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
secs = json.loads((H / "sections" / "out" / "sections.json").read_text(encoding="utf-8"))["points"]
secs = secs if isinstance(secs, dict) else {s["id"]: s for s in secs if isinstance(s, dict)}
pub = json.loads((H / "sections" / "out" / "published_flows.json").read_text(encoding="utf-8"))
cps = json.loads((H / "catchments" / "out" / "control_points.json").read_text(encoding="utf-8"))
cps = cps["points"] if isinstance(cps, dict) else cps

out, rejected = {}, {}
for p in cps:
    i = p["id"]
    area = p.get("area_unregulated_km2") or p["area_km2"]
    env = 100.0 * max(area, 1.0) ** -0.4 * area          # m3/s, envelope peak for that area
    s = secs.get(i) or {}
    rec = (s.get("capacity_recommended") or {}).get("value_m3s") or s.get("q_bankfull")
    if rec and float(rec) <= env and s.get("confidence") != "low":
        out[i] = {"q": round(float(rec), 1), "source": "sección LiDAR + Manning", "kind": "section",
                  "confidence": s.get("confidence"), "rating": True}
        continue
    if rec:
        rejected[i] = f"section {float(rec):.0f} m3/s vs envelope {env:.0f} (confidence {s.get('confidence')})"
    entries = [e for e in pub.get(i, []) if isinstance(e, dict) and e.get("value_m3s")]
    cap = [e for e in entries if e["quantity"] == "channel_capacity" and e["value_m3s"] <= env]
    if cap:
        e = min(cap, key=lambda e: e["value_m3s"])
        out[i] = {"q": float(e["value_m3s"]), "source": "capacidad publicada", "kind": "published",
                  "url": e.get("source_url"), "rating": False}
(H / "capacity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
kinds = {}
for v in out.values():
    kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
print(f"{len(out)} of {len(cps)} points with capacity: {kinds}")
print("rejected sections:", json.dumps(rejected, indent=1, ensure_ascii=False))
print("level by unit discharge:", len(cps) - len(out))
