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

cm = json.loads((H / "caumax_points.json").read_text(encoding="utf-8")) if (H / "caumax_points.json").exists() else {}
out, rejected = {}, {}
for p in cps:
    i = p["id"]
    area = p.get("area_unregulated_km2") or p["area_km2"]
    env = 100.0 * max(area, 1.0) ** -0.4 * area          # m3/s, envelope peak for that area
    s = secs.get(i) or {}
    rec = (s.get("capacity_recommended") or {}).get("value_m3s") or s.get("q_bankfull")
    qt = {k: v for k, v in (cm.get(i) or {}).items() if k.startswith("T")}
    lo = 0.5 * qt["T2"] if qt else 0.0
    hi = env                    # engineered channels may exceed T500 (new Turia channel: 5000 m3/s)
    if rec and lo <= float(rec) <= hi and s.get("confidence") != "low" and not s.get("valley_confined"):
        out[i] = {"q": round(float(rec), 1), "source": "sección LiDAR + Manning", "kind": "section",
                  "confidence": s.get("confidence"), "rating": True}
        continue
    if rec:
        rejected[i] = f"section {float(rec):.0f} m3/s vs plausible {lo:.0f}-{hi:.0f} (confidence {s.get('confidence')}{', confined' if s.get('valley_confined') else ''})"
    entries = [e for e in pub.get(i, []) if isinstance(e, dict) and e.get("value_m3s")]
    cap = [e for e in entries if e["quantity"] == "channel_capacity" and lo <= e["value_m3s"] <= hi]
    if cap:
        e = min(cap, key=lambda e: e["value_m3s"])
        out[i] = {"q": float(e["value_m3s"]), "source": "capacidad publicada", "kind": "published",
                  "url": e.get("source_url"), "rating": False}
# 3. a capacity measured by a real overflow beats both: the flow measured when the water started to leave the
#    channel at the control point (q9-floods, coord/findings/q9-floods.md). The rating curve of the section is kept for
#    the depths, only the overflow flow changes.
OBSERVED = {
    "segura-orihuela": {
        "q": 125.0,
        "note": "13 Sept 2019: the Segura overflowed in Orihuela 'unicamente ... y muy ligeramente entre el puente de "
                "Levante y el puente del Rey' (CHS Comisaria de Aguas) at 07:30-07:40 local; CEDEX anuario: 123.4 m3/s "
                "at Azud de los Huertos (Orihuela) that day (daily mean 120.7). Section value was 286.",
        "url": "https://alicanteplaza.es/la-gota-fria-sigue-y-mantiene-en-vilo-a-la-vega-baja-ante-el-desborde-del-rio-segura",
    },
    "saleta-aldaia": {
        "q": 20.0,
        "note": "CHJ: in Aldaia the ravine loses its channel entirely, causing frequent and serious flooding 'de ocurrencia casi "
                "anual' (about the 2-year flood, CAUMAX Q2 19.7). The new works are sized for 80 + 15 m3/s.",
        "url": "https://chj.es/es-es/ciudadano/participacion_publica/Documents/Descripci%C3%B3n%20de%20las%20actuaciones.pdf",
    },
    "girona-verger": {
        "q": 200.0,
        "note": "CHJ Plan Director Marina Alta: 'desbordamientos generalizados para caudales superiores a 200 m3/s'. Section value was 269.",
        "url": "https://www.chj.es",
    },
    "barxeta-carcaixent": {
        "q": 40.0,
        "note": "CHJ/MITECO viability report: 'no cuenta con un cauce definido, mas alla de una acequia'; the planned channel "
                "carries about 40 m3/s. The LiDAR section value (265) was not a channel.",
        "url": "https://www.chj.es",
    },
}
for i, o in OBSERVED.items():
    prev = out.get(i, {})
    out[i] = {"q": o["q"], "source": "desbordamiento observado", "kind": "observed", "confidence": "medium",
              "rating": prev.get("rating", False), "url": o["url"], "note": o["note"],
              "replaces": {k: prev.get(k) for k in ("q", "kind")}}
for i, v in cm.items():
    qt = {k: x for k, x in v.items() if k.startswith("T")}
    if qt:
        out.setdefault(i, {"q": None, "source": None, "kind": None, "rating": False})
        out[i]["qT"] = qt
        out[i]["qT_how"] = v.get("how")
(H / "capacity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
kinds = {}
for v in out.values():
    kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
print(f"capacity: {kinds}; with return-period flows: {sum(1 for v in out.values() if v.get('qT'))} of {len(cps)}")
print("rejected sections:", json.dumps(rejected, indent=1, ensure_ascii=False))
print("level by return period (no capacity):", sum(1 for v in out.values() if v.get("q") is None))
