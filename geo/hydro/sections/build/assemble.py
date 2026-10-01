"""Step 4: assemble out/sections.json from the per-point results, the literature figures
(scratch/h2-sections/misc/published.json, copied to out/published_flows.json) and the analyst
configuration; assign the confidence flag and the recommended capacity; build the contact sheet.
Usage: py -3.11 assemble.py
"""
import json
import os
import shutil

from PIL import Image

from common import MISC, OUT, PLOTS, SCRATCH, jdump, load_config, load_seed

RESULTS = os.path.join(SCRATCH, "results")

# Published figure taken as THE reference capacity of the reach at our section
# (index into published.json[id]); kind: "existing" = capacity/design of the channel as built,
# "projected" = design flow of works not (fully) built -> listed, never used as fallback.
PUBLISHED_PICK = {
    "poyo-torrent": (1, "existing"),
    "poyo-paiporta": (3, "existing"),
    "poyo-catarroja": (1, "existing"),
    "carraixet-alfara": (1, "existing"),
    "turia-nuevo-cauce": (1, "existing"),
    "magro-utiel": (0, "existing"),
    "jucar-alzira": (2, "existing"),
    "jucar-sueca": (1, "existing"),
    "barxeta-carcaixent": (0, "projected"),
    "vaca-tavernes": (0, "existing"),
    "palancia-sagunt": (0, "projected"),
    "sec-borriana": (0, "existing"),
    "sec-castello": (0, "existing"),
    "ovejas-alacant": (1, "existing"),
    "abanilla-benferri": (0, "projected"),
    "segura-orihuela": (0, "existing"),
}


def confidence(r, cfg):
    reasons = []
    level = 2  # 2 high, 1 medium, 0 low

    def down(to, why):
        nonlocal level
        level = min(level, to)
        reasons.append(why)

    w = r["bankfull_top_width_m"]
    if r["covered_reach_at_section"]:
        down(0, "section crosses a covered reach")
    if w < 5:
        down(0, f"channel only {w:.0f} m wide (< 5 DTM pixels)")
    elif w < 10:
        down(1, f"channel only {w:.0f} m wide (few DTM pixels)")
    if not (r["bank_left"]["crest_found"] and r["bank_right"]["crest_found"]):
        down(1, "one bank is a valley side / no crest found within the search distance")
    q3 = [q for q in r["q_bankfull_3_sections"] if q]
    if len(q3) == 3 and min(q3) > 0:
        f = max(q3) / min(q3)
        if f > 2.5:
            down(1, f"the three sections disagree by a factor {f:.1f}")
    qs = sorted(q for q in r["reach_scan"]["q_bankfull"] if q > 0)
    if len(qs) >= 6:
        p25, p75 = qs[len(qs) // 4], qs[(3 * len(qs)) // 4]
        f = p75 / p25
        if f > 2.5:
            down(1, f"capacity varies by a factor {f:.1f} (p25-p75) between sections 50 m apart along the reach")
        med = qs[len(qs) // 2]
        if not (0.5 <= r["q_bankfull"] / med <= 2.0):
            down(1, f"chosen section ({r['q_bankfull']:.0f}) is not typical of the reach (median {med:.0f} m3/s)")
    if r["perennial"]:
        down(1, "perennial river: LiDAR saw the water surface, submerged area not included (capacity underestimated)")
    if r["froude_capped_at_bankfull"]:
        down(1, "steep reach: channel flow limited to Froude 1")
    if any("floored" in n for n in r["auto_notes"]):
        down(1, "flat reach: slope floored at 0.0002, backwater-controlled in reality")
    if r["bankfull_depth_m"] < 1.0:
        down(0, f"bankfull depth only {r['bankfull_depth_m']:.1f} m: DTM noise and vegetation dominate")
    if level == 2:
        reasons.append("lined channel well resolved by the DTM, sections agree" if r["lining"] in ("concrete", "masonry")
                       else "channel well resolved by the DTM, sections agree")
    if "confidence" in cfg:
        level = {"high": 2, "medium": 1, "low": 0}[cfg["confidence"]]
        reasons.insert(0, "analyst: " + cfg.get("confidence_reason", ""))
    return ["low", "medium", "high"][level], "; ".join(reasons)


def main():
    cfg = load_config()
    with open(os.path.join(MISC, "published.json"), encoding="utf-8") as f:
        pub = json.load(f)
    shutil.copyfile(os.path.join(MISC, "published.json"), os.path.join(OUT, "published_flows.json"))
    out = {"_meta": {
        "generated_by": "geo/hydro/sections/build (h2-sections)",
        "units": "m, m3/s; elevations orthometric from the ICV 1 m LiDAR DTM",
        "method": "see README.md",
        "fields_note": "q_bankfull = Manning capacity of the DTM section up to the lower bank crest; "
                       "capacity_recommended = value to use as overflow threshold (falls back to published/CAUMAX when confidence is low); "
                       "h_over_bank: water level above the lower bank crest for flows above capacity",
    }, "points": {}}
    rows = []
    for p in load_seed():
        pid = p["id"]
        path = os.path.join(RESULTS, pid + ".json")
        if not os.path.exists(path):
            out["points"][pid] = {"id": pid, "stream": p["stream"], "town": p["town"], "status": "not processed"}
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        c = cfg[pid]
        conf, why = confidence(r, c)
        r["confidence"], r["confidence_reason"] = conf, why
        r["analyst_note"] = c.get("note")
        # published figures
        plist = pub.get(pid, [])
        r["published"] = [{k: e.get(k) for k in ("quantity", "value_m3s", "value_range_m3s", "values_by_T", "where",
                                                 "event_or_T", "source_url", "source_type", "quote", "verified_by_fetch")
                           if e.get(k) is not None} for e in plist]
        ref = None
        if pid in PUBLISHED_PICK:
            i, kind = PUBLISHED_PICK[pid]
            e = plist[i]
            ref = {"value_m3s": e["value_m3s"], "range_m3s": e.get("value_range_m3s"), "kind": kind,
                   "where": e["where"], "source_url": e["source_url"], "quote": e.get("quote")}
            ref["ratio_dtm_to_published"] = round(r["q_bankfull"] / e["value_m3s"], 2)
        r["published_capacity"] = ref
        # recommended capacity
        cm = r.get("caumax")
        if conf != "low":
            rec = {"value_m3s": r["q_bankfull"], "range_m3s": r["q_bankfull_range"], "basis": "DTM section + Manning"}
        elif ref and ref["kind"] == "existing":
            rg = [v for v in (ref["range_m3s"] or []) if v] or [ref["value_m3s"]]
            rec = {"value_m3s": ref["value_m3s"], "range_m3s": [min(rg + [ref["value_m3s"]]), max(rg + [ref["value_m3s"]])],
                   "basis": "published design/channel capacity (DTM section not reliable)", "source_url": ref["source_url"]}
        elif cm:
            rec = {"value_m3s": round((cm["T5"] * cm["T10"]) ** 0.5, 1), "range_m3s": [cm["T5"], cm["T10"]],
                   "basis": "PROXY: CAUMAX T5-T10 natural-regime flow as bankfull proxy (DTM section not reliable, no published capacity)"}
        else:
            rec = {"value_m3s": r["q_bankfull"], "range_m3s": r["q_bankfull_range"],
                   "basis": "DTM section + Manning, LOW CONFIDENCE (no published capacity, basin < 50 km2 so no CAUMAX value)"}
        confined = c.get("confined")
        if confined is None:   # automatic: far deeper and larger than any plausible flood
            confined = bool(cm and r["bankfull_depth_m"] > 8 and r["q_bankfull"] > 2 * cm["T500"])
        if confined:
            rec = {"value_m3s": None, "range_m3s": None,
                   "basis": "valley/gorge-confined reach: no overbank threshold exists at the section (bankfull capacity is more than "
                            "twice the T500 flow); use the rating h(Q) - low riverside buildings, terraces and bridges are "
                            "affected long before the detected crest"}
            r["confidence_reason"] += "; valley-confined: q_bankfull is not an overflow threshold"
        r["valley_confined"] = bool(confined)
        if "capacity_override" in c:
            rec = c["capacity_override"]
        r["capacity_recommended"] = rec
        out["points"][pid] = r
        rows.append((pid, r["q_bankfull"], r["q_bankfull_range"], rec["value_m3s"], rec["basis"][:22], conf,
                     ref["value_m3s"] if ref else None, (cm or {}).get("T10"), (cm or {}).get("T100"), (cm or {}).get("T500")))
    jdump(out, os.path.join(OUT, "sections.json"))
    print(f'{"id":24s} {"Qbf":>7s} {"range":>14s} {"rec":>7s} {"basis":22s} {"conf":6s} {"publ":>6s} {"T10":>6s} {"T100":>6s} {"T500":>6s}')
    for x in rows:
        print(f'{x[0]:24s} {x[1]:7.0f} {str([round(v) for v in x[2]]):>14s} {x[3] if x[3] is not None else float("nan"):7.0f} {x[4]:22s} {x[5]:6s} '
              f'{str(x[6]):>6s} {str(x[7]):>6s} {str(x[8]):>6s} {str(x[9]):>6s}')
    # contact sheet
    ids = [p["id"] for p in load_seed() if os.path.exists(os.path.join(PLOTS, p["id"] + ".png"))]
    if ids:
        tw, th, ncol = 1140, 384, 2
        sheet = Image.new("RGB", (tw * ncol, th * ((len(ids) + ncol - 1) // ncol)), "white")
        for k, pid in enumerate(ids):
            im = Image.open(os.path.join(PLOTS, pid + ".png")).convert("RGB").resize((tw, th))
            sheet.paste(im, ((k % ncol) * tw, (k // ncol) * th))
        sheet.save(os.path.join(PLOTS, "_contact_sheet.png"))
        print("contact sheet:", len(ids), "plots")


if __name__ == "__main__":
    main()
