"""Assemble geo/hydro/dams/dams.json: everything the reservoir model and the page need to know about each dam.

Merges
  dams_sites.json        which reservoirs, their SAIH station and variables                    (written by hand)
  out/dam_points.json    catchment, dams above with travel time, control points below          (build_dams.py)
  out/curves.json        level-volume curve fitted to SAIH pairs                               (build_curves.py)
  out/saih_static.json   what SAIH Júcar itself says: volume at the maximum normal level, spillway level,
                         thresholds of the outflow (refreshed from the live page with `refresh`)
  dams_static.json       engineering data from the inventories (spillway, outlets, crest, reserves), with sources
  out/fit.json           quick-runoff share alpha per dam catchment, fitted to measured inflows  (check_events.py)

How the three SAIH Júcar numbers are read (checked on Forata: 27.5 hm3 at the 379.2 m "cota de vertido", 37.34 hm3
= "volumen NMN" at 384 m, the top of its gates):
  V(cota de vertido) <  0.97 NMN   gated spillway: sill at the cota de vertido, the gates hold up to the NMN volume
  V(cota de vertido) >  1.03 NMN   the NMN volume is an operating limit below the spillway (Contreras, Escalona):
                                   it is the flood-reserve limit, the spill starts at V(cota de vertido)
  otherwise                        free spillway at the NMN volume

    py -3.11 geo/hydro/dams/build_json.py [refresh]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"


def load(p, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def saih_static(refresh):
    f = OUT / "saih_static.json"
    if refresh or not f.exists():
        sys.path.insert(0, str(ROOT / "backend"))
        from riua.sources import gauges as G
        rows = {}
        for s in G._chj_embedded("mapa-embalses", "embalses"):
            rows[s["fldTCodigo"]] = dict(nmn_hm3=s.get("fldFVolumenNMN"), spill_level_m=s.get("fldFCotaVertido"),
                                         thr=[s.get("umbralBajoCaudalSalidaRio"), s.get("umbralMedioCaudalSalidaRio"),
                                              s.get("umbralAltoCaudalSalidaRio")],
                                         town=s.get("fldTPoblacion"), province=s.get("fldTProvincia"))
        seg = {}
        for it in json.loads(G._post(G.SEG_IVISOR, "segura_embalses.json", "action=consultar_embalses")):
            v, p = G._num(it.get("UltimoDatoVolumen")), G._num(it.get("PorcentajeCapacidad"))
            seg[it["CodPuntoMedicion"]] = dict(capacity_from_pct_hm3=round(100.0 * v / p, 2) if v and p and p >= 5 else None)
        f.write_text(json.dumps(dict(fetched=__import__("time").strftime("%Y-%m-%d"), chj=rows, seg=seg), ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return load(f)


def curve_v(c, h):
    return c["a"] * max(h - c["h0"], 0.0) ** c["b"]


def curve_h(c, v):
    return c["h0"] + (max(v, 0.0) / c["a"]) ** (1.0 / c["b"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sites = load(HERE / "dams_sites.json")["dams"]
    pts = {p["id"]: p for p in load(OUT / "dam_points.json")["points"]}
    curves = load(OUT / "curves.json", {})
    saih = saih_static(len(sys.argv) > 1 and sys.argv[1] == "refresh")
    static = {d["id"]: d for d in (load(HERE / "dams_static.json", {"dams": []})["dams"])}
    fit = load(OUT / "fit.json", {})
    h1 = {d["id"]: d for d in load(HERE.parent / "catchments" / "build" / "dams_def.json")["dams"]}
    dams, rows = [], []
    for s in sites:
        p, st, c = pts[s["id"]], static.get(s["id"], {}), curves.get(s["id"])
        net, code = s["saih"].split(":")
        d = dict(id=s["id"], name=s["name"], river=s["river"], lat=p["lat"], lon=p["lon"], saih=s["saih"], vars=s["vars"],
                 source="saih_chj" if net == "chj" else "saih_segura",
                 area_km2=p["area_km2"], own_km2=p["own_km2"], inside_box_fraction=p["inside_box_fraction"],
                 t_longest_h=p["t_longest_h"], cuts=p["cuts"], upstream=p["upstream"], next_dam=p["next_dam"], points=p["points"])
        unverified = list(st.get("unverified", []))
        sp = dict(st.get("spill") or {})
        nmn = sill_level = None
        if net == "chj":
            ss = saih["chj"].get(code, {})
            nmn, sill_level = ss.get("nmn_hm3"), ss.get("spill_level_m")
            d["thr"] = ss.get("thr")
            d["town"] = ss.get("town")
        else:
            nmn = st.get("capacity_hm3") or (saih["seg"].get(code) or {}).get("capacity_from_pct_hm3") or (h1.get(s["id"]) or {}).get("capacity_hm3")
            if not st.get("capacity_hm3"):
                unverified.append("capacity_hm3 (from the SAIH Segura percentage or embalses.net)")
        d["nmn_hm3"] = nmn
        kind = "unknown"
        v_spill, v_res = nmn, None
        if st.get("spill", {}).get("gated") is not None:
            sp["gated"] = st["spill"]["gated"]
        if c:
            d["curve"] = dict(a=c["a"], b=c["b"], h0=c["h0"], level_min=c["level_min"], level_max=c["level_max"], rel_rms=round(c["rel_rms"], 4),
                              source="power law fitted to SAIH level-volume pairs (build_curves.py)")
            if sill_level is not None and nmn:
                v_sill = curve_v(c, sill_level)
                d["v_sill_hm3"] = round(v_sill, 2)
                if v_sill < 0.97 * nmn:
                    kind, v_spill = "gated", nmn
                    sp["gated"] = True
                elif v_sill > 1.03 * nmn:
                    kind, v_spill, v_res = "operating limit below the spillway", v_sill, nmn
                else:
                    kind = "free"
                sp.setdefault("sill_level_m", sill_level)
                if st.get("reserve_at_nmn") and st.get("crest_level_m"):
                    sill = float(st["crest_level_m"]) - 3.0
                    kind, v_spill, v_res = "operating limit below the spillway", curve_v(c, sill), nmn
                    sp["sill_level_m"] = sill
                    unverified.append("spill level (generic: crest - 3 m)")
        elif nmn and st.get("crest_level_m") and st.get("bed_level_m") is not None:
            # no level above sea level in SAIH Segura: generic curve V = V_s ((h - bed) / h_s)^2.5, spill level 3 m below the crest
            hs = float(st["crest_level_m"]) - 3.0 - float(st["bed_level_m"])
            if hs > 2.0:
                d["curve"] = dict(a=nmn / hs ** 2.5, b=2.5, h0=float(st["bed_level_m"]),
                                  source="GENERIC: V = V_s ((h - bed) / h_s)^2.5, spill level = crest - 3 m (SEPREM crest and bed levels)")
                sp.setdefault("sill_level_m", round(float(st["crest_level_m"]) - 3.0, 2))
                kind = "free"
                unverified.append("curve and spill level (generic)")
        d["spill_kind"] = kind
        d["v_spill_hm3"] = round(v_spill, 3) if v_spill else None
        if st.get("reserve"):
            d["reserve"] = st["reserve"]
        elif v_res:
            d["reserve"] = [dict(months=list(range(1, 13)), max_hm3=round(v_res, 2),
                                 source="SAIH Júcar: volume at the maximum normal (operating) level, below the spillway")]
        # spillway rating: design capacity at the design head
        cv = d.get("curve")
        if sp.get("capacity_m3s") and cv and v_spill and not sp.get("design_head_m"):
            sill = sp.get("sill_level_m") if sp.get("sill_level_m") is not None else curve_h(cv, v_spill)
            h_top = curve_h(cv, v_spill)
            if kind == "gated" and h_top - sill > 0.3:
                sp["design_head_m"], sp["design_head_source"] = round(h_top - sill, 2), "capacity reached at the maximum normal level (gates open)"
            elif st.get("crest_level_m") and st["crest_level_m"] - 1.5 > sill:
                sp["design_head_m"], sp["design_head_source"] = round(st["crest_level_m"] - 1.5 - sill, 2), "capacity reached 1.5 m below the crest"
            unverified.append("spill.design_head_m (assumed, see design_head_source)")
        if st.get("rating_c"):
            sp["rating_c"], sp["rating_note"] = st["rating_c"], st.get("rating_note")
        if sp:
            d["spill"] = sp
        if cv and st.get("crest_level_m") and v_spill and curve_v(cv, st["crest_level_m"]) > 1.02 * v_spill:
            d["v_top_hm3"] = round(curve_v(cv, st["crest_level_m"]), 2)
        for k in ("owner", "year", "dam_type", "height_m", "crest_level_m", "nmn_level_m", "nap_level_m", "capacity_total_hm3",
                  "dead_hm3", "outlet_m3s", "outlet_open", "flood_only", "catchment_inventory_km2", "design_flood_m3s", "uses", "bed_level_m", "reserve_note", "notes", "sources"):
            if st.get(k) is not None:
                d[k] = st[k]
        if s["id"] in fit and fit[s["id"]].get("alpha") is not None:
            d["alpha"] = fit[s["id"]]["alpha"]
        d["unverified"] = sorted(set(unverified))
        dams.append(d)
        rows.append(f"{d['id']:15s} {kind[:10]:10s} NMN={nmn}  V_sill={d.get('v_sill_hm3')}  V_spill={d['v_spill_hm3']}  "
                    f"res={d.get('reserve', [{}])[0].get('max_hm3')}  Qd={sp.get('capacity_m3s')}  Hd={sp.get('design_head_m')}  "
                    f"top={d.get('v_top_hm3')}  alpha={d.get('alpha')}")
    meta = dict(generated=__import__("time").strftime("%Y-%m-%d"), agent="q10-dams",
                note="built by geo/hydro/dams/build_json.py; edit dams_sites.json / dams_static.json, not this file",
                not_modelled="dams of the h1 inventory without live data (water passes through): Cortes II, El Naranjero, El Molinar, "
                             "Tibi, Elx, Relleu, Isbert, Taibilla, Anchuricas")
    (HERE / "dams.json").write_text(json.dumps(dict(meta=meta, dams=dams), ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(rows))
    print(len(dams), "dams ->", HERE / "dams.json")
