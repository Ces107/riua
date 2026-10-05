"""dams_static.json from the SEPREM dam fichas (Sociedad Española de Presas y Embalses, https://www.seprem.es/ficha.php?idpresa=N).

Input : scratch/q10-dams/raw/inv/seprem_parsed.json   (the ficha fields as published, one dict per dam; downloaded 2026-10-02)
Output: geo/hydro/dams/dams_static.json

Taken from the ficha: crest level ("Cota coronación"), river-bed level ("Cota cauce"), height, spillway capacity (sum of the
spillways), regulation (gates or a fixed lip), bottom-outlet capacity (sum), catchment, owner, year, design flood, uses.
Manual overrides (OVERRIDE below) are measured facts with their source.

Assumptions (marked in `unverified`): a reservoir whose only use is flood defence ("Defensa frente a avenidas") is a
flood-control dam kept empty with its bottom outlet open; the SEPREM spillway capacity is reached with the reservoir at
its maximum normal level for a gated spillway, and 1.5 m below the crest for a fixed lip.

    py -3.11 geo/hydro/dams/build_static.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "scratch" / "q10-dams" / "raw" / "inv" / "seprem_parsed.json"

# measured facts that override or complete the fichas
OVERRIDE = {
    "tous": {"reserve_at_nmn": True, "reserve_note": "SAIH Júcar NMN 378.6 hm3 at 130 m is the flood-control operating limit of Tous "
                                                    "(crest 162.5 m); the spillway sill is not published here: generic, crest - 3 m"},
    "forata": {"rating_c": 80.0, "rating_note": "1092 m3/s measured over the spillway on 29 Oct 2024 with the level at 384.9 m, sill 379.2 m "
                                                "(CHJ figures quoted in coord/findings/q5-science.md A.3): C = 1092 / 5.7^1.5"},
}


def num(s):
    """'1300,000 - 0,000' -> [1300.0, 0.0]"""
    out = []
    for x in re.findall(r"-?\d[\d.]*,?\d*", s or ""):
        try:
            out.append(float(x.replace(".", "").replace(",", ".")))
        except ValueError:
            pass
    return out


def fix(s):
    try:
        return s.encode("latin-1").decode("utf-8") if "Ã" in s else s
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    src = json.loads(SRC.read_text(encoding="utf-8"))
    out = []
    for did, f in src.items():
        g = lambda k: f.get(k) or ""
        crest = (num(g("Cota coronación (m)")) or [None])[0]
        bed = (num(g("Cota cauce (m)")) or [None])[0]
        sp = [x for x in num(g("Capacidad aliviaderos (m3/s)")) if x > 0]
        ds = [x for x in num(g("Capacidad desagüe (m3/s)")) if x > 0]
        reg = fix(g("Regulación"))
        uses = fix(g("Usos del embalse"))
        use_list = [u.strip() for u in uses.split("-") if u.strip()]
        flood_only = bool(use_list) and all("avenidas" in u for u in use_list)
        d = dict(id=did, owner=fix(g("Titular de la presa")).title() or None, dam_type=fix(g("Tipo de Presa")) or None,
                 height_m=(num(g("Altura desde cimientos (m)")) or [None])[0], crest_level_m=crest, bed_level_m=bed,
                 catchment_inventory_km2=(num(g("Superficie de la cuenca (km2)")) or [None])[0],
                 design_flood_m3s=(num(g("Avenida de Proyecto (m3/s)")) or [None])[0],
                 year=int(g("Fin de las obras")[-4:]) if re.search(r"\d{4}$", g("Fin de las obras")) else None,
                 uses=use_list, flood_only=flood_only, outlet_m3s=round(sum(ds), 1) if ds else None,
                 sources={"ficha": f.get("_url")}, unverified=[])
        spill = {}
        if sp:
            spill["capacity_m3s"] = round(sum(sp), 1)
        if "ompuert" in reg:
            spill["gated"] = True
        elif "abio fijo" in reg:
            spill["gated"] = False
        spill["regulation"] = reg
        if spill:
            d["spill"] = spill
        if flood_only:
            d["outlet_open"] = True
            d["unverified"].append("outlet_open (assumed: flood-control dam kept empty with the bottom outlet open)")
        if not sp:
            d["unverified"].append("spill.capacity_m3s (not in the ficha)")
        if did in OVERRIDE:
            d.update(OVERRIDE[did])
        out.append(d)
        print(f"{did:15s} crest={crest} bed={bed} spill={spill.get('capacity_m3s')} gated={spill.get('gated')} outlet={d['outlet_m3s']} "
              f"flood_only={flood_only} A={d['catchment_inventory_km2']}")
    meta = dict(source="SEPREM fichas (https://www.seprem.es/ficha.php?idpresa=N), parsed 2026-10-02; built by build_static.py")
    (HERE / "dams_static.json").write_text(json.dumps(dict(meta=meta, dams=out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(out), "dams")
