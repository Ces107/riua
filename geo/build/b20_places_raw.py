"""Municipalities: town-centre coordinates (IGN núcleos de población) + population (INE padrón).

Output (intermediate): scratch/r5-geo/places_raw.json  — basin/zone ids are added later by b60_finalize.py
Sources:
  - IGN OGC API Features, collection `nuc` (Núcleos de población, IGN/CNIG, CC-BY 4.0 scne.es)
    https://api-features.ign.es/collections/nuc/items?cpro=<prov>&skipGeometry=true
    fields: codine (11 digits = 5 INE municipality + 6 entity), capital, latitud, longitud, altitud, habitantes
  - INE, Padrón municipal "Población por sexo, municipios" tables per province (servicios.ine.es/wstempus JSON API)
"""
from __future__ import annotations

import json
import time
import unicodedata

import requests

from common import RAW, UA, write_json

# province code -> (name, INE table id)
PROVS = {
    "03": ("Alicante", 2856), "12": ("Castellón", 2865), "46": ("Valencia", 2903),
    # upstream / neighbouring provinces inside the box
    "02": ("Albacete", 2855), "16": ("Cuenca", 2869), "30": ("Murcia", 2883),
    "44": ("Teruel", 2899), "43": ("Tarragona", 2900),
}
CV = ("03", "12", "46")
S = requests.Session()
S.headers["User-Agent"] = UA


def get(url):
    for attempt in range(4):
        try:
            r = S.get(url, timeout=180)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # transient resets seen on api-features.ign.es
            print(f"  retry {attempt + 1}: {type(e).__name__}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def cached_json(path, fetch):
    if path.exists() and path.stat().st_size > 100:
        return json.loads(path.read_text(encoding="utf-8"))
    obj = fetch()
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return obj


def fetch_nuc(cpro: str) -> list[dict]:
    out, url = [], f"https://api-features.ign.es/collections/nuc/items?f=json&limit=1000&cpro={int(cpro)}&skipGeometry=true"
    while url:
        print("  GET", url)
        d = get(url).json()
        out += [f["properties"] for f in d["features"]]
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
        time.sleep(1.0)
    return out


def fetch_ine(table: int) -> list[dict]:
    url = f"https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/{table}?nult=1&tip=AM"
    print("  GET", url)
    d = get(url).json()
    time.sleep(1.0)
    return d


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if c.isalnum())


def main():
    places, stats = [], {}
    for cpro, (pname, table) in PROVS.items():
        ine = cached_json(RAW / f"ine_{table}.json", lambda: fetch_ine(table))
        pop, year = {}, None
        for row in ine:
            md = {m["T3_Variable"]: m for m in row["MetaData"]}
            if md.get("Sexo", {}).get("Nombre") != "Total":
                continue
            if "Municipios" in md and row["Data"]:
                code = md["Municipios"]["Codigo"]
                pop[code] = (md["Municipios"]["Nombre"], int(row["Data"][0]["Valor"]))
                year = row["Data"][0]["Anyo"]
            elif "Provincias" in md:
                assert md["Provincias"]["Codigo"] == cpro, (table, md["Provincias"])
        nuc = cached_json(RAW / f"ign_nuc_{cpro}.json", lambda: fetch_nuc(cpro))
        by_muni: dict[str, list[dict]] = {}
        for n in nuc:
            by_muni.setdefault(n["codine"][:5], []).append(n)
        n_cap = 0
        for code, (mname, p) in sorted(pop.items()):
            cands = by_muni.get(code, [])
            if not cands:
                print(f"  !! no IGN nucleus for {code} {mname}")
                continue
            caps = [n for n in cands if n.get("capital") not in (None, "", "000000")]
            if caps:
                n_cap += 1
                c = max(caps, key=lambda n: n.get("habitantes") or 0)
                how = "capital"
            else:  # fall back: nucleus named like the municipality, else the most populated one
                same = [n for n in cands if norm(n["nombre"]) in [norm(x) for x in mname.split("/")]]
                c = max(same or cands, key=lambda n: n.get("habitantes") or 0)
                how = "name" if same else "largest"
            places.append({
                "ine": code, "name": mname, "centre_name": c["nombre"], "centre_how": how,
                "lat": round(c["latitud"], 5), "lon": round(c["longitud"], 5),
                "alt_m": None if c.get("altitud") is None else round(c["altitud"]),
                "pop": p, "pop_year": year, "province": pname, "province_code": cpro, "in_cv": cpro in CV,
            })
        stats[cpro] = (pname, len(pop), n_cap, year)
    for cpro, s in stats.items():
        print(cpro, s)
    ncv = sum(p["in_cv"] for p in places)
    print("places:", len(places), "CV municipalities:", ncv)
    assert ncv == 542, ncv
    write_json(RAW / "places_raw.json", places)


if __name__ == "__main__":
    main()
