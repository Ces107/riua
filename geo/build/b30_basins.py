"""Basin units + topology for flash-flood aggregation.

Outputs: geo/out/basins.geojson, geo/out/basins_topology.json
Intermediate (used by b40/b50/b60): scratch/r5-geo/basin_raster.npz (label raster 0.001 deg), basins_stageA.pkl

Method (see geo/out/SOURCES.md and coord/findings/r5-geo.md):
  * Inside the Júcar river-basin district (CHJ; Cenia .. Vinalopó, includes Poyo, Carraixet, Magro, Turia, Júcar,
    Mijares, Palancia, Serpis, Gorgos, Girona...):
      - unit polygons = official catchments of the surface water bodies of the Plan Hidrológico del Júcar 2022-2027
        (CHJ layer "Masas de agua superficial red Cuenca PHJ22", official names);
      - topology  = CEDEX/DGA 1:25.000 sub-basins with modified Pfafstetter codes (25 149 leaves): leaf -> leaf
        downstream pointers are decoded from the codes; a unit's downstream unit is the unit of the leaf reached
        through its exit leaf with the largest accumulated upstream area;
      - areas not covered by a water-body catchment (small coastal ravines, coastal strips, endorheic bits) are
        filled with CEDEX leaves (named after the CEDEX root river) or merged into the neighbour;
      - very large water-body catchments are split by their named CEDEX tributaries; very small ones are merged
        into their downstream (or, at the sea, upstream) unit.
  * Outside the Júcar district (Segura basin, Ebro tributaries of Castellón, small coastal basins of the south):
    HydroBASINS level 12 (NEXT_DOWN topology), coarsened by Pfafstetter prefix away from the CV; names from
    OpenStreetMap named waterways.
  * Everything is finally burnt into one 0.001 deg label raster, sieved, polygonised and simplified as a
    coverage (no slivers / overlaps between units).
"""
from __future__ import annotations

import json
import pickle
import re
import unicodedata
import warnings
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import rasterio.features
import rasterio.transform
import scipy.ndimage as ndi
import shapely
import shapely.ops
from shapely.geometry import LineString, Point, mapping, shape

from common import OUT, RAW, download, find_one, unzip, write_geojson, write_json

warnings.filterwarnings("ignore")

CHJ = "https://aps.chj.es/down/SHP/"
URLS = {
    "wb": CHJ + "F2333_PHJ_2022_2027_Masas_de_agua_superficial.zip",
    "sub": CHJ + "F851_Subcuencas_1_25000_DGA_CEDEX.zip",
    "riv": CHJ + "F850_Rios_1_25000_DGA_CEDEX.zip",
    "dem": CHJ + "F162_Ambito_de_la_Demarcacion_Hidrografica_del_Jucar.zip",
    "hybas": "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_eu_lev12_v1c.zip",
    "hyriv": "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_eu_shp.zip",
}

# label raster (bigger than the Riuà box so that whole basins fit)
EXT = (-3.2, 37.3, 1.0, 41.2)  # lon0, lat0, lon1, lat1
RES = 0.001
W = round((EXT[2] - EXT[0]) / RES)
H = round((EXT[3] - EXT[1]) / RES)
TR = rasterio.transform.from_origin(EXT[0], EXT[3], RES, RES)

SMALL_KM2 = 25.0      # water-body units below this are merged
BIG_KM2 = 450.0       # water-body units above this are split by named tributaries
SPLIT_MIN, SPLIT_MAX = 90.0, 450.0
GAP_RIVER_MIN = 20.0  # CEDEX root river outside any water-body catchment becomes a unit above this
PATCH_MIN = 30.0      # leftover coastal/endorheic patch becomes a unit above this
HY_SLIVER = 15.0
HY_GROUP_DIGITS = 9   # HydroBASINS units far from the CV are grouped by this Pfafstetter level


# ----------------------------------------------------------------------------- helpers
def cell_area_rows() -> np.ndarray:
    lat = EXT[3] - (np.arange(H) + 0.5) * RES
    return (RES * 111.195) * (RES * 111.195 * np.cos(np.radians(lat)))  # km2 per cell, per row


ROW_KM2 = cell_area_rows()


def label_areas(lab: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n + 1)
    for r0 in range(0, H, 200):
        blk = lab[r0:r0 + 200]
        w = np.repeat(ROW_KM2[r0:r0 + 200], W)
        out += np.bincount(blk.ravel(), weights=w, minlength=n + 1)[: n + 1]
    return out


def rasterize(geoms, values, dtype="int32"):
    return rasterio.features.rasterize(
        ((g, int(v)) for g, v in zip(geoms, values) if g is not None and not g.is_empty),
        out_shape=(H, W), transform=TR, fill=0, dtype=dtype)


def sample(lab, lon, lat):
    c = int((lon - EXT[0]) / RES)
    r = int((EXT[3] - lat) / RES)
    if 0 <= r < H and 0 <= c < W:
        return int(lab[r, c])
    return 0


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


LOWER = {"de", "del", "la", "las", "los", "el", "els", "les", "dels", "o", "y", "i", "en", "al", "a", "d'", "l'"}
FIXED = {"rio": "Río", "riu": "Riu", "canada": "Cañada"}


def build_vocab(texts) -> dict[str, str]:
    """unaccented lowercase word -> accented spelling (only if unambiguous), learnt from official mixed-case names."""
    seen = defaultdict(Counter)
    for t in texts:
        for w in re.findall(r"[^\W\d_]+", t):
            if len(w) > 2:
                seen[strip_accents(w).lower()][w] += 1
    vocab = {}
    for k, c in seen.items():
        forms = {w.lower() for w in c}
        if len(forms) == 1:
            w = c.most_common(1)[0][0]
            if strip_accents(w) != w:  # only keep words that actually carry an accent
                vocab[k] = w.lower()
    return vocab


def pretty(name: str, vocab: dict[str, str]) -> str:
    """'RAMBLA DE CHIVA O DE POYO' -> 'Rambla de Chiva o de Poyo' (accents restored only when unambiguous)."""
    out = []
    for i, tok in enumerate(re.split(r"(\s+|'|-)", name.strip())):
        if not tok or tok.isspace() or tok in "'-":
            out.append(tok)
            continue
        low = tok.lower()
        key = strip_accents(low)
        if key in FIXED:
            out.append(FIXED[key])
        elif i > 0 and (low in LOWER or low in ("d", "l")):
            out.append(low)
        else:
            w = vocab.get(key, low)
            out.append(w[:1].upper() + w[1:])
    return "".join(out)


# ----------------------------------------------------------------------------- stage A: load
def stage_a():
    cache = RAW / "basins_stageA.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    d = {}
    for k, u in URLS.items():
        z = download(u, RAW / ("chj" if u.startswith(CHJ) else "") / u.rsplit("/", 1)[1])
        d[k] = unzip(z, z.with_suffix(""))
    print("reading CEDEX sub-basins ...")
    sub = gpd.read_file(find_one(d["sub"], "*.shp"))
    sub["geometry"] = shapely.simplify(sub.geometry.values, 10.0)
    sub = sub.to_crs(4326)
    print("reading CEDEX rivers ...")
    riv = gpd.read_file(find_one(d["riv"], "*.shp"))
    riv["geometry"] = shapely.simplify(riv.geometry.values, 15.0)
    riv = riv.to_crs(4326)
    print("reading PHJ water-body catchments ...")
    wb = gpd.read_file(find_one(d["wb"], "*red_Cuenca_PHJ22.shp"))
    wb["geometry"] = shapely.simplify(wb.geometry.values, 10.0)
    wb = wb.to_crs(4326)
    wbl = gpd.read_file(find_one(d["wb"], "*red_R*o_PHJ22.shp"))
    wbl["geometry"] = shapely.simplify(wbl.geometry.values, 15.0)
    wbl = wbl.to_crs(4326)
    dem = gpd.read_file(find_one(d["dem"], "F162C175_Demarcacion.shp")).to_crs(4326)
    print("reading HydroBASINS / HydroRIVERS ...")
    hy = gpd.read_file(find_one(d["hybas"], "hybas_eu_lev12_v1c.shp"), bbox=EXT)
    hr = gpd.read_file(find_one(d["hyriv"], "HydroRIVERS_v10_eu.shp"), bbox=EXT)
    hr = hr[hr.UPLAND_SKM >= 20].copy()
    st = {
        "sub": sub[["PFAFCUEN", "PFAFRIO", "NomRio", "CuencaKm2", "geometry"]],
        "riv": riv[["PFAFRIO", "NomRio", "LongRioKm", "geometry"]],
        "wb": wb[["CodMasaAgu", "MasaAguSup", "Categoria", "Naturaleza", "SistExplot", "geometry"]],
        "wbl": wbl[["CodMasaAgu", "MasaAguSup", "geometry"]],
        "dem": dem, "hy": hy, "hr": hr,
    }
    cache.write_bytes(pickle.dumps(st, protocol=4))
    return st


# ----------------------------------------------------------------------------- Pfafstetter leaf graph
def leaf_graph(codes: list[str], rios: list[str], river_set: set[str]):
    """Return down[i] (index or -1) and term[i] ('sea'|'endo') for terminal leaves."""
    by_river = defaultdict(list)
    for i, (c, r) in enumerate(zip(codes, rios)):
        assert c.startswith(r), (c, r)
        by_river[r].append((c[len(r):], i))
    for r in by_river:
        by_river[r].sort()
    import bisect

    def lower(r, suf):
        lst = by_river.get(r)
        if not lst:
            return -1
        k = bisect.bisect_left(lst, (suf, -1))
        return lst[k - 1][1] if k > 0 else -1

    def parent(r):
        for k in range(len(r) - 1, 3, -1):
            if r[:k] in river_set:
                return r[:k]
        return None

    down = np.full(len(codes), -1, dtype=np.int64)
    term = [None] * len(codes)
    for i, (c, r) in enumerate(zip(codes, rios)):
        if r not in river_set:  # coastal strip or closed leaf: PFAFRIO is not a real river
            term[i] = "endo" if "0" in c[4:] else "sea"
            continue
        rr, suf = r, c[len(r):]
        while True:
            j = lower(rr, suf) if suf else -1
            if j >= 0:
                down[i] = j
                break
            p = parent(rr)
            if p is None:
                term[i] = "endo" if "0" in rr[4:] else "sea"
                break
            suf, rr = rr[len(p):], p
            if suf.startswith("0") or "0" in suf:  # closed (endorheic) system hanging from river p
                term[i] = "endo"
                break
    return down, term


def accumulate(down: np.ndarray, area: np.ndarray) -> np.ndarray:
    n = len(down)
    indeg = np.zeros(n, dtype=np.int64)
    for d in down:
        if d >= 0:
            indeg[d] += 1
    cum = area.astype(float).copy()
    stack = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while stack:
        i = stack.pop()
        seen += 1
        d = down[i]
        if d >= 0:
            cum[d] += cum[i]
            indeg[d] -= 1
            if indeg[d] == 0:
                stack.append(d)
    assert seen == n, "cycle in the leaf graph"
    return cum


NAME_RE = re.compile(r"^(?P<river>[^:]+):\s*(?P<a>.+?)\s+-\s+(?P<b>.+)$")


def merge_names(up_name: str, down_name: str, keep: str) -> str:
    a, b = NAME_RE.match(up_name), NAME_RE.match(down_name)
    if a and b and a["river"].strip() == b["river"].strip():
        key = lambda s: strip_accents(s).lower().strip()  # noqa: E731
        if key(b["b"]) == key(a["a"]) and key(a["b"]) != key(b["a"]):
            a, b = b, a  # the two reaches were handed over in reverse order (unit without own leaf)
        return f"{a['river'].strip()}: {a['a'].strip()} - {b['b'].strip()}"
    return keep


# ----------------------------------------------------------------------------- main
def main():
    st = stage_a()
    sub, riv, wb, hy, hr = st["sub"], st["riv"], st["wb"], st["hy"], st["hr"]
    places = json.loads((RAW / "places_raw.json").read_text(encoding="utf-8"))
    vocab = build_vocab(list(wb.MasaAguSup) + [p["name"] for p in places] + [p["centre_name"] for p in places])

    n_leaf, n_wb = len(sub), len(wb)
    codes, rios = list(sub.PFAFCUEN), list(sub.PFAFRIO)
    leaf_area = sub.CuencaKm2.values.astype(float)
    river_set = set(riv.PFAFRIO)
    river_name = dict(zip(riv.PFAFRIO, riv.NomRio))
    river_geom = dict(zip(riv.PFAFRIO, riv.geometry))
    down, term = leaf_graph(codes, rios, river_set)
    cum = accumulate(down, leaf_area)
    print(f"leaves {n_leaf}: terminal sea {sum(t == 'sea' for t in term)}, endo {sum(t == 'endo' for t in term)}")
    jucar = [i for i in range(n_leaf) if rios[i] == "2002"]
    print(f"  check: max accumulated area on the Júcar main stem = {max(cum[i] for i in jucar):.0f} km2 (official basin 21 579 km2)")

    # ---- rasters
    print("rasterising ...")
    wb_r = rasterize(wb.geometry.values, np.arange(1, n_wb + 1), "int16")
    leaf_r = rasterize(sub.geometry.values, np.arange(1, n_leaf + 1), "int32")
    pair = np.bincount((leaf_r.astype(np.int64) * (n_wb + 1) + wb_r).ravel(), minlength=(n_leaf + 1) * (n_wb + 1))
    pair = pair.reshape(n_leaf + 1, n_wb + 1)
    leaf_wb = pair[1:].argmax(axis=1)  # 0 = none, else wb index + 1
    empty = pair[1:].sum(axis=1) == 0
    for i in np.where(empty)[0]:  # leaves smaller than a raster cell
        p = sub.geometry.values[i].representative_point()
        leaf_wb[i] = sample(wb_r, p.x, p.y)
    # Main-stem leaves follow the water body their own river runs through (the official catchments of short
    # main-stem water bodies are narrow corridors; area majority would hand the leaf to a neighbouring tributary).
    leaf_geoms = sub.geometry.values
    n_byline = 0
    for i in range(n_leaf):
        line = river_geom.get(rios[i])
        if line is None or line.is_empty:
            continue
        seg = line.intersection(leaf_geoms[i])
        if seg.is_empty or seg.geom_type not in ("LineString", "MultiLineString") or seg.length < 0.003:
            continue
        nper = max(5, int(seg.length / 0.002))
        vals = []
        for part in getattr(seg, "geoms", [seg]):
            m = max(2, int(round(nper * part.length / seg.length)))
            pts = shapely.line_interpolate_point(part, np.linspace(0.03, 0.97, m), normalized=True)
            vals += [sample(wb_r, p.x, p.y) for p in pts]
        vals = np.array(vals)
        nzv = vals[vals > 0]
        if len(nzv) * 2 >= len(vals):
            k = int(np.bincount(nzv).argmax())
            if k != leaf_wb[i]:
                n_byline += 1
            leaf_wb[i] = k
    print(f"  leaves relabelled by their river line: {n_byline}")
    wb_ids = ["J" + c for c in wb.CodMasaAgu]
    wb_name = dict(zip(wb_ids, wb.MasaAguSup))

    # ---- leaf -> unit labels
    unit_of_leaf: list[str | None] = [wb_ids[k - 1] if k > 0 else None for k in leaf_wb]
    gap = [i for i in range(n_leaf) if unit_of_leaf[i] is None]
    groups = defaultdict(list)
    for i in gap:  # follow each uncovered leaf downstream until it enters a water-body catchment or ends
        j = i
        while True:
            d = down[j]
            if d < 0:
                root = rios[j]
                groups[("river", root, None) if root in river_set else ("strip", "", None)].append(i)
                break
            if leaf_wb[d] > 0:
                groups[("trib", rios[j], unit_of_leaf[d])].append(i)
                break
            j = d
    units: dict[str, dict] = {}
    for k, uid in enumerate(wb_ids):
        units[uid] = {"name": wb.MasaAguSup.values[k], "kind": "wb", "category": wb.Categoria.values[k],
                      "members": [wb.CodMasaAgu.values[k]], "source": "CHJ PHJ 2022-2027 masas de agua superficial (cuenca)"}
    n_attach = 0
    for (kind, key, tgt), idxs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        a = leaf_area[idxs].sum()
        if kind in ("river", "trib") and a >= GAP_RIVER_MIN and key in river_set and "JG" + key not in units:
            uid = "JG" + key
            print(f"  gap unit {uid} {river_name.get(key)} {a:.0f} km2" + (f" -> {tgt} {wb_name[tgt]}" if tgt else " -> terminal"))
            nm = river_name.get(key, "SIN NOMBRE")
            # irrigation canals (Sequia Gran ...) are not basin names -> treated as unnamed
            unnamed = nm.startswith("SIN ") or re.match(r"^(SEQUIA|ACEQUIA|CANAL|AZARBE)\b", nm)
            units[uid] = {"name": None if unnamed else pretty(nm, vocab), "kind": "gap_river",
                          "members": [key], "source": "CEDEX/DGA subcuencas 1:25.000 (río " + key + ")",
                          "endo": "0" in key[4:]}
            for i in idxs:
                unit_of_leaf[i] = uid
        elif kind == "trib":
            for i in idxs:
                unit_of_leaf[i] = tgt
            n_attach += len(idxs)
        else:
            for i in idxs:
                unit_of_leaf[i] = "STRIP"
    print(f"gap leaves {len(gap)} ({leaf_area[gap].sum():.0f} km2): attached {n_attach}, "
          f"gap-river units {sum(u['kind'] == 'gap_river' for u in units.values())}, "
          f"strip leaves {sum(u == 'STRIP' for u in unit_of_leaf)}")

    # ---- split very large water-body units by named tributaries
    leaves_of = defaultdict(list)
    for i, u in enumerate(unit_of_leaf):
        leaves_of[u].append(i)
    prefix_total = Counter()
    for c, a in zip(codes, leaf_area):  # total area under every river prefix
        for k in range(5, len(c) + 1):
            if c[:k] in river_set:
                prefix_total[c[:k]] += a
    child_parent = {}
    for uid in list(units):
        if units[uid]["kind"] != "wb":
            continue
        L = leaves_of.get(uid, [])
        tot = leaf_area[L].sum() if L else 0
        if tot <= BIG_KM2:
            continue
        inside = Counter()
        for i in L:
            c = codes[i]
            for k in range(5, len(c) + 1):
                if c[:k] in river_set:
                    inside[c[:k]] += leaf_area[i]
        cands = [(a, r) for r, a in inside.items()
                 if SPLIT_MIN <= a <= SPLIT_MAX and abs(a - prefix_total[r]) < 0.5
                 and not river_name.get(r, "SIN").startswith("SIN ")]
        chosen = []
        for a, r in sorted(cands, reverse=True):
            if any(r.startswith(c) or c.startswith(r) for _, c in chosen):
                continue
            if tot - sum(x for x, _ in chosen) - a < 50:
                continue
            chosen.append((a, r))
        for a, r in chosen:
            cid = f"{uid}~{r}"
            units[cid] = {"name": pretty(river_name[r], vocab), "kind": "split", "members": [r], "within": units[uid]["name"],
                          "source": "CEDEX/DGA subcuencas 1:25.000 (río " + r + ") inside CHJ water-body catchment " + uid[1:]}
            child_parent[cid] = uid
            for i in L:
                if codes[i].startswith(r):
                    unit_of_leaf[i] = cid
        if chosen:
            print(f"  split {uid} {units[uid]['name']} ({tot:.0f} km2) -> " + ", ".join(f"{units[uid + '~' + r]['name']} {a:.0f}" for a, r in chosen))

    # ---- label raster
    uid_list = list(units)
    uidx = {u: k + 1 for k, u in enumerate(uid_list)}
    STRIP = 32000
    leaf_lab = np.zeros(n_leaf + 1, dtype=np.int32)
    leaf_is_child = np.zeros(n_leaf + 1, dtype=bool)
    for i, u in enumerate(unit_of_leaf):
        leaf_lab[i + 1] = STRIP if u == "STRIP" else uidx[u]
        leaf_is_child[i + 1] = u in child_parent
    wb_lab = np.zeros(n_wb + 1, dtype=np.int32)
    for k, u in enumerate(wb_ids):
        wb_lab[k + 1] = uidx[u]
    child_par_idx = np.zeros(len(uid_list) + 1, dtype=np.int32)
    for c, p in child_parent.items():
        child_par_idx[uidx[c]] = uidx[p]
    lab = wb_lab[wb_r]
    ll = leaf_lab[leaf_r]
    use_leaf = (lab == 0) | (leaf_is_child[leaf_r] & (child_par_idx[np.minimum(ll, len(uid_list))] == lab))
    lab = np.where(use_leaf, ll, lab).astype(np.int32)
    del ll, use_leaf

    # leftover patches (coastal strips, closed bits, tiny rivers): own unit if big, else nearest unit
    strip_mask = lab == STRIP
    comp, ncomp = ndi.label(strip_mask)
    comp_area = np.zeros(ncomp + 1)
    for r0 in range(0, H, 200):
        comp_area += np.bincount(comp[r0:r0 + 200].ravel(), weights=np.repeat(ROW_KM2[r0:r0 + 200], W), minlength=ncomp + 1)
    patch_units = {}
    for c in range(1, ncomp + 1):
        if comp_area[c] >= PATCH_MIN:
            uid = f"JP{len(patch_units) + 1:02d}"  # (JLxx are official lake water bodies)
            units[uid] = {"name": None, "kind": "patch", "members": [], "source": "CEDEX/DGA subcuencas 1:25.000 (franja sin masa de agua)"}
            uid_list.append(uid)
            uidx[uid] = len(uid_list)
            patch_units[c] = uidx[uid]
    remap = np.zeros(ncomp + 1, dtype=np.int32)
    for c, k in patch_units.items():
        remap[c] = k
    lab = np.where(strip_mask, remap[comp], lab)
    # leaves that are majority inside a patch unit take that unit (they are terminal anyway)
    lp = np.bincount((leaf_r.astype(np.int64))[strip_mask], minlength=n_leaf + 1)
    for c, k in patch_units.items():
        m = comp == c
        cnt = np.bincount(leaf_r[m], minlength=n_leaf + 1)
        for li in np.where((cnt > 0) & (cnt * 2 > lp))[0]:
            if li > 0 and unit_of_leaf[li - 1] == "STRIP":
                unit_of_leaf[li - 1] = uid_list[k - 1]
                units[uid_list[k - 1]]["endo"] = units[uid_list[k - 1]].get("endo", True) and ("0" in codes[li - 1][4:])
    todo = strip_mask & (lab == 0)
    print(f"leftover patches: {ncomp}, as units {len(patch_units)}, nearest-filled {(todo * ROW_KM2[:, None]).sum():.0f} km2")
    if todo.any():
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = np.where(todo, lab[idx[0], idx[1]], lab)
        del idx
    del comp, strip_mask, todo
    chj_mask = lab > 0

    # ---- unit topology inside CHJ (exit leaf with the largest accumulated area)
    leaves_of = defaultdict(list)
    for i, u in enumerate(unit_of_leaf):
        leaves_of[u].append(i)

    def exit_of(uid):
        best, bi = -1.0, None
        for i in leaves_of.get(uid, []):
            d = down[i]
            if (d < 0 or unit_of_leaf[d] != uid) and cum[i] > best:
                best, bi = cum[i], i
        return bi

    def walk_next(i):
        """first unit different from STRIP reached downstream of exit leaf i; (uid|None, terminal kind)"""
        j = i
        while True:
            d = down[j]
            if d < 0:
                return None, term[j]
            if unit_of_leaf[d] != "STRIP":
                return unit_of_leaf[d], None
            j = d

    nxt, terminal, exit_leaf = {}, {}, {}
    for uid in units:
        e = exit_of(uid)
        exit_leaf[uid] = e
        if e is None:  # no leaf has this unit as majority: it sits inside a bigger leaf -> flows to that leaf's unit
            k = uidx[uid]
            if uid in wb_ids:
                col = pair[1:, wb_ids.index(uid) + 1]
                li = int(col.argmax())
                v = unit_of_leaf[li] if col[li] > 0 else None
                if v == "STRIP" or v == uid:
                    v = None
                nxt[uid], terminal[uid] = v, None if v else "sea"
            else:
                nxt[uid], terminal[uid] = None, "endo" if units[uid].get("endo") else "sea"
            continue
        v, t = walk_next(e)
        if v == uid:
            v = None
        nxt[uid], terminal[uid] = v, t
    # sanity: no cycles
    for uid in units:
        seen, u = set(), uid
        while u is not None:
            assert u not in seen, f"cycle at {uid}: {seen}"
            seen.add(u)
            u = nxt.get(u)

    # ---- HydroBASINS outside the Júcar district
    zones = json.loads((OUT / "zones.geojson").read_text(encoding="utf-8"))
    cv_r = rasterize([shape(f["geometry"]) for f in zones["features"]], [1] * len(zones["features"]), "uint8").astype(bool)
    hy = hy.reset_index(drop=True)
    hy_r = rasterize(hy.geometry.values, np.arange(1, len(hy) + 1), "int32")
    free = lab == 0
    w = np.repeat(ROW_KM2, W)
    tot = np.bincount(hy_r.ravel(), weights=w, minlength=len(hy) + 1)
    fre = np.bincount(hy_r.ravel(), weights=w * free.ravel(), minlength=len(hy) + 1)
    cvf = np.bincount(hy_r.ravel(), weights=w * (free & cv_r).ravel(), minlength=len(hy) + 1)
    hid = list(hy.HYBAS_ID.astype(np.int64))
    hpos = {h: k for k, h in enumerate(hid)}
    # ENDO == 2 marks an endorheic sink: its NEXT_DOWN is only a *virtual* link to the outlet of the main basin
    # (not adjacent, no real flow) -> treated as a terminal closed basin.
    hsink = (hy.ENDO.values == 2)
    hdown = [-1 if hsink[k] else hpos.get(int(x), -1) for k, x in enumerate(hy.NEXT_DOWN)]
    mostly_free = fre[1:] > 0.5 * np.maximum(tot[1:], 1e-9)
    near = (cvf[1:] >= 2.0)
    S = set(np.where(near & mostly_free & (fre[1:] >= HY_SLIVER))[0])
    incl = set(S)
    reach = {}

    def reaches_S(k):
        path = []
        while k >= 0 and k not in reach:
            path.append(k)
            if k in S:
                reach[k] = True
                break
            k = hdown[k]
        res = reach.get(k, False) if k >= 0 else False
        for p in path:
            reach[p] = res
        return res

    for k in range(len(hy)):
        if mostly_free[k] and reaches_S(k):
            incl.add(k)
    # coarsen far units by Pfafstetter prefix
    pf = [str(int(x)) for x in hy.PFAF_ID]
    grp = defaultdict(list)
    for k in incl:
        grp[pf[k][:HY_GROUP_DIGITS]].append(k)
    hy_unit = {}
    for g, ks in grp.items():
        if len(ks) > 1 and not any(k in S for k in ks):
            uid = "HG" + g
            for k in ks:
                hy_unit[k] = uid
            units[uid] = {"name": None, "kind": "hybas_group", "members": [int(hid[k]) for k in ks],
                          "source": f"HydroBASINS v1c level 12, grouped at Pfafstetter level {HY_GROUP_DIGITS}"}
        else:
            for k in ks:
                uid = f"H{hid[k]}"
                hy_unit[k] = uid
                units[uid] = {"name": None, "kind": "hybas", "members": [int(hid[k])], "source": "HydroBASINS v1c level 12"}
    hy_lab = np.zeros(len(hy) + 1, dtype=np.int32)
    for k, uid in hy_unit.items():
        if uid not in uidx:
            uid_list.append(uid)
            uidx[uid] = len(uid_list)
        hy_lab[k + 1] = uidx[uid]
    lab_j = lab.copy()
    lab = np.where(free, hy_lab[hy_r], lab)
    for uid in set(hy_unit.values()):
        ks = [k for k, u in hy_unit.items() if u == uid]
        kout = max(ks, key=lambda k: hy.UP_AREA.values[k])
        d = hdown[kout]
        units[uid]["_kout"] = kout
        if d < 0:
            nd = int(hy.NEXT_DOWN.values[kout])
            nxt[uid] = None
            terminal[uid] = "endo" if hsink[kout] else ("sea" if nd == 0 else "outside")
        elif d in hy_unit:
            nxt[uid], terminal[uid] = hy_unit[d], None
        else:
            # downstream HydroBASINS unit is not ours: if it lies (mostly) inside the Júcar district, link to the
            # Júcar unit that covers most of it, otherwise the flow leaves the domain (Ebro...)
            cnt = np.bincount(lab_j[hy_r == d + 1], minlength=2)
            tot_d = cnt.sum()
            cnt[0] = 0
            if tot_d and cnt.max() > 0 and cnt.sum() > 0.5 * tot_d:
                nxt[uid], terminal[uid] = uid_list[int(cnt.argmax()) - 1], None
            elif int(hy.ENDO.values[kout]) >= 1:  # part of a closed basin whose sink is not in the domain
                nxt[uid], terminal[uid] = None, "endo"
            else:
                nxt[uid], terminal[uid] = None, "outside"
    # CV land still unlabelled (slivers along the district divide / coast): nearest unit
    todo = cv_r & (lab == 0)
    print(f"HydroBASINS: near-CV units {len(S)}, included lev12 {len(incl)}, final H units {len(set(hy_unit.values()))}; "
          f"CV cells still unlabelled {(todo * ROW_KM2[:, None]).sum():.0f} km2 -> nearest unit")
    if todo.any():
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = np.where(todo, lab[idx[0], idx[1]], lab)
        del idx

    # ---- merge small units
    area = dict(zip(uid_list, label_areas(lab, len(uid_list))[1:]))
    merged_into = {}
    relabel = np.arange(len(uid_list) + 1, dtype=np.int32)
    ups = defaultdict(set)
    for u, v in nxt.items():
        if v:
            ups[v].add(u)

    def up_area(u, memo={}):
        return area[u] + sum(up_area(x) for x in ups[u])

    while True:
        small = sorted((a, u) for u, a in area.items() if a < SMALL_KM2 and u not in merged_into and units[u]["kind"] in ("wb", "split", "gap_river", "hybas", "hybas_group"))
        done = False
        for a, u in small:
            tgt, direction = None, None
            if nxt.get(u):
                tgt, direction = nxt[u], "down"
            elif ups[u]:
                tgt, direction = max(ups[u], key=up_area), "up"
            if tgt is None or area[tgt] + a > 600:
                continue
            if direction == "down":
                new_name = merge_names(units[u]["name"] or "", units[tgt]["name"] or "", units[tgt]["name"])
                for x in ups[u]:
                    nxt[x] = tgt
                    ups[tgt].add(x)
                ups[tgt].discard(u)
            else:
                new_name = merge_names(units[tgt]["name"] or "", units[u]["name"] or "", units[tgt]["name"])
                for x in ups[u] - {tgt}:
                    nxt[x] = tgt
                    ups[tgt].add(x)
                nxt[tgt], terminal[tgt] = nxt[u], terminal[u]
                if "_kout" in units[u] and "_kout" in units[tgt]:
                    units[tgt]["_kout"] = units[u]["_kout"]
            units[tgt]["name"] = new_name
            units[tgt]["members"] = units[tgt]["members"] + units[u]["members"]
            units[tgt].setdefault("merged_names", []).append(units[u]["name"])
            area[tgt] += a
            merged_into[u] = tgt
            relabel[relabel == uidx[u]] = uidx[tgt]
            for i in leaves_of.get(u, []):
                unit_of_leaf[i] = tgt
            leaves_of[tgt] += leaves_of.pop(u, [])
            ups.pop(u, None)
            del area[u], nxt[u]
            done = True
            break
        if not done:
            break
    lab = relabel[lab]
    print(f"merged {len(merged_into)} small units; units now {len(area)}")


    # ---- clean raster, compact ids
    lab = rasterio.features.sieve(lab.astype(np.int32), size=60, connectivity=4)
    present = np.unique(lab)
    present = present[present > 0]
    final_ids = [uid_list[k - 1] for k in present]
    assert not any(u in merged_into for u in final_ids)
    comp_map = np.zeros(len(uid_list) + 1, dtype=np.int16)
    for new, k in enumerate(present, start=1):
        comp_map[k] = new
    lab = comp_map[lab]
    n_units = len(final_ids)
    area_f = label_areas(lab, n_units)[1:]
    for u in list(nxt):
        if u not in final_ids:
            nxt.pop(u)
    for u in final_ids:
        v = nxt.get(u)
        while v is not None and v not in final_ids:
            v = merged_into.get(v)
        nxt[u] = v
    print(f"FINAL units: {n_units}; area stats km2 min {area_f.min():.1f} median {np.median(area_f):.0f} max {area_f.max():.0f}; "
          f"<25: {(area_f < 25).sum()}, 25-50: {((area_f >= 25) & (area_f < 50)).sum()}, >500: {(area_f > 500).sum()}")

    # ---- names
    by_unit_place = defaultdict(list)
    for p in places:
        k = sample(lab, p["lon"], p["lat"])
        if k:
            by_unit_place[final_ids[k - 1]].append(p)

    def town(uid):
        ps = by_unit_place.get(uid)
        if not ps:
            return None
        nm = max(ps, key=lambda p: p["pop"])["name"]
        nm = nm.split("/")[0]
        m = re.match(r"^(.*), (.+)$", nm)  # "Vall d'Uixó, la" -> "la Vall d'Uixó"
        if m:
            art = m.group(2)
            nm = art + ("" if art.endswith("'") else " ") + m.group(1)
        return nm

    # OSM named waterways for the HydroBASINS units
    osm_path = RAW / "osm_ww.json"
    osm = json.loads(osm_path.read_text(encoding="utf-8"))["elements"] if osm_path.exists() else []
    ww_len = defaultdict(lambda: defaultdict(float))  # unit -> (class, name) -> km
    for el in osm:
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        nm, cls = el["tags"].get("name"), el["tags"].get("waterway")
        for a, b in zip(g[:-1], g[1:]):
            k = sample(lab, (a["lon"] + b["lon"]) / 2, (a["lat"] + b["lat"]) / 2)
            if k and final_ids[k - 1].startswith("H"):
                dx = (a["lon"] - b["lon"]) * 111.195 * np.cos(np.radians(a["lat"]))
                dy = (a["lat"] - b["lat"]) * 111.195
                ww_len[final_ids[k - 1]][(cls, nm)] += float(np.hypot(dx, dy))
    for uid in final_ids:
        u = units[uid]
        t = town(uid)
        u["town"] = t
        if uid.startswith("H"):
            cand = ww_len.get(uid, {})
            best = None
            for cls in ("river", "stream"):
                c = [(l, nm) for (c_, nm), l in cand.items() if c_ == cls and l >= 3.0]
                if c:
                    best = max(c)[1]
                    break
            u["river"] = best
            u["name"] = (f"{best} ({t})" if t else best) if best else None
        elif u["kind"] == "patch":
            base = "Zona endorreica" if u.get("endo") else "Cuencas litorales"
            u["river"] = None
            u["name"] = f"{base} ({t})" if t else base
        elif u["kind"] in ("gap_river", "split"):
            u["river"] = u["name"]
        else:
            u["river"] = u["name"].split(":")[0].strip()
    # centroid of every unit (from the raster) -> nearest municipality, for disambiguation only
    rr_i, cc_i = np.nonzero(lab)
    kk = lab[rr_i, cc_i].astype(np.int64)
    cnt_u = np.maximum(np.bincount(kk, minlength=n_units + 1), 1)
    cen_lat = EXT[3] - (np.bincount(kk, weights=rr_i, minlength=n_units + 1) / cnt_u + 0.5) * RES
    cen_lon = EXT[0] + (np.bincount(kk, weights=cc_i, minlength=n_units + 1) / cnt_u + 0.5) * RES
    del rr_i, cc_i, kk
    pl_xy = np.array([[p["lon"], p["lat"]] for p in places])

    def near_town(uid):
        k = final_ids.index(uid) + 1
        d2 = ((pl_xy[:, 0] - cen_lon[k]) * 0.78) ** 2 + (pl_xy[:, 1] - cen_lat[k]) ** 2
        nm = places[int(d2.argmin())]["name"].split("/")[0]
        m = re.match(r"^(.*), (.+)$", nm)
        if m:
            nm = m.group(2) + ("" if m.group(2).endswith("'") else " ") + m.group(1)
        return nm

    # units without any reliable name stay generic (never invented): town if there is one, else where they flow to
    for uid in final_ids:
        u = units[uid]
        if u["name"] is not None:
            continue
        t = u["town"]
        v, ref = nxt.get(uid), None
        while v is not None and ref is None:
            ref = units[v].get("river")
            v = nxt.get(v)
        if u["kind"] == "gap_river" and nxt.get(uid) is None and not u.get("endo"):
            base = "Cuencas litorales"
        else:
            base = "Cuenca sin nombre"
        if t:
            u["name"] = f"{base} ({t})"
        elif ref:
            u["name"] = f"{base} (hacia {ref})"
        else:
            u["name"] = f"{base} (cerca de {near_town(uid)})"
    # make names unique (same river + same town): number them from upstream to downstream
    ups = defaultdict(set)
    for u, v in nxt.items():
        if v:
            ups[v].add(u)
    idx_of = {u: k for k, u in enumerate(final_ids)}
    up_all, up_km2 = {}, {}

    def collect(u):
        if u in up_all:
            return up_all[u]
        s = set()
        for x in ups[u]:
            s.add(x)
            s |= collect(x)
        up_all[u] = s
        up_km2[u] = area_f[idx_of[u]] + sum(area_f[idx_of[x]] for x in s)
        return s

    for u in final_ids:
        collect(u)
    dup = defaultdict(list)
    for u in final_ids:
        dup[units[u]["name"]].append(u)
    for nm, us in dup.items():
        if len(us) > 1:
            us.sort(key=lambda u: (up_km2[u], u))
            # reaches of one and the same river (each one upstream of the next) are numbered; unrelated units that
            # merely share a name are told apart by the nearest municipality
            chain = all(us[i] in up_all[us[i + 1]] for i in range(len(us) - 1))
            for n, u in enumerate(us, start=1):
                units[u]["name"] = f"{nm} — tramo {n} de {len(us)}" if chain else f"{nm} (cerca de {near_town(u)})"
    dup = defaultdict(list)
    for u in final_ids:
        dup[units[u]["name"]].append(u)
    for nm, us in dup.items():
        if len(us) > 1:
            for n, u in enumerate(sorted(us), start=1):
                units[u]["name"] = f"{nm} #{n}"

    # ---- outlets
    hr_by = {}
    for hb, upl, geom in zip(hr.HYBAS_L12.astype(np.int64), hr.UPLAND_SKM, hr.geometry):
        if hb not in hr_by or upl > hr_by[hb][0]:
            hr_by[hb] = (upl, geom)
    leaf_geom = sub.geometry.values
    outlet = {}
    for uid in final_ids:
        pt, how = None, "approx"
        if uid.startswith("H"):
            kout = units[uid]["_kout"]
            h = hr_by.get(int(hid[kout]))
            if h is not None:
                c = list(h[1].coords)
                pt, how = Point(c[-1]), "hydrorivers"
        else:
            e = exit_of(uid)
            if e is not None:
                d = down[e]
                line = river_geom.get(rios[e])
                eg = leaf_geom[e]
                if line is not None and not line.is_empty:
                    cands = [Point(line.coords[0]), Point(line.coords[-1])] if isinstance(line, LineString) else []
                    x = line.intersection(eg.boundary)
                    cands += [g if isinstance(g, Point) else g.representative_point() for g in getattr(x, "geoms", [x]) if not g.is_empty]
                    if cands:
                        ref = leaf_geom[d] if d >= 0 else eg.boundary
                        pt, how = min(cands, key=lambda p: p.distance(ref)), "cedex river"
                if pt is None and d >= 0:
                    pt = shapely.ops.nearest_points(eg, leaf_geom[d])[0]
        if pt is None:
            rr, cc = np.where(lab == idx_of[uid] + 1)
            pt = Point(EXT[0] + (cc.mean() + 0.5) * RES, EXT[3] - (rr.mean() + 0.5) * RES)
        outlet[uid] = (round(pt.x, 4), round(pt.y, 4), how)

    # ---- polygons from the raster (noded -> valid coverage -> simplify)
    print("polygonising ...")
    polys = defaultdict(list)
    for geom, val in rasterio.features.shapes(lab.astype(np.int16), mask=lab > 0, connectivity=4, transform=TR):
        polys[int(val)].append(shape(geom))
    geoms = [shapely.union_all(polys[k + 1]) if len(polys[k + 1]) > 1 else polys[k + 1][0] for k in range(n_units)]
    lines = shapely.union_all([g.boundary for g in geoms])  # nodes every junction
    faces = list(shapely.polygonize([lines]).geoms)
    by = defaultdict(list)
    for f in faces:
        p = f.representative_point()
        k = sample(lab, p.x, p.y)
        if k:
            by[k].append(f)
    cov = np.array([shapely.MultiPolygon(by[k + 1]) if len(by[k + 1]) > 1 else by[k + 1][0] for k in range(n_units)], dtype=object)
    print("  coverage valid before simplify:", bool(np.all(shapely.coverage_is_valid(cov))))
    # refine outlets of the Júcar-district units: where the river leaves the unit polygon towards the next unit
    n_ref = 0
    for k, uid in enumerate(final_ids):
        if uid.startswith("H") or outlet[uid][2] != "cedex river":
            continue
        e = exit_of(uid)
        d = down[e]
        lines = [river_geom.get(rios[e])]
        if d >= 0 and rios[d] != rios[e]:
            lines.append(river_geom.get(rios[d]))
        bnd = cov[k].boundary
        pts = []
        for ln in lines:
            if ln is None or ln.is_empty:
                continue
            x = ln.intersection(bnd)
            pts += [g if isinstance(g, Point) else g.representative_point() for g in getattr(x, "geoms", [x]) if not g.is_empty]
        if not pts:
            continue
        old = Point(outlet[uid][0], outlet[uid][1])
        v = nxt.get(uid)
        if v is not None:
            tgt = cov[idx_of[v]]
            best = min(pts, key=lambda p: (round(p.distance(tgt), 3), p.distance(old)))
            if best.distance(tgt) > 0.01:
                continue
        else:
            best = min(pts, key=lambda p: p.distance(old))
            if best.distance(old) > 0.05:
                continue
        outlet[uid] = (round(best.x, 4), round(best.y, 4), "cedex river x unit boundary")
        n_ref += 1
    print(f"  outlets refined on the unit boundary: {n_ref}")
    simp = shapely.coverage_simplify(cov, 0.0022)
    print("  coverage valid after simplify:", bool(np.all(shapely.coverage_is_valid(simp))))

    # ---- write
    def term_chain(u):
        chain = []
        while nxt.get(u):
            u = nxt[u]
            chain.append(u)
        return chain

    feats, topo = [], {}
    for k, uid in enumerate(final_ids):
        u = units[uid]
        chain = term_chain(uid)
        last = chain[-1] if chain else uid
        tk = terminal.get(last) or "sea"
        g = simp[k]
        lp = g.representative_point()
        props = {
            "id": uid, "idx": k, "name": u["name"], "river": u.get("river"), "town": u.get("town"),
            "kind": u["kind"], "area_km2": round(float(area_f[k]), 1), "up_area_km2": round(float(up_km2[uid]), 1),
            "next": nxt.get(uid), "terminal_unit": last, "terminal": tk, "drains_to_sea": tk == "sea",
            "n_upstream": len(up_all[uid]), "outlet": [outlet[uid][0], outlet[uid][1]], "outlet_how": outlet[uid][2],
            "label": [round(lp.x, 4), round(lp.y, 4)], "source": u["source"],
        }
        if u.get("within"):
            props["within"] = u["within"]
        if u.get("merged_names"):
            props["merged_names"] = u["merged_names"]
        if u["kind"] in ("wb",):
            props["chj_codes"] = u["members"]
        feats.append({"type": "Feature", "properties": props, "geometry": mapping(g)})
        topo[uid] = {
            "idx": k, "name": u["name"], "area_km2": props["area_km2"], "next": nxt.get(uid),
            "downstream_chain": chain, "terminal": tk,
            "upstream_direct": sorted(ups[uid], key=lambda x: idx_of[x]),
            "upstream_all": sorted(up_all[uid], key=lambda x: idx_of[x]),
            "up_area_km2": props["up_area_km2"],
        }
    n = write_geojson(OUT / "basins.geojson", feats, meta={
        "name": "Riuà basin units",
        "attribution": "© Confederación Hidrográfica del Júcar (PHJ 2022-2027); CEDEX/DGA-MITECO subcuencas 1:25.000; "
                       "HydroBASINS/HydroRIVERS © WWF/HydroSHEDS; names outside the Júcar district © OpenStreetMap contributors (ODbL)",
    })
    print(f"basins.geojson {n / 1e3:.0f} KB")
    n = write_json(OUT / "basins_topology.json", {
        "note": "next = downstream unit id (null = sea / closed basin / leaves the domain, see 'terminal'). "
                "upstream_all is transitive. up_area_km2 includes the unit itself. Order of 'order' = idx used in terrain.npz.",
        "order": final_ids, "units": topo})
    print(f"basins_topology.json {n / 1e3:.0f} KB")
    np.savez_compressed(RAW / "basin_raster.npz", lab=lab.astype(np.int16), ids=np.array(final_ids), ext=np.array(EXT), res=RES)


if __name__ == "__main__":
    main()
