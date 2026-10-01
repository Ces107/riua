"""Load every output back with json + numpy ONLY (as the server will), print sanity tables, draw overview PNGs.

PNG outputs: scratch/r5-geo/overview.png, overview_poyo.png, overview_grid.png
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT, RAW = ROOT / "geo" / "out", ROOT / "scratch" / "r5-geo"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def rings(geom):
    if geom["type"] == "Polygon":
        yield from geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield from poly
    elif geom["type"] == "LineString":
        yield geom["coordinates"]
    elif geom["type"] == "MultiLineString":
        yield from geom["coordinates"]


def main():
    print("=== files")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e3:8.1f} KB")
    zones, thr, basins, topo = load("zones.geojson"), load("thresholds.json"), load("basins.geojson"), load("basins_topology.json")
    rivers, places, boundary = load("rivers.geojson"), load("places.json"), load("boundary.geojson")
    T = np.load(OUT / "terrain.npz")  # no allow_pickle needed

    print("\n=== zones + thresholds  (source:", thr["source"]["document"], "v" + thr["source"]["version"], thr["source"]["document_date"] + ")")
    print(f"  {'code':7s} {'name':30s} {'prov':10s} {'km2':>7s}  1h y/o/r    12h y/o/r")
    for f in zones["features"]:
        p = f["properties"]
        t = thr["zones"][p["code"]]
        a, b = t["precip_1h_mm"], t["precip_12h_mm"]
        print(f"  {p['code']:7s} {p['name']:30s} {p['province']:10s} {p['area_km2']:7.0f}  {a['yellow']}/{a['orange']}/{a['red']}   {b['yellow']}/{b['orange']}/{b['red']}")
    assert len(zones["features"]) == 11 and set(thr["zones"]) == {f["properties"]["code"] for f in zones["features"]}

    print("\n=== boundary")
    for f in boundary["features"]:
        print("  ", f["properties"], f["geometry"]["type"])

    print("\n=== basins")
    P = {f["properties"]["id"]: f["properties"] for f in basins["features"]}
    U = topo["units"]
    order = topo["order"]
    assert list(P) == order == [str(x) for x in T["basin_ids"]]
    areas = np.array([P[u]["area_km2"] for u in order])
    print(f"  units {len(order)}; km2 min {areas.min():.1f} p10 {np.percentile(areas, 10):.0f} median {np.median(areas):.0f} "
          f"p90 {np.percentile(areas, 90):.0f} max {areas.max():.0f}; total {areas.sum():.0f} km2")
    print("  kind:", dict(Counter(p["kind"] for p in P.values())))
    print("  terminal:", dict(Counter(p["terminal"] for p in P.values())), " outlet_how:", dict(Counter(p["outlet_how"] for p in P.values())))
    # topology consistency
    for u in order:
        n = U[u]["next"]
        assert n is None or n in U, (u, n)
        seen, v = {u}, n
        while v is not None:
            assert v not in seen, f"cycle {u}"
            seen.add(v)
            v = U[v]["next"]
        assert U[u]["downstream_chain"] == [x for x in _chain(U, u)]
        ups = {x for x in order if u in U[x]["downstream_chain"]}
        assert ups == set(U[u]["upstream_all"]), u
        tot = P[u]["area_km2"] + sum(P[x]["area_km2"] for x in ups)
        assert abs(tot - U[u]["up_area_km2"]) < 0.6 + 0.001 * tot, (u, tot, U[u]["up_area_km2"])
    print("  topology consistent: next ids exist, no cycles, upstream_all == inverse of downstream chains, cumulative areas add up")
    names = [P[u]["name"] for u in order]
    print("  unique names:", len(set(names)) == len(names), "| units without town:", sum(P[u]["town"] is None for u in order))

    PL = {p["name"]: p for p in places["places"]}
    print("\n=== Rambla del Poyo check (town -> unit -> downstream chain)")
    for town in ("Chiva", "Cheste", "Buñol", "Godelleta", "Torrent", "Aldaia", "Alaquàs", "Picanya", "Paiporta", "Massanassa", "Catarroja", "Albal"):
        p = PL[town]
        u = p["basin"]
        chain = " -> ".join([P[u]["name"]] + [P[x]["name"] for x in U[u]["downstream_chain"]])
        print(f"  {town:10s} [{u}] {chain} -> {P[u]['terminal']}")
    poyo = [u for u in order if (P[u]["river"] or "").startswith("Rambla Poyo")]
    print("  Poyo units:")
    for u in sorted(poyo, key=lambda u: P[u]["up_area_km2"]):
        print(f"    {u:8s} {P[u]['name']:60s} area {P[u]['area_km2']:6.1f} up {P[u]['up_area_km2']:6.1f} next {P[u]['next']} outlet {P[u]['outlet']}")

    print("\n=== key rivers / ravines (units whose river name matches; upstream -> downstream)")
    for q in ("Carraixet", "Magro", "Rambla de la Viuda", "Río Sec", "Girona", "Gorgos", "Serpis", "Vinalopó", "Segura",
              "Servol", "Sénia", "Júcar", "Turia", "Mijares", "Palancia", "Cabriel", "Albaida", "Cànyoles", "Montnegre", "Algar"):
        us = sorted([u for u in order if q.lower() in (P[u]["river"] or "").lower()], key=lambda u: P[u]["up_area_km2"])
        if not us:
            print(f"  {q}: NONE")
            continue
        last = us[-1]
        print(f"  {q}: {len(us)} units; most downstream [{last}] '{P[last]['name']}' up_area {P[last]['up_area_km2']:.0f} km2 -> {P[last]['terminal']}")

    print("\n=== places")
    pl = places["places"]
    print("  ", places["meta"]["n_cv"], "CV +", places["meta"]["n_upstream"], "upstream;",
          "CV population", sum(p["pop"] for p in pl if p["in_cv"]))
    print("   per province:", dict(Counter(p["province"] for p in pl if p["in_cv"])))
    print("   per zone:", dict(sorted(Counter(p["zone"] for p in pl if p["in_cv"]).items())))
    assert all(p["basin"] in U for p in pl) and all(p["zone"] in thr["zones"] for p in pl if p["in_cv"])
    print("   biggest upstream towns:", [(p["name"], p["pop"], P[p["basin"]]["name"]) for p in sorted((p for p in pl if not p["in_cv"]), key=lambda p: -p["pop"])[:8]])

    print("\n=== rivers:", len(rivers["features"]), "lines;", dict(Counter(f["properties"]["source"].split(" ")[0] for f in rivers["features"])))

    print("\n=== terrain.npz")
    for k in T.files:
        a = T[k]
        extra = f" min {np.nanmin(a):.2f} max {np.nanmax(a):.2f}" if a.dtype.kind in "fi" and a.size else ""
        print(f"  {k:18s} {str(a.dtype):8s} {str(a.shape):12s}{extra}")
    NY, NX = T["elev_mean"].shape
    assert (NY, NX) == (68, 64) and abs(T["lon"][0] + 2.375) < 1e-9 and abs(T["lat"][0] - 37.625) < 1e-9
    ptr, cell, w = T["basin_w_ptr"], T["basin_w_cell"], T["basin_w"]
    sums = np.array([w[ptr[k]:ptr[k + 1]].sum() for k in range(len(order))])
    has = np.diff(ptr) > 0
    assert np.allclose(sums[has], 1.0, atol=1e-4)
    cell_km2 = (0.05 * 111.195) ** 2 * np.cos(np.radians(T["lat"]))[:, None] * np.ones((1, NX))
    k = order.index("J16-01")
    est = (T["basin_w_cellfrac"][ptr[k]:ptr[k + 1]] * cell_km2.ravel()[cell[ptr[k]:ptr[k + 1]]]).sum()
    print(f"  weights: {has.sum()} units have cells, weights sum to 1; J16-01 area from cell fractions {est:.1f} km2 vs {P['J16-01']['area_km2']} km2")
    print(f"  in_domain cells {int(T['in_domain'].sum())}, land cells {int((T['land_frac'] > 0).sum())}, CV cells (cv_frac>0.5) {int((T['cv_frac'] > 0.5).sum())}")
    # demo: basin-mean of a field using the weights (here: elevation)
    j, i = np.unravel_index(cell[ptr[k]:ptr[k + 1]], (NY, NX))
    print(f"  demo basin-mean elevation of J16-01 = {(T['elev_mean'][j, i] * w[ptr[k]:ptr[k + 1]]).sum():.0f} m")

    # ------------------------------------------------------------------ PNGs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    def draw(ax, extent, labels=False):
        rng = np.random.default_rng(3)
        cols = rng.uniform(0.35, 0.95, (len(order), 3))
        polys, fc = [], []
        for k, f in enumerate(basins["features"]):
            for ring in ([poly[0] for poly in f["geometry"]["coordinates"]] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"][0]]):
                polys.append(np.asarray(ring))
                fc.append(cols[k])
        ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors="k", linewidths=0.25))
        ax.add_collection(LineCollection([np.asarray(r) for f in rivers["features"] for r in rings(f["geometry"])], colors="#0b3d91", linewidths=0.6))
        ax.add_collection(LineCollection([np.asarray(r) for f in zones["features"] for r in rings(f["geometry"])], colors="red", linewidths=1.3))
        for f in basins["features"]:
            p = f["properties"]
            if p["next"]:
                a, b = p["label"], P[p["next"]]["label"]
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="k", lw=0.5, alpha=0.6))
            ax.plot(*p["outlet"], "k.", ms=2)
            if labels and extent[0] < p["label"][0] < extent[1] and extent[2] < p["label"][1] < extent[3]:
                ax.text(*p["label"], f"{p['id']}\n{p['name'][:38]}", fontsize=5, ha="center")
        for p in pl:
            if extent[0] < p["lon"] < extent[1] and extent[2] < p["lat"] < extent[3] and (labels and p["pop"] > 8000 or p["pop"] > 60000):
                ax.plot(p["lon"], p["lat"], "o", color="yellow", mec="k", ms=3)
                ax.text(p["lon"], p["lat"], p["names"][0], fontsize=6, color="darkred")
        ax.plot([-2.4, 0.8, 0.8, -2.4, -2.4], [37.6, 37.6, 41.0, 41.0, 37.6], "m--", lw=1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(1 / np.cos(np.radians(39.3)))

    fig, ax = plt.subplots(figsize=(15, 17))
    draw(ax, (-3.1, 1.0, 37.4, 41.2))
    ax.set_title(f"Riuà geodata: {len(order)} basin units, rivers, AEMET zones (red), box (magenta)")
    fig.savefig(RAW / "overview.png", dpi=110, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(16, 12))
    draw(ax, (-1.05, -0.2, 39.1, 39.75), labels=True)
    ax.set_title("Zoom: Poyo / Turia / Magro / Albufera")
    fig.savefig(RAW / "overview_poyo.png", dpi=120, bbox_inches="tight")

    fig, axs = plt.subplots(2, 3, figsize=(18, 13))
    ext = (-2.4, 0.8, 37.6, 41.0)
    land = T["land_frac"] > 0
    def show(ax, a, title, **kw):
        im = ax.imshow(np.where(land, a, np.nan), origin="lower", extent=ext, **kw)
        ax.contour(T["lon"], T["lat"], T["in_domain"].astype(float), levels=[0.5], colors="k", linewidths=0.8)
        ax.set_title(title)
        ax.set_aspect(1 / np.cos(np.radians(39.3)))
        plt.colorbar(im, ax=ax, shrink=0.7)
    show(axs[0, 0], T["elev_mean"], "elev_mean (m) + in_domain outline", cmap="terrain")
    show(axs[0, 1], T["elev_std"], "elev_std (m)", cmap="magma")
    show(axs[0, 2], T["slope_mean_deg"], "slope_mean_deg", cmap="viridis")
    show(axs[1, 0], T["elev_smooth"], "elev_smooth + gradient (dzdx, dzdy)", cmap="terrain")
    axs[1, 0].quiver(T["lon"][::2], T["lat"][::2], T["dzdx"][::2, ::2], T["dzdy"][::2, ::2], scale=900, width=0.002)
    show(axs[1, 1], np.where(T["zone_idx"] >= 0, T["zone_idx"], np.nan), "zone_idx", cmap="tab20")
    show(axs[1, 2], np.where(T["basin_idx"] >= 0, (T["basin_idx"] * 37) % 101, np.nan), "basin_idx (hashed colours)", cmap="nipy_spectral")
    fig.savefig(RAW / "overview_grid.png", dpi=90, bbox_inches="tight")
    print("\nPNG written: scratch/r5-geo/overview.png, overview_poyo.png, overview_grid.png")


def _chain(U, u):
    v = U[u]["next"]
    while v is not None:
        yield v
        v = U[v]["next"]


if __name__ == "__main__":
    main()
