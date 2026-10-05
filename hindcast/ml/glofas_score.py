"""Score GloFAS v4 (Open-Meteo Flood API archive, daily mean discharge of the ~5-km cell at the gauge) against the SAIH
event peaks of the ravine domain.

    py -3.11 hindcast/ml/glofas_score.py        # needs cache/glofas/*.json (fetch_openmeteo.py glofas)

Per event: GloFAS rise = largest daily discharge inside the response window minus the value of the day before the rain.
Output: out/glofas_score.json and a `glofas_rise` column that evaluate.py can use as a feature.
"""
import numpy as np
from scipy.stats import spearmanr

from common import CACHE, OUT, auc, counts, read_json, write_json


def load():
    rows = [r for r in read_json(OUT / "dataset.json") if r["clean"] and r["domain"] and r["usable"]]
    t_all = np.load(CACHE / "sim_series.npz")["t"]
    out = []
    for r in rows:
        f = CACHE / "glofas" / f"{r['pid']}_{r['t0']}.json"
        if not f.exists():
            continue
        js = read_json(f)
        d = np.array(js["daily"]["time"], "datetime64[D]")
        q = np.array([np.nan if v is None else v for v in js["daily"]["river_discharge"]], float)
        d0 = np.datetime64(r["t0"][:10]); dw = t_all[r["iw"]].astype("datetime64[D]")
        inside = (d >= d0) & (d <= dw + 1)
        if not inside.any() or not np.isfinite(q[inside]).any():
            continue
        base = q[d < d0][-1] if (d < d0).any() else q[0]
        out.append(dict(r, glofas_max=float(np.nanmax(q[inside])), glofas_base=float(base),
                        glofas_rise=float(max(np.nanmax(q[inside]) - base, 0.0)), glofas_lat=js["latitude"], glofas_lon=js["longitude"]))
    return out


def main():
    rows = load()
    g = lambda k: np.array([r[k] for r in rows], float)       # noqa: E731
    obs, c, thr2, rise, gmax, A = g("q_obs"), g("c"), g("thr2"), g("glofas_rise"), g("glofas_max"), g("area")
    sim = g("sim_dep")
    res = dict(n=len(rows), n_catchments=len({r["pid"] for r in rows}))
    for name, y, thr in (("runs", obs >= c, c), ("level2", obs >= thr2, thr2)):
        d = dict(n_events=int(y.sum()),
                 auc_glofas_rise_over_threshold=round(auc(rise / thr, y), 3), auc_glofas_max_over_threshold=round(auc(gmax / thr, y), 3),
                 auc_conceptual=round(auc(sim / thr, y), 3), auc_event_rain=round(auc(g("rain"), y), 3))
        # GloFAS as its own alarm: rise >= k x threshold, k scanned; report the k that reaches POD >= 0.5 and 0.8
        for target in (0.5, 0.8):
            for k in np.sort(np.unique(rise / thr))[::-1]:
                cc = counts(rise / thr >= k, y)
                if cc["pod"] >= target:
                    break
            d[f"at_pod_{target}"] = dict(k=round(float(k), 4), **{a: (round(b, 2) if isinstance(b, float) else b) for a, b in cc.items()})
        d["plain_rise_ge_threshold"] = counts(rise >= thr, y)
        res[name] = d
    resp = obs >= c
    res["spearman_all"] = round(float(spearmanr(rise / A ** 0.75, obs / A ** 0.75)[0]), 3)
    res["spearman_responses"] = round(float(spearmanr(rise[resp] / A[resp] ** 0.75, obs[resp] / A[resp] ** 0.75)[0]), 3)
    res["spearman_conceptual_all"] = round(float(spearmanr(sim / A ** 0.75, obs / A ** 0.75)[0]), 3)
    res["spearman_conceptual_responses"] = round(float(spearmanr(sim[resp] / A[resp] ** 0.75, obs[resp] / A[resp] ** 0.75)[0]), 3)
    res["median_ratio_glofas_rise_over_observed_peak_responses"] = round(float(np.median(rise[resp] / obs[resp])), 3)
    per = {}
    for pid in sorted({r["pid"] for r in rows}):
        m = np.array([r["pid"] == pid for r in rows])
        if m.sum() >= 5 and obs[m].std() > 0 and rise[m].std() > 0:
            per[pid] = dict(name=rows[int(np.nonzero(m)[0][0])]["name"], area=float(A[m][0]), n=int(m.sum()),
                            spearman=round(float(spearmanr(rise[m], obs[m])[0]), 2), max_obs=float(obs[m].max()), max_glofas=float(gmax[m].max()))
    res["per_catchment"] = per
    res["biggest_observed"] = [dict(pid=r["pid"], name=r["name"], t0=r["t0"], area=r["area"], rain=r["rain"], observed=r["q_obs"],
                                    conceptual=r["sim_dep"], glofas_max=round(r["glofas_max"], 1), glofas_base=round(r["glofas_base"], 1))
                               for r in sorted(rows, key=lambda r: -r["q_obs"] / r["c"])[:15]]
    write_json(OUT / "glofas_score.json", res, indent=1)
    write_json(OUT / "glofas_rows.json", {f"{r['pid']}|{r['t0']}": [r["glofas_max"], r["glofas_base"]] for r in rows})
    for k, v in res.items():
        if k not in ("per_catchment", "biggest_observed"):
            print(k, v)
    print("per catchment Spearman (>= 5 events):", {v["name"][:14]: v["spearman"] for v in per.values()})
    for e in res["biggest_observed"]:
        print("  ", e)


if __name__ == "__main__":
    main()
