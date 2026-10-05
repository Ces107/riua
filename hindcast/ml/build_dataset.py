"""One row per (gauged catchment, rain event): what was measured, what the conceptual model said, and the features a light
statistical model may use.  Floods AND non-floods.

    py -3.11 hindcast/ml/build_dataset.py            # ~3 min (reads hindcast/obs/flows, cached by calibrate_hydro.py)

Inputs (all read-only): geo/hydro/gauges (network, attributes), hindcast/obs/flows/rain_grid.npz (hourly rain: SAIH
pluviometers + the truth analysis inside episodes), the SAIH flow series, geo/hydro/loss_params.json via
hindcast/obs/flows/alpha_cells.npy (the per-cell quick share as deployed), hindcast/hydro_fit.json (alpha regression).
Outputs: hindcast/ml/out/dataset.csv (+ dataset.json), hindcast/ml/cache/gauge_qT.json.

Three simulated peaks per row:
  sim_dep   as deployed: per-cell alpha mosaic (a gauged stream carries the alpha fitted to its own floods);
  sim_reg   as an UNGAUGED point would run: alpha from the regression on CEDEX P0i and area (no per-gauge fitting);
  q0        the same rain routed with no losses at all (the "potential" peak: depth, intensity, position and routing in
            one number).
Thresholds at the gauges follow core/hydro.py::level_thresholds for a point without channel capacity: levels 2/3/4/5 at
the CAUMAX 2/5/25/100-year floods, with the floors 0.6 and 1.5 x A^0.6; `c` is q7's 'runs' threshold 5 (A/184)^0.75.
"""
import csv
import sys
import time

import numpy as np
from scipy.signal import lfilter

from common import CACHE, FLOWS, GAUGES, OUT, ROOT, c_off, read_json, write_json

import calibrate_hydro as CH                                # noqa: E402

TAU_H = 72.0
DUR = (1, 3, 6, 12, 24)
RP_LEVELS = (2, 5, 25, 100)
MIN_Q2, MIN_Q3 = 0.6, 1.5


def caumax_qT(points):
    """CAUMAX flood quantiles T2..T500 at the gauges; same matching rule as geo/hydro/gauges/build_attributes.py::caumax_q2
    (which only kept T2).  Cached."""
    f = CACHE / "gauge_qT.json"
    if f.exists():
        return read_json(f)
    import pyflwdir
    import rasterio
    from pyproj import Transformer
    src = ROOT / "scratch" / "h2-sections" / "misc" / "caumax" / "data"
    tr = Transformer.from_crs(4326, 25830, always_xy=True)
    x0, y0 = tr.transform(-4.6, 37.2)
    x1, y1 = tr.transform(1.0, 41.4)
    with rasterio.open(src / "dir.tif") as r:
        win = rasterio.windows.from_bounds(x0, y0, x1, y1, r.transform).round_offsets().round_lengths()
        d8 = r.read(1, window=win); tf = r.window_transform(win); nod = r.nodata
    q = {}
    for T in (2, 5, 10, 25, 100, 500):
        with rasterio.open(src / f"q{T}.tif") as r:
            dc = int(round((tf.c - r.transform.c) / r.transform.a)); dr = int(round((tf.f - r.transform.f) / r.transform.e))
            a = r.read(1, window=rasterio.windows.Window(dc, dr, win.width, win.height), boundless=True, fill_value=r.nodata).astype(np.float64)
            a[(a < 0) | (a > 1e6)] = np.nan
            q[T] = a
    d8u = np.where(d8 == nod, 247, np.where(d8 == 3, 0, d8)).astype(np.uint8)
    area = pyflwdir.from_array(d8u, ftype="d8", transform=tf, latlon=False).upstream_area(unit="km2")
    ny, nx = d8.shape
    ys = tf.f + (np.arange(ny) + 0.5) * tf.e
    xs = tf.c + (np.arange(nx) + 0.5) * tf.a
    out = {}
    for p in points:
        A = float(p["area_km2"]); Au = float(p.get("area_unregulated_km2") or A)
        px, py = tr.transform(p["lon"], p["lat"])
        i0, j0 = int((px - tf.c) / tf.a), int((py - tf.f) / tf.e)
        for rad, limit in ((4, 1.6), (8, 2.5)):
            sl_ = (slice(max(j0 - rad, 0), j0 + rad + 1), slice(max(i0 - rad, 0), i0 + rad + 1))
            jj, ii = np.mgrid[sl_]
            cand = np.isfinite(q[2][sl_]) & (area[sl_] > 0)
            if not cand.any():
                continue
            dist = np.hypot(xs[ii] - px, ys[jj] - py) / 1000.0
            score = np.abs(np.log(area[sl_] / max(A, 1.0))) + dist / 4.0
            score[~cand] = np.inf
            k = np.unravel_index(np.argmin(score), score.shape)
            ratio = area[sl_][k] / max(A, 1.0)
            if np.isfinite(score[k]) and 1 / limit <= ratio <= limit:
                f_ = (A / float(area[sl_][k])) ** 0.75 * (Au / A) ** 0.75
                out[p["id"]] = {f"T{T}": round(float(q[T][sl_][k]) * f_, 2) for T in q}
                break
    write_json(f, out, indent=1)
    return out


def thresholds(area, qT):
    """(c, thr2, thr3, thr4, thr5) for a gauge of unregulated area `area` with CAUMAX quantiles qT (dict or None)"""
    a06 = max(area, 1.0) ** 0.6
    c = float(c_off(area))
    if not qT or not all(np.isfinite(qT.get(f"T{T}", np.nan)) for T in RP_LEVELS):
        env = 100.0 * max(area, 1.0) ** 0.6
        t = [f * env for f in (0.08, 0.18, 0.35, 0.6)]
    else:
        t = [qT[f"T{T}"] for T in RP_LEVELS]
    t[0] = max(t[0], MIN_Q2 * a06); t[1] = max(t[1], MIN_Q3 * a06); t[2] = max(t[2], t[1]); t[3] = max(t[3], t[2])
    return [c] + [float(x) for x in t]


def roll_max(x, n, a, b):
    """largest sum of n consecutive hours of x that END inside [a, b] (hours before index 0 count as dry)"""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    e = np.arange(a, b + 1) + 1
    return float((cs[e] - cs[np.maximum(e - n, 0)]).max())


def main():
    t_start = time.time()
    D, cats, rows, anchors, cnet, cps = CH.collect(offline=True)
    attrs = CH.attributes()
    raw_attr = read_json(GAUGES / "out" / "attributes.json")
    fit = read_json(ROOT / "hindcast" / "hydro_fit.json")
    reg = fit["validation"]["regression"]; reg = dict(names=reg["attributes"], beta=reg["beta"], mean=reg["mean"], sd=reg["sd"])
    qT = caumax_qT(D.pts)
    alpha_cells = np.load(FLOWS / "alpha_cells.npy").astype(float)
    decay = float(np.exp(-1.0 / TAU_H))
    kd = CH.API_K_DAY ** (1.0 / 24.0)
    is_clean = {id(r) for r in CH.clean(rows)}
    ep_all = CH.episodes(rows)
    by = {}
    for k, r in enumerate(rows):
        by.setdefault(r["pid"], []).append(k)
    out = []
    series = {}
    for pid, ks in by.items():
        c = cats[pid]
        n = len(c["cells"]); T = c["p"].shape[0]
        one, K = np.ones(T), max(0.3 * c["tl"], 0.25)
        a_reg = float(np.clip(CH.ALPHA_REF * np.exp(CH.predict_lnf(reg, attrs[pid])), 0.0, 0.5))
        a_dep = alpha_cells[c["cells"]]
        args = (c["lag"], c["loc"], c["a"], K)
        q_dep = CH.sim(c["p"], one, np.full(n, 150.0), np.full(n, 150.0), decay, np.full(n, 45.0), 1, a_dep, 10.0, 100.0, *args)
        q_reg = CH.sim(c["p"], one, np.full(n, 150.0), np.full(n, 150.0), decay, np.full(n, 45.0), 1, np.full(n, a_reg), 10.0, 100.0, *args)
        q0 = CH.sim(c["p"], one, np.zeros(n), np.full(n, 1e-3), decay, np.zeros(n), 0, np.zeros(n), 0.0, 1.0, *args)
        R = np.nan_to_num(c["R"])
        Wb = lfilter([1.0], [1.0, -decay], R)                       # basin rain remembered with the model's 72-h memory
        api = lfilter([1.0], [1.0, -kd], R)                         # antecedent precipitation index, k = 0.98 / day
        known = np.concatenate([[0.0], np.cumsum(np.isfinite(c["R"]))])
        w = c["w"]; wn = w / w.sum()
        # mean travel time (h) from each cell to the gauge, and its catchment moments (Zoccatelli et al. 2011)
        d = np.bincount(c["loc"], c["a"] * c["lag"], minlength=n) / np.maximum(np.bincount(c["loc"], c["a"], minlength=n), 1e-9)
        g1 = float((wn * d).sum()); g2 = float((wn * d * d).sum())
        at = raw_attr[pid]
        thr = thresholds(c["area"], qT.get(pid))
        series[pid] = dict(q_dep=q_dep.astype(np.float32), q_reg=q_reg.astype(np.float32), q0=q0.astype(np.float32))
        for k in ks:
            r = rows[k]
            a, b, iw = r["i0"], r["i1"], r["iw"]
            pe = c["p"][a:b + 1].astype(float)                      # (hours, cells)
            Rt = R[a:b + 1]
            tot_c = pe.sum(axis=0)
            tot = float((tot_c * wn).sum())
            row = dict(pid=pid, name=r["name"], t0=r["t0"], t1=r["t1"], i0=a, i1=b, iw=iw, episode=int(ep_all[k]), src=r["src"],
                       clean=id(r) in is_clean, domain=bool(CH.in_domain(pid, attrs)), usable=bool(CH.usable(r)),
                       regulated=bool(r["regulated"]), rain_doubt=bool(CH.rain_doubt(r)), rain_gauges=r.get("rain_gauges"),
                       area=round(c["area"], 1), tl=round(c["tl"], 1),
                       q_obs=r["q_obs"], base=r["base"], runoff_mm=r["runoff_mm"], rc=r["rc"], lag_i1_h=r.get("lag_i1_h"),
                       c=round(thr[0], 2), thr2=round(thr[1], 1), thr3=round(thr[2], 1), thr4=round(thr[3], 1), thr5=round(thr[4], 1),
                       sim_dep=round(float(q_dep[a:iw + 1].max()), 2), sim_reg=round(float(q_reg[a:iw + 1].max()), 2),
                       q0=round(float(q0[a:iw + 1].max()), 2),
                       alpha_dep=round(float((a_dep * wn).sum()), 4), alpha_reg=round(a_reg, 4),
                       rain=round(tot, 1), dur_h=int(b - a + 1), wb=round(float(Wb[a:b + 1].max()), 1))
            for h in DUR:
                row[f"r{h}"] = round(roll_max(R, h, a, b), 1)
            row["cell1"] = round(float(pe.max()), 1)                               # wettest cell-hour
            row["cell_tot"] = round(float(tot_c.max()), 1)                         # wettest cell, event total
            row["sp_conc"] = round(float(tot_c.max() / max(tot, 0.1)), 2)          # spatial concentration: wettest cell / mean
            row["sp_frac50"] = round(float(wn[tot_c >= 0.5 * tot_c.max()].sum()), 2)   # share of the basin with >= half the max
            row["t_conc6"] = round(row["r6"] / max(tot, 0.1), 2)                   # share of the event that fell in its wettest 6 h
            # where and how the rain moved along the channel (travel-time coordinate d): Zoccatelli et al. 2011
            pw = pe * wn[None, :]
            m0 = pw.sum()
            if m0 > 0 and g1 > 0:
                th1 = float((pw * d[None, :]).sum() / m0)
                th2 = float((pw * d[None, :] ** 2).sum() / m0)
                row["d1"] = round(th1 / g1, 3)                                     # <1: rain near the outlet, >1: in the headwaters
                row["d2"] = round((th2 - th1 ** 2) / max(g2 - g1 ** 2, 1e-9), 3)   # <1: concentrated along the channel
                wt = Rt / max(Rt.mean(), 1e-9)                                     # rain weight of each hour
                wet = Rt > 0.05
                tt = np.arange(len(Rt), dtype=float)
                if wet.sum() >= 3 and np.var(tt) > 0:
                    d1t = np.where(wet, (pw * d[None, :]).sum(axis=1) / np.maximum(pw.sum(axis=1), 1e-12) / g1, 0.0)
                    cov = lambda x, y: float(np.mean(x * y) - np.mean(x) * np.mean(y))        # noqa: E731
                    vs = g1 * (cov(tt, d1t * wt) / np.var(tt) - cov(tt, wt) / np.var(tt) * row["d1"])
                    row["storm_v"] = round(vs, 3)        # travel-hours per hour; negative = the storm moves downstream
                else:
                    row["storm_v"] = 0.0
            else:
                row.update(d1=1.0, d2=1.0, storm_v=0.0)
            hist_h = int(known[a])
            row["hist_days"] = round(hist_h / 24.0, 1)
            row["api"] = round(float(api[a - 1]), 1) if a > 0 else 0.0
            row["rain5d"] = round(float(R[max(a - 120, 0):a].sum()), 1)
            row["rain30d"] = round(float(R[max(a - 720, 0):a].sum()), 1)
            row["rain90d"] = round(float(R[max(a - 2160, 0):a].sum()), 1)
            mth = int(r["t0"][5:7]); doy = (np.datetime64(r["t0"][:10]) - np.datetime64(r["t0"][:4] + "-01-01")).astype(int)
            row["month"] = mth
            row["season_cos"] = round(float(np.cos(2 * np.pi * (doy - 15) / 365.25)), 3)   # +1 mid-January, -1 mid-July
            row["season_sin"] = round(float(np.sin(2 * np.pi * (doy - 15) / 365.25)), 3)   # +1 mid-April, -1 mid-October
            for kk in ("lat", "lon", "elev_mean_m", "slope", "p0i_mm", "perm_rank", "carb_high", "carb_med", "marl_clay", "quaternary",
                       "urban", "crops", "forest", "scrub", "map_mm", "q2_m3s", "q2_spec"):
                row[kk] = at.get(kk)
            out.append(row)
    out.sort(key=lambda r: (r["pid"], r["t0"]))
    keys = list(out[0].keys())
    for r in out:
        for k_ in r:
            if k_ not in keys:
                keys.append(k_)
    with open(OUT / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader(); wr.writerows(out)
    write_json(OUT / "dataset.json", out)
    np.savez_compressed(CACHE / "sim_series.npz", t=D.t, **{f"{pid}|{k}": v for pid, s in series.items() for k, v in s.items()})
    # the 2024 anchors at control points (for the 'do not break 2024' check): peaks with the deployed alpha and with no losses
    anc = []
    n_all = alpha_cells.size
    par = dict(p0=np.full(n_all, 150.0), s=np.full(n_all, 150.0), phi=np.full(n_all, 45.0), alpha=alpha_cells, form=1, tau=TAU_H,
               p0b=10.0, sb=100.0, kf=0.3)
    pk = CH.run_cells(cats, anchors, par)
    for e, p_ in zip(anchors, pk):
        c = cats[e["pid"]]
        R = np.nan_to_num(c["R"])
        anc.append(dict(pid=e["pid"], area=c["area"], tl=c["tl"], official=e["official"], zero_loss=e["zero_loss"], target=round(e["q_obs"]),
                        sim_dep=round(float(p_), 1), rain=e["rain"], i0=e["i0"], i1=e["i1"], iw=e["iw"], t0=e["t0"],
                        **{f"r{h}": round(roll_max(R, h, e["i0"], e["i1"]), 1) for h in DUR}))
    write_json(OUT / "anchors_2024.json", anc, indent=1)
    cl = [r for r in out if r["clean"]]
    dom = [r for r in cl if r["domain"]]
    print(f"{len(out)} rows with flow, {len(cl)} clean, {len(dom)} clean in the ravine domain ({len({r['pid'] for r in dom})} catchments, "
          f"{len({r['episode'] for r in dom})} episodes); {time.time() - t_start:.0f} s")
    for name, key in (("runs (>= c)", "c"), ("level 2", "thr2"), ("level 3", "thr3"), ("level 4", "thr4")):
        o = np.array([r["q_obs"] >= r[key] for r in dom]); s = np.array([r["sim_dep"] >= r[key] for r in dom])
        print(f"  {name:12s}: observed {int(o.sum()):3d} | deployed model hits {int((o & s).sum())} misses {int((o & ~s).sum())} false {int((~o & s).sum())}")


if __name__ == "__main__":
    sys.exit(main())
