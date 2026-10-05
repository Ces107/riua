"""Check the reservoir inflow model against what the reservoirs themselves measured, and fit the quick-runoff
share alpha of each dam catchment.

For every episode with a rain analysis (hindcast/truth/*.npz) and every SAIH Júcar reservoir:
  measured  inflow(t) = dV/dt + outflow (hourly, V from the level through the fitted curve where SAIH clips the volume),
            minus what the dams directly above let down the river (their measured outflow, lagged), minus the base
            inflow of the 6 h before the rain;  event volume = its integral from the first rain to 72 h after the last
  simulated own-catchment inflow with the production losses (core/reservoirs.py::inflow), split in the two components
            of the loss model so that any alpha is a linear mix:  q = (1 - alpha) q_main + alpha q_quick

  fit       alpha per dam minimising sum over episodes of ln((V_sim + c) / (V_obs + c))^2, c = 1 mm over the catchment,
            shrunk towards the regional reference (0.03) with the weight of two episodes
Outputs: geo/hydro/dams/out/fit.json, geo/hydro/dams/out/events.json, scratch/q10-dams/events.txt

    py -3.11 geo/hydro/dams/check_events.py           # table + fit
    py -3.11 geo/hydro/dams/check_events.py forata    # the 29 Oct 2024 filling of Forata, hour by hour (after the fit)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from riua import params as P                                # noqa: E402
from riua.core import hydro as H, reservoirs as R           # noqa: E402

CACHE = ROOT / "scratch" / "q10-dams" / "series"
TRUTH = ROOT / "hindcast" / "truth"
TAIL_H = 72
ALPHAS = [0.003, 0.005, 0.008, 0.012, 0.016, 0.022, 0.03, 0.04, 0.055, 0.075, 0.10, 0.13, 0.17, 0.22, 0.28, 0.35]
ALPHA_REF, SHRINK = 0.03, 2.0
HM3 = R.HM3_PER_M3S_H


def series(var):
    f = CACHE / f"chj_{var}.npz"
    if not var or not f.exists():
        return None
    z = np.load(f)
    return z["t"], z["v"].astype(float)


def hourly_last(t, v, hours):
    """value at each hour end (last sample inside the hour), NaN where the hour has no sample"""
    hh = (t + np.timedelta64(59, "m")).astype("datetime64[h]")
    out = np.full(len(hours), np.nan)
    pos = np.searchsorted(hours, hh)
    ok = (pos < len(hours)) & (hours[np.minimum(pos, len(hours) - 1)] == hh)
    out[pos[ok]] = v[ok]            # samples are time-ordered: the last one of the hour wins
    return out


def hourly_mean(t, v, hours):
    hh = (t + np.timedelta64(59, "m")).astype("datetime64[h]")
    pos = np.searchsorted(hours, hh)
    ok = (pos < len(hours)) & (hours[np.minimum(pos, len(hours) - 1)] == hh)
    s = np.bincount(pos[ok], weights=v[ok], minlength=len(hours))
    n = np.bincount(pos[ok], minlength=len(hours))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def measured(d, hours):
    """hourly volume (hm3, NaN gaps) and total outflow (m3/s) of dam d on the hour axis"""
    lv, vv, ov = series(d["vars"].get("level")), series(d["vars"].get("volume")), series(d["vars"].get("outflow"))
    if vv is None:
        return None
    v = hourly_last(vv[0], np.where(vv[1] > 0, vv[1], np.nan), hours)
    c = d.get("curve")
    if lv is not None and c and c.get("h0"):
        h = hourly_last(lv[0], np.where(lv[1] > 0, lv[1], np.nan), hours)
        vc = c["a"] * np.maximum(h - c["h0"], 0.0) ** c["b"]
        clip = np.nanmax(v) if np.isfinite(v).any() else np.inf
        v = np.where(np.isfinite(h) & (v >= 0.999 * clip) & (vc > v), vc, v)      # above the clipped table: the curve
        v = np.where(np.isfinite(v), v, vc)
    q = hourly_mean(ov[0], np.maximum(ov[1], 0.0), hours) if ov is not None else np.zeros(len(hours))
    rv = series(d["vars"].get("river"))
    r = hourly_mean(rv[0], np.maximum(rv[1], 0.0), hours) if rv is not None else np.full(len(hours), np.nan)
    return v, np.nan_to_num(q), np.where(np.isfinite(r), r, np.nan_to_num(q))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    params = P.load()
    hp = params["hydro"]
    dn = R.load_dams()
    D = dn.n
    n_cell = dn.net.lags[0].shape[1]
    a0, a1 = np.zeros(n_cell, np.float32), np.ones(n_cell, np.float32)
    rows = []
    for f in sorted(TRUTH.glob("*.npz")):
        z = np.load(f)
        t = z["t_end"]
        p = np.nan_to_num(z["o_mean"]).reshape(len(t), -1).astype(np.float32)
        hours = np.concatenate([t, t[-1] + np.arange(1, TAIL_H + 1) * np.timedelta64(1, "h")])
        pp = np.concatenate([p, np.zeros((TAIL_H, p.shape[1]), np.float32)])
        dn.alpha = a0; q_main = R.inflow(pp, dn, hp)
        dn.alpha = a1; q_quick = R.inflow(pp, dn, hp)
        ext = np.concatenate([t[0] - np.arange(6, 0, -1) * np.timedelta64(1, "h"), hours])     # 6 h before, for the base flow
        meas = [measured(m, ext) if m["source"] == "saih_chj" else None for m in dn.meta]
        for k, m in enumerate(dn.meta):
            if meas[k] is None:
                continue
            v, qo, _ = meas[k]
            if np.isfinite(v).sum() < 0.7 * len(v):
                continue
            idx = np.arange(len(v))
            v = np.interp(idx, idx[np.isfinite(v)], v[np.isfinite(v)])
            qin = np.diff(v) / HM3 + qo[1:]                       # (len(ext) - 1,) hourly inflow, aligned with ext[1:]
            for u, lag in dn.up[k]:
                if meas[u] is not None:
                    r = meas[u][2]
                    qin = qin - np.concatenate([np.full(lag, r[0]), r[:-lag]])[1:]
            base = float(np.median(qin[:5]))
            ev = qin[5:] - base                                   # aligned with `hours`
            rain = float((dn.net.lags[0].getrow(k) * 0).sum())     # placeholder, replaced below
            A = np.zeros(n_cell)
            for L in dn.net.lags:
                A += np.asarray(L.getrow(k).todense()).ravel()
            rain = float((p.sum(axis=0) * A).sum() / max(A.sum(), 1e-9))
            rows.append(dict(case=f.stem, dam=m["id"], k=k, rain_mm=round(rain, 1), own_km2=m["own_km2"],
                             v_obs=round(float(ev.sum() * HM3), 3), q_obs=round(float(np.max(ev)), 1),
                             v_main=float(q_main[:, k].sum() * HM3), v_quick=float(q_quick[:, k].sum() * HM3),
                             qp_main=float(q_main[:, k].max()), qp_quick=float(q_quick[:, k].max()),
                             dv=round(float(np.nanmax(v) - v[5]), 3), base=round(base, 2)))
        print(f.stem, "done", flush=True)
    # ---- fit alpha per dam
    fit = {}
    for k, m in enumerate(dn.meta):
        rr = [r for r in rows if r["k"] == k and r["rain_mm"] >= 15.0]
        dry = [abs(r["v_obs"]) for r in rows if r["k"] == k and r["rain_mm"] < 3.0]
        c = 0.001 * m["own_km2"]
        noise = float(np.median(dry)) if dry else 0.0
        if not rr or noise > c:
            # the measured balance moves by more than 1 mm of the catchment on dry days (transfers, turbines, sensor
            # noise of a large lake): its event volumes cannot identify alpha
            fit[m["id"]] = dict(alpha=None, episodes=len(rr), dry_noise_hm3=round(noise, 3), note="not fitted: measured balance too noisy")
            continue
        vo = np.maximum(np.array([r["v_obs"] for r in rr]), 0.0)
        vm, vq = np.array([r["v_main"] for r in rr]), np.array([r["v_quick"] for r in rr])
        cost = [float(np.mean(np.log(((1 - a) * vm + a * vq + c) / (vo + c)) ** 2)) for a in ALPHAS]
        best = ALPHAS[int(np.argmin(cost))]
        n_inf = int((vo > c).sum())
        alpha = float(np.exp((n_inf * np.log(best) + SHRINK * np.log(ALPHA_REF)) / (n_inf + SHRINK)))
        e_def = float(np.sqrt(np.mean(np.log(((1 - 0.016) * vm + 0.016 * vq + c) / (vo + c)) ** 2)))
        e_fit = float(np.sqrt(np.mean(np.log(((1 - alpha) * vm + alpha * vq + c) / (vo + c)) ** 2)))
        fit[m["id"]] = dict(alpha=round(alpha, 4), alpha_unshrunk=best, episodes=len(rr), informative=n_inf, dry_noise_hm3=round(noise, 3),
                            rms_ln_default=round(e_def, 3), rms_ln_fit=round(e_fit, 3))
    (HERE / "out" / "fit.json").write_text(json.dumps(fit, indent=1), encoding="utf-8")
    for r in rows:
        a = fit.get(r["dam"], {}).get("alpha") or 0.016
        r["v_sim"] = round((1 - a) * r["v_main"] + a * r["v_quick"], 3)
        r["q_sim"] = round((1 - a) * r["qp_main"] + a * r["qp_quick"], 1)
        r["v_sim_default"] = round((1 - 0.016) * r["v_main"] + 0.016 * r["v_quick"], 3)
        for key in ("v_main", "v_quick", "qp_main", "qp_quick", "k"):
            r.pop(key)
    (HERE / "out" / "events.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    lines = [f"{'dam':15s} alpha  (raw)  n inf  rms ln default -> fit"]
    lines += [f"{i:15s} {v['alpha']:.4f} {v['alpha_unshrunk']:.3f} {v['episodes']:3d} {v['informative']:3d}   {v['rms_ln_default']:.3f} -> {v['rms_ln_fit']:.3f}"
              if v.get("alpha") else f"{i:15s} not fitted (dry-day noise {v['dry_noise_hm3']} hm3, {v['episodes']} episodes)" for i, v in fit.items()]
    lines.append("")
    lines.append(f"{'case':13s} {'dam':15s} {'rain':>6s} {'V_obs':>8s} {'V_sim':>8s} {'V_def':>8s} {'Qp_obs':>7s} {'Qp_sim':>7s} {'dV':>7s}")
    for r in sorted(rows, key=lambda r: -r["v_obs"]):
        if r["rain_mm"] >= 20 or r["v_obs"] > 0.5:
            lines.append(f"{r['case']:13s} {r['dam']:15s} {r['rain_mm']:6.1f} {r['v_obs']:8.2f} {r['v_sim']:8.2f} {r['v_sim_default']:8.2f} "
                         f"{r['q_obs']:7.0f} {r['q_sim']:7.0f} {r['dv']:7.2f}")
    txt = "\n".join(lines)
    (ROOT / "scratch" / "q10-dams" / "events.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt[:6000])


def forata(dam="forata", case="2024-10-dana", start="2024-10-29T00"):
    """One dam through one episode with the full storage model, against its measured level and volume."""
    sys.stdout.reconfigure(encoding="utf-8")
    params = P.load()
    hp = params["hydro"]
    dn = R.load_dams()
    k = dn.ids.index(dam)
    z = np.load(TRUTH / f"{case}.npz")
    t = z["t_end"]
    p = np.nan_to_num(z["o_mean"]).reshape(len(t), -1).astype(np.float32)
    q = R.inflow(p, dn, hp)                                         # (T, D)
    i0 = int(np.searchsorted(t, np.datetime64(start, "h")))
    meas = [measured(m, t) if m["source"] == "saih_chj" else None for m in dn.meta]
    v0 = np.array([m[0][i0] if m is not None and np.isfinite(m[0][i0]) else 0.0 for m in meas])
    rel = np.array([m[1][i0] if m is not None else 0.0 for m in meas])
    sig = hp["sigma"]["now"]
    from scipy.special import ndtri
    mult = np.exp(sig * ndtri((np.arange(7) + 0.5) / 7))
    qs = q[i0 + 1:][None] * mult[:, None, None]
    v, qi, qo, qsp = R.integrate(qs.astype(np.float32), v0, rel, rel, dn)
    vm = meas[k][0]
    A = np.zeros(p.shape[1])
    for L in dn.net.lags:
        A += np.asarray(L.getrow(k).todense()).ravel()
    rain = (p * A).sum(axis=1) / A.sum()
    print(f"{dam} {case}: catchment {dn.meta[k]['own_km2']} km2, alpha {dn.meta[k].get('alpha')}, V_spill {dn.v_spill[k]} hm3, "
          f"rain {rain.sum():.0f} mm = {rain.sum() * A.sum() / 1000:.0f} hm3, simulated inflow {q[:, k].sum() * HM3:.1f} hm3 "
          f"(runoff coefficient {q[:, k].sum() * HM3 / (rain.sum() * A.sum() / 1000):.2f}), peak {q[:, k].max():.0f} m3/s")
    print("hour(UTC)      rain  Qin_sim | V_meas  level_meas | V_sim x0.47 x1.0 x2.1 | spill x1.0 x2.1")
    lv = series(dn.meta[k]["vars"]["level"])
    hm = hourly_last(lv[0], np.where(lv[1] > 0, lv[1], np.nan), t)
    for j in range(i0 + 1, min(i0 + 49, len(t))):
        s = j - i0 - 1
        print(f"{str(t[j])[5:13]}  {rain[j]:6.1f} {q[j, k]:8.0f} | {vm[j]:6.2f}  {hm[j]:8.2f}   | {v[0, s, k]:6.2f} {v[3, s, k]:6.2f} {v[6, s, k]:6.2f} | "
              f"{qsp[3, s, k]:6.0f} {qsp[6, s, k]:6.0f}")
    for name, kk in (("x0.47", 0), ("x1.0", 3), ("x2.1", 6)):
        full = np.nonzero(v[kk, :, k] >= dn.v_spill[k] * 0.9995)[0]
        print(f"multiplier {name}: reaches the spill volume {'at ' + str(t[i0 + 1 + full[0]]) if len(full) else 'never'}; "
              f"peak inflow {qi[kk, :, k].max():.0f} m3/s, peak spill {qsp[kk, :, k].max():.0f} m3/s, max volume {v[kk, :, k].max():.1f} hm3")
    mfull = np.nonzero(vm[i0:] >= dn.v_spill[k] * 0.9995)[0]
    print(f"measured: V {vm[i0]:.2f} hm3 at {t[i0]}, reaches {dn.v_spill[k]} hm3 at {t[i0 + mfull[0]] if len(mfull) else 'never'}, "
          f"max level {np.nanmax(hm[i0:]):.2f} m, max hourly gain {np.nanmax(np.diff(vm[i0:i0 + 30])):.2f} hm3/h "
          f"= {np.nanmax(np.diff(vm[i0:i0 + 30])) / HM3:.0f} m3/s net")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        forata(*sys.argv[1:])
    else:
        main()
