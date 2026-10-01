"""Calibrate the rain -> discharge losses against measured flows (SAIH Júcar 5-min series).

    py -3.11 hindcast/calibrate_hydro.py            # fetch flows (cached), simulate, fit, write hindcast/hydro_fit.json

For every control point with a gauge on the same stream and every hindcast case with flow data:
the radar-gauge rain analysis is run through core/hydro.py and the simulated event peak is
compared with the measured one (pre-event flow removed).
"""
import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from riua import params as P, static                      # noqa: E402
from riua.core import hydro as H                          # noqa: E402
from riua.sources import gauges as G                      # noqa: E402

# control point -> SAIH variable id of the flow gauge on the same stream
PAIRS = {"poyo-ribarroja": "13873", "carraixet-betera": "13792", "castellarda-lliria": "13896",
         "magro-requena": "13403", "sellent-carcer": "2701", "albaida-manuel": "2443",
         "canyoles-canals": "13102", "vernissa-real": "13568", "magro-carlet": "14551"}
FLOWS = ROOT / "hindcast" / "obs" / "flows"


def flow_series(var: str, a: datetime, b: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Hourly maximum of the 5-min flow (m3/s): t_end datetime64[h], q."""
    f = FLOWS / f"{var}_{a:%Y%m%d}_{b:%Y%m%d}.json"
    if not f.exists():
        FLOWS.mkdir(parents=True, exist_ok=True)
        out, t = [], a
        while t < b:                      # the endpoint thins long windows: ask two days at a time
            u = min(t + timedelta(days=2), b)
            qa, qb = (requests.utils.quote(x.strftime("%Y-%m-%d %H:%M:%S")) for x in (t, u))
            try:
                out += json.loads(G._get(f"{G.CHJ}/admin/variables/valor/{var}/{qa}/{qb}", "flow.json").decode("utf-8"))
            except Exception as e:
                print("  flow", var, t, type(e).__name__)
            t = u
        f.write_text(json.dumps(out), encoding="utf-8")
    d = [(np.datetime64(p["fecha"][:19]), p["valor"]) for p in json.loads(f.read_text(encoding="utf-8")) if p.get("valor") is not None]
    if not d:
        return np.array([], "datetime64[h]"), np.array([])
    tt = np.array([x[0] for x in d]); v = np.array([x[1] for x in d], float)
    hh = (tt + np.timedelta64(3599, "s")).astype("datetime64[h]")
    u = np.unique(hh)
    return u, np.array([v[hh == x].max() for x in u])


def load_cases():
    cases = []
    for f in sorted(glob.glob(str(ROOT / "hindcast" / "truth" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if len(z["t_end"]) < 12:
            continue
        cases.append((Path(f).stem, z["t_end"], np.nan_to_num(z["o_mean"]).astype(np.float32)))
    return cases


def simulate(net, flat, hp, **kw):
    return H.route(H.net_rain(flat, **kw), net, hp["clark_k"])


def collect():
    net, _ = static.hydro_net()
    rows = []
    for name, t, p in load_cases():
        a = datetime.fromisoformat(str(t[0])) - timedelta(hours=6)
        b = datetime.fromisoformat(str(t[-1])) + timedelta(hours=30)
        flat = p.reshape(len(t), -1)
        for pid, var in PAIRS.items():
            k = net.ids.index(pid)
            tq, q = flow_series(var, a, b)
            if len(q) < 12:
                continue
            base = float(np.median(q[: max(3, len(q) // 10)]))
            catch = sum(A[k].toarray().ravel() for A in net.lags)           # km2 per cell
            rain = float((flat.sum(axis=0) * catch).sum() / max(catch.sum(), 1e-9))
            rmax12 = float(max(((flat[max(0, i - 12):i].sum(axis=0) * catch).sum() / catch.sum()) for i in range(1, len(t) + 1)))
            rows.append(dict(case=name, pid=pid, k=k, area=float(net.area[k]), rain=rain, rain12=rmax12,
                             q_obs=float(q.max() - base), base=base, flat=flat))
    return net, rows


if __name__ == "__main__":
    hp = P.load()["hydro"]
    net, rows = collect()
    flats = {r["case"]: r["flat"] for r in rows}
    k_poyo = net.ids.index("poyo-ribarroja")
    # flows that are not this catchment's runoff: regulated (Bellus dam above Manuel), sensor spikes without rain
    use = [r for r in rows if r["pid"] != "albaida-manuel" and r["rain"] >= 10]
    # 29 Oct 2024, Poyo at the A-3: 2283 m3/s when the sensor was lost, still rising
    use.append(dict(case="2024-10-dana", pid="poyo-ribarroja", k=k_poyo, q_obs=2700.0, rain=0, weight=6.0))
    best = []
    for p0 in (100.0, 120.0, 140.0, 160.0):
        for S in (75.0, 100.0, 150.0, 200.0):
            for phi in (None,):
                for tau in (72.0, 96.0):
                    peak = {c: simulate(net, f, hp, p0=p0, tau_h=tau, phi=phi, s=S).max(axis=0) for c, f in flats.items()}
                    err = [(r.get("weight", 1.0), abs(np.log((peak[r["case"]][r["k"]] + 5.0) / (r["q_obs"] + 5.0)))) for r in use]
                    score = sum(w * e for w, e in err) / sum(w for w, e in err)
                    best.append((round(float(score), 3), p0, S, phi, tau, round(float(peak["2024-10-dana"][k_poyo]))))
    best.sort(key=lambda x: x[0])
    for x in best[:25]:
        print(x)
    score, p0, S, phi, tau, poyo = best[0]
    peak = {c: simulate(net, f, hp, p0=p0, tau_h=tau, phi=phi, s=S).max(axis=0) for c, f in flats.items()}
    old = {c: simulate(net, f, hp, p0=25.0, tau_h=72.0).max(axis=0) for c, f in flats.items()}
    table = [dict(case=r["case"], point=r["pid"], rain_mm=round(r["rain"]), observed=round(r["q_obs"], 1),
                  before=round(float(old[r["case"]][r["k"]]), 1), after=round(float(peak[r["case"]][r["k"]]), 1))
             for r in use if r["rain"] >= 30 or r.get("weight")]
    for x in table:
        print(x)
    res = np.array([np.log((peak[r["case"]][r["k"]] + 5.0) / (r["q_obs"] + 5.0)) for r in use])
    resid = {"mean_ln": round(float(res.mean()), 2), "sd_ln": round(float(res.std()), 2)}
    print("residuals ln((sim+5)/(obs+5)):", resid)
    (ROOT / "hindcast" / "hydro_fit.json").write_text(json.dumps(
        {"p0_mm": p0, "s_mm": S, "phi_mmh": phi, "wet_memory_h": tau, "score": round(score, 3), "poyo_2024_m3s": round(poyo),
         "n": len(use), "residuals": resid, "events": table}, indent=1), encoding="utf-8")
