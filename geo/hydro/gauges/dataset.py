"""Gauged catchments as a data set for the flood model: network, observed hourly flows, rain, and the event table.

    from dataset import gauge_net, Data
    net, pts = gauge_net()                 # HydroNet for the gauge points, loadable like static.hydro_net()
    D = Data()                             # continuous hourly rain per catchment + observed flow, Sep 2024 -> now
    rows = D.events()                      # one row per catchment and rain event (also written to out/events.csv)

    py -3.11 geo/hydro/gauges/dataset.py   # builds the table and prints a summary
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))
from riua.core import grid                                # noqa: E402
from riua.core.hydro import HydroNet                      # noqa: E402
import saih_series as SS                                  # noqa: E402

OUT = HERE / "out"
FLOWS = ROOT / "hindcast" / "obs" / "flows"
API_K_DAY = 0.98            # antecedent precipitation index, decay per day (Tramblay et al. 2010: best single predictor from rain)
MIN_EVENT_MM = 15.0         # basin-mean rain of an event
WET_MMH = 0.2               # an hour counts as rain above this basin mean


def gauge_net():
    """Gauge-point network with the same HydroNet.build inputs as static.hydro_net() (scope 1 = unregulated catchment)."""
    gp = json.loads((OUT / "gauge_points.json").read_text(encoding="utf-8"))["points"]
    ta = np.load(OUT / "time_area.npz", allow_pickle=False)
    ids = [str(x) for x in ta["point_ids"]]
    sel = ta["scope"] == 1
    P = len(ids)
    by = {p["id"]: p for p in gp}
    nan = [np.nan] * P
    net = HydroNet.build(ids, [by[i]["area_unregulated_km2"] for i in ids], [by[i]["t_longest_unregulated_h"] for i in ids],
                         ta["point_idx"][sel].astype(int), ta["cell"][sel].astype(int), ta["lag_h"][sel].astype(int),
                         ta["area_km2"][sel].astype(float), nan, [None] * P, [None] * P, nan, None)
    return net, [by[i] for i in ids]


def catchment(net, k):
    """cells (n,), km2 per cell (n,), and the lag table as (lag, local cell index, km2) arrays"""
    lag, cell, a = [], [], []
    for L, A in enumerate(net.lags):
        row = A.getrow(k)
        lag.append(np.full(row.nnz, L)); cell.append(row.indices); a.append(row.data)
    lag, cell, a = np.concatenate(lag), np.concatenate(cell), np.concatenate(a).astype(np.float64)
    cells, loc = np.unique(cell, return_inverse=True)
    w = np.bincount(loc, a)
    return cells, w, lag.astype(np.int64), loc.astype(np.int64), a


def _median3(v):
    if len(v) < 3:
        return v
    s = np.stack([v[:-2], v[1:-1], v[2:]])
    out = v.copy()
    out[1:-1] = np.median(s, axis=0)
    return out


def despike(v):
    """Remove sensor faults from a 5-min flow series (returns a copy with NaN): a flat plateau that is entered and left by a
    jump within one 5-min step (level sensors that latch a false echo: MC Traiguera reads 0.04 -> 235 -> 0.04 m3/s). A real
    flood can rise that fast but never falls by 60 % in five minutes."""
    v = v.astype(np.float64).copy()
    n = len(v)
    bad = np.zeros(n, bool)
    for _ in range(4):                                           # stacked plateaus (0 -> 408 -> 852 -> 0) need several passes
        idx = np.nonzero(~bad)[0]
        w = v[idx]
        downs = np.nonzero((w[1:] < 0.4 * w[:-1]) & (w[:-1] - w[1:] > 2.0))[0]
        hit = False
        for i in downs:
            lvl = w[i]
            j = i
            while j > 0 and i - j < 72 and abs(w[j - 1] - lvl) <= 0.12 * lvl:
                j -= 1
            if j > 0 and w[j - 1] < 0.5 * lvl:
                bad[idx[max(j - 2, 0)]:idx[min(i + 2, len(idx) - 1)] + 1] = True     # the plateau and the transition samples around it
                hit = True
        if not hit:
            break
    v[bad] = np.nan
    return v


class Data:
    def __init__(self, offline=False):
        self.offline = offline                              # True: never download, gauges without a cached series are skipped
        z = np.load(FLOWS / "rain_grid.npz")
        self.t = z["t_end"]; self.src = z["src"]
        self.p_all = z["p"]                                 # (T, NY*NX), NaN where no rain information
        self.p_gauges = z["p_gauges"] if "p_gauges" in z.files else None
        self.net, self.pts = gauge_net()
        self.dec = json.loads((HERE / "gauges_decisions.json").read_text(encoding="utf-8"))["gauges"]
        self.T = len(self.t)
        self.kd = API_K_DAY ** (1.0 / 24.0)
        self._obs, self._cat = {}, {}

    # ---- rain ---------------------------------------------------------------------------------------
    def cat(self, k, net=None):
        """catchment k of `net` (default: gauges): dict(cells, w, lag, loc, a, p (T, n) with NaN -> 0, cover (T,), R (T,))"""
        key = (id(net) if net is not None else 0, k)
        if key not in self._cat:
            n_ = net or self.net
            cells, w, lag, loc, a = catchment(n_, k)
            p = self.p_all[:, cells]
            fin = np.isfinite(p)
            cover = (fin * w).sum(axis=1) / w.sum()
            p0 = np.where(fin, p, 0.0)
            # basin mean over the cells that have data
            R = (p0 * w).sum(axis=1) / np.maximum((fin * w).sum(axis=1), 1e-9)
            R[cover < 0.7] = np.nan
            # cells without data take the basin mean of the hour (so the volume is right)
            p_fill = np.where(fin, p, np.nan_to_num(R)[:, None]).astype(np.float32)
            # the pluviometer interpolation alone in the truth hours (NaN elsewhere): a second opinion on the truth analysis
            Rg = np.full(self.T, np.nan)
            if self.p_gauges is not None:
                g = self.p_gauges[:, cells]
                fg = np.isfinite(g)
                cg = (fg * w).sum(axis=1) / w.sum()
                Rg[self.src == 1] = np.where(cg >= 0.7, (np.where(fg, g, 0.0) * w).sum(axis=1) / np.maximum((fg * w).sum(axis=1), 1e-9), np.nan)
            self._cat[key] = dict(cells=cells, w=w, lag=lag, loc=loc, a=a, p=p_fill, cover=cover, R=R, Rg=Rg,
                                  area=float(n_.area[k]), tl=float(n_.tc_h[k]))
        return self._cat[key]

    # ---- observed flow ------------------------------------------------------------------------------
    def obs(self, pid):
        """hourly observed flow on the rain axis: dict(qmax, qmean, n (samples per hour), ceil (rating ceiling or None))"""
        if pid in self._obs:
            return self._obs[pid]
        p = next(x for x in self.pts if x["id"] == pid)
        f = FLOWS / (f"seg_{p['var']}.npz" if p["source"] == "saih_segura" else f"chj_{p['var']}.npz")
        rels = (self.dec.get(p["var"], {}).get("release") or {}).values()
        if p["source"] == "virtual" or (self.offline and not (f.exists() and all((FLOWS / f"chj_{v}.npz").exists() for v in rels))):
            self._obs[pid] = dict(qmax=np.full(self.T, np.nan), qmean=np.full(self.T, np.nan), n=np.zeros(self.T, int),
                                  ceil=None, release=np.zeros(self.T))
            return self._obs[pid]
        t, v, _ = SS.segura_series(p["var"]) if p["source"] == "saih_segura" else SS.series(p["var"])
        v = v.astype(np.float64)
        ceil = None
        if len(v):
            vmax = v.max()
            if (v >= vmax * 0.999).sum() >= 6 and vmax > 5:        # the same top value many times: top of the rating table / error code
                ceil = float(vmax)
        v = despike(v)
        keep = np.isfinite(v)
        self.n_spike = getattr(self, "n_spike", {}); self.n_spike[pid] = int((~keep).sum())
        t, v = t[keep], _median3(v[keep])                          # single-sample spikes
        qmax, qmean, n = (np.full(self.T, np.nan), np.full(self.T, np.nan), np.zeros(self.T, int))
        for how, dst in (("max", qmax), ("mean", qmean)):
            u, val, cnt = SS.hourly(t, v, how)
            pos = (u - self.t[0]).astype(int)
            ok = (pos >= 0) & (pos < self.T) & (cnt >= 6)
            dst[pos[ok]] = val[ok]
            n[pos[ok]] = cnt[ok]
        rel = np.zeros(self.T)
        dec = self.dec.get(p["var"], {})
        for dam, var in (dec.get("release") or {}).items():
            lag = int(round(next((d["travel_h"] for d in p["dams"] if d["id"] == dam), 0)))
            tr, vr, _ = SS.series(var)
            u, val, cnt = SS.hourly(tr, _median3(vr.astype(np.float64)), "mean")
            pos = (u - self.t[0]).astype(int) + lag
            ok = (pos >= 0) & (pos < self.T)
            r = np.zeros(self.T); r[pos[ok]] = val[ok]
            rel += r
        self._obs[pid] = dict(qmax=qmax, qmean=qmean, n=n, ceil=ceil, release=rel)
        return self._obs[pid]

    # ---- events -------------------------------------------------------------------------------------
    def rain_events(self, c):
        """Rain events of one catchment (dict from cat()): rows with hour indices i0..i1 (rain), iw (end of the response window)
        and the rain statistics. Events are separated by max(12 h, 2 x longest travel time) without basin rain."""
        from scipy.signal import lfilter
        R = c["R"]
        Rz = np.nan_to_num(R)
        known = np.concatenate([[0.0], np.cumsum(np.isfinite(R))])
        gap = int(max(12, round(2 * c["tl"])))
        wet = np.nonzero(Rz >= WET_MMH)[0]
        if wet.size == 0:
            return []
        api = lfilter([1.0], [1.0, -self.kd], Rz)
        cs = np.concatenate([[0.0], np.cumsum(Rz)])
        breaks = np.nonzero(np.diff(wet) > gap)[0]
        starts = np.concatenate([[wet[0]], wet[breaks + 1]]); ends = np.concatenate([wet[breaks], [wet[-1]]])
        rows = []
        for e, (a, b) in enumerate(zip(starts, ends)):
            tot = cs[b + 1] - cs[a]
            if tot < MIN_EVENT_MM or np.isnan(R[a:b + 1]).mean() > 0.1:
                continue
            w1 = int(min(b + max(24, round(3 * c["tl"] + 12)), self.T - 1))
            if e + 1 < len(starts):
                w1 = min(w1, int(starts[e + 1]) - 1)
            w1 = max(w1, min(b + 3, self.T - 1))
            seg = Rz[a:b + 1]
            cen = float((seg * np.arange(a, b + 1)).sum() / seg.sum())
            r12 = float(max(cs[min(i + 12, b + 1)] - cs[i] for i in range(a, b + 1)))
            k5, k30 = (known[a] - known[max(a - 120, 0)]) / 120.0, (known[a] - known[max(a - 720, 0)]) / 720.0
            rows.append(dict(
                area=c["area"], t0=str(self.t[a]), t1=str(self.t[b]), i0=int(a), i1=int(b), iw=int(w1), cen=cen, imax=int(a + np.argmax(seg)),
                src="truth" if self.src[a:b + 1].mean() > 0.5 else "gauges",
                rain=round(float(tot), 1), rain1=round(float(seg.max()), 1), rain12=round(r12, 1),
                rain1_cell=round(float(c["p"][a:b + 1].max()), 1), dur_h=int(b - a + 1),
                rain5d=round(float(cs[a] - cs[max(a - 120, 0)]), 1) if k5 > 0.9 else None,
                rain30d=round(float(cs[a] - cs[max(a - 720, 0)]), 1) if k30 > 0.9 else None,
                api=round(float(api[a - 1]), 1) if (a > 0 and k30 > 0.9) else None,
                cover=round(float(np.nanmean(c["cover"][a:b + 1])), 2),
                # same event from the SAIH pluviometers alone, when they cover it (only inside truth episodes)
                rain_gauges=(round(float(c["Rg"][a:b + 1].sum()), 1)
                             if (self.src[a:b + 1] == 1).all() and np.isfinite(c["Rg"][a:b + 1]).all() else None)))
        return rows

    def events(self, write=True):
        rows = []
        for k, p in enumerate(self.pts):
            c = self.cat(k)
            o = self.obs(p["id"])
            for row in self.rain_events(c):
                a, b, w1, cen, tot = row["i0"], row["i1"], row["iw"], row["cen"], row["rain"]
                pre = slice(max(a - 6, 0), a + 1)
                win = slice(a, w1 + 1)
                qm, qx, nn, rel = o["qmean"][win], o["qmax"][win], o["n"][win], o["release"][win]
                cov = float((nn >= 6).mean())
                base = float(np.nanmedian(o["qmean"][pre])) if np.isfinite(o["qmean"][pre]).any() else np.nan
                rel_base = float(np.nanmedian(o["release"][pre]))
                row.update(pid=p["id"], name=p["name"], regulated=bool(p.get("regulated")), obs_cov=round(cov, 2))
                for b0, b1, why in self.dec.get(p.get("var", ""), {}).get("bad_periods", []):       # hand-checked sensor faults / dam spills
                    if self.t[a] <= np.datetime64(b1) and self.t[w1] >= np.datetime64(b0):
                        row["excluded"] = why
                if cov >= 0.8 and np.isfinite(base):
                    nat = np.maximum(qx - base - np.maximum(rel - rel_base, 0.0), 0.0)
                    natm = np.maximum(qm - base - np.maximum(rel - rel_base, 0.0), 0.0)
                    ip = int(np.nanargmax(nat))
                    vol = float(np.nansum(natm) * 3600.0)
                    row.update(base=round(base, 2), q_obs=round(float(nat[ip]), 2), lag_h=round(float(a + ip - cen), 1),
                               lag_i1_h=int(a + ip - row["imax"]), ip=int(a + ip),
                               runoff_mm=round(vol / (c["area"] * 1e3), 2), rc=round(vol / (c["area"] * 1e3) / tot, 4),
                               censored=bool(o["ceil"] is not None and np.nanmax(qx) >= 0.999 * o["ceil"]),
                               release_peak=round(float(np.nanmax(rel) - rel_base), 2),
                               # a peak with almost no volume behind it is a sensor pulse, not a flood (equivalent duration < 30 min)
                               glitch=bool(nat[ip] > 5.0 and vol < nat[ip] * 1800.0),
                               flat=bool(np.nanstd(qm) == 0 and base > 0))
                rows.append(row)
        if write:
            OUT.mkdir(exist_ok=True)
            (OUT / "events.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            keys = ["pid", "name", "area", "regulated", "t0", "t1", "src", "rain", "rain1", "rain12", "rain1_cell", "dur_h", "rain5d",
                    "rain30d", "api", "cover", "obs_cov", "base", "q_obs", "runoff_mm", "rc", "lag_h", "lag_i1_h", "censored",
                    "release_peak", "flat", "glitch", "excluded", "rain_gauges"]
            with open(OUT / "events.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, keys, extrasaction="ignore")
                wr.writeheader(); wr.writerows(rows)
        return rows


if __name__ == "__main__":
    D = Data()
    rows = D.events()
    ok = [r for r in rows if "q_obs" in r]
    print(len(rows), "catchment-events,", len(ok), "with observed flow;", len({r["pid"] for r in ok}), "catchments")
    print("with rain >= 50 mm:", sum(r["rain"] >= 50 for r in ok), "; with peak >= 5 m3/s:", sum(r["q_obs"] >= 5 for r in ok),
          "; censored:", sum(r["censored"] for r in ok))
