"""Products C and D: ERA5 versus reality, and the translation model space -> warning thresholds.

    python climate/build/build_scaling.py       -> scratch/c1-climate/parts/scaling.npz, events.json, scaling_report.txt

C1  ERA5-space values and return periods of the historic catastrophes and of the 2023-2026 catalogue (r6 findings).
C2  Scan of the whole record: every hour, the largest ERA5-space return period of the 24-h (and 12-h) sum over the
    225 grid points; declustered (72 h) list of the top events 1950-2026 -> how well the measure separates the
    catastrophes from ordinary heavy rain, and how often each level is reached.
C3  ERA5 return levels against rain-gauge return levels (GHCN-Daily stations + AEMET regional GPD of Ruiz & Nunez).
D   Gauge-space GEV of the daily annual maximum on the ERA5 grid (ERA5 fixed-day GEV x station-derived ratio
    fields, regional shape) and, by quantile mapping, the ERA5-space 12-h and 24-h amounts that are as frequent as
    60 / 100 / 180 / 300 mm at a gauge.
X   Operational IFS 0.25 deg (2024-02 .. 2026-09) against ERA5: how different the two "model spaces" are.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

import lmom
import stations as st
from common import PARTS, SCRATCH, cell, hour_index, index_time, precip, rolling_sum, safe_savez

DUR_H = [1, 6, 12, 24, 48, 72]
OBS_THR = np.array([60.0, 100.0, 180.0, 300.0])
XI_AEMET = 0.18          # regional GPD shape of Ruiz Garcia & Nunez Mora (AEMET): k = -0.18 +- 0.03
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


# ----------------------------------------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------------------------------------
#: (label, first day, last day, place, lat, lon, observed, source)
HISTORIC = [
    ("Turis/Chiva 29-Oct-2024", (2024, 10, 28), (2024, 10, 30), "Turis", 39.39, -0.71, "771.8 mm/24 h, 184.6 mm/1 h (Turis)", "r6 M1"),
    ("Oliva/Gandia 3-Nov-1987", (1987, 11, 2), (1987, 11, 5), "Oliva", 38.92, -0.12, "817 mm/24 h (Oliva)", "r6 B.3"),
    ("Tous 20-Oct-1982", (1982, 10, 19), (1982, 10, 21), "Muela de Cortes", 39.20, -0.90, "~900-1100 mm est. (Casa del Baron); Alicante 220 mm (19th)", "r6 B.3"),
    ("Vega Baja 12/13-Sep-2019", (2019, 9, 11), (2019, 9, 14), "Orihuela", 38.08, -0.95, "521.6 mm episode, >500 mm/48 h (Orihuela)", "r6 B.3"),
    ("Xabia 1/2-Oct-1957", (1957, 9, 30), (1957, 10, 3), "Xabia", 38.79, 0.16, "878 mm/48 h (Xabia)", "r6 B.3"),
    ("Marina Alta 12-Oct-2007", (2007, 10, 11), (2007, 10, 13), "Orba", 38.78, -0.06, ">400 mm in ~12 h", "r6 B.3"),
    ("Oct-2000 episode (21-25)", (2000, 10, 20), (2000, 10, 26), "Castello/Valencia", 39.90, -0.20, ">500 mm episode, up to 300 mm/24 h", "r6 C.4"),
    ("Vinaros 19-Oct-2018", (2018, 10, 18), (2018, 10, 20), "Vinaros", 40.47, 0.47, "159 mm/1 h", "r6 B.3"),
    ("Alicante 30-Sep-1997", (1997, 9, 29), (1997, 10, 1), "Alicante", 38.37, -0.49, "270 mm/day (GHCN SPE00101043)", "GHCN-D"),
    ("4/7-Sep-1989", (1989, 9, 3), (1989, 9, 8), "Torrevieja", 37.98, -0.71, "240 mm/day Torrevieja, 176 mm Valencia apt (GHCN)", "GHCN-D"),
    ("Valencia 17-Nov-1956", (1956, 11, 16), (1956, 11, 18), "Valencia", 39.48, -0.37, "263 mm/day (GHCN SP000008416)", "GHCN-D"),
    ("Valencia 13/14-Oct-1957 (Turia flood)", (1957, 10, 12), (1957, 10, 15), "Valencia", 39.48, -0.37, "great Turia flood (date from general knowledge, UNVERIFIED here)", "UNVERIFIED"),
]
CATALOGUE = [
    ("M1 29-Oct-2024 DANA", (2024, 10, 28), (2024, 10, 30), 39.39, -0.71, "EXTREME: Turis 772 mm, ~228 deaths"),
    ("M2 28/29-Sep-2025", (2025, 9, 28), (2025, 9, 30), 39.02, -0.30, "red verified locally: Barx 239 mm/4.5 h, modest impact"),
    ("M3 9-13-Oct-2025 Alice", (2025, 10, 9), (2025, 10, 14), 38.97, -0.18, "three >100 mm/1 h bursts; Carcaixent 168 mm"),
    ("M4 28-Dec-2025", (2025, 12, 27), (2025, 12, 29), 39.10, -0.45, ">200 mm in <12 h Ribera Alta, Magro overflow"),
    ("D1 13/14-Nov-2024", (2024, 11, 13), (2024, 11, 15), 39.55, -0.57, "red issued; ~95-100 mm/1 h locally"),
    ("D2 31-Oct-2024 N Castello", (2024, 10, 31), (2024, 11, 1), 40.47, 0.02, "red; Tirig >200 mm, Cati 176 mm"),
    ("D3 2/3-Mar-2025 Espada", (2025, 3, 2), (2025, 3, 4), 39.90, -0.35, "red (persistent); >180 mm"),
    ("D4 14-Sep-2023 Algemesi", (2023, 9, 13), (2023, 9, 15), 39.19, -0.44, "isolated storm, 170 mm in 1.5 h"),
    ("D5 2/3-Sep-2023", (2023, 9, 2), (2023, 9, 4), 39.48, -0.37, "orange; Valencia apt 71 mm, 57 mm/1 h"),
    ("D6 22-26-May-2023", (2023, 5, 22), (2023, 5, 27), 38.82, -0.61, "wettest May day on record at many stations"),
    ("D7 15-Aug-2026 Bejis", (2026, 8, 15), (2026, 8, 16), 39.91, -0.71, "isolated supercell 186 mm/2.5 h"),
    ("D12 3-5-Jan-2026 Francis", (2026, 1, 3), (2026, 1, 6), 39.02, -0.30, "persistent, Barx 186 mm in 3 days, floods nothing"),
    ("D13 10/11-Mar-2026", (2026, 3, 10), (2026, 3, 12), 38.84, -0.12, "Pego 139 mm"),
    ("D14 16/17-Jan-2025", (2025, 1, 16), (2025, 1, 18), 38.98, -0.33, "moderate persistent"),
    ("N1 14-Dec-2025 Emilia", (2025, 12, 14), (2025, 12, 16), 38.95, -0.20, "red issued, orange-level outcome (146 mm max)"),
    ("N2 3-Nov-2024", (2024, 11, 3), (2024, 11, 4), 39.10, -0.30, "false alarm for red"),
    ("N3 8-Sep-2025", (2025, 9, 8), (2025, 9, 9), 40.00, 0.00, "probable false alarm for orange"),
    ("O1 29-Jan-2023", (2023, 1, 29), (2023, 1, 30), 38.95, -0.15, "ordinary, 30 mm"),
    ("O2 11-Dec-2024", (2024, 12, 11), (2024, 12, 12), 38.95, -0.15, "ordinary, >50 mm Safor"),
    ("O6 12-Apr-2026", (2026, 4, 12), (2026, 4, 13), 38.50, -0.50, "ordinary general rain ~50 mm"),
    ("O7 5-Jun-2026", (2026, 6, 5), (2026, 6, 6), 38.90, -0.15, "upper yellow, 81 mm"),
    ("Z1 25-Oct-2023", (2023, 10, 25), (2023, 10, 26), 39.40, -0.50, "dry autumn day"),
    ("Z2 15-Nov-2023", (2023, 11, 15), (2023, 11, 16), 39.40, -0.50, "dry autumn day"),
]
#: labels attached to the top events of the scan when the date falls inside (first day, last day)
KNOWN = [((1957, 9, 30), (1957, 10, 3), "Xabia 878 mm/48 h [r6]"), ((1957, 10, 12), (1957, 10, 15), "Valencia Turia flood [UNVERIFIED date]"),
         ((1982, 10, 19), (1982, 10, 21), "Tous dam break [r6]"), ((1987, 11, 2), (1987, 11, 5), "Oliva 817 mm [r6]"),
         ((2000, 10, 20), (2000, 10, 26), "Oct 2000 episode [r6]"), ((2007, 10, 11), (2007, 10, 13), "Marina Alta >400 mm [r6]"),
         ((2019, 9, 11), (2019, 9, 14), "Vega Baja 2019 [r6]"), ((2024, 10, 28), (2024, 10, 30), "29-Oct-2024 DANA [r6]"),
         ((2018, 10, 18), (2018, 10, 20), "Vinaros 159 mm/1 h [r6]"), ((1997, 9, 29), (1997, 10, 1), "Alicante 270 mm [GHCN]"),
         ((1989, 9, 3), (1989, 9, 8), "Sep 1989: Torrevieja 240 mm [GHCN]"), ((1956, 11, 16), (1956, 11, 18), "Valencia 263 mm [GHCN]"),
         ((2012, 9, 27), (2012, 9, 29), "28-Sep-2012: Valencia apt 189 mm [GHCN]"), ((2016, 12, 16), (2016, 12, 20), "Dec 2016: San Javier 179 mm [GHCN]"),
         ((2025, 9, 28), (2025, 9, 30), "M2 Sep 2025 [r6]"), ((2025, 10, 9), (2025, 10, 14), "M3 Alice [r6]"),
         ((2025, 12, 27), (2025, 12, 29), "M4 Dec 2025 [r6]"), ((2024, 11, 13), (2024, 11, 15), "D1 Nov 2024 [r6]"),
         ((2025, 3, 2), (2025, 3, 4), "D3 Mar 2025 [r6]"), ((2023, 5, 22), (2023, 5, 27), "D6 May 2023 [r6]; Castello 198 mm [GHCN]"),
         ((2025, 12, 14), (2025, 12, 16), "N1 Emilia [r6]"), ((2024, 10, 31), (2024, 11, 1), "D2 N Castello [r6]")]


def rp_interval(amount, T, rl_lo, rl_hi):
    """Return-period range implied by the stored 90 % band of the return levels (log-linear in T).
    lower T: where the upper band reaches the amount; upper T: where the lower band reaches it."""
    lt = np.log(T)

    def inv(curve):
        if amount <= curve[0]:
            return float(T[0])
        if amount >= curve[-1]:
            return float("inf")
        return float(np.exp(np.interp(amount, curve, lt)))
    return inv(rl_hi), inv(rl_lo)


def bilin(field, la, lo, glat, glon):
    fy = np.clip((la - glat[0]) / 0.25, 0, len(glat) - 1.0)
    fx = np.clip((lo - glon[0]) / 0.25, 0, len(glon) - 1.0)
    y0, x0 = min(int(fy), len(glat) - 2), min(int(fx), len(glon) - 2)
    wy, wx = fy - y0, fx - x0
    return (field[..., y0, x0] * (1 - wy) * (1 - wx) + field[..., y0, x0 + 1] * (1 - wy) * wx
            + field[..., y0 + 1, x0] * wy * (1 - wx) + field[..., y0 + 1, x0 + 1] * wy * wx)


def fmt_T(t):
    if not np.isfinite(t) or t >= 9999:
        return ">9999"
    return f"{t:.0f}" if t >= 10 else f"{t:.1f}"


def main():
    pr, lat, lon = precip()
    x = np.array(pr)
    x[x == 32767] = 0
    ev = np.load(PARTS / "eva.npz")
    T = ev["eva_T"]
    loc, scale, xi = ev["gev_loc"].astype(float), ev["gev_scale"].astype(float), ev["gev_shape"].astype(float)
    rl, rl_lo, rl_hi = ev["rl"].astype(float), ev["rl_lo"].astype(float), ev["rl_hi"].astype(float)
    ny, nx = len(lat), len(lon)
    out_json: dict = {}

    # ------------------------------------------------------------------------------------------------ C1
    say("=" * 110)
    say("C1. HISTORIC EVENTS IN ERA5 SPACE (GEV 1950-2025, regional shape). amount mm / return period yr [90 % range]")
    say("    'cell' = grid point nearest to the place; 'nbh' = worst of the 3 x 3 points around it; 'box' = worst of the 225 points")
    say("=" * 110)

    def event_stats(d0, d1, la, lo):
        a, b = hour_index(*d0), hour_index(*d1) + 24
        i, j = cell(lat, lon, la, lo)
        res = {}
        for k, w in enumerate(DUR_H):
            seg = rolling_sum(x[a - w:b].astype(np.int32), w)[w:] / 10.0          # windows ending inside [a, b)
            amax = seg.max(axis=0)                                                # [ny, nx]
            when = seg.argmax(axis=0)
            rp = lmom.return_period(amax, loc[k], scale[k], xi[k])
            ii, jj = slice(max(0, i - 1), i + 2), slice(max(0, j - 1), j + 2)
            sub = rp[ii, jj]
            n_i, n_j = np.unravel_index(np.argmax(sub), sub.shape)
            n_i, n_j = n_i + max(0, i - 1), n_j + max(0, j - 1)
            b_i, b_j = np.unravel_index(np.argmax(rp), rp.shape)
            lo_c, hi_c = rp_interval(amax[n_i, n_j], T, rl_lo[k, :, n_i, n_j], rl_hi[k, :, n_i, n_j])
            res[w] = dict(cell_mm=float(amax[i, j]), cell_T=float(rp[i, j]), nbh_mm=float(amax[n_i, n_j]), nbh_T=float(rp[n_i, n_j]),
                          nbh_T_lo=lo_c, nbh_T_hi=hi_c, nbh_lat=float(lat[n_i]), nbh_lon=float(lon[n_j]),
                          box_mm=float(amax[b_i, b_j]), box_T=float(rp[b_i, b_j]), box_lat=float(lat[b_i]), box_lon=float(lon[b_j]),
                          box_end=str(index_time(a + when[b_i, b_j])), n_cells_T5=int((rp >= 5).sum()), n_cells_T10=int((rp >= 10).sum()),
                          n_cells_T25=int((rp >= 25).sum()))
        return res

    hist = []
    for label, d0, d1, place, la, lo, obs, src in HISTORIC:
        r = event_stats(d0, d1, la, lo)
        hist.append(dict(label=label, place=place, lat=la, lon=lo, observed=obs, source=src, era5={str(k): v for k, v in r.items()}))
        say(f"{label}  [{place} {la:.2f},{lo:+.2f}]  observed: {obs}  ({src})")
        for w in DUR_H:
            v = r[w]
            say(f"   {w:>2} h: cell {v['cell_mm']:6.1f} mm T={fmt_T(v['cell_T']):>5} | nbh {v['nbh_mm']:6.1f} mm T={fmt_T(v['nbh_T']):>5} "
                f"[{fmt_T(v['nbh_T_lo'])}-{fmt_T(v['nbh_T_hi'])}] | box {v['box_mm']:6.1f} mm T={fmt_T(v['box_T']):>5} at {v['box_lat']:.2f},{v['box_lon']:+.2f} "
                f"| cells T>=10: {v['n_cells_T10']:3d}, T>=25: {v['n_cells_T25']:3d}")
    out_json["historic"] = hist

    say()
    say("C1b. CATALOGUE 2023-2026 (r6 Part A) IN ERA5 SPACE: 12-h and 24-h, worst 3x3 point near the place and worst point of the box")
    cat = []
    say(f"{'event':28s} {'nbh12 mm':>8s} {'T12':>6s} {'nbh24 mm':>8s} {'T24':>6s} | {'box24 mm':>8s} {'T24box':>6s} {'T12box':>6s} {'T1box':>6s} n(T24>=10) | outcome")
    for label, d0, d1, la, lo, outcome in CATALOGUE:
        r = event_stats(d0, d1, la, lo)
        cat.append(dict(label=label, lat=la, lon=lo, outcome=outcome, era5={str(k): v for k, v in r.items()}))
        say(f"{label:28s} {r[12]['nbh_mm']:8.1f} {fmt_T(r[12]['nbh_T']):>6s} {r[24]['nbh_mm']:8.1f} {fmt_T(r[24]['nbh_T']):>6s} | "
            f"{r[24]['box_mm']:8.1f} {fmt_T(r[24]['box_T']):>6s} {fmt_T(r[12]['box_T']):>6s} {fmt_T(r[1]['box_T']):>6s} {r[24]['n_cells_T10']:6d}     | {outcome}")
    out_json["catalogue"] = cat

    # ------------------------------------------------------------------------------------------------ C2
    say()
    say("=" * 110)
    say("C2. SCAN OF THE WHOLE RECORD: hourly maximum over the box of the ERA5-space return period")
    say("=" * 110)
    n = x.shape[0]
    h1950, h_end = hour_index(1950), hour_index(2026, 9, 25)
    scans = {}
    for w in (24, 12):
        k = DUR_H.index(w)
        best = np.ones(n, np.float32)
        arg = np.zeros(n, np.int16)
        amt = np.zeros(n, np.float32)
        n10 = np.zeros(n, np.int16)
        for i in range(ny):
            rs = rolling_sum(x[:, i, :].astype(np.int32), w) / 10.0
            rp = lmom.return_period(rs, loc[k, i], scale[k, i], xi[k, i]).astype(np.float32)
            rp[:w] = 1.0
            n10 += (rp >= 10).sum(axis=1).astype(np.int16)
            jb = rp.argmax(axis=1)
            vb = rp[np.arange(n), jb]
            upd = vb > best
            best[upd] = vb[upd]
            arg[upd] = (i * nx + jb[upd]).astype(np.int16)
            amt[upd] = rs[np.arange(n), jb][upd]
        # decluster: greedy, peaks at least 72 h apart
        cand = np.nonzero(best >= 1.5)[0]
        cand = cand[(cand >= h1950) & (cand < h_end)]
        order = cand[np.argsort(-best[cand], kind="stable")]
        taken = np.zeros(n, bool)
        events = []
        for t in order:
            if taken[max(0, t - 72):t + 73].any():
                continue
            taken[t] = True
            a, b = max(0, t - 72), min(n, t + 73)
            events.append((int(t), float(best[t]), float(amt[t]), int(arg[t]), int(n10[a:b].max())))
        scans[w] = events
        yrs = (h_end - h1950) / 8766.0
        say(f"--- {w}-h sums, 1950-01 .. 2026-09 ({yrs:.1f} yr), events = peaks >= 72 h apart")
        for thr in (2, 5, 10, 25, 50, 100, 200):
            c = sum(1 for e in events if e[1] >= thr)
            say(f"   box-max T >= {thr:>3} yr: {c:4d} events = {c / yrs:5.2f} per year")
        say(f"   top 40 by box-max return period ({w} h):")
        say(f"   {'rank':>4s} {'window centre (UTC)':19s} {'T yr':>6s} {'mm':>6s} {'cell':>13s} {'n cells T>=10':>13s}  known event")
        toplist = []
        for r, (t, v, am, c, nn) in enumerate(events[:40], 1):
            when = index_time(t) - dt.timedelta(hours=w // 2)
            lab = ""
            for d0, d1, name in KNOWN:
                if dt.datetime(*d0) <= when <= dt.datetime(*d1) + dt.timedelta(days=1):
                    lab = name
            toplist.append(dict(rank=r, centre=str(when), T=v, mm=am, lat=float(lat[c // nx]), lon=float(lon[c % nx]), n_cells_T10=nn, known=lab))
            say(f"   {r:4d} {str(when):19s} {fmt_T(v):>6s} {am:6.1f} {lat[c // nx]:6.2f},{lon[c % nx]:+5.2f} {nn:13d}  {lab}")
        # rank of each known event
        say(f"   rank of the known events in the {w}-h list (of {len(events)} events with T >= 1.5):")
        ranks = []
        for d0, d1, name in KNOWN:
            a, b = dt.datetime(*d0), dt.datetime(*d1) + dt.timedelta(days=1)
            hit = [(r, e) for r, e in enumerate(events, 1) if a <= index_time(e[0]) - dt.timedelta(hours=w // 2) <= b]
            if hit:
                r, e = hit[0]
                say(f"     {name:45s} rank {r:4d}  T={fmt_T(e[1]):>5s}  {e[2]:6.1f} mm  n(T>=10)={e[4]}")
                ranks.append(dict(name=name, rank=r, T=e[1], mm=e[2], n_cells_T10=e[4]))
            else:
                say(f"     {name:45s} not in the list (box-max T < 1.5 yr)")
                ranks.append(dict(name=name, rank=None))
        out_json[f"scan_{w}h"] = dict(top=toplist, known_ranks=ranks,
                                      rates_per_year={str(thr): sum(1 for e in events if e[1] >= thr) / yrs for thr in (2, 5, 10, 25, 50, 100)})

    # ------------------------------------------------------------------------------------------------ C3 + D
    say()
    say("=" * 110)
    say("C3. ERA5 AGAINST RAIN GAUGES: daily annual maxima (gauge: pluviometric day; ERA5: fixed 07-07 UTC day, bilinear GEV)")
    say("=" * 110)
    kday = list(ev["eva_durations"]).index("day")
    e_loc, e_scale, e_xi = loc[kday], scale[kday], xi[kday]
    am = np.load(PARTS / "annmax.npz")
    am_day, am_years = am["am"][kday], am["years"]
    rows = []
    for sid, (name, la, lo, el) in st.STATIONS.items():
        y, a, _d = st.annual_maxima(sid)
        if len(y) < 25:
            continue
        l1, l2, t3, t4 = [float(v) for v in lmom.sample_lmom(a)]
        s_loc, s_scale, s_xi = [float(v) for v in lmom.gev_fit(a)]
        # ERA5 at the station, same years where possible (empirical) and GEV 1950-2025 (bilinear parameters)
        el1 = float(bilin(am_day[np.isin(am_years, y)].mean(axis=0), la, lo, lat, lon))
        p_e = [float(bilin(f, la, lo, lat, lon)) for f in (e_loc, e_scale, e_xi)]
        rows.append(dict(id=sid, name=name, lat=la, lon=lo, elev=el, n=len(y), y0=int(y.min()), y1=int(y.max()), l1=l1, l2=l2, t3=t3, t4=t4,
                         site=(s_loc, s_scale, s_xi), era=p_e, era_l1_same_years=el1, am_max=float(a.max())))
    coastal = [r for r in rows if r["elev"] < 200]
    wsum = sum(r["n"] for r in coastal)
    t3_reg = sum(r["t3"] * r["n"] for r in coastal) / wsum
    xi_reg_obs = -float(lmom.gev_k_from_t3(t3_reg))
    inland = [r for r in rows if r["elev"] >= 200]
    t3_in = sum(r["t3"] * r["n"] for r in inland) / sum(r["n"] for r in inland)
    xi_in_obs = -float(lmom.gev_k_from_t3(t3_in))
    say(f"Gauge regional shape (record-length-weighted L-skewness): coastal/lowland stations (n={len(coastal)}) t3={t3_reg:.3f} -> xi={xi_reg_obs:+.3f}; "
        f"interior plateau stations (n={len(inland)}) t3={t3_in:.3f} -> xi={xi_in_obs:+.3f}.  AEMET regional GPD: xi=+0.18 +- 0.03")
    say(f"{'station':22s} {'yrs':>4s} {'meanAM':>6s} {'xi_site':>7s} | obs T10 / T100 (regional xi) | ERA5 day T10 / T100 | ratio obs/ERA5: mean  T10  T100 | max obs")
    for r in rows:
        xr = xi_reg_obs if r["elev"] < 200 else xi_in_obs
        r["reg"] = [float(v) for v in lmom.gev_from_lmom(r["l1"], r["l2"], -xr)]
        o10, o100 = [float(lmom.return_level(t, *r["reg"])) for t in (10, 100)]
        e10, e100 = [float(lmom.return_level(t, *r["era"])) for t in (10, 100)]
        r.update(o10=o10, o100=o100, e10=e10, e100=e100)
        say(f"{r['name']:22s} {r['n']:4d} {r['l1']:6.1f} {r['site'][2]:+7.2f} | {o10:6.0f} / {o100:6.0f}             | {e10:6.1f} / {e100:6.1f}      | "
            f"{r['l1'] / r['era_l1_same_years']:5.2f} {o10 / e10:5.2f} {o100 / e100:5.2f} | {r['am_max']:.0f}")
    # AEMET regional GPD anchors -> GEV of the annual maximum:  loc = u + s (lam^xi - 1) / xi, scale = s lam^xi, s = <x> (1 - xi)
    def gpd_to_gev(lam, mean_excess, u=30.0, xi_=XI_AEMET):
        s = mean_excess * (1 - xi_)
        return u + s * (lam ** xi_ - 1) / xi_, s * lam ** xi_, xi_
    anchors = {"Alicante (AEMET GPD lam=2, <x>=20)": (38.37, -0.49, gpd_to_gev(2, 20)),
               "Oliva / Safor coast (AEMET GPD lam=8, <x>=35)": (38.92, -0.12, gpd_to_gev(8, 35))}
    for name, (la, lo, g) in anchors.items():
        p_e = [float(bilin(f, la, lo, lat, lon)) for f in (e_loc, e_scale, e_xi)]
        p24 = [float(bilin(f[DUR_H.index(24)], la, lo, lat, lon)) for f in (loc, scale, xi)]
        say(f"{name}: GEV loc {g[0]:.1f} scale {g[1]:.1f} xi {g[2]:.2f}")
        for t in (10, 25, 100, 500):
            o, e, e24 = float(lmom.return_level(t, *g)), float(lmom.return_level(t, *p_e)), float(lmom.return_level(t, *p24))
            say(f"     T={t:>3}: gauge {o:6.0f} mm | ERA5 fixed day {e:6.1f} mm (ratio {o / e:.2f}) | ERA5 sliding 24 h {e24:6.1f} mm (ratio {o / e24:.2f})")
    ali = next(r for r in rows if r["id"] == "SPE00101043")
    say(f"Check Alicante: AEMET formula T100 = {float(lmom.return_level(100, *gpd_to_gev(2, 20))):.0f} mm; GHCN station fit (regional xi) T100 = {ali['o100']:.0f} mm, "
        f"at-site xi T100 = {float(lmom.return_level(100, *ali['site'])):.0f} mm")

    say()
    say("=" * 110)
    say("D. GAUGE-SPACE GEV ON THE GRID AND MODEL-SPACE EQUIVALENTS OF THE WARNING THRESHOLDS")
    say("=" * 110)
    # control points: stations (regional-xi GEV) + the Safor anchor; ratios to ERA5 fixed-day GEV at the same place
    ctrl = []
    for r in rows:
        ctrl.append(dict(name=r["name"], lat=r["lat"], lon=r["lon"], w=min(r["n"], 60) / 60.0, gev=r["reg"], era=r["era"]))
    la, lo, g = anchors["Oliva / Safor coast (AEMET GPD lam=8, <x>=35)"]
    ctrl.append(dict(name="Safor coast (AEMET GPD)", lat=la, lon=lo, w=1.0, gev=list(g), era=[float(bilin(f, la, lo, lat, lon)) for f in (e_loc, e_scale, e_xi)]))
    c_lat = np.array([c["lat"] for c in ctrl]); c_lon = np.array([c["lon"] for c in ctrl]); c_w = np.array([c["w"] for c in ctrl])
    c_rloc = np.log(np.array([c["gev"][0] / c["era"][0] for c in ctrl]))
    c_rsc = np.log(np.array([c["gev"][1] / c["era"][1] for c in ctrl]))
    c_xi = np.array([c["gev"][2] for c in ctrl])

    def idw(la_, lo_, vals, skip=None, d0=25.0):
        dy = (c_lat - la_) * 111.2
        dx = (c_lon - lo_) * 111.2 * np.cos(np.radians(39.3))
        wgt = c_w / (dx * dx + dy * dy + d0 * d0)
        if skip is not None:
            wgt = wgt.copy(); wgt[skip] = 0.0
        return float((wgt * vals).sum() / wgt.sum())

    say(f"{'control point':26s} ratio gauge/ERA5-day: loc  scale | xi_obs | leave-one-out prediction error ln(pred/own): T10   T100")
    loo = []
    for s, c in enumerate(ctrl):
        same = [k for k, c2 in enumerate(ctrl) if abs(c2["lat"] - c["lat"]) < 0.15 and abs(c2["lon"] - c["lon"]) < 0.15]   # drop the twin station too
        pl = c["era"][0] * np.exp(idw(c["lat"], c["lon"], c_rloc, same))
        ps = c["era"][1] * np.exp(idw(c["lat"], c["lon"], c_rsc, same))
        px = idw(c["lat"], c["lon"], c_xi, same)
        e10 = float(np.log(lmom.return_level(10, pl, ps, px) / lmom.return_level(10, *c["gev"])))
        e100 = float(np.log(lmom.return_level(100, pl, ps, px) / lmom.return_level(100, *c["gev"])))
        loo.append((np.log(pl / c["gev"][0]), np.log(ps / c["gev"][1]), e10, e100))
        say(f"{c['name']:26s}                     {np.exp(c_rloc[s]):5.2f} {np.exp(c_rsc[s]):5.2f} | {c_xi[s]:+.2f}  |                                         {e10:+.2f}  {e100:+.2f}")
    loo = np.array(loo)
    s_loc, s_sc = float(np.sqrt((loo[:, 0] ** 2).mean())), float(np.sqrt((loo[:, 1] ** 2).mean()))
    say(f"leave-one-out RMS of ln: loc {s_loc:.2f}, scale {s_sc:.2f}, T10 {np.sqrt((loo[:, 2] ** 2).mean()):.2f}, T100 {np.sqrt((loo[:, 3] ** 2).mean()):.2f} "
        f"(i.e. the gauge-space T100 at an ungauged grid point is known to within a factor ~{np.exp(np.sqrt((loo[:, 3] ** 2).mean())):.2f})")

    o_loc = np.zeros((ny, nx)); o_scale = np.zeros((ny, nx)); o_xi = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            o_loc[i, j] = e_loc[i, j] * np.exp(idw(lat[i], lon[j], c_rloc))
            o_scale[i, j] = e_scale[i, j] * np.exp(idw(lat[i], lon[j], c_rsc))
            o_xi[i, j] = idw(lat[i], lon[j], c_xi)
    F = np.stack([lmom.gev_cdf(t, o_loc, o_scale, o_xi) for t in OBS_THR])
    Fc = np.clip(F, 0.02, 1 - 1 / 2000.0)
    obs_T = 1.0 / (1.0 - Fc)
    k12, k24 = DUR_H.index(12), DUR_H.index(24)
    eq12 = lmom.gev_quantile(Fc, loc[k12], scale[k12], xi[k12])
    eq24 = lmom.gev_quantile(Fc, loc[k24], scale[k24], xi[k24])
    # Monte Carlo: gauge-side uncertainty (LOO spread of the ratio fields, xi +- 0.05) and ERA5 shape uncertainty
    rng = np.random.default_rng(7)
    nmc = 400
    sh_sd = (ev["gev_shape_hi"].astype(float) - ev["gev_shape_lo"].astype(float)) / 3.29
    mc12 = np.zeros((nmc,) + eq12.shape); mc24 = np.zeros_like(mc12)
    for m in range(nmc):
        ol = o_loc * np.exp(rng.normal(0, s_loc)); os_ = o_scale * np.exp(rng.normal(0, s_sc)); ox = o_xi + rng.normal(0, 0.05)
        Fm = np.clip(np.stack([lmom.gev_cdf(t, ol, os_, ox) for t in OBS_THR]), 0.02, 1 - 1 / 2000.0)
        z = rng.normal()
        mc12[m] = lmom.gev_quantile(Fm, loc[k12], scale[k12], xi[k12] + z * sh_sd[k12])
        mc24[m] = lmom.gev_quantile(Fm, loc[k24], scale[k24], xi[k24] + z * sh_sd[k24])
    eq12_lo, eq12_hi = np.percentile(mc12, [5, 95], axis=0)
    eq24_lo, eq24_hi = np.percentile(mc24, [5, 95], axis=0)

    pts = {"Valencia": (39.48, -0.37), "Turis": (39.39, -0.71), "Oliva (Safor)": (38.92, -0.12), "Alicante": (38.37, -0.49),
           "Castello": (39.99, -0.04), "Orihuela": (38.08, -0.95), "Utiel": (39.57, -1.20), "Morella": (40.62, -0.10), "Denia": (38.84, 0.11)}
    say()
    say("Gauge thresholds -> gauge return period (yr, annual-maximum sense) -> ERA5-space 12-h / 24-h equivalents (mm) [90 % range]")
    table = {}
    for name, (la, lo) in pts.items():
        i, j = cell(lat, lon, la, lo)
        say(f"  {name} (grid point {lat[i]:.2f},{lon[j]:+.2f}; gauge-space GEV loc {o_loc[i, j]:.0f} scale {o_scale[i, j]:.0f} xi {o_xi[i, j]:+.2f}; "
            f"ERA5 24 h T2/T10/T100 = {rl[k24, 0, i, j]:.0f}/{rl[k24, 2, i, j]:.0f}/{rl[k24, 5, i, j]:.0f} mm)")
        table[name] = {}
        for t, thr in enumerate(OBS_THR):
            flag = " (clipped)" if (F[t, i, j] <= 0.02 or F[t, i, j] >= 1 - 1 / 2000.0) else ""
            say(f"     {thr:5.0f} mm at a gauge: T = {fmt_T(obs_T[t, i, j]):>5s} yr{flag} -> ERA5 12 h {eq12[t, i, j]:6.1f} [{eq12_lo[t, i, j]:5.1f}-{eq12_hi[t, i, j]:5.1f}]"
                f"   24 h {eq24[t, i, j]:6.1f} [{eq24_lo[t, i, j]:5.1f}-{eq24_hi[t, i, j]:5.1f}]")
            table[name][str(int(thr))] = dict(T_obs=float(obs_T[t, i, j]), era12=float(eq12[t, i, j]), era24=float(eq24[t, i, j]),
                                              era12_range=[float(eq12_lo[t, i, j]), float(eq12_hi[t, i, j])], era24_range=[float(eq24_lo[t, i, j]), float(eq24_hi[t, i, j])])
    say()
    say("Box summary of the equivalents (median [p10-p90] over the 225 grid points):")
    for t, thr in enumerate(OBS_THR):
        say(f"   {thr:5.0f} mm observed -> ERA5 12 h {np.median(eq12[t]):5.1f} [{np.percentile(eq12[t], 10):5.1f}-{np.percentile(eq12[t], 90):5.1f}] mm; "
            f"24 h {np.median(eq24[t]):5.1f} [{np.percentile(eq24[t], 10):5.1f}-{np.percentile(eq24[t], 90):5.1f}] mm; "
            f"gauge T median {fmt_T(float(np.median(obs_T[t])))} yr; ratio obs/ERA5-24h median {np.median(thr / eq24[t]):.2f}")
    out_json["equivalents_points"] = table
    out_json["stations"] = [{k: v for k, v in r.items()} for r in rows]

    # ------------------------------------------------------------------------------------------------ X
    say()
    say("=" * 110)
    say("X. OPERATIONAL IFS 0.25 deg (stitched 0-6 h forecasts, 2024-02 .. 2026-09) AGAINST ERA5, same grid points")
    say("=" * 110)
    qmap = {}
    f_ifs = SCRATCH / "ifs025_precip.npz"
    if f_ifs.exists():
        zi = np.load(f_ifs)
        d, hrs = zi["data"], zi["hours"]
        h0 = hour_index(1940) + 0
        epoch_off = (dt.datetime(1940, 1, 1) - dt.datetime(1970, 1, 1)).days * 24      # hours from 1970 to 1940 (negative)
        ok = np.isfinite(d).all(axis=(1, 2)) & (hrs - epoch_off < h_end)
        first, last = np.nonzero(ok)[0][0], np.nonzero(ok)[0][-1]
        d, hrs = d[first:last + 1], hrs[first:last + 1]
        assert np.isfinite(d).all() and (np.diff(hrs) == 3).all()
        for w in (24, 12):
            steps = w // 3
            c = np.cumsum(d, axis=0, dtype=np.float64)
            ifs_sum = c[steps:] - c[:-steps]                       # window ending at hrs[steps:]
            ends = hrs[steps:] - epoch_off                         # index on the 1940 axis; value at index t = hour ending t
            sel = ends % 6 == 0
            ifs_sum, ends = ifs_sum[sel], ends[sel]
            base = int(ends.min()) - w
            cs = np.cumsum(x[base:int(ends.max()) + 1], axis=0, dtype=np.int64)
            era_sum = (cs[ends - base] - cs[ends - base - w]) / 10.0
            del cs
            say(f"--- {w}-h sums ending 00/06/12/18 UTC: {len(ends)} windows x 225 points, {index_time(int(ends[0]))} .. {index_time(int(ends[-1]))}")
            say(f"   mean: IFS {ifs_sum.mean():.3f} mm, ERA5 {era_sum.mean():.3f} mm (ratio {ifs_sum.mean() / era_sum.mean():.2f}); "
                f"correlation of the window sums {np.corrcoef(ifs_sum.ravel(), era_sum.ravel())[0, 1]:.2f}")
            ps = np.array([0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999])
            qi, qe = np.quantile(ifs_sum, ps), np.quantile(era_sum, ps)
            say("   pooled quantiles     p: " + " ".join(f"{p:8.5f}" for p in ps))
            say("            ERA5 (mm)    : " + " ".join(f"{v:8.1f}" for v in qe))
            say("            IFS025 (mm)  : " + " ".join(f"{v:8.1f}" for v in qi))
            say("            IFS / ERA5   : " + " ".join(f"{(a / b if b > 0 else np.nan):8.2f}" for a, b in zip(qi, qe)))
            # per-point maxima over the common period
            mi, me = ifs_sum.max(axis=0), era_sum.max(axis=0)
            say(f"   per-point maximum of the period: IFS median {np.median(mi):.1f} mm, ERA5 median {np.median(me):.1f} mm, "
                f"ratio median {np.median(mi / me):.2f} (p10 {np.percentile(mi / me, 10):.2f}, p90 {np.percentile(mi / me, 90):.2f})")
            # mean of the 10 largest values at each point (more stable than the single maximum)
            ti = np.sort(ifs_sum, axis=0)[-10:].mean(axis=0); te = np.sort(era_sum, axis=0)[-10:].mean(axis=0)
            say(f"   per-point mean of the 10 largest: ratio IFS/ERA5 median {np.median(ti / te):.2f} (p10 {np.percentile(ti / te, 10):.2f}, p90 {np.percentile(ti / te, 90):.2f})")
            dense = np.concatenate([np.linspace(0.5, 0.99, 50), 1 - np.logspace(-2, -5, 25)[1:]])
            qmap[f"ifs_qmap{w}_era5"] = np.quantile(era_sum, dense).astype(np.float32)
            qmap[f"ifs_qmap{w}_ifs"] = np.quantile(ifs_sum, dense).astype(np.float32)
            if w == 24:
                # 29 Oct 2024: 24 h ending 30 Oct 00 UTC
                t_ev = hour_index(2024, 10, 30)
                kk = int(np.nonzero(ends == t_ev)[0][0])
                i, j = cell(lat, lon, 39.39, -0.71)
                say(f"   29-Oct-2024 00-24 UTC: Turis point IFS {ifs_sum[kk, i, j]:.1f} mm vs ERA5 {era_sum[kk, i, j]:.1f} mm; box max IFS {ifs_sum[kk].max():.1f} vs ERA5 {era_sum[kk].max():.1f}")
    else:
        say("ifs025_precip.npz not found: section skipped")

    safe_savez(PARTS / "scaling.npz", obs_thresholds_mm=OBS_THR, obs_gev_loc=o_loc.astype(np.float32), obs_gev_scale=o_scale.astype(np.float32),
               obs_gev_shape=o_xi.astype(np.float32), obs_T=obs_T.astype(np.float32),
               equiv12=eq12.astype(np.float32), equiv12_lo=eq12_lo.astype(np.float32), equiv12_hi=eq12_hi.astype(np.float32),
               equiv24=eq24.astype(np.float32), equiv24_lo=eq24_lo.astype(np.float32), equiv24_hi=eq24_hi.astype(np.float32), **qmap)
    (PARTS / "events.json").write_text(json.dumps(out_json, indent=1), encoding="utf-8")
    (PARTS / "scaling_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("saved scaling.npz, events.json, scaling_report.txt")


if __name__ == "__main__":
    main()
