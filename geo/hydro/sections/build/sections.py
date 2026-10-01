"""Step 3: thalweg trace, 3 cross-sections, banks, slope, Manning rating, CAUMAX, QA plot.

Per point writes
  out/profiles/<id>.json                      raw profiles of the 3 sections
  scratch/h2-sections/results/<id>.json       hydraulic result (assembled later by assemble.py)
  scratch/h2-sections/plots/<id>.png          QA figure (DTM map, orthophoto, profiles)
Usage: py -3.11 sections.py [id ...]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from shapely.geometry import LineString

import hydraulics as H
from common import (DTM_DIR, MISC, OSM_DIR, PLOTS, PROFILES, SCRATCH, UA, jdump, load_config, load_seed, utm2ll)

RESULTS = os.path.join(SCRATCH, "results")
ORTHO = os.path.join(SCRATCH, "ortho")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(ORTHO, exist_ok=True)

C_SEC = ["#2a78b5", "#d1495b", "#3a9b5c"]   # upstream, chosen, downstream
LBL_SEC = ["upstream", "chosen", "downstream"]


# ----------------------------------------------------------------------------- raster helpers
def bilinear(arr, tr, x, y):
    col = (np.asarray(x) - tr.c) / tr.a - 0.5
    row = (tr.f - np.asarray(y)) / (-tr.e) - 0.5
    c0 = np.floor(col).astype(int)
    r0 = np.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    h, w = arr.shape
    ok = (c0 >= 0) & (c0 < w - 1) & (r0 >= 0) & (r0 < h - 1)
    c0c, r0c = np.clip(c0, 0, w - 2), np.clip(r0, 0, h - 2)
    v = (arr[r0c, c0c] * (1 - fc) * (1 - fr) + arr[r0c, c0c + 1] * fc * (1 - fr)
         + arr[r0c + 1, c0c] * (1 - fc) * fr + arr[r0c + 1, c0c + 1] * fc * fr)
    return np.where(ok, v, np.nan)


def moving(a, k):
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    return np.convolve(ap, np.ones(k) / k, mode="valid")


def medfilt(a, k):
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    return np.array([np.median(ap[i:i + k]) for i in range(len(a))])


# ----------------------------------------------------------------------------- thalweg
def trace_thalweg(anchor, arr, tr, snap_w):
    xy = np.array(anchor["xy"])
    chain = np.array(anchor["chain"])
    cov = np.array(anchor["covered"], bool)
    k = 21 if len(xy) > 25 else 5
    sx, sy = moving(xy[:, 0], k), moving(xy[:, 1], k)
    tx, ty = np.gradient(sx), np.gradient(sy)
    nrm = np.hypot(tx, ty)
    tx, ty = tx / nrm, ty / nrm
    offs = np.arange(-snap_w, snap_w + 1)
    X = xy[:, 0][:, None] + offs[None, :] * ty[:, None]
    Y = xy[:, 1][:, None] - offs[None, :] * tx[:, None]
    Z = bilinear(arr, tr, X, Y)
    Zs = np.stack([moving(np.where(np.isnan(r), np.nanmax(r) if np.any(~np.isnan(r)) else 0, r), 5) for r in Z])
    Zs = Zs + 0.004 * np.abs(offs)[None, :]          # mild preference for staying near the mapped line
    o = offs[np.argmin(Zs, axis=1)].astype(float)
    o = moving(medfilt(o, 21), 9)                     # ~100 m median + 45 m mean: no lateral jumps
    thx = xy[:, 0] + o * ty
    thy = xy[:, 1] - o * tx
    # thalweg elevation = min within +-4 m of the smoothed position
    zz = []
    for i in range(len(xy)):
        oo = np.arange(o[i] - 4, o[i] + 5)
        v = bilinear(arr, tr, xy[i, 0] + oo * ty[i], xy[i, 1] - oo * tx[i])
        zz.append(np.nanmin(v) if np.any(~np.isnan(v)) else np.nan)
    thz = np.array(zz)
    slope_all = H.theil_sen(chain, thz)
    flipped = False
    if slope_all > 0:      # OSM way drawn upstream -> flip so that chainage increases downstream
        flipped = True
        chain, thx, thy, thz, tx, ty, cov = -chain[::-1], thx[::-1], thy[::-1], thz[::-1], -tx[::-1], -ty[::-1], cov[::-1]
    return dict(chain=chain, x=thx, y=thy, z=thz, tx=tx, ty=ty, covered=cov, flipped=flipped)


def slopes(th, c):
    out = {}
    for half in (250, 500, 750):
        m = np.abs(th["chain"] - c) <= half
        out[half] = -H.theil_sen(th["chain"][m], th["z"][m]) if m.sum() > 8 else np.nan
    return out


# ----------------------------------------------------------------------------- one section
def cut_section(th, c, arr, tr, half_len):
    i = int(np.argmin(np.abs(th["chain"] - c)))
    cx, cy, tx, ty = th["x"][i], th["y"][i], th["tx"][i], th["ty"][i]
    s = np.arange(-half_len, half_len + 1.0)
    # right-hand normal looking downstream = (ty, -tx); station grows from left to right
    X = cx + s * ty
    Y = cy - s * tx
    z = bilinear(arr, tr, X, Y)
    return dict(i=i, s=s + half_len, x=X, y=Y, z=z, centre_idx=int(half_len))


def analyse(sec, n_tab, S, S_lo, S_hi, cfg, caps=(None, None, None)):
    s, z = sec["s"], sec["z"]
    zf = z.copy()
    # fill short NaN gaps for the walk
    if np.isnan(zf).any():
        ok = ~np.isnan(zf)
        if ok.sum() < 20:
            return None
        zf = np.interp(s, s[ok], zf[ok], left=np.nan, right=np.nan)
    zs = H.smooth(zf, 3)
    i0 = H.find_thalweg(s, zs, sec["centre_idx"], cfg.get("thalweg_search", 15))
    if np.isnan(zs[max(0, i0 - 30):i0 + 31]).any():
        return None        # section centre at the edge of the DTM window
    bs = int(cfg.get("bank_search", 200))
    md = cfg.get("min_depth", 0.6)
    wr = cfg.get("wall_ratio", 2.0)
    iL, fL = H.find_bank(zs, i0, -1, bs, min_depth=md, wall_ratio=wr)
    iR, fR = H.find_bank(zs, i0, +1, bs, min_depth=md, wall_ratio=wr)
    if "bank_left_off" in cfg:
        iL, fL = int(i0 - cfg["bank_left_off"]), True
    if "bank_right_off" in cfg:
        iR, fR = int(i0 + cfg["bank_right_off"]), True
    z0 = float(zs[i0])
    zL, zR = float(zs[iL]), float(zs[iR])
    zb = min(zL, zR)
    n_ch, n_lo, n_hi = n_tab
    nf, nf_lo, nf_hi = H.N_FLOODPLAIN
    r = H.discharge(s, zs, zb, iL, iR, n_ch, nf, S, caps[0])
    q_lo = H.discharge(s, zs, zb, iL, iR, n_hi, nf_hi, S_lo, caps[1])["Q"]
    q_hi = H.discharge(s, zs, zb, iL, iR, n_lo, nf_lo, S_hi, caps[2])["Q"]
    A, T = r["A_channel"], r["T_channel"]
    V = r["Q"] / A if A > 0 else 0.0
    Fr = V / np.sqrt(H.G * A / T) if A > 0 and T > 0 else 0.0
    hs, qs, unb = H.rating(s, zs, z0, zb, iL, iR, n_ch, nf, S, fr_cap=caps[0])
    _, qs_lo, _ = H.rating(s, zs, z0, zb, iL, iR, n_hi, nf_hi, S_lo, fr_cap=caps[1])
    _, qs_hi, _ = H.rating(s, zs, z0, zb, iL, iR, n_lo, nf_lo, S_hi, fr_cap=caps[2])
    depth = zb - z0
    over = hs > depth + 1e-9
    # flat bottom (water surface seen by the LiDAR?)
    flat = float(np.sum(zs[iL:iR + 1] <= z0 + 0.05))
    return dict(
        i0=i0, iL=iL, iR=iR, z0=z0, zL=zL, zR=zR, zb=zb, found_L=bool(fL), found_R=bool(fR),
        froude_capped=bool(r["froude_capped"]), depth=depth, top_width=T, area=A, q_bf=r["Q"], q_bf_lo=q_lo, q_bf_hi=q_hi, V=V, Fr=Fr,
        h=hs, q=qs, q_lo=qs_lo, q_hi=qs_hi, unbounded=unb,
        over_q=np.concatenate([[r["Q"]], qs[over]]), over_h=np.concatenate([[0.0], hs[over] - depth]),
        over_q_lo=np.concatenate([[q_lo], qs_lo[over]]), over_q_hi=np.concatenate([[q_hi], qs_hi[over]]),
        flat_bottom_m=flat, zs=zs,
    )


# ----------------------------------------------------------------------------- bridges / choice
def bridge_chainages(pid, th):
    path = os.path.join(OSM_DIR, pid + "_bridges.json")
    if not os.path.exists(path):
        return None
    from common import ll2utm
    line = LineString(np.c_[th["x"], th["y"]])
    cum = np.r_[0, np.cumsum(np.hypot(np.diff(th["x"]), np.diff(th["y"])))]
    out = []
    with open(path, encoding="utf-8") as f:
        els = json.load(f)["elements"]
    for e in els:
        g = e.get("geometry") or []
        if len(g) < 2:
            continue
        b = LineString([ll2utm(q["lon"], q["lat"]) for q in g])
        if b.distance(line) < 12:
            from shapely.ops import nearest_points
            p = nearest_points(line, b)[0]
            d = line.project(p)
            out.append(float(np.interp(d, cum, th["chain"])))
    return sorted(out)


def choose_chainage(th, bridges, cfg, dx):
    if "offset_m" in cfg:
        return float(cfg["offset_m"]), "manual offset"
    lo, hi = th["chain"][0] + dx + 60, th["chain"][-1] - dx - 60
    cands = sorted(np.arange(-500, 501, 25.0), key=lambda c: (abs(c), c))
    for c in cands:
        if c < lo or c > hi:
            continue
        ok = True
        for cc in (c - dx, c, c + dx):
            i = int(np.argmin(np.abs(th["chain"] - cc)))
            if th["covered"][i]:
                ok = False
            if bridges and min(abs(cc - b) for b in bridges) < 35:
                ok = False
        if ok:
            return float(c), "auto (nearest position clear of bridges/covered reaches)"
    return float(np.clip(0.0, lo, hi)), "auto (no fully clear position found)"


# ----------------------------------------------------------------------------- CAUMAX
_cm = {}


def caumax(x, y, radius=1000.0):
    res = {}
    best = None
    for T in (2, 5, 10, 25, 100, 500):
        if T not in _cm:
            d = rasterio.open(os.path.join(MISC, "caumax", "data", f"q{T}.tif"))
            _cm[T] = (d.read(1), d.transform)
        a, tr = _cm[T]
        if best is None:
            r, c = int((tr.f - y) / 500), int((x - tr.c) / 500)
            k = int(radius // 500) + 1
            cand = []
            for rr in range(r - k, r + k + 1):
                for cc in range(c - k, c + k + 1):
                    v = a[rr, cc]
                    if v > 0:
                        px, py = tr.c + (cc + 0.5) * 500, tr.f - (rr + 0.5) * 500
                        cand.append((np.hypot(px - x, py - y), rr, cc, float(v)))
            cand = [q for q in cand if q[0] <= radius]
            if not cand:
                return None
            best = min(cand)
            res["_pixel_dist_m"] = round(best[0])
            res["_all_T2_candidates"] = sorted({round(q[3]) for q in cand})
        res[f"T{T}"] = round(float(a[best[1], best[2]]), 1)
    return res


# ----------------------------------------------------------------------------- ortho
def ortho(pid, cx, cy, half=300, px=900):
    path = os.path.join(ORTHO, f"{pid}_{int(cx)}_{int(cy)}.jpg")
    if not os.path.exists(path):
        url = ("https://www.ign.es/wms-inspire/pnoa-ma?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
               "&LAYERS=OI.OrthoimageCoverage&STYLES=&CRS=EPSG:25830"
               f"&BBOX={cx - half},{cy - half},{cx + half},{cy + half}&WIDTH={px}&HEIGHT={px}&FORMAT=image/jpeg")
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                with open(path, "wb") as f:
                    f.write(r.content)
            else:
                return None
        except Exception:  # noqa
            return None
    return plt.imread(path)


def hillshade(a):
    gy, gx = np.gradient(np.nan_to_num(a, nan=np.nanmean(a)))
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy) * 2)
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    return np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)


# ----------------------------------------------------------------------------- per point
def run_point(p, cfg, anchor):
    pid = p["id"]
    with rasterio.open(os.path.join(DTM_DIR, pid + ".tif")) as d:
        arr, tr = d.read(1), d.transform
        src = json.loads(d.tags().get("riua_source", "{}"))
    snap_w = int(cfg.get("snap_w", 40))
    half_len = int(cfg.get("half_len", 450))
    dx = float(cfg.get("dx_sec", 125))
    th = trace_thalweg(anchor, arr, tr, snap_w)
    bridges = bridge_chainages(pid, th)
    c0, how = choose_chainage(th, bridges, cfg, dx)
    # reach-scale slope (several km, includes drop structures) from longprofile.py
    S_reach = None
    lp_path = os.path.join(MISC, "longprofile.json")
    if os.path.exists(lp_path):
        with open(lp_path, encoding="utf-8") as f:
            lp = json.load(f).get(pid, {}).get("points", [])
        if len(lp) >= 3:
            sign = -1.0 if th["flipped"] else 1.0
            xs = [sign * q["chain"] for q in lp] + [0.0]
            i_a = int(np.argmin(np.abs(th["chain"])))
            zs_ = [q["z_min"] for q in lp] + [float(np.nanmin(th["z"][max(0, i_a - 5):i_a + 6]))]
            S_reach = float(-np.polyfit(xs, zs_, 1)[0])

    def slope_at(c):
        sl_ = slopes(th, c)
        S_ = sl_[500] if np.isfinite(sl_[500]) else np.nanmedian(list(sl_.values()))
        notes_ = []
        if S_reach is not None:
            sl_["reach_km"] = S_reach
        if cfg.get("slope_from") == "reach" and S_reach and S_reach > 0:
            S_ = S_reach
            notes_.append("slope = reach-average over +-3 km (includes drop structures), not the local thalweg slope")
        if not np.isfinite(S_) or S_ < 2e-4:
            notes_.append(f"measured thalweg slope {S_:.5f} below 0.0002: floored at 0.0002 (flat reach / water surface)")
            S_ = 2e-4
        if "slope" in cfg:
            S_ = float(cfg["slope"])
            notes_.append("slope set manually: " + cfg.get("slope_note", ""))
        # range: 500 m and 750 m local windows and the reach slope (the 250 m window is too noisy)
        vals = [v for k, v in sl_.items() if k != 250 and v is not None and np.isfinite(v) and v > 0] + [S_]
        if cfg.get("slope_from") == "reach":
            vals = [v for v in vals if v >= 0.5 * S_]   # stepped channel: tread slopes are not the energy slope
        vals = [v for v in vals if 0.33 * S_ <= v <= 3.0 * S_]
        return S_, max(1e-4, min(min(vals), 0.8 * S_)), max(max(vals), 1.2 * S_), sl_, notes_

    S, S_lo, S_hi, sl, notes = slope_at(c0)
    lining = cfg.get("lining", "natural")
    n_tab = H.N_TABLE[lining]
    if "n" in cfg:
        n_tab = tuple(cfg["n"])
    caps = H.FR_CAP[lining]
    # reach scan: bankfull capacity every 50 m over +-500 m around the first-guess position
    # (automatic banks, same n and S), skipping covered reaches and positions within 35 m of a bridge
    cfg_auto = {k: v for k, v in cfg.items() if not k.startswith("bank_")}

    def clear(c):
        if c < th["chain"][0] + 20 or c > th["chain"][-1] - 20:
            return False
        i = int(np.argmin(np.abs(th["chain"] - c)))
        return not (th["covered"][i] or (bridges and min(abs(c - b) for b in bridges) < 35))

    scan_c, scan_q = [], []
    for c in np.arange(c0 - 500, c0 + 501, 50.0):
        if not clear(c):
            continue
        r_ = analyse(cut_section(th, c, arr, tr, half_len), n_tab, S, S_lo, S_hi, cfg_auto, caps)
        if r_ is not None:
            scan_c.append(float(c))
            scan_q.append(round(float(r_["q_bf"]), 1))
    if "offset_m" not in cfg and len(scan_q) >= 5:
        # representative section = the clear position within +-300 m whose capacity is closest to the reach median
        med = float(np.median(scan_q))
        cand = [(abs(np.log(max(q, 0.1) / max(med, 0.1))), abs(c - c0), c) for c, q in zip(scan_c, scan_q)
                if abs(c - c0) <= 300 and clear(c - dx) and clear(c + dx)]
        if cand:
            c_new = min(cand)[2]
            if c_new != c0:
                c0 = c_new
                S, S_lo, S_hi, sl, notes = slope_at(c0)
            how = "auto: clear of bridges/covered reaches, capacity closest to the median of the +-500 m reach scan"
    secs, res = [], []
    for c in (c0 - dx, c0, c0 + dx):
        sec = cut_section(th, c, arr, tr, half_len)
        r = analyse(sec, n_tab, S, S_lo, S_hi, cfg if c == c0 else {k: v for k, v in cfg.items() if not k.startswith("bank_")}, caps)
        secs.append(sec)
        res.append(r)
    main = res[1]
    if main is None:
        raise RuntimeError("chosen section has no data")
    sec = secs[1]

    def ll(i, s_=sec):
        lon, lat = utm2ll(s_["x"][i], s_["y"][i])
        return [round(lat, 6), round(lon, 6)]

    qs3 = [r["q_bf"] if r else None for r in res]
    covered_here = bool(th["covered"][sec["i"]])
    scan_c = [c - c0 for c in scan_c]
    cm = caumax(sec["x"][main["i0"]], sec["y"][main["i0"]])
    out = {
        "id": pid, "stream": p["stream"], "town": p["town"],
        "start_source": anchor["start_source"],
        "section_choice": how, "section_chainage_from_anchor_m": c0,
        "section_left_end": ll(0), "section_right_end": ll(len(sec["s"]) - 1),
        "thalweg": {"lat": ll(main["i0"])[0], "lon": ll(main["i0"])[1], "elev_m": round(main["z0"], 2)},
        "bank_left": {"lat": ll(main["iL"])[0], "lon": ll(main["iL"])[1], "elev_m": round(main["zL"], 2), "crest_found": main["found_L"]},
        "bank_right": {"lat": ll(main["iR"])[0], "lon": ll(main["iR"])[1], "elev_m": round(main["zR"], 2), "crest_found": main["found_R"]},
        "bankfull_elev_m": round(main["zb"], 2), "bankfull_side": "left" if main["zL"] <= main["zR"] else "right",
        "bankfull_depth_m": round(main["depth"], 2), "bankfull_top_width_m": round(main["top_width"], 1),
        "bankfull_area_m2": round(main["area"], 1),
        "slope": round(S, 5), "slope_range": [round(S_lo, 5), round(S_hi, 5)],
        "slope_windows": {str(k): (round(v, 5) if v is not None and np.isfinite(v) else None) for k, v in sl.items()},
        "manning_n_channel": n_tab[0], "manning_n_channel_range": [n_tab[1], n_tab[2]],
        "manning_n_floodplain": H.N_FLOODPLAIN[0], "manning_n_floodplain_range": list(H.N_FLOODPLAIN[1:]),
        "lining": lining,
        "q_bankfull": round(main["q_bf"], 1), "q_bankfull_range": [round(main["q_bf_lo"], 1), round(main["q_bf_hi"], 1)],
        "q_bankfull_3_sections": [round(q, 1) if q is not None else None for q in qs3],
        "reach_scan": {"chainage_from_section_m": scan_c, "q_bankfull": scan_q,
                       "median": round(float(np.median(scan_q)), 1) if scan_q else None,
                       "p25": round(float(np.percentile(scan_q, 25)), 1) if scan_q else None,
                       "min": min(scan_q) if scan_q else None,
                       "note": "automatic bankfull capacity of sections every 50 m over +-500 m; the weakest sections (p25/min) are where overflow starts first"},
        "bankfull_velocity_ms": round(main["V"], 2), "bankfull_froude": round(main["Fr"], 2),
        "froude_cap": caps[0], "froude_capped_at_bankfull": main["froude_capped"],
        "rating": {"h_m": np.round(main["h"], 2), "q_m3s": np.round(main["q"], 1),
                   "q_low_m3s": np.round(main["q_lo"], 1), "q_high_m3s": np.round(main["q_hi"], 1),
                   "floodplain_unbounded_in_section": main["unbounded"].astype(bool).tolist()},
        "h_over_bank": {"q_m3s": np.round(main["over_q"], 1), "h_over_bank_m": np.round(main["over_h"], 2),
                        "q_low_m3s": np.round(main["over_q_lo"], 1), "q_high_m3s": np.round(main["over_q_hi"], 1)},
        "caumax": cm,
        "dtm": {"source": src.get("name"), "resolution_m": 1.0, "licence": src.get("licence"),
                "sheets": [s_["hoja"] for s_ in src.get("sheets", [])]},
        "perennial": bool(cfg.get("perennial", False)),
        "flat_bottom_width_m": main["flat_bottom_m"],
        "covered_reach_at_section": covered_here,
        "bridges_chainage_m": bridges,
        "auto_notes": notes,
    }
    jdump(out, os.path.join(RESULTS, pid + ".json"))
    prof = {"id": pid, "crs_note": "station in m from the left end looking downstream; elevation m (ICV 1 m LiDAR DTM, orthometric)",
            "sections": []}
    for k, (s_, r_) in enumerate(zip(secs, res)):
        e0, e1 = utm2ll(s_["x"][0], s_["y"][0]), utm2ll(s_["x"][-1], s_["y"][-1])
        prof["sections"].append({
            "role": LBL_SEC[k], "chainage_from_chosen_m": (k - 1) * dx,
            "left_end": [round(e0[1], 6), round(e0[0], 6)], "right_end": [round(e1[1], 6), round(e1[0], 6)],
            "station_m": s_["s"], "elevation_m": np.round(s_["z"], 2),
            "thalweg_station_m": float(s_["s"][r_["i0"]]) if r_ else None,
            "bank_left_station_m": float(s_["s"][r_["iL"]]) if r_ else None,
            "bank_right_station_m": float(s_["s"][r_["iR"]]) if r_ else None,
            "bankfull_elev_m": round(r_["zb"], 2) if r_ else None,
            "q_bankfull": round(r_["q_bf"], 1) if r_ else None,
        })
    jdump(prof, os.path.join(PROFILES, pid + ".json"))
    plot_point(pid, p, arr, tr, anchor, th, secs, res, out, bridges)
    return out


def plot_point(pid, p, arr, tr, anchor, th, secs, res, out, bridges):
    fig = plt.figure(figsize=(19, 6.4), dpi=80)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    sec, main = secs[1], res[1]
    cx, cy = sec["x"][main["i0"]], sec["y"][main["i0"]]
    # --- DTM
    ax = fig.add_subplot(gs[0])
    half = 450
    c = int((cx - tr.c)), int((tr.f - cy))
    r0, r1 = max(0, c[1] - half), min(arr.shape[0], c[1] + half)
    c0, c1 = max(0, c[0] - half), min(arr.shape[1], c[0] + half)
    sub = arr[r0:r1, c0:c1]
    ext = [tr.c + c0, tr.c + c1, tr.f - r1, tr.f - r0]
    zc = main["z0"]
    ax.imshow(sub, extent=ext, cmap="viridis", vmin=zc - 1, vmax=zc + max(14, 2.2 * main["depth"]))
    ax.imshow(hillshade(sub), extent=ext, cmap="gray", alpha=0.45)
    axy = np.array(anchor["xy"])
    ax.plot(axy[:, 0], axy[:, 1], color="white", lw=0.8, ls="--", label="OSM line")
    ax.plot(th["x"], th["y"], color="#ffd166", lw=1.0, label="DTM thalweg")
    for k, s_ in enumerate(secs):
        ax.plot(s_["x"], s_["y"], color=C_SEC[k], lw=1.6 if k == 1 else 1.0)
    if bridges:
        for b in bridges:
            i = int(np.argmin(np.abs(th["chain"] - b)))
            ax.plot(th["x"][i], th["y"][i], marker="s", color="white", ms=6, mec="black")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title("DTM 1 m (thalweg-1 .. +14 m), 900 m", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])
    # --- ortho
    ax = fig.add_subplot(gs[1])
    img = ortho(pid, round(cx), round(cy))
    if img is not None:
        ax.imshow(img, extent=[round(cx) - 300, round(cx) + 300, round(cy) - 300, round(cy) + 300])
        for k, s_ in enumerate(secs):
            ax.plot(s_["x"], s_["y"], color=C_SEC[k], lw=1.2 if k == 1 else 0.8, alpha=0.9)
        for key, i in (("L", main["iL"]), ("R", main["iR"])):
            ax.plot(sec["x"][i], sec["y"][i], "o", color="white", mec="black", ms=5)
        ax.set_xlim(round(cx) - 300, round(cx) + 300); ax.set_ylim(round(cy) - 300, round(cy) + 300)
    ax.set_title("PNOA orthophoto (IGN), 600 m; dots = bank crests", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # --- profiles
    ax = fig.add_subplot(gs[2])
    span = max(60, 1.6 * max(main["i0"] - main["iL"], main["iR"] - main["i0"]) + 30)
    for k, (s_, r_) in enumerate(zip(secs, res)):
        if r_ is None:
            continue
        x = s_["s"] - s_["s"][r_["i0"]]
        dz = 0.0
        ax.plot(x, s_["z"] + dz, color=C_SEC[k], lw=2.0 if k == 1 else 1.0,
                label=f'{LBL_SEC[k]}: Qbf {r_["q_bf"]:.0f} m3/s, depth {r_["depth"]:.1f} m, width {r_["top_width"]:.0f} m')
        ax.plot([x[r_["iL"]], x[r_["iR"]]], [s_["z"][r_["iL"]], s_["z"][r_["iR"]]], "o", color=C_SEC[k], ms=7 if k == 1 else 4, mec="black")
    x = sec["s"] - sec["s"][main["i0"]]
    ax.hlines(main["zb"], x[main["iL"]], x[main["iR"]], color=C_SEC[1], lw=1.0, ls="--")
    ax.set_xlim(-span, span)
    m = (x > -span) & (x < span)
    zz = sec["z"][m]
    ax.set_ylim(np.nanmin(zz) - 1, max(np.nanmax(zz), main["zb"] + 3) + 1)
    ax.grid(alpha=0.25)
    ax.set_xlabel("distance from thalweg (m), left bank <- looking downstream -> right bank")
    ax.set_ylabel("elevation (m)")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_title(f'{pid} | {p["stream"]} @ {p["town"]}\nQbf {out["q_bankfull"]:.0f} [{out["q_bankfull_range"][0]:.0f}-{out["q_bankfull_range"][1]:.0f}] m3/s  '
                 f'S {out["slope"]:.4f}  n {out["manning_n_channel"]} ({out["lining"]})  V {out["bankfull_velocity_ms"]} m/s Fr {out["bankfull_froude"]}  '
                 f'CAUMAX T10 {out["caumax"]["T10"] if out["caumax"] else "-"} T100 {out["caumax"]["T100"] if out["caumax"] else "-"}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, pid + ".png"))
    plt.close(fig)


def main():
    only = set(sys.argv[1:])
    cfg = load_config()
    with open(os.path.join(MISC, "anchors.json"), encoding="utf-8") as f:
        anchors = json.load(f)
    for p in load_seed():
        pid = p["id"]
        if only and pid not in only:
            continue
        if pid not in anchors or not os.path.exists(os.path.join(DTM_DIR, pid + ".tif")):
            if only:
                print(pid, "missing anchor or DTM")
            continue
        try:
            o = run_point(p, cfg[pid], anchors[pid])
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            print(pid, "FAILED", repr(e)[:200])
            continue
        cm = o["caumax"] or {}
        print(f'{pid:22s} c0 {o["section_chainage_from_anchor_m"]:5.0f} Qbf {o["q_bankfull"]:7.0f} [{o["q_bankfull_range"][0]:6.0f}-{o["q_bankfull_range"][1]:6.0f}] '
              f'3sec {[int(q) if q is not None else None for q in o["q_bankfull_3_sections"]]} d {o["bankfull_depth_m"]:4.1f} W {o["bankfull_top_width_m"]:5.0f} S {o["slope"]:.4f} '
              f'V {o["bankfull_velocity_ms"]:.1f} Fr {o["bankfull_froude"]:.1f} banks {o["bank_left"]["crest_found"]}/{o["bank_right"]["crest_found"]} '
              f'scan p25/med {o["reach_scan"]["p25"]}/{o["reach_scan"]["median"]} '
              f'T10 {cm.get("T10")} T100 {cm.get("T100")} T500 {cm.get("T500")}', flush=True)


if __name__ == "__main__":
    main()
