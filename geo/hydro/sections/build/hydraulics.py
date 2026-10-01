"""Pure-numpy hydraulics for the cross-section pipeline (no I/O).

Conventions
-----------
* A profile is (s, z): station s in metres from the LEFT end of the section looking
  downstream, 1 m spacing; z ground elevation (m, orthometric) from the 1 m LiDAR DTM.
* Stage h is measured above the thalweg (lowest point of the main channel).
* Manning, divided-channel method:  Q = sum_i (1/n_i) A_i R_i^(2/3) S^(1/2),  R_i = A_i / P_i
  with three sub-sections: left overbank | main channel (between the two bank crests) | right overbank.
  The vertical water interfaces between sub-sections are not counted in P.
"""
import numpy as np

G = 9.81

# Manning n: (central, low, high)
N_TABLE = {
    "concrete": (0.020, 0.015, 0.025),      # smooth concrete lined trapezoid/rectangle
    "masonry": (0.030, 0.024, 0.038),       # masonry / riprap / gabion banks, gravel or earth bed
    "natural": (0.040, 0.030, 0.055),       # dry gravel-bed ravine, scattered vegetation
    "vegetated": (0.055, 0.040, 0.080),     # cane (Arundo donax) / reed choked bed
    "river": (0.040, 0.032, 0.055),         # perennial lowland river, vegetated banks
}
# Froude cap of the main-channel flow (central, low, high) for unlined channels; lined: no cap
FR_CAP = {"natural": (1.0, 0.8, 1.3), "vegetated": (1.0, 0.8, 1.3), "river": (1.0, 0.8, 1.3),
          "masonry": (None, None, None), "concrete": (None, None, None)}
N_FLOODPLAIN = (0.10, 0.08, 0.12)           # urban fabric / orchards (bare-earth DTM, buildings removed)


def smooth(z, k=3):
    if k <= 1:
        return z.copy()
    pad = k // 2
    zp = np.pad(z, pad, mode="edge")
    return np.array([np.nanmedian(zp[i:i + k]) for i in range(len(z))])


def find_thalweg(s, z, centre_idx, search):
    lo, hi = max(0, centre_idx - search), min(len(z), centre_idx + search + 1)
    zs = smooth(z, 5)
    seg = zs[lo:hi]
    if np.all(np.isnan(seg)):
        return centre_idx
    return lo + int(np.nanargmin(seg))


def find_bank(z, i0, direction, max_dist, min_depth=0.6, tol=0.30, d_min=12, d_frac=0.4, fp_slope=0.08,
              wall_ratio=2.0):
    """Walk outward from the thalweg index i0 (direction -1 = left, +1 = right).

    The bank crest is the first point at least `min_depth` above the thalweg from which the
    ground does not rise more than max(tol, fp_slope * D) over the next D metres
    (D = max(d_min, d_frac * distance from the thalweg)): i.e. the break of slope from the
    channel wall (steeper than ~8 %) to the floodplain / street level, or the top of a levee. Returns (index, found). If no break is found within max_dist the
    highest point reached is returned with found=False (valley side: flow stays confined).
    """
    n = len(z)
    z0 = z[i0]
    run_max = -np.inf
    j = i0
    cands = []
    for k in range(1, max_dist + 1):
        j = i0 + direction * k
        if j < 1 or j > n - 2 or np.isnan(z[j]):
            j -= direction
            break
        if z[j] >= run_max:
            run_max = z[j]
            if z[j] - z0 >= min_depth:
                d = int(max(d_min, d_frac * k))
                a, b = (j + 1, min(n, j + 1 + d)) if direction > 0 else (max(0, j - d), j)
                ahead = z[a:b]
                ahead = ahead[~np.isnan(ahead)]
                if len(ahead) >= 3 and ahead.max() <= z[j] + max(tol, fp_slope * d):
                    # crest = highest point of the next D metres if it is a small levee top,
                    # otherwise (floodplain rising gently away) the slope break itself
                    if z[j] < ahead.max() <= z[j] + tol:
                        best = a + int(np.nanargmax(z[a:b]))
                    else:
                        best = j
                    if not cands or z[best] > z[cands[-1]] + 0.2:
                        cands.append(best)
    lo, hi = (i0, j + 1) if direction > 0 else (j, i0 + 1)
    i_top = lo + int(np.nanargmax(z[lo:hi]))
    if not cands:
        return i_top, False
    # Several slope breaks (bars, benches, terraces): keep the first one, but move outward to a
    # later break when it is a real confinement, i.e. it at least doubles the depth (wall_ratio).
    cur, found = cands[0], True
    for c in cands[1:]:
        if z[c] - z0 >= wall_ratio * (z[cur] - z0):
            cur = c
    if z[i_top] - z0 >= wall_ratio * (z[cur] - z0) and abs(i_top - i0) > abs(cur - i0):
        cur, found = i_top, False      # still rising at the end of the search: valley side
    return cur, found


def zone_AP(s, z, wse, i_from, i_to):
    """Area, wetted perimeter and top width of ground below wse between stations [i_from, i_to]."""
    if i_to <= i_from:
        return 0.0, 0.0, 0.0
    zz = z[i_from:i_to + 1]
    ss = s[i_from:i_to + 1]
    d0 = wse - zz[:-1]
    d1 = wse - zz[1:]
    dx = np.diff(ss)
    A = P = T = 0.0
    both = (d0 > 0) & (d1 > 0)
    A += float(np.sum(0.5 * (d0[both] + d1[both]) * dx[both]))
    P += float(np.sum(np.hypot(dx[both], (zz[1:] - zz[:-1])[both])))
    T += float(np.sum(dx[both]))
    part = (d0 > 0) ^ (d1 > 0)
    if part.any():
        dd = np.where(d0[part] > 0, d0[part], d1[part])
        frac = dd / np.abs(d0[part] - d1[part])
        A += float(np.sum(0.5 * dd * frac * dx[part]))
        P += float(np.sum(np.hypot(frac * dx[part], dd)))
        T += float(np.sum(frac * dx[part]))
    return A, P, T


def connected_extent(z, wse, start, direction):
    """Index reached by water spreading outward from `start` while ground < wse."""
    n = len(z)
    j = start
    while 0 < j < n - 1:
        nj = j + direction
        if np.isnan(z[nj]) or z[nj] >= wse:
            return nj, False
        j = nj
    return j, True   # reached the end of the section: unbounded


def discharge(s, z, wse, iL, iR, n_ch, n_fp, S, fr_cap=None):
    """Divided-channel Manning discharge at water-surface elevation wse.
    fr_cap: if given, the main-channel discharge is limited to fr_cap * A * sqrt(g A / T)
    (natural steep channels do not sustain Froude numbers much above 1: Grant 1997, Jarrett 1984).
    Returns dict with Q and the pieces."""
    out = {}
    A, P, T = zone_AP(s, z, wse, iL, iR)
    q_ch = (A / n_ch) * (A / P) ** (2.0 / 3.0) * np.sqrt(S) if A > 0 and P > 0 else 0.0
    capped = False
    if fr_cap and A > 0 and T > 0:
        q_c = fr_cap * A * np.sqrt(G * A / T)
        if q_ch > q_c:
            q_ch, capped = q_c, True
    out["froude_capped"] = capped
    q = q_ch
    unb = False
    a_fp = t_fp = 0.0
    q_fp = 0.0
    for side, i_bank, direction in (("L", iL, -1), ("R", iR, +1)):
        if wse > z[i_bank]:
            j, u = connected_extent(z, wse, i_bank, direction)
            unb = unb or u
            a, b = (j, i_bank) if direction < 0 else (i_bank, j)
            Af, Pf, Tf = zone_AP(s, z, wse, a, b)
            if Af > 0 and Pf > 0:
                q_fp += (Af / n_fp) * (Af / Pf) ** (2.0 / 3.0) * np.sqrt(S)
                a_fp += Af
                t_fp += Tf
    out.update(Q=q_ch + q_fp, Q_channel=q_ch, Q_floodplain=q_fp, A_channel=A, P_channel=P, T_channel=T,
               A_floodplain=a_fp, T_floodplain=t_fp, unbounded=unb)
    return out


def rating(s, z, z0, zb, iL, iR, n_ch, n_fp, S, extra=4.0, dh=0.1, fr_cap=None):
    depth_bf = zb - z0
    hmax = np.ceil((depth_bf + extra) / dh) * dh
    hs = np.round(np.arange(0.0, hmax + dh / 2, dh), 3)
    qs, unb = [], []
    for h in hs:
        r = discharge(s, z, z0 + h, iL, iR, n_ch, n_fp, S, fr_cap)
        qs.append(r["Q"])
        unb.append(r["unbounded"])
    qs = np.maximum.accumulate(np.array(qs))
    return hs, qs, np.array(unb)


def theil_sen(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    m = np.abs(dx) > 20.0     # only pairs more than 20 m apart
    return float(np.median((y[j] - y[i])[m] / dx[m]))
