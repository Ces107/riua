"""Product B: extreme-value analysis of ERA5 precipitation, 0.25 deg box, annual maxima 1950-2025.

    python climate/build/build_eva.py am      -> scratch/c1-climate/parts/annmax.npz   (annual maxima 1940-2025)
    python climate/build/build_eva.py fit     -> scratch/c1-climate/parts/eva.npz      (GEV, return levels, CI, diagnostics)

Durations: sliding 1, 6, 12, 24, 48, 72 h (hourly steps; a window belongs to the year of its END) and 'day' = fixed
07-07 UTC day (the Spanish pluviometric day), needed to compare like with like against rain-gauge daily maxima.

Three estimators are computed at every grid point:
  site      GEV by L-moments of the point's own 76 annual maxima (lmoments3).
  rshape    regional shape: L-skewness averaged over the region of influence (5 x 5 points, +-0.5 deg, clipped at the
            box edge) -> one shape; location and scale from the point's own l1, l2.            [stored as the product]
  iflood    full Hosking-Wallis index flood: regional L-CV and L-skewness -> growth curve x point mean.
Diagnostics: Hosking-Wallis heterogeneity H1 and goodness of fit Z(GEV) per region (kappa simulations),
split-sample stability of the 100-year level (1950-87 vs 1988-2025), sensitivity to the period, Mann-Kendall trend.
Confidence intervals (90 %): parametric bootstrap. 'site': independent samples from the fitted GEV. 'rshape':
samples for the whole region drawn through a Gaussian copula with the empirical inter-site correlation of the
annual maxima (Hosking & Wallis 1997, sect. 6.4), so that the strong dependence between neighbouring grid
points does not shrink the intervals.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import stats

import lmom
from common import PARTS, hour_index, precip, rolling_sum, safe_savez

DURS = (1, 6, 12, 24, 48, 72)
DUR_NAMES = ["1", "6", "12", "24", "48", "72", "day"]
YEARS_ALL = np.arange(1940, 2026)
FIT0, FIT1 = 1950, 2025
T_LIST = np.array([2, 5, 10, 25, 50, 100, 500], float)
ROI = 2                     # half-width of the region of influence in grid points (2 -> 5 x 5)
N_BOOT = 500
N_SIM_H = 300
SEED = 20261001


def build_am():
    pr, lat, lon = precip()
    x = np.array(pr)                                  # int16 [T, 15, 15], 343 MB
    x[x == 32767] = 0                                 # first 7 h of 1940 and the tail after 2026-09-24
    nd = len(DURS) + 1
    am = np.zeros((nd, len(YEARS_ALL), len(lat), len(lon)), np.float32)
    when = np.zeros(am.shape, np.int32)               # hour index (since 1940-01-01 00) of the window END
    bounds = [hour_index(y) for y in range(YEARS_ALL[0], YEARS_ALL[-1] + 2)]
    for i in range(len(lat)):
        row = x[:, i, :].astype(np.int32)
        for d, w in enumerate(DURS):
            rs = rolling_sum(row, w)
            for k in range(len(YEARS_ALL)):
                seg = rs[bounds[k]:bounds[k + 1]]
                arg = seg.argmax(axis=0)
                am[d, k, i] = seg[arg, np.arange(seg.shape[1])] / 10.0
                when[d, k, i] = bounds[k] + arg
        rs = rolling_sum(row, 24)
        for k in range(len(YEARS_ALL)):
            idx = np.arange(bounds[k] + 7, bounds[k + 1], 24)     # windows ending 07 UTC inside the year
            seg = rs[idx]
            arg = seg.argmax(axis=0)
            am[nd - 1, k, i] = seg[arg, np.arange(seg.shape[1])] / 10.0
            when[nd - 1, k, i] = idx[arg]
        print(f"row {i} lat {lat[i]}: 24 h annual max mean {am[3, :, i].mean():.1f} mm", flush=True)
    safe_savez(PARTS / "annmax.npz", am=am, when=when, years=YEARS_ALL, lat=lat, lon=lon,
                        durations=np.array(DUR_NAMES))
    print("saved annmax; ratio sliding-24h / fixed-day annual maxima (mean over points and years):",
          float((am[3] / np.maximum(am[nd - 1], 0.1)).mean()))


# ----------------------------------------------------------------------------------------------------------
def roi_indices(ny, nx, half=ROI):
    out = []
    for i in range(ny):
        for j in range(nx):
            ii, jj = np.meshgrid(np.arange(max(0, i - half), min(ny, i + half + 1)),
                                 np.arange(max(0, j - half), min(nx, j + half + 1)), indexing="ij")
            out.append((ii.ravel() * nx + jj.ravel()))
    return out


def regional_fits(x, regions):
    """x [n, P] annual maxima. Returns dict of per-point parameter arrays for the three estimators."""
    l1, l2, t3, t4 = lmom.sample_lmom(x, axis=0)
    t = l2 / l1
    k_site = lmom.gev_k_from_t3(t3)
    t3r = np.array([t3[r].mean() for r in regions])
    t4r = np.array([t4[r].mean() for r in regions])
    tr = np.array([t[r].mean() for r in regions])
    k_r = lmom.gev_k_from_t3(t3r)
    site = lmom.gev_from_lmom(l1, l2, k_site)
    rshape = lmom.gev_from_lmom(l1, l2, k_r)
    g_loc, g_scale, g_xi = lmom.gev_from_lmom(1.0, tr, k_r)      # growth curve of the standardised variable
    iflood = (g_loc * l1, g_scale * l1, g_xi)
    return dict(site=site, rshape=rshape, iflood=iflood, l1=l1, l2=l2, t=t, t3=t3, t4=t4, tr=tr, t3r=t3r, t4r=t4r)


def heterogeneity(x, regions, fits, rng):
    """Hosking-Wallis H1 and Z(GEV) for the region of every point (independent kappa simulations)."""
    from lmoments3 import distr

    n, P = x.shape
    H = np.full(P, np.nan)
    Z = np.full(P, np.nan)
    for p, r in enumerate(regions):
        m = len(r)
        tR, t3R, t4R = fits["tr"][p], fits["t3r"][p], fits["t4r"][p]
        V = np.sqrt(((fits["t"][r] - tR) ** 2).mean())
        try:
            kp = distr.kap.lmom_fit(lmom_ratios=[1.0, tR, t3R, t4R])
            sim = distr.kap.ppf(rng.uniform(size=(N_SIM_H, n, m)), **kp)
        except Exception:  # noqa: BLE001 - kappa not fittable (t4 too large): generalised logistic, as HW recommend
            try:
                gp = distr.glo.lmom_fit(lmom_ratios=[1.0, tR, t3R])
                sim = distr.glo.ppf(rng.uniform(size=(N_SIM_H, n, m)), **gp)
            except Exception:  # noqa: BLE001
                continue
        s1, s2, s3, s4 = lmom.sample_lmom(sim, axis=1)            # [N_SIM, m]
        ts = s2 / s1
        Vs = np.sqrt(((ts - ts.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
        H[p] = (V - Vs.mean()) / Vs.std()
        t4s = s4.mean(axis=1)
        k_r = lmom.gev_k_from_t3(t3R)
        Z[p] = (lmom.gev_t4_from_k(k_r) - t4R + (t4s.mean() - t4R)) / t4s.std()
    return H, Z


def boot_site(x, rng, nboot=1000):
    n, P = x.shape
    loc, scale, xi = lmom.gev_fit(x, axis=0)
    u = rng.uniform(size=(nboot, n, P))
    sim = lmom.gev_quantile(u, loc, scale, xi)
    bl, bs, bx = lmom.gev_fit(sim, axis=1)                        # [nboot, P]
    rl = lmom.return_level(T_LIST[:, None, None], bl, bs, bx)     # [T, nboot, P]
    return np.nanpercentile(rl, [5, 95], axis=1)                  # [2, T, P]


def boot_rshape(x, regions, fits, rng, nboot=N_BOOT):
    """Parametric bootstrap of the regional-shape estimator with inter-site dependence (Gaussian copula)."""
    n, P = x.shape
    loc, scale, xi = fits["rshape"]
    ranks = stats.rankdata(x, axis=0)
    z = stats.norm.ppf((ranks - 0.375) / (n + 0.25))              # normal scores (Blom)
    out = np.zeros((2, len(T_LIST), P))
    shape_ci = np.zeros((2, P))
    for p, r in enumerate(regions):
        m = len(r)
        C = np.corrcoef(z[:, r], rowvar=False)
        w, v = np.linalg.eigh(C)
        L = v * np.sqrt(np.clip(w, 1e-6, None))                   # C ~ L L^T (PSD-safe)
        g = rng.standard_normal((nboot, n, m)) @ L.T
        g /= np.sqrt(np.clip(np.diag(L @ L.T), 1e-9, None))
        u = np.clip(stats.norm.cdf(g), 1e-9, 1 - 1e-9)
        sim = lmom.gev_quantile(u, loc[r], scale[r], xi[r])       # every site with its own fitted marginal
        s1, s2, s3, _s4 = lmom.sample_lmom(sim, axis=1)           # [nboot, m]
        k_r = lmom.gev_k_from_t3(s3.mean(axis=1))
        c = int(np.nonzero(r == p)[0][0])
        bl, bs, bx = lmom.gev_from_lmom(s1[:, c], s2[:, c], k_r)
        rl = lmom.return_level(T_LIST[:, None], bl, bs, bx)       # [T, nboot]
        out[:, :, p] = np.nanpercentile(rl, [5, 95], axis=1)
        shape_ci[:, p] = np.percentile(bx, [5, 95])
    return out, shape_ci


def fit_all():
    z = np.load(PARTS / "annmax.npz")
    am, years, lat, lon = z["am"], z["years"], z["lat"], z["lon"]
    ny, nx = len(lat), len(lon)
    P = ny * nx
    regions = roi_indices(ny, nx)
    regions3 = roi_indices(ny, nx, 1)
    sel = (years >= FIT0) & (years <= FIT1)
    nd = am.shape[0]
    rng = np.random.default_rng(SEED)
    keys = ["loc", "scale", "shape"]
    res = {f"gev_{k}": np.zeros((nd, ny, nx), np.float32) for k in keys}
    res.update({f"gev_site_{k}": np.zeros((nd, ny, nx), np.float32) for k in keys})
    res.update({f"gev_iflood_{k}": np.zeros((nd, ny, nx), np.float32) for k in keys})
    for k in ("rl", "rl_lo", "rl_hi", "rl_site", "rl_site_lo", "rl_site_hi"):
        res[k] = np.zeros((nd, len(T_LIST), ny, nx), np.float32)
    res["gev_shape_lo"] = np.zeros((nd, ny, nx), np.float32)
    res["gev_shape_hi"] = np.zeros((nd, ny, nx), np.float32)
    res["hw_H1"] = np.zeros((nd, ny, nx), np.float32)
    res["hw_Zgev"] = np.zeros((nd, ny, nx), np.float32)
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    say(f"GEV fits on annual maxima {FIT0}-{FIT1} (n = {int(sel.sum())}), {P} grid points, ROI {2 * ROI + 1}x{2 * ROI + 1}")
    for d in range(nd):
        t0 = time.time()
        x = am[d][sel].reshape(sel.sum(), P).astype(np.float64)
        fits = regional_fits(x, regions)
        # cross-check of the vectorised at-site fit against lmoments3 at every point
        from lmoments3 import distr
        dmax = 0.0
        for p in range(0, P, 7):
            lp = distr.gev.lmom_fit(x[:, p])
            dmax = max(dmax, abs(-lp["c"] - fits["site"][2][p]), abs(lp["loc"] - fits["site"][0][p]) / lp["loc"])
        for name, key in (("", "rshape"), ("site_", "site"), ("iflood_", "iflood")):
            for k, v in zip(keys, fits[key]):
                res[f"gev_{name}{k}"][d] = v.reshape(ny, nx)
        rl = lmom.return_level(T_LIST[:, None], *fits["rshape"])
        rl_site = lmom.return_level(T_LIST[:, None], *fits["site"])
        rl_if = lmom.return_level(T_LIST[:, None], *fits["iflood"])
        res["rl"][d] = rl.reshape(-1, ny, nx)
        res["rl_site"][d] = rl_site.reshape(-1, ny, nx)
        ci_site = boot_site(x, rng)
        res["rl_site_lo"][d], res["rl_site_hi"][d] = ci_site[0].reshape(-1, ny, nx), ci_site[1].reshape(-1, ny, nx)
        ci, sci = boot_rshape(x, regions, fits, rng)
        res["rl_lo"][d], res["rl_hi"][d] = ci[0].reshape(-1, ny, nx), ci[1].reshape(-1, ny, nx)
        res["gev_shape_lo"][d], res["gev_shape_hi"][d] = sci[0].reshape(ny, nx), sci[1].reshape(ny, nx)
        H, Z = heterogeneity(x, regions, fits, rng)
        res["hw_H1"][d], res["hw_Zgev"][d] = H.reshape(ny, nx), Z.reshape(ny, nx)

        # --- diagnostics -------------------------------------------------------------------------------
        xi_s, xi_r = fits["site"][2], fits["rshape"][2]
        xi_dom = -float(lmom.gev_k_from_t3(fits["t3"].mean()))
        xi_r3 = -lmom.gev_k_from_t3(np.array([fits["t3"][r].mean() for r in regions3]))
        i100 = int(np.nonzero(T_LIST == 100)[0][0])
        half = sel.sum() // 2
        fa, fb = regional_fits(x[:half], regions), regional_fits(x[half:], regions)
        stab = {}
        for key in ("site", "rshape", "iflood"):
            ra = lmom.return_level(100.0, *fa[key])
            rb = lmom.return_level(100.0, *fb[key])
            stab[key] = float(np.sqrt(np.mean(np.log(ra / rb) ** 2)))
            stab[key + "_bias"] = float(np.mean(np.log(rb / ra)))
        wid_site = float(np.median(ci_site[1][i100] / ci_site[0][i100]))
        wid_r = float(np.median(ci[1][i100] / ci[0][i100]))
        mk = np.array([stats.kendalltau(np.arange(x.shape[0]), x[:, p]) for p in range(P)])
        say(f"--- duration {DUR_NAMES[d]} h ({time.time() - t0:.0f}s); lmoments3 cross-check max diff {dmax:.2e}")
        say(f"  mean annual max {fits['l1'].mean():.1f} mm (min {fits['l1'].min():.1f}, max {fits['l1'].max():.1f}); "
            f"L-CV {fits['t'].mean():.3f}; L-skew {fits['t3'].mean():.3f}; L-kurt {fits['t4'].mean():.3f}")
        say(f"  shape xi  at-site: mean {xi_s.mean():+.3f} sd {xi_s.std():.3f} range {xi_s.min():+.2f}..{xi_s.max():+.2f} | "
            f"ROI 3x3: sd {xi_r3.std():.3f} | ROI 5x5: mean {xi_r.mean():+.3f} sd {xi_r.std():.3f} "
            f"range {xi_r.min():+.2f}..{xi_r.max():+.2f} | whole box: {xi_dom:+.3f}")
        say(f"  bootstrap 90% CI of xi (ROI 5x5, dependent): median half-width {np.median((sci[1] - sci[0]) / 2):.3f}")
        say(f"  T100 (mm) median over points: site {np.median(rl_site[i100]):.1f}, rshape {np.median(rl[i100]):.1f}, "
            f"iflood {np.median(rl_if[i100]):.1f}; |ln(site/rshape)| mean {np.abs(np.log(rl_site[i100] / rl[i100])).mean():.3f}")
        say(f"  split-sample RMS ln(T100 first half / second half): site {stab['site']:.3f}, rshape {stab['rshape']:.3f}, "
            f"iflood {stab['iflood']:.3f}; mean ln(second/first) rshape {stab['rshape_bias']:+.3f}")
        say(f"  CI width T100 (hi/lo, median): site (independent bootstrap) {wid_site:.2f}, rshape (dependent bootstrap) {wid_r:.2f}")
        say(f"  HW heterogeneity H1 over the 5x5 regions: median {np.nanmedian(H):.2f}, share H<1 {np.nanmean(H < 1):.2f}, "
            f"1<=H<2 {np.nanmean((H >= 1) & (H < 2)):.2f}, H>=2 {np.nanmean(H >= 2):.2f} | Z(GEV) median {np.nanmedian(Z):+.2f}, "
            f"share |Z|<=1.64 {np.nanmean(np.abs(Z) <= 1.64):.2f}")
        say(f"  Mann-Kendall on annual maxima: tau mean {mk[:, 0].mean():+.3f}, share p<0.05 {np.mean(mk[:, 1] < 0.05):.2f} "
            f"(positive {np.mean((mk[:, 1] < 0.05) & (mk[:, 0] > 0)):.2f})")
        for a, b in ((1940, 2025), (1979, 2025), (1950, 1987), (1988, 2025)):
            s2 = (years >= a) & (years <= b)
            f2 = regional_fits(am[d][s2].reshape(s2.sum(), P).astype(np.float64), regions)
            r2 = lmom.return_level(100.0, *f2["rshape"])
            say(f"  period {a}-{b}: median T100 {np.median(r2):.1f} mm, ratio to {FIT0}-{FIT1} median {np.median(r2 / rl[i100]):.3f} "
                f"(p10 {np.percentile(r2 / rl[i100], 10):.2f}, p90 {np.percentile(r2 / rl[i100], 90):.2f}); xi mean {f2['rshape'][2].mean():+.3f}")
        safe_savez(PARTS / "eva.npz", lat=lat, lon=lon, eva_T=T_LIST, eva_durations=np.array(DUR_NAMES),
                            fit_years=np.array([FIT0, FIT1]), **res)
    (PARTS / "eva_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("saved", PARTS / "eva.npz")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "am":
        build_am()
    elif what == "fit":
        fit_all()
