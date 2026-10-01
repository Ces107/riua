"""Verification of ``backend/riua/static/climate_era5.npz`` on real data (prints, no test framework).

    cd <repo> && python climate/build/verify.py

Needs only the npz, ``backend/riua/core/efi.py``, ``backend/riua/core/eva.py`` and ``climate/efi_reviewed.py``.
If r2's cached ECMWF ensemble of 27-Oct-2024 00Z is present (hindcast/cache/ens) the real forecast is scored too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "climate"))
from riua.core import efi as efi_orig  # noqa: E402
from riua.core import eva  # noqa: E402
import efi_reviewed as efi_new  # noqa: E402

Z = np.load(ROOT / "backend" / "riua" / "static" / "climate_era5.npz")
lat, lon = Z["lat"], Z["lon"]
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def cell(la, lo):
    return int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))


def clim_q(window, doy):
    return efi_new.clim_at(doy, Z["clim_doy"], Z[f"pr_q{window}"]) / float(Z["pr_q_scale"])


def clim_max(window, doy):
    return efi_new.clim_at(doy, Z["clim_doy"], Z[f"pr_max{window}"][:, None])[0] / float(Z["pr_q_scale"])


def fmt_T(t):
    return ">9999" if (not np.isfinite(t) or t >= 9999) else (f"{t:.0f}" if t >= 10 else f"{t:.1f}")


def main():
    meta = json.loads(str(Z["meta_json"]))
    print("=" * 100)
    print("1. FILE, GRID, RECORD")
    print("=" * 100)
    f = ROOT / "backend" / "riua" / "static" / "climate_era5.npz"
    print(f"{f}  {f.stat().st_size / 1e6:.2f} MB, {len(Z.files)} arrays")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  precipitation grid: lat {lat[0]}..{lat[-1]} ({len(lat)}), lon {lon[0]}..{lon[-1]} ({len(lon)}), step {lat[1] - lat[0]}")
    print(f"  TCWV grid: lat {Z['tcwv_lat'][0]}..{Z['tcwv_lat'][-1]} ({len(Z['tcwv_lat'])}), lon {Z['tcwv_lon'][0]}..{Z['tcwv_lon'][-1]} ({len(Z['tcwv_lon'])})")
    print(f"  annual maxima: {Z['annmax_years'][0]}..{Z['annmax_years'][-1]} ({len(Z['annmax_years'])} yr), GEV fitted on {Z['eva_fit_years'][0]}..{Z['eva_fit_years'][1]}")
    print(f"  clim_p identical to backend/riua/core/efi.py CLIM_P: {np.allclose(Z['clim_p'], efi_orig.CLIM_P)} ({len(Z['clim_p'])} levels); "
          f"day-of-year nodes: {Z['clim_doy'][0]}, {Z['clim_doy'][1]}, ..., {Z['clim_doy'][-1]} ({len(Z['clim_doy'])})")

    print()
    print("=" * 100)
    print("2. CLIMATOLOGICAL SANITY VALUES (ERA5 1991-2020)")
    print("=" * 100)
    mon = Z["pr_monthly_mean"]
    for name, la, lo, ref in [("Valencia", 39.48, -0.37, "obs 1991-2020 normal ~ 450-480 mm"), ("Alicante", 38.37, -0.49, "obs ~ 280-310 mm"),
                              ("Castello", 39.99, -0.04, "obs ~ 440-470 mm"), ("Oliva/Safor", 38.92, -0.12, "obs ~ 700-800 mm"), ("Turis", 39.39, -0.71, "")]:
        i, j = cell(la, lo)
        m = mon[:, i, j]
        print(f"  {name:12s} point {lat[i]:.2f},{lon[j]:+.2f}: annual {m.sum():5.0f} mm; wettest month {MONTHS[int(m.argmax())]} ({m.max():.0f} mm), "
              f"driest {MONTHS[int(m.argmin())]} ({m.min():.0f} mm)   [{ref}]")
    bm = mon.mean(axis=(1, 2))
    print("  box-mean monthly totals (mm): " + " ".join(f"{MONTHS[k]} {bm[k]:.0f}" for k in range(12)) + f" | annual {bm.sum():.0f}")
    print(f"  wettest month of the box mean: {MONTHS[int(bm.argmax())]}")
    tm, ts = Z["tcwv_mean"], Z["tcwv_std"]
    ti, tj = int(np.argmin(np.abs(Z["tcwv_lat"] - 39.5))), int(np.argmin(np.abs(Z["tcwv_lon"] + 0.25)))
    for label, a, b in (("January", 0, 31), ("August", 213, 244), ("October", 274, 305)):
        print(f"  TCWV {label:8s}: Valencia point mean {tm[a:b, ti, tj].mean():.1f} kg/m2, std {ts[a:b, ti, tj].mean():.1f}; "
              f"box mean {tm[a:b].mean():.1f} (land+sea)")
    print("  (expected: August ~ 28-32 at the coast, January ~ 12-15)")
    for w in (12, 24, 48, 72):
        q = clim_q(w, 302)  # 29 October
        i, j = cell(39.39, -0.71)
        k50, k90, k99 = (int(np.argmin(np.abs(Z["clim_p"] - p))) for p in (0.5, 0.9, 0.99))
        print(f"  {w:>2}-h climate at Turis on 29 Oct: p50 {q[k50, i, j]:.1f}, p90 {q[k90, i, j]:.1f}, p99 {q[k99, i, j]:.1f}, p99.9 {q[-1, i, j]:.1f}, "
              f"max {clim_max(w, 302)[i, j]:.1f} mm; share of dry (0 mm) quantile levels {np.mean(q[:, i, j] == 0):.2f}")

    print()
    print("=" * 100)
    print("3. EXTREME VALUES (ERA5 space) AND riua.core.eva")
    print("=" * 100)
    T = Z["eva_T"]
    print("  return levels 24 h (mm) [90 % CI] at T = " + ", ".join(f"{t:.0f}" for t in T))
    for name, la, lo in [("Valencia", 39.48, -0.37), ("Turis", 39.39, -0.71), ("Oliva", 38.92, -0.12), ("Alicante", 38.37, -0.49), ("Orihuela", 38.08, -0.95)]:
        vals = [float(eva.return_level(t, 24, la, lo)) for t in T]
        ci = [eva.return_level_ci(float(t), 24, la, lo) for t in T]
        print(f"   {name:9s}: " + "  ".join(f"{v:.0f} [{float(c[0]):.0f}-{float(c[1]):.0f}]" for v, c in zip(vals, ci))
              + f" | xi {float(eva.gev_params(24, la, lo)[2]):+.2f}")
    i, j = cell(39.39, -0.71)
    print("  consistency at a grid point (Turis point): eva.return_level(100, d) vs stored rl: " +
          ", ".join(f"{d:.0f} h {float(eva.return_level(100, float(d), lat[i], lon[j])):.1f}/{Z['rl'][k, 5, i, j]:.1f}" for k, d in enumerate(Z["eva_durations_h"])))
    rt = [float(eva.return_period(float(eva.return_level(t, 24, 39.39, -0.71)), 24, 39.39, -0.71)) for t in (5, 50, 500)]
    print(f"  round trip return_period(return_level(T)) for T = 5, 50, 500: {rt[0]:.2f}, {rt[1]:.2f}, {rt[2]:.2f}")
    print(f"  intermediate duration (log interpolation): T100 3 h {float(eva.return_level(100, 3, 39.39, -0.71)):.1f} mm, 18 h {float(eva.return_level(100, 18, 39.39, -0.71)):.1f} mm")
    amounts = np.array([20, 50, 80, 100, 150, 200.0])
    print("  return period (yr) of a 24-h ERA5-space amount at Turis: " + ", ".join(f"{a:.0f} mm -> {fmt_T(float(eva.return_period(a, 24, 39.39, -0.71)))}" for a in amounts))
    print("  model-space equivalents of gauge amounts, 12-h window (mm): " + "; ".join(
        f"{name}: " + "/".join(f"{float(eva.model_equivalent(t, 12, la, lo)):.0f}" for t in (60, 100, 180, 300))
        for name, la, lo in [("Valencia", 39.48, -0.37), ("Turis", 39.39, -0.71), ("Oliva", 38.92, -0.12), ("Alicante", 38.37, -0.49)]) + "  (for 60/100/180/300 mm)")

    print()
    print("=" * 100)
    print("4. EVENT TABLE (part C): ERA5-space amounts and return periods")
    print("=" * 100)
    ev = json.loads(str(Z["events_json"]))
    print(f"  {'event':38s} {'observed':52s} | 24 h: point mm (T) | worst 3x3 mm (T) | box mm (T) | 12 h box T | 1 h box T")
    for e in ev["historic"]:
        r24, r12, r1 = e["era5"]["24"], e["era5"]["12"], e["era5"]["1"]
        print(f"  {e['label']:38s} {e['observed'][:52]:52s} | {r24['cell_mm']:6.1f} ({fmt_T(r24['cell_T']):>5s}) | {r24['nbh_mm']:6.1f} ({fmt_T(r24['nbh_T']):>5s}) "
              f"| {r24['box_mm']:6.1f} ({fmt_T(r24['box_T']):>5s}) | {fmt_T(r12['box_T']):>5s} | {fmt_T(r1['box_T']):>5s}")
    print("  catalogue 2023-2026:")
    for e in ev["catalogue"]:
        r24, r12 = e["era5"]["24"], e["era5"]["12"]
        print(f"  {e['label']:28s} 24 h worst 3x3 {r24['nbh_mm']:6.1f} mm T={fmt_T(r24['nbh_T']):>5s} | box {r24['box_mm']:6.1f} mm T={fmt_T(r24['box_T']):>5s} "
              f"| 12 h box T={fmt_T(r12['box_T']):>5s} | cells T24>=10: {r24['n_cells_T10']:3d} | {e['outcome']}")
    sc = ev["scan_24h"]
    print("  how often the box maximum of the 24-h return period reaches a level (events per year, 1950-2026): " +
          ", ".join(f"T>={k}: {v:.2f}" for k, v in sc["rates_per_year"].items()))
    print("  top 15 events of the 24-h scan:")
    for t in sc["top"][:15]:
        print(f"   {t['rank']:3d} {t['centre'][:13]} T={fmt_T(t['T']):>5s} {t['mm']:6.1f} mm at {t['lat']:.2f},{t['lon']:+.2f} n(T>=10)={t['n_cells_T10']:3d}  {t['known']}")

    print()
    print("=" * 100)
    print("5. EFI / SOT FOR 29-OCT-2024")
    print("=" * 100)
    doy = 302  # 29 October on the 366-day calendar (0 = 1 Jan, 59 = 29 Feb)
    rng = np.random.default_rng(1)
    for w, key in ((24, "demo_20241029_pr24"), (12, "demo_20241029_pr12"), (72, "demo_20241029_pr72")):
        field = Z[key].astype(np.float64)
        q, qmax = clim_q(w, doy), clim_max(w, doy)
        ens = field[None] * rng.lognormal(0.0, 0.05, size=(51, 1, 1))        # synthetic ensemble = ERA5 itself (+-5 %)
        e0 = efi_orig.efi(ens, q); s0 = efi_orig.sot(ens, q)
        e1 = efi_new.efi(ens, q, clim_max=qmax); s1 = efi_new.sot(ens, q)
        print(f"  synthetic ensemble = ERA5 {w}-h field ending {'30 Oct 00' if w < 72 else '31 Oct 00'} UTC (box max {field.max():.1f} mm):")
        for name, la, lo in [("Turis", 39.39, -0.71), ("Utiel", 39.57, -1.20), ("Valencia", 39.48, -0.37), ("Alicante", 38.37, -0.49), ("Morella", 40.62, -0.10)]:
            i, j = cell(la, lo)
            print(f"     {name:9s} {field[i, j]:6.1f} mm | efi.py: EFI {e0[i, j]:+.2f} SOT {s0[i, j]:+6.2f} | efi_reviewed: EFI {e1[i, j]:+.2f} SOT {s1[i, j]:+6.2f}")
        print(f"     box: efi.py EFI max {e0.max():+.3f}, share of points > 0.8: {np.mean(e0 > 0.8):.2f} | reviewed EFI max {np.nanmax(e1):+.3f}, share > 0.8: {np.nanmean(e1 > 0.8):.2f}")
    # behaviour on trivial forecasts (why the dry-climate treatment matters)
    q, qmax = clim_q(24, doy), clim_max(24, doy)
    q_aug = clim_q(24, 222)
    for label, val, qq in (("all members 0.0 mm (29 Oct)", 0.0, q), ("all members 0.5 mm (29 Oct)", 0.5, q), ("all members 2 mm (29 Oct)", 2.0, q),
                           ("all members 0.5 mm (10 Aug)", 0.5, q_aug), ("all members 5 mm (10 Aug)", 5.0, q_aug)):
        ens = np.full((51,) + q.shape[1:], val)
        e0, s0 = efi_orig.efi(ens, qq), efi_orig.sot(ens, qq)
        e1, s1 = efi_new.efi(ens, qq), efi_new.sot(ens, qq)
        print(f"  {label:28s}: efi.py EFI box mean {e0.mean():+.2f}, SOT {np.mean(s0):+.2f} | reviewed EFI {np.nanmean(e1):+.2f} "
              f"(NaN share {np.mean(np.isnan(e1)):.2f}), SOT {np.nanmean(s1) if np.isfinite(s1).any() else float('nan'):+.2f} (NaN share {np.mean(np.isnan(s1)):.2f})")
    ens = np.full((51,) + q.shape[1:], 1e4)
    print(f"  every member above the climate maximum: efi.py EFI = {efi_orig.efi(ens, q).mean():+.4f} (should be +1) | reviewed = {np.nanmean(efi_new.efi(ens, q, clim_max=qmax)):+.4f}")

    f_ens = ROOT / "hindcast" / "cache" / "ens" / "ifsens_tp_2024102700.npz"
    if f_ens.exists():
        zz = np.load(f_ens, allow_pickle=True)
        steps = list(zz["steps"])
        if 48 in steps and 72 in steps and np.allclose(zz["lat"], lat) and np.allclose(zz["lon"], lon):
            acc = zz["data"][:, steps.index(72)] - zz["data"][:, steps.index(48)]       # 29 Oct 00 -> 30 Oct 00 UTC
            q, qmax = clim_q(24, doy), clim_max(24, doy)
            e0, s0 = efi_orig.efi(acc, q), efi_orig.sot(acc, q)
            e1, s1 = efi_new.efi(acc, q, clim_max=qmax), efi_new.sot(acc, q)
            print(f"  REAL ECMWF ENS run 27-Oct-2024 00Z, +48..+72 h ({acc.shape[0]} members, r2's cache):")
            for name, la, lo in [("Turis", 39.39, -0.71), ("Utiel", 39.57, -1.20), ("Valencia", 39.48, -0.37), ("Alicante", 38.37, -0.49)]:
                i, j = cell(la, lo)
                m = acc[:, i, j]
                T50 = float(eva.return_period(float(np.median(m)), 24, lat[i], lon[j]))
                T90 = float(eva.return_period(float(np.quantile(m, 0.9)), 24, lat[i], lon[j]))
                print(f"     {name:9s} members mean {m.mean():6.1f} p90 {np.quantile(m, 0.9):6.1f} max {m.max():6.1f} mm | efi.py EFI {e0[i, j]:+.2f} SOT {s0[i, j]:+.2f} "
                      f"| reviewed EFI {e1[i, j]:+.2f} SOT {s1[i, j]:+.2f} | ERA5-space T of median {fmt_T(T50)} yr, of p90 {fmt_T(T90)} yr")
            rp = eva.return_period(np.quantile(acc, 0.9, axis=0), 24, lat[:, None], lon[None, :])
            print(f"     box: reviewed EFI max {np.nanmax(e1):+.2f}; points with EFI > 0.8: {int(np.nansum(e1 > 0.8))}; points whose ensemble p90 exceeds the ERA5-space T=10 / T=50 / T=100 level: "
                  f"{int((rp >= 10).sum())} / {int((rp >= 50).sum())} / {int((rp >= 100).sum())}")
    else:
        print("  (no cached real ensemble found; skipped)")

    print()
    print("=" * 100)
    print("6. TCWV STANDARDISED ANOMALY, 29-OCT-2024 12 UTC")
    print("=" * 100)
    tc = Z["demo_20241029_tcwv_12utc"]
    anom = (tc - tm[doy]) / ts[doy]
    print(f"  Valencia point: TCWV {tc[ti, tj]:.1f} kg/m2, climate mean {tm[doy, ti, tj]:.1f}, std {ts[doy, ti, tj]:.1f} -> {anom[ti, tj]:+.2f} sigma; "
          f"box max {tc.max():.1f} kg/m2, anomaly max {anom.max():+.2f} sigma, box mean {anom.mean():+.2f} sigma")


if __name__ == "__main__":
    main()
