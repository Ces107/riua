# snapshot.json — the contract between the pipeline and the web page

Produced by `backend/riua/product.py`. A real one is at `scratch/out/snapshot.json` (and `explain-<hz>.bin` next to it).

```
v                 1
generated         "2026-10-01T10:05Z"  (UTC)
params_version    string
grid              {lon0:-2.4, lat0:37.6, d:0.05, nx:64, ny:68}     cell (j,i): j south→north, i west→east,
                  centre = (lat0+(j+.5)d, lon0+(i+.5)d), flat index c = j*nx+i
mask              base64 of a bit array (np.packbits, MSB first) over the nx*ny cells: 1 = cell is shown
n_cells           N = number of 1-bits. Every per-cell array below lists ONLY masked cells, in increasing c.
thresholds        {zones:{code:{"1h":[y,o,r],"12h":[y,o,r]}}, extreme:{x1h,x12h,...}, source:{document,url,...}}
horizons          {now:{...}, mid:{...}, long:{...}}
explain           {now:{M,F,N,layout}, ...}  header of explain-<hz>.bin
obs               {hours, last_hour, o1, o12, o24}   base64 uint8[N] rain already fallen: last hour (cell max), last 12 h, last 24 h (cell mean);
                  mm = (v/8)^2; hours = analysed hours kept, last_hour = end of the last one (UTC ISO)
gauges            [{id,name,lat,lon,source,t_utc,p_1h,p_12h,p_24h}]
rivers            [{id,name,river,lat,lon,source,t_utc,level_m,flow_m3s,thr_low,thr_mid,thr_high,...}]  (keys vary by source)
warnings          [{zone_code,zone_name,level:"yellow|orange|red",onset,expires,text,params,...}]  official AEMET rain warnings
drivers           optional: ingredients block (see backend/riua/diagnostics), may be missing
reservoirs        optional: reservoirs, live state and filling / spill forecast (see the end of this file); may be missing
sources           [{id,label,ok,runs?,n?,error?,...}]
notes             [string]
timing_s          {...}
```

Each horizon:

```
frames   [{t0,t1 (UTC ISO, window (t0,t1]), ok:bool, has_1h:bool, method:"dressing"|"emos",
           cal:{c1,c1_from12,c12,rho} | null}]   F frames. method says how P was computed for the frame:
           "emos" = the calibrated distribution `cal` (see the end of this file); "dressing" = scenario dressing
           with sigma, bias and fam below (cal is null). The page's audit view follows it.
tau      {"2":p,"3":p,"4":p,"5":p}    probability needed to issue each level
sigma, bias          kernel of the 12-h term of the dressing for this horizon (log-normal spread, multiplicative bias)
sigma1h, bias1h      kernel of the 1-h term (equal to sigma, bias where params does not set them)
sigma_obs            spread left when the whole 12-h amount is already measured (0.15)
obs12_factor         bias applied to the measured share of the 12-h amount (0.88)
level_cap            null, or the highest level the horizon may publish (3 for `long`): level = min(level, cap)
fam      {family:{s1h,s12h}}          representativeness factors applied to the 1-h and 12-h amounts of each
                                      family of scenarios (radar, cp, regional, global, ens, eps)
cells    level  base64 uint8[F*N]      0 = no data, 1..5 (after level_cap)
         p      base64 uint8[4*F*N]    P(>=2),P(>=3),P(>=4),P(>=5); P = v/200, rounded DOWN, so that
                                       `p >= tau` on the byte agrees with the level
         o12    base64 uint8[F*N]      mm already measured inside the 12-h amount of the frame (largest over the
                                       scenarios); mm = (v/8)^2
         e1,e12 base64 uint8[2*F*N]    amount the thresholds are applied to: [median, 1-in-10 high] over the scenarios of
                                       the largest 1-h and 12-h amounts within the neighbourhood (scaled by fam, or the
                                       calibrated distribution when method = "emos"); mm = (v/8)^2
         acc    base64 uint8[2*F*N]    rain expected AT the cell inside each frame, no neighbourhood: [median, 1-in-10 high]
                                       over the scenarios; mm = (v/8)^2   (product.py::frame_accumulation)
         acc_total base64 uint8[2*N]   the same over the whole period covered by the frames (first t0 .. last t1)
         m1,q1,m12,q12  base64 uint8[F*N]  raw ensemble mean / 90th percentile of the scenarios (mm = (v/8)^2;
                                       255 = n/a, real amounts stop at 254 = 1008 mm)
members  [{name,family,model,run,step_h,neigh_km,w_raw,w:[per frame normalised weight]}]   M scenarios
         The members of an ensemble are named "<ensemble> mNN <rest>" ("ENS m07 · 01/10 12Z",
         "Radar STEPS m03 → AROME-HD 1,3 km · 01/10 18Z"); the page shows those that share the name without
         the member number as one row. Radar members carry the family and model of the run they blend into.
basins   level[F][B], p[4][F][B], own12[2][F][B] (mm, median/p90 12-h mean rain over the unit),
         up12[2][F][B] (same over unit+upstream), q[2][F][B] (m3/s/km2), upstream[F][B] (0..1 share of scenarios where upstream rain dominates)
         B and order = geo/out/basins.geojson features order (property idx)
points   (only when the control-point network exists) level[F][P], p[4][F][P], qpeak[2][F][P] m3/s,
         hover[2][F][P] m above bank (negative = below), t[T] ISO, q[3][T][P] (p10,p50,p90 hydrograph m3/s),
         rp[2][F][P] | null  return period in years of the median / p90 peak (CAUMAX quantiles, log-log;
                              1 = below the 2-year flood, capped at 1000; null without quantiles),
         cap[P] (optional)   channel capacity m3/s; when absent the page uses web/geo/points.json
         qpeak and hover are [median, p90]; p is P(>=2..5), and p[2] (level 4) is the probability of overflow
         where the point has a capacity
         P and order = geo/hydro/catchments/out/control_points.json (= web/geo/points.json)
```

Frames: `now` = 6 hourly frames from the top of the current hour; `mid` = 14 three-hour frames from +6 h; `long` = 6 UTC days (day+2 … day+7).

A horizon may be missing (`long` when the ECMWF ensemble is not available): the page disables its tab.

explain-<hz>.bin (lazy-loaded for the audit view), head in `explain[hz]` = `{M, M1, has1:[bool × M], F, N, layout}`:
uint8 `a1[M1,F,N]` — only the scenarios with `has1`, in order — then `a12[M,F,N]`, `p1[4,F,N]`, `p12[4,F,N]`: per-scenario
1-h and 12-h amounts (after neighbourhood; 255 = not available, amounts stop at 254) and the two marginal exceedance
probabilities before the union. Length `(M1 + M + 8)·F·N`. A scenario without `has1` has no 1-h amount.
(Before 2026-10-02 the head had no `M1` and the file was `a1[M,F,N]`, `a12[M,F,N]`, …, length `(2M + 8)·F·N`; the page reads both.)

Scenario dressing (`method: "dressing"`, `risk.py::dressed_probabilities`, `web/js/maths.js::dressScenario`), per scenario
with amounts `a1`, `a12`, family factors `s1h`, `s12h`, thresholds `T1`, `T12` of the level and `o12` of the cell and frame:

```
A12 = s12h·a12        phi = min(1, o12 / A12)            measured share of the 12-h amount (0 if A12 = 0)
b12 = phi·obs12_factor + (1 − phi)·bias                  sg12 = max(sigma·(1 − phi), min(sigma, sigma_obs))
p12 = Φ(ln(b12·A12 / T12) / sg12)                        p1 = Φ(ln(bias1h·s1h·a1 / T1) / sigma1h)   (0 if a1 n/a or 0)
P(>= L) = Σ w·max(p1, p12) / Σ w   over the scenarios with weight and an amount;   then minimum.accumulate over the levels
level = highest L with P(>= L) >= tau[L], then min(level, level_cap)
```

With `method: "emos"` the probability of a level is NOT a count of scenarios: it comes from the calibrated distribution `cal` (censored shifted gamma): with `m`,`q` the ensemble mean and 90th percentile, `mu = a0 + a1*m + a2*q`, `sigma = b0*sqrt(mu) + b1*(q-m)`, shape `k = mu²/sigma²`, scale `θ = sigma²/mu`, `P(amount >= T) = Q(k, (T - delta)/θ)` (upper regularised incomplete gamma). The two marginals (1 h, 12 h) are joined with a Gaussian copula of correlation `rho`. Level = highest L with `P(>=L) >= tau[L]`.

## reservoirs (optional, `core/reservoirs.py`, `sources/reservoirs.py`, static data `geo/hydro/dams/dams.json`)

```
dams[D]   {id, name, river, lat, lon, source: saih_chj|saih_segura,
           kind: "gated" | "free" | "operating limit below the spillway" | "unknown",
           cap   hm3 at the spill level (lip of a free spillway / top of the gates): the 100 % of every percentage below
           res   hm3 flood-reserve limit this month (or null)        top  hm3 with the water at the dam crest (or null)
           qd    design spillway capacity m3/s (or null)             thr  [low, mid, high] outflow thresholds m3/s (SAIH Júcar;
                                                                         elsewhere 0.6 / 1.5 / 3 x A^0.6)
           area  km2 natural catchment     up [ids of the dams directly above]     next  id of the next dam below (or null)
           now   null | {t (UTC ISO), v hm3, pct (% of cap), level m a.s.l. (SAIH Júcar only), qin, qout, qriv m3/s
                         (inflow, total outflow, outflow to the river; SAIH Júcar only), rate hm3/h (mean of the last 3 h),
                         sv [[hour ISO, % of cap], ...] measured volume of the last 12 h}}
points    {control point id: [[dam index, share of the point's natural catchment above the dam, lag h dam -> point, cuts 0/1]]}
          only the dams with no other modelled dam between them and the point. cuts = 1: the catchment above the dam is
          NOT in the point's own discharge (control_points.json scope 1), so what the dam lets out adds to it.
horizons  {now|mid|long: per horizon, D = order of dams, F = frames of that horizon
           level[F][D]       0 (not computed: no live volume, no capacity, or < 50 % of the catchment in the rain grid) or 1..5
           p[4][F][D]        P(level >= 2..5)
           p_reserve[F][D]   P(inside the flood reserve)   p_spill[F][D] P(spilling)   p_design[F][D] P(spill >= design capacity)
           qin[2][F][D], qout[2][F][D], qspill[2][F][D]   median / p90 of the frame's peak inflow, outflow to the river, spill (m3/s)
           pct[3][F][D]      p10 / p50 / p90 volume at the end of the frame, % of cap
           p_spill_any[D]    P(spilling at any time of the horizon)
           t_spill[D]        median hour of the first spill among the scenarios that spill (UTC ISO) or null
           t[T]              hours of the series (thinned to at most 48)
           v[3][T][D]        p10 / p50 / p90 volume, % of cap      qi[2][T][D], qo[2][T][D]  p50 / p90 inflow, outflow m3/s
           ok[D]             dam computed}
```

Levels of a reservoir: 2 = fills fast (gains 10 % of cap, or rises into its flood reserve) or outflow >= thr low;
3 = reaches the spill level or outflow >= thr mid; 4 = outflow >= thr high; 5 = spill >= design capacity or water at the crest.
Probabilities are weighted shares over the scenarios x 7 equal-probability multipliers of the simulated inflow (log-normal,
sigma of the control points); level = highest L with P >= tau of the horizon.
