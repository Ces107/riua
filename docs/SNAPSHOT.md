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
obs               {hours, last_hour, o1, o12, o24}   base64 uint8[N] rain already fallen: last hour (cell max), last 12 h, last 24 h (cell mean)
gauges            [{id,name,lat,lon,source,t_utc,p_1h,p_12h,p_24h}]
rivers            [{id,name,river,lat,lon,source,t_utc,level_m,flow_m3s,thr_low,thr_mid,thr_high,...}]  (keys vary by source)
warnings          [{zone_code,zone_name,level:"yellow|orange|red",onset,expires,text,params,...}]  official AEMET rain warnings
drivers           optional: ingredients block (see backend/riua/diagnostics), may be missing
sources           [{id,label,ok,runs?,n?,error?,...}]
notes             [string]
timing_s          {...}
```

Each horizon:

```
frames   [{t0,t1 (UTC ISO, window (t0,t1]), ok:bool, has_1h:bool, cal:{c1,c1_from12,c12,rho}}]   F frames
tau      {"2":p,"3":p,"4":p,"5":p}    probability needed to issue each level
cells    level  base64 uint8[F*N]      0 = no data, 1..5
         p      base64 uint8[4*F*N]    P(>=2),P(>=3),P(>=4),P(>=5); P = v/200
         e1,e12 base64 uint8[2*F*N]    calibrated expected amount: [median, 1-in-10 high], 1 h and 12 h; mm = (v/8)^2
         m1,q1,m12,q12  base64 uint8[F*N]  raw ensemble mean / 90th percentile of the scenarios (mm = (v/8)^2; 255 = n/a)
members  [{name,family,model,run,step_h,neigh_km,w_raw,w:[per frame normalised weight]}]   M scenarios
basins   level[F][B], p[4][F][B], own12[2][F][B] (mm, median/p90 12-h mean rain over the unit),
         up12[2][F][B] (same over unit+upstream), q[2][F][B] (m3/s/km2), upstream[F][B] (0..1 share of scenarios where upstream rain dominates)
         B and order = geo/out/basins.geojson features order (property idx)
points   (only when the control-point network exists) level[F][P], p[4][F][P], qpeak[2][F][P] m3/s,
         hover[2][F][P] m above bank (negative = below), t[T] ISO, q[3][T][P] (p10,p50,p90 hydrograph m3/s)
         P and order = geo/hydro/catchments/out/control_points.json
```

Frames: `now` = 6 hourly frames from the top of the current hour; `mid` = 14 three-hour frames from +6 h; `long` = 6 UTC days (day+2 … day+7).

explain-<hz>.bin (lazy-loaded for the audit view): uint8 `a1[M,F,N]` (255 = that scenario has no 1-h information), `a12[M,F,N]`, `p1[4,F,N]`, `p12[4,F,N]` — per-scenario 1-h and 12-h amounts (after neighbourhood), and the two marginal exceedance probabilities before the union.

The probability of a level is NOT a count of scenarios: it comes from the calibrated distribution `cal` (censored shifted gamma): with `m`,`q` the ensemble mean and 90th percentile, `mu = a0 + a1*m + a2*q`, `sigma = b0*sqrt(mu) + b1*(q-m)`, shape `k = mu²/sigma²`, scale `θ = sigma²/mu`, `P(amount >= T) = Q(k, (T - delta)/θ)` (upper regularised incomplete gamma). The two marginals (1 h, 12 h) are joined with a Gaussian copula of correlation `rho`. Level = highest L with `P(>=L) >= tau[L]`.
