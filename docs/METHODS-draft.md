# Riuà — methods (draft for expert review)

Status: DRAFT written before any tuning. Code: `backend/riua/core/*.py`, parameters: `backend/riua/params.py`.

## 1. Purpose and output

Probabilistic risk of extreme precipitation and flash flood for the Comunitat Valenciana, three horizons:
nowcast 0–6 h (hourly frames), short range 6–48 h (3-h frames), medium/long range days 2–7 (daily frames).
Three spatial supports, each with its own level 1–5 and probabilities P(≥2), P(≥3), P(≥4), P(≥5):

- **Cell**: regular 0.05° grid (64 × 68, lon −2.4..0.8, lat 37.6..41.0).
- **Basin unit**: ~150–400 catchment units with upstream topology.
- **Control point**: ~50 sections where a ravine/river meets a town; discharge, overflow probability, water level.

Levels: 1 none; 2, 3, 4 = the yellow / orange / red thresholds of AEMET's Plan Meteoalerta for the warning zone
(1-h and 12-h accumulation, zone-specific); 5 "extreme" = considerably beyond red.

## 2. Inputs (planned)

- Convection-permitting NWP: AROME-France HD 1.3 km and AROME 2.5 km (Météo-France), successive runs (time-lagged ensemble); other regional models that cover the area (ICON-EU 6.5 km, ARPEGE-Europe, HARMONIE-AROME Europe if coverage allows).
- Global deterministic: IFS 9 km / 0.25°, GFS 13 km.
- Ensembles: ECMWF ENS 51 members 0.25° (also ICON-EU-EPS, GEFS as available).
- Observations: radar composite (reflectivity → rain rate, Z = 200 R^1.6), rain gauges (SAIH, AVAMET), river gauges, official AEMET warnings (displayed, never mixed into the level).
- Static: AEMET zones + thresholds, DEM, HydroSHEDS-type flow directions, LiDAR DTM cross-sections.

All models are accessed through Open-Meteo's open-data mirror (S3) for live use and its Previous-Runs / Historical-Forecast archive for hindcasts. Fine-grid models are **sampled** at Riuà cell centres (nearest native point), not block-averaged, so live and hindcast inputs are statistically identical; coarse models are interpolated.

## 3. Cell-level probability

For member *m*, frame *f*, cell *x*:

- a1 = largest 1-h accumulation ending in the frame; a12 = largest 12-h accumulation ending in the frame (window may start before the frame).
- For convection-permitting members, a1 and a12 are replaced by their maximum within radius R (neighbourhood maximum; R = 10 km nowcast, 20 km short range) — Schwartz & Sobash (2017) NMEP semantics: "the event occurs within R of here".
- Family representativeness factors s1h, s12h (≥ 1 for coarse models) multiply the amounts. Families with long native steps or coarse grids (global, ENS) are judged on 12 h only.
- Hazard ratio at level L: ρ_L = max(s1h·a1 / T1h_L(zone), s12h·a12 / T12h_L(zone)). Extreme: T_5 = 1.5 × red (1 h), 1.67 × red (12 h).
- Kernel dressing (Roulston & Smith 2003): each member contributes Φ(ln(b·ρ_L)/σ) instead of the indicator 1[ρ_L ≥ 1]; σ and b depend on the horizon (σ = 0.35 / 0.50 / 0.65, to be tuned).
- P(≥L) = Σ w_m Φ(·) / Σ w_m with w_m = family weight × 0.5^(run age / half-life). Probabilities are forced to be nested in L.

## 4. Decision rule

level = max{ L : P(≥L) ≥ τ_L(horizon) }, else 1. τ decreases with severity (defaults nowcast 0.50/0.40/0.30/0.20, short 0.45/0.35/0.25/0.15, long 0.40/0.30/0.20/0.10), justified by cost–loss: the optimal probability threshold equals C/L and the loss of a missed level-5 event dwarfs the cost of a false alarm. τ, σ, b and the family factors are to be tuned on hindcasts.

## 5. Basin units

Per member: hourly basin-mean rain over the unit (own) and over unit + everything upstream (up). For durations d ∈ {1, 3, 6, 12} h, ratio = S_d / (K_A · T_L(d)), with T_L(d) a power-law interpolation between the official 1-h and 12-h thresholds, and K_A = 1 − log10(A)/15 (areal reduction factor of Norma 5.2-IC). Upstream windows may end up to tc = 0.5·A^0.38 h before the frame. Displacement uncertainty for convection-permitting members: the field is also evaluated shifted ±Δ N/S/E/W (weights 0.4 unshifted, 0.15 each). Additional path to level 5: unit peak discharge q from SCS-type losses (P0 = 25 mm) and mean net intensity over tc, compared with 0.35 × the Gaume et al. (2009) Mediterranean envelope q = 100·A^−0.4 m³ s⁻¹ km⁻².

## 6. Control points (rain → discharge → stage)

- Losses per cell: wetness W(t) = W(t−1)·e^(−1/72 h) + p(t); runoff share = marginal coefficient of E = (P−P0)²/(P+4P0) evaluated at W.
- Routing: time–area method with travel-time bands derived from the DEM flow network (hillslope + channel velocity law calibrated on Témez tc), then a linear reservoir K = 0.3·tc.
- Catchments exclude the area above large dams ("unregulated" scope); dam releases are not modelled.
- Hydraulics: LiDAR cross-section, Manning compound section → rating curve, bankfull capacity Qb.
- Levels: 2 if Q ≥ 0.25 Qb; 3 if Q ≥ 0.6 Qb; 4 if Q ≥ Qb (overflow); 5 if water ≥ 1.0 m above the bank crest (or Q ≥ 2 Qb when no section). Probability by the same dressing with σ_hydro = 0.5 / 0.65 / 0.8.

## 7. Extremeness and ingredients (shown as drivers, not used in the level)

- EFI and SOT of the ENS 24-h precipitation against an ERA5 daily climate (±15 days, 30 years) — reanalysis climate, not model climate.
- PWAT and its standardized anomaly; IVT; 925-hPa wind and upslope flow V·∇h; most-unstable CAPE, normalised CAPE (tall-skinny profile), warm-cloud depth (0 °C height − LCL); Corfidi upwind vector speed (back-building potential); 500-hPa cut-off low position and depth.

## 8. Nowcast

Members: (a) radar extrapolation (optical-flow advection of the rain-rate field, persisted for 1–2 h, weight decaying to zero by ~3 h), (b) the newest convection-permitting runs. Hours already elapsed are filled with observed rain (radar QPE adjusted to gauges) so that 12-h accumulations and soil wetness include what has already fallen.

## 9. Validation plan

Hindcast with forecasts really issued at the time (archives from 2023): 29 Oct 2024 plus ≥ 2 other Valencian events and null cases. Truth: station observations (AVAMET / SAIH / AEMET) converted to observed level per AEMET zone and per cell neighbourhood. Scores per horizon and level: hits, misses, false alarms, POD, FAR, CSI, bias; reliability of probabilities. Thresholds tuned on the hindcasts.

## Known weak points (author's own list)

1. Very small event sample for tuning (risk of overfitting).
2. Kernel-dressing σ and family factors are assumed constant in space and lead time within a horizon.
3. Time-lagged ensemble members are not independent; weights are heuristic.
4. Reanalysis climate for EFI.
5. No hydraulic modelling beyond normal depth; no dam operations; constant P0.
6. Archive of convection-permitting runs only reaches lead day 1 for 2024.
