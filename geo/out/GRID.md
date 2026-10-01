# Riuà grid and file conventions

All files in this folder load with `json` + `numpy` only (no geopandas needed).
Coordinates are WGS84 / ETRS89 longitude, latitude in degrees (GeoJSON order `[lon, lat]`).

## The 0.05° grid (terrain.npz)

- Box: lon -2.4 .. 0.8, lat 37.6 .. 41.0. Cell size 0.05° x 0.05° (about 4.3 km x 5.6 km at 39.3°N).
- `NX = 64` columns, `NY = 68` rows. Every 2-D array has shape `(NY, NX) = (68, 64)`, indexed `[j, i]`.
- **`j` runs south -> north, `i` runs west -> east.** Row 0 is the southernmost row.
- Cell centres: `lon[i] = -2.375 + 0.05*i` (i = 0..63), `lat[j] = 37.625 + 0.05*j` (j = 0..67).
  Cell (j, i) covers lon `[-2.4 + 0.05*i, -2.4 + 0.05*(i+1)]`, lat `[37.6 + 0.05*j, 37.6 + 0.05*(j+1)]`.
- Flat cell index used by the weight tables: `c = j*NX + i`  (`j, i = divmod(c, 64)`).
- Point -> cell: `i = floor((lon + 2.4)/0.05)`, `j = floor((lat - 37.6)/0.05)`.
- With a north-up raster (row 0 = north), flip it first: `a[::-1]`.

### Arrays

| name | dtype, shape | meaning |
|---|---|---|
| `lon`, `lat` | f8 (64,), (68,) | cell-centre coordinates |
| `grid` | f8 (5,) | `[lon0, lon1, lat0, lat1, dx]` = `[-2.4, 0.8, 37.6, 41.0, 0.05]` |
| `elev_mean` | f4 (68,64) | mean elevation of the land pixels of the cell, m (0 where the cell has no land) |
| `elev_std` | f4 | standard deviation of the 90 m elevations inside the cell, m (sub-grid relief) |
| `elev_min`, `elev_max`, `relief` | f4 | min, max and max-min of the land pixels, m |
| `slope_mean_deg` | f4 | mean slope of the 90 m DEM inside the cell, degrees |
| `land_frac` | f4 | fraction of the cell that is land (0..1); land/sea mask = `land_frac > 0.5` |
| `elev_smooth` | f4 | elevation smoothed with a Gaussian of sigma 5 km (FWHM about 12 km), sea = 0 m |
| `dzdx`, `dzdy` | f4 | gradient of `elev_smooth` in **m per km**; `dzdx > 0` = terrain rises towards the EAST, `dzdy > 0` = rises towards the NORTH. Orographic ascent: `w = u*dzdx + v*dzdy` with the wind (u eastward, v northward) in m/s gives `w` in mm/s (positive = upslope flow). |
| `zone_idx` | i2 | index into `zone_codes` of the AEMET zone covering most of the cell, -1 = none |
| `basin_idx` | i2 | index into `basin_ids` of the basin unit covering most of the cell, -1 = none |
| `cv_frac` | f4 | fraction of the cell inside the Comunitat Valenciana |
| `domain_frac` | f4 | fraction of the cell that is CV or belongs to a basin unit draining into the CV |
| `in_domain` | bool | `domain_frac > 0.02` and the cell has land |

### Lookups (1-D, same order as `basins_topology.json["order"]` / `zones.geojson`)

| name | meaning |
|---|---|
| `basin_ids` (U), `basin_names` (U) | id and name of basin unit k (k = `idx` in basins.geojson) |
| `basin_area_km2` | total area of the unit |
| `basin_frac_in_box` | share of the unit's area that lies inside the box (one Segura headwater unit is 0, a few are < 1) |
| `basin_in_cv`, `basin_drains_to_cv` | unit overlaps the CV (>= 1 km2) / unit or any downstream unit overlaps the CV |
| `zone_codes` (U), `zone_names` (U) | the 11 AEMET zones |

### Area weights (CSR layout) — basin-mean and zone-mean rainfall

For unit `k`: `s = slice(basin_w_ptr[k], basin_w_ptr[k+1])`

- `basin_w_cell[s]` flat cell indices `c = j*NX + i`
- `basin_w[s]` weights that sum to 1 (area of the unit inside each cell / area of the unit inside the box)
- `basin_w_cellfrac[s]` fraction of each cell covered by the unit (use it for "max over the cells that are mostly in the unit")

```python
import numpy as np, json
T = np.load("terrain.npz")
NY, NX = T["elev_mean"].shape
def basin_mean(field, k):                     # field: (68, 64), south-up
    s = slice(T["basin_w_ptr"][k], T["basin_w_ptr"][k + 1])
    return float((field.ravel()[T["basin_w_cell"][s]] * T["basin_w"][s]).sum())
# whole upstream catchment of a unit (area-weighted over unit means):
topo = json.load(open("basins_topology.json", encoding="utf-8"))
order = topo["order"]; idx = {u: k for k, u in enumerate(order)}
def upstream_mean(field, uid):
    ks = [idx[uid]] + [idx[u] for u in topo["units"][uid]["upstream_all"]]
    a = T["basin_area_km2"][ks] * T["basin_frac_in_box"][ks]
    return float(sum(basin_mean(field, k) * w for k, w in zip(ks, a) if w > 0) / a.sum())
```

The same three arrays exist for the zones: `zone_w_ptr`, `zone_w_cell`, `zone_w`, `zone_w_cellfrac`.
The membership was computed on a 0.001° raster (50 x 50 sub-cells per grid cell), so fractions are exact to 0.04 %.

## Other files

- `zones.geojson` — 11 AEMET Meteoalerta zones. Properties: `code`, `name`, `province`, `province_code`, `area_km2`, `label_lon`, `label_lat`.
- `thresholds.json` — `zones[code].precip_1h_mm / precip_12h_mm = {yellow, orange, red}` (mm) + `source` (document, version, date, URL).
- `basins.geojson` — 420 basin units (a gap-free, overlap-free coverage). Properties:
  `id`, `idx`, `name`, `river`, `town` (largest municipality centre inside, may be null), `kind`, `area_km2`,
  `up_area_km2` (unit + everything upstream), `next` (downstream unit id or null), `terminal_unit`,
  `terminal` (`sea` / `endo` = closed basin / `outside` = flows out of the domain, i.e. to the Ebro),
  `drains_to_sea`, `n_upstream`, `outlet` `[lon, lat]`, `outlet_how`, `label` `[lon, lat]`, `source`,
  and when relevant `chj_codes`, `merged_names`, `within`.
  `kind`: `wb` official CHJ water-body catchment; `split` named tributary cut out of a very large one;
  `gap_river` / `patch` CEDEX sub-basins outside any water-body catchment (coastal ravines, coastal strips, closed basins);
  `hybas` / `hybas_group` HydroBASINS (Segura, Ebro tributaries).
- `basins_topology.json` — `order` (list of ids = idx) and `units[id] = {idx, name, area_km2, next, downstream_chain, terminal, upstream_direct, upstream_all, up_area_km2}`.
- `rivers.geojson` — named lines: `name`, `kind` (`rio`/`rambla`/`barranco`/`arroyo`/`canal`/`otro`), `length_km`, `source`.
- `places.json` — `places[]`: `name`, `names` (the two official forms when bilingual), `ine`, `lat`, `lon`, `alt_m`, `pop`, `province`, `province_code`, `in_cv`, `basin` (unit id), `zone` (AEMET code, null outside the CV).
- `boundary.geojson` — features by `kind`: `province` (3), `region` (CV), `coastline` (CV coast, lines), `land_box` (all land in the box + 0.3° margin, for masking the sea).
- `SOURCES.md` — licences and attribution texts.
