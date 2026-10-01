"""Observation space: long daily rain-gauge series from NOAA GHCN-Daily (keyless) -> annual maxima of daily rain.

    https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/<ID>.csv

PRCP is in tenths of mm for the station's observation day (Spanish pluviometric day, 07-07 UTC). Values with a
quality flag are dropped. A year is kept when it has >= 350 valid days, or >= 330 with September-December complete
to 95 % (the season of the extremes). GHCN lists several of these stations with the longitude sign wrong
(east instead of west); the coordinates below are corrected.
"""
from __future__ import annotations

import csv

import numpy as np

from common import SCRATCH

GHCN = SCRATCH / "ghcn"
#: id -> (name, lat, lon corrected, elevation m)
STATIONS = {
    "SP000008416": ("Valencia", 39.4806, -0.3664, 11),
    "SPE00120584": ("Valencia aeropuerto", 39.4867, -0.4731, 69),
    "SPE00101043": ("Alicante", 38.3725, -0.4942, 81),
    "SPE00101035": ("Alicante El Altet", 38.2828, -0.5706, 43),
    "SPE00120008": ("Castello de la Plana", 39.9892, -0.0406, 25),
    "SPE00119999": ("Castello Almassora", 39.9500, -0.0714, 35),
    "SP000009981": ("Tortosa (Obs. Ebre)", 40.8206, 0.4914, 44),
    "SP000007038": ("Torrevieja", 37.9769, -0.7106, 1),
    "SPE00120323": ("Murcia", 38.0028, -1.1692, 61),
    "SPE00120332": ("Murcia Alcantarilla", 37.9578, -1.2294, 85),
    "SPE00120341": ("Murcia San Javier", 37.7889, -0.8031, 4),
    "SP000008280": ("Albacete Los Llanos", 38.9519, -1.8631, 704),
    "SPE00119756": ("Albacete Obs.", 39.0067, -1.8606, 674),
    "SPE00120557": ("Teruel", 40.3494, -1.1167, 900),
    "SPE00120062": ("Cuenca", 40.0667, -2.1381, 945),
    "SPE00120125": ("Molina de Aragon", 40.8442, -1.8853, 1056),
}


def read_station(sid: str):
    """-> (dates datetime64[D], prcp mm float32 with NaN)."""
    dates, vals = [], []
    with open(GHCN / f"{sid}.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = row.get("PRCP", "").strip()
            if not v:
                continue
            attr = (row.get("PRCP_ATTRIBUTES") or "").split(",")
            if len(attr) > 1 and attr[1].strip():
                continue
            dates.append(row["DATE"])
            vals.append(int(v) / 10.0)
    return np.array(dates, "datetime64[D]"), np.array(vals, np.float32)


def annual_maxima(sid: str):
    """-> (years, annual max mm, date of the max) for accepted years."""
    d, v = read_station(sid)
    yrs = d.astype("datetime64[Y]").astype(int) + 1970
    mon = d.astype("datetime64[M]").astype(int) % 12 + 1
    out_y, out_v, out_d = [], [], []
    for y in np.unique(yrs):
        s = yrs == y
        n = int(s.sum())
        n_aut = int((s & (mon >= 9)).sum())
        if n >= 350 or (n >= 330 and n_aut >= 116):
            k = int(np.argmax(v[s]))
            out_y.append(int(y))
            out_v.append(float(v[s][k]))
            out_d.append(d[s][k])
    return np.array(out_y), np.array(out_v), np.array(out_d)


if __name__ == "__main__":
    for sid, (name, la, lo, el) in STATIONS.items():
        y, a, dd = annual_maxima(sid)
        top = np.argsort(a)[::-1][:3]
        print(f"{sid} {name:22s} {la:.2f} {lo:+.2f}: {len(y)} yr {y.min()}-{y.max()}, mean AM {a.mean():.1f}, "
              f"top: " + ", ".join(f"{a[k]:.0f} mm ({dd[k]})" for k in top))
