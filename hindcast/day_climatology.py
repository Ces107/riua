"""How often each kind of day happens: the weights that turn the hindcast sample into a year.

    py -3.11 hindcast/day_climatology.py [scratch/q4-hindcast/chj_daily.csv]

The hindcast cases were chosen because something happened (or was warned of); ordinary and dry days are a
minority of the sample and the large majority of a year. Every verification day is put in a stratum by the
largest 24-h total of the SAIH Júcar gauge network that civil day, the one quantity known for EVERY day of
a long period (q4's daily scan, 2025-01-01 .. 2026-09-30, 144-175 gauges). The frequency of each stratum is
counted per calendar month and the twelve months are averaged, so that autumn is not under-represented by
a period with two springs and one autumn. Output: hindcast/day_climatology.json.

`stratum_of(day)` gives the stratum of a sample day from the same network (hindcast/obs/<day>.csv).
"""
from __future__ import annotations

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hindcast" / "day_climatology.json"
EDGES = (0.0, 1.0, 10.0, 30.0, 60.0, 100.0)          # mm in 24 h at the wettest SAIH gauge; the last stratum is open
LABELS = ("dry <1", "1-10", "10-30", "30-60", "60-100", ">=100")
EPISODE_FROM = 3                                     # strata 30 mm and above are "rain episode" days
# the AVAMET network is 5 times denser and its daily maximum larger: median SAIH / AVAMET over the 188 sample
# days that have both = 0.72 (same stratum on 100 of them)
AVAMET_TO_SAIH = 0.72


def stratum(mm: float) -> int:
    return max(k for k, e in enumerate(EDGES) if mm >= e)


@lru_cache(maxsize=None)
def stratum_of(day: str) -> int | None:
    """Stratum of one civil day from the gauge file q4 keeps for it (None: no SAIH data that day)."""
    f = ROOT / "hindcast" / "obs" / f"{day}.csv"
    if not f.exists():
        # the GitHub hindcast ships the class of each day, never the gauge files (hindcast/gha.py)
        s = ROOT / "hindcast" / "cache" / "day_strata.json"
        return json.loads(s.read_text(encoding="utf-8")).get(day) if s.exists() else None
    best = other = None
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["precip_24h_mm"])
            except ValueError:
                continue
            if r["source"].startswith("saih_chj"):
                best = v if best is None else max(best, v)
            elif r["source"] == "avamet":
                other = v if other is None else max(other, v)
    if best is None and other is not None:
        best = AVAMET_TO_SAIH * other           # 21 days of 2024-25 have no SAIH file
    return None if best is None else stratum(best)


def load() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scratch" / "q4-hindcast" / "chj_daily.csv"
    month = {m: [0] * len(EDGES) for m in range(1, 13)}
    days = []
    with open(src, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                mm = float(r["max"])
            except (ValueError, TypeError):
                continue
            month[int(r["date"][5:7])][stratum(mm)] += 1
            days.append(r["date"])
    freq_m = {m: [c / max(sum(v), 1) for c in v] for m, v in month.items()}
    annual = [sum(freq_m[m][k] for m in freq_m) / 12.0 for k in range(len(EDGES))]
    raw = [sum(month[m][k] for m in month) / len(days) for k in range(len(EDGES))]
    out = {"source": "SAIH Jucar daily maxima, q4 scan (scratch/q4-hindcast/chj_daily.csv)", "period": [min(days), max(days)],
           "n_days": len(days), "edges_mm": list(EDGES), "labels": list(LABELS), "episode_from": EPISODE_FROM,
           "frequency": [round(x, 5) for x in annual], "frequency_unweighted": [round(x, 5) for x in raw],
           "days_per_year": [round(365.25 * x, 1) for x in annual], "counts_by_month": month}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, lab in enumerate(LABELS):
        print(f"{lab:>8}: {annual[k]:.4f} of the days ({365.25 * annual[k]:5.1f} per year; unweighted {raw[k]:.4f})")
