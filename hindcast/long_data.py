"""Archive inputs for the days 2-7 hindcast of one case (run on GitHub Actions by .github/workflows/long.yml).

    python hindcast/long_data.py fetch <case id> [--no-aifs] [--no-ingr]
    python hindcast/long_data.py plan <case id>                        what would be fetched

For every target day D of the case and lead L = 2..7 the 00Z run of day D-L is the run production would use at
08Z (frames = UTC days +2 .. +7; ENS complete ~8 h after run time). Fetched from ECMWF open data on Google Cloud
(CC BY 4.0, archive since 2023), only the part of each message north of 37.3 N (sources/extra_models.py):
  ENS tp      50 members, steps 24L-12 .. 24L+24 every 12 h -> hindcast/cache/ens/ifsens_tp_<run>.npz
              (the format product.ens_npz_members reads, as production's long horizon)
  IFS tp      the deterministic 0.25 deg run, production's "global" members of the long horizon (2 lagged runs:
              00Z of the issue day and 12Z of the day before), 6-hourly -> ifsoper_tp_<run>.npz
  AIFS-ENS tp control + 50, 12-hourly, runs from 2025-07-02 on -> aifsens_tp_<run>.npz
  ingredients 5 ENS members at 12Z of D: total column water vapour, q / u / v at 850 hPa, z500, CAPE
              -> ingr_<run>_<step>.npz
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HC = Path(__file__).resolve().parent
ROOT = HC.parent
sys.path.insert(0, str(ROOT / "backend"))

from riua.sources import extra_models as X  # noqa: E402

ENS = HC / "cache" / "ens"
PARTS = HC / "cache" / "ens_parts"
LEADS = range(2, 8)
AIFS_FROM = datetime(2025, 7, 2)
INGR_MEMBERS = (1, 2, 3, 4, 5)
INGR = {"tcwv": ("tcwv", None), "q850": ("q", "850"), "u850": ("u", "850"), "v850": ("v", "850"), "z500": ("gh", "500"),
        "cape": ("mucape", None)}
GCS = X.ECMWF_MIRRORS["gcs"]


def case_days(cid: str) -> list[datetime]:
    c = next(x for x in json.loads((HC / "cases.json").read_text(encoding="utf-8"))["cases"] if x["id"] == cid)
    return [datetime.fromisoformat(d) for d in c["days"]]


def plan(cid: str) -> dict:
    """run -> {"ens": steps, "ifs": steps, "ingr": steps}."""
    need: dict[datetime, dict[str, set]] = {}
    for D in case_days(cid):
        for L in LEADS:
            run = D - timedelta(days=L)
            r = need.setdefault(run, {"ens": set(), "ifs": set(), "ingr": set()})
            r["ens"].update({24 * L - 12, 24 * L, 24 * L + 12, 24 * L + 24})
            r["ifs"].update(range(24 * L - 12, 24 * L + 25, 6))
            r["ingr"].add(24 * L + 12)
            prev = need.setdefault(run - timedelta(hours=12), {"ens": set(), "ifs": set(), "ingr": set()})
            prev["ifs"].update(range(24 * L, 24 * L + 37, 6))
    return need


_IDX: dict[str, list[dict]] = {}


def _index(stem: str) -> list[dict]:
    if stem not in _IDX:
        raw = X._get(stem + ".index")
        if raw is None:
            raise FileNotFoundError(stem + ".index")
        _IDX[stem] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if len(_IDX) > 8:
            _IDX.pop(next(iter(_IDX)))
    return _IDX[stem]


def _save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def fetch_ens(run: datetime, steps: list[int], model: str) -> str:
    name = {"ifs": "ifsens", "aifs": "aifsens"}[model]
    out = ENS / f"{name}_tp_{run:%Y%m%d%H}.npz"
    if out.exists() and set(steps) <= set(np.load(out)["steps"].astype(int).tolist()):
        return "have"
    z = X.ecmwf_ens_fetch(run, sorted(steps), PARTS, model=model, workers=8)
    _save_npz(out, data=z["acc"], members=np.asarray(z["numbers"]), steps=np.array(z["steps"]), lat=z["lat"], lon=z["lon"],
              meta=json.dumps({"model": model, "run": f"{run:%Y-%m-%dT%HZ}", "source": "ECMWF open data (GCS), CC BY 4.0"}))
    return "ok"


def fetch_ifs(run: datetime, steps: list[int]) -> str:
    out = ENS / f"ifsoper_tp_{run:%Y%m%d%H}.npz"
    if out.exists() and set(steps) <= set(np.load(out)["steps"].astype(int).tolist()):
        return "have"
    stream = "oper" if run.hour in (0, 12) else "scda"
    todo = []
    for s in sorted(steps):
        stem = f"{GCS}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/{stream}/{run:%Y%m%d%H}0000-{s}h-{stream}-fc"
        j = next(j for j in _index(stem) if j.get("param") == "tp" and j.get("levtype") == "sfc")
        todo.append((stem + ".grib2", j["_offset"], j["_length"]))
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda t: X._ecmwf_field(*t), todo))
    acc = np.stack([r[0] for r in res])[None]                # (1, S, ny, nx) mm since run start
    _save_npz(out, data=acc, members=np.array([0]), steps=np.array(sorted(steps)), lat=res[0][1], lon=res[0][2],
              meta=json.dumps({"model": "ifs_oper", "run": f"{run:%Y-%m-%dT%HZ}"}))
    return "ok"


def fetch_ingr(run: datetime, step: int) -> str:
    out = ENS / f"ingr_{run:%Y%m%d%H}_{step:03d}.npz"
    if out.exists():
        return "have"
    stem = f"{GCS}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/enfo/{run:%Y%m%d%H}0000-{step}h-enfo-ef"
    idx = _index(stem)
    todo = []
    for key, (param, lev) in INGR.items():
        for num in INGR_MEMBERS:
            cand = [j for j in idx if j.get("type") == "pf" and int(j.get("number", -1)) == num and
                    j.get("param") in ((param, "cape") if key == "cape" else (param,)) and (lev is None or j.get("levelist") == lev)]
            if cand:
                j = cand[0]
                todo.append((key, num, stem + ".grib2", j["_offset"], j["_length"]))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda t: X._ecmwf_field(t[2], t[3], t[4]), todo))
    arrays = {}
    for (key, num, *_), r in zip(todo, res):
        arrays.setdefault(key, []).append(r[0])
    if not arrays:
        return "none"
    lat, lon = res[0][1], res[0][2]
    _save_npz(out, lat=lat, lon=lon, **{k: np.stack(v) for k, v in arrays.items()})
    return "ok " + ",".join(f"{k}:{len(v)}" for k, v in arrays.items())


def fetch(cid: str, aifs: bool = True, ingr: bool = True) -> None:
    ENS.mkdir(parents=True, exist_ok=True)
    need = plan(cid)
    for run in sorted(need):
        r = need[run]
        jobs = []
        if r["ens"]:
            jobs.append(("ens", lambda: fetch_ens(run, sorted(r["ens"]), "ifs")))
            if aifs and run >= AIFS_FROM:
                jobs.append(("aifs", lambda: fetch_ens(run, sorted(r["ens"]), "aifs")))
        if r["ifs"]:
            jobs.append(("ifs", lambda: fetch_ifs(run, sorted(r["ifs"]))))
        if ingr:
            for s in sorted(r["ingr"]):
                jobs.append((f"ingr{s}", lambda s=s: fetch_ingr(run, s)))
        for name, fn in jobs:
            try:
                msg = fn()
            except Exception as e:  # noqa: BLE001  one missing file must not stop the case
                msg = f"FAILED {type(e).__name__}: {e}"[:200]
            print(f"{run:%Y-%m-%d %HZ} {name}: {msg}", flush=True)
    rq, by = X.transfer()
    print(f"transfer: {rq} requests, {by / 1e6:.0f} MB", flush=True)


# ------------------------------------------------------------------------------ ingredients per day

_SEA = None


def ingredient_scalars(run: datetime, step: int) -> dict | None:
    """Region summaries of the ingredient members (mean over members): largest total column water vapour, largest
    easterly 850-hPa moisture flux over the sea strip off the coast (g/kg x m/s), largest CAPE, mean z500 (gpm)."""
    f = ENS / f"ingr_{run:%Y%m%d%H}_{step:03d}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    lat, lon = z["lat"], z["lon"]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    box = (LA >= 37.6) & (LA <= 41.0) & (LO >= -2.4) & (LO <= 0.8)
    sea = (LA >= 38.4) & (LA <= 40.6) & (LO >= -0.1)
    out = {}
    if "tcwv" in z.files:
        out["tcwv"] = float(np.nanmean(np.nanmax(z["tcwv"][:, box], axis=1)))
    if all(k in z.files for k in ("q850", "u850", "v850")):
        n = min(len(z["q850"]), len(z["u850"]), len(z["v850"]))
        # onshore for the Valencian coast (facing ESE): the wind component from 110 degrees
        d = np.deg2rad(110.0)
        on = -(z["u850"][:n] * np.sin(d) + z["v850"][:n] * np.cos(d))
        flux = z["q850"][:n] * 1000.0 * np.maximum(on, 0.0)
        out["flux850"] = float(np.nanmean(np.nanmax(flux[:, sea], axis=1)))
    if "cape" in z.files:
        out["cape"] = float(np.nanmean(np.nanmax(z["cape"][:, box], axis=1)))
    if "z500" in z.files:
        out["z500"] = float(np.nanmean(np.nanmean(z["z500"][:, box], axis=1)))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "plan":
        for run, r in sorted(plan(a[1]).items()):
            print(run, {k: sorted(v) for k, v in r.items()})
    elif a[0] == "fetch":
        fetch(a[1], "--no-aifs" not in a, "--no-ingr" not in a)
