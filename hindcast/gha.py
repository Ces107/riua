"""Driver of the hindcast on GitHub Actions (.github/workflows/hindcast.yml). Linux runners have pysteps.

    python hindcast/gha.py matrix                    JSON list of case ids for the job matrix
    python hindcast/gha.py inputs                    download + unpack the release assets (forecast archives, radar)
    python hindcast/gha.py case <id> [--no-ens]      one case: gauges, truth, ENS, AROME-HD runs, blocks
    python hindcast/gha.py collect <dir>             final job: unpack every case artifact into the hindcast tree

Data rules (coord/BATCH-Q.md): only public sources on GitHub.
  - gauges: SAIH Júcar only (fetch_obs.rows_chj_api / rows_chj_pdf); AVAMET and Meteoclimatic (CC BY-NC-ND)
    are never fetched here, so the truth built on a runner is radar + SAIH, like production. SAIH's daily API
    starts in 2025: for 2024 days only the 08-08 PDF reports exist, which build_truth does not use for a civil
    day -> 2024 truths are radar with the climatological gauge factor.
  - forecasts: the Open-Meteo Previous Runs archive is NOT called from runners (that would spread one quota
    over many IPs). The series q4 fetched locally (CC BY 4.0) are a release asset; complete runs are read from
    the Open-Meteo S3 mirror; ECMWF ENS from ECMWF open data on Google Cloud (CC BY 4.0).
  - what a job uploads: blocks, the truth it built (radar + SAIH), and the stratum of each day (a class
    0-5, hindcast/cache/day_strata.json). No gauge file leaves the runner.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HC = ROOT / "hindcast"
REPO = "Ces107/riua"
TAG = "hindcast-inputs"
ASSETS = ("forecasts.tar", "radar.tar", "acc.tar")
STRATA = HC / "cache" / "day_strata.json"
PY = sys.executable


def cases() -> list[dict]:
    return json.loads((HC / "cases.json").read_text(encoding="utf-8"))["cases"]


def sh(*args, check=True, timeout=None):
    print("+", " ".join(str(a) for a in args), flush=True)
    return subprocess.run([str(a) for a in args], cwd=ROOT, check=check, timeout=timeout)


def inputs() -> None:
    for name in ASSETS:
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}"
        dst = ROOT / "scratch" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("download", url, flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:  # noqa: BLE001
            # without the release: radar and ENS are fetched by build_truth / fetch_ens*, complete runs from the S3
            # mirror; the Previous Runs series of the older cases (tier B) are then missing
            print("no release asset", name, type(e).__name__, e, flush=True)
            continue
        with tarfile.open(dst) as t:
            t.extractall(ROOT, filter="data")
        dst.unlink()


def gauges(days: list[str]) -> None:
    """SAIH Júcar only: the daily API (2025 on) and the PDF reports."""
    sys.path.insert(0, str(HC))
    import fetch_obs as F
    import day_climatology as DC
    strata = json.loads(STRATA.read_text(encoding="utf-8")) if STRATA.exists() else {}
    for d in days:
        if (F.OUT / f"{d}.csv").exists():
            rows = None
        else:
            rows = []
            for fn in (lambda: F.rows_chj_api(d, False, 40.0), lambda: F.rows_chj_pdf(d)):
                try:
                    rows += fn()
                except Exception as e:  # noqa: BLE001  one source down must not stop the job
                    print("gauges", d, type(e).__name__, e, flush=True)
            F.write_csv(d, rows)
        DC.stratum_of.cache_clear()
        s = DC.stratum_of(d)
        if s is not None:
            strata[d] = s
        print("gauges", d, "rows" if rows is None else len(rows), "stratum", s, flush=True)
    STRATA.parent.mkdir(parents=True, exist_ok=True)
    STRATA.write_text(json.dumps(strata, indent=0, sort_keys=True), encoding="utf-8")


def arome_hd_runs(c: dict) -> None:
    """AROME-HD complete runs from the S3 mirror (kept there ~94 days): production's heaviest family in 0-48 h."""
    sys.path.insert(0, str(HC))
    import fetch_openmeteo_archive as A
    d0 = date.fromisoformat(c["days"][0]) - timedelta(days=3)
    d1 = date.fromisoformat(c["days"][-1])
    if d1 < date.today() - timedelta(days=92):
        return
    model = "meteofrance_arome_france_hd"
    n = 0
    d = d0
    while d <= d1:
        try:
            for run in A.list_runs(model, d):
                if A.fetch_run_box(model, run) is not None:
                    n += 1
        except Exception as e:  # noqa: BLE001
            print("arome_hd", d, type(e).__name__, e, flush=True)
        d += timedelta(days=1)
    print("arome_hd runs", n, flush=True)


def case(cid: str, with_ens: bool = True) -> None:
    c = next(x for x in cases() if x["id"] == cid)
    days = list(c["days"])
    gauges(days + [(date.fromisoformat(days[0]) - timedelta(days=k)).isoformat() for k in (1, 2)])
    sh(PY, "hindcast/build_truth.py", "--acc-only", "--jobs", "3", cid, check=False)
    sh(PY, "hindcast/build_truth.py", cid, check=False)
    if with_ens:
        sh(PY, "hindcast/fetch_ens3h.py", cid, check=False, timeout=3 * 3600)
        sh(PY, "hindcast/fetch_ens_cases.py", f"{days[0]}:{days[-1]}", check=False, timeout=3600)
    arome_hd_runs(c)
    sh(PY, "hindcast/run.py", "build", "mid", "long", "--cases", cid, "--jobs", "3", check=False)
    sh(PY, "hindcast/run.py", "build", "now", "--cases", cid, "--jobs", "2", check=False)
    out = ROOT / "scratch" / "artifact"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out / f"{cid}.tar", "w") as t:
        for p in sorted((HC / "cache" / "blocks").rglob("[0-9]*.npz")):
            if ".tmp" not in p.name:
                t.add(p, p.relative_to(ROOT))
        tr = HC / "truth" / f"{cid}.npz"
        if tr.exists():
            t.add(tr, tr.relative_to(ROOT))
        if STRATA.exists():
            t.add(STRATA, Path("hindcast/cache/strata") / f"{cid}.json")
    print("artifact", out / f"{cid}.tar", flush=True)


def collect(src: str) -> None:
    strata = {}
    for f in sorted(Path(src).rglob("*.tar")):
        with tarfile.open(f) as t:
            t.extractall(ROOT, filter="data")
    for f in (HC / "cache" / "strata").glob("*.json"):
        strata.update(json.loads(f.read_text(encoding="utf-8")))
    STRATA.write_text(json.dumps(strata, indent=0, sort_keys=True), encoding="utf-8")
    for hz in ("now", "mid", "long"):
        print(hz, len(list((HC / "cache" / "blocks" / hz).glob("*.npz"))), "blocks", flush=True)
    print(len(list((HC / "truth").glob("*.npz"))), "truth files,", len(strata), "day strata", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "matrix":
        print(json.dumps([c["id"] for c in cases()]))
    elif a[0] == "inputs":
        inputs()
    elif a[0] == "case":
        case(a[1], "--no-ens" not in a)
    elif a[0] == "collect":
        collect(a[1])
