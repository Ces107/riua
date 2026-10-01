"""Shared helpers for the Riuà geodata build (geo/build/*.py).

Run every script with `py -3.11 geo/build/<script>.py` from the repo root (any cwd works).
Raw downloads are cached in scratch/r5-geo/ (gitignored); outputs go to geo/out/.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "scratch" / "r5-geo"
OUT = ROOT / "geo" / "out"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) riua-geo-build/0.1"

# Domain box (coord/PROTOCOL.md)
LON0, LON1, LAT0, LAT1 = -2.4, 0.8, 37.6, 41.0
DX = 0.05
NX = 64  # round((LON1-LON0)/DX)
NY = 68  # round((LAT1-LAT0)/DX)

try:  # make prints safe on a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


def download(url: str, dest: Path, *, min_bytes: int = 1000, pause: float = 1.0) -> Path:
    """Download url to dest unless already cached. Polite: one request, then a pause."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    for attempt in range(4):
        try:
            with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=180) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
            break
        except requests.ConnectionError as e:  # transient DNS / reset errors
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1}: {type(e).__name__}")
            time.sleep(5 * (attempt + 1))
    if dest.stat().st_size < min_bytes:
        raise RuntimeError(f"{url}: suspiciously small download ({dest.stat().st_size} B)")
    time.sleep(pause)
    return dest


def unzip(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir = Path(dest_dir)
    if not dest_dir.exists() or not any(dest_dir.iterdir()):
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest_dir)
    return dest_dir


def find_one(folder: Path, pattern: str) -> Path:
    hits = sorted(Path(folder).rglob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern} in {folder}: {len(hits)} matches {hits[:5]}")
    return hits[0]


def round_coords(obj, nd: int = 4):
    """Round every coordinate in a GeoJSON-like nested list."""
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), nd) for v in obj]
        return [round_coords(o, nd) for o in obj]
    return obj


def write_geojson(path: Path, features: list[dict], *, nd: int = 4, meta: dict | None = None) -> int:
    fc = {"type": "FeatureCollection"}
    if meta:
        fc.update(meta)
    fc["features"] = features
    for f in features:
        g = f["geometry"]
        if g is not None:
            g["coordinates"] = round_coords(g["coordinates"], nd)
    txt = json.dumps(fc, ensure_ascii=False, separators=(",", ":"))
    Path(path).write_text(txt, encoding="utf-8")
    return len(txt.encode("utf-8"))


def write_json(path: Path, obj, *, indent=None) -> int:
    txt = json.dumps(obj, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))
    Path(path).write_text(txt, encoding="utf-8")
    return len(txt.encode("utf-8"))
