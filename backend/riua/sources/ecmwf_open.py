#!/usr/bin/env python
"""Download archived ECMWF ENS (IFS open data) fields for all 51 members, cropped to the Riuà box.

Source: ECMWF open-data archive, free and keyless (CC-BY-4.0).
  primary : https://storage.googleapis.com/ecmwf-open-data   (complete from 2023-07-12 12z, verified)
  fallback: https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com (often answers 503 SlowDown)

How it works
  1. fetch `<run>-<step>h-enfo-ef.index` (JSON lines: one line per GRIB message with _offset/_length)
  2. keep the lines of the wanted param (default `tp`, total precipitation accumulated since run start)
  3. HTTP Range request for each member's message. The grid is scanned north -> south and packed with
     CCSDS, so only the first ~40 % of the message is needed to reach latitude 37.5N. We download that
     prefix, pad the rest, decode twice with two different paddings and accept the result only if both
     decodes agree down to a safety margin below the box (otherwise more bytes are fetched). `--full`
     disables the trick and downloads whole messages.
  4. crop to the box, cache one small .npy per (step, member) -> resumable; assemble the .npz.

Output npz
  data    float32 [member, step, lat, lon]   (tp: mm accumulated since run start)
  members int16   [member]  0 = control forecast (cf), 1..50 = perturbed (pf)
  steps   int16   [step]    forecast hours
  lat     float32 ascending, lon float32 ascending (degrees east, -180..180)
  valid_time  str [step] ISO UTC;  meta: json string (run, param, units, source, grid, bytes downloaded)

Other models (same output format)
  --model ifs-hres : ECMWF deterministic 0.25 deg (0.4 deg in 2023), streams oper (00/12, to 240 h) and
                     scda (06/18, to 90 h); 1 "member". Some 06/18 cycles are missing in the archive.
  --model gefs     : NOAA GEFS 0.25 deg, 31 members, https://noaa-gefs-pds.s3.amazonaws.com. tp is NOT
                     accumulated since run start: each step holds the 6-h bucket ending at that step
                     (meta.bucket_hours = 6), steps must be multiples of 6. Whole messages (~0.27 MB each).

Examples
  py -3.11 hindcast/fetch_ens.py --run 2024-10-27T00 --steps 48,54,60,66,72,78 \
      --out hindcast/cache/ens/ifsens_tp_2024102700.npz --point 39.47,-0.72
  py -3.11 hindcast/fetch_ens.py --model gefs --run 2024-10-27T00 --steps 54,60,66,72 --point 39.47,-0.72
  py -3.11 hindcast/fetch_ens.py --model ifs-hres --run 2024-10-27T00 --steps 48,72 --point 39.47,-0.72
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

BOX = dict(lon_min=-2.4, lon_max=0.8, lat_min=37.6, lat_max=41.0)
MIRRORS = {
    "ecmwf": "https://data.ecmwf.int/forecasts",              # live only (last ~4 days), fastest to update
    "gcs": "https://storage.googleapis.com/ecmwf-open-data",
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
}
UA = "riua-hindcast/0.1 (flood-risk research; python-requests)"
import os
CACHE = Path(os.environ.get("RIUA_ENS_CACHE", Path(__file__).resolve().parents[3] / "hindcast" / "cache" / "ens"))
UNIT_SCALE = {"tp": (1000.0, "mm"), "ro": (1000.0, "mm")}  # metres -> mm

_local = threading.local()
_bytes = {"n": 0}
_bytes_lock = threading.Lock()


def session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = UA
        _local.s = s
    return s


def http_get(url: str, rng: tuple[int, int] | None = None, tries: int = 8) -> bytes | None:
    """GET with exponential backoff on 429/5xx (S3 SlowDown = 503). Returns None on 404."""
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    delay = 2.0
    for attempt in range(tries):
        try:
            r = session().get(url, headers=headers, timeout=(15, 120))
            if r.status_code in (200, 206):
                if rng and len(r.content) != rng[1] - rng[0] + 1:
                    raise IOError(f"short read {len(r.content)}")
                with _bytes_lock:
                    _bytes["n"] += len(r.content)
                return r.content
            if r.status_code in (403, 404):
                return None
            if r.status_code not in (429, 500, 502, 503, 504):
                raise IOError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
            why = f"HTTP {r.status_code}"
        except (requests.RequestException, IOError) as e:  # network hiccup -> retry
            why = repr(e)[:120]
        if attempt == tries - 1:
            break
        wait = delay * (1 + random.random())
        print(f"    backoff {wait:4.1f}s ({why})", file=sys.stderr)
        time.sleep(wait)
        delay = min(delay * 2, 60)
    raise IOError(f"giving up on {url} after {tries} tries")


GEFS_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_PARAM = {"tp": "APCP:surface", "cape": "CAPE:surface", "pwat": "PWAT:entire atmosphere"}
MODEL_NAMES = {"ifs-ens": "ECMWF IFS ENS (open data, 51 members)",
               "ifs-hres": "ECMWF IFS HRES deterministic (open data)",
               "gefs": "NOAA GEFS 0.25 deg (31 members, pgrb2sp25)"}


def _stream(run: dt.datetime, model: str) -> tuple[str, str]:
    if model == "ifs-hres":  # 00/12 UTC: oper (to 240 h); 06/18 UTC: scda (to 90 h)
        s = "oper" if run.hour in (0, 12) else "scda"
        return s, f"{s}-fc"
    return "enfo", "enfo-ef"


def candidate_dirs(run: dt.datetime, model: str = "ifs-ens") -> list[str]:
    """Directory layouts used by the archive over time (verified by listing the bucket)."""
    d, hh = run.strftime("%Y%m%d"), run.strftime("%H")
    s = _stream(run, model)[0]
    return [
        f"{d}/{hh}z/ifs/0p25/{s}",   # from 2024-02-28 06z
        f"{d}/{hh}z/0p25/{s}",       # 2024-01-31 06z .. 2024-02-28 00z
        f"{d}/{hh}z/0p4-beta/{s}",   # 2023-07-12 12z .. 2024-01-31 (0.4 deg)
        f"{d}/{hh}z/ifs/0p4-beta/{s}",
    ]


def load_index_gefs(run: dt.datetime, step: int, param: str) -> dict:
    """GEFS: one file per member, wgrib2-style .idx (`n:offset:d=..:APCP:surface:42-48 hour acc fcst:ENS=+1`)."""
    want = GEFS_PARAM[param]
    msgs = []
    for num in range(31):
        mem = "gec00" if num == 0 else f"gep{num:02d}"
        base = (f"{GEFS_URL}/gefs.{run:%Y%m%d}/{run:%H}/atmos/pgrb2sp25/"
                f"{mem}.t{run:%H}z.pgrb2s.0p25.f{step:03d}")
        raw = http_get(base + ".idx")
        if raw is None:
            raise IOError(f"missing {base}.idx")
        lines = raw.decode().strip().splitlines()
        for i, line in enumerate(lines):
            p = line.split(":")
            if f"{p[3]}:{p[4]}".startswith(want):
                if i + 1 >= len(lines):
                    raise IOError("message is last in file, length unknown")
                off = int(p[1])
                msgs.append(dict(number=num, _offset=off, _length=int(lines[i + 1].split(":")[1]) - off,
                                 url=base, what=p[5]))
                break
        else:
            raise IOError(f"{want} not in {base}.idx")
    return dict(url=msgs[0]["url"], dir="pgrb2sp25", mirror="noaa-gefs-pds", msgs=msgs)


def load_index(run: dt.datetime, step: int, param: str, mirrors: list[str], model: str = "ifs-ens") -> dict:
    """Return {'url': grib url, 'msgs': [{number,_offset,_length}]} for one step; cached on disk."""
    tag = run.strftime("%Y%m%d%H")
    pre = "" if model == "ifs-ens" else model + "_"
    f = CACHE / "idx" / f"{pre}{tag}_{param}_{step:03d}.json"
    if f.exists():
        return json.loads(f.read_text())
    if model == "gefs":
        out = load_index_gefs(run, step, param)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out))
        return out
    stem = f"{run:%Y%m%d%H}0000-{step}h-{_stream(run, model)[1]}"
    for m in mirrors:
        for d in candidate_dirs(run, model):
            base = f"{MIRRORS[m]}/{d}/{stem}"
            try:
                raw = http_get(base + ".index", tries=4 if m == "aws" else 8)
            except IOError as e:
                print(f"  index failed on {m}: {e}", file=sys.stderr)
                break  # next mirror
            if raw is None:
                continue
            msgs = []
            for line in raw.decode().splitlines():
                if not line.strip():
                    continue
                j = json.loads(line)
                if j.get("param") == param and j.get("levtype") == "sfc":
                    num = int(j["number"]) if j["type"] == "pf" else 0
                    msgs.append(dict(number=num, _offset=j["_offset"], _length=j["_length"]))
            if not msgs:
                raise IOError(f"param {param} not in {base}.index")
            msgs.sort(key=lambda x: x["number"])
            out = dict(url=base + ".grib2", dir=d, mirror=m, msgs=msgs)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(out))
            return out
    raise IOError(f"no index found for run {tag} step {step} (tried {mirrors})")


_ec_lock = threading.Lock()  # eccodes is not thread-safe on Windows (crashes on first use from 2 threads)


def _decode(msg: bytes):
    with _ec_lock:
        return _decode_locked(msg)


def _decode_locked(msg: bytes):
    import eccodes as ec

    h = ec.codes_new_from_message(msg)
    try:
        nj, ni = ec.codes_get(h, "Nj"), ec.codes_get(h, "Ni")
        if ec.codes_get(h, "gridType") != "regular_ll" or ec.codes_get(h, "jScansPositively") != 0:
            raise IOError("unexpected grid (need regular_ll scanned north->south)")
        lat = np.asarray(ec.codes_get_array(h, "distinctLatitudes"), dtype=np.float64)
        lon = np.asarray(ec.codes_get_array(h, "distinctLongitudes"), dtype=np.float64)
        if lat[0] < lat[-1]:
            lat = lat[::-1]
        vals = ec.codes_get_values(h).reshape(nj, ni)
        if lon.max() > 180:  # 0..360 grids (GEFS): rotate to -180..180 so the box can straddle Greenwich
            lon = ((lon + 180.0) % 360.0) - 180.0
            order = np.argsort(lon)
            lon, vals = lon[order], vals[:, order]
        info = dict(shortName=ec.codes_get(h, "shortName"), units=ec.codes_get(h, "units"),
                    dataDate=ec.codes_get(h, "dataDate"), dataTime=ec.codes_get(h, "dataTime"),
                    stepRange=str(ec.codes_get(h, "stepRange")), Ni=ni, Nj=nj,
                    dataType=ec.codes_get(h, "dataType"),
                    number=ec.codes_get(h, "number") if ec.codes_is_defined(h, "number") else 0)
        return vals, lat, lon, info
    finally:
        ec.codes_release(h)


def _box_slices(lat: np.ndarray, lon: np.ndarray):
    """Rows/cols of the smallest grid rectangle that encloses the box (lat is north->south)."""
    eps = 1e-6
    rows = np.where((lat >= BOX["lat_min"] - eps) & (lat <= BOX["lat_max"] + eps))[0]
    r0, r1 = max(rows[0] - (lat[rows[0]] < BOX["lat_max"] - eps), 0), rows[-1]
    if lat[r1] > BOX["lat_min"] + eps:
        r1 += 1
    cols = np.where((lon >= BOX["lon_min"] - eps) & (lon <= BOX["lon_max"] + eps))[0]
    c0, c1 = cols[0], cols[-1]
    if lon[c0] > BOX["lon_min"] + eps:
        c0 -= 1
    if lon[c1] < BOX["lon_max"] - eps:
        c1 += 1
    return int(r0), int(r1), int(c0), int(c1)


def fetch_member(url: str, off: int, length: int, full: bool):
    """Download (a prefix of) one GRIB message and return cropped field + lat/lon + info + bytes."""
    if full:
        vals, lat, lon, info = _decode(http_get(url, (off, off + length - 1)))
        r0, r1, c0, c1 = _box_slices(lat, lon)
        return vals[r0:r1 + 1, c0:c1 + 1], lat[r0:r1 + 1], lon[c0:c1 + 1], info, length
    frac = min(1.0, (90.0 - BOX["lat_min"]) / 180.0 + 0.11)  # ~0.40 for 37.6N
    have = b""
    while True:
        want = length if frac >= 0.999 else int(length * frac)
        have += http_get(url, (off + len(have), off + want - 1))
        if len(have) == length:
            vals, lat, lon, info = _decode(have)
            r0, r1, c0, c1 = _box_slices(lat, lon)
            return vals[r0:r1 + 1, c0:c1 + 1], lat[r0:r1 + 1], lon[c0:c1 + 1], info, len(have)
        dec = []
        for pad in (b"\xff", b"\x55", b"\xaa"):
            try:
                dec.append(_decode(have + pad * (length - len(have) - 4) + b"7777"))
            except Exception:
                continue
            if len(dec) == 2:
                break
        if len(dec) == 2:
            (a, lat, lon, info), (b, _, _, _) = dec
            r0, r1, c0, c1 = _box_slices(lat, lon)
            diff = np.where((a != b).any(axis=1))[0]
            first_bad = int(diff[0]) if len(diff) else a.shape[0]
            if first_bad >= r1 + 6:  # both paddings agree well below the box -> real data
                return a[r0:r1 + 1, c0:c1 + 1], lat[r0:r1 + 1], lon[c0:c1 + 1], info, len(have)
        frac = min(1.0, frac + 0.15)


def fetch_run(run: dt.datetime, steps: list[int], param: str = "tp", out: Path | None = None,
              mirrors: list[str] | None = None, workers: int = 4, full: bool = False,
              members: list[int] | None = None, verbose: bool = True, model: str = "ifs-ens") -> Path:
    mirrors = mirrors or ["gcs", "aws"]
    tag = run.strftime("%Y%m%d%H")
    pre = "" if model == "ifs-ens" else model + "_"
    out = Path(out) if out else CACHE / f"{model.replace('-', '')}_{param}_{tag}.npz"
    parts = CACHE / "parts" / f"{pre}{tag}"
    parts.mkdir(parents=True, exist_ok=True)
    scale, units = UNIT_SCALE.get(param, (1.0, None))
    if model == "gefs":
        scale, units = 1.0, ("mm" if param == "tp" else None)  # APCP is kg m-2 = mm, 6-h buckets
        full = True  # complex packing, the prefix trick only works for CCSDS
    t0 = time.time()
    grid = {}
    src = {}
    fields: dict[tuple[int, int], np.ndarray] = {}
    for step in steps:
        idx = load_index(run, step, param, mirrors, model)
        src[step] = idx["url"]
        todo = []
        for m in idx["msgs"]:
            if members is not None and m["number"] not in members:
                continue
            pf = parts / f"{param}_{step:03d}_{m['number']:02d}.npy"
            gf = parts / f"{param}_grid.npz"
            if pf.exists() and gf.exists():
                fields[(m["number"], step)] = np.load(pf)
                if not grid:
                    g = np.load(gf)
                    grid = dict(lat=g["lat"], lon=g["lon"])
            else:
                todo.append((m, pf))
        if todo:
            def job(item):
                m, pf = item
                v, lat, lon, info, nbytes = fetch_member(m.get("url", idx["url"]), m["_offset"], m["_length"], full)
                want_num = m["number"]
                got_num = int(info["number"]) if info["dataType"] == "pf" else 0
                if info["shortName"] != param or (model == "ifs-ens" and got_num != want_num):
                    raise IOError(f"index/message mismatch: {info}")
                if model == "gefs" and param == "tp" and info["stepRange"] != f"{step - 6}-{step}":
                    raise IOError(f"GEFS APCP bucket is {info['stepRange']}, use steps that are multiples of 6")
                return m, pf, (v * scale).astype(np.float32), lat, lon, info

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(job, it) for it in todo]
                for fu in as_completed(futs):
                    m, pf, v, lat, lon, info = fu.result()
                    if not grid:
                        grid = dict(lat=lat, lon=lon)
                        np.savez(parts / f"{param}_grid.npz", lat=lat, lon=lon)
                    elif v.shape != (len(grid["lat"]), len(grid["lon"])):
                        raise IOError("grid changed between messages")
                    tmp = pf.with_suffix(".tmp.npy")
                    np.save(tmp, v)
                    tmp.replace(pf)
                    fields[(m["number"], step)] = v
        if verbose:
            print(f"  step {step:3d}h: {len(idx['msgs'])} msgs, {len(todo)} downloaded, "
                  f"total {_bytes['n'] / 1e6:6.1f} MB, {time.time() - t0:5.0f}s", flush=True)
    nums = sorted({k[0] for k in fields})
    data = np.full((len(nums), len(steps), len(grid["lat"]), len(grid["lon"])), np.nan, np.float32)
    for (n, s), v in fields.items():
        data[nums.index(n), steps.index(s)] = v
    data = data[:, :, ::-1, :]  # latitude ascending
    lat = np.asarray(grid["lat"][::-1], np.float32)
    lon = np.asarray(grid["lon"], np.float32)
    meta = dict(model=MODEL_NAMES[model], run=run.strftime("%Y-%m-%dT%H:00Z"), param=param,
                units=units or "native",
                accumulated_since_run_start=param in ("tp", "ro") and model != "gefs",
                bucket_hours=6 if (model == "gefs" and param == "tp") else None,
                source=src, box=BOX, grid_step_deg=float(abs(lat[1] - lat[0])),
                bytes_downloaded_this_call=_bytes["n"], partial_messages=not full,
                created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    valid = np.array([(run + dt.timedelta(hours=s)).strftime("%Y-%m-%dT%H:00Z") for s in steps])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, data=data, members=np.array(nums, np.int16), steps=np.array(steps, np.int16),
                        lat=lat, lon=lon, valid_time=valid, meta=json.dumps(meta))
    tmp.replace(out)
    if verbose:
        print(f"  wrote {out} shape {data.shape} ({out.stat().st_size / 1e3:.0f} kB), "
              f"downloaded {_bytes['n'] / 1e6:.1f} MB in {time.time() - t0:.0f}s")
    return out


def summarize(npz: Path, point: tuple[float, float], window_h: int = 24, radius_deg: float = 0.5):
    """Print what the ensemble knew: accumulations over `window_h` at / around a point."""
    z = np.load(npz, allow_pickle=False)
    d, steps, lat, lon = z["data"], list(z["steps"]), z["lat"], z["lon"]
    meta = json.loads(str(z["meta"]))
    run = dt.datetime.strptime(meta["run"], "%Y-%m-%dT%H:00Z")
    if not (lat[0] <= point[0] <= lat[-1] and lon[0] <= point[1] <= lon[-1]):
        print(f"point {point} is outside the cropped grid, no summary")
        return
    i, j = int(np.abs(lat - point[0]).argmin()), int(np.abs(lon - point[1]).argmin())
    near = (np.abs(lat[:, None] - point[0]) <= radius_deg) & (np.abs(lon[None, :] - point[1]) <= radius_deg)
    print(f"\n{meta['model']} run {meta['run']}  {d.shape[0]} members, grid {meta['grid_step_deg']:.2f} deg; "
          f"nearest grid point to ({point[0]}, {point[1]}) = ({lat[i]:.2f}, {lon[j]:.2f})")
    print(f"  {'window (UTC)':33s} lead_h   | at point: mean  p50   p90   max  ctrl | "
          f"max within +-{radius_deg} deg: mean  p90   max | box max: mean  max | P(pt>=40) P(area>=100)")
    bucket = meta.get("bucket_hours")
    pairs = []
    if bucket:  # GEFS: each step holds one 6-h bucket -> sum the buckets inside the window
        for sb in steps:
            need = list(range(sb - window_h + bucket, sb + 1, bucket))
            if all(s in steps for s in need):
                pairs.append((sb - window_h, sb, sum(d[:, steps.index(s)] for s in need)))
    else:
        pairs = [(sa, sb, d[:, b] - d[:, a]) for a, sa in enumerate(steps) for b, sb in enumerate(steps)
                 if sb - sa == window_h]
    for sa, sb, acc in pairs:
        if True:
            pt = acc[:, i, j]
            ar = acc[:, near].max(axis=1)
            bx = acc.reshape(acc.shape[0], -1).max(axis=1)
            ta, tb = run + dt.timedelta(hours=int(sa)), run + dt.timedelta(hours=int(sb))
            c = list(z["members"]).index(0) if 0 in z["members"] else 0
            print(f"  {ta:%Y-%m-%d %HZ} -> {tb:%Y-%m-%d %HZ}  {int(sa):3d}-{int(sb):3d} | "
                  f"{pt.mean():13.1f} {np.percentile(pt, 50):5.1f} {np.percentile(pt, 90):5.1f} {pt.max():5.1f} {pt[c]:5.1f} | "
                  f"{ar.mean():28.1f} {np.percentile(ar, 90):5.1f} {ar.max():5.1f} | {bx.mean():13.1f} {bx.max():5.1f} | "
                  f"{(pt >= 40).mean():8.2f} {(ar >= 100).mean():12.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run time UTC, e.g. 2024-10-27T00 (00/12 go to 360 h; 06/18 to 144 h)")
    ap.add_argument("--steps", required=True, help="comma list of forecast hours, e.g. 48,54,60,66,72")
    ap.add_argument("--out", default=None, help="output .npz (default hindcast/cache/ens/ifsens_<param>_<run>.npz)")
    ap.add_argument("--param", default="tp", help="surface param short name in the index (tp, cape, tcwv, 2t...)")
    ap.add_argument("--mirror", default="gcs,aws", help="mirror order (gcs,aws)")
    ap.add_argument("--workers", type=int, default=4, help="parallel range requests (keep small)")
    ap.add_argument("--full", action="store_true", help="download whole messages (no prefix trick)")
    ap.add_argument("--members", default=None, help="subset, e.g. 0,1,2 (0 = control)")
    ap.add_argument("--point", default=None, help="lat,lon to summarise, e.g. 39.47,-0.72")
    ap.add_argument("--window", type=int, default=24, help="accumulation window in hours for the summary")
    ap.add_argument("--model", default="ifs-ens", choices=sorted(MODEL_NAMES),
                    help="ifs-ens (default, 51 members), ifs-hres (deterministic 0.25/0.4 deg, all leads to 240 h), "
                         "gefs (NOAA, 31 members, tp = 6-h buckets, steps must be multiples of 6)")
    a = ap.parse_args(argv)
    run = dt.datetime.strptime(a.run, "%Y-%m-%dT%H")
    steps = sorted(int(s) for s in a.steps.split(","))
    members = [int(x) for x in a.members.split(",")] if a.members else None
    print(f"{MODEL_NAMES[a.model]} {a.param} run {run:%Y-%m-%d %HZ} steps {steps}")
    out = fetch_run(run, steps, a.param, Path(a.out) if a.out else None, a.mirror.split(","),
                    a.workers, a.full, members, model=a.model)
    if a.point:
        la, lo = (float(x) for x in a.point.split(","))
        summarize(out, (la, lo), a.window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
