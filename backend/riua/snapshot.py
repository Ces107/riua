"""The snapshot: one compact JSON document with everything the web page shows.

It is what the API serves at /v1/snapshot and what gets mirrored to the static site, so
the page works from a CDN even when the server sleeps. Arrays over grid cells only carry
the cells inside the domain mask, in row-major order (south to north, west to east), and
are packed as base64 bytes:

  level        uint8, 0 = no data, 1..5
  probability  uint8, value = round(P * 200)            (0.5 % steps)
  rain amount  uint8, value = round(8 * sqrt(mm))       (0..1016 mm, ~1 mm steps at 16 mm)

Small per-basin / per-point tables are plain JSON numbers.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone

import numpy as np

SNAPSHOT_VERSION = 1


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def code_prob(p: np.ndarray) -> np.ndarray:
    # floor: a published probability never reads as having reached a minimum that the exact value did not
    return np.clip(np.floor(np.nan_to_num(p) * 200.0 + 1e-6), 0, 200).astype(np.uint8)


def code_mm(mm: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(8.0 * np.sqrt(np.maximum(np.nan_to_num(mm), 0.0))), 0, 255).astype(np.uint8)


def iso(t) -> str:
    if isinstance(t, np.datetime64):
        return str(t.astype("datetime64[m]")) + "Z"
    if isinstance(t, datetime):
        return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ") if t.tzinfo else t.strftime("%Y-%m-%dT%H:%MZ")
    return str(t)


def r(a, nd=1):
    """ndarray -> nested lists of rounded floats (NaN -> None)."""
    a = np.asarray(a, np.float64)
    out = np.round(a, nd)
    if nd == 0:
        return [None if not np.isfinite(x) else int(x) for x in out.ravel()] if a.ndim == 1 else [r(x, nd) for x in a]
    return [None if not np.isfinite(x) else float(x) for x in out.ravel()] if a.ndim == 1 else [r(x, nd) for x in a]
