"""Shared paths and helpers for the h2-sections pipeline."""
import json
import os

import numpy as np
from pyproj import Transformer

ROOT = r"C:\Users\cpereiro\IdeaProjects\riua"
SEED = os.path.join(ROOT, "geo", "hydro", "control_points_seed.json")
SNAPPED = os.path.join(ROOT, "geo", "hydro", "catchments", "out", "control_points.json")
HERE = os.path.join(ROOT, "geo", "hydro", "sections")
BUILD = os.path.join(HERE, "build")
OUT = os.path.join(HERE, "out")
PROFILES = os.path.join(OUT, "profiles")
SCRATCH = os.path.join(ROOT, "scratch", "h2-sections")
DTM_DIR = os.path.join(SCRATCH, "dtm")
OSM_DIR = os.path.join(SCRATCH, "osm")
PLOTS = os.path.join(SCRATCH, "plots")
MISC = os.path.join(SCRATCH, "misc")
UA = "Mozilla/5.0 (riua-research; flood-forecast study; contact via github ces107)"

for _d in (OUT, PROFILES, DTM_DIR, OSM_DIR, PLOTS, MISC):
    os.makedirs(_d, exist_ok=True)

_to_utm = Transformer.from_crs(4326, 25830, always_xy=True)
_to_ll = Transformer.from_crs(25830, 4326, always_xy=True)


def ll2utm(lon, lat):
    return _to_utm.transform(lon, lat)


def utm2ll(x, y):
    """returns lon, lat"""
    return _to_ll.transform(x, y)


def load_seed():
    """Control points: the final snapped list of h1-catchments (60 points) when present, else the seed."""
    if os.path.exists(SNAPPED):
        with open(SNAPPED, encoding="utf-8") as f:
            return json.load(f)["points"]
    with open(SEED, encoding="utf-8") as f:
        return json.load(f)["points"]


def load_config():
    """Per-point analyst configuration (stream name regex, lining, manual section centre...)."""
    with open(os.path.join(BUILD, "points_config.json"), encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=conv)
    os.replace(tmp, path)
