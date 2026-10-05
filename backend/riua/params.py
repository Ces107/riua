"""Every tunable number of the risk model, in one place.

The defaults below are the starting point; `hindcast/tune.py` overwrites the tuned ones
in `backend/riua/params.json`, which is what the server loads. Nothing else in the code
base hides a threshold.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

PARAMS_FILE = Path(__file__).with_name("params.json")

HORIZONS = ("now", "mid", "long")

DEFAULTS: dict = {
    "version": "untuned-defaults",
    # --- Decision rule -------------------------------------------------------------
    # A level L is issued when P(>= L) reaches tau[L]. Still slightly asymmetric (a
    # missed severe event costs more than a false alarm, cost/loss: threshold = C/L),
    # but never below 40 %: below that the product cries wolf, which costs trust.
    "tau": {
        "now":  {"2": 0.60, "3": 0.55, "4": 0.50, "5": 0.45},
        "mid":  {"2": 0.55, "3": 0.50, "4": 0.45, "5": 0.40},
        "long": {"2": 0.50, "3": 0.45, "4": 0.40, "5": 0.40},
    },
    # --- What "extreme" means (level 5) ----------------------------------------------
    # Cell scale: accumulation at least this multiple of the zone's RED threshold.
    "extreme": {
        "x1h": 1.5,     # 1-h accumulation >= 1.5 x red (135 mm where red is 90 mm)
        "x12h": 5 / 3,  # 12-h accumulation >= 5/3 x red (300 mm where red is 180 mm)
        # Basin scale: estimated unit peak discharge at least this fraction of the
        # envelope of the largest flash floods observed in the Mediterranean,
        # q_env = 100 * A^-0.4 m3/s/km2 (Gaume et al. 2009).
        "envelope_fraction": 0.6,     # same as level 5 at the control points
        "envelope_c": 100.0,
        "envelope_exp": -0.4,
    },
    # --- Uncertainty dressing --------------------------------------------------------
    # Used while no fitted calibration exists (params["calibration"] empty).
    # Each scenario's hazard ratio r = s * amount / threshold is the median of a
    # log-normal error: P(exceed | scenario) = Phi(ln(bias * r) / sigma);
    # P(>= L) = weighted mean over scenarios. sigma 0.3 = a scenario at 0.7 x the
    # threshold still gives 12 %, one at 1.4 x gives 87 %.
    "sigma": {"now": 0.25, "mid": 0.30, "long": 0.35},
    "bias": {"now": 1.0, "mid": 1.0, "long": 1.0},
    # --- Neighbourhood radius (km) by horizon, for convection-permitting members -----
    # A 12-km maximum already covers a whole Riuà cell plus a typical 6-12 h displacement
    # error; larger radii inflate every amount relative to what falls in one cell.
    "radius_km": {"now": 6.0, "mid": 12.0, "long": 12.0},
    # Displacement applied to each convection-permitting member for basin means (km).
    # a storm's position is uncertain by more than a small catchment: 10 km within 6 h, 20 km a day ahead
    "basin_shift_km": {"now": 10.0, "mid": 20.0, "long": 0.0},
    # --- Model families ---------------------------------------------------------------
    # weight: relative weight of ONE member of that family, before run-age decay.
    # s1h / s12h: multiplicative representativeness factors applied to the model's
    # 1-h and 12-h amounts (coarse models cannot produce point-scale convective peaks).
    # use_1h: whether the 1-h criterion is evaluated at all for that family.
    "families": {
        "radar":    {"weight": 0.15, "s1h": 1.0, "s12h": 1.0, "use_1h": True,  "neigh": True},
        "cp":       {"weight": 1.0, "s1h": 1.0, "s12h": 1.0, "use_1h": True,  "neigh": True,  "sampled": True},
        "regional": {"weight": 0.6, "s1h": 1.6, "s12h": 1.2, "use_1h": True,  "neigh": True, "sampled": True},
        "global":   {"weight": 0.5, "s1h": 2.2, "s12h": 1.3, "use_1h": False, "neigh": True, "sampled": True},
        "ens":      {"weight": 0.04, "s1h": 2.2, "s12h": 1.5, "use_1h": False, "neigh": True},
        # regional ensemble at ~13 km (ICON-EU-EPS): finer than the 25-km ENS, so a smaller factor
        "eps":      {"weight": 0.04, "s1h": 1.8, "s12h": 1.3, "use_1h": False, "neigh": True},
    },
    # radius used only to bridge the 0.1 deg sampling lattice for members without neighbourhood
    "lattice_fill_km": 8.0,
    # rain already measured counts fully in a 12-h amount when the scenario still brings this much (mm)
    "obs_gate_mm": 5.0,       # hindcast (q8): 5 mm scores 0.47 against 0.34 for 20 mm in 0-6 h
    "obs12_factor": 1.0,      # 0.88 if the measured hours were hourly cell maxima; they are pixel-consistent increments
    "sigma_obs": 0.15,
    # the 1-h criterion with its own kernel a day ahead: better in 7 of 7 hindcast cases
    "sigma1h": {"mid": 1.0}, "bias1h": {"mid": 1.0},
    # days 2-7: no skill above "alto" outside the 2024 DANA (BSS about 0): the level stops at 3
    "level_cap": {"long": 3},
    # at least 2 different models (an ensemble counts as one when 30 % of its weight agrees) reaching the red or
    # extreme threshold outright raise the level to 4 or 5, in every horizon
    "agreement": {"on": True, "models": 2, "ens_share": 0.3},
    "age_halflife_h": {"now": 3.0, "mid": 12.0, "long": 36.0},
    # --- Hydrology ----------------------------------------------------------------------
    "hydro": {
        # losses E = (W - P0)^2 / (W - P0 + S), fitted to measured SAIH flows in 8 gauged catchments
        # over 10 rain episodes plus the Poyo flood of 29 Oct 2024 (hindcast/calibrate_hydro.py).
        # The design value P0 = 25 mm gave 146 m3/s on the Poyo for an episode that measured 0.5.
        # 26 ravine-like gauged catchments, 200 measured floods in 22 episodes + the 2024 anchors: one regional
        # threshold cannot be right (the spread is between catchments), a per-catchment quick share is
        # (leave-one-episode-out error of ln peak 0.58 -> 0.38; floods hit / missed 20 / 53 -> 42 / 31).
        "p0_mm": 150.0,       # runoff threshold of most of the ground
        "s_mm": 150.0,        # retention still to fill above the threshold
        "ml_vote": True,      # max(physical, logistic vote on rain arriving and simulated peak) on levels 2-3 only
        "p0b_mm": 10.0, "sb_mm": 100.0,   # the quick share alpha of each catchment (geo/hydro/loss_params.json)
        # intensity excess: rain above this rate on a 5-km cell-hour runs off even on dry ground
        # (flow starts after 10 mm at 139 mm/h, after 72 mm at 57 mm/h: Camarasa-Belmonte 2021).
        # Costs nothing on the ordinary episodes and adds 160-300 m3/s to the Poyo of 2024.
        "phi_mmh": 45.0,
        "tc_a": 0.5,          # response time tc = tc_a * A^tc_b hours (A in km2)
        "tc_b": 0.38,
        "tc_max_h": 24.0,
        "durations_h": [1, 3, 6, 12],
        # control points (rain -> discharge -> water level)
        "wet_memory_h": 72.0,  # e-folding time of the soil's memory of earlier rain
        "clark_k": 0.3,        # linear-reservoir constant as a fraction of tc
        "f2": 0.25, "f3": 0.6, "f5": 2.0,   # level starts, as fractions of channel capacity
        "d5_m": 1.0,           # level 5: water this far above the bank crest
        # points without a usable channel capacity: levels 2..5 at these fractions of the
        # envelope Q = env_c * A^(1+env_e) (Gaume et al. 2009, unit q = 100 A^-0.4).
        # 2024 Poyo at the A-3 gauge: 2283 m3/s on 184 km2, i.e. 1.0 x the envelope.
        "env_c": 100.0, "env_e": -0.4, "env_fractions": [0.08, 0.18, 0.35, 0.6],
        # points without a usable capacity but with CAUMAX flood quantiles (all 60 today):
        # level 2 at the 2-year flood, 3 at 5 years, 4 at 25 years, 5 at 100 years (EFAS-like)
        "rp_levels": [2, 5, 25, 100],
        # spread of ln(discharge): the fit leaves a residual of 0.71, larger than the rain uncertainty
        "sigma": {"now": 0.7, "mid": 0.8, "long": 0.9},
        # sub-hourly bursts (coord/findings/q12-subhourly.md): OFF. A number turns it on: the intensity excess is then the
        # rain above phi_sub mm/h inside the hour from the radar rate frames (radar/qpe.py::burst_ladder), not the hourly
        # mean above phi_mmh. Tested on 200 ordinary floods + q9's big floods: no phi lifts the big floods without
        # worsening the ordinary ones (best candidate 80: >= 100 m3/s median x0.39 -> x0.43, ordinary error 0.378 -> 0.400).
        "phi_sub_mmh": None,
        # forecast hours without a radar ladder: X = p G(phi_sub / p), mean over 1.1 M radar cell-hours 2024-26
        "burst_rel": [[0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0],
                      [0.651, 0.553, 0.427, 0.300, 0.183, 0.104, 0.0527, 0.0260, 0.0119, 0.0059, 0.0023, 0.0010, 0.0003]],
    },
}


def load() -> dict:
    p = copy.deepcopy(DEFAULTS)
    if PARAMS_FILE.exists():
        _merge(p, json.loads(PARAMS_FILE.read_text(encoding="utf-8")))
    return p


def save(p: dict) -> None:
    PARAMS_FILE.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _merge(base: dict, over: dict) -> None:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
