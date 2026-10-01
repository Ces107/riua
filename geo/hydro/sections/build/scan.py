"""QA helper: bankfull geometry every 50 m along the traced thalweg of one point, to choose a
representative chainage (config offset_m). Usage: py -3.11 scan.py <id> [step]
"""
import json
import os
import sys

import numpy as np
import rasterio

import hydraulics as H
import sections as S
from common import DTM_DIR, MISC, load_config


def main():
    pid = sys.argv[1]
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    cfg = load_config()[pid]
    with open(os.path.join(MISC, "anchors.json"), encoding="utf-8") as f:
        anchor = json.load(f)[pid]
    with rasterio.open(os.path.join(DTM_DIR, pid + ".tif")) as d:
        arr, tr = d.read(1), d.transform
    th = S.trace_thalweg(anchor, arr, tr, int(cfg.get("snap_w", 40)))
    br = S.bridge_chainages(pid, th)
    print("flipped", th["flipped"], "bridges at", [round(b) for b in br] if br else br)
    lining = cfg.get("lining", "natural")
    c = {k: v for k, v in cfg.items() if not k.startswith("bank_")}
    for ch in np.arange(np.ceil(th["chain"][0] / step) * step, th["chain"][-1], step):
        sl = S.slopes(th, ch)
        Sx = sl[500] if np.isfinite(sl[500]) and sl[500] > 2e-4 else 2e-4
        sec = S.cut_section(th, ch, arr, tr, int(cfg.get("half_len", 450)))
        r = S.analyse(sec, H.N_TABLE[lining], Sx, Sx, Sx, c, H.FR_CAP[lining])
        if r is None:
            print(f"{ch:6.0f}  no data")
            continue
        i = sec["i"]
        print(f'{ch:6.0f} zth {th["z"][i]:7.2f} cov {int(th["covered"][i])} S {Sx:.4f} depth {r["depth"]:5.1f} W {r["top_width"]:5.0f} '
              f'L {r["i0"] - r["iL"]:4d} R {r["iR"] - r["i0"]:4d} zL {r["zL"]:7.2f} zR {r["zR"]:7.2f} Qbf {r["q_bf"]:7.0f}')


if __name__ == "__main__":
    main()
