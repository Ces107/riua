"""web/geo/zi500/index.json: the tiles that exist, so the page asks only for those.

    py -3.11 geo/floodzones/build_index.py
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "web" / "geo" / "zi500"
idx = {z.name: sorted(f"{x.name}/{f.stem}" for x in z.iterdir() for f in x.glob("*.png"))
       for z in sorted(D.iterdir(), key=lambda p: int(p.name)) if z.is_dir()}
(D / "index.json").write_text(json.dumps(idx, separators=(",", ":")), encoding="utf-8")
print({z: len(v) for z, v in idx.items()})
