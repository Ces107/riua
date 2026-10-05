"""Everything of hindcast/ml in one command, from cached data (no network unless a cache file is missing):

    py -3.11 hindcast/ml/run_all.py          # ~6 min: dataset (3.5 min), GloFAS score, evaluation, final model + report
"""
import runpy
from pathlib import Path

H = Path(__file__).resolve().parent
for s in ("build_dataset", "glofas_score", "evaluate", "train"):
    print(f"\n######## {s}")
    try:
        runpy.run_path(str(H / f"{s}.py"), run_name="__main__")
    except SystemExit as e:
        if e.code:
            raise
