"""CLI wrapper: the downloader lives in backend/riua/sources/ecmwf_open.py (shared with the live pipeline)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from riua.sources.ecmwf_open import main  # noqa: E402

if __name__ == "__main__":
    main()
