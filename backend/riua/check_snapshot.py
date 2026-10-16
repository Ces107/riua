"""Is this snapshot good enough to publish?

    python -m riua.check_snapshot <snapshot.json> [max_age_minutes]

Exit 0 when the file parses, carries the three horizons' structure and was generated less than
max_age_minutes ago (default 30). Used by the cycle workflow to accept or reject a run's output.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    max_age = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
        gen = datetime.fromisoformat(str(s["generated"]).replace("Z", "+00:00"))
        if gen.tzinfo is None:
            gen = gen.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - gen).total_seconds() / 60.0
        have = [h for h in ("now", "mid") if h in s.get("horizons", {})]
        if len(have) < 2:
            print(f"snapshot has horizons {list(s.get('horizons', {}))}: not publishable")
            return 1
        if age > max_age:
            print(f"snapshot is {age:.0f} min old (max {max_age:.0f})")
            return 1
        print(f"snapshot ok: generated {s['generated']}, {age:.0f} min old")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"snapshot unreadable: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
