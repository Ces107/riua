"""Process points one by one and rewrite out/sections.json after EACH point (live pipeline reads it).
Usage: py -3.11 run_points.py id [id ...]
"""
import subprocess
import sys

for pid in sys.argv[1:]:
    r = subprocess.run([sys.executable, "sections.py", pid], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    print(lines[-1][:240] if lines else pid + " no output " + r.stderr[-300:], flush=True)
    a = subprocess.run([sys.executable, "assemble.py", "--fast"], capture_output=True, text=True)
    row = [l for l in a.stdout.splitlines() if l.startswith(pid + " ")]
    print("   assembled:", row[0][:160] if row else a.stderr[-300:], flush=True)
