#!/usr/bin/env bash
# Production loop on this machine (WSL), until the GitHub Actions cron takes over:
# every 10 minutes compute the product and publish it to GitHub Pages.
#   nohup deploy/loop.sh > /tmp/riua-loop.log 2>&1 &
REPO=/mnt/c/Users/cpereiro/IdeaProjects/riua
exec >> "$HOME/riua-loop.log" 2>&1
OUT=$HOME/riua-out
mkdir -p "$OUT"
while true; do
  start=$(date +%s)
  cd "$REPO/backend"
  if timeout 1500 ~/riua-venv/bin/python -m riua.product --state ~/riua-state --out "$OUT"; then
    ~/riua-venv/bin/python -c "import json,pathlib; from riua import card; card.from_snapshot(json.loads(pathlib.Path('$OUT/snapshot.json').read_text(encoding='utf-8')), pathlib.Path('$OUT/og.png'))" || true
    bash "$REPO/deploy/publish.sh" "$OUT" || echo "publish failed"
  else
    echo "cycle failed $(date -u +%H:%MZ)"
  fi
  left=$(( 600 - ($(date +%s) - start) ))
  [ $left -gt 0 ] && sleep $left
done
