#!/usr/bin/env bash
# Publish web/ + the latest product to the gh-pages branch (GitHub Pages).
# Usage (WSL):  deploy/publish.sh <product_dir>
set -euo pipefail
REPO=/mnt/c/Users/cpereiro/IdeaProjects/riua
PAGES=/mnt/c/Users/cpereiro/IdeaProjects/riua-pages
OUT=${1:-$REPO/scratch/out}
GIT=git.exe   # Windows git: uses the Windows credential manager

if [ ! -d "$PAGES/.git" ]; then
  cd /mnt/c/Users/cpereiro/IdeaProjects
  $GIT clone -q https://github.com/Ces107/riua.git riua-pages
  cd "$PAGES"
  $GIT checkout -q --orphan gh-pages
  $GIT rm -rq --cached . >/dev/null 2>&1 || true
  find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
fi
cd "$PAGES"
# site files (no dev fixtures)
rsync -a --delete --exclude .git --exclude dev/ --exclude data/ "$REPO/web/" "$PAGES/"
mkdir -p data
cp "$OUT/snapshot.json" "$OUT"/explain-*.bin data/
[ -f "$OUT/og.png" ] && cp "$OUT/og.png" data/
touch .nojekyll
$GIT add -A .
if $GIT diff --cached --quiet; then echo "nothing to publish"; exit 0; fi
$GIT -c user.name="César Pereiro" -c user.email="cesar0407p@gmail.com" commit -q -m "producto $(date -u +%Y-%m-%dT%H:%MZ)"
# keep the branch small: one commit only
$GIT reset -q --soft "$($GIT rev-list --max-parents=0 HEAD | tail -1)" 2>/dev/null || true
$GIT -c user.name="César Pereiro" -c user.email="cesar0407p@gmail.com" commit -q --amend -m "producto $(date -u +%Y-%m-%dT%H:%MZ)" 2>/dev/null || true
$GIT push -q -f origin gh-pages
echo "published $(date -u +%H:%MZ)"
