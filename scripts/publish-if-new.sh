#!/usr/bin/env bash
# Publish a Rust crate to crates.io, but skip if the same version is already
# there. Lets the publish workflow be re-driven (workflow_dispatch / re-run)
# without choking on "crate version already uploaded" errors when a previous
# run published some sub-crates and then failed downstream.
#
# Usage:
#   bash scripts/publish-if-new.sh <manifest-path> <crate-name>
#
# Reads $CARGO_REGISTRY_TOKEN from the env (set by the workflow step).

set -euo pipefail

MANIFEST="${1:?manifest path required}"
CRATE="${2:?crate name required}"

VERSION=$(cargo metadata --no-deps --format-version 1 --manifest-path "$MANIFEST" \
  | python3 -c "import sys, json; pkgs = json.load(sys.stdin)['packages']; print(pkgs[0]['version'])")

# Prefer GET: crates.io often returns 404/405 on HEAD even when the version exists.
if curl -sf "https://crates.io/api/v1/crates/$CRATE/$VERSION" >/dev/null; then
  echo "✓ $CRATE@$VERSION already on crates.io; skipping publish"
  exit 0
fi

echo "→ publishing $CRATE@$VERSION"
set +e
OUT=$(cargo publish --manifest-path "$MANIFEST" 2>&1)
CODE=$?
set -e
printf '%s\n' "$OUT"
if [ "$CODE" -eq 0 ]; then
  exit 0
fi
# Idempotent re-run: race / index lag after a previous successful publish.
if printf '%s\n' "$OUT" | grep -qiE 'already exists|already uploaded'; then
  echo "✓ $CRATE@$VERSION already on crates.io (cargo reported exists); treating as success"
  exit 0
fi
exit "$CODE"
