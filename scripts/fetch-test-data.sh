#!/usr/bin/env bash
set -euo pipefail

# Shared, binding-neutral test data: cloned once to the workspace root so every
# crate (and the Python / C / WASM bindings) resolves the same copy, and
# `cargo clean` does not wipe it.
#
# Fetch-once: if a valid clone is already present it is reused. Pre-push therefore
# does NOT re-clone on every run (the old `rm -rf` + clone was slow, network-flaky,
# and left no data if the clone failed mid-way). To force a refresh, delete the
# directory; to share an existing copy, point MOLRS_TESTS_DATA at it.
REPO_URL="https://github.com/MolCrafts/tests-data.git"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${MOLRS_TESTS_DATA:-$PROJECT_ROOT/tests-data}"

# Reuse only a *valid* clone; a broken/partial one (from an interrupted run)
# must be re-fetched, not reused.
if [ -d "$TARGET_DIR/.git" ] && git -C "$TARGET_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "Test data already present at $TARGET_DIR — reusing."
    exit 0
fi

echo "Fetching test data to $TARGET_DIR..."
rm -rf "$TARGET_DIR"   # drop any broken/partial clone before re-fetching
mkdir -p "$(dirname "$TARGET_DIR")"
git clone --depth=1 "$REPO_URL" "$TARGET_DIR"
echo "Done. Run: cargo test"
