#!/usr/bin/env bash
set -euo pipefail

# Shared, binding-neutral test data: cloned once to the workspace root so every
# crate (and the Python / C / WASM bindings) resolves the same copy, and
# `cargo clean` does not wipe it.
#
# Never delete an existing clone: if a valid one is present, just update it to
# the latest (so newly added fixtures land without a full re-clone). Only clone
# from scratch when it is missing or broken. Point MOLRS_TESTS_DATA at an
# existing copy to share it.
REPO_URL="https://github.com/MolCrafts/tests-data.git"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${MOLRS_TESTS_DATA:-$PROJECT_ROOT/tests-data}"

# Present and valid → refresh in place (fetch the latest tip, fast-forward to
# it). A broken/partial clone (from an interrupted run) falls through to a
# clean re-clone below.
if [ -d "$TARGET_DIR/.git" ] && git -C "$TARGET_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "Test data present at $TARGET_DIR — updating to latest."
    git -C "$TARGET_DIR" fetch --depth=1 origin
    git -C "$TARGET_DIR" reset --hard "@{u}"
    exit 0
fi

echo "Cloning test data to $TARGET_DIR..."
rm -rf "$TARGET_DIR"   # only reached when missing/broken
mkdir -p "$(dirname "$TARGET_DIR")"
git clone --depth=1 "$REPO_URL" "$TARGET_DIR"
echo "Done. Run: cargo test"
