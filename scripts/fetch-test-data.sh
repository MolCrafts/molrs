#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/MolCrafts/tests-data.git"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/molrs-core/target/tests-data"

echo "Fetching test data to $TARGET_DIR..."

mkdir -p "$(dirname "$TARGET_DIR")"
rm -rf "$TARGET_DIR"
git clone --depth=1 "$REPO_URL" "$TARGET_DIR"

echo "Done. Run: cargo test"
