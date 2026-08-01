#!/usr/bin/env bash
set -euo pipefail

# Git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE to hook processes, and
# those take precedence over `git -C <dir>` for repository discovery. This
# script runs from the `cargo-test-unit` pre-push hook, so without clearing
# them every `git -C "$TARGET_DIR" ...` below silently targets the *enclosing*
# repository instead of the test-data clone — and `reset --hard "@{u}"` then
# resets the developer's branch to its upstream, discarding committed work and
# the working tree with it.
#
# It hides on master, where `@{u}` == HEAD makes the reset a no-op. It only
# fires on a branch with unpushed commits, i.e. exactly when there is work to
# lose. Observed: two commits destroyed mid-push before the cause was found.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_OBJECT_DIRECTORY

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
    # Belt and braces for the hook-environment hazard above: refuse to touch
    # anything that is not the test-data clone's own root. `reset --hard` on
    # the wrong repository is unrecoverable from the working tree, so verify
    # rather than trust.
    toplevel="$(git -C "$TARGET_DIR" rev-parse --show-toplevel)"
    if [ "$toplevel" != "$(cd "$TARGET_DIR" && pwd)" ]; then
        echo "refusing to update: '$TARGET_DIR' resolves to repository" \
             "'$toplevel', not itself. Leaked git env (GIT_DIR=${GIT_DIR:-unset})?" >&2
        exit 1
    fi
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
