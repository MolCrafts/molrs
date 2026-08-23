#!/usr/bin/env bash
set -euo pipefail

# Re-resolve every native workspace root's Cargo.lock in ONE registry sweep.
#
# ── HAZARD (unit-identity axis 1: lockfile version drift) ────────────────────
# The shared-dylib invariant needs all native roots to compile the SAME molrs
# unit. cargo's `-C metadata` fingerprint is RECURSIVE: it mixes in the
# fingerprints of every dependency unit, so molrs's identity moves if any crate
# anywhere in its closure resolves to a different version.
#
# Every root here keeps its own Cargo.lock, and those locks are deliberately
# UNTRACKED (see .gitignore) — binders and molpack are libraries, and committing
# their locks would state a release-level fact to fix a machine-local problem.
# Untracked means each lock is written whenever that root first got built, i.e.
# at a different registry moment. Measured drift: molrs-ffi's lock pinned
# log 0.4.34 while molrs-python's still pinned log 0.4.33 — one patch bump, and
# all 54 shared crates fingerprinted differently, so the two roots built two
# distinct molrs units writing the same hashless deps/libmolrs_ffi.dylib. The
# second build silently OVERWRITES the first; nothing errors.
#
# The fix is therefore SYNCHRONISATION, not tracking: delete all locks, then
# regenerate them back to back against one registry snapshot. `cargo metadata`
# is the cheapest lock-writing command (it resolves and writes the lock without
# compiling anything).
#
# Failure policy: refuse and explain (as scripts/fetch-test-data.sh does). A
# partial sweep leaves exactly the skew this script exists to remove, so a
# missing molpack checkout is a hard failure, never a silent skip.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Locate the sibling molpack checkout (same resolution as
#    scripts/verify-shared-dylib.sh, so both gates agree on one molpack) ──────
MOLPACK_ROOT="${MOLPACK_ROOT:-$PROJECT_ROOT/../molpack}"

if [ ! -d "$MOLPACK_ROOT" ] || [ ! -f "$MOLPACK_ROOT/python/Cargo.toml" ]; then
    echo "sync-dylib-locks: cannot find a usable molpack checkout." >&2
    echo "  looked for directory: $MOLPACK_ROOT" >&2
    echo "  requiring manifest:   $MOLPACK_ROOT/python/Cargo.toml" >&2
    echo >&2
    echo "molpack's two roots are part of the shared-dylib graph, so syncing" >&2
    echo "only the molrs-side locks would leave the very version skew this" >&2
    echo "script removes — a green run that fixed nothing." >&2
    echo >&2
    echo "Point it at an existing checkout:" >&2
    echo "  MOLPACK_ROOT=/path bash scripts/sync-dylib-locks.sh" >&2
    echo "or clone it next to molrs:" >&2
    echo "  git clone https://github.com/MolCrafts/molpack" >&2
    exit 1
fi
MOLPACK_ROOT="$(cd "$MOLPACK_ROOT" && pwd)"

# The seven native roots of the dynamic graph.
#
# molrs-wasm is DELIBERATELY excluded: it is the existing browser exemption
# (`default-features = false` for bundle size), it carries no
# `[profile.release]` of its own, and it builds for wasm32, which has no dynamic
# linking and so can never take the opt-in dynamic link form (docs/interop.md).
# Its molrs unit is SUPPOSED to diverge, so dragging it into the sweep would add
# a unit that can never converge and invite someone to "fix" the divergence by
# widening the browser bundle.
ROOTS=(
    "$PROJECT_ROOT"                 # molrs root workspace (molrs itself)
    "$PROJECT_ROOT/molrs-ffi"       # the dylib host
    "$PROJECT_ROOT/molrs-python"    # molrs wheel
    "$PROJECT_ROOT/molrs-capi"      # C API
    "$PROJECT_ROOT/molrs-cxxapi"    # CXX bridge (own workspace since axis 4)
    "$MOLPACK_ROOT"                 # molpack root
    "$MOLPACK_ROOT/python"          # molpack wheel
)

echo "sync-dylib-locks: molrs   root = $PROJECT_ROOT"
echo "sync-dylib-locks: molpack root = $MOLPACK_ROOT"

for root in "${ROOTS[@]}"; do
    if [ ! -f "$root/Cargo.toml" ]; then
        echo "sync-dylib-locks: no manifest at $root/Cargo.toml" >&2
        echo "  Every root listed in this script must exist; a skipped root" >&2
        echo "  keeps its stale lock and re-introduces the skew." >&2
        exit 1
    fi
done

# ── Delete first, regenerate second ─────────────────────────────────────────
# Deleting ALL locks before regenerating ANY is the point: a root re-resolved
# while another root still holds a stale lock would just be re-pinned to that
# same stale set the next time the stale root gets touched.
echo "sync-dylib-locks: deleting ${#ROOTS[@]} Cargo.lock files"
for root in "${ROOTS[@]}"; do
    rm -f "$root/Cargo.lock"
done

# Back to back, so every root resolves against ONE registry snapshot: the first
# invocation refreshes the index, the rest read that same cache seconds later.
echo "sync-dylib-locks: regenerating in one registry sweep"
for root in "${ROOTS[@]}"; do
    echo "  resolving $root"
    cargo metadata --format-version 1 --manifest-path "$root/Cargo.toml" >/dev/null
done

echo "sync-dylib-locks: done — ${#ROOTS[@]} locks re-resolved at one registry moment."
echo "Next: rm -rf $PROJECT_ROOT/target/debug && bash scripts/verify-shared-dylib.sh"
