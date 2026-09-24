#!/usr/bin/env bash
# Run only the molrs unit tests that belong to the modules you touched.
#
#   scripts/test-scope.sh                            # modules changed vs HEAD (incl. untracked)
#   scripts/test-scope.sh origin/dev                 # modules changed vs a revision
#   scripts/test-scope.sh ff::potential              # explicit module path(s)
#   scripts/test-scope.sh molrs/src/io/data/xyz.rs   # explicit file path(s)
#
# Why a script and not a hand-typed `cargo test <module>`: the feature string
# must never vary. Every run here goes through `cargo mrs-test`
# (see .cargo/config.toml), so the crate is compiled once and each later run is
# a filter over the same test binary — ~0.1 s instead of a 67 s rebuild. A bare
# `cargo test <module>` resolves a different feature set (no stream/serde, plus
# the doctest and bin targets) and recompiles all 293k lines.
#
# Filters are libtest substrings, not anchored patterns, so a scope can pull in
# a few unrelated tests whose path happens to contain the same text. It can
# never drop a test in a module you changed, which is the direction that
# matters. The full gate is still `cargo mrs-test && cargo mrs-doctest`.
set -euo pipefail
cd "$(dirname "$0")/.."

# molrs/src/ff/potential/lj.rs -> ff::potential::lj ; molrs/src/ff/mod.rs -> ff
to_module() {
    sed -e 's#^molrs/src/##' -e 's#\.rs$##' -e 's#/mod$##' -e 's#/#::#g'
}

paths=()
modules=()
rev=""
for arg in "$@"; do
    case "$arg" in
        *::*) modules+=("$arg") ;;
        *.rs | molrs/*) paths+=("$arg") ;;
        *) rev="$arg" ;;
    esac
done

if [ ${#modules[@]} -eq 0 ] && [ ${#paths[@]} -eq 0 ]; then
    while IFS= read -r f; do
        [ -n "$f" ] && paths+=("$f")
    done < <(
        {
            git diff --name-only "${rev:-HEAD}" -- molrs/src
            git ls-files --others --exclude-standard -- molrs/src
        } | sort -u
    )
fi

if [ ${#paths[@]} -gt 0 ]; then
    while IFS= read -r m; do
        [ -n "$m" ] && modules+=("$m")
    done < <(printf '%s\n' "${paths[@]}" | grep '^molrs/src/.*\.rs$' | to_module | sort -u)
fi

if [ ${#modules[@]} -eq 0 ]; then
    echo "test-scope: no molrs/src changes — nothing to run (full suite: cargo mrs-test)"
    exit 0
fi

# lib.rs is the crate root: every module hangs off it, so scoping is meaningless.
for m in "${modules[@]}"; do
    if [ "$m" = "lib" ]; then
        echo "test-scope: molrs/src/lib.rs changed — running the full suite"
        exec cargo mrs-test
    fi
done

filters=()
for m in "${modules[@]}"; do
    filters+=("$m::")
done

echo "test-scope: ${filters[*]}"
exec cargo mrs-test -- "${filters[@]}"
