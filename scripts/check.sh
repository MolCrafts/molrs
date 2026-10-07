#!/usr/bin/env bash
# The one definition of every gate. prek (.pre-commit-config.yaml) and CI
# (.github/workflows/ci-*.yml) both call this script, and rust-toolchain.toml
# pins the compiler for both, so a gate that passes locally is the same
# command on the same rustc / clippy / rustdoc that CI runs.
#
#   scripts/check.sh fmt clippy     # run the named gates, in order
#   scripts/check.sh all            # every gate (CI parity)
#
# Gates: fmt ruff partners clippy doc test features package ffi cxx ext python capi
# wasm mrec docs. Root-workspace cargo calls go through the `cargo mrs-*`
# aliases (.cargo/config.toml) so they share one feature set and one build.
# Every cargo / maturin / wasm-pack call is --locked and every uv call runs
# on CI's Python (3.12) against the committed lock: a gate that would have to
# change a lock file fails instead.
#
# Dispatch: `fmt`, `ruff` and `partners` compile nothing and run wherever this script
# is called. When the environment names a runner in MOLCRAFTS_HOOK_RUNNER and
# this is not already a Slurm job, any other gate hands the whole call to it.
# The MolCrafts cluster's shared git hooks set it to a launcher that runs its
# arguments on a compute node, so a commit touching only cheap gates never
# waits for Slurm. CI and other machines leave it unset: nothing changes there.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_PYTHON=3.12
# Run from a git hook, GIT_DIR, GIT_INDEX_FILE, ... name the hooked
# repository; a gate's own git calls (scripts/partners.py fetching molrec, uv
# fetching a git dependency) must not inherit them.
unset $(git rev-parse --local-env-vars)
CLEANUP=()
trap '[ "${#CLEANUP[@]}" -eq 0 ] || rm -rf "${CLEANUP[@]}"' EXIT

BINDERS=(molrs-ffi molrs-python molrs-wasm molrs-capi molrs-cxxapi)
# Every standalone workspace root besides the root one: the binders, and the
# force-field IR extension proof crate.
ROOTS=("${BINDERS[@]}" molrs-ext-example)
# wasm-opt release the wasm gate runs; CI installs exactly this one.
BINARYEN_VERSION=version_133
TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target}

clippy_binder() {
    cargo clippy --locked --manifest-path "$1/Cargo.toml" --all-targets "${@:2}" -- -D warnings
}

# Sets WORK to a fresh temp dir, removed when the script exits.
scratch() {
    WORK=$(mktemp -d "${TMPDIR:-/tmp}/molrs-check.XXXXXX")
    CLEANUP+=("$WORK")
}

gate_fmt() {
    cargo fmt --all --check
    for crate in "${ROOTS[@]}"; do
        cargo fmt --manifest-path "$crate/Cargo.toml" --check
    done
}

# Python lint (ruff.toml): text-mode file I/O names its encoding, so a read
# that passes on Linux cannot fail only on Windows (cp1252).
gate_ruff() {
    uvx ruff@0.16.5 check .
}

# Every partner in .github/partners.env resolves, no path dependency points
# where CI has no checkout, no workflow spells a partner commit of its own.
gate_partners() {
    python3 scripts/partners.py check
}

gate_clippy() {
    cargo --locked mrs-clippy -- -D warnings
}

gate_doc() {
    RUSTDOCFLAGS="-D warnings" cargo --locked mrs-doc
}

# --lib does not run rustdoc examples, so the doctests are their own step.
gate_test() {
    cargo --locked mrs-test
    cargo --locked mrs-doctest
}

# Each sub-system must build on its own, without molrs's native defaults,
# and so must its tests: --all-targets, so a test that reaches past its
# feature (an `ff` test naming `io::mrec`, which is `zarr`'s) fails here.
features_clippy() {
    cargo clippy --locked -p molcrafts-molrs --all-targets "$@" -- -D warnings
}

gate_features() {
    features_clippy
    features_clippy --no-default-features
    for feature in io smiles signal compute ff conformer md builder serde stream zarr filesystem voronoi full; do
        features_clippy --no-default-features --features "$feature"
    done
}

# Compiles the unpacked crates.io archive, catching files left out of it.
gate_package() {
    cargo package --locked --allow-dirty --manifest-path molrs/Cargo.toml
    python3 - "$TARGET_DIR" <<'PY'
import sys
import tarfile
import tomllib
from pathlib import Path

version = tomllib.loads(Path("Cargo.toml").read_text())["workspace"]["package"]["version"]
name = f"molcrafts-molrs-{version}"
with tarfile.open(Path(sys.argv[1]) / "package" / f"{name}.crate") as package:
    license_file = package.extractfile(f"{name}/LICENSE")
    assert license_file is not None, "Package is missing LICENSE"
    assert license_file.read() == Path("LICENSE").read_bytes(), "Package license differs from repository license"
PY
}

gate_ffi() {
    clippy_binder molrs-ffi
    cargo test --locked --manifest-path molrs-ffi/Cargo.toml
}

gate_cxx() {
    clippy_binder molrs-cxxapi
    cargo test --locked --manifest-path molrs-cxxapi/Cargo.toml
}

# The force-field IR as a protocol (ff-ir-02-protocol, P-Rust): a third
# party's crate extending it through molrs's pub API alone — a pair style, a
# new category, an expression style — priced against pinned LAMMPS numbers
# (scripts/ff_ir_extension_lammps_check.sh), persisted, and refused by name.
gate_ext() {
    clippy_binder molrs-ext-example
    cargo test --locked --manifest-path molrs-ext-example/Cargo.toml
}

# Tools only (no project install), so tox builds the wheel once.
gate_python() {
    clippy_binder molrs-python
    uv --directory molrs-python sync --locked --no-install-project --extra dev
    uv --directory molrs-python run --no-sync tox -e py
}

# Profile and target dir are explicit: CMake caches both, and a build-test/
# configured once for release would otherwise keep rebuilding the release lib.
gate_capi() {
    clippy_binder molrs-capi
    cargo test --locked --manifest-path molrs-capi/Cargo.toml
    cargo build --locked --manifest-path molrs-capi/Cargo.toml
    cmake -S molrs-capi/tests/cpp -B molrs-capi/build-test \
        -DCARGO_PROFILE=debug -DCARGO_TARGET_DIR="$TARGET_DIR"
    cmake --build molrs-capi/build-test
    ctest --test-dir molrs-capi/build-test --output-on-failure
}

# Building proves the wasm compiles; the Node suite proves it works.
gate_wasm() {
    local have
    have=$(wasm-opt --version | awk '{print $NF}' | tr -d '()')
    if [ "$have" != "$BINARYEN_VERSION" ]; then
        echo "wasm-opt is $have, the gate pins $BINARYEN_VERSION:" >&2
        echo "https://github.com/WebAssembly/binaryen/releases/tag/$BINARYEN_VERSION" >&2
        return 1
    fi
    clippy_binder molrs-wasm --target wasm32-unknown-unknown
    (cd molrs-wasm && wasm-pack build --release --target bundler --scope molcrafts --out-name molrs -- --locked)
    (cd molrs-wasm && wasm-pack test --node -- --locked)
}

# molrec's conformance suite through molrs.io.mrec, as ci-snapshot.yml's mrec
# step runs it: molrec at the commit scripts/partners.py resolves (fetched into
# a temp dir, never a sibling's working tree), the extension built by `maturin
# develop` into a fresh venv on CI's Python, scripts/ci-conformance.py judging
# every case. Any case that does not pass fails the gate.
gate_mrec() {
    scratch
    local work=$WORK
    python3 scripts/partners.py fetch MOLREC "$work/molrec"
    uv venv -q --seed "$work/venv"
    # shellcheck disable=SC1091
    source "$work/venv/bin/activate"
    uv pip install -q maturin "$work/molrec" \
        "molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@master"
    maturin develop --locked --manifest-path molrs-python/Cargo.toml
    python scripts/ci-conformance.py --suite "$work/molrec/tests" --out "$work/conformance.json"
    deactivate
}

# The docs site as Cloudflare Pages builds it -- `pip install ".[doc]"` in a
# fresh env, then `zensical build --clean` -- with --strict, so any warning
# (an unresolved mkdocstrings reference included) fails. mkdocstrings imports
# the compiled extension; it is built in the dev profile, which has the same
# API surface as the release wheel at a fraction of the compile.
gate_docs() {
    scratch
    local work=$WORK
    uv venv -q "$work/venv"
    uv pip install -q --python "$work/venv/bin/python" maturin
    "$work/venv/bin/maturin" build --locked --manifest-path molrs-python/Cargo.toml \
        --interpreter "$work/venv/bin/python" --out "$work/wheels"
    local wheel
    wheel=$(ls "$work"/wheels/molcrafts_molrs-*.whl)
    uv pip install -q --python "$work/venv/bin/python" "$wheel[doc]"
    (cd molrs-python && "$work/venv/bin/zensical" build --clean --strict)
}

ALL=(fmt ruff partners clippy doc test features package ffi cxx ext python capi wasm mrec docs)
# Gates that compile nothing; everything else goes to MOLCRAFTS_HOOK_RUNNER.
CHEAP=(fmt ruff partners)

[ "$#" -gt 0 ] || { echo "usage: $0 <gate>... | all   (gates: ${ALL[*]})" >&2; exit 2; }
[ "$1" = all ] && set -- "${ALL[@]}"

if [ -n "${MOLCRAFTS_HOOK_RUNNER:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    for gate in "$@"; do
        if [[ " ${CHEAP[*]} " != *" $gate "* ]]; then
            exec "$MOLCRAFTS_HOOK_RUNNER" "$PWD/scripts/check.sh" "$@"
        fi
    done
fi

for gate in "$@"; do
    if ! declare -F "gate_$gate" >/dev/null; then
        echo "unknown gate: $gate (gates: ${ALL[*]})" >&2
        exit 2
    fi
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::group::check $gate"; else echo "== check $gate"; fi
    "gate_$gate"
    [ -z "${GITHUB_ACTIONS:-}" ] || echo "::endgroup::"
done
