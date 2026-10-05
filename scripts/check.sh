#!/usr/bin/env bash
# The one definition of every gate. prek (.pre-commit-config.yaml) and CI
# (.github/workflows/ci-*.yml) both call this script, and rust-toolchain.toml
# pins the compiler for both, so a gate that passes locally is the same
# command on the same rustc / clippy / rustdoc that CI runs.
#
#   scripts/check.sh fmt clippy     # run the named gates, in order
#   scripts/check.sh all            # every gate (CI parity)
#
# Gates: fmt clippy doc test features package ffi cxx python capi wasm.
# Root-workspace cargo calls go through the `cargo mrs-*` aliases
# (.cargo/config.toml) so they share one feature set and one build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BINDERS=(molrs-ffi molrs-python molrs-wasm molrs-capi molrs-cxxapi)
# wasm-opt release the wasm gate runs; CI installs exactly this one.
BINARYEN_VERSION=version_133
TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target}

clippy_binder() {
    cargo clippy --manifest-path "$1/Cargo.toml" --all-targets "${@:2}" -- -D warnings
}

gate_fmt() {
    cargo fmt --all --check
    for crate in "${BINDERS[@]}"; do
        cargo fmt --manifest-path "$crate/Cargo.toml" --check
    done
}

gate_clippy() {
    cargo mrs-clippy -- -D warnings
}

gate_doc() {
    RUSTDOCFLAGS="-D warnings" cargo mrs-doc
}

# --lib does not run rustdoc examples, so the doctests are their own step.
gate_test() {
    cargo mrs-test
    cargo mrs-doctest
}

# Each sub-system must build on its own, without molrs's native defaults.
gate_features() {
    cargo check -p molcrafts-molrs
    cargo check -p molcrafts-molrs --no-default-features
    for feature in io smiles signal compute ff conformer md builder serde stream zarr filesystem voronoi full; do
        cargo check -p molcrafts-molrs --no-default-features --features "$feature"
    done
}

# Compiles the unpacked crates.io archive, catching files left out of it.
gate_package() {
    cargo package --allow-dirty --manifest-path molrs/Cargo.toml
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
    cargo test --manifest-path molrs-ffi/Cargo.toml
}

gate_cxx() {
    clippy_binder molrs-cxxapi
    cargo test --manifest-path molrs-cxxapi/Cargo.toml
}

# Tools only (no project install), so tox builds the wheel once.
gate_python() {
    clippy_binder molrs-python
    uv --directory molrs-python sync --no-install-project --extra dev
    uv --directory molrs-python run --no-sync tox -e py
}

# Profile and target dir are explicit: CMake caches both, and a build-test/
# configured once for release would otherwise keep rebuilding the release lib.
gate_capi() {
    clippy_binder molrs-capi
    cargo test --manifest-path molrs-capi/Cargo.toml
    cargo build --manifest-path molrs-capi/Cargo.toml
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
    (cd molrs-wasm && wasm-pack build --release --target bundler --scope molcrafts --out-name molrs)
    (cd molrs-wasm && wasm-pack test --node)
}

ALL=(fmt clippy doc test features package ffi cxx python capi wasm)

[ "$#" -gt 0 ] || { echo "usage: $0 <gate>... | all   (gates: ${ALL[*]})" >&2; exit 2; }
[ "$1" = all ] && set -- "${ALL[@]}"

for gate in "$@"; do
    if ! declare -F "gate_$gate" >/dev/null; then
        echo "unknown gate: $gate (gates: ${ALL[*]})" >&2
        exit 2
    fi
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::group::check $gate"; else echo "== check $gate"; fi
    "gate_$gate"
    [ -z "${GITHUB_ACTIONS:-}" ] || echo "::endgroup::"
done
