#!/usr/bin/env bash
set -euo pipefail

# Cross-repo gate: the molrs and molpack Python extensions, installed into ONE
# venv, must dynamically link the SAME libmolrs_ffi.
#
# ── HAZARD ───────────────────────────────────────────────────────────────────
# Rust has no stable ABI, so one molrs dylib is reusable only inside a single
# (rustc, features, profile) fingerprint. If the two build graphs disagree on
# even one molrs feature — or on one [profile.release] knob — rustc happily
# emits two differently-fingerprinted molrs instances and each extension
# statically embeds its own copy. Nothing errors. Nothing conflicts. The
# shared-dylib work simply stops happening, silently.
#
# Worse, under the shared target dir the dylib's install_name is a *hashless*
# absolute path (<shared-target>/release/deps/libmolrs_ffi.dylib), so a
# mismatched second build OVERWRITES the same file that the first extension's
# @rpath resolves at runtime. Degradation is therefore not merely wasteful, it
# is a correctness hazard.
#
# That hashlessness also dictates where this gate's teeth are. Comparing the
# libmolrs_ffi PATH recorded in the two extensions proves nothing: install_name
# is a hashless absolute path, so the two are equal even when the second build
# overwrote the file out from under the first. The identity evidence is instead
# the sha256 of the shared dylib taken AFTER the molrs wheel build and again
# AFTER the molpack wheel build — if those differ, the molrs extension is now
# loading a library it was not linked against.
#
# None of this is observable from any single-package pytest: it is a
# link-time / process-time property of TWO wheels coexisting in ONE
# interpreter. molrs-python's test suite cannot see molpack; molpack's cannot
# see how molrs was linked. Hence a shell gate that builds both, installs both,
# and interrogates the actual Mach-O / ELF load commands.
#
# Failure policy: refuse and explain. This script never silently skips — a
# green run that proved nothing is exactly the "test that cannot fail" this
# repo forbids.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── 0. The dynamic opt-in, carried by this gate ──────────────────────────────
# Nothing committed selects the dynamic form any more: a zero-argument build is
# static everywhere (local, CI and publish pass byte-identical argument lists —
# namely none). The shared dylib is a pure command-line opt-in, so this script,
# being the only thing that proves that form still works, must carry it itself.
# Do not re-commit it into a .cargo/config.toml, an exported env var or a cargo
# alias: a committed switch is exactly the drift this arrangement removes.
#
# The opt-in is TWO things, and they travel by TWO DIFFERENT mechanisms because
# maturin is in the middle. Neither half is sufficient alone:
#
#   1. DYN_RUSTFLAGS — `-C prefer-dynamic` is what makes a consumer resolve
#      molrs out of the one libmolrs_ffi dylib instead of embedding its own
#      rlib copy. rustc is offered BOTH --extern forms (…/libmolrs_ffi.dylib
#      and …/libmolrs_ffi.rlib) and without this flag it takes the rlib.
#      It must reach cargo as RUSTFLAGS, NOT as --config: maturin
#      unconditionally sets CARGO_ENCODED_RUSTFLAGS (it adds its own
#      `-C link-arg=-undefined -C link-arg=dynamic_lookup`), and env-level
#      rustflags REPLACE config-level rustflags wholesale — so a rustflags
#      array supplied through --config never reaches rustc under maturin.
#      Measured on the molrs_python invocation of `maturin build --release -v`:
#      via --config, `-C prefer-dynamic` is ABSENT; via RUSTFLAGS it is
#      PRESENT, because maturin appends its link-args to an inherited
#      RUSTFLAGS rather than discarding it. (It used to work from
#      .cargo/config.toml only because maturin READS that file and merges it
#      into its own computation; a --config CLI override never enters it.)
#      Set inline on the two maturin calls below and nowhere else — never
#      exported, so it cannot leak into any other build in this shell.
#
#   2. DYN_CONFIG — `lto=false` is mandatory the moment that dylib enters a
#      cdylib's graph: rustc refuses outright with "only 'staticlib', 'bin',
#      and 'cdylib' outputs are supported with LTO". The seven native manifests
#      now carry lto = "thin" in [profile.release], and a cargo --config
#      profile key is the only thing that overrides a manifest profile in
#      place. This half DOES survive maturin: it is a profile key, not
#      rustflags, so the CARGO_ENCODED_RUSTFLAGS clobber above cannot touch it.
#      maturin 1.13.3 exposes cargo's own --config <KEY=VALUE> directly
#      (verified against `maturin build --help`), so it needs no `--`
#      passthrough.
#
# Human-readable contract: docs/interop.md, § "Local link form".
DYN_RUSTFLAGS='-C prefer-dynamic'
DYN_CONFIG=(--config 'profile.release.lto=false')

# ── 1. Locate the sibling molpack checkout ───────────────────────────────────
MOLPACK_ROOT="${MOLPACK_ROOT:-$PROJECT_ROOT/../molpack}"

if [ ! -d "$MOLPACK_ROOT" ] || [ ! -f "$MOLPACK_ROOT/python/Cargo.toml" ]; then
    echo "verify-shared-dylib: cannot find a usable molpack checkout." >&2
    echo "  looked for directory: $MOLPACK_ROOT" >&2
    echo "  requiring manifest:   $MOLPACK_ROOT/python/Cargo.toml" >&2
    echo >&2
    echo "The shared-dylib invariant is cross-repo by construction: it only" >&2
    echo "exists once molrs's and molpack's extensions coexist in one venv." >&2
    echo "There is nothing to weaken it down to, so this is a hard failure," >&2
    echo "not a skip." >&2
    echo >&2
    echo "Point it at an existing checkout:" >&2
    echo "  MOLPACK_ROOT=/path bash scripts/verify-shared-dylib.sh" >&2
    echo "or clone it next to molrs:" >&2
    echo "  git clone https://github.com/MolCrafts/molpack" >&2
    exit 1
fi
MOLPACK_ROOT="$(cd "$MOLPACK_ROOT" && pwd)"
echo "verify-shared-dylib: molrs   root = $PROJECT_ROOT"
echo "verify-shared-dylib: molpack root = $MOLPACK_ROOT"

# ── 2. Required tools ────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "verify-shared-dylib: 'uv' is not on PATH." >&2
    echo "  It provisions the throwaway interpreter both wheels are built" >&2
    echo "  against; a system python would leave the venv un-reproducible." >&2
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if ! command -v maturin >/dev/null 2>&1; then
    echo "verify-shared-dylib: 'maturin' is not on PATH." >&2
    echo "  Both extensions must be built by the same maturin against the" >&2
    echo "  same interpreter, so a pre-built wheel is not a substitute." >&2
    echo "  Install: uv tool install maturin" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin)
        LINKER_TOOL="otool"
        LINKER_CMD=(otool -L)
        FFI_LIB_SUFFIX=".dylib"
        if ! command -v otool >/dev/null 2>&1; then
            echo "verify-shared-dylib: 'otool' is not on PATH (macOS)." >&2
            echo "  Install the Command Line Tools: xcode-select --install" >&2
            exit 1
        fi
        ;;
    Linux)
        LINKER_TOOL="ldd"
        LINKER_CMD=(ldd)
        FFI_LIB_SUFFIX=".so"
        if ! command -v ldd >/dev/null 2>&1; then
            echo "verify-shared-dylib: 'ldd' is not on PATH (Linux)." >&2
            echo "  It ships with glibc; install your distro's libc-bin." >&2
            exit 1
        fi
        ;;
    *)
        echo "verify-shared-dylib: unsupported platform '$(uname -s)'." >&2
        echo "  This gate reads dynamic-link entries via otool (macOS) or" >&2
        echo "  ldd (Linux). Add a branch here before running elsewhere." >&2
        exit 1
        ;;
esac

if command -v shasum >/dev/null 2>&1; then
    SHA256_CMD=(shasum -a 256)
elif command -v sha256sum >/dev/null 2>&1; then
    SHA256_CMD=(sha256sum)
else
    echo "verify-shared-dylib: neither 'shasum' nor 'sha256sum' is on PATH." >&2
    echo "  The identity evidence of this gate is a byte hash of the shared" >&2
    echo "  libmolrs_ffi taken between the two wheel builds; without a hasher" >&2
    echo "  the run could only re-assert the vacuous path compare, which is" >&2
    echo "  exactly the 'test that cannot fail' this repo forbids." >&2
    echo "  Install: macOS ships shasum with perl; Linux: coreutils." >&2
    exit 1
fi

# The ONE shared dylib both extensions' @rpath resolves at runtime. Its name is
# hashless by construction (cargo does not fingerprint dylib file names), which
# is precisely why its CONTENT has to be watched.
FFI_DYLIB_BASE="$PROJECT_ROOT/target/release/deps/libmolrs_ffi"

resolve_shared_ffi_dylib() {
    local candidate
    for candidate in "$FFI_DYLIB_BASE$FFI_LIB_SUFFIX" "$FFI_DYLIB_BASE.dylib" "$FFI_DYLIB_BASE.so"; do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

sha256_of() {
    "${SHA256_CMD[@]}" "$1" | awk '{print $1}'
}

# ── 3. Throwaway venv ────────────────────────────────────────────────────────
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "verify-shared-dylib: creating venv (python 3.12) in $WORK/venv"
if ! uv venv --python 3.12 "$WORK/venv" >/dev/null; then
    echo "verify-shared-dylib: uv could not provision CPython 3.12." >&2
    echo "  Both wheels must target one interpreter, so falling back to" >&2
    echo "  whatever python happens to be around would defeat the gate." >&2
    echo "  Fix: uv python install 3.12" >&2
    exit 1
fi
VENV_PY="$WORK/venv/bin/python"

# ── 4. Build both wheels, release profile, same interpreter ──────────────────
# Release specifically: the dylib/profile fingerprint alignment this gate
# checks is declared per-[profile.release] across the seven native workspace
# roots enumerated by scripts/sync-dylib-locks.sh (ROOTS), so a debug build
# would not exercise the invariant under test. Those manifests carry
# lto = "thin", which is why the release build here must also carry
# profile.release.lto=false — see DYN_RUSTFLAGS / DYN_CONFIG at the top of
# this file for both halves of the opt-in and why they travel differently.
mkdir -p "$WORK/wheels"

echo "verify-shared-dylib: building molrs wheel (release, dynamic opt-in) ..."
(cd "$PROJECT_ROOT/molrs-python" && RUSTFLAGS="$DYN_RUSTFLAGS" maturin build --release "${DYN_CONFIG[@]}" -i "$VENV_PY" -o "$WORK/wheels")

FFI_DYLIB="$(resolve_shared_ffi_dylib || true)"
if [ -z "$FFI_DYLIB" ]; then
    echo "verify-shared-dylib: FAIL — the molrs wheel build produced no shared" >&2
    echo "libmolrs_ffi in the shared target." >&2
    echo >&2
    echo "  expected: $FFI_DYLIB_BASE$FFI_LIB_SUFFIX" >&2
    echo "  observed: no such file (the wheel build emitted no shared dylib)" >&2
    echo >&2
    echo "Remediation, in this order:" >&2
    echo "  1. bash scripts/sync-dylib-locks.sh   # rebuild every native root's" >&2
    echo "     Cargo.lock in one registry sweep, then rebuild" >&2
    echo "  2. confirm the maturin call above still carries BOTH halves of the" >&2
    echo "     opt-in: RUSTFLAGS=\"\$DYN_RUSTFLAGS\" AND \${DYN_CONFIG[@]} (top of" >&2
    echo "     this script). Since the static default landed, NOTHING committed" >&2
    echo "     turns the dynamic form on — lose either half and this branch is" >&2
    echo "     the guaranteed outcome" >&2
    echo "  3. check molrs-ffi's crate-type = [\"dylib\", \"rlib\"]" >&2
    echo "  4. check your shell for an exported CARGO_ENCODED_RUSTFLAGS: cargo" >&2
    echo "     prefers it over RUSTFLAGS, so it displaces the value this script" >&2
    echo "     sets inline for these two builds. Unset it and rerun" >&2
    echo "  Contract: docs/interop.md, § \"Local link form\"" >&2
    exit 1
fi
FFI_SHA_AFTER_MOLRS="$(sha256_of "$FFI_DYLIB")"
echo "verify-shared-dylib: sha256 after molrs wheel   = $FFI_SHA_AFTER_MOLRS"
echo "                     ($FFI_DYLIB)"

echo "verify-shared-dylib: building molpack wheel (release, dynamic opt-in) ..."
(cd "$MOLPACK_ROOT/python" && RUSTFLAGS="$DYN_RUSTFLAGS" maturin build --release "${DYN_CONFIG[@]}" -i "$VENV_PY" -o "$WORK/wheels")

# ── 4b. Unit identity: the shared dylib must not have moved a single byte ────
# This is THE identity assertion of this gate. The dylib's install_name carries
# no hash, so a second build graph that resolves molrs to a different unit
# (drifted Cargo.lock, divergent feature set, sibling-induced third-party
# widening, workspace-member flavour) does not conflict and does not error — it
# simply OVERWRITES this file. The molrs extension then loads, at runtime, a
# library it was never linked against.
FFI_DYLIB_AFTER="$(resolve_shared_ffi_dylib || true)"
if [ -z "$FFI_DYLIB_AFTER" ]; then
    echo "verify-shared-dylib: FAIL — the shared libmolrs_ffi disappeared during" >&2
    echo "the molpack wheel build." >&2
    echo >&2
    echo "  before: $FFI_DYLIB" >&2
    echo "  after:  <none matching $FFI_DYLIB_BASE$FFI_LIB_SUFFIX>" >&2
    echo >&2
    echo "Remediation, in this order:" >&2
    echo "  1. bash scripts/sync-dylib-locks.sh   # then rebuild" >&2
    echo "  2. confirm molpack's .cargo/config.toml still routes its target dir" >&2
    echo "     into $PROJECT_ROOT/target (a stray CARGO_TARGET_DIR breaks it)" >&2
    exit 1
fi
FFI_SHA_AFTER_MOLPACK="$(sha256_of "$FFI_DYLIB_AFTER")"
echo "verify-shared-dylib: sha256 after molpack wheel = $FFI_SHA_AFTER_MOLPACK"

if [ "$FFI_SHA_AFTER_MOLRS" != "$FFI_SHA_AFTER_MOLPACK" ]; then
    echo "verify-shared-dylib: FAIL — the shared libmolrs_ffi changed between the" >&2
    echo "two wheel builds. This is a unit-identity breach." >&2
    echo >&2
    echo "  after molrs wheel   : $FFI_SHA_AFTER_MOLRS" >&2
    echo "  after molpack wheel : $FFI_SHA_AFTER_MOLPACK" >&2
    echo "  file                : $FFI_DYLIB_AFTER" >&2
    echo >&2
    echo "The dylib's install_name is a HASHLESS absolute path, so cargo has no" >&2
    echo "way to keep two differently-fingerprinted builds apart: molpack's graph" >&2
    echo "resolved a molrs unit that differs from the one molrs-python resolved," >&2
    echo "and its build overwrote this file. Nothing errored. But the already-" >&2
    echo "linked molrs extension now loads, at runtime, a library it was NOT" >&2
    echo "linked against — mismatched layouts across an FFI boundary." >&2
    echo >&2
    echo "Remediation, in this order:" >&2
    echo "  1. bash scripts/sync-dylib-locks.sh   # deletes and rebuilds every" >&2
    echo "     native root's untracked Cargo.lock in ONE registry sweep (version" >&2
    echo "     drift between those locks is the dominant cause), then rerun this" >&2
    echo "     gate" >&2
    echo "  2. still red? census the shared target (this gate builds release):" >&2
    echo "       ls $PROJECT_ROOT/target/release/deps/libmolrs-*.rlib | wc -l" >&2
    echo "     anything but 1 means another fork axis is live. Unit identity" >&2
    echo "     has five measured axes and every one must match across the" >&2
    echo "     seven native roots: lockfile drift, the literal 'default'" >&2
    echo "     feature label, sibling-induced third-party feature widening," >&2
    echo "     workspace-member flavour, and the RUSTFLAGS régime (a bare" >&2
    echo "     cargo build and a maturin build are two different régimes)." >&2
    echo "     Word-diff the rustc invocations of the diverged crate and" >&2
    echo "     widen molrs-ffi's 'unify' pins" >&2
    exit 1
fi

MOLRS_WHEEL="$(find "$WORK/wheels" -maxdepth 1 -type f -name 'molcrafts_molrs-*.whl' | head -n 1)"
if [ -z "$MOLRS_WHEEL" ]; then
    echo "verify-shared-dylib: no molcrafts_molrs-*.whl in $WORK/wheels" >&2
    ls -la "$WORK/wheels" >&2
    exit 1
fi

MOLPACK_WHEEL="$(find "$WORK/wheels" -maxdepth 1 -type f -name 'molcrafts_molpack-*.whl' | head -n 1)"
if [ -z "$MOLPACK_WHEEL" ]; then
    echo "verify-shared-dylib: no molcrafts_molpack-*.whl in $WORK/wheels" >&2
    ls -la "$WORK/wheels" >&2
    exit 1
fi

# ── 5. Install ───────────────────────────────────────────────────────────────
# numpy is installed explicitly because the wheels go in with --no-deps:
# molpack's metadata requires molcrafts-molpy>=0.14,<0.15, which is not on
# PyPI yet, so any dependency resolution would fail on a package that this
# gate does not need. --no-deps also guarantees the molrs in this venv is the
# wheel we just built, never a resolved PyPI release.
echo "verify-shared-dylib: installing numpy + both wheels (--no-deps)"
uv pip install --python "$VENV_PY" numpy >/dev/null
uv pip install --python "$VENV_PY" --no-deps "$MOLRS_WHEEL" "$MOLPACK_WHEEL" >/dev/null

# ── 6. Locate the two extension modules ──────────────────────────────────────
echo "verify-shared-dylib: locating extension modules"
MOLRS_EXT="$("$VENV_PY" -c 'import importlib.util; s = importlib.util.find_spec("molrs._lib"); print(s.origin if s and s.origin else "")')"
MOLPACK_EXT="$("$VENV_PY" -c 'import importlib.util; s = importlib.util.find_spec("molpack.molpack"); print(s.origin if s and s.origin else "")')"

if [ -z "$MOLRS_EXT" ] || [ -z "$MOLPACK_EXT" ]; then
    echo "verify-shared-dylib: could not resolve both extension modules." >&2
    echo "  molrs._lib        -> ${MOLRS_EXT:-<unresolved>}" >&2
    echo "  molpack.molpack   -> ${MOLPACK_EXT:-<unresolved>}" >&2
    exit 1
fi
echo "  molrs._lib      = $MOLRS_EXT"
echo "  molpack.molpack = $MOLPACK_EXT"

# ── 7. Read the dynamic-link entries ─────────────────────────────────────────
echo "verify-shared-dylib: reading ${LINKER_CMD[*]} entries"
MOLRS_LINKS="$("${LINKER_CMD[@]}" "$MOLRS_EXT" 2>&1 || true)"
MOLPACK_LINKS="$("${LINKER_CMD[@]}" "$MOLPACK_EXT" 2>&1 || true)"

# otool -L:  "\t/abs/path/libmolrs_ffi.dylib (compatibility version ...)"
# ldd:       "\tlibmolrs_ffi.so => /abs/path/libmolrs_ffi.so (0x00007f...)"
#
# The `|| true` is load-bearing: no match is the *expected* red outcome, and
# under `set -o pipefail` a failing grep would abort the script before the
# diagnostic below ever ran — a red that prints nothing.
extract_ffi_path() {
    if [ "$LINKER_TOOL" = "otool" ]; then
        printf '%s\n' "$1" | grep 'libmolrs_ffi' | head -n 1 | awk '{print $1}' || true
    else
        printf '%s\n' "$1" | grep 'libmolrs_ffi' | head -n 1 | awk -F'=>' '{print $2}' | awk '{print $1}' || true
    fi
}

MOLRS_FFI="$(extract_ffi_path "$MOLRS_LINKS")"
MOLPACK_FFI="$(extract_ffi_path "$MOLPACK_LINKS")"

if [ -z "$MOLRS_FFI" ] || [ -z "$MOLPACK_FFI" ]; then
    echo "verify-shared-dylib: FAIL — no libmolrs_ffi dynamic-link entry." >&2
    echo >&2
    echo "  observed  molrs._lib      libmolrs_ffi entry: ${MOLRS_FFI:-<none>}" >&2
    echo "  observed  molpack.molpack libmolrs_ffi entry: ${MOLPACK_FFI:-<none>}" >&2
    echo "  expected  one libmolrs_ffi entry in EACH extension" >&2
    echo >&2
    echo "An extension with no libmolrs_ffi entry has molrs STATICALLY linked" >&2
    echo "into it: the two wheels then carry two independent molrs images and" >&2
    echo "share nothing. Static is the zero-argument default of both repos, so" >&2
    echo "this is the outcome whenever the opt-in fails to reach rustc." >&2
    echo >&2
    echo "Remediation, in this order:" >&2
    echo "  1. bash scripts/sync-dylib-locks.sh   # rebuild every native root's" >&2
    echo "     Cargo.lock in ONE registry sweep, then rebuild. Do this first:" >&2
    echo "     until every root resolves one molrs unit, nothing below can be" >&2
    echo "     diagnosed reliably" >&2
    echo "  2. the dynamic opt-in itself — BOTH halves, on BOTH maturin build" >&2
    echo "     --release calls: RUSTFLAGS=\"\$DYN_RUSTFLAGS\" (prefer-dynamic has" >&2
    echo "     to travel as env rustflags; maturin's own" >&2
    echo "     CARGO_ENCODED_RUSTFLAGS discards a --config rustflags array)" >&2
    echo "     plus \${DYN_CONFIG[@]} (the profile key, which does survive)." >&2
    echo "     Nothing else selects the dynamic form — no .cargo/config, no" >&2
    echo "     workflow env, no alias does it any more" >&2
    echo "  3. an exported CARGO_ENCODED_RUSTFLAGS in your shell: cargo prefers" >&2
    echo "     it over RUSTFLAGS, so it displaces the value this script sets" >&2
    echo "     inline. Unset it and rerun" >&2
    echo "  4. molrs-ffi's crate-type = [\"dylib\", \"rlib\"]" >&2
    echo "  5. every consumer graph resolving the SAME molrs feature union" >&2
    echo "  Contract: docs/interop.md, § \"Local link form\"" >&2
    echo >&2
    echo "--- ${LINKER_CMD[*]} $MOLRS_EXT ---" >&2
    printf '%s\n' "$MOLRS_LINKS" >&2
    echo >&2
    echo "--- ${LINKER_CMD[*]} $MOLPACK_EXT ---" >&2
    printf '%s\n' "$MOLPACK_LINKS" >&2
    exit 1
fi

# ── 8. Sanity print of the two recorded paths (NOT identity evidence) ────────
# Kept only for visibility. The compare it used to perform is trivially always
# equal: the dylib's install_name is a hashless ABSOLUTE path, so both
# extensions record the same string even when the second build overwrote the
# file. The identity proof is the sha256 pair asserted in step 4b; per this
# repo's "tests that cannot fail" rule an assertion that cannot expose the
# defect is replaced, not trusted.
MOLRS_FFI_REAL="$("$VENV_PY" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$MOLRS_FFI")"
MOLPACK_FFI_REAL="$("$VENV_PY" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$MOLPACK_FFI")"
echo "verify-shared-dylib: recorded install_name (sanity print only)"
echo "  molrs._lib      -> $MOLRS_FFI_REAL"
echo "  molpack.molpack -> $MOLPACK_FFI_REAL"

# What DOES have teeth here: each recorded library must exist and be byte-wise
# the object step 4b hashed. Byte identity rather than path identity, because
# cargo uplifts the dylib by hardlink — a second path to the same content is
# fine, different content behind either path is not (it would mean the wheels
# were built into a target dir other than the shared one).
for recorded in "$MOLRS_FFI_REAL" "$MOLPACK_FFI_REAL"; do
    if [ ! -f "$recorded" ] || [ "$(sha256_of "$recorded")" != "$FFI_SHA_AFTER_MOLPACK" ]; then
        echo "verify-shared-dylib: FAIL — an extension records a libmolrs_ffi that" >&2
        echo "is not the shared object this gate hashed." >&2
        echo >&2
        echo "  observed  recorded     -> $recorded" >&2
        echo "  observed  exists       -> $([ -f "$recorded" ] && echo yes || echo no)" >&2
        echo "  observed  its sha256   -> $([ -f "$recorded" ] && sha256_of "$recorded" || echo '<none>')" >&2
        echo "  expected  shared dylib -> $FFI_DYLIB_AFTER" >&2
        echo "  expected  its sha256   -> $FFI_SHA_AFTER_MOLPACK" >&2
        echo >&2
        echo "Remediation, in this order:" >&2
        echo "  1. bash scripts/sync-dylib-locks.sh   # then rebuild and rerun" >&2
        echo "  2. confirm both repos' .cargo/config.toml still route builds into" >&2
        echo "     $PROJECT_ROOT/target ([build] target-dir is the only key those" >&2
        echo "     files still carry), and that no CARGO_TARGET_DIR is set" >&2
        echo "  3. confirm both maturin calls in this script took the SAME" >&2
        echo "     RUSTFLAGS=\"\$DYN_RUSTFLAGS\" and the SAME \${DYN_CONFIG[@]}:" >&2
        echo "     two different flag sets are two rustflags régimes fighting" >&2
        echo "     over one hashless dylib path" >&2
        echo "  Contract: docs/interop.md, § \"Local link form\"" >&2
        echo >&2
        echo "--- ${LINKER_CMD[*]} $MOLRS_EXT ---" >&2
        printf '%s\n' "$MOLRS_LINKS" >&2
        echo >&2
        echo "--- ${LINKER_CMD[*]} $MOLPACK_EXT ---" >&2
        printf '%s\n' "$MOLPACK_LINKS" >&2
        exit 1
    fi
done

# ── 9. Runtime half: one interpreter, both packages, a real capsule crossing ─
echo "verify-shared-dylib: running regressions/ffi-shared-dylib.py"
"$VENV_PY" "$PROJECT_ROOT/regressions/ffi-shared-dylib.py"

# ── 10. Done ─────────────────────────────────────────────────────────────────
echo "verify-shared-dylib: shared dylib = $FFI_DYLIB_AFTER"
echo "verify-shared-dylib: sha256       = $FFI_SHA_AFTER_MOLPACK (stable across both wheel builds)"
echo "verify-shared-dylib: OK"
