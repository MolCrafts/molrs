# 0.14.0 release preflight — 2026-09-08

Final archive contents and workflow syntax rechecked on 2026-09-09.

Local release preparation on `dev`, preserving the existing uncommitted I/O
and WASM work. Package versions remain 0.14.0. No commit, tag, push, or registry
publication was performed.

## Changes

- Appended LAMMPS logs use binary search over sorted invocation boundaries.
  WASM thermo serialization moves owned rows and column names instead of
  cloning the full table. A WASM test checks the exported object shape,
  numeric values, empty output, and retained run indices.
- Simplified fixed-width vector copies and removed an empty optimizer
  conditional. Applied rustfmt across all six workspace roots.
- Cleaned Python binding documentation and a nested condition. Existing
  Python keyword signatures and tuple return formats have narrowly scoped,
  documented Clippy exceptions; public signatures are unchanged.
- C header generation now fails when cbindgen fails. CMake builds the requested
  debug/release profile, respects an isolated Cargo target directory, and asks
  Cargo to check source changes even when a library already exists.
- Moved the PME source-documentation check into the architecture suite,
  retaining its assertion. Fixed stale API links and executable quickstarts.
- CI covers every workspace's formatting, independent core features, binding
  linting, architecture/CXX tests, strict rustdoc, and unpacked-crate builds.
  Publishing waits for CI and checks tag/version/master ancestry. Branch
  dispatches build artifacts without publishing.
- Added the missing Rust crate LICENSE file, matching the repository license;
  CI verifies its contents inside the actual archive.

## Verification

Environment: Linux x86_64, Rust 1.98.1, Python 3.12.13 for wheel tests, CPython
3.14 for maturin, Node 26.3.0, wasm-pack 0.13.1. The release profiles and linker
flags are unchanged. A shared-filesystem build-lock wait was avoided with the
temporary `CARGO_TARGET_DIR=/tmp/molrs-release-check`; repository Cargo
configuration was not changed.

| Check | Result |
| --- | --- |
| rustfmt: root + five standalone binding workspaces | Passed |
| Clippy with `-D warnings`: core, CXX, C, Python, wasm32 | Passed |
| Core alone + 13 independent feature configurations | Passed |
| Core unit tests | 1,894 passed |
| Architecture tests | 15 passed; 2 existing ignored cases |
| Core doctests | 73 passed; 13 existing ignored examples |
| FFI tests and ABI layout snapshot | 17 passed + 1 doctest |
| CXX tests | 20 passed; 2 existing ignored cases |
| C API Rust tests | 17 passed |
| C++ tests against the release C library | 28 passed |
| WASM tests under Node | 32 passed |
| Installed abi3 wheel under Python 3.12 | 629 passed |
| Static wheel linkage and Python README quickstart | Passed |
| Public rustdoc with `RUSTDOCFLAGS="-D warnings"` | Passed |
| `cargo package --allow-dirty` | 466 files including LICENSE; unpacked crate builds |
| Optimized WASM bundler package | Builds; Node import and SMILES/3D/XYZ smoke pass |
| `npm pack --dry-run` | Correct name/version and seven distribution files |
| Workflow YAML, embedded shell syntax, tag guard | Passed locally |
| `git diff --check` | Passed |

The local, ignored root lockfile referenced yanked `chacha20` 0.10.1. Updating
it to the compatible 0.10.2 removed the packaging warning; core tests and the
unpacked-package build were rerun. No dependency constraints or tracked lockfiles
were changed.

Session build products and logs are under `/tmp/molrs-release-*`. The Python
wheel is in `/tmp/molrs-release-wheels`, the npm package in
`/tmp/molrs-release-wasm-pkg`, and the C library in
`/tmp/molrs-release-check/release`. These are temporary local artifacts.

## Remaining release checks

The existing ignored tests/examples were not enabled or weakened. This run
does not exercise optional BLAS/slow tests, the opt-in cross-repository dynamic
link form, Windows/macOS/ARM, or Pyodide. Rust 1.91 dependency metadata was
checked, but the minimum compiler itself was not run. GitHub Actions workflows
were validated locally, not executed remotely.

Follow [the release checklist](../docs/releasing.md): review the diff, complete
the remote platform matrix, merge the reviewed revision to master, then tag
and publish molrs before updating downstream minor-version pins.
