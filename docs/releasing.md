# Release checklist

molrs uses one version across the Rust library, standalone binding manifests,
and `molrs-python/pyproject.toml`. Only `molcrafts-molrs` is published to
crates.io; Python, WASM, and C API artifacts use their own distribution channels.
The CXX bridge is built from source. Release history lives in git tags and
GitHub Releases.

## Local verification

Run from the repository root with the toolchain in `rust-toolchain.toml`:

```bash
for manifest in Cargo.toml molrs-ffi/Cargo.toml molrs-python/Cargo.toml \
  molrs-wasm/Cargo.toml molrs-capi/Cargo.toml molrs-cxxapi/Cargo.toml; do
  cargo fmt --manifest-path "$manifest" --check || exit 1
done
cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings
cargo clippy --manifest-path molrs-cxxapi/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path molrs-capi/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path molrs-wasm/Cargo.toml --target wasm32-unknown-unknown --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p molcrafts-molrs --no-deps --features full,filesystem,stream
bash scripts/fetch-test-data.sh
cargo test -p molcrafts-molrs --features full,filesystem,stream
cargo test --manifest-path molrs-ffi/Cargo.toml
cargo test --manifest-path molrs-cxxapi/Cargo.toml
cargo package --manifest-path molrs/Cargo.toml
```

`cargo package` compiles the unpacked archive, catching files accidentally
omitted from the release. Inspect `cargo package --list --manifest-path
molrs/Cargo.toml` as well. `--all-features` includes `blas` and `slow-tests`;
those require a separate setup and are outside the default CI gate.

Verify the installed Python wheel, browser bindings, and C ABI:

```bash
uv --directory molrs-python sync --no-install-project --extra dev
uv --directory molrs-python run --no-sync tox -e py
(cd molrs-wasm && wasm-pack build --release --target bundler --scope molcrafts --out-name molrs)
(cd molrs-wasm && wasm-pack test --node)
cmake -S molrs-capi/tests/cpp -B molrs-capi/build-test -DCARGO_PROFILE=release
cmake --build molrs-capi/build-test
ctest --test-dir molrs-capi/build-test --output-on-failure
```

Native artifacts use static Rust linkage by default. Run
`regressions/link-mode-static-default.py` in an environment containing the
new wheel. Changes to the optional shared-library integration also need
`bash scripts/verify-shared-dylib.sh` and a compatible sibling molpack checkout;
see [interop.md](interop.md).

CI additionally checks each independent feature with defaults disabled. Check
version metadata across all manifests before tagging; downstream molpy pins the
major.minor ABI line and must be released after molrs.

## Publishing

1. Finish the checks and review the release diff, including API migrations.
2. Run **Publish** manually on a branch for a build rehearsal. It runs CI and
   builds artifacts without uploading to registries or creating a release.
3. Merge the reviewed revision into `master`, then create and push `vX.Y.Z`,
   matching `[workspace.package].version` in `Cargo.toml`.
4. Wait for **Publish** to finish: crates.io, npm, all desktop/Pyodide wheels
   on PyPI, and C API archives with checksums on GitHub Releases.
5. Smoke-test the published artifacts before updating downstream minor pins.

The workflow checks that the tag matches the package version and points to a
commit on `master`. Registry publications wait for CI. To retry a partial release, rerun
the failed tag workflow, or dispatch **Publish** against that same tag; existing
registry versions are skipped. A branch dispatch never publishes.
