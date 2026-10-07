# Release checklist

molrs uses one version across the Rust library, standalone binding manifests,
and `molrs-python/pyproject.toml`. Only `molcrafts-molrs` is published to
crates.io; Python, WASM, and C API artifacts use their own distribution channels.
The CXX bridge is built from source. Release history lives in git tags and
GitHub Releases.

## Local verification

Run every CI gate from the repository root:

```bash
scripts/check.sh all
```

It is the same script, on the same pinned `rust-toolchain.toml` compiler, that
the prek hooks and the CI workflows call. The `package` gate compiles the
unpacked crates.io archive, catching files accidentally omitted from the
release; inspect `cargo package --list --manifest-path molrs/Cargo.toml` as
well. Downstream molpy pins the major.minor ABI line and must be released
after molrs.

Check version metadata before tagging. One version appears in:

- `[workspace.package].version` in `Cargo.toml` (inherited by `molrs/`);
- `version` and the `molcrafts-molrs` / `molcrafts-molrs-ffi` dependency
  versions in `molrs-ffi/`, `molrs-python/`, `molrs-wasm/`, `molrs-capi/`
  and `molrs-cxxapi/Cargo.toml` (the npm `package.json` is generated from
  `molrs-wasm/Cargo.toml` by wasm-pack), and in
  `molrs-ext-example/Cargo.toml` (unpublished, a standalone workspace kept
  in step);
- `version` in `molrs-python/pyproject.toml`;
- the `molcrafts-molrs*` entries of every committed `Cargo.lock` (the root
  one and one per binder), and the editable `molcrafts-molrs` entry of
  `molrs-python/uv.lock`;
- the version-pinned examples in `README.md` and the documentation site
  (`version = "X.Y"` dependency lines, `>=X.Y.0,<X.(Y+1)` pins).

The documentation site must build exactly as Cloudflare Pages builds it, in
a fresh virtualenv:

```bash
cd molrs-python
pip install ".[doc]"
zensical build --clean      # must end with "No issues found"
```

## Publishing

1. Finish the checks and review the release diff. The GitHub Release created
   by **Publish** links to the documentation site.
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
