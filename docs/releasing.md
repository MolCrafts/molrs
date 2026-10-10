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

## Partners

`.github/partners.env` pins full commits on development and release branches.
The same pins are used by pre-push, normal CI, release tags and nightly. Update
and validate them explicitly; never restore floating refs after a release.

## Publishing

`.github/workflows/release.yml` holds every channel: crates.io
(`molcrafts-molrs`), npm (`@molcrafts/molrs`), PyPI (desktop abi3 wheels and
the Pyodide wheel), and the GitHub Release with the C API archives.

| run | what it does |
| --- | --- |
| `v*` tag on MolCrafts | `release / guard` (tag = `v` + the workspace version, on `master`), `lint.yml` and the full `test.yml` tier, the builds and dry runs, then every upload (`release / crate`, `npm`, `pypi`, `github`) |
| dispatch (any branch, fork or upstream) | the dry run: the same guard, gates and builds, `release / crate (dry run)` (`cargo publish --dry-run`) and `release / npm (dry run)` (`npm pack --dry-run`); the upload jobs are skipped, so no upload and no Release |

The upload jobs run only when `release / context` reports `publish` (a `v*`
tag pushed to MolCrafts, from `MolCrafts/molcrafts-ci/actions/ci-context`);
`lint / hooks` fails any upload job that tests the event or ref itself.

1. Finish the checks and review the release diff.
2. Dispatch **release** on the branch for a rehearsal (a fork is fine).
3. Merge the reviewed revision into `master`, then create and push `vX.Y.Z`,
   matching `[workspace.package].version` in `Cargo.toml`.
4. Wait for **release** to finish, then smoke-test the published artifacts
   before updating downstream minor pins.

Every upload skips a version the registry already has: retry a partial
release by re-running the tag's run. Trusted publishing on crates.io, PyPI
and npm names the workflow file, `release.yml`, and one environment per registry:
`crates-io`, `npm` and `pypi`.

Nightly wheels (`molcrafts-molrs-nightly` on PyPI) come from `nightly.yml`
on a push to the `nightly` branch; see "CI" in
`molrs-python/docs/contributing.md`.
