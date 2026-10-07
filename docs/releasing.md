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

On `dev`, `.github/partners.env` tracks molrec's `dev` (`MOLREC_REF=dev`).
A release is judged against a fixed molrec instead: the release commit on
`master` sets `MOLREC_REF` to the molrec tag or full commit the release was
checked against, so the tag's run (`release.yml`, through `test / mrec`)
fetches exactly that. When `master` is merged back into `dev`, keep
`MOLREC_REF=dev` there.

## Publishing

`.github/workflows/release.yml` holds every channel: crates.io
(`molcrafts-molrs`), npm (`@molcrafts/molrs`), PyPI (desktop abi3 wheels and
the Pyodide wheel), and the GitHub Release with the C API archives.

| run | what it does |
| --- | --- |
| `v*` tag on MolCrafts | `release / guard` (tag = `v` + the workspace version, on `master`), `lint.yml` and the full `test.yml` tier, the builds, then every upload |
| dispatch (any branch, fork or upstream) | the dry run: the same guard, gates and builds, `cargo publish --dry-run`, no upload and no Release |

1. Finish the checks and review the release diff.
2. Dispatch **release** on the branch for a rehearsal (a fork is fine).
3. Merge the reviewed revision into `master`, then create and push `vX.Y.Z`,
   matching `[workspace.package].version` in `Cargo.toml`.
4. Wait for **release** to finish, then smoke-test the published artifacts
   before updating downstream minor pins.

Every upload skips a version the registry already has: retry a partial
release by re-running the tag's run. Trusted publishing on crates.io, PyPI
and npm names the workflow file, `release.yml`, and the environments
`release` (crates.io, npm) and `pypi`.

Nightly wheels (`molcrafts-molrs-nightly` on PyPI) come from `nightly.yml`
on a push to the `nightly` branch; see "CI" in
`molrs-python/docs/contributing.md`.
