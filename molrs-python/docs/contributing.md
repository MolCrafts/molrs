# Contributing to Documentation

The documentation system has one central invariant: public API prose starts in
Rust `///` comments. Zensical pages may explain workflows and concepts, but
reference pages should inject generated API docs or link to generated API docs
instead of copying signatures by hand.

For Python, keep `molrs-python/python/molrs/_native.pyi` synchronized with the
PyO3 module exports. `molrs-python/tests/test_stub_parity.py` is the freshness
guard and runs in `tox -e py`: it fails when a compiled export is missing from
the stub, and when a parameter name differs between the stub and the compiled
signature.

For WASM, build declarations with the same `wasm-pack` flags used by npm
publishing. The generated `pkg/` directory is ignored and must not be committed.

Local documentation loop. The Zensical config lives at
`molrs-python/zensical.toml`, so build and serve from that directory (its
`[doc]` extra pulls in Zensical plus the mkdocstrings Python handler):

```bash
cd molrs-python
pip install -e ".[doc]"                 # zensical + mkdocstrings[python]
maturin develop                         # build the molrs module so the API reference can introspect it
zensical build                          # reads ./zensical.toml, writes ./site
zensical serve -a localhost:8000        # live preview
```

## Hooks

Every CI gate is a `scripts/check.sh <gate>` call, and `.pre-commit-config.yaml`
runs the same calls as git hooks. Install both hook types once with
`prek install` (or `pre-commit install`). **Never `git commit --no-verify` or
`git push --no-verify`, and never merge a red pull request**: a hook that
fails is a CI job that would have failed.

| Stage | Hooks |
| --- | --- |
| pre-commit | file hygiene (whitespace, final newline, YAML/TOML, merge markers, line endings), `fmt`, and `os-cfg`. Nothing compiles. |
| pre-push | the pre-commit hooks again on `--all-files`; `partners`; `clippy doc test` (molrs core); `ffi`, `cxx`, `python`, `capi`, `wasm` when that binder's files changed, and `ext` (the force-field IR extension proof crate, `molrs-ext-example`) when it or molrs changed; `mrec`; `docs`. |
| CI only | `features` and `package` — run them by hand (`scripts/check.sh features package`) when touching Cargo features or the crate's file list. |

- `partners` — every partner in `.github/partners.env` resolves (see
  [Partners](#partners)), no path dependency points where CI has no checkout,
  and no workflow spells a partner commit of its own.
- `mrec` — molrec's conformance suite through `molrs.io.mrec`, exactly as
  `ci-snapshot.yml` runs it: molrec fetched at its resolved commit into a temp
  dir (never your sibling's working tree), `maturin develop` into a fresh
  Python 3.12 venv, `scripts/ci-conformance.py`. Any case that does not pass
  fails it.
- `docs` — the site as Cloudflare Pages builds it (`.[doc]` in a fresh venv,
  then `zensical build --clean`), with `--strict`, so an mkdocstrings
  reference to a symbol that does not exist fails the push.
- `os-cfg` — `std::os::unix` / `std::os::windows` only directly under a
  `#[cfg(...)]` attribute (line above or same line). A use inside an item that
  is gated further up is fine: end that line with `// os-gated: <why>`.
- Every cargo, maturin and wasm-pack call is `--locked`, every uv call runs on
  Python 3.12 against the committed lock, and the compiler is the one
  `rust-toolchain.toml` pins — the same as CI.

**Dispatch on the MolCrafts cluster.** The checkouts' shared `core.hooksPath`
runs the hooks in place on the login node and sets `MOLCRAFTS_HOOK_RUNNER`.
`scripts/check.sh` keeps `fmt` and `partners` in place and hands every other
gate to that runner, which runs it on a compute node (it reuses the
`$USER-hooks` allocation, or requests one and fails after 20 minutes without a
node — it never passes a gate it did not run). So a commit never waits for
Slurm, and a push waits only when a compiling gate is in scope. Anywhere else
the variable is unset and every gate runs locally, exactly as CI runs it.

## Partners

molrs is judged against molrec's conformance suite (`mrec`). On `dev`,
partners are tracked, not pinned: `.github/partners.env` names molrec's branch
(`MOLREC_REF=dev`), and `scripts/partners.py` resolves it — for CI
(`partners.py resolve`, appended to `$GITHUB_ENV`) and for the hooks
(`partners.py fetch`) alike — to the first of:

1. molrec's branch named like the one being built (CI: the pushed branch or a
   pull request's head branch; locally: the checked-out branch), when
   MolCrafts/molrec has one;
2. outside CI only, that branch in your sibling clone `../molrec`, when it has
   one and the remote does not yet;
3. molrec's `dev`.

So a change that breaks the contract between the two repositories lands as two
same-named branches, never by skipping a gate:

1. Create the same branch (say `converge/x`) in both checkouts and commit each
   side.
2. Push molrs's branch: its pre-push `mrec` gate takes molrec's `converge/x`
   from your sibling clone. Push molrec's: its gates take molrs's from the
   remote.
3. Open both pull requests into `dev`. Each CI run resolves the other's
   `converge/x`. Wait until both are green.
4. Fast-forward both `dev`s to their branch (or merge both pull requests) and
   delete the branches. A `dev` push whose partner's `dev` has not caught up
   yet is re-run once both have landed.

A release judges against fixed partners: see `docs/releasing.md` in the
repository.
