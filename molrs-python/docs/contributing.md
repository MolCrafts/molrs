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
| pre-commit | file hygiene (whitespace, final newline, YAML/TOML, merge markers, line endings), `fmt`, `ruff`, and `os-cfg`. Nothing compiles. |
| pre-push | the pre-commit hooks again on `--all-files`; `partners`; `clippy doc test` (molrs core); `ffi`, `cxx`, `python`, `capi`, `wasm` when that binder's files changed, and `ext` (the force-field IR extension proof crate, `molrs-ext-example`) when it or molrs changed; `mrec`; `docs`. |
| CI only | `features` and `package` — run them by hand (`scripts/check.sh features package`) when touching Cargo features or the crate's file list. |

- `ruff` — `ruff.toml`'s one rule, PLW1514: text-mode file I/O (`open`,
  `read_text`, `write_text`) names its encoding. Unnamed, Python decodes with
  the locale, cp1252 on Windows, so the read fails only on CI's Windows leg.
  Ruff sees only receivers it can type (`Path(...)`, annotated names); spell
  `encoding="utf-8"` on every text read and write regardless.
- `partners` — every partner in `.github/partners.env` resolves (see
  [Partners](#partners)), no path dependency points where CI has no checkout,
  and no workflow spells a partner commit of its own.
- `mrec` — molrec's conformance suite through `molrs.io.mrec`, exactly as
  `test.yml`'s `test / mrec` runs it: molrec fetched at its resolved commit into a temp
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
`scripts/check.sh` keeps `fmt`, `ruff` and `partners` in place and hands every other
gate to that runner, which runs it on a compute node (it reuses the
`$USER-hooks` allocation, or requests one and fails after 20 minutes without a
node — it never passes a gate it did not run). So a commit never waits for
Slurm, and a push waits only when a compiling gate is in scope. Anywhere else
the variable is unset and every gate runs locally, exactly as CI runs it.

## CI

One workflow per kind of work, each job a `scripts/check.sh` call. Every
push of any branch runs `lint`, `test` and `docs`, on a fork as on
MolCrafts. A pull request into `dev` or `master` runs them again unless it is
a pull request inside a fork (that was already run by its push). Those
decisions (tier, fork or upstream, the duplicate pull request) are made in
one place: every workflow's first job, `<file> / context`, runs
`MolCrafts/molcrafts-ci/actions/ci-context@master`, and every other job reads
its outputs.

| workflow | feature-branch push to MolCrafts | everything else: any push to a fork, `dev`/`master` on MolCrafts, pull requests, tags, dispatches | upstream only |
| --- | --- | --- | --- |
| `lint.yml` | `lint / hooks` (commit hooks on every file, `partners`), `lint / clippy` (`clippy doc`), `lint / workflows` (`check-workflows`) | same | — |
| `test.yml` | fast: `test / rust` (`test`), `test / python (ubuntu-latest)` | full: `test / rust` (+ `ffi cxx ext package`), `test / python` on Linux, macOS and Windows, `test / features`, `test / capi`, `test / wasm`, `test / mrec` | — |
| `docs.yml` | `docs / build` (`docs`) | same | Cloudflare Pages deploys the site from MolCrafts |
| `nightly.yml` | — | — | nightly: tests, coverage and conformance snapshots to molcrafts-ci; a `nightly` branch push: wheels to `molcrafts-molrs-nightly` |
| `release.yml` | — | dispatch: dry run (builds, uploads nothing) | `v*` tag: crates.io, npm, PyPI, GitHub Release (`docs/releasing.md`) |

So a fork branch gets the full tier on its push: push to your fork, wait for
green, then open the pull request into MolCrafts `dev`. Branches pushed to
MolCrafts itself (Dependabot's) get the fast tier, and their pull requests the
full one. The `require-green-ci` (`dev`) and `protect-master` rulesets require
`test / context` and the full tier's jobs. Shared setup is
`MolCrafts/molcrafts-ci/actions/<name>@master` (`setup-rust`, `setup-python`);
only `setup-wasm` is molrs's own, in `.github/actions/`.

## Partners

molrs is judged against molrec's conformance suite (`mrec`). On `dev`,
partners are tracked, not pinned: `.github/partners.env` names molrec's branch
(`MOLREC_REF=dev`), and `scripts/partners.py` resolves it — for CI and the
hooks alike (`partners.py fetch`) — to the first of:

1. molrec's branch named like the one being built (CI: the pushed branch or a
   pull request's head branch; locally: the checked-out branch), looked up
   first on the fork the build comes from (`<owner>/molrec`, where `<owner>`
   owns the pull request's head repository or the repository CI runs in; in a
   git hook, the remote being pushed to), then on MolCrafts/molrec;
2. outside CI only, that branch in your sibling clone `../molrec`, when it has
   one and neither remote does yet;
3. MolCrafts/molrec's `dev`.

So a change that breaks the contract between the two repositories lands as two
same-named branches, never by skipping a gate:

1. Create the same branch (say `converge/x`) in both checkouts and commit each
   side.
2. Push both branches to your forks, never to MolCrafts. molrs's pre-push
   `mrec` gate takes molrec's `converge/x` from your sibling clone (or your
   fork, once pushed); molrec's gates take molrs's from your fork.
3. Run CI on the forks: each push runs the full tier there (see [CI](#ci)),
   and each run resolves the other's `converge/x` on your fork.
4. Only once both forks are green, open the pull requests from the forks into
   MolCrafts `dev`; their CI again resolves each other's branch on your fork.
   Merge both once green (never a red one), then delete the branches. A `dev`
   push whose partner's `dev` has not caught up yet is re-run once both have
   landed.

A release judges against fixed partners: see `docs/releasing.md` in the
repository.
