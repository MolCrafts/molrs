# Spec: molrs Unified Documentation System

**Status**: Draft
**Date**: 2026-04-16
**Author**: MolCrafts

## Summary

Replace the ad-hoc per-binding docs (rustdoc-only for Rust, none for Python/WASM)
with a single Zensical site that hosts narrative guides and injects auto-generated
Python and TypeScript API reference via `mkdocstrings`. Rust API reference continues
to live on `docs.rs`. The source of truth for every public binding's documentation
is the Rust `///` comment; PyO3 and wasm-bindgen propagate it downstream.

## Motivation

- **Discoverability**: today, Python and WASM users have no hosted API reference at
  all. Python users fall back to `help()` at the REPL; WASM users read the raw
  `pkg/molwasm.d.ts`.
- **Drift**: any hand-written reference would immediately drift from the Rust
  source. We want a single authoring surface (`///`) that flows to all three
  audiences.
- **README overhead**: per-crate READMEs duplicate build instructions and miss
  cross-links between language bindings.

## Scope

- **Crates affected**:
  - `molrs-python` — keep `python/molrs/molrs.pyi` synced with PyO3 source by
    hand (committed, not generated); audit PyO3 doc comments.
  - `molrs-wasm` — audit wasm-bindgen doc comments (TSDoc tags).
  - No changes to `molrs-core`, `molrs-io`, `molrs-compute`, `molrs-smiles`,
    `molrs-ff`, `molrs-embed`, `molrs-cxxapi` beyond routine rustdoc audits covered
    by the existing `molrs-doc` skill.
- **Traits extended / created**: none.
- **Data structures**: none; the hand-maintained `molrs.pyi` is the one
  authored artifact, and the `.d.ts` / HTML are generated.
- **Feature flags**: none.
- **Out of scope** (see §Migration & Compatibility and the Open Questions table):
  - Translated docs, private docs, versioned docs (`/v0.1/`, `/v0.2/`).
  - Documenting inactive crates (`molrs-ffi`, `molrs-capi`) — they remain
    undocumented until promoted to workspace members.
  - `cargo test --doc` policy changes (tracked separately in the Rust CI config).

## Technical Design

### API Surface

No new public Rust API. The comment-to-docs pipeline has three author-facing
contracts (the "surface" for this spec):

**1. PyO3 items** (`molrs-python/src/**.rs`) — Google-style sections:

```rust
/// Parses a SMILES string and returns a molecular graph IR.
///
/// Args:
///     smiles: SMILES string, e.g. `"CCO"` for ethanol.
///
/// Returns:
///     A `SmilesIR`; call `.to_frame()` to materialize a `Frame`.
///
/// Raises:
///     ValueError: if `smiles` fails to parse.
///
/// Example:
///     ```python
///     import molrs
///     ir = molrs.parse_smiles("CCO")
///     frame = ir.to_frame()
///     ```
#[pyfunction]
pub fn parse_smiles(smiles: &str) -> PyResult<SmilesIR> { ... }
```

**2. wasm-bindgen items** (`molrs-wasm/src/**.rs`) — TSDoc tags:

```rust
/// Parses a SMILES string and returns an intermediate representation.
///
/// @param smiles - SMILES string, e.g. `"CCO"` for ethanol.
/// @returns `SmilesIR` — call `.toFrame()` to obtain a `Frame`.
/// @throws if `smiles` fails to parse.
///
/// @example
/// ```ts
/// const ir = parseSMILES("CCO");
/// const frame = ir.toFrame();
/// ```
#[wasm_bindgen(js_name = parseSMILES)]
pub fn parse_smiles(smiles: &str) -> Result<SmilesIR, JsValue> { ... }
```

**3. Rust public items** — unchanged; follow the existing `molrs-doc` skill.

### Data Flow

```
 Rust ///  ──┬─► (cargo doc)          ─► docs.rs              (Rust reference)
             │
             ├─► (PyO3 compile)       ─► module __doc__
             │                            + hand-maintained molrs.pyi (committed)
             │                            ─► mkdocstrings-python
             │                            ─► Zensical /reference/python/
             │
             └─► (wasm-bindgen)       ─► pkg/molwasm.d.ts (TSDoc)
                                         ─► mkdocstrings-typescript (TypeDoc)
                                         ─► Zensical /reference/wasm/

 docs/**.md (narrative)               ─► Zensical /                (guides, getting-started, changelog)
```

One site, one deployment, three audiences. The Python `.pyi` is the one
hand-maintained artifact — every PR that touches `#[pyfunction]`/`#[pyclass]`
must also update `molrs-python/python/molrs/molrs.pyi`.

### Tooling Choices

| Layer | Choice | Rationale |
|---|---|---|
| Rust API reference | `cargo doc` → `docs.rs` | Free, authoritative, already works on every release. No `mkdocstrings-rust` handler exists. |
| Python API reference | `mkdocstrings-python` | Introspects installed module + `.pyi` stubs; renders Google-style docstrings. |
| Python type stubs | **Hand-maintained `molrs.pyi`** | `pyo3-stub-gen` 0.22 does not support `bool` NumPy arrays (private `NumPyScalar` trait), and our PBC API relies on them; maintaining per-method `skip` workarounds + a local fork is more expensive than syncing the `.pyi` by hand. Owned by the `molrs-documentor` agent. |
| WASM API reference | `mkdocstrings-typescript` (wraps TypeDoc) | Reads wasm-pack's generated `.d.ts`. **Prototype** — fallback plan documented below. |
| Site generator | Zensical (`pip install zensical`) | Material-family MkDocs-compatible, supports `mkdocstrings` plugins natively. |
| Hosting | GitHub Pages (`gh-pages` branch of this repo) | Already configured for org; zero infra cost. |

**Fallback for mkdocstrings-typescript**: its README self-describes as
"prototyping phase." If integration fails in Phase 3, we publish a standalone
TypeDoc site at `/reference/wasm/` merged into the deployed artifact — same URL,
separate build step. The comment pipeline (Rust `///` → `.d.ts`) is unaffected.

### Site Structure

```
docs/
├── zensical.toml                 # Zensical + mkdocstrings config
├── index.md                      # Landing page with three-audience picker
├── getting-started/
│   ├── installation.md           # cargo add / pip install / npm install
│   ├── quickstart-rust.md
│   ├── quickstart-python.md
│   └── quickstart-wasm.md
├── guides/
│   ├── data-model.md             # Frame / Block / SimBox
│   ├── smiles-and-topology.md
│   ├── neighbor-search.md
│   ├── embed-3d.md
│   ├── force-field.md
│   ├── io-formats.md
│   └── trajectory-analysis.md
├── reference/
│   ├── rust.md                   # External link to docs.rs (one crate per row)
│   ├── python.md                 # ::: molrs.Frame / ::: molrs.Block / ...
│   └── wasm.md                   # ::: @molcrafts/molrs / (or TypeDoc iframe)
├── contributing.md
└── changelog.md
```

URL map:

| Audience | URL |
|----------|-----|
| Landing | `https://molcrafts.github.io/molrs/` |
| Python API | `https://molcrafts.github.io/molrs/reference/python/` |
| WASM API | `https://molcrafts.github.io/molrs/reference/wasm/` |
| Rust API | `https://docs.rs/molcrafts-molrs-core` (and sibling crates) |

### CI/CD Pipeline

New workflow `.github/workflows/docs.yml`. Triggers: push to `master`, manual
dispatch, and PRs (PRs run steps 1–6 only, no deploy).

```
1. checkout (submodules: true)
2. install Rust toolchain
3. maturin develop -m molrs-python/Cargo.toml
     (the committed molrs-python/python/molrs/molrs.pyi is consumed as-is)
4. cd molrs-wasm && wasm-pack build --release --target bundler --scope molcrafts
     → molrs-wasm/pkg/molwasm.d.ts
5. pip install zensical mkdocstrings mkdocstrings-python mkdocstrings-typescript
6. zensical build -f docs/zensical.toml --strict
     → site/  (mkdocstrings runs inline)
7. peaceiris/actions-gh-pages — deploy site/ to gh-pages branch (master push only)
```

`zensical build --strict` makes missing cross-links and introspection failures
hard-fail, so a broken `///` → `:::` injection blocks the merge.

### Integration Points

- **Existing CI workflows** (`ci-rust.yml`, `ci-python.yml`, `ci-wasm.yml`) are
  untouched. `docs.yml` is an independent job.
- **Release workflows** (`publish-molrs-*.yml`) remain the triggers for docs.rs
  and npm/PyPI artifacts; docs site is decoupled.
- **README overhaul** (three files) lands in the same PR as Phase 6 so README
  links and Zensical go-live together:
  - Root `README.md`: new "Documentation" section above "Build & Test";
    docs.rs + GitHub Pages badges; crate-dependency diagram (mermaid);
    move verbose build commands to `CONTRIBUTING.md`.
  - `molrs-python/README.md`: link to Zensical Python API page; `help(molrs.Frame)`
    offline tip; Python/PyO3/maturin version compat table.
  - `molrs-wasm/README.md`: link to Zensical WASM API page; TypeScript example;
    feature-flag build variants; browser compat (ES2017+, WebAssembly MVP).

## Constraints & Invariants

- **Single source of truth**: every exported binding's doc content lives in the
  Rust `///` comment on the Rust item. No hand-written Markdown duplicates the
  function/class reference. Docs pages contain `:::` injections or external
  links, never transcribed signatures.
- **Hand-maintained `.pyi`**: `molrs-python/python/molrs/molrs.pyi` is the one
  docs-adjacent artifact we author by hand — kept in sync with the Rust PyO3
  source on every PR that touches `#[pyfunction]` / `#[pyclass]`. All other
  per-run artifacts (`pkg/molwasm.d.ts`, `site/`) are CI-generated and
  `.gitignore`d.
- **Zero behavior change in published artifacts**: `maturin develop` and
  `wasm-pack build` flags in CI match the publish workflows byte-for-byte (same
  features, same profile), so the `.d.ts` that docs sees is exactly what
  downstream consumers see.
- **Inactive crates stay invisible**: the docs site must not render
  `molrs-ffi` or `molrs-capi` public items — they are not workspace members per
  CLAUDE.md and their surfaces are not supported.
- **CLAUDE.md IO testing rule does not apply here**: no new file-format readers
  are introduced.

## Test Criteria

### Unit Tests
- `molrs-python`: existing PyO3 test suite (`molrs-python/tests/`) continues to
  pass.
- `molrs-wasm`: existing `wasm-bindgen-test` suite continues to pass.

### Integration Tests (run in `docs.yml` PR job)
1. **`.pyi` freshness guard**: a CI script imports every public symbol listed
   in `molrs-python/src/lib.rs`'s `#[pymodule]` block and asserts each one is
   declared in `molrs-python/python/molrs/molrs.pyi`. Missing entries fail the
   build and flag the author to update the stub.
2. **`.d.ts` generation**: `wasm-pack build` produces `molrs-wasm/pkg/molwasm.d.ts`
   containing at least one `@param` tag (proves wasm-bindgen propagated a TSDoc
   comment).
3. **Site build**: `zensical build --strict` exits 0 with no broken links or
   unresolved `:::` injections.
4. **Anchor sanity check**: post-build, a small script greps `site/reference/python/index.html`
   for the class names exported from `molrs-python/src/lib.rs` (`Frame`, `Block`,
   `SimBox`, `MolGraph`, ...) and fails if any are missing.
5. **Equivalent pair check**: for every public `#[pyfunction]` whose Rust name
   also appears as a `#[wasm_bindgen]` export, the two reference pages both
   render and their TOC entries cross-link. A script enumerates Rust items and
   asserts the HTML contains the expected anchors on both pages.

### Doc-build Validation (stricter than "it compiles")
- **Docstring lint**: CI step runs `ruff` (or `pydocstyle`) across the installed
  `molrs` module; fails on missing `Args:` / `Returns:` sections for public
  functions.
- **TSDoc lint**: CI step runs `typedoc --validation.invalidLink --validation.notExported`
  (or equivalent) against `pkg/molwasm.d.ts`; fails on broken `@see` references.
- **rustdoc**: `cargo doc --no-deps --all-features -D warnings` as a separate
  step (covers the docs.rs surface) — broken intra-doc links block merge.
- **Doctests**: `cargo test --doc --all-features` remains in the main Rust CI
  (`ci-rust.yml`), unchanged.

### Manual validation (once per milestone)
- Open each quickstart in the deployed site and execute the code verbatim in a
  fresh env (cargo project / `pip install` / `npm install`).
- Confirm every "equivalent API" cross-link lands on the intended sibling page.

## Performance Requirements

- **CI wall time**: `docs.yml` end-to-end target ≤ 8 minutes on `ubuntu-latest`
  (baseline: current `ci-python.yml` ≈ 4 min + `ci-wasm.yml` ≈ 3 min; docs reuses
  their outputs in one job).
- **Site build**: `zensical build` on the full site ≤ 60 s locally.
- **Deployed asset size**: single-page weight ≤ 1 MB gzipped. The raw `.d.ts`
  (currently 78 KB) is rendered, not served.
- No runtime perf impact on `molrs-core` or any other crate.

## Migration & Compatibility

- **No existing docs site exists**, so there is no URL migration. The
  `gh-pages` branch is created by the first successful deploy.
- **Breaking changes**: none for library consumers.
- **Deprecations**: none.
- **Rollback plan**: disable `docs.yml`; `gh-pages` branch remains as a static
  snapshot. docs.rs is unaffected.

## Implementation Phases

Track these as checkboxes in the PR description; each phase is one commit or
one stacked PR.

- **Phase 0 — Source audit**: reconcile `molrs-python/python/molrs/molrs.pyi`
  with the current PyO3 source (fix the stale `np.float32` → `np.float64`,
  add any missing exports); audit PyO3 and wasm-bindgen `///` comments against
  the standards above. Land the freshness-guard CI step (Integration Test #1).
- **Phase 1 — Site skeleton**: install Zensical + plugins; scaffold
  `docs/zensical.toml`; write `index.md` and `getting-started/installation.md`;
  local `zensical build` must succeed.
- **Phase 2 — Python reference**: write `docs/reference/python.md` with `:::`
  injections; resolve any introspection failures (opaque types, missing stubs).
- **Phase 3 — WASM reference**: configure `mkdocstrings-typescript`; write
  `docs/reference/wasm.md`; if the handler is too unstable, switch to standalone
  TypeDoc at `/reference/wasm/` before Phase 4.
- **Phase 4 — Guides**: write the seven guide pages; write `changelog.md` from
  git history; verify all cross-links.
- **Phase 5 — CI/CD**: land `docs.yml`; enable GitHub Pages; verify end-to-end
  deploy from a throwaway branch first.
- **Phase 6 — README overhaul**: update the three READMEs; this is the go-live
  commit and must land *after* Phase 5's first successful deploy.

## Open Questions

| # | Question | Default assumption |
|---|----------|--------------------|
| 1 | Is `mkdocstrings-typescript` stable enough for `master`? | Try in Phase 3; fall back to standalone TypeDoc under the same URL. |
| 2 | Custom domain vs. `molcrafts.github.io/molrs`? | GitHub Pages default for now; revisit after 0.1. |
| 3 | Versioned docs (`/v0.1/`, `/v0.2/`)? | Defer until post-0.1. |
| 4 | Should `docs.yml` PR failures block merge or be advisory? | **Block.** A broken site build on `master` is a worse outcome than a blocked PR. |
| 5 | Should `cargo test --doc` live in `docs.yml` or stay in `ci-rust.yml`? | Stay in `ci-rust.yml` — it's a Rust correctness concern, not a site-build concern. |
