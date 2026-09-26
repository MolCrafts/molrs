---
title: "cgsmiles-03: release molrs 0.15.0"
slug: cgsmiles-03-release
status: in-progress
created: 2026-09-21
chain: cgsmiles (01a → 01b → 01c → 01d → 01e → 02a → 02b → 02c → 02d → 03)
depends_on: cgsmiles-01a-descriptors, cgsmiles-01b-graph, cgsmiles-01c-fragments, cgsmiles-01d-resolve, cgsmiles-01e-python-ir, cgsmiles-02a-fragment-core, cgsmiles-02b-to-fragment, cgsmiles-02c-conformer-fragment, cgsmiles-02d-python-fragment, frame-meta-dict-parity-01-ordered, frame-meta-dict-parity-02-binder-order, frame-meta-dict-parity-03-untyped-write, frame-meta-dict-parity-04-dict-views, frame-meta-dict-parity-05-document
---

# cgsmiles-03: release molrs 0.15.0

## Summary

After every other link of the `cgsmiles-` chain has landed, cut molrs `0.15.0`: move the hand-maintained version literal from `0.14.0` to `0.15.0` at its 16 sites in 7 manifests, move the four user-facing install pins from `version = "0.14"` to `version = "0.15"`, refresh the one tracked lock file, append a `## v0.15.0` section to `.claude/notes/release.md` recording what this release adds, removes and still defers, run the full release gate from `docs/releasing.md:9-64` against the exact tree that will be tagged, land that tree on `master` through a PR, tag `v0.15.0`, and wait for the **Publish** workflow to finish crates.io / npm / PyPI. No behaviour change and no API change **from this link**; the only product is a tagged, published `0.15.0` that unblocks the two downstream repos — molpy's `backmap-` chain (pin `molcrafts-molrs>=0.15.0,<0.16`) and molpack's three `^0.14` path pins, which break in the shared checkout the moment `Cargo.toml:12` reads 0.15.0.

**Revised 2026-09-22.** The `frame-meta-dict-parity-*` chain lands on this same unreleased 0.15 tree and carries breaking Python, C and C++ changes — key enumeration order, the `frame.meta` dtype rule, view return types, and frozen metadata values. That chain does not edit this spec, does not add a version, and does not own a release-notes link. The `## v0.15.0` section and the pre-tag gate stay this spec's tasks; run them against the tree after that chain, not against a later minor.

## Domain basis

None — release mechanics. No equation, no physical unit, no literature reference is involved; the science of the chain was settled in links 01a–02d.

## Design

**Constitution.** `.claude/notes/law.md` does not exist in this repo (checked); the governing rules are `CLAUDE.md` § *Design preferences* — in particular the iron law *no silent debt*, which is why every stale literal this link leaves behind is enumerated and named below rather than passed over — plus `CLAUDE.md` § *Release before molpy* (molrs tag and Publish first; downstream minor pins only after) and `.claude/notes/release.md:1-8`.

**Entities touched.** Four: the manifest version literal, the user-facing install pins, the release-notes section, and the gate run.

### 1. The manifest version literal — 16 sites, 7 files

`Cargo.toml:12` (`[workspace.package] version = "0.14.0"`) is the SSOT the tag must match (`docs/releasing.md:56`), enforced in CI by `.github/workflows/publish.yml:25-37`, which reads `Cargo.toml → workspace.package.version` and rejects any tag that is not `v{version}`. Verified sites:

| File | Lines |
|---|---|
| `Cargo.toml` | 12 |
| `molrs-ffi/Cargo.toml` | 3, 41 |
| `molrs-python/Cargo.toml` | 3, 31, 45 |
| `molrs-python/pyproject.toml` | 7 |
| `molrs-wasm/Cargo.toml` | 3, 47, 48 |
| `molrs-capi/Cargo.toml` | 3, 32, 33 |
| `molrs-cxxapi/Cargo.toml` | 8, 42, 43 |

Each binder is a standalone workspace (`Cargo.toml:3-6`), so it inherits nothing from `[workspace.package]`; its own `version` and each `version = "0.14.0"` inside a `path` dependency on `molcrafts-molrs` / `molcrafts-molrs-ffi` is hand-maintained. The one manifest that *does* inherit is the published crate itself, `molrs/Cargo.toml:3` (`version.workspace = true`) — not edited, and its absence from the diff is a check.

### 2. The install pins — 4 sites, `version = "0.14"` (no patch component)

A `0\.14\.0` sweep cannot see these, because they pin the minor line the way a consumer would write it:

- `README.md:70` — `molcrafts-molrs = { version = "0.14", default-features = false, features = ["io", "smiles", "conformer"] }`
- `docs/interop.md:27` — `molrs = { package = "molcrafts-molrs", version = "0.14", … }`
- `molrs-python/docs/getting-started/quickstart-rust.md:11` — `molrs = { package = "molcrafts-molrs", version = "0.14", features = ["full"] }`
- `molrs/src/lib.rs:8` — the crate-level rustdoc install snippet, inside a ` ```toml ` fence at `:7-9` (the docs.rs front page of `molcrafts-molrs`)

Bumping them is established release practice, not scope creep: `git log -L68,72:README.md` shows af6ea62f moving 0.11→0.12 and d04d558a moving 0.1→0.11, and `docs/releasing.md:47` requires checking version metadata across all manifests before tagging. The fourth site sits in `src/`, which the scope line declared untouched — it is included anyway, under a narrow and stated carve-out: it is a `//!` comment inside a **toml** fence (not a Rust doctest, not compiled, no symbol, no API), it is the same *class* of artifact as the other three (an install pin a user copies), and it is the single most-read install line molrs publishes. Leaving it at 0.14 while `README.md` reads 0.15 is the silent-debt trap the iron law names. If the coordinator declines the `src/` touch, this site moves verbatim into the survivor allowlist in § 4 below plus a `/mol:docs` follow-up, and the diff bound in Task 2 drops from 11 files / 20 lines to 10 / 19.

What is deliberately **not** bumped is ABI-line prose, which describes history rather than pinning a version: `docs/interop.md:123,127,160-162`, `molrs-ffi/src/abi.rs:15`, `molrs-cxxapi/src/lib.rs:920`, plus the migration-guide links (`README.md:154`, `molrs-python/docs/index.md:79`, `molrs-python/zensical.toml:18`) and the `0.14 cut` comments at `molrs-wasm/src/core/block/mod.rs:373,653`.

**Scripted or by hand — decision: scripted, path-restricted, diff-bounded.** Twenty identical literals across eleven files is exactly the shape a hand edit fails at (one missed dependency pin ships a crate whose own version and whose sibling pin disagree). A repo-wide `sed` is worse: `0.14` also appears in GAFF2 force-constant rows (`molrs/src/ff/params/gaff2.rs`), in a `rgba(…, 0.14)` colour (`molrs-python/zensical.toml:43`), in upload timestamps inside `molrs-python/uv.lock`, in historical notes and in the ABI prose above. The rule: two `sed -i` invocations, each given **exactly** its path list, no glob and no recursion — `s/"0\.14\.0"/"0.15.0"/g` over the seven manifests, `s/version = "0\.14"/version = "0.15"/g` over the four pin sites — bounded afterwards by `git diff --name-only` equalling those eleven paths and `git diff --numstat` summing to twenty changed lines. A careful hand edit passes the same bound; the bound, not the tool, is the contract.

### 3. Lock files — one tracked, six ignored

`.gitignore:69-74` ignores `Cargo.lock` and `uv.lock` repo-wide, with a single un-ignore for `molrs-python/uv.lock` (commit 7969d5c2, *chore: stop committing lockfiles*: "molrs ships as a library, so a lock pins nothing downstream; CI resolves fresh"). Therefore:

- **Tracked, and part of this link's commit:** `molrs-python/uv.lock`, whose editable `molcrafts-molrs` entry reads `version = "0.14.0"` at `:294-297`. Regenerated by `uv --directory molrs-python lock`, never hand-edited. Leaving it stale desynchronizes the `tox -e py` gate from `molrs-python/pyproject.toml:7`.
- **Ignored, never committed:** the six `Cargo.lock` files at the six workspace roots. They are regenerated locally (`cargo update -w --manifest-path <root>/Cargo.toml`, or implicitly by any build) purely so the gate run resolves the new version; they are a working-tree side effect, not a deliverable. A commit that contains one is a defect.

### 4. The `0.14.0` sweep and its allowed survivors

The check is `git grep -n '0\.14\.0'`. It will not reach zero, and demanding zero would be wrong. Verified post-bump survivors, all legitimate — note that `git grep` never sees the six ignored `Cargo.lock` files at all, so third-party pins living in them (e.g. `itertools 0.14.0`) are simply not part of this question:

- `.claude/notes/release.md:65,81,106,109,112,115` — the historical `v0.14.0` sections. History; never rewritten.
- `.claude/specs/INDEX.md:17,19,22` — the 0.14.0 release entry. History.
- `.claude/notes/architecture.md:3` — "Generated for 0.14.0", a `/mol:map` regeneration stamp, refreshed by that command, not by this link.
- `molrs-python/src/lib.rs:107` and `molrs-capi/src/lib.rs:154` — illustrative `e.g.` text showing the *shape* of `_ffi_abi_token()`'s tuple and of `molrs_version()`'s string. Neither is a doctest and neither pins anything: both compute from `molrs::VERSION` / `MOLRS_C_API_VERSION` at runtime. They stay, and per the iron law are named here as known stale-by-example text handed to the orchestrator as a `/mol:docs` follow-up, not quietly ignored. (Unlike `molrs/src/lib.rs:8`, which *is* a pin, which is why that one is bumped.)
- `molrs-wasm/pkg/package.json:7` — gitignored (`.gitignore:59`), regenerated by `wasm-pack build`. It cannot appear in `git grep` and must not be edited.

`molrs-python/uv.lock:296` is **not** a survivor: it is tracked and must read `0.15.0` after the regeneration.

### 5. The release-notes section

There is no CHANGELOG (`CLAUDE.md` § *Consuming molrs from other projects*; `docs/releasing.md:7`: history lives in git tags). The equivalent is a `## v0.15.0 (releasing — <date>)` section appended to `.claude/notes/release.md` in the style of `:31-120`: a bullet list of user-visible surface, then the gate table. It lists `parse_cgsmiles` / `CGSmilesIR` (Rust plus `molrs.io.CGSmilesIR`); the fragment SMILES dialect (`parse_fragment_smiles`, `fragment_to_atomistic`, `write_fragment_smiles`); `core::Fragment` with `PortKind`, plus `molrs.Fragment` and `views.Port`; `Conformer::generate<M: ElementGraph>` and the Python `Conformer.generate` widening; `SmilesError.notation` as a **required** constructor argument (a source-level breaking change for any external constructor — none known); `SmilesErrorKind` gaining variants (public, not `#[non_exhaustive]`, so exhaustive matches break); and the deletion of `core::system::mapping::{CGMapping, WeightScheme}` (zero consumers, verified across sibling repos 2026-09-21). Every bullet is a claim about the tagged tree and is verified there first — as of drafting, `molrs/src/core/system/mapping.rs` still exists and `parse_cgsmiles` resolves nowhere, because 01a–02d have not landed.

The section must also carry one **re-deferral**. `.claude/notes/notes.md:130-135` and `.claude/notes/release.md:75` both promise "the wasm `NeighborQuery` symmetry gate is deferred to **0.15**". This link ships no source change, so 0.15.0 does not deliver it; a v0.15.0 section silent about it would let a promise expire unrecorded. The bullet re-defers it explicitly with its reason — wasm still has no consumer (facade-first), and `notes.md:131-135` rules deletion out because the in-tree consumers are `compute/hbond/detect.rs`, `compute/rdf/mod.rs`, `compute/dynamics/van_hove.rs` and `ff/potential/soft.rs`.

The section refers to the previous release as `v0.14` (no patch component) wherever it compares against it, so the `0.14.0` sweep in § 4 stays exact. The section records **what shipped**, not what was noticed. The deferred items the nine implementation summaries flagged for `/mol:note` (the two SMARTS parsers, `[#6]`/`Element::by_number`, the `%n` mislabel, module inception, `frag_id` vs `res_id`, the trait-principle-1 amendment, the `read_frame` skip, the `emit_column` mask drop, the six `let _ =`, the `InvalidElement` mislabel, the `add_hydrogens` bond-number/mass rot, the docstring dead names, stub-parity method level, `smiles_error_to_pyerr` flattening) belong to the orchestrator's `/mol:note` run. The notes section must not name any of them as fixed — a release note claiming an unfixed item is silent debt inverted.

### 6. The gate run — the iron law of this link

Nothing lands on a red gate. Every command in `docs/releasing.md:9-64` runs against the exact commit that gets tagged: six per-manifest `cargo fmt --check`; five `cargo clippy` runs (`molcrafts-molrs`, cxxapi, python, capi, and wasm with `--target wasm32-unknown-unknown`); `RUSTDOCFLAGS="-D warnings" cargo doc`; `cargo test -p molcrafts-molrs --features full,filesystem,stream` (`releasing.md:24` — no `--lib`, so this one command covers lib tests *and* doctests; a separate `--doc` run would be redundant) plus `cargo test` for molrs-ffi and molrs-cxxapi; `cargo package --manifest-path molrs/Cargo.toml` and `--list`; `tox -e py` in molrs-python; `wasm-pack build --release --target bundler` and `wasm-pack test --node`; the CMake/ctest C-ABI suite. Recorded counts are **measured at that tree**, never carried forward — `.claude/notes/release.md:95-98` documents that exact mistake on 0.14.0 (2068 and 1188 both stale). The 0.14.0 table (`:89-91`: 2045 lib / 74 doctests / 557 python) was moreover measured under `full,filesystem` **without** `stream`, so it is not even comparable to this gate's numbers; it is a precedent for the format, not a baseline.

### 7. Publish, and the two downstream repos

`.github/workflows/publish.yml` triggers on `push` of tags matching `v*` (`:5-8`) and on `workflow_dispatch`. `guard` (`:15-46`) asserts the tag equals `v{[workspace.package].version}` and that the tagged commit is an ancestor of `origin/master`; `ci` (`:48-50`) must be green before any registry job runs. Publishing jobs: `publish-molrs` → crates.io (`molcrafts-molrs`), `publish-wasm` → npm (`@molcrafts/molrs`), `build-python` + `build-python-pyodide` → wheel artifacts, `publish-python` → PyPI via trusted publishing, `build-capi` / `release-capi` → C API archives on GitHub Releases. A branch dispatch never publishes. `master` is protected, so the tree lands by PR (as 0.14.0 did).

Two downstream repos are named here and edited by neither this link nor this repo:

- **molpy** — `molpy/pyproject.toml:34`, currently `molcrafts-molrs>=0.14.0,<0.15`; moves to `>=0.15.0,<0.16` in the `backmap-` chain, only after Publish (the release iron law).
- **molpack** — three `^0.14` requirements on **path** dependencies into this very checkout: `molpack/Cargo.toml:26` (`molrs = { path = "../molrs/molrs", version = "0.14", package = "molcrafts-molrs" }`), `molpack/python/Cargo.toml:22` (same crate) and `:28` (`molcrafts-molrs-ffi`). A path dep still honours its version requirement, so every molpack build in the shared checkout hard-fails the moment `Cargo.toml:12` reads 0.15.0 — not at publish time, immediately. molpack also reads the `_ffi_abi_token()` handshake (`molrs-python/src/lib.rs:104-116`), whose `abi_line` moves 0.14 → 0.15, so a 0.14-built molpack wheel raises `ImportError` against a 0.15 molrs wheel by design. This is expected and correct, but it means molpack is *broken from the bump commit until it follows*; Task 8's report must say so explicitly so the operator sequences it.

The uv-path consumers molrec / molab / molhub / molexp carry no molrs version requirement and are unaffected.

### Reuse decision

- `pattern` — the per-version section style at `.claude/notes/release.md:31-120`: **reuse** verbatim (heading `## vX.Y.Z (releasing — YYYY-MM-DD)`, bullet list, gate table). No new notes format.
- `reuse` — the checklist at `docs/releasing.md:9-64` plus `.claude/notes/release.md:1-8`, executed as written. No new script and no helper: `.claude/notes/release.md:26-29` rules out publish helper scripts and `CLAUDE.md` § *Release before molpy* rules out pin-parity scripts. The two `sed` invocations are commands run once, not committed artifacts.
- `reuse` — the version/branch guard in `.github/workflows/publish.yml:25-46`; the local checks read the same `Cargo.toml:12` rather than reimplementing the invariant.
- `reuse` — `cargo update -w` and `uv lock` for lock regeneration, `wasm-pack` for `molrs-wasm/pkg/package.json`.
- `reuse` — the install-pin bump precedent from af6ea62f / d04d558a (`git log -L68,72:README.md`).
- `new` — none. This link introduces no symbol and no file.

## Files to create or modify

Hand-edited manifests (16 literals, `"0.14.0"` → `"0.15.0"`):

- `Cargo.toml` — `[workspace.package] version` at :12 (the tag SSOT)
- `molrs-ffi/Cargo.toml` — :3, :41
- `molrs-python/Cargo.toml` — :3, :31, :45
- `molrs-python/pyproject.toml` — :7
- `molrs-wasm/Cargo.toml` — :3, :47, :48
- `molrs-capi/Cargo.toml` — :3, :32, :33
- `molrs-cxxapi/Cargo.toml` — :8, :42, :43

Hand-edited install pins (4 sites, `version = "0.14"` → `version = "0.15"`):

- `README.md` — :70
- `docs/interop.md` — :27
- `molrs-python/docs/getting-started/quickstart-rust.md` — :11
- `molrs/src/lib.rs` — :8 (rustdoc toml fence at :7-9; carve-out stated in Design § 2)

Regenerated by tooling, tracked, committed unedited:

- `molrs-python/uv.lock` — `molcrafts-molrs` entry at :294-297

Documentation:

- `.claude/notes/release.md` — append a `## v0.15.0` section after :120

Regenerated locally for the gate run, **never committed** (ignored via `.gitignore:69-72`): `Cargo.lock`, `molrs-ffi/Cargo.lock`, `molrs-python/Cargo.lock`, `molrs-wasm/Cargo.lock`, `molrs-capi/Cargo.lock`, `molrs-cxxapi/Cargo.lock`.

Read but not modified: `docs/releasing.md`; `.github/workflows/publish.yml`; `molrs/Cargo.toml` (`:3` inherits); `molrs-wasm/pkg/package.json` (ignored, `wasm-pack`-generated); `molrs-python/src/lib.rs:107`, `molrs-capi/src/lib.rs:154`, `molrs-ffi/src/abi.rs:15`, `molrs-cxxapi/src/lib.rs:920`, `docs/interop.md:123,127,160-162` (illustrative / historical ABI prose); `molpy/pyproject.toml:34`, `molpack/Cargo.toml:26`, `molpack/python/Cargo.toml:22,28` (other repos, other chains).

## Tasks

- [x] Verify the chain has landed: `parse_cgsmiles`, `CGSmilesIR`, `parse_fragment_smiles`, `fragment_to_atomistic`, `write_fragment_smiles`, `core::Fragment`, `PortKind` resolve in `molrs/src/`, `SmilesError::new` requires `notation`, and `molrs/src/core/system/mapping.rs` with `CGMapping`/`WeightScheme` is gone — abort this link if any claim fails
- [x] Bump the 16 `"0.14.0"` literals in `Cargo.toml`, `molrs-ffi/Cargo.toml`, `molrs-python/Cargo.toml`, `molrs-python/pyproject.toml`, `molrs-wasm/Cargo.toml`, `molrs-capi/Cargo.toml`, `molrs-cxxapi/Cargo.toml` and the 4 `version = "0.14"` install pins in `README.md:70`, `docs/interop.md:27`, `molrs-python/docs/getting-started/quickstart-rust.md:11`, `molrs/src/lib.rs:8` via two path-restricted `sed` runs, then bound the diff (`git diff --name-only` = those 11 paths, `git diff --numstat` = 20 lines, `molrs/Cargo.toml` absent)
- [x] Regenerate the one tracked lock file, `molrs-python/uv.lock`, with `uv --directory molrs-python lock` and commit it unedited; refresh the six ignored `Cargo.lock` files locally with `cargo update -w --manifest-path <root>/Cargo.toml` for the gate run only, and confirm none of them enters the commit
- [x] Prove the sweeps: `git grep -n 'version = "0\.14"'` returns nothing; every line of `git grep -n '0\.14\.0'` is an allowed survivor (`.claude/notes/release.md:65,81,106,109,112,115`; `.claude/specs/INDEX.md:17,19,22`; `.claude/notes/architecture.md:3`; `molrs-python/src/lib.rs:107`; `molrs-capi/src/lib.rs:154`); and `cargo metadata` at each of the six roots resolves every workspace member and every `molcrafts-molrs*` dependency to 0.15.0
- [x] Append the `## v0.15.0` section to `.claude/notes/release.md` in the `:31-120` style, naming each user-visible addition, the `SmilesError.notation` / `SmilesErrorKind` breaking changes, the `CGMapping`/`WeightScheme` deletion, and the explicit re-deferral of the wasm `NeighborQuery` symmetry gate (`notes.md:130-135`, `release.md:75`) — claiming no deferred `/mol:note` item as fixed
- [x] Run every gate in `docs/releasing.md:9-64` against the exact tree to be tagged (6× `cargo fmt --check`, 5× `cargo clippy -D warnings` incl. the wasm target, `RUSTDOCFLAGS="-D warnings" cargo doc`, `cargo test -p molcrafts-molrs --features full,filesystem,stream`, `cargo test` for molrs-ffi and molrs-cxxapi, `cargo package` and `--list`, `tox -e py`, `wasm-pack build` + `test --node`, `ctest`) and record the freshly measured numbers into the new `.claude/notes/release.md` section per `:95-98`
- [ ] Open the PR into protected `master`, merge green, then create and push tag `v0.15.0` matching `Cargo.toml:12`
- [ ] Wait for **Publish** (`.github/workflows/publish.yml`, trigger `push` tags `v*`) to finish crates.io + npm + PyPI incl. Pyodide + C API assets, smoke-test the published crate and wheel, and report molrs 0.15.0 released — naming both downstream follow-ups: `molpy/pyproject.toml:34` may move to `>=0.15.0,<0.16`, and molpack must move `molpack/Cargo.toml:26` and `molpack/python/Cargo.toml:22,28` off `^0.14` (broken in the shared checkout from the bump commit onward) and rebuild against the 0.15 `_ffi_abi_token` handshake line

## Testing strategy

There is no code under test: this link changes no compiled line (`molrs/src/lib.rs:8` is a toml fence in a `//!` block), so there is no unit test to write first and none to add — `CLAUDE.md` § *Testing Rules* puts molrs unit tests in `#[cfg(test)]` next to the code they exercise. Verification is the release gate plus a set of file-state checks, each a shell command whose exit status is the verdict.

- **Manifest consistency (happy path).** `git grep -n '"0\.14\.0"' -- '*Cargo.toml' '*pyproject.toml'` returns nothing; `git grep -c '"0\.15\.0"' -- '*Cargo.toml' '*pyproject.toml'` returns 1, 2, 3, 1, 3, 3, 3 for the seven manifests in the order listed above (16 total). The path restriction matters on both clauses: without it, `molrs-python/uv.lock` adds an eighth row and the count no longer reads as a per-manifest tally.
- **Resolver consistency.** For each of the six roots, `cargo metadata --format-version 1 --no-deps --manifest-path <root>/Cargo.toml | jq -r '.packages[] | "\(.name) \(.version)"'` shows 0.15.0 on **every** row — no `startswith("molcrafts")` filter, which would print nothing at all for the python root (member `molrs-python`) and the wasm root (member `molrs`) and so pass vacuously. Separately, the with-deps form `cargo metadata --format-version 1 --manifest-path <root>/Cargo.toml` must exit 0 (a path dep whose `version` requirement no longer matches its target is a hard resolve error — this is what catches a missed pin among the nine path-dependency sites) and every `molcrafts-molrs*` package it lists reads 0.15.0.
- **Install pins.** `git grep -n 'version = "0\.14"'` returns nothing; the four sites read `version = "0.15"`. Verified that no other line in the repo has that exact form, so the sweep needs no path restriction — the neighbouring `0.14` strings (`rgba(…, 0.14)`, migration-guide links, GAFF2 rows, ABI prose) do not match it.
- **Edge case — the inheriting manifest.** `molrs/Cargo.toml` must not appear in `git diff --name-only`; its `version.workspace = true` (`:3`) resolves through `Cargo.toml:12`. Its presence means the scripted replacement escaped its path list.
- **Edge case — lock-file tracking.** `git ls-files --error-unmatch molrs-python/uv.lock` exits 0 (tracked; must be in the commit, reading 0.15.0), while `git ls-files --error-unmatch Cargo.lock` and each binder `Cargo.lock` exit non-zero (ignored per `.gitignore:69-72`; regenerated locally, never committed). `git status --porcelain` on an ignored path is tautologically empty and proves nothing — tracking is asserted with `ls-files`, not `status`.
- **Edge case — the generated wasm package.** `git ls-files --error-unmatch molrs-wasm/pkg/package.json` exits non-zero (ignored per `.gitignore:59`); after `wasm-pack build` its `version` reads `0.15.0` with no edit.
- **Edge case — stale literal sweep.** `git grep -n '0\.14\.0'` output is compared against the five-entry survivor list; any hit outside it fails the link. Note that `git grep` sees no lock file except `molrs-python/uv.lock`, so third-party versions inside the ignored `Cargo.lock` files are outside this check entirely.
- **Release notes (docs).** The new section exists, is headed `## v0.15.0`, and names each of: `parse_cgsmiles`, `CGSmilesIR`, `parse_fragment_smiles`, `fragment_to_atomistic`, `write_fragment_smiles`, `Fragment`, `PortKind`, `Conformer::generate`, `SmilesError`, `notation`, `SmilesErrorKind`, `CGMapping`, `WeightScheme`, and `NeighborQuery` (the last as a re-deferral with its reason, not as a shipped item). Negative check: none of the deferred `/mol:note` item names (`Element::by_number`, `%n`, `frag_id`, `read_frame`, `emit_column`, `let _ =`, `InvalidElement`, `add_hydrogens`, `smiles_error_to_pyerr`) appears as fixed.
- **Gate (runtime).** Every command in `docs/releasing.md:9-64` exits 0 on the tagged tree; the numbers written into the notes table come from that run (`git rev-parse HEAD` at measurement equals the tagged commit). The 0.14.0 figures are not a baseline: `.claude/notes/release.md:89-91` measured them under `full,filesystem` without `stream`, and `:95-98` records the cost of trusting a carried-over number.
- **Tag and publish (runtime).** `git describe --exact-match --tags` on the merge commit is `v0.15.0`; `"v" + tomllib(Cargo.toml).workspace.package.version` equals the tag — the same assertion `publish.yml:25-37` makes; the tagged commit is an ancestor of `origin/master`; all Publish jobs conclude green; `cargo search molcrafts-molrs`, `npm view @molcrafts/molrs version` and `pip index versions molcrafts-molrs` report 0.15.0.

## Out of scope

- **Behaviour, API and compiled code.** Nothing under any `src/` tree changes except the four-word install pin in the `//!` toml fence at `molrs/src/lib.rs:8`; no symbol, signature or test moves. The two illustrative `e.g.` version literals (`molrs-python/src/lib.rs:107`, `molrs-capi/src/lib.rs:154`) are named in the Design and handed to `/mol:docs`, not fixed here.
- **ABI-line and migration prose.** `docs/interop.md:123,127,160-162`, `molrs-ffi/src/abi.rs:15`, `molrs-cxxapi/src/lib.rs:920`, `README.md:154`, `molrs-python/docs/index.md:79`, `molrs-python/zensical.toml:18`, `molrs-wasm/src/core/block/mod.rs:373,653` — historical statements, verified, untouched. Writing a 0.13→0.14-style migration page for 0.15 is a separate `/mol:docs` job.
- **A CHANGELOG.** Refused by `CLAUDE.md` and `docs/releasing.md:7`; considered and rejected because it creates a second history that drifts from the tags. The `.claude/notes/release.md` section is the equivalent.
- **Committing any `Cargo.lock`.** Ignored by design (7969d5c2); regenerated locally for the gate only.
- **Editing `molrs-wasm/pkg/package.json`** (ignored, `wasm-pack`-generated) **or hand-editing `molrs-python/uv.lock`** (tool-regenerated).
- **molpy.** `molpy/pyproject.toml:34` is another repo and the `backmap-` chain's work; by the release iron law it moves only after Publish completes here.
- **molpack.** `molpack/Cargo.toml:26` and `molpack/python/Cargo.toml:22,28` pin `^0.14` on path dependencies into this checkout and will fail to resolve from the bump commit onward; fixing them is molpack's own change, sequenced after Publish, and this link only reports it.
- **Delivering the wasm `NeighborQuery` symmetry gate.** No source change ships here; the release notes re-defer it with its reason.
- **Recording the deferred implementation-summary items.** `/mol:note`, run by the orchestrator; this link only proves the release notes do not claim them.
- **Refreshing `.claude/notes/architecture.md:3`.** That stamp is `/mol:map`'s.
- **A `regressions/` example.** molrs has no `regressions/` tree (`CLAUDE.md`: the regression and benchmark systems are being redesigned outside this repo), and a release link with no behaviour change has nothing to pin; the gate run is the verification.

## Re-run after the assembly chain

The `assembly-*` chain (assembly-01 §0.12) changes the public surface of the
unreleased 0.15 tree after this link verified its notes and gate. The
`## v0.15.0` section of `.claude/notes/release.md` and the gate are re-run on
the post-chain tree (ac-007, ac-009, ac-010). The section must add:

- the new always-on `molrs::op` module (numeric base beneath `core`);
- the assembly surface: `FragGraph`, `Mapping`, `FragLibrary.map` (coarse-type
  → template-label rules), `TracePlacer` / orienters, `PortReacter`,
  `Finalizer`, `Assembler`, `MolGraph::replicate`, `CGSmilesIR::to_template` /
  `to_frag_graph`, `CoarseGrain::from_atom_frame`, `Frame::convert_units`, LJ
  `lj_mass` / `lj_charge` / `define_lj_sigma`;
- the retirements: `SiteMap`, the `site` / `q0` keys, `LineOrienter` /
  `TangOrienter`, the whole-graph `TracePlacer`, `replicate(n)`,
  `compute::density::kabsch`, the `core::math` pure functions,
  `Trace::from_arrays` / `tangent`;
- `FRAME_VOCAB_VERSION` 2;
- `ColumnSpec.unit` replaced by a typed `dimension`, with the schema
  document's unit column now derived;
- the LAMMPS data reader refusing unknown sections, incomplete / duplicate /
  unknown-id per-atom rows, repeated sections, and header lines it does not
  read (incl. the general-triclinic `avec` / `bvec` / `cvec` / `abc origin`
  keywords, previously ignored with the box left unset); `with_skipped_section`;
- the Python builder classes moving under `molrs.builder`.
- `ScaleLjError::InvalidMass` (a non-finite fragment mass is refused).
