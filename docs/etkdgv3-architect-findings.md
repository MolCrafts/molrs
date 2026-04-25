# ETKDGv3 Spec — Architect Review Findings

Review of `docs/etkdgv3-port-spec.md` against the `molrs-arch` skill, performed by the `molrs-architect` agent at the start of session 1.

This document captures findings that were **not resolved in session 1** because their fix belongs to a later session (typically the session that adds the relevant inter-crate edge or module). Findings that WERE addressed in session 1 are crossed out.

## CRITICAL

### C1. Undocumented inter-crate edges violate the arch DAG

Spec §4/§5.2 has `molrs-embed` depend on both `molrs-ff` (UFF table) and `molrs-smiles` (SMARTS matcher). The arch skill only permits sibling edges `molrs-ff → molrs-io` and `molrs-cxxapi → molrs-core, molrs-io`; everything else may depend on `molrs-core` only.

**Fix required before wiring the deps**: in the same PR that adds `molrs-ff` and `molrs-smiles` to `molrs-embed/Cargo.toml`, update:
- `.claude/skills/molrs-arch/SKILL.md` lines 23–29 (dependency rules)
- `CLAUDE.md` workspace-crates section (lines ~90–100) — add `molrs-embed → molrs-ff` and `molrs-embed → molrs-smiles` to the DAG diagram

Non-compliance on day one if not done.

## HIGH

### ~~H1. `SubstructureMatcher` Send+Sync obligation only implicit~~

**Resolved in session 1**: trait declared as `pub trait SubstructureMatcher: Send + Sync` in `molrs-smiles/src/smarts/matcher.rs`. Compile-time assertion on `SmartsPattern: Send + Sync` added in `molrs-smiles/src/smarts/pattern.rs`.

### ~~H2. `chem/rings.rs` risks owning ring perception twice~~

**Resolved in session 1**: `chem/` module created without `rings.rs`. Ring perception stays owned by `molrs-core::rings`; any future `chem::rings` will be a pure `pub use molrs_core::rings::{…}` façade per architect recommendation.

### H3. `EmbedReport` public-field replacement is unflagged breaking change

Current `EmbedReport` (`molrs-embed/src/report.rs` L41–52) exposes `embed_algorithm_used`, `forcefield_used`, `stages: Vec<StageReport>`, `final_energy`. Spec §5.1 replaces every field without an explicit breaking-change subsection. `StageReport` and `StageKind` also public, disappear.

**Fix required during embed rewrite**: add "Removed public items" subsection to spec §9 listing `StageKind`, `StageReport`, and every removed `EmbedReport` field. Downstream `molrs-python/src/embed.rs` uses these — migration notes required.

### H4. Generated tables will exceed 800-line ceiling

254 SMARTS × 6 Fourier terms with provenance comments + fallback generics typically expand to >1500 lines after `build.rs` emits the static table. Same for UFF (~13k lines when expanded).

**Fix required before M2 landing**: emit generated tables to `OUT_DIR/etkdg_v3_table.rs`, `OUT_DIR/uff_params.rs`, etc., and `include!` them. Keep `torsions/library.rs` and `uff/params.rs` as loader shells only.

### H5. `minimize/dg_ff.rs` + `etk_ff.rs` + `contribs.rs` risk exceeding 800 lines each

Current `molrs-embed/src/optimizer.rs` is 26 KB alone. Spec concentrates DG terms, chirality, 4D, torsion, improper into two files.

**Fix required during minimization-module rollout**: split `contribs.rs` into per-term files (`contrib_dist.rs`, `contrib_chiral.rs`, `contrib_fourth.rs`, `contrib_torsion.rs`, `contrib_improper.rs`), each 150–300 lines.

## MEDIUM

### M1. `EtkdgStage` trait name violates naming convention

Skill §"Naming Conventions" says traits are "PascalCase capability noun" — `NbListAlgo`, `Potential`, `Typifier`, `Restraint`. `EtkdgStage` is domain-prefixed and not a capability.

**Fix**: rename to `MinimizationStage` OR drop the trait entirely (three stages sharing a thin interface rarely earn dyn-dispatch; an enum works).

### M2. `EtkdgForceField` enum is a closed-set toggle — OK, but document it

Spec §5.1 introduces `force_field: EtkdgForceField { Etkdg | EtkdgPlusUff }`. Workspace convention for open-ended compositional force fields is `KernelRegistry`. With only two variants this is fine today.

**Fix**: acknowledge in spec §5.1 that this is a closed-set user-facing toggle that will NOT grow; or route through `KernelRegistry` if a third option ever appears.

### M3. ~~`Conformer` public name collision~~ — non-issue

Grep confirms zero existing `struct Conformer` / `pub struct Conformer` in the workspace. Closed.

### M4. UFF diverges from MMFF's runtime-XML approach — justify in spec

MMFF parameters currently embedded via XML `include_str!` at runtime. Spec introduces first `build.rs` in `molrs-ff`.

**Fix**: spec §5.3 should justify why UFF uses text-table + `build.rs` (answer: this is the stated future standard; MMFF should eventually migrate). Add one-line migration note for MMFF as a follow-up item.

### M5. `EmbedFailureCause` requires Hash/Eq derives

Spec uses `HashMap<EmbedFailureCause, usize>` without listing derives.

**Fix**: state `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]` in §5.1.

### M6. File-count inflation in `etkdg/bounds/` and `etkdg/embed/` — NIT only

Eight+ small files is on the high side of "many small files" guidance but acceptable.

## LOW

### L1. `force_field: EtkdgForceField` lacks default

Knobs should document `Default` impl. **Fix**: add `Default` clause to spec §5.1.

### ~~L2. `EtkdgOptions` `#[derive(Debug, Clone)]` missing from spec~~

Workspace convention is `Debug + Clone` on every public struct. Already incorporated in session-1 SMARTS stubs; document in spec §5.1.

### L3. Spec uses `f64` in public signatures — grey area

CLAUDE.md says algorithm code uses `F`/`I`/`U` aliases. Existing `EmbedReport.final_energy: Option<f64>` sets a precedent for public API.

**Fix**: note explicitly in §6 that public API uses bare `f64` for stability while internal algorithm code uses `F`.

### L4. ~~FFI confirmed clean~~

`molrs-cxxapi/src/bridge.rs` touches `Frame`/`FrameView` only; embed not re-exported across the CXX bridge. Spec §4 correctly lists no cxxapi changes.

## NIT

### N1. Spec §5.2 comment "may proxy to molrs-core::rings" — decide now

**Decision recorded in session 1**: `chem/` does NOT own ring perception. If a `chem::rings` module ever appears, it is a pure re-export façade (`pub use molrs_core::rings::{…}`) — no duplication of logic.

### N2. Crate layout diagram in §5.2 omits `build.rs` files

Spec's module trees for `molrs-ff` and `molrs-embed` do not show `build.rs`, though §5.3 requires them. **Fix**: add to module-tree diagrams.

### N3. `etkdg/` sub-module asymmetry

Bare `chirality.rs` and `stereo.rs` at the same level as `bounds/`, `embed/`, `minimize/`, `torsions/` directories is asymmetric. **Fix**: either nest as `etkdg/verify/{chirality,stereo}.rs` or flatten the rest. Pick one.

## Skill amendment scheduled

The arch skill (`.claude/skills/molrs-arch/SKILL.md` lines 23–29, 33–42) must be updated in the same PR as C1: add `molrs-embed` sibling edges to ff and smiles, and list `uff`, `etkdg`, SMARTS `matcher`/`predicate`/`compile`/`recursive` under Module Ownership.
