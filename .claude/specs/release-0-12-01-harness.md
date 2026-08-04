---
slug: release-0-12-01-harness
status: done
created: 2026-08-04
grilled: false
revised: 2026-08-04
---

# release-0-12-01-harness — 0.12 harness map & unit SSOT

## Summary

Regenerate the agent harness so notes and CLAUDE.md match the single-crate molrs tree and the project unit system (analysis time = **fs**). After this spec, architects, librarians, and later 0.12 tasks navigate real modules instead of the deleted multi-crate layout.

## Domain basis

Project analysis units (binding for all later science specs):

| Quantity | Unit |
|---|---|
| Length | Å |
| Energy | kcal/mol |
| Time (analysis `dt`, lags) | **fs** |
| Velocity (LAMMPS `real`) | Å/fs |
| Charge | e |
| Temperature | K |

LAMMPS `units real` time is femtoseconds ([docs.lammps.org/units.html](https://docs.lammps.org/units.html)). Dual docs that claim dielectric/conductivity use **ps** are obsolete and must be removed from notes in this spec; code prefactor migration is **release-0-12-04**.

## Design

- Run inventory of live crates/modules under `molrs/`, binders (`molrs-python`, `molrs-wasm`, `molrs-capi`, `molrs-cxxapi`, `molrs-ffi`).
- Rewrite `.claude/notes/architecture.md` managed blueprint (module graph, not multi-crate).
- Rewrite `.claude/notes/architecture-rules.md` ownership table to single-crate spine: `core → perceive → {io, ff} → conformer`; `compute → signal`; binders separate.
- Rewrite `.claude/notes/ffi.md` to list **four** active binders + shared `molrs-ffi` handles; keep “no panics in extern”.
- Update `.claude/notes/science.md` unit table if needed (confirm Time=fs; remove any ps dual for analysis).
- Patch CLAUDE.md residual lies only: “optimize always-on”, Gasteiger under perceive, SMARTS still in `core/chem`, multi-crate wording.
- Do **not** implement science kernels or delete APIs here.

### Reuse decision

- `reuse` CLAUDE.md crate-structure section as SSOT for module list when rewriting rules.
- `reuse` ffi.md Rule 3 (no panics) — keep, expand binder inventory only.
- `generalize` science.md Time=fs row as the unit SSOT referenced by later science specs.
- `new` — none.

## Files to create or modify

- `.claude/notes/architecture.md`
- `.claude/notes/architecture-rules.md`
- `.claude/notes/ffi.md`
- `.claude/notes/science.md`
- `.claude/notes/performance.md` (path fixes only: `molrs/src/**` not `molrs-ff/src/**`)
- `CLAUDE.md` (residual factual lies only)

## Tasks

- [x] Inventory live workspace members and `molrs/src` top-level modules against CLAUDE crate structure
- [x] Rewrite `.claude/notes/architecture.md` managed inventory (single-crate + binders)
- [x] Rewrite `.claude/notes/architecture-rules.md` ownership / import rules for the merged crate
- [x] Update `.claude/notes/ffi.md` active-binder list and path diagrams
- [x] Align `.claude/notes/science.md` + `.claude/notes/performance.md` with fs analysis unit and `molrs/src/**` paths
- [x] Patch CLAUDE.md factual lies (optimize gate, Gasteiger layer, SMARTS path)
- [x] Add regression example `regressions/release-0-12-01-harness-check.md` listing SSOT files and required unit table rows
- [x] Run full check + test suite (harness-only: `cargo test -p molcrafts-molrs --lib --features full,filesystem` must still pass unchanged)

## Testing strategy

Harness artifacts are verified by grep gates in the regression checklist:

- No remaining `molrs-core` / `molrs-io` / `molrs-ff` crate-as-package claims in architecture-rules.
- science.md Time row = fs.
- ffi.md mentions python, wasm, capi, cxxapi.

No production code change required for green suite.

## Out of scope

- VACF / MSD / dielectric code fixes (03–04)
- Legacy API deletion (02)
- cxxapi panic elimination (05)
- User-facing site-src rewrite (06)
- Closing unrelated open specs (`chem-perceive-15`, `net-streaming`, …)
