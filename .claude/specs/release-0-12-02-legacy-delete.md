---
slug: release-0-12-02-legacy-delete
status: done
created: 2026-08-04
grilled: false
revised: 2026-08-04
depends_on:
  - release-0-12-01-harness
---

# release-0-12-02-legacy-delete — delete all legacy public APIs

## Summary

Remove every dual-name, façade, and unused public symbol identified for the 0.12 break window. Callers use the single compose path already enforced for MMFF: typify → pairs → `ForceField::to_potentials` → `calc_energy_forces`. No deprecation shims.

## Domain basis

N/A (API surface hygiene). Product rule: experimental stage allows hard breaks; iron law forbids dual public names.

## Design

**Delete (definition sites):**

| Symbol / module | Replacement |
|---|---|
| `forcefield_method_json` + `ff/forcefield_meta.rs` | drop; no provenance JSON door |
| `ff::potential::kernels` (`PairLJ126`, `pair_lj126_ctor`) | `PairLJCut` / `pair_lj_cut_ctor` |
| `read_trajectory_store` | `read_record_store` / `read_frame_from_store` / `read_trajectory_file` |
| `OPLSAATypifier::build` (Rust + Python) | same compose path as MMFF |
| Python `PairLJ126Style`, `PairLJ126CoulCutStyle`, `PairLJ126CoulLongStyle` | `PairStyle` / style strings `lj/cut`, … |
| Python `Entity` / `Entities` (and any `Link` dual if present) compat aliases in `views.py` | `NodeRef` / `Refs` |
| Any remaining pyi free-fn ghosts (`parse_smiles`, free perceive, free gasteiger) | class APIs only |

**Tests:** migrate PME test off `PairLJ126`; rewrite `test_opls_typifier` build path to compose; add absence gates (MMFF-style) for deleted symbols.

### Reuse decision

- `reuse` MMFF orthogonal delete pattern (`mmff-orthogonal-02` / `test_mmff_public_surface.py`) for OPLS `build` removal + absence gate.
- `reuse` `PairLJCut` as sole LJ pair kernel.
- `reuse` `read_record_store` / `read_frame_from_store` as Zarr doors.
- `new` — none.

## Files to create or modify

- `molrs/src/ff/forcefield_meta.rs` (delete)
- `molrs/src/ff/mod.rs`
- `molrs/src/ff/potential/mod.rs`
- `molrs/src/ff/potential/kspace/pme.rs` (test only)
- `molrs/src/ff/typifier/opls/mod.rs`
- `molrs/src/io/store/zarr/mod.rs`
- `molrs/src/io/store/zarr/record_io.rs`
- `molrs-python/src/ff/mod.rs` (OPLS build pymethod)
- `molrs-python/python/molrs/ff/forcefield.py`
- `molrs-python/python/molrs/ff/__init__.py`
- `molrs-python/python/molrs/views.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_opls_typifier.py`
- `molrs-python/tests/test_mmff_public_surface.py` (extend pattern) or `tests/test_opls_public_surface.py` (new)

## Tasks

- [x] Write failing absence-gate tests for deleted OPLS `build`, `PairLJ126*`, `forcefield_method_json`, `read_trajectory_store`, Entity aliases
- [x] Delete `forcefield_meta` module and re-export
- [x] Delete `potential::kernels`; fix PME unit test to `PairLJCut`
- [x] Delete `read_trajectory_store` and re-exports
- [x] Delete `OPLSAATypifier::build` (Rust + PyO3 + pyi); rewrite OPLS tests to compose path
- [x] Delete Python `PairLJ126*` styles and `Entity`/`Entities` aliases; clean `__all__`
- [x] Sweep remaining pyi / doctest references to deleted symbols
- [x] Add regression example `regressions/release-0-12-02-legacy-delete.py` (compose-only OPLS path; no `build`)
- [x] Run full check + test suite (`cargo test` 1449 ok; maturin reinstall)

## Testing strategy

- Unit: absence gates (`not hasattr`, compile-fail via Python AttributeError, Rust tests compile without kernels).
- OPLS energy path: typify → frame → pairs → to_potentials → energy finite on ethanol hard-coded fixture.
- Regression under `regressions/` only for public API story.

## Out of scope

- Docs site-src narrative (06) — but fix tests/examples that would fail to compile
- Science kernel math (03–04)
- cxxapi panics (05)
- molpy dual APIs (separate molpy chain)
