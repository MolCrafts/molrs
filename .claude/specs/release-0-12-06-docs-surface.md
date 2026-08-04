---
slug: release-0-12-06-docs-surface
status: done
created: 2026-08-04
grilled: false
revised: 2026-08-04
depends_on:
  - release-0-12-02-legacy-delete
  - release-0-12-03-science-vacf-msd
  - release-0-12-04-science-dielectric-zarr
  - release-0-12-05-cxxapi-panic-free
---

# release-0-12-06-docs-surface — rewrite public docs for 0.12

## Summary

Rewrite all user-facing molrs documentation and examples so they teach only the landed 0.12 surface: single crate `molcrafts-molrs`, Python package layout (`molrs.io.SmilesIR`, `molrs.ff.*`, `molrs.conformer`, `calc_energy_forces`), version pin `0.12`, and correct science contracts (unbiased VACF, fs time, unwrapped MSD).

## Domain basis

Science contracts from specs 03–04 (fs, VACF, MSD unwrap). No new physics.

## Design

**Must rewrite / fix:**

- `molrs-python/README.md` — kill `parse_smiles` free fn, `generate_3d`, `EmbedOptions`, typifier `.build()`, top-level I/O
- `molrs-python/site-src/**` — index, quickstarts, installation, smiles, embed-3d, force-field, transport, reference/python, reference/rust
- Root `README.md`, `docs/interop.md`, crate `lib.rs` version pins → `0.12`
- `molrs-python/examples/*` imports → subpackages
- Transport guide: unbiased VACF + \(D=\frac1d\int C\); time fs
- Force-field: `calc_energy_forces` / `(N,3)` forces; no `.eval`
- Rust reference: single crate only

### Reuse decision

- `reuse` actual registered Python API from `molrs-python/src/lib.rs` and `python/molrs/__init__.py`
- `reuse` force-field compose path from MMFF tests
- `new` — none

## Files to create or modify

- `README.md`
- `docs/interop.md`
- `molrs/src/lib.rs` (doctest pin)
- `molrs-python/README.md`
- `molrs-python/site-src/index.md`
- `molrs-python/site-src/getting-started/*.md`
- `molrs-python/site-src/guides/*.md`
- `molrs-python/site-src/reference/*.md`
- `molrs-python/examples/*.py`
- `molrs-python/python/molrs/compute/dielectric.py` (docstring paths)

## Tasks

- [x] Rewrite molrs-python README + examples to 0.12 surface
- [x] Rewrite site-src getting-started + index (SmilesIR, conformer, pins)
- [x] Rewrite force-field + smiles + embed-3d guides
- [x] Rewrite transport guide for VACF/fs/MSD contracts
- [x] Rewrite reference/python and reference/rust for single crate + live symbols
- [x] Bump version pins in root README, interop.md, lib.rs doctest
- [x] Grep-gate: no `parse_smiles(`, `generate_3d`, `potentials.eval`, multi-crate docs.rs links, version 0.0.15/0.10/0.11 in published docs
- [x] Add regression `regressions/release-0-12-06-docs-surface.md` (checklist of grep gates)
- [x] Run full check + test suite

## Testing strategy

Docs verified by grep gates + manual run of rewritten example snippets if tests exist. No autodoc of deleted free functions.

## Out of scope

- molpy docs (separate chain)
- Closing unrelated open specs
- Changing public API beyond what 02–05 already landed
