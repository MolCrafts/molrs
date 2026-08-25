# Project blueprint — molrs

> Generated for 0.14.0. Refresh with `/mol:map` when the tree drifts.

**Inventory date:** 2026-08-25
**Layout:** single published crate `molcrafts-molrs` + binder workspaces.

## Workspace members

| Path | Role |
|---|---|
| `molrs/` | Core library (`molcrafts-molrs`) — all science |
| `molrs-ffi/` | Shared handle/store layer (own workspace; path dep) |
| `molrs-python/` | PyO3 wheel `molcrafts-molrs` |
| `molrs-wasm/` | wasm-bindgen `@molcrafts/molrs` |
| `molrs-capi/` | C ABI |
| `molrs-cxxapi/` | CXX bridge (Atomiverse) |

## Core module graph (`molrs/src`)

```
core (always) ──► perceive (always)
                ├── io (feature)
                ├── ff (feature) ──► optimize (with ff)
                │                 └── md (feature, needs ff)
                └── conformer (feature, needs ff)
compute (feature) ──► signal
stream / serialize (optional)
```

| Module | Public surface (summary) |
|---|---|
| `core` | Frame, Block, MolGraph, Atomistic, Box, schema, generate, units, `UnitPreset` |
| `perceive` | rings, aromaticity, SMARTS, stereo, hydrogens, bond types |
| `io` | readers/writers, SMILES, trajectory, Record store adapter (`io::store::zarr`) |
| `ff` | ForceField, potentials, typifiers (MMFF, OPLS, UFF, ATD), charge; pair styles include `coul/long/pme` |
| `md` | VelocityVerlet / Langevin, `LJCut` re-export, MaxwellBoltzmann(`kbt`) |
| `compute` | transport, MSD, RDF, dielectric, spectra, shape, … |
| `signal` | ACF FFT primitives |
| `conformer` | ETKDG-style 3D generation |
| `optimize` | LBFGS / potentials-driven minimize |

## Python package layout (`molrs-python/python/molrs`)

- Top level: core only (`Frame`, `Block`, `Atomistic`, …)
- Subpackages: `ff/`, `io/`, `compute/`, `conformer`, `perceive`, `md` (experimental), `optimize`, `signal`
- No flat free functions for SMILES/perceive; use `molrs.io.SmilesIR`, `molrs.perceive.Perceive`

## Layer rules (import direction)

- `core` must not import `io` / `ff` / `compute`
- `perceive` may use `core` only
- `io` / `ff` may use `core` + `perceive`
- `compute` may use `core` + `signal`
- Binders depend on `molcrafts-molrs` + `molrs-ffi`; never reverse

## Analysis units (SSOT)

See `.claude/notes/science.md`: Time = **fs**, Length = Å, Charge = e, Energy = kcal/mol.
Dielectric / conductivity SI paths must use fs (not ps).

## Out of inventory

Historical multi-crate names (`molrs-core`, `molrs-io`, …) are deleted. Do not link docs.rs packages with those names.
