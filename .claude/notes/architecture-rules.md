# Architecture Rules

Project standard for the molrs Rust workspace — module dependency rules,
ownership, trait conventions, FFI boundaries, naming. Applied by the
`mol:architect` agent and `/mol:review`. (The inventory blueprint lives in
`.claude/notes/architecture.md`.)

## Single crate (0.12+)

Published library: **`molcrafts-molrs`** (`molrs/`). Binders are separate
workspace roots / path deps, not multi-crate science packages.

```
molrs/src modules:
  core (always) ──► perceive (always)
                 ├── io (feature)
                 ├── ff (feature) ──► optimize (with ff)
                 │                 └─► md (feature → ff)
                 └── conformer (feature → ff)
  compute (feature) ──► signal
  stream / serialize (optional)

binders (depend on molcrafts-molrs + molrs-ffi):
  molrs-python · molrs-wasm · molrs-capi · molrs-cxxapi
```

### Module dependency rules (ENFORCED)

- `core` depends on no other molrs module.
- `perceive` may depend on `core` only.
- `io` / `ff` may depend on `core` + `perceive`.
- `compute` depends on `signal` (+ `core` for Frame access).
- `conformer` requires `ff`.
- `optimize` is behind `ff` (not always-on).
- `md` requires `ff`, and may depend on `core` + `ff` only. **`ff` must never
  name `md`** — a pair kernel tallies a virial and a bonded kernel takes an
  index table, and neither may reach up to the loop that runs it. Gated by
  `ff_names_no_md` beside `core_names_no_other_module`; a `md`-defined `Virial`
  leaked into `core` once already, which is why both gates exist.
- No cyclic module edges in library code.

### Binder rules

- All binders go through `molrs-ffi` handles where ownership crosses languages.
- No panics on fallible paths in `extern "C"` / CXX / wasm exports (see `ffi.md`).
- Python package layout: top-level = core; domain under `molrs.ff` / `molrs.io` / …

## Module ownership

| Module | Owns |
|---|---|
| `core` | Frame, Block, MolGraph, MolRec, Topology, Element, SimBox, neighbors, schema, generate, units |
| `perceive` | rings, aromaticity, SMARTS, stereo, hydrogens, bond types, equivalence |
| `io` | format readers/writers, SMILES, trajectory, Zarr/MolRec |
| `ff` | ForceField, potentials, typifiers, charge (Gasteiger/BCC), scale_lj |
| `signal` | FFT ACF, windows, frequency grids |
| `compute` | RDF, MSD, transport, dielectric, spectra, shape, cluster, … |
| `conformer` | distance geometry / ETKDG-style pipeline |
| `optimize` | LBFGS / potential-driven minimize |
| `md` | integrators, `ForceProvider` and the minimum-image / ghost régimes, the halo (`Comm`), bonded index lists, special-bonds weights, Maxwell-Boltzmann |

### Potential's three traits

`ff::potential` splits the capability, not the type:

- `Potential` — energy and forces from coordinates. Every kernel.
- `IndexedTerms: Potential` — the rows are named by an index table the caller
  may replace. Every bonded kernel; no pair kernel.
- `PairDriven: Potential` — the sum runs over whatever pairs a neighbour search
  turns up. Every pair kernel; nothing else (PME reads its own exclusions, so
  it is `Member::Plain` despite registering under the `pair` category).

`Member` is the three as one value, chosen by the kernel's **constructor** —
which is why `KernelConstructor` returns `Member` and not `Box<dyn Potential>`.
A `Box<dyn Potential>` cannot be asked which of the two it also is, and the
question used to be put to `terms()`, a method whose job is to return a table
and which allocated one per bonded member per step to answer it. Two
`holds_indices: Vec<bool>` fields existed to cache that answer.

**Rule**: a capability only some potentials have is its own trait, and the
variant is decided once, at construction. A default implementation that is
correct for half the implementors and silently wrong for the other half is the
shape this replaced.

## Trait design principles

1. **Object-safe**: no `Self` in return position, no generic methods on the trait.
2. **`Send + Sync` for shared trait objects**: required for rayon and cross-thread use.
3. **Owned returns at API boundary**.
4. **Open-ended dispatch via registration**, not enum match: `KernelRegistry`, etc.
5. **Coordinate format**: structural code uses ndarray (`F3`, `FNx3`).

## Naming

- Snake_case modules; PascalCase public types.
- Pair styles use LAMMPS names (`lj/cut`), not `LJ126` dual aliases.
- No dual public names for the same symbol (0.12: delete façades, not deprecate).

## Analysis units

SSOT: `.claude/notes/science.md` — Time = **fs**, Length = Å, Charge = e,
Energy = kcal/mol. Conductivity/dielectric SI paths must use fs (not ps).
