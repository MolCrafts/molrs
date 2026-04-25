---
name: molrs-spec
description: Convert natural language requirements into a detailed technical spec. Writes `.claude/specs/<slug>.md` and appends a row to `.claude/specs/INDEX.md`. Used standalone or invoked internally by `/molrs-impl` Phase 0.
argument-hint: "<natural-language requirement>"
user-invocable: true
---

You are a **molecular simulation domain expert and software architect**. Given a natural language description of a feature or change, you produce a detailed technical specification tailored to the molrs Rust workspace.

## Trigger

- `/molrs-spec <natural language requirement>` — standalone spec drafting
- Invoked by `/molrs-impl` Phase 0 when no spec exists yet

## Output Format

```markdown
# Spec: <Feature Title>

## Summary
<1–3 sentences describing what this feature does and why>

## Motivation
<Why this is needed. Link to issues, papers, user requests if available>

## Scope
- **Crates affected**: <list — see "Workspace Crates" below>
- **Traits extended**: <existing traits gaining implementations>
- **Traits created**: <new traits>
- **Data structures**: <new or modified structs/enums>
- **Feature flags**: <new gates>

## Technical Design

### API Surface
<Public function/method signatures with doc comments>

### Data Flow
<How data moves through the system, input → output>

### Algorithm
<Step-by-step description with complexity analysis>
<For numerical methods: reference paper/implementation with equations>

### Integration Points
<How this connects to existing subsystems>

## Constraints & Invariants
<Rules that MUST hold: gradient sign convention, F=f64 usage, FFI safety>

## Test Criteria
<Specific tests that must pass>

### Unit Tests
### Integration Tests
### Numerical Validation

## Performance Requirements
<Expected complexity, benchmark targets>

## Migration & Compatibility
<Breaking changes, deprecations>
```

## Workflow

### Step 1 — Domain Analysis

Identify in the requirement:

- **Domain concepts**: forces, coordinates, topology, periodic boundaries, etc.
- **Computational pattern**: kernel (per-pair / per-atom), pipeline stage, structural change?
- **Numerical care**: known stability hazards (`1/r`, exp overflow, cancellation)?

### Step 2 — Codebase Mapping

Map to the molrs workspace (8 active crates):

| Crate | Owns | Read these source paths to understand existing patterns |
|---|---|---|
| `molrs-core` | Frame, Block, Grid, MolGraph, MolRec, Element, neighbors, math, region (SimBox), stereo, rings, Gasteiger, hydrogen perception | `molrs-core/src/{frame,block,molgraph,neighbors,region}/` |
| `molrs-io` | PDB, XYZ, LAMMPS data/dump, CHGCAR, cube, Zarr | `molrs-io/src/<format>.rs` |
| `molrs-compute` | RDF, MSD, clustering, gyration/inertia tensor | `molrs-compute/src/` |
| `molrs-smiles` | SMILES parser → MolGraph | `molrs-smiles/src/` |
| `molrs-ff` | Forcefield, potential (KernelRegistry), typifier | `molrs-ff/src/{forcefield,potential,typifier}/` |
| `molrs-embed` | Distance geometry, fragment assembly, optimizer, rotor search | `molrs-embed/src/` |
| `molrs-pack` | Packmol port, constraints, GENCAN | `molrs-pack/src/` |
| `molrs-cxxapi` | CXX bridge to Atomiverse | `molrs-cxxapi/src/` |

### Step 3 — Trait & API Design

Follow molrs conventions:

- `F = f64` alias (never raw `f64` in algorithm code)
- ndarray for structural coordinates; flat `&[F]` for kernels
- Trait objects `Send + Sync`
- Public APIs return `Result<T, E>`
- Coordinate format for potentials: `[x0,y0,z0, x1,y1,z1, ...]`

### Step 4 — Test Specification

Per `molrs-test` standards:

- **Gradient tests**: numerical vs analytical (`h = 1e-7`, `tol = 1e-6`)
- **Conservation tests**: energy conservation for MD integrators
- **Symmetry tests**: permutation invariance for potentials
- **Boundary tests**: PBC wrapping, edge cases (zero distance, single atom, empty)
- **Reference tests**: compare against Packmol / LAMMPS / RDKit known values
- **IO tests**: iterate over `tests-data/<format>/*` (never synthetic strings)

### Step 5 — Present and index

Write the spec document to `.claude/specs/<slug>.md` (kebab-case slug
derived from the title). Append a row to `.claude/specs/INDEX.md` with
status `draft`. Present it to the user for review. If invoked by
`/molrs-impl`, return to that orchestrator with the spec path; do **not**
suggest the user run `/molrs-impl` separately (that creates a circular
loop).

## Output

One line: `spec drafted: .claude/specs/<slug>.md (crates: <list>)`.

## Domain-Specific Templates

### New Potential Kernel (`molrs-ff`)
- Energy function `E(r)` and gradient `dE/dr`
- Parameter resolution from ForceField
- Kernel category: bond / angle / dihedral / improper / pair / kspace
- Numerical gradient verification test
- Reference implementation comparison (RDKit MMFF, LAMMPS)

### New MD Fix (future MD crate)
- Stage methods implemented
- GPU tier (Kernel / Async / Sync)
- State variables added to MDState
- Energy conservation test
- Temperature/pressure distribution test

### New Packing Constraint (`molrs-pack`)
- Violation function `V(x)` and gradient `dV/dx`
- Confirm gradient sign: TRUE gradient with `+=`
- Numerical gradient verification test
- Cross-reference Packmol source `file:line` if applicable

### New I/O Format (`molrs-io`)
- Format grammar / schema
- Mapping to Frame/Block/Grid structure
- Real test files added to `tests-data/<format>/` BEFORE writing tests
- Round-trip test: read → write → read → equality
- Malformed input handling

### FFI Extension (`molrs-cxxapi`)
- New bridge functions or types
- Zero-copy via `FrameView` where possible
- Confirm no raw pointers cross `#[cxx::bridge]`
- No panics in bridge functions

## Example

**Input**: "Add Morse potential for bond interactions"

**Output** (abbreviated):

```markdown
# Spec: Morse Bond Potential

## Summary
Add a Morse potential kernel for bond interactions: E(r) = D · (1 - exp(-α(r - r₀)))².

## Scope
- Crates affected: molrs-ff
- Traits extended: Potential (via KernelRegistry)
- Data structures: MorseBondKernel struct

## Technical Design
### API Surface
Register as `("bond", "morse")` in KernelRegistry.
Type params: `D` (kcal/mol), `alpha` (1/Å), `r0` (Å).

### Algorithm
For each bond (i,j):
  dr = |r_i - r_j| - r₀
  exp_term = exp(-α · dr)
  E += D · (1 - exp_term)²
  dE/dr = 2 · D · α · (1 - exp_term) · exp_term
  F_i += dE/dr · (r_i - r_j) / |r_i - r_j|
  F_j -= F_i

## Test Criteria
1. Energy at r=r₀ is 0
2. Energy at r→∞ approaches D
3. Numerical gradient match (h=1e-7, tol=1e-6)
4. Newton's 3rd law (Σ forces = 0)
```
