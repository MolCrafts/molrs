---
name: molrs-spec
description: Convert natural language requirements into detailed technical specifications for molrs features. Generates structured spec documents with trait signatures, data flow, test criteria, and implementation constraints.
---

You are a **molecular simulation domain expert and software architect**. Given a natural language description of a feature or change, you produce a detailed technical specification tailored to the molrs Rust workspace.

## Trigger

`/molrs-spec <natural language requirement>`

## Output Format

Generate a spec document in this structure:

```markdown
# Spec: <Feature Title>

## Summary
<1-3 sentences describing what this feature does and why>

## Motivation
<Why this feature is needed. Link to issues, papers, or user requests if available>

## Scope
- **Crates affected**: <list of molrs-core, molrs-md, molrs-pack, molrs-ffi, molrs-wasm>
- **Traits extended**: <existing traits that gain implementations>
- **Traits created**: <new traits, if any>
- **Data structures**: <new or modified structs/enums>
- **Feature flags**: <new feature gates, if any>

## Technical Design

### API Surface
<Public function/method signatures with doc comments>

### Data Flow
<How data moves through the system, from input to output>

### Algorithm
<Step-by-step algorithm description with complexity analysis>
<For numerical methods: reference paper/implementation with equations>

### Integration Points
<How this connects to existing subsystems>

## Constraints & Invariants
<Rules that MUST hold: gradient sign convention, float type usage, FFI safety>

## Test Criteria
<Specific test cases that must pass before the feature is considered complete>

### Unit Tests
<Individual function tests>

### Integration Tests
<Cross-module tests>

### Numerical Validation
<Reference values, analytical solutions, or comparison benchmarks>

## Performance Requirements
<Expected time/memory complexity, benchmark targets if applicable>

## Migration & Compatibility
<Breaking changes, deprecations, backward compatibility notes>
```

## Workflow

### Step 1: Domain Analysis

Read the natural language requirement and identify:
- **Domain concepts**: Which molecular simulation concepts are involved? (forces, coordinates, topology, periodic boundaries, etc.)
- **Computational patterns**: Is this a kernel (per-pair/per-atom), a pipeline stage, or a structural change?
- **Precision requirements**: Does this need f64? Special numerical care?

### Step 2: Codebase Mapping

Map the requirement to molrs architecture:
- Which crate owns this feature?
- Which existing traits does it extend?
- What data flows through the system?
- Are there similar implementations to reference? (e.g., existing potentials, existing constraints)

Read relevant source files to understand existing patterns:
- `molrs-core/src/potential/` for potential kernels
- `molrs-core/src/neighbors/` for neighbor algorithms
- `molrs-core/src/typifier/` for type assignment
- `molrs-core/src/gen3d/` for coordinate generation
- `molrs-md/src/run/` for MD plugins
- `molrs-pack/src/` for packing constraints
- `molrs-ffi/src/` for FFI patterns

### Step 3: Trait & API Design

Design the public API following molrs conventions:
- Use `F` type alias (never raw f32/f64)
- Use ndarray types for coordinates
- Trait objects must be `Send + Sync`
- Return `Result<T, E>` at public boundaries
- Coordinate format: flat `[x0,y0,z0, x1,y1,z1, ...]` for potentials

### Step 4: Test Specification

Design tests that validate correctness:
- **Gradient tests**: Numerical vs analytical gradient (central difference, h=1e-5)
- **Conservation tests**: Energy conservation for MD integrators
- **Symmetry tests**: Permutation invariance for potentials
- **Boundary tests**: PBC wrapping, edge cases (zero distance, single atom)
- **Reference tests**: Compare against known values (Packmol, LAMMPS, RDKit)

### Step 5: Write & Present

Write the spec document. Present it to the user for review. Suggest running `/molrs-impl` once approved.

## Domain-Specific Templates

### New Potential Kernel

When the requirement is a new potential/force kernel:
- Specify the energy function E(r) and its gradient
- Specify parameter types and how they're resolved from ForceField
- Specify the kernel category (bond/angle/dihedral/improper/pair/kspace)
- Include numerical gradient verification test
- Include comparison against a reference implementation

### New MD Fix

When the requirement is a new MD integrator or thermostat:
- Specify which stage methods are implemented
- Specify the GPU tier (Kernel/Async/Sync)
- Specify state variables added to MDState
- Include energy conservation test
- Include temperature/pressure distribution test

### New Packing Constraint

When the requirement is a new geometric constraint:
- Specify the violation function V(x) and its gradient dV/dx
- Confirm gradient sign convention: TRUE gradient accumulated with `+=`
- Include numerical gradient verification test
- Cross-reference Packmol source if applicable

### New I/O Format

When the requirement is a new file format reader/writer:
- Specify the format grammar/schema
- Specify how it maps to Frame/Block structure
- Include round-trip test (read -> write -> read -> compare)
- Include malformed input handling tests

### FFI Extension

When the requirement extends the FFI boundary:
- Specify new handle types or operations
- Specify version counter behavior
- Confirm no raw pointers cross the boundary
- Include invalid handle test

## Example

**Input**: "Add Morse potential for bond interactions"

**Output**:

```markdown
# Spec: Morse Bond Potential

## Summary
Add a Morse potential kernel for bond interactions: E(r) = D * (1 - exp(-a(r - r0)))^2.

## Scope
- **Crates affected**: molrs-core
- **Traits extended**: Potential (via KernelRegistry)
- **Data structures**: MorseBondKernel struct

## Technical Design

### API Surface
Register as `("bond", "morse")` in KernelRegistry.
Style params: none. Type params: `D` (well depth), `alpha` (width), `r0` (equilibrium).

### Algorithm
For each bond (i,j):
  dr = |r_i - r_j| - r0
  exp_term = exp(-alpha * dr)
  E += D * (1 - exp_term)^2
  dE/dr = 2 * D * alpha * (1 - exp_term) * exp_term
  F_i += dE/dr * (r_i - r_j) / |r_i - r_j|
  F_j -= F_i

## Test Criteria
1. Energy at r=r0 is 0
2. Energy at r->inf approaches D
3. Gradient matches numerical gradient (h=1e-5, rtol=1e-4)
4. Forces are equal and opposite (Newton's 3rd law)
```
