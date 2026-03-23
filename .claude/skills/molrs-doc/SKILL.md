---
name: molrs-doc
description: Documentation and docstring standards for molrs. Covers Rust doc comments, mathematical notation, module-level docs, examples, and API documentation for molecular simulation code.
---

You are a **scientific software documentation specialist** for the molrs Rust workspace.

## Trigger

Use when writing documentation, reviewing docstrings, or updating module-level docs.

## Docstring Tiers

### Tier 1: Public API (REQUIRED)

All `pub` items MUST have doc comments:

```rust
/// Evaluate energy and forces for the given coordinates.
///
/// Coordinates are a flat array: `[x0, y0, z0, x1, y1, z1, ...]` (3N elements).
///
/// Returns `(energy, forces)` where forces has the same layout as coordinates.
///
/// # Panics
///
/// Panics if `coords.len()` is not a multiple of 3.
fn eval(&self, coords: &[F]) -> (F, Vec<F>);
```

### Tier 2: Complex Algorithms (REQUIRED)

Any non-trivial algorithm MUST document:
- **What** it computes (equation if applicable)
- **How** it works (algorithm sketch)
- **Why** a particular approach was chosen (if non-obvious)
- **Reference** (paper, Packmol source location, etc.)

```rust
/// Compute the Lennard-Jones 12-6 potential and gradient.
///
/// Energy: `E(r) = 4e [(s/r)^12 - (s/r)^6]`
///
/// Reference: Allen & Tildesley, "Computer Simulation of Liquids", Eq. 1.2
```

### Tier 3: Internal Helpers (OPTIONAL)

Private functions that are short and self-explanatory may omit docstrings. Add them when:
- The function name doesn't fully convey its purpose
- The implementation uses a non-obvious trick
- The function has surprising edge case behavior

## Mathematical Notation

Use Unicode math in doc comments for readability. For complex equations, use inline code blocks:
```rust
/// Energy: `E = D * (1 - exp(-a(r - r0)))^2`
```

## Module-Level Documentation

Each module (`mod.rs` or file-level) should have a module doc comment:

```rust
//! # Potential Kernels
//!
//! This module provides the [`Potential`] trait and [`KernelRegistry`] for
//! energy/force evaluation in molecular simulations.
//!
//! ## Adding a New Potential
//!
//! 1. Implement the [`Potential`] trait
//! 2. Register in [`register_builtins`]
//! 3. Add tests (numerical gradient + Newton's 3rd law)
```

## Units Convention

Document units explicitly in docstrings:

| Quantity | Unit | Note |
|---|---|---|
| Distance | Angstroms | Unless otherwise stated |
| Energy | kcal/mol | MMFF convention |
| Force | kcal/(mol*A) | Energy/distance |
| Angle | Radians | Internal; degrees in I/O |
| Mass | amu | atomic mass units |
| Temperature | Kelvin | In MD |
| Time | femtoseconds | In MD |

```rust
/// Set the cutoff distance for pair interactions.
///
/// # Arguments
///
/// * `cutoff` -- Maximum interaction distance in Angstroms
pub fn set_cutoff(&mut self, cutoff: F) { ... }
```

## Documentation Checklist

- [ ] All `pub` items have `///` doc comments
- [ ] Algorithm functions include equations and references
- [ ] Module-level `//!` docs explain the module's purpose and architecture
- [ ] Crate-level `//!` docs list key types and subsystems
- [ ] `# Panics` section for functions that can panic
- [ ] `# Errors` section for functions returning `Result`
- [ ] Examples are included for key public APIs
- [ ] Units are documented (Angstroms, kcal/mol, etc.)
- [ ] Parameter names match their mathematical symbols where possible
