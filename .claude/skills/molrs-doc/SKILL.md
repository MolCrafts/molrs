---
name: molrs-doc
description: Documentation standards for molrs — rustdoc tiers, mathematical notation, units, references. Reference document only; no procedural workflow.
---

Reference standard for molrs documentation. The `molrs-documenter` agent applies these rules; this file defines them.

## Docstring Tiers

### Tier 1 — Public API (REQUIRED)

Every `pub` item carries a `///` doc comment.

```rust
/// Evaluate energy and forces for the given coordinates.
///
/// `coords` is a flat array `[x0, y0, z0, x1, y1, z1, ...]` (3N elements).
/// Returns `(energy, forces)` with forces in the same layout.
///
/// # Panics
///
/// Panics if `coords.len()` is not a multiple of 3.
fn eval(&self, coords: &[F]) -> (F, Vec<F>);
```

Sections:

- `# Arguments` — non-obvious parameters
- `# Returns` — non-obvious returns
- `# Panics` — when the function can panic
- `# Errors` — for `Result`-returning functions
- `# Safety` — REQUIRED for `unsafe fn`
- `# Examples` — encouraged for key APIs

### Tier 2 — Complex Algorithms (REQUIRED)

Non-trivial algorithms document **what** (equation), **how** (algorithm sketch), **why** (if non-obvious), and **reference** (paper, Packmol source `file:line`, RDKit method).

```rust
/// Lennard-Jones 12-6 pair potential.
///
/// `E(r) = 4ε [(σ/r)¹² - (σ/r)⁶]`
///
/// Reference: Allen & Tildesley, *Computer Simulation of Liquids*, Eq. 1.2.
```

### Tier 3 — Internal Helpers (OPTIONAL)

Private functions may omit docstrings when the name fully conveys purpose. Add when:

- The implementation uses a non-obvious trick
- Edge-case behavior would surprise a reader
- The function is more than ~30 lines

## Mathematical Notation

Unicode in inline code blocks for readability:

```rust
/// Energy: `E = D · (1 - exp(-α(r - r₀)))²`
```

ASCII fallback acceptable: `E = D * (1 - exp(-a * (r - r0)))^2`.

## Module-Level Docs

Every `mod.rs` and crate `lib.rs` carries `//!` documentation:

```rust
//! # Potential Kernels
//!
//! Provides the [`Potential`] trait and [`KernelRegistry`] for
//! energy/force evaluation in molecular simulations.
//!
//! ## Adding a Kernel
//!
//! 1. Implement [`Potential`]
//! 2. Register in [`register_builtins`]
//! 3. Add tests (numerical gradient + Newton's 3rd law)
```

## Units Convention

molrs uses real units. Document units explicitly in every numeric API:

| Quantity | Unit | Note |
|---|---|---|
| Distance | Å | unless stated otherwise |
| Energy | kcal/mol | MMFF convention |
| Force | kcal/(mol·Å) | energy / distance |
| Angle | radians | internal; degrees in I/O |
| Mass | amu | atomic mass units |
| Temperature | K | MD |
| Time | fs | MD |
| Charge | e | elementary charge units |

```rust
/// Construct a kernel with the given cutoff (Å).
pub fn with_cutoff(self, cutoff: F) -> Self { /* ... */ }
```

Note: prefer immutable builder-style (`with_*` returning `Self`) over `&mut self` mutators — see workspace coding-style rules.

## References

For published methods, cite DOI / arXiv / Packmol source line:

```rust
/// ETKDG distance geometry.
///
/// Reference: Riniker & Landrum (2015), J. Chem. Inf. Model. 55, 2562–2574.
/// DOI: 10.1021/acs.jcim.5b00654
```

## Compliance Checklist

- [ ] All `pub` items have `///` docs
- [ ] Algorithm functions include equations + references
- [ ] Module-level `//!` explains purpose and architecture
- [ ] Crate-level `//!` lists key types and subsystems
- [ ] `# Panics` for fallible-by-panic functions
- [ ] `# Errors` for `Result` returners
- [ ] `# Safety` for `unsafe fn`
- [ ] `# Examples` for key public APIs
- [ ] Units documented for every numeric quantity
- [ ] Parameter names match mathematical symbols where possible
