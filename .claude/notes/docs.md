# Documentation Standards

Project standard for molrs documentation. Applied by the `mol:documenter`
agent and `/mol:docs`. Two parts: (A) rustdoc rules for Rust source, (B) the
docs-system (Zensical site) rules. The full docs-system spec lives in
`docs/doc-system-spec.md` — that file is authoritative for invariants; this
page summarizes and adds prose-style rules not covered there.

## Part A — Rustdoc Rules

### Docstring Tiers

#### Tier 1 — Public API (REQUIRED)

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

#### Tier 2 — Complex Algorithms (REQUIRED)

Non-trivial algorithms document **what** (equation), **how** (algorithm
sketch), **why** (if non-obvious), and **reference** (paper, Packmol source
`file:line`, RDKit method).

```rust
/// Lennard-Jones 12-6 pair potential.
///
/// `E(r) = 4ε [(σ/r)¹² - (σ/r)⁶]`
///
/// Reference: Allen & Tildesley, *Computer Simulation of Liquids*, Eq. 1.2.
```

#### Tier 3 — Internal Helpers (OPTIONAL)

Private functions may omit docstrings when the name fully conveys purpose.
Add when:

- The implementation uses a non-obvious trick
- Edge-case behavior would surprise a reader
- The function is more than ~30 lines

### Mathematical Notation

Unicode in inline code blocks for readability:

```rust
/// Energy: `E = D · (1 - exp(-α(r - r₀)))²`
```

ASCII fallback acceptable: `E = D * (1 - exp(-a * (r - r0)))^2`.

### Module-Level Docs

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

### Units Convention

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

Note: prefer immutable builder-style (`with_*` returning `Self`) over
`&mut self` mutators — see workspace coding-style rules.

### References

For published methods, cite DOI / arXiv / Packmol source line:

```rust
/// ETKDG distance geometry.
///
/// Reference: Riniker & Landrum (2015), J. Chem. Inf. Model. 55, 2562–2574.
/// DOI: 10.1021/acs.jcim.5b00654
```

Never invent references. If the source of an algorithm is unknown, say so
explicitly rather than guessing a citation.

### Rustdoc Compliance Checklist

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

## Part B — Docs System (Zensical site)

Authoritative spec: `docs/doc-system-spec.md` (read it in full before changing
the docs system; cite section numbers when a change lands against an
invariant).

### Owned surfaces

| Surface | Notes |
|---|---|
| `molrs-python/zensical.toml`, `molrs-python/site-src/**` (getting-started, guides, reference, contributing) | Zensical site content and config (no changelog page — history is git log / tags) |
| `docs/interop.md`, `docs/attribution.md` | Consumer interop + attributions (not the Zensical tree) |
| `.github/workflows/docs.yml` | Build + deploy to GitHub Pages |
| `README.md`, `molrs-python/README.md`, `molrs-wasm/README.md` | Root + binding READMEs |
| `molrs-python/python/molrs/_lib.pyi` | Hand-maintained Python type stubs |

Rust `///` comments in `molrs-*/src/**.rs` are the rustdoc axis (Part A above)
— do not mix the two in one change.

### Key invariants (summary — full list in spec §Constraints & Invariants)

- Single source of truth = Rust `///`. Never duplicate prose between `///` and
  `docs/**.md`; always inject with `::: module.Symbol`.
- `_lib.pyi` is hand-maintained and committed; every export in
  `molrs-python/src/lib.rs`'s `#[pymodule]` must appear in the stub.
- `pkg/molrs.d.ts` and `site/` are CI-generated and `.gitignore`d — never
  commit them.
- Inactive crates (`molrs-ffi`, `molrs-capi`) must not surface on the site.
- When a binding exists in both Python and WASM, the two reference pages must
  cross-link.
- `docs.yml` flags must match `publish-molrs-python.yml` /
  `publish-molrs-wasm.yml` byte-for-byte where the spec requires it.

Verification: `.pyi` change → freshness-guard check (spec §Test Criteria
Integration Test #1); content/`zensical.toml` change → `zensical build
--strict` locally, otherwise defer to CI explicitly.

### Prose Style (tutorials and conceptual docs)

API docstrings follow Rust rustdoc style. Tutorials, guides, and conceptual
pages use textbook prose — not bullet-heavy AI-generated lists.

**Structure.** Every section moves through concept → motivation → mechanics.
The heading names the concept, not the phase. Write "Neighbor Lists and
Cutoffs", not "What Are Neighbor Lists / Why We Need Them / How They Work".

**Prefer prose over lists.** A paragraph explaining why two things interact
(e.g. how `SimBox` and `LinkCell` share the cutoff value) is better than three
bullets that name the parts. Use lists only for genuinely enumerable items:
CLI flags, crate names, sequential setup steps.

**Motivation before mechanics.** A reader who understands why `F = f64` always
(not `f32`) can reconstruct the precision contract. A reader who only knows
the alias cannot.

**Complete the thought.** A section that says "molrs uses kcal/mol" without
explaining when that breaks interop with other codebases is incomplete. Every
paragraph must leave the reader with a usable mental model.

**No filler.** Cut: "it is worth noting that", "in order to", "as mentioned
above".
