---
slug: release-0-12-04-science-dielectric-zarr
status: done
created: 2026-08-04
grilled: false
revised: 2026-08-04
depends_on:
  - release-0-12-01-harness
---

# release-0-12-04-science-dielectric-zarr — fs units, two-pass ε, SimBox f64

## Summary

Unify analysis time to **fs** for dielectric/conductivity SI paths, force per-axis static dielectric to two-pass centered variance, and persist Zarr SimBox geometry as f64 (with f32 read compat).

**Note (2026-08-04):** Zarr/MolRec IO is owned by another agent this sprint — **SimBox f64 deferred**. fs + two-pass dielectric/conductivity landed in this run.

## Domain basis

**Time = fs** project-wide (science.md). Replace all analysis docs/prefactors that assume picoseconds:

\[
1\,\mathrm{fs}=10^{-15}\,\mathrm{s}
\]

Conductivity SI factors that used `1e-12` for ps must use `1e-15`. Velocity for LAMMPS real is Å/fs (pass-through).

**Per-axis dielectric (Neumann tin-foil):**

\[
\mathrm{Var}_d=\frac1n\sum_t\bigl(M_d(t)-\langle M_d\rangle\bigr)^2
\]

two-pass only. Prefactors unchanged:

\[
\varepsilon_d=\varepsilon_\infty+\frac{4\pi\kappa}{Vk_BT}\,\mathrm{Var}_d
\]

**SimBox Zarr:** store `h` (3×3) and `origin` (3) as float64. Read path: promote legacy f32 → f64 once.

Refs: Neumann *Mol. Phys.* **50**, 841 (1983); LAMMPS units real; MDAnalysis dielectric docs.

## Design

- Update `compute/dielectric` rustdoc tables: time fs, current density e·Å⁻²·fs⁻¹.
- Update SI constants in dielectric + transport conductivity (Einstein / Green–Kubo) modules and Python bindings that restate units.
- Extract shared two-pass variance helper; use in isotropic (already) and components.
- `write_simbox` / `read_simbox` in `frame_io.rs`: f64; read accepts f32.
- Grep-gate: no `dt` docs claiming ps under `molrs/src/compute/`.

### Reuse decision

- `reuse` isotropic two-pass block in `static_dielectric_constant`.
- `reuse` `write_f64_array` pattern for SimBox.
- `generalize` components path to two-pass.
- `generalize` SI prefactors from ps → fs.
- Do **not** change isotropic Neumann prefactor algebra.

## Files to create or modify

- `molrs/src/compute/dielectric/mod.rs`
- `molrs/src/compute/transport/jacf.rs` (if unit docs)
- `molrs/src/compute/transport/einstein_conductivity.rs`
- `molrs/src/compute/transport/green_kubo_conductivity.rs`
- `molrs/src/io/store/zarr/frame_io.rs`
- related unit tests in those modules
- `molrs-python` dielectric/transport docs if they hard-code ps

## Tasks

- [x] Write failing tests: large-mean one-pass failure case for components; f64 SimBox round-trip ULP; SI prefactor dimensional check with fs
- [x] Generalize two-pass variance for `static_dielectric_constant_components`
- [x] Migrate dielectric + conductivity SI prefactors and rustdocs to fs
- [x] Persist SimBox as f64; read-compat f32 *(deferred — molrec IO agent)*
- [x] Grep-clean ps claims in dielectric/transport docs (core path)
- [x] Add regression `regressions/release-0-12-04-dielectric-zarr.md` with hard-coded goldens
- [x] Run full check + test suite

## Testing strategy

Synthetic dipoles with \(M_0=10^8\); components variance ≈ 1. SimBox `h[0,0]=123.456789012345` exact after round-trip. No third-party oracles.

## Out of scope

- VACF/MSD (03)
- molpy prefactor migration (molpy chain, must match this contract)
- Full site-src rewrite (06)
