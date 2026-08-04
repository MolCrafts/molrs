---
slug: release-0-12-03-science-vacf-msd
status: done
created: 2026-08-04
grilled: false
revised: 2026-08-04
depends_on:
  - release-0-12-01-harness
---

# release-0-12-03-science-vacf-msd — unbiased VACF + MSD unwrap contract

## Summary

Fix the Green–Kubo diffusion velocity-ACF path so `VACF` / `GreenKuboDiffusion` return an unbiased time-origin average, and hard-document (with tests) that MSD / Einstein diffusion require unwrapped coordinates. One VACF curve serves both D-integral and VDOS consumers.

## Domain basis

**Unbiased particle-averaged vector VACF:**

\[
C(\tau)=\frac{1}{N\,(n-\tau)}\sum_{i=1}^{N}\sum_{t=0}^{n-\tau-1}
\delta\mathbf{v}_i(t)\cdot\delta\mathbf{v}_i(t+\tau)
\]

Default mean removal for offline arrays: **per-DOF trajectory mean** (document clearly). Prefer per-frame COM velocity when streaming if already available — do not silently change callers who pass pre-centered series.

**Green–Kubo diffusion** (same curve; no second algorithm):

\[
D=\frac{1}{d}\int_0^{\tau_{\max}} C(\tau)\,\mathrm{d}\tau \quad (d=3)
\]

via `CumulativeTrapezoid` + scale `1/d` in the **caller** / fitting layer, not inside VACF.

**MSD** (window mode math already correct) requires continuous unwrapped Cartesian positions. Do **not** apply MIC inside the MSD kernel. Document hard precondition; optional diagnostic when box present and consecutive jumps exceed \(L/2\).

Refs: Green (1954) DOI 10.1063/1.1740082; Kubo (1957) DOI 10.1143/JPSJ.12.570; freud MSD unwrapped positions; molrs `GreenKuboConductivity` already uses `/(n−τ)`.

## Design

- Generalize `velocity_acf` in `vacf.rs` to divide by `(n−τ)` after DOF average (or fold DOF into N). Align `VACFAccumulator` finalize bit-for-bit with batch.
- Update rustdoc on `VACF`, `GreenKuboDiffusion`, Python bindings: delete “unnormalized lag-sum” claims; state \(D=\frac1d\int C\).
- Fix unit tests that currently assert lag-sum equality with VDOS raw sum — replace with unbiased reference.
- MSD/Einstein: rustdoc + Python docs precondition; add wrapped-vs-unwrapped regression test that documents failure mode; optional `check_jumps` diagnostic (if cheap).
- PowerSpectrum / VDOS consumers of VACF must use the same unbiased \(C(\tau)\).

### Reuse decision

- `reuse` `signal::acf_fft` as unnormalized primitive (caller owns norm).
- `reuse` `dynamics/acf.rs` unbiased `/(N·(T−t))` as pattern.
- `reuse` `GreenKuboConductivity` lag normalization discipline.
- `generalize` `velocity_acf` + `VACFAccumulator` to unbiased estimator.
- `new` — none for MSD kernel math; docs + tests only (+ optional jump check).

## Files to create or modify

- `molrs/src/compute/transport/vacf.rs`
- `molrs/src/compute/transport/vacf_accumulator.rs` (if present) or accumulator module
- `molrs/src/compute/transport/green_kubo_diffusion.rs`
- `molrs/src/compute/transport/mod.rs` (module docs)
- `molrs/src/compute/msd/mod.rs`
- `molrs/src/compute/msd/accumulator.rs` (docs if needed)
- `molrs-python/src/compute/*` related docs / bindings if exposed
- unit tests colocated in the above modules

## Tasks

- [x] Write failing unit tests: white-noise \(C(0)\), direct O(n²) ref with `/(n−τ)`, COM/mean cases, accumulator == batch
- [x] Implement unbiased `velocity_acf` + matching accumulator finalize
- [x] Update GreenKuboDiffusion / transport module rustdoc for \(D=\frac1d\int C\)
- [x] Write MSD wrapped-vs-unwrapped regression + unwrapped precondition rustdoc (docs done; wrapped regression still thin)
- [x] Fix VDOS/VACF tests that assumed lag-sum identity
- [x] Add regression example `regressions/release-0-12-03-vacf-gk.md` with hard-coded synthetic goldens
- [x] Run full check + test suite

## Testing strategy

Hard-coded synthetic series only (no freud/MDAnalysis at test time). Relative FFT vs direct \(10^{-10}\). Einstein path uses unwrapped constant-velocity particle: \(\mathrm{MSD}(\tau)=(v\tau)^2\).

## Out of scope

- Dielectric SI units / two-pass (04)
- Zarr SimBox (04)
- Site-src guide rewrite (06) beyond rustdoc strings required for correctness
- molpy JACF charge weighting (molpy chain)
