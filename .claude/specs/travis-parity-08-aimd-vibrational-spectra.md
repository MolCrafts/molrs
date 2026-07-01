---
title: AIMD vibrational spectra — VCD, ROA, and resonance Raman
status: draft
created: 2026-06-26
---

# AIMD vibrational spectra — VCD, ROA, and resonance Raman

## Summary
Extend molrs's spectral suite from IR + Raman to the three chiral/advanced
vibrational spectra TRAVIS is known for: **VCD** (vibrational circular dichroism),
**ROA** (Raman optical activity), and **resonance Raman**. All are computed from
time-correlation functions of the molecular electromagnetic moments produced in
link 07, and all slot into molrs's existing `fit` pattern: a raw `Compute` emits an
auto/cross-correlation curve, a `Fit` windows + FFTs it into a spectrum
(reusing `fit::spectral::window_and_fft`, cm⁻¹ frequency grid, and the IR/Raman
prefactor conventions already in the crate).

molrs today ships IR (`IRFlux`→`IRSpectrum`) and Raman (`RamanTensor`→`RamanSpectrum`).
This link adds the magnetic-moment and field-derivative cross-correlations those
two do not cover.

Library feature only — **no CLI**, no bindings.

## Domain basis
- AIMD spectra from MD time-correlation functions (the unifying framework):
  Thomas, Brehm, Weber, Kirchner, *Phys. Chem. Chem. Phys.* **2013**, 15, 6608
  ("Computing vibrational spectra from ab initio molecular dynamics"); Brehm et al.,
  *J. Chem. Phys.* **2020**, 152, 164105.
- **VCD**: cross-correlation of the electric dipole derivative and the **magnetic**
  dipole moment — `Δε(ω) ∝ FT⟨ μ̇(0)·m(t) ⟩` (the rotational-strength analogue of
  the IR dipole ACF). Requires per-molecule magnetic moments `m` (from the current
  density / atomic-current decomposition).
- **ROA**: cross-correlation involving the electric-dipole–magnetic-dipole and
  electric-dipole–electric-quadrupole polarizability derivatives (the optical-
  activity tensors G′ and A) — the chiral analogue of the Raman polarizability ACF.
- **Resonance Raman**: the Raman polarizability ACF evaluated with the
  excitation-frequency-dependent (resonant) polarizability tensor supplied by the
  caller (molrs does the correlation + transform, not the excited-state response).
- **Units / conventions**: cm⁻¹ frequency axis, quantum/temperature prefactors and
  windowing identical to the existing `IRSpectrum`/`RamanSpectrum` so all spectra are
  mutually consistent.

## Design
Extend `compute/fit/` (the established raw-compute → spectral-fit home).

New raw `Compute` structs (in `fit/raw_computes.rs` alongside `IRFlux`/`RamanTensor`):
- `VcdCrossFlux` — emits the `⟨μ̇(0)·m(t)⟩` cross-correlation from per-molecule
  electric-dipole and magnetic-dipole time series (link 07 + a current-density-derived
  magnetic moment; the magnetic-moment computation reuses `dielectric::compute_current_density`/`decompose_current`).
- `RoaCrossTensor` — emits the ROA cross-correlations from polarizability + optical-
  activity tensor (G′/A) time series.
- `ResonanceRamanTensor` — like `RamanTensor` but consuming a resonant (frequency-
  dependent) polarizability series.

New `Fit` structs (in `fit/spectral.rs` alongside `IRSpectrum`/`RamanSpectrum`):
- `VcdSpectrum`, `RoaSpectrum`, `ResonanceRamanSpectrum` — window + FFT + the
  spectrum-specific prefactor, reusing `window_and_fft` / `forward_fft_onesided`
  and the cm⁻¹ frequency grid. Outputs reuse `SpectrumResult` /
  `RamanSpectrumResult` (extended with the chiral component where needed).

No new tessellation/IO; this link is correlation + transform over link-07 moments
plus the existing magnetic-moment/current machinery in `dielectric`.

Layer discipline: entirely within `compute` (feature `compute`; `voronoi` only for
the moment *inputs*, which are passed in as time series); reuses `signal`/`rustfft`;
WASM-clean.

## Files to create or modify
- `molrs/src/compute/fit/raw_computes.rs` (modify) — add `VcdCrossFlux`, `RoaCrossTensor`, `ResonanceRamanTensor`.
- `molrs/src/compute/fit/spectral.rs` (modify) — add `VcdSpectrum`, `RoaSpectrum`, `ResonanceRamanSpectrum`.
- `molrs/src/compute/fit/mod.rs` (modify) — re-exports + doc table rows.
- `molrs/src/compute/spectra.rs` (modify) — extend result types if a chiral component is needed.
- `molrs/tests/compute/spectra_chiral.rs` (new) — symmetry + consistency tests.

## Tasks
- [ ] Write failing VCD tests: an achiral (mirror-symmetric) synthetic system gives ~zero VCD; an enantiomer pair gives equal-and-opposite VCD spectra (the defining chiral signature).
- [ ] Write failing ROA test: enantiomers give sign-flipped ROA; an achiral system gives ~zero.
- [ ] Write failing consistency test: peak positions (cm⁻¹) of VCD/ROA coincide with the IR/Raman peaks of the same trajectory (same vibrational frequencies, different intensities/signs).
- [ ] Implement `VcdCrossFlux`/`RoaCrossTensor`/`ResonanceRamanTensor` raw computes (reusing `dielectric` current/magnetic machinery).
- [ ] Implement `VcdSpectrum`/`RoaSpectrum`/`ResonanceRamanSpectrum` fits reusing `window_and_fft` + cm⁻¹ grid + IR/Raman prefactor conventions.
- [ ] Rustdoc with the cross-correlation definitions + citations + the shared-prefactor note; run fmt/clippy/test.

## Testing strategy
- **Chiral sign law** (`scientific`): enantiomers ⇒ equal-and-opposite VCD/ROA;
  achiral ⇒ ~zero. This is the headline correctness property.
- **Peak-position consistency** (`scientific`): VCD/ROA/resonance-Raman peaks fall at
  the same cm⁻¹ frequencies as IR/Raman for the same trajectory (the moments differ,
  the normal-mode frequencies do not).
- **Prefactor/window parity**: identical windowing + frequency grid to the existing
  `IRSpectrum`/`RamanSpectrum` (shared-helper call verified).
- **Edge cases**: zero-length series → typed error; a single molecule → defined
  (no cross-molecule terms).

## Third-party library analysis
- **FFT / windowing**: reuse the existing `rustfft` dependency and
  `fit::spectral::window_and_fft` / `forward_fft_onesided`; no new crate.
- **Magnetic moment / current density**: reuse `compute::dielectric::compute_current_density`
  + `decompose_current`; no new crate.
- No third-party dependency is introduced — this link is composition over link-07
  moments + the existing spectral and dielectric machinery.
- *(Versions to confirm at implementation time.)*

## Out of scope
- Computing excited-state / resonant polarizabilities or magnetic response from first
  principles (caller supplies the field/current/tensor time series; molrs correlates +
  transforms).
- Normal-mode decomposition of the spectra (a possible later link).
- IR/Raman themselves (already shipped).
- CLI, bindings.
