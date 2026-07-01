---
slug: travis-parity-08-aimd-vibrational-spectra
criteria:
  - id: ac-001
    summary: VCD obeys the enantiomer sign law
    type: scientific
    pass_when: |
      For a synthetic enantiomer pair, VcdSpectrum produces equal-and-opposite
      spectra (sign-flipped within tolerance), and an achiral / mirror-symmetric
      system produces a VCD spectrum near zero.
    status: pending
  - id: ac-002
    summary: ROA obeys the enantiomer sign law
    type: scientific
    pass_when: |
      For a synthetic enantiomer pair, RoaSpectrum produces sign-flipped spectra, and
      an achiral system produces ~zero ROA.
    status: pending
  - id: ac-003
    summary: peak positions coincide with IR/Raman
    type: scientific
    pass_when: |
      For the same trajectory, the cm⁻¹ peak positions of VCD/ROA/resonance-Raman
      coincide (within one frequency bin) with the IR/Raman peaks from the existing
      IRSpectrum/RamanSpectrum — same vibrational frequencies, different intensities.
    status: pending
  - id: ac-004
    summary: windowing + frequency grid reuse the existing spectral helpers
    type: code
    pass_when: |
      The new fits call fit::spectral::window_and_fft / forward_fft_onesided and emit
      the same cm⁻¹ frequency grid as IRSpectrum/RamanSpectrum (verified by identical
      grids on identical input length/dt).
    status: pending
  - id: ac-005
    summary: edge cases + full check green
    type: runtime
    pass_when: |
      A zero-length input series returns a typed ComputeError; a single-molecule input
      is well-defined (no cross-molecule terms); the module stays WASM-clean and
      cargo fmt --check + clippy -D warnings + cargo test --features compute pass.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** are the defining physical signatures of VCD and ROA — chirality
  must flip the sign between enantiomers and vanish for achiral systems. Without this
  the spectra are meaningless.
- **ac-003** ties the new spectra to the existing IR/Raman frequency axis (the normal
  modes are shared; only the intensities/signs differ).
- **ac-004** enforces reuse of the established spectral helpers so all spectra share
  one windowing/FFT/prefactor convention.
- **ac-005** covers degenerate inputs and the standard WASM-clean build gate.
