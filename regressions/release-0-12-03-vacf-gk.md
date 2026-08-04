# VACF / Green–Kubo / MSD contracts

## VACF
- Unbiased \(C(\tau)\) with `1/(n-τ)` after DOF average.
- Batch VACF matches VACFAccumulator finalize.
- PowerSpectrum consumes the same unbiased curve.

## GreenKuboDiffusion
- Same curve as VACF; \(D = (1/d)\int C\,d\tau\) is the caller fit step.

## MSD
- Hard precondition: **unwrapped** positions; no MIC inside the kernel.

Verified by: `cargo test -p molcrafts-molrs --lib --features full,filesystem vacf`
and full suite.
