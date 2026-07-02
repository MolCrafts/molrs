---
title: Van Hove correlation + Legendre reorientational dynamics
status: draft
created: 2026-06-26
---

# Van Hove correlation + Legendre reorientational dynamics

## Summary
Add two space-/time-correlation analyses that molrs lacks:

1. **Van Hove correlation function** `G(r, t)` — the self part `G_s(r, t)`
   (probability a tagged particle has moved a distance r in time t) and the distinct
   part `G_d(r, t)` (density of *other* particles at distance r at lag t). `G(r,0)`
   reduces to a delta (self) plus `g(r)` (distinct), tying it to the existing RDF;
   the time-resolved decay is the natural bridge between RDF (structure) and MSD
   (dynamics), neither of which molrs currently connects.
2. **Legendre reorientational TCFs** `C_ℓ(t) = ⟨P_ℓ(û(0)·û(t))⟩` for ℓ = 1, 2 of a
   selected molecular **vector** û (e.g. an O–H bond or dipole axis). These are the
   NMR/IR reorientation observables. molrs's existing `order::RotationalAutocorrelation`
   is freud's *quaternion rigid-body* ACF — a different quantity; this adds the
   molecular-vector Legendre form that spectroscopy and NMR relaxation use.

Library feature only — **no CLI**, no bindings.

## Domain basis
- Van Hove: L. van Hove, *Phys. Rev.* **1954**, 95, 249; Hansen & McDonald,
  *Theory of Simple Liquids* (4th ed.), §7. Definitions (number density N, volume V):
  `G_s(r,t) = (1/N) Σ_i ⟨δ(r − |r_i(t) − r_i(0)|)⟩`,
  `G_d(r,t) = (1/N) Σ_i Σ_{j≠i} ⟨δ(r − |r_j(t) − r_i(0)|)⟩`, with `G_d(r,0) = ρ g(r)`.
- Legendre reorientation: Berne & Pecora, *Dynamic Light Scattering*; standard NMR
  relaxation theory — `C_1 = ⟨cosθ(t)⟩`, `C_2 = ⟨(3cos²θ(t) − 1)/2⟩`, where
  `cosθ(t) = û(0)·û(t)`. The `C_2` decay gives the rotational correlation time τ_c.
- Both reuse molrs unit conventions (Å, fs); time lags are frame-index × dt.

## Design
- **Van Hove** — new `compute/van_hove.rs` (`compute` feature). `VanHove` implements
  `Compute` over a trajectory: for a set of lag times, bin self-displacements
  (`G_s`) and cross-displacements (`G_d`) into r-histograms reusing the link-01/RDF
  `histogram1d` + min-image conventions. `VanHoveResult` holds the `(n_lags × n_rbins)`
  arrays + r-edges + lag times. `G_d` reuses the existing neighbor/RDF pair machinery
  at each lag origin; multiple time origins are averaged.
- **Legendre reorientation** — new `compute/order/reorientation_legendre.rs`. A
  `LegendreReorientation` `Compute` takes a per-frame **vector** observable (a
  thin reuse of the link-01 selection: a unit vector per molecule from an atom pair
  or a supplied direction) and produces `C_1(t)` and `C_2(t)` via multi-origin
  averaging. It is deliberately separate from the quaternion `RotationalAutocorrelation`
  (documented cross-reference clarifying the two are different observables).
- The raw `C_ℓ(t)` curves are `Fit`-ready: an exponential/τ_c fit can reuse
  `fit::DebyeFit` (already ships a normalized-Φ(t) → τ fit) — so reorientation times
  fall out of existing infrastructure, not new fitting code.

Layer discipline: `compute` → (`signal` for any FFT-based origin averaging) → `core`;
WASM-clean.

## Files to create or modify
- `molrs/src/compute/van_hove.rs` (new) — `VanHove` + `VanHoveResult`.
- `molrs/src/compute/order/reorientation_legendre.rs` (new) — `LegendreReorientation`.
- `molrs/src/compute/order/mod.rs` (modify) — re-export.
- `molrs/src/compute/mod.rs` (modify) — re-exports.
- `molrs/tests/compute/van_hove.rs`, `molrs/tests/compute/reorientation.rs` (new).

## Tasks
- [ ] Write failing Van Hove t=0 test: `G_d(r,0)` equals ρ·g(r) from the existing RDF (within binning tolerance) and `G_s(r,0)` is a spike in the first bin.
- [ ] Write failing Van Hove dynamics test: free-particle / known-diffusion trajectory gives a Gaussian `G_s(r,t)` whose width grows as √(2·d·D·t) (matches the MSD slope).
- [ ] Write failing Legendre test: a vector rotating at constant rate gives `C_1(t)=cos(ωt)`, `C_2(t)=(3cos²(ωt)−1)/2`; a static vector gives `C_ℓ ≡ 1`.
- [ ] Implement `VanHove` (self + distinct, multi-origin) reusing RDF histogram + neighbor machinery.
- [ ] Implement `LegendreReorientation` (P1/P2, multi-origin) + DebyeFit-based τ_c example.
- [ ] Rustdoc with cited definitions + the explicit contrast vs. `RotationalAutocorrelation`; run fmt/clippy/test.

## Testing strategy
- **RDF consistency** (`scientific`): `G_d(r,0) = ρ g(r)` cross-checked against
  `compute::rdf` on the same frame.
- **MSD consistency** (`scientific`): `∫ r² G_s(r,t) dr` equals the MSD from
  `compute::msd` at each lag within tolerance.
- **Analytic reorientation**: constant-rate rotation reproduces `C_1`/`C_2` to 1e-9.
- **Multi-origin**: results invariant (within noise) to the number of time origins.
- **Edge cases**: single frame → only t=0; zero-length vector → typed error.

## Third-party library analysis
- **No new crate.** Van Hove reuses the existing RDF/`histogram1d` + neighbor-list
  code; Legendre `P_ℓ` are closed-form polynomials of `cosθ`; multi-origin averaging
  reuses `signal`'s FFT autocorrelation where a windowed-FFT origin average helps,
  else a direct double sum. `fit::DebyeFit` supplies τ_c — no new fitting dependency.
- *(Versions to confirm at implementation time.)*

## Out of scope
- Self-intermediate scattering function `F_s(k,t)` / dynamic structure factor
  `S(k,ω)` (a possible later link; note it as a natural FFT extension of Van Hove).
- NMR relaxation-rate (T1/T2) prediction from `C_2` (downstream of τ_c).
- Reorientation of full rigid-body quaternions (already `RotationalAutocorrelation`).
- CLI, bindings.
