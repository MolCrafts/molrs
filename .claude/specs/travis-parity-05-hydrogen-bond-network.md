---
title: Hydrogen-bond network detection + dynamics
status: draft
created: 2026-06-26
---

# Hydrogen-bond network detection + dynamics

## Summary
Add **hydrogen-bond analysis** — geometric detection plus network topology plus
lifetime dynamics. molrs has no H-bond surface today, a hard gap for water,
protic-solvent, and ionic-liquid work. Three pieces:

1. **Detection** — per frame, identify donor–H···acceptor (D–H···A) bonds from a
   geometric criterion (D···A or H···A distance cutoff + D–H···A angle cutoff) over
   donor/acceptor atom selections, using the existing neighbor search for the
   candidate pairing.
2. **Network** — assemble the per-frame H-bond graph and report connected
   components / aggregate sizes via molrs's native `core` `Topology` connectivity
   (connected-components / BFS), **not** petgraph (petgraph is gated behind the
   `smiles` feature and must not leak into `compute`).
3. **Dynamics** — H-bond **lifetimes** via the continuous and intermittent
   hydrogen-bond time-correlation functions, reusing the existing
   `compute::persist::pair_survival_tcf` (`SurvivalMethod`) machinery that molrs
   already ships for pair survival.

Library feature only — **no CLI**, no bindings.

## Domain basis
- Geometric criterion: Luzar & Chandler, *Nature* **1996**, 379, 55 and *Phys. Rev.
  Lett.* **1996**, 76, 928 (the continuous/intermittent H-bond TCF definitions);
  common defaults r(O···O) ≤ 3.5 Å and ∠(O–H···O) ≥ 150° (≤ 30° from linear) —
  parameterized, with these as documented defaults.
- TRAVIS H-bond network/aggregation topology: Brehm et al., *J. Chem. Phys.* **2020**,
  152, 164105.
- **Lifetime TCFs**: continuous `S_HB(t) = ⟨h(0)h(t)·Θ_continuous⟩/⟨h⟩`,
  intermittent `C_HB(t) = ⟨h(0)h(t)⟩/⟨h⟩`, where `h` is the binary bond indicator;
  the intermittent integral gives the H-bond lifetime τ_HB.

## Design
New submodule `compute/hbond/` (`compute` feature).

New symbols:
- `HBondCriterion { dist_cutoff, dist_kind: DonorAcceptor|HydrogenAcceptor, angle_cutoff }`
  — frozen geometric parameters (Luzar–Chandler defaults).
- `HBonds` — implements `Compute`; args: donor (D,H) groups + acceptor (A) groups +
  criterion. Per frame returns the list of satisfied `(donor, hydrogen, acceptor)`
  triples (with D···A distance + angle), found via the existing `NeighborQuery`
  candidate search + min-image geometry. `HBondsResult` = per-frame bond lists +
  per-frame counts.
- `HBondNetwork` — builds the per-frame undirected molecule graph from the bond
  triples and returns connected-component sizes / counts using `core::Topology`
  connectivity (native; no petgraph).
- **Lifetimes** — a thin adapter mapping each ordered (D,A) bond to a binary
  presence series and feeding `compute::persist::pair_survival_tcf` with
  `SurvivalMethod::{Continuous, Intermittent}`, yielding `S_HB(t)` / `C_HB(t)` +
  τ_HB — no new TCF code.

Layer discipline: `compute` → `core` (`Topology`, `NeighborQuery`); explicitly
**no** petgraph (keep it confined to `io::smiles`); WASM-clean.

## Files to create or modify
- `molrs/src/compute/hbond/mod.rs` (new) — re-exports.
- `molrs/src/compute/hbond/criterion.rs` (new) — `HBondCriterion` + defaults.
- `molrs/src/compute/hbond/detect.rs` (new) — `HBonds` detection over `NeighborQuery`.
- `molrs/src/compute/hbond/network.rs` (new) — `HBondNetwork` via `core::Topology`.
- `molrs/src/compute/hbond/lifetime.rs` (new) — adapter to `persist::pair_survival_tcf`.
- `molrs/src/compute/mod.rs` (modify) — re-exports.
- `molrs/tests/compute/hbond.rs` (new) — geometry + network + lifetime tests.

## Tasks
- [ ] Write failing detection tests: a constructed water dimer at a known geometry is detected as one H-bond; widening the angle past the cutoff drops it; PBC-separated pair is detected via min-image.
- [ ] Write failing network test: a constructed chain of N H-bonded molecules forms one component of size N; breaking the middle bond splits it into two.
- [ ] Write failing lifetime test: a synthetic on/off bond series yields the analytically expected continuous vs. intermittent TCF and τ_HB ordering (τ_intermittent ≥ τ_continuous).
- [ ] Implement `HBondCriterion` + `HBonds` detection over `NeighborQuery` with min-image angle/distance.
- [ ] Implement `HBondNetwork` via `core::Topology` connected components (no petgraph).
- [ ] Implement the lifetime adapter over `persist::pair_survival_tcf`; rustdoc with criteria + TCF citations; run fmt/clippy/test.

## Testing strategy
- **Geometry** (`scientific`): known water-dimer geometry detected; angle/distance
  cutoff boundaries behave exactly; min-image under PBC.
- **Network**: chain/ring/cluster connectivity matches hand-computed component sizes;
  no petgraph symbol appears in the `compute` dependency (AST/grep guard).
- **Dynamics**: synthetic indicator series reproduces continuous vs. intermittent
  TCFs and τ ordering.
- **Edge cases**: no donors/acceptors → empty; self-bond excluded; symmetric
  double-counting avoided.

## Third-party library analysis
- **Graph/connectivity**: evaluated `petgraph` (already a dep, but **only under the
  `smiles` feature** for the SMARTS VF2 matcher). **Recommend molrs's native
  `core::Topology`** (connected components / BFS already implemented there) — keeps
  `compute` free of petgraph and avoids forcing the `smiles` feature on H-bond users.
- **H-bond detection crate**: none exists in Rust (no MDAnalysis/freud equivalent) —
  implement natively over the existing `NeighborQuery`.
- **Lifetime TCF**: reuse the shipped `compute::persist::pair_survival_tcf`; no new
  crate.
- *(Versions to confirm at implementation time.)*

## Out of scope
- Electronic/charge-transfer H-bond definitions (geometric criterion only).
- Hydrogen-bond free-energy / reactive-flux rate constant k(t) (a possible later
  link building on the intermittent TCF).
- Sankey/aggregate visual diagrams (visualization layer).
- CLI, bindings.
