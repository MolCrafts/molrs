# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

The `travis-parity-*` series (2026-06-26) closes feature gaps against the TRAVIS
trajectory analyzer (travis-analyzer.de). It is dependency-ordered (01 → 08) and
listed in that order for readability. All target the merged `molcrafts-molrs` crate
(`compute`, plus `io` and a new `voronoi` feature); none add a CLI; each carries a
third-party-library analysis.

| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-06-26 | travis-parity-01-geometric-distributions | draft | molcrafts-molrs | ADF / dihedral / distance distribution functions + reusable `Observable` extractors (foundation for CDF/SDF). |
| 2026-06-26 | travis-parity-02-combined-distribution-functions | draft | molcrafts-molrs | Joint 2-D/3-D histograms (CDF) correlating 2–3 observables; TRAVIS's most-used analysis. |
| 2026-06-26 | travis-parity-03-spatial-distribution-function | draft | molcrafts-molrs | Reference-molecule-frame 3-D density + solvent orientation via native (BLAS-free) Kabsch. |
| 2026-06-26 | travis-parity-04-van-hove-and-reorientation | draft | molcrafts-molrs | Van Hove G_s/G_d(r,t) + Legendre P1/P2 reorientational TCFs (bridges RDF↔MSD; NMR/IR reorientation). |
| 2026-06-26 | travis-parity-05-hydrogen-bond-network | draft | molcrafts-molrs | Geometric D–H···A detection + native-`Topology` network + continuous/intermittent lifetime TCFs. |
| 2026-06-26 | travis-parity-06-radical-voronoi | draft | molcrafts-molrs | 3-D periodic radical (Laguerre) Voronoi core + domain/void analysis; native pure-Rust (WASM-clean). |
| 2026-06-26 | travis-parity-07-voronoi-electron-integration | draft | molcrafts-molrs | Cube-trajectory IO + per-molecule charge/dipole/polarizability via Voronoi integration of electron density. |
| 2026-06-26 | travis-parity-08-aimd-vibrational-spectra | draft | molcrafts-molrs | VCD / ROA / resonance-Raman spectra from EM-moment cross-correlations (extends the IR/Raman `fit` suite). |

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
