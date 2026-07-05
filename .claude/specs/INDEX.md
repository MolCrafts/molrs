# molrs — Spec Index

One row per spec produced by `/mol:spec`. Newest on top.

The `analysis-parity-*` series (2026-06-26) closes feature gaps against the reference implementation
trajectory analyzer (the upstream analyzer site). It is dependency-ordered (01 → 08) and
listed in that order for readability. All target the merged `molcrafts-molrs` crate
(`compute`, plus `io` and a new `voronoi` feature); none add a CLI; each carries a
third-party-library analysis.

| Date | Slug | Status | Owner crate(s) | Summary |
|---|---|---|---|---|
| 2026-07-05 | region-support-01-graph-hash | draft | molcrafts-molrs, molrs-python | Isomorphism-invariant structural graph hash (WL/Morgan) + canonical node order + `is_isomorphic` on `MolGraph` (AA+CG), exposed to Python. The dedup key for molpy incremental-typification (`AffectedRegion` hashes by it → identical polymer junctions retype once). Greenfield; reuses topo/neighbor kernels. See molpy notes/incremental-typification-design.md. |
| 2026-07-05 | region-support-02-reaction-touched | draft | molcrafts-molrs, molrs-python | `Reaction.apply` returns the touched atom handles (bond-forming endpoints + added atoms + deleted-atom surviving neighbors + prop-set atoms) so molpy can extract the retype-safe region. Small, on reaction-smarts-02. Return type None→list[int]. |
| 2026-07-05 | reaction-smarts-01-python-matcher | in-flight | molcrafts-molrs, molrs-python | Expose the existing atom-map-aware SMARTS engine (`molrs::SmartsPattern`, `core/chem/smarts/`, already parses `[C:1]`) to Python as `SmartsPattern` with map-keyed matches; add `PyAtomistic` graph-edit conveniences (`remove_atom`/`remove_bond`/`set_bond_order`/`copy`). Pure binding, no algorithm change. Consumed by molpy `crosslink-*`. |
| 2026-07-05 | reaction-smarts-02-smirks-applier | in-flight | molcrafts-molrs, molrs-python | Daylight reaction-SMARTS (SMIRKS) engine: parse `LHS>>RHS`, compile the atom-map diff to a `Transform` (Daylight SMIRKS semantics: pairwise maps preserved, unmapped LHS deleted, unmapped RHS added, bond diff→form/break/order), apply to one occurrence via existing core edits + `generate_topology`. Expose `Reaction` to Python. Greenfield, on top of 01. Permissive reaction SMARTS (no strict mode). Depends on 01. |
| 2026-06-26 | analysis-parity-01-geometric-distributions | draft | molcrafts-molrs | ADF / dihedral / distance distribution functions + reusable `Observable` extractors (foundation for CDF/SDF). |
| 2026-06-26 | analysis-parity-02-combined-distribution-functions | draft | molcrafts-molrs | Joint 2-D/3-D histograms (CDF) correlating 2–3 observables; the reference implementation's most-used analysis. |
| 2026-06-26 | analysis-parity-03-spatial-distribution-function | draft | molcrafts-molrs | Reference-molecule-frame 3-D density + solvent orientation via native (BLAS-free) Kabsch. |
| 2026-06-26 | analysis-parity-04-van-hove-and-reorientation | draft | molcrafts-molrs | Van Hove G_s/G_d(r,t) + Legendre P1/P2 reorientational TCFs (bridges RDF↔MSD; NMR/IR reorientation). |
| 2026-06-26 | analysis-parity-05-hydrogen-bond-network | draft | molcrafts-molrs | Geometric D–H···A detection + native-`Topology` network + continuous/intermittent lifetime TCFs. |
| 2026-06-26 | analysis-parity-06-radical-voronoi | draft | molcrafts-molrs | 3-D periodic radical (Laguerre) Voronoi core + domain/void analysis; native pure-Rust (WASM-clean). |
| 2026-06-26 | analysis-parity-07-voronoi-electron-integration | draft | molcrafts-molrs | Cube-trajectory IO + per-molecule charge/dipole/polarizability via Voronoi integration of electron density. |
| 2026-06-26 | analysis-parity-08-aimd-vibrational-spectra | draft | molcrafts-molrs | VCD / ROA / resonance-Raman spectra from EM-moment cross-correlations (extends the IR/Raman `fit` suite). |

<!--
Status values:
  draft      — spec written, not yet implemented
  in-flight  — /mol:impl started against this spec
  shipped    — merged to master
  superseded — replaced by a later spec (link it in Summary)
-->
