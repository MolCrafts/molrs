---
slug: core-generate-saw-path
criteria:
  - id: ac-001
    summary: Same seed and config yields byte-identical paths
    type: code
    evaluator_hint: ""
    pass_when: |
      Calling SelfAvoidingWalk::generate() twice with an identical config
      (same n_chains, chain_length, bond_length, target_density, seed,
      strategy) produces WalkOutput.paths that are exactly equal coordinate
      by coordinate (no tolerance) for both FccLattice and OffLattice.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-002
    summary: Consecutive intra-chain points are exactly bond_length apart
    type: code
    evaluator_hint: ""
    pass_when: |
      For every chain and every consecutive point pair, the SimBox PBC
      distance equals bond_length within 1e-9 for OffLattice and exactly
      (within 1e-12) for FccLattice.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-003
    summary: OffLattice respects excluded volume under minimum image
    type: code
    evaluator_hint: ""
    pass_when: |
      For an OffLattice { excluded_radius } run, no two distinct placed
      monomers across all chains have a SimBox minimum-image distance less
      than excluded_radius - 1e-9.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-004
    summary: FccLattice places no two monomers on the same lattice site
    type: code
    evaluator_hint: ""
    pass_when: |
      For an FccLattice run, no two distinct placed monomers (across all
      chains, minimum image) coincide; all pairwise minimum-image distances
      are >= the FCC nearest-neighbor spacing minus 1e-9.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-005
    summary: Output has exactly n_chains chains of chain_length points
    type: code
    evaluator_hint: ""
    pass_when: |
      WalkOutput.paths.len() == n_chains and every inner Vec has length
      chain_length for both strategies on a successful generate().
    status: verified
    last_checked: "2026-06-07"
  - id: ac-006
    summary: Derived box volume matches n_total / target_density and contains all points
    type: code
    evaluator_hint: ""
    pass_when: |
      For OffLattice, WalkOutput.simbox.volume() equals (n_chains *
      chain_length) / target_density within 1e-6 relative tolerance. For
      FccLattice (lattice-commensurate box), volume is >= that requested
      value. All output points lie inside the box.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-007
    summary: Strategy is struct-injected with no free-function factories
    type: code
    evaluator_hint: ""
    pass_when: |
      Tests construct SelfAvoidingWalk via struct literal with strategy: S;
      FccLattice and OffLattice are structs implementing GrowthStrategy; and
      the module's production (non-#[cfg(test)]) surface exposes no `fn make_`
      or free function returning SelfAvoidingWalk — construction is struct
      literal only. (Test-only fixtures that build configs via struct literal
      are permitted.)
    status: verified
    last_checked: "2026-06-07"
  - id: ac-008
    summary: generate module is pure (no molrs-io, no Frame/Topology, paths+SimBox only)
    type: code
    evaluator_hint: ""
    pass_when: |
      molrs-core/src/generate/ has no reference to molrs_io, Frame, or
      Topology; WalkOutput contains only paths: Vec<Vec<F3>> and simbox:
      SimBox; the module performs no file IO.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-009
    summary: Invalid config and exhausted dead-end return WalkError
    type: code
    evaluator_hint: ""
    pass_when: |
      generate() returns Err(WalkError) for target_density <= 0,
      bond_length <= 0, or chain_length == 0; and returns
      Err(WalkError::DeadEnd { .. }) when retries/backtracks/restarts are
      exhausted, deterministically under a fixed seed.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-010
    summary: Public API documented with units and density convention
    type: docs
    evaluator_hint: ""
    pass_when: |
      SelfAvoidingWalk, GrowthStrategy, FccLattice, OffLattice, WalkOutput
      carry rustdoc per repo doc style; target_density doc explicitly states
      monomers-per-volume units and the a = (n_total/target_density).cbrt()
      box convention; cargo doc builds without warnings.
    status: pending
    last_checked: ""
  - id: ac-011
    summary: Full check and test suite pass
    type: runtime
    evaluator_hint: ""
    pass_when: |
      cargo fmt --all --check, cargo clippy -- -D warnings, and
      cargo test -p molcrafts-molrs-core all succeed.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-012
    summary: Per-axis boundary — periodic wraps, non-periodic reflects, output in-box
    type: code
    evaluator_hint: ""
    pass_when: |
      For BOTH FccLattice and OffLattice, with pbc=[true,true,true] and
      pbc=[false,false,false], every output coordinate lies in [0, edge) on
      every axis, and the consecutive-bond min-image distance equals
      bond_length within 1e-9 under both settings (periodic wrap and
      reflective step-flip both preserve the bond). FCC on reflective walls
      also keeps all pairwise separations >= bond_length.
    status: verified
    last_checked: "2026-06-07"
  - id: ac-013
    summary: Overlap is occupancy-based — no distance or neighbor-list in the overlap path
    type: code
    evaluator_hint: ""
    pass_when: |
      Self-avoidance is decided by OccupancyGrid cell occupancy
      (SameCell / BlockClear). The production overlap path in
      molrs-core/src/generate/ contains no NeighborQuery, no LinkCell, and no
      shortest_vector/calc_distance distance computation used to judge
      overlap. (Tests may still measure distance to verify the geometric
      property.)
    status: verified
    last_checked: "2026-06-07"
---

# Acceptance criteria

- **ac-001 Determinism** — reproducibility under `seed`, both strategies.
- **ac-002 Bond length** — fixed-bond invariant per consecutive pair.
- **ac-003 / ac-004 Self-avoidance** — excluded-radius (OffLattice) and lattice-occupancy (FccLattice), periodic.
- **ac-005 Shape** — exact chain count and length.
- **ac-006 Density/box** — `volume == n_total/target_density`; containment.
- **ac-007 No-factory** — struct literal + struct-injected strategy; no free-function factory.
- **ac-008 Purity** — no `molrs-io`, no Frame/Topology; return is paths + SimBox.
- **ac-009 Errors** — invalid config and exhausted dead-end yield `WalkError`.
- **ac-010 Docs** — rustdoc with units and density convention.
- **ac-011 Build** — fmt + clippy + test suite green.
