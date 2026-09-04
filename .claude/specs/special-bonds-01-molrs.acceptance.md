---
slug: special-bonds-01-molrs
created: 2026-09-04
criteria:
  - id: ac-001
    summary: from_frame takes n_atoms from atoms.nrows() without coordinates
    type: code
    pass_when: |
      Topology::from_frame(&frame) on a frame whose "atoms" block has
      nrows() == Some(12) from a uint "id" column and no x/y/z columns, and
      whose "bonds" block carries uint atomi/atomj for the 11 pairs (i, i+1),
      returns Ok with n_atoms() == 12 and n_bonds() == 11.
    status: verified
    last_checked: 2026-09-04
  - id: ac-002
    summary: adjacency follows bonds-block insertion order, never sorted
    type: code
    pass_when: |
      For a C12 chain frame whose bonds block lists (5,6) before (4,5),
      topo.neighbors(5) == vec![6, 4]; for the same frame in file order
      (4,5) before (5,6), topo.neighbors(5) == vec![4, 6]. A #[cfg(test)]
      test asserts both, and the rustdoc on Topology::from_frame states that
      sorting the neighbour slices would reshape a consumer's growth tree.
    status: verified
    last_checked: 2026-09-04
  - id: ac-003
    summary: frame problems are named MolRsError; (n,n) is Validation
    type: code
    pass_when: |
      from_frame returns Err(MolRsError::NotFound { entity: "atoms", .. })
      for a frame with no atoms block; Err(MolRsError::Validation { .. })
      whose message contains "atoms" and "nrows" for atoms inserted as
      Block::new() (nrows() == None); Err(MolRsError::Validation { .. })
      whose message contains both endpoint indices and the atom count for a
      bond naming an atom outside the frame, including a self-loop (n, n);
      Err(MolRsError::Validation { .. }) whose message names "atomi" and/or
      "atomj" when a present bonds block with nrows() == Some(n>0) lacks
      those uint columns (including a block that only has "i"/"j"); Ok with
      n_bonds() == 0 for a frame with no bonds block or an empty one; and an
      in-range self-loop (a, a) with a < n is dropped so n_bonds() counts
      only the real edges.
    status: verified
    last_checked: 2026-09-04
  - id: ac-004
    summary: BondDistanceWeights validates and carries the Cassandra 1-N tail
    type: code
    pass_when: |
      BondDistanceWeights::new(vec![]) and new(vec![0.0, 1.5]) and
      new(vec![0.0, f64::NAN]) each return Err(MolRsError::Validation { .. });
      from_exclusion_depth(3).as_slice() == [0.0, 0.0, 0.0, 1.0];
      weight(0) == 0.0, weight(1) == 0.0, weight(3) == 0.0, weight(4) == 1.0
      and weight(97) == 1.0 for that table; new(vec![0.0]) and
      new(vec![0.0, 0.0, 0.0]) succeed with weight(10) == 0.0; the type has
      no Default impl.
    status: verified
    last_checked: 2026-09-04
  - id: ac-005
    summary: exclusions are root-inclusive, ascending, and keyed by weight == 0
    type: code
    pass_when: |
      On the C12 chain with from_exclusion_depth(3):
      exclusions[0] == [0,1,2,3], exclusions[5] == [2,3,4,5,6,7,8],
      exclusions[11] == [8,9,10,11]; with from_exclusion_depth(1)
      exclusions[5] == [4,5,6]; with from_exclusion_depth(2)
      exclusions[5] == [3,4,5,6,7]; every list contains its own root and is
      sorted ascending. With new(vec![0.0,0.0,0.5,1.0]) the 1-4 partners
      are absent from every list (0.5 is not an exemption).
    status: verified
    last_checked: 2026-09-04
  - id: ac-006
    summary: exclusions follow weight(d)==0 including a zero tail and hole tables
    type: code
    pass_when: |
      On C12, new(vec![0.0]) puts 11 in exclusions[0]; new(vec![1.0, 0.0])
      puts 11 in exclusions[0] and does not put 1; new(vec![0.0, 0.5, 0.0, 1.0])
      puts the 1-4 partner in exclusions[0] and does not put the 1-3 partner.
      On a two-component graph, a zero-tail table lists only same-component
      partners. For every tested root r and partner p, p is in exclusions[r]
      iff distances(r)[p] >= 0 and weights.weight(distances(r)[p] as usize) == 0.0.
      molrs/src contains no `fn exclusions(&self, depth` entry point.
    status: verified
    last_checked: 2026-09-04
  - id: ac-007
    summary: core table is ff-free; SpecialBonds struct Default accessors frozen
    type: code
    pass_when: |
      molrs/src/core/system/bond_weights.rs and the from_frame / exclusions
      bodies in topology.rs contain no `use crate::ff` and no
      cfg(feature = "ff"); cargo check -p molcrafts-molrs
      --no-default-features compiles the new module; SpecialBonds still has
      pub lj: [f64; 3], pub coul: [f64; 3], Default { lj: [0.0, 0.0, 1.0],
      coul: [0.0, 0.0, 1.0] }, and unchanged lj_14 / coul_14 /
      ForceField::{special_bonds, set_special_bonds}; there is no From or
      Into between BondDistanceWeights and SpecialBonds; molrs::BondDistanceWeights
      and molrs::Topology both resolve at the crate root.
    status: verified
    last_checked: 2026-09-04
  - id: ac-008
    summary: the graph half of molpack's topology suite now lives here
    type: code
    pass_when: |
      molrs/src/core/system/topology.rs has #[cfg(test)] tests covering all
      of: linear / branched / ring frame reads, bonds-block neighbour order,
      the error and Ok-empty cases of ac-003, self-loop dropping,
      isolated-atom detection via n_components, root-inclusive sorted
      exclusions, the C12 depth literals, ring closure, branched-template
      exclusions, and the zero-tail / [1,0] / hole-table cases of ac-006.
      Each such test targets a single method.
    status: verified
    last_checked: 2026-09-04
  - id: ac-009
    summary: regression example reproduces the C12 goldens with no third party
    type: runtime
    pass_when: |
      regressions/special-bonds-01-molrs.md names the public-API scenario
      (Topology::from_frame -> exclusions(&BondDistanceWeights::
      from_exclusion_depth(3))), embeds the literals [0,1,2,3] /
      [2,3,4,5,6,7,8] / [8,9,10,11] as hard-coded goldens, and its stated
      gate `cargo test -p molcrafts-molrs --lib --features full,filesystem
      topology::tests::` passes; the file references no external tool at
      run time.
    status: verified
    last_checked: 2026-09-04
  - id: ac-010
    summary: rustdoc states units, tail, LAMMPS pitfall, and both relationships
    type: docs
    pass_when: |
      BondDistanceWeights, its four methods, Topology::from_frame and
      Topology::exclusions all carry rustdoc; the weight docs say the values
      are dimensionless in [0, 1], the index is an integer bond distance,
      the last entry is the 1-N tail, and a length-3 vector is not a LAMMPS
      special_bonds triple (charmm 0 0 0 is [0,0,0,1] here). Core rustdoc
      names molrs::ff::forcefield::SpecialBonds in code-spans with no
      crate::ff intra-doc links and states there is no From/Into.
      Topology::exclusions rustdoc states it does not read or write
      frame["exclusions"]. SpecialBonds rustdoc links to
      crate::BondDistanceWeights. docs/interop.md keeps the ForceField
      special_bonds bullet and adds a separate BondDistanceWeights bullet.
      RUSTDOCFLAGS='-D warnings' cargo doc --no-deps -p molcrafts-molrs
      --all-features emits no warning.
    status: verified
    last_checked: 2026-09-04
  - id: ac-011
    summary: full check and test suite pass, doctests included
    type: runtime
    pass_when: |
      `cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets
      --features full,filesystem -- -D warnings`, `cargo test -p molcrafts-molrs
      --lib --features full,filesystem` and `cargo test --doc -p molcrafts-molrs
      --features full,filesystem` all succeed on the branch tip.
    status: verified
    last_checked: 2026-09-04
  - id: ac-012
    summary: C12 hop counts match Cassandra depth-3 exemption (Boon 2017)
    type: scientific
    pass_when: |
      from_exclusion_depth(3) is bit-identical to [0.0, 0.0, 0.0, 1.0];
      weight(4) == weight(9) == 1.0; on the C12 chain the exempt set at
      each root equals the atoms whose bond-graph hop count is ≤ 3
      (hard-coded [0,1,2,3] / [2,3,4,5,6,7,8] / [8,9,10,11] at roots
      0 / 5 / 11); new([0.0, 0.0, 0.5, 1.0]) does not exempt 1-4 partners;
      new([0.0]) exempts the whole C12 component including atom 11 from
      root 0.
    status: verified
    last_checked: 2026-09-04
out_of_scope:
  - "Reshaping SpecialBonds struct / Default / accessors or any force-field reader"
  - "From/Into between BondDistanceWeights and SpecialBonds"
  - "Re-expressing intramolecular_pairs on Topology::exclusions"
  - "Exposing Topology or BondDistanceWeights in molrs-python"
  - "A frame geometry reader in core (frame_positions stays in molpack)"
  - "Version bump, tag, or the molpack CI ref bump (manual, molpack law P3)"
  - "WasmTopology::from_frame (still reads i/j; later must delegate to Topology::from_frame)"
  - "exclusions(&self, depth) entry point or a required private ball symbol"
---

# Acceptance — special-bonds-01-molrs

Done means molrs owns the bond graph and the Cassandra-tailed bond-distance
weight table as additive, `ff`-free `core` API. Exemption has one home:
`weight(distance) == 0.0`, including a zero tail and hole tables.
`SpecialBonds` keeps its struct, `Default` and accessors; only rustdoc may change.
