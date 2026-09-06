---
slug: amber-prmtop-complete-01-structure
criteria:
  - id: ac-001
    summary: exclude_14 is a registered Bool column in sorted position
    type: code
    pass_when: |
      molrs/src/core/store/schema/mod.rs contains a col! row with key
      "exclude_14", const_name "EXCLUDE_14", dtype DType::Bool, shape Scalar,
      unit "", placed between the "element" and "id" rows; consts::EXCLUDE_14
      exists and appears in the consts_agree_with_the_table list; "exclude_14"
      appears in the optional list of both the dihedrals and impropers
      BlockSpec; and `cargo test -p molcrafts-molrs --lib --features
      full,filesystem store::schema` passes with
      columns_sorted_unique_and_documented and block_column_references_resolve
      green.
    status: pending
  - id: ac-002
    summary: atoms block carries res_name, mol_id, tree and the GB pair
    type: code
    pass_when: |
      Reading the extended LiTFSI fixture through
      read_amber_prmtop_from_reader yields an "atoms" block whose res_name is
      15x"TF" then "LI", whose mol_id is 15x1 then 2 (UInt), and which carries
      String "tree" plus Float "gb_radius" and "gb_screen" of NATOM rows; and
      removing ATOMS_PER_MOLECULE, TREE_CHAIN_CLASSIFICATION, RADII or SCREEN
      from the fixture makes the corresponding column absent (get_* returns
      None) rather than zero-filled — mol_id is never derived from bonds.
    status: pending
  - id: ac-003
    summary: exclusions block matches the existing BlockSpec and PME consumer
    type: code
    pass_when: |
      A fixture carrying NUMBER_EXCLUDED_ATOMS and EXCLUDED_ATOMS_LIST yields a
      frame["exclusions"] block with UInt "atomi"/"atomj" columns, exactly
      (NNB minus the count of 0 placeholders) rows, atomi < atomj on every row,
      values equal to the file's 1-based numbers minus one, an empty
      schema-typed block when the list is all zeros, and an InvalidData error
      naming both counts when the flattened list length differs from NNB.
    status: pending
  - id: ac-004
    summary: exclude_14 comes from one builder and agrees across both blocks
    type: code
    pass_when: |
      On LITFSI_HEAD, frame["dihedrals"] has 27 rows and get_bool("exclude_14")
      is true at exactly row indices 12, 14 and 16 and false elsewhere; on the
      improper fixture, every row of frame["impropers"] has the same
      exclude_14 value as the dihedrals row with the same
      (atomi, atomj, atomk, atoml, type_id); and build_dihedral_block is the
      only function in prmtop.rs that writes the exclude_14 column.
    status: pending
  - id: ac-005
    summary: prmtop box reproduces the ortho and truncated-octahedron cells
    type: scientific
    pass_when: |
      IFBOX=1 with BOX_DIMENSIONS "90.0 30.0 30.0 30.0" gives a SimBox with
      lengths (30.0, 30.0, 30.0) and volume 27000.0 within 1e-9; IFBOX=2 with
      "109.4712190 30.0 30.0 30.0" gives the matrix from
      matrix_from_lengths_angles([30;3], [109.4712206;3]) whose first row is
      [30.0, -10.0, -10.0] within 1e-9 and whose volume is 20784.6096908265
      within 1e-6; IFBOX=0 or a missing BOX_DIMENSIONS leaves frame.simbox
      None; and charge[15] == 1.0 within 1e-12 with the 16-atom sum below 0.01,
      proving the 18.2223 literal was not replaced by sqrt(332.0637133).
    status: pending
  - id: ac-006
    summary: prmtop meta carries radius_set, oldbeta and the solvent pointers
    type: code
    pass_when: |
      frame.meta["radius_set"] is the verbatim RADIUS_SET line as a String,
      frame.meta["oldbeta"] is the f64 first BOX_DIMENSIONS value (90.0 for the
      IFBOX=1 fixture, 109.4712190 for the IFBOX=2 fixture), and
      frame.meta["solvent_iptres"], ["solvent_nspm"], ["solvent_nspsol"] are
      the three SOLVENT_POINTERS integers as i64; each key is absent when its
      section is absent.
    status: pending
  - id: ac-007
    summary: unsupported prmtop features are refused with a named message
    type: code
    pass_when: |
      Each of CMAP_COUNT>0, IFPERT>0, IFCAP>0, IPOL=1, a CTITLE or
      FORCE_FIELD_TYPE flag, a negative NONBONDED_PARM_INDEX entry, a
      LENNARD_JONES_CCOEF flag, and IFBOX=3 makes
      read_amber_prmtop_from_reader return Err with kind InvalidData whose
      message contains the feature subject and the phrase "are not supported";
      and JOIN_ARRAY or IROTAT of length != NATOM returns InvalidData while the
      right length produces no column.
    status: pending
  - id: ac-008
    summary: the read_prmtop facade and the duplicate a4 parser are gone
    type: code
    pass_when: |
      `rg "fn read_prmtop" molrs/src` returns no match, `rg "fn
      read_a4_names" molrs/src` returns no match, every 20a4 section in
      prmtop.rs is parsed via prmtop_tables::parse_a4_names, and both
      `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem
      -- -D warnings` and `cargo test --doc -p molcrafts-molrs --features
      full,filesystem` pass.
    status: pending
  - id: ac-009
    summary: rustdoc names the real column set on both reader and consumer
    type: docs
    pass_when: |
      molrs/src/ff/potential/kspace/pme.rs lines 10 and 857 say atomi/atomj
      (no occurrence of the "i", "j" spelling for the exclusions columns
      remains in that file), and the prmtop.rs module header documents the new
      atoms columns, states that tree/gb_radius/gb_screen are intentionally
      unregistered format-local columns with no cross-format consumer, and
      states that an inpcrd box takes precedence over the prmtop box.
    status: pending
  - id: ac-010
    summary: regression example reproduces the hard-coded prmtop goldens
    type: runtime
    pass_when: |
      `uv --directory molrs-python run python
      ../regressions/amber-prmtop-complete-01-structure.py` exits 0 and prints
      its ok line, having asserted against literals embedded in the script:
      16 atoms; res_name 15x"TF" then "LI"; mol_id 15x1 then 2; 27 dihedral
      rows with exclude_14 true at exactly {12, 14, 16}; impropers exclude_14
      equal to the matching dihedral rows; the expected exclusions row count
      and first row; simbox lengths (30.0, 30.0, 30.0); meta["oldbeta"] == 90.0;
      charge[15] == 1.0 within 1e-12. The script imports molrs only — no
      AmberTools, RDKit, OpenMM or other third-party scientific package is
      imported or subprocessed.
    status: pending
  - id: ac-011
    summary: full check and unit-test gate green
    type: code
    pass_when: |
      `cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets
      --features full,filesystem -- -D warnings`, `cargo test -p
      molcrafts-molrs --lib --features full,filesystem` and `cargo test --doc
      -p molcrafts-molrs --features full,filesystem` all pass, with the four
      pre-existing prmtop unit tests (litfsi_counts, litfsi_charge_and_li,
      bond_first_pair_zero_based, missing_pointers) still green.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-004** are the pair that makes the flag real: registration gives
  `exclude_14` a key-global dtype so no future writer can land it as UInt, and the
  parity assertion proves the two-homes mirror cannot desynchronize while one
  builder owns the column.
- **ac-002 / ac-003 / ac-006** are the completeness criteria — each names the exact
  absent-when behaviour, because a fabricated column is the failure mode this spec
  exists to prevent.
- **ac-005** is the only `scientific` criterion: box geometry is the one place where
  a wrong constant produces a plausible-looking but incorrect cell, so it carries
  hard numbers (`−10.0` exactly, `20784.6096908265`) rather than a shape check. The
  charge assertion rides here to pin the `18.2223` literal against "correction".
- **ac-007** is graded on message text, not just error kind: a refusal that does not
  say what is unsupported is not a refusal a user can act on.
- **ac-008 / ac-009** discharge the architect's Iron-law findings 1 and 4; ac-009 is
  `docs` rather than `code` because nothing executes, but it is binding — this spec
  is what makes the `exclusions` block appear in a real Frame.
- **ac-010** is verified by `/mol:impl` at delivery. It is `runtime`, not
  `scientific`: the script lives in this repo's `regressions/` and reproduces
  literals derived from the format spec, not values from an external bench.
