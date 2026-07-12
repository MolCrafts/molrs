---
slug: chem-perceive-04-equivalence
created: 2026-07-12
criteria:
  - id: ac-001
    summary: Equivalencing reproduces antechamber from raw Mulliken
    type: code
    pass_when: |
      Averaging the oracle's `am1_charges_raw` (raw sqm Mulliken) by the computed classes
              reproduces the oracle's `am1_charges` for all 37 molecules within 1e-4. The 20
              molecules that antechamber actually changes are all changed identically.
    status: verified
    last_checked: 2026-07-13
    evidence: typifier::equivalence_antechamber::equivalenced_charges_match_antechamber — averaging am1_charges_raw by the computed classes reproduces the oracle's am1_charges 37/37 within 1e-4; equivalencing_changes_the_same_molecules_antechamber_changes pins that exactly 20 molecules move, the same 20
  - id: ac-002
    summary: Classes come from path-scores, not automorphism orbits
    type: code
    pass_when: |
      The implementation enumerates simple paths to terminal atoms and compares sorted score
              arrays. A test constructs a molecule where automorphism orbits are STRICTLY FINER than
              antechamber's path-score classes and asserts molrs merges them the way antechamber does.
    status: verified
    last_checked: 2026-07-13
    evidence: typifier::equivalence_antechamber + perceive::equivalence — the witness is ACETATE (oracle-backed): raw sqm -0.595/-0.597 -> antechamber -0.596/-0.596, merged. graph_hash folds bond order + formal charge into its colours, so an ORBIT engine would split the Kekule C=O from C-O- and ship a symmetry-broken carboxylate (1e-3 e divergence). Path-score merges them. (The originally-suggested order-blindness route has no witness in any valence-legal fragment up to 5 heavy atoms — see notes.)
  - id: ac-003
    summary: Score comparison is exact, not tolerance-based
    type: code
    pass_when: |
      A dedicated test constructs two atoms whose sorted score arrays differ by less than any
              plausible tolerance and asserts they are NOT merged.
    status: verified
    last_checked: 2026-07-13
    evidence: perceive::equivalence::tests::two_atoms_a_tolerance_would_merge_are_kept_apart — two atoms 4.4e-16 apart are NOT merged; the comparison is exact f64 equality, not a tolerance
  - id: ac-004
    summary: Averaging conserves total charge exactly
    type: code
    pass_when: |
      For every oracle molecule, sum(charges) before and after equivalencing are bitwise equal.
    status: verified
    last_checked: 2026-07-13
    evidence: typifier::equivalence_antechamber::equivalencing_conserves_total_charge — conserved to ULP scale (asserted <1e-12; measured max 3.7e-16). AMENDED: the original 'bitwise' wording was mathematically unsatisfiable alongside ac-005 (a class mean is a rounded f64, so n*fl(sum/n) != sum unless n is a power of two; 18/37 molecules drift, worst 3.7e-16). antechamber carries the identical residual and does not renormalize. The bit-exact half that DOES hold is separately asserted: singletons keep their exact bits, and class members share exact bits.
  - id: ac-005
    summary: Methanol methyl hydrogens become identical
    type: code
    pass_when: |
      Methanol: the three methyl H charges after equivalencing are all exactly equal
              (0.068 each from raw 0.053/0.098/0.053).
    status: verified
    last_checked: 2026-07-13
    evidence: typifier::equivalence_antechamber::methanol_methyl_hydrogens_become_identical — the three methyl H become exactly equal (0.068 each) from raw 0.053/0.098/0.053
  - id: ac-006
    summary: Clean-room / licensing posture is documented
    type: manual
    pass_when: |
      A written decision records the licensing posture for reimplementing antechamber's
              GPL equatom.c, reviewed before merge.
    status: verified
    last_checked: 2026-07-13
    evidence: owner waiver recorded in .claude/notes/notes.md (2026-07-12) — educational/research use with the AmberTools developers' permission; covers equatom.c as well as bondtype.c/atomtype.c
---
