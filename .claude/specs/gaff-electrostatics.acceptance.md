---
slug: gaff-electrostatics
created: 2026-07-15
criteria:
  - id: ac-001
    summary: The GAFF force field declares a Coulomb style
    type: code
    pass_when: |
      `gaff_forcefield(...)` declares `pair/coul/cut` with `coulomb`, `dielectric` and `delta = 0`
      (AMBER's Coulomb is unbuffered). `end_to_end::the_force_field_the_chain_builds_declares_its_
      electrostatics` goes green.
      No new kernel: `mmff-ele-compose` already generalized `pair/coul/cut` to `E = k*qi*qj/(D*(r+delta))`.
      AMBER is just `delta = 0`. What was missing was never a kernel — it was the force field SAYING SO.
    status: pending
    last_checked:
    evidence:

  - id: ac-002
    summary: AMBER's Coulomb constant is MEASURED, not copied from a doc
    type: code
    pass_when: |
      The constant is determined empirically (sander/antechamber single-point on a known molecule, backed
      out) and the derivation is recorded in the code.
      This is not pedantry. AMBER's parm files use 18.2223^2 = 332.0522..., CODATA is 332.06371 — DIFFERENT
      numbers. `mmff-ele-compose` proved a 2.4e-5 relative difference is worth 0.0036 kcal/mol on a
      -150 kcal/mol electrostatic term, ABOVE the parity tolerance. Guessing here reproduces the exact
      class of defect this chain exists to kill.
    status: pending
    last_checked:
    evidence:

  - id: ac-003
    summary: An EXTERNAL energy oracle for the GAFF chain — the real gap
    type: runtime
    pass_when: |
      The chain Perceive -> AtdTypifier -> BccModel -> ForceField -> Potentials -> energy is asserted
      against an oracle molrs did not produce (sander single-point, or a documented subset with the reason
      stated IN THE TEST).
      THIS IS THE ACTUAL DEFECT. Every stage of this chain had a test. The composition never did — nothing
      had ever run the GAFF chain to an energy, which is exactly how a missing Coulomb term survived 37/37
      green oracle checks. On this chain, "no external oracle" has now let a defect live three times: BCC
      bond-type perception, charge equivalencing, and the generic MMFF path's missing electrostatics.
      Asserting what the code computed is how you dig the next hole.
    status: pending
    last_checked:
    evidence:

  - id: ac-004
    summary: The IONS are asserted specifically
    type: runtime
    pass_when: |
      acetate (net -1, sum|q| = 2.85 e), methylammonium (+1) and imidazolium (+1) each get a NON-ZERO
      electrostatic energy, asserted against the oracle.
      A neutral-only test would be the `["e_ethane"]` mistake for the third time: pick an input that cannot
      fail, then report coverage. Dropping the Coulomb term costs a neutral molecule far less than it costs
      an ion — the ions are where this defect screams, and they are exactly what a lazy fixture list omits.
    status: pending
    last_checked:
    evidence:

  - id: ac-005
    summary: special_bonds.coul[2] is actually CONSUMED
    type: runtime
    pass_when: |
      The 1-4 Coulomb scale (`AMBER_COUL_14 = 1/1.2`, AMBER's SCEE) demonstrably CHANGES the energy of a
      molecule carrying a 1-4 pair.
      It was declared and consumed by NOTHING — the tell that was sitting in the tree the whole time:
      `coul: [0.0, 0.0, AMBER_COUL_14]` is a 1-4 scale factor for a Coulomb term that does not exist.
      A constant nothing consumes is the same smell as 4,065 XML rows nothing read.
    status: pending
    last_checked:
    evidence:
---

# Acceptance criteria

这是 `chem-perceive-15-final-acceptance` 的整体验收抓到的第一个缺陷，**也是这条链一直在打的同一个洞，只是换了个力场**。

- **ac-003 是真正的缺陷**。不是"少了一个 style"——是**这条链从来没有被跑到过能量**。每一段都有测试，组合没有。这正是 150 kcal/mol 那个洞当初活下来的方式。
- **ac-004** 防的是同一个反模式的第三次重演：**选一个不可能失败的输入，然后声称覆盖了。**
- **ac-005** 把那个"没人消费的常数"变成"必须被消费"。
