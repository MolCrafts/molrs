---
slug: release-0-14-14-pair-kernel-merge
created: 2026-08-25
criteria:
  - id: ac-001
    summary: exactly one LJ pair kernel type exists
    type: code
    pass_when: |
      molrs/src/md/lj_cut.rs does not exist; a single LJCut type lives in
      molrs/src/ff/potential/pair/lj_cut.rs and is re-exported as molrs::md::
      LJCut; no PairLJCut type survives anywhere under molrs/src.
    status: pending
  - id: ac-002
    summary: the md call path moved no bits
    type: scientific
    pass_when: |
      A #[cfg(test)] test asserts the merged kernel's energy and force arrays
      over a fixed box, coordinate set and skin are bit-identical
      (assert_eq!, no tolerance) to the pre-merge md::LJCut values recorded in
      the test.
    status: pending
  - id: ac-003
    summary: the ff call path moves at most one ulp, and it is explained
    type: scientific
    pass_when: |
      For a fixed pairs block and coordinates, merged-kernel energies and forces
      equal the pre-merge PairLJCut values bitwise wherever the two arithmetic
      forms agree; every disagreement is <= 1 ulp and is reproduced in the test
      body by evaluating both x/r2 and x*(1/r2). Any larger difference, or any
      difference in the cutoff mask, 1-4 scaling or Lorentz-Berthelot mixing,
      fails.
    status: pending
  - id: ac-004
    summary: no pairs snapshot and no MIC inside any potential
    type: code
    pass_when: |
      ripgrep finds zero occurrences of set_pairs across molrs/src,
      molrs-python/src and molrs-python/python (outside .claude/notes history);
      no file under molrs/src/ff/potential or molrs/src/md calls mic( or
      minimum_image; LJCut holds no stored pairs or SimBox field.
    status: pending
  - id: ac-005
    summary: one per-step pair dataset, shared by every pair potential
    type: runtime
    pass_when: |
      With two pair potentials in one Potentials driven by a VerletSkin, both
      receive the same &Neighbors for a step and the skin materializes the
      dataset exactly once per step (materialization counter == 1).
    status: pending
  - id: ac-006
    summary: the pair source is fixed at construction
    type: runtime
    pass_when: |
      A compiled-source LJ kernel evaluated inside a skin-driven Potentials
      returns exactly the energy it returns standalone over its compiled pairs,
      proving it did not also consume the per-step dataset; a loop-source kernel
      with no dataset returns an explicit 0.0.
    status: pending
  - id: ac-007
    summary: the kspace name is gone from the ForceField surface
    type: code
    pass_when: |
      StyleDefs has no KSpace variant, no category() returns "kspace", and
      def_kspacestyle exists in none of molrs/src/ff, molrs-python/src,
      molrs-python/python or molrs-capi/src; nothing def_kspace-shaped replaces
      it.
    status: pending
  - id: ac-008
    summary: PME is a pair style and its numbers did not move
    type: scientific
    pass_when: |
      lookup_kernel("pair", "coul/long/pme") resolves with
      ParamSource::PerInstance and lookup_kernel("kspace", "pme") is None; PME
      energy and forces on a fixed charged frame are bit-identical
      (assert_eq!) before and after the re-registration.
    status: pending
  - id: ac-009
    summary: the kspace module survives as the FFT compilation unit
    type: code
    pass_when: |
      molrs/src/ff/potential/kspace/{mod.rs,pme.rs} still exist and are declared
      by ff; the module header states that the boundary exists to keep the FFT
      dependency feature-gateable, and a grep gate asserts that sentence is
      present.
    status: pending
  - id: ac-010
    summary: no new physics test was added and the NVE bar is untouched
    type: runtime
    pass_when: |
      The existing molrs/src/md/integrators.rs conservation test passes with its
      5e-5 threshold unedited, and this spec's diff adds no new NVE, drift or
      long-run physics assertion.
    status: pending
  - id: ac-011
    summary: pair-merge regression gets one number from two paths
    type: runtime
    pass_when: |
      `python regressions/release-0-14-14-pair-kernel-merge.py` exits 0, imports
      no third-party scientific package, and reproduces its embedded analytic LJ
      golden to 1e-15 relative from both the md.LJCut path and the
      ForceField pair:lj/cut path, whose results differ by at most 1 ulp.
    status: pending
  - id: ac-012
    summary: full gate green after the merge
    type: runtime
    pass_when: |
      cargo lib tests, cargo doc tests, the architecture_gate target, the capi
      build and the molrs-python tox py env all pass.
    status: pending
out_of_scope:
  - new NVE / conservation / long-run physics tests
  - gating the FFT dependency out of the ff feature (0.15)
  - mapping LAMMPS kspace_style onto a pair style
  - per-type mixing and special_bonds exclusion on the driver pair path
  - Ewald / PPPM kernel implementations
  - wasm / capi md binding surface
---

# Acceptance — release-0-14-14-pair-kernel-merge

LJ 只剩一个 kernel、pair 数据每步只算一次且被所有 pair 势共享、势里再没有快照和 MIC、`kspace` 只作为编译单元活着而 PME 以 pair style 的身份归位——全部以逐位相等证明没动物理，唯一动不了逐位的一处（乘倒数 vs 除法）被量到 1 ulp 并当场解释，而不是藏进容差。
