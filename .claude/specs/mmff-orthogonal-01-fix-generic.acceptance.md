---
slug: mmff-orthogonal-01-fix-generic
created: 2026-07-14
criteria:
  - id: ac-001
    summary: Generic path fixture list is directory-scanned, the false comment is gone
    type: code
    pass_when: |
      `molrs/tests/ff/mmff/energy.rs` contains no hardcoded fixture-name array in
      `generic_path_total_energy_matches_rdkit`; the list is produced by scanning
      `molrs/tests/ff/mmff/fixtures/*.energy.json`, so a subset assertion is structurally
      impossible. The false comment at the old `:443-449` ("stretch-bend + torsion eq-fallback
      label resolution") is deleted — stbn and torsion already agree with bespoke to five
      decimals on every fixture; the real defect was the missing electrostatic term.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Fixture list in `generic_path_total_energy_matches_rdkit` is now produced by scanning
      `molrs/tests/ff/mmff/fixtures/*.energy.json` — a subset assertion is STRUCTURALLY impossible to
      write. (The bare-named fixtures carry a `.json`, not a `.energy.json`, so the scan separates the 11
      energy fixtures from the typifier ones.) The false comment blaming "stretch-bend + torsion
      eq-fallback label resolution" is deleted; stbn and torsion always agreed with bespoke to five
      decimals — the real defect was an entire missing energy term.

  - id: ac-002
    summary: 11/11 RDKit total-energy parity on the generic path
    type: runtime
    pass_when: |
      All 11 fixtures asserted (e_ethane, e_butane, e_ethylene, e_benzene, e_caffeine, e_big,
      e_acetonitrile, s_acetamide, s_aniline, s_nmethylacetamide, s_urea), each
      |E_generic - E_rdkit| < 1e-3 kcal/mol.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      11/11 RDKit total-energy parity. Worst |delta| = 3.1e-9 (e_big) against a 1e-3 tolerance.
      Before: 9/11 failed — caffeine 150.5, urea 116.6, NMA 28.3, ethylene 8.1, e_big 3.3, benzene 3.1.

  - id: ac-003
    summary: Per-style breakdown matches the frozen bespoke reference
    type: runtime
    pass_when: |
      For every fixture, each of the 7 style energies (bond, angle, stbn, torsion, oop, vdw, ele)
      computed via `Style::to_potential` differs from `<name>.breakdown.json` by < 1e-6 kcal/mol.
      A total-energy assertion alone is NOT sufficient: the total hid a 28 kcal/mol electrostatic
      hole behind partially-cancelling terms for a month.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Per-style breakdown matches the frozen bespoke reference to < 1e-6 on all 7 terms x 11 fixtures.
      NMA: bond 3.166364, angle 4.493303, stbn -0.315962, torsion -1.182367, oop 0.0, vdw 3.855594,
      ele -28.165949. The `ele` column did not exist at all before this spec.

  - id: ac-004
    summary: Frozen breakdown fixtures are self-validated against the RDKit oracle
    type: runtime
    pass_when: |
      For each of the 11 `<name>.breakdown.json`, the sum of its 7 terms equals the
      `mmff94_total_energy` in the matching `<name>.energy.json` to within 1e-6 kcal/mol,
      asserted IN-TEST (not by inspection). This is what makes the frozen reference legitimate
      after spec 02 deletes the bespoke path that produced it.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Each of the 11 frozen breakdowns self-validates: sum(7 terms) == the RDKit
      `mmff94_total_energy`, worst case 1.4e-13 (tol 1e-6). The tester went further than the criterion and
      cross-checked every frozen TERM against RDKit's own per-term oracle (SetMMFF*Term toggles): max
      |delta| = 1.9e-14, term by term, every fixture. So the freeze is RDKit-faithful per term, not merely
      self-summing — which is what makes it a legitimate reference after spec 02 deletes the bespoke path
      that produced it.

  - id: ac-005
    summary: pair/mmff_ele is DEFINED by a ForceField, not merely registered
    type: code
    pass_when: |
      `grep -rn 'def_pairstyle("mmff_ele"' molrs/src/ff/forcefield/xml.rs` returns >= 1 hit, and
      `molrs/data/mmff94.xml` + `molrs/data/mmff94s.xml` + the root `data/` copies each contain
      exactly one `<ElectrostaticParams` element, emitted by `scripts/mmff_to_xml.py`.
      The kernel was registered from the day it was written and never wired to a style — that is
      the whole 150 kcal/mol caffeine error.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `<ElectrostaticParams dielectric="1.0" delta="0.05" scale14="0.75"/>` in all four XMLs (both
      molrs/data and the root data/ copies), emitted by scripts/mmff_to_xml.py so a regeneration cannot
      silently delete it. `read_forcefield_xml_str` dispatches to `parse_mmff_ele` -> `def_pairstyle(
      "mmff_ele", ...)` and calls `set_special_bonds(lj14 = 1.0, coul14 = 0.75)`. `mmff_ele_ctor` reads
      dielectric / delta / coulomb14scale from `sp`; the hardcoded 0.75 is gone.
      The kernel had been REGISTERED since the day it was written and never wired to a style. That is the
      entire 150 kcal/mol caffeine error.

  - id: ac-006
    summary: Zero-charge molecules get EXACTLY zero electrostatic energy
    type: runtime
    pass_when: |
      For e_ethane and e_butane (the only two fixtures with sum|q| = 0) the `pair/mmff_ele` style
      energy is exactly 0.0, and both totals still match RDKit within 1e-3 kcal/mol. This is the
      reverse assertion: it proves the new term was added in the right place, not merely added.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      e_ethane and e_butane (the only two fixtures with sum|q| = 0) get EXACTLY 0.0 from the
      `pair/mmff_ele` style, and their totals still match RDKit. This is the reverse assertion: it proves
      the term was added in the right place, not merely added.

  - id: ac-007
    summary: MMFF vdW applies the donor/acceptor rule
    type: code
    pass_when: |
      `parse_mmff_vdw` reads the `da` attribute into the PairType params, and `vdw_combining` reads
      B/Beta/DARAD/DAEPS from the style params (`sp`, no longer `_sp`), suppressing the B-term for
      donors and scaling R*/epsilon for donor-acceptor pairs.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `parse_mmff_vdw` reads the `da` attribute into PairType.params; `vdw_combining(pi, pj, sp)`
      takes B / Beta / DARAD / DAEPS from the style params (no longer `_sp`) and applies donor R*
      suppression plus donor-acceptor scaling. Implementation note worth keeping: epsilon is evaluated at
      the UNSCALED R* — scaling first would cost a spurious DARAD^-6 (~3.8x). Measured recovery: urea
      0.588, e_big 1.261, NMA 0.116 — matching the predicted costs exactly.

  - id: ac-008
    summary: The angle cubic constant is -0.4 rad^-1 exactly
    type: code
    pass_when: |
      `molrs/src/ff/potential/angle/mmff.rs` declares `CB_RAD = -0.4` (not -0.40107) with a comment
      deriving it as -0.006981317 deg^-1 * 180/pi, and a unit test asserts `CB_RAD == -0.4`.
      The original author rounded the constant BEFORE converting; the comment `// = -0.007 * 180/pi`
      records the mistake in plain sight.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `CB_RAD = -0.4` exactly, with the deg->rad derivation and the original mistake recorded in the
      doc comment. The old value -0.40107 came from `// = -0.007 * 180/pi` — the author ROUNDED the
      constant (-0.006981317) BEFORE converting. 0.27% error in the anharmonic term, worth 0.0073 kcal/mol
      on e_big: 7x the test tolerance, so e_big would have failed on this alone even after the
      electrostatics and vdW were fixed.

  - id: ac-009
    summary: Linear-angle branch is exercised by a BENT acetonitrile fixture
    type: runtime
    pass_when: |
      `molrs/tests/ff/mmff/fixtures/e_acetonitrile.{sdf,energy.json}` exist; the test asserts the
      C-C#N angle deviates >= 2 degrees from 180 (committed: 169.997 deg), and that the generic total
      matches RDKit within 1e-3 kcal/mol.
      The >= 2 degree guard is load-bearing: at ETKDG's own 179.14 deg the linear and cubic forms differ
      by only ~1e-4*ka — BELOW the 1e-3 tolerance — so a straight-from-ETKDG fixture would silently pass
      whether or not the defect is fixed. It would be a fake fixture wearing the badge of the very defect
      it was added to catch.
      AMENDED 2026-07-14 — two errors of mine, both caught by measurement:
      (a) The original clause "the fixture's stretch-bend style energy is exactly 0.0" is FALSE. RDKit's
      oracle gives stbn(CH3-C#N) = -0.010941: the methyl H-C-H / H-C-C angles carry stretch-bend like any
      sp3 centre. ONLY THE LINEAR CENTRE is skipped. Asserting the literal clause would have forced the
      implementer to break the physics to go green. The correct, oracle-backed form is: stbn is
      bit-for-bit INVARIANT under displacing the nitrogen (RDKit: delta = 0.000000, while bond moves
      +15.91 and angle -0.90 under the same displacement), with non-vacuity guards (stbn != 0,
      |delta_bond| > 1).
      (b) The oracle energy is 1.50754068, not the 1.50630 I measured. Mine came from the full-precision
      in-memory conformer; the committed SDF carries RDKit's standard 4-decimal coordinates, and the
      0.0012 kcal/mol rounding shift is ABOVE the 1e-3 parity tolerance — freezing my number would have
      produced a fixture that FAILS BY CONSTRUCTION even after all four defects are fixed. House
      convention, verified against all 10 pre-existing fixtures: RDKit re-read from the committed .sdf
      reproduces the stored mmff94_total_energy to exactly 0.0. So the oracle must be taken on the
      RE-READ geometry, never on the in-memory one.
      Also measured: acetonitrile's electrostatic energy is exactly 0 (MMFF gives its methyl hydrogens
      zero charge), so this fixture cannot expose defect 1 either — it is purely the linear-angle probe.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `e_acetonitrile.{sdf,energy.json}` committed; C-C#N angle 169.997 deg (deviation 10.0 >= 2).
      Generic total matches RDKit within 1e-3.
      Two of MY errors were caught by measurement here and are recorded in the pass_when above: (a) "stbn
      must be exactly 0" is FALSE — the methyl centres carry stbn (-0.010941); only the LINEAR centre is
      skipped, and implementing my literal clause would have broken the physics; (b) the oracle is
      1.50754068 on the RE-READ geometry, not the 1.50630 I measured in memory — the committed SDF's
      4-decimal coordinates shift the energy by 0.0012, which is ABOVE the 1e-3 tolerance, so freezing my
      number would have produced a fixture that fails BY CONSTRUCTION even after every defect is fixed.

  - id: ac-010
    summary: Generic-path forces are finite-difference consistent on every fixture
    type: runtime
    pass_when: |
      For all 11 fixtures, max|F_analytic + dE/dx_fd| < 1e-5 with h = 1e-5 — covering the newly
      added analytic gradient of the linear-angle branch.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Finite-difference gradients consistent on all 11 fixtures (h = 1e-5, max_err < 1e-5), covering
      the new analytic linear-angle gradient dE/dtheta = -143.9325 * ka * sin(theta).

  - id: ac-011
    summary: The bespoke path is UNTOUCHED and still exactly reproduces RDKit
    type: runtime
    pass_when: |
      `git diff --stat` shows no change under `molrs/src/ff/mmff/`, and the existing bespoke parity
      test passes with max |delta| = 0.00000 across all fixtures.
      This is the ordering discipline in machine-checkable form: NEVER delete or disturb the only
      correct implementation before the replacement is proven.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `git status --short molrs/src/ff/mmff/` returns ZERO entries, verified independently. The bespoke
      parity test still passes with max |delta| = 0.00000 (worst 1.1e-13), and `bespoke_gate.rs` +
      `BESPOKE.sha256` pin all 16 files.
      This is the ordering discipline in machine-checkable form: the only correct implementation is the
      measuring stick, and 01 must not disturb it. The generic kernels re-derive the same formulas
      independently — they are NOT copied — so the < 1e-6 breakdown agreement is a genuine second opinion,
      which is what makes it safe for spec 02 to delete the bespoke path.

  - id: ac-012
    summary: Full molrs gates green, no regression against the 1914 baseline
    type: runtime
    pass_when: |
      `cargo fmt --all --check`; `cargo clippy --workspace --all-targets -- -D warnings`;
      `cargo clippy -p molcrafts-molrs --all-targets --features full -- -D warnings`;
      `RUSTDOCFLAGS='-D warnings' cargo doc`; and `cargo test -p molcrafts-molrs --features full`
      reports 0 failed with passed >= 1914 (the 15 parked chem-perceive-14 REDs excluded as before).
    status: verified
    last_checked: 2026-07-14
    evidence: |
      1926 passed / 0 failed (1913 baseline + 13 new). cargo fmt --check, both clippy invocations
      with -D warnings, and RUSTDOCFLAGS='-D warnings' cargo doc all exit 0. The 15 chem-perceive-14 REDs
      remain parked out of the tree.
---

# Acceptance criteria

- **ac-001 / ac-002** 是本 spec 的存在理由：把"只断言 e_ethane"这一反模式**结构性**消灭（列表由目录扫描产生，写不出子集），并要求 11/11。今天那个唯一被断言的分子，恰好是唯一**不可能暴露主缺陷**的那一类。
- **ac-003 / ac-004** 是给 spec 02 留的安全网：逐项分解冻结自 bespoke，且每个冻结文件由外部 RDKit 总能量自证（七项之和 == oracle）。bespoke 删除后参照物仍然成立。**总和是最容易撒谎的那个数**——它把 150 kcal/mol 的空洞藏了一个月。
- **ac-005 ~ ac-009** 一一对应四个已测缺陷。ac-006 与 ac-009 是**反向断言**：零电荷分子的静电必须恰好为 0，线性中心的 stbn 必须恰好为 0 —— 防止"加了项但加错地方"。
- **ac-011** 是排序纪律的机器可查形式：**01 不许碰 bespoke**。
