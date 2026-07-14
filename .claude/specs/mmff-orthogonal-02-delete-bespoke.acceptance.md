---
slug: mmff-orthogonal-02-delete-bespoke
created: 2026-07-14
criteria:
  - id: ac-001
    summary: Re-assert 11/11 RDKit parity + frozen breakdown BEFORE any deletion
    type: runtime
    pass_when: |
      On the commit that STARTS this spec (bespoke still present), the generic-path total-energy test
      passes on all 11 fixtures (< 1e-3 kcal/mol) and the per-style breakdown test passes (< 1e-6).
      Recorded before any deletion commit.
      This is the ordering discipline made executable: PROVE, then delete. Deletion changes the label
      source in frame_builder, the conformer call site and the XML line count — every one of those needs
      a known-green baseline BEHIND it, not ahead of it.
    status: pending
    last_checked:
    evidence:

  - id: ac-002
    summary: A registered kernel constructor that ignores `tp` is not a Style
    type: runtime
    pass_when: |
      `molrs/tests/ff/potential/param_source_gate.rs` passes: for every `*_ctor` in
      `molrs/src/ff/potential/*/*.rs`, "second parameter is `_tp` (ignored)" holds IF AND ONLY IF its
      registration in `registry.rs` is `ParamSource::PerInstance`. BOTH directions asserted.
      This turns the whole bug class into a red light. Today the grep returns exactly the five MMFF
      files; every other kernel family in molrs really does resolve from `tp`.
    status: pending
    last_checked:
    evidence:

  - id: ac-003
    summary: The six MMFF per-instance styles carry zero type-def rows
    type: runtime
    pass_when: |
      For a ForceField parsed from the MMFF94 parameter set, the styles `bond/mmff_bond`,
      `angle/mmff_angle`, `angle/mmff_stbn`, `dihedral/mmff_torsion`, `improper/mmff_oop`,
      `pair/mmff_ele` each have 0 type definitions and still compile to a Potential;
      `pair/mmff_vdw` still has exactly 95 (vdW genuinely IS a type-row lookup).
    status: pending
    last_checked:
    evidence:

  - id: ac-004
    summary: A TypeRows style with no type definitions still errors
    type: runtime
    pass_when: |
      Constructing a `TypeRows` bonded style with an empty type-def list still returns
      Err("... has no type definitions"). The ParamSource relaxation must apply ONLY to per-instance
      styles — it must not quietly disable the empty-table check for everyone.
    status: pending
    last_checked:
    evidence:

  - id: ac-005
    summary: The 4,065 dead XML rows and their five readers are gone
    type: code
    pass_when: |
      In both MMFF XMLs and their root `data/` copies: grep -c for `<Bond `, `<Angle `,
      `<StretchBend `, `<Torsion `, `<Oop ` each returns 0, while `<VdW ` returns 95 and
      `<ElectrostaticParams` returns 1. `parse_mmff_bonds|parse_mmff_angles|parse_mmff_stbn|
      parse_mmff_torsions|parse_mmff_oop` returns 0 hits in `forcefield/xml.rs`, and
      `scripts/mmff_to_xml.py` no longer emits those five sections (or regeneration invites them back).
    status: pending
    last_checked:
    evidence:

  - id: ac-006
    summary: Bespoke energy path and the wrong classifiers are deleted repo-wide
    type: code
    pass_when: |
      `grep -rn 'MmffForceField|MmffEnergyBreakdown|build_mmff_potentials'` over molrs/src,
      molrs-python/{src,python,site-src,examples}, molrs/examples, molrs/tests and docs returns 0 hits;
      `molrs/src/ff/mmff/energy/` does not exist; `molrs/src/ff/typifier/mmff/classify.rs` does not
      exist; `MMFF94Typifier` exposes no `build` / `typify_bond` / `typify_angle` / `typify_dihedral`.
      (CHANGELOG.md excluded.)
    status: pending
    last_checked:
    evidence:

  - id: ac-007
    summary: The RDKit-faithful resolver SURVIVES, out of `energy/`
    type: code
    pass_when: |
      `molrs/src/ff/mmff/params.rs` exists (the moved 826-line resolver), is imported by
      `typifier/mmff/frame_builder.rs`, and frame_builder derives every bond / angle / dihedral type
      label from it — no second classifier anywhere in the tree.
      This is the REVERSE protection: the deletion list must not swallow the one correct
      implementation of the context rules. It lived under `energy/` but it is a parameter resolver,
      not an energy file.
    status: pending
    last_checked:
    evidence:

  - id: ac-008
    summary: Energies are bit-for-bit unchanged after the deletion
    type: runtime
    pass_when: |
      After all deletions, the same 11 fixtures still match RDKit within 1e-3 kcal/mol and the
      per-style breakdown still matches the frozen `<name>.breakdown.json` within 1e-6, with NO
      assertion value edited — the diff of the tolerance / expected constants in
      `molrs/tests/ff/mmff/energy.rs` is empty. Loosening a tolerance to make this green is a failure,
      not a pass.
    status: pending
    last_checked:
    evidence:

  - id: ac-009
    summary: conformer MMFF cleanup runs on the generic route, typifier hoisted
    type: runtime
    pass_when: |
      The ETKDG / conformer suites pass unchanged; `conformer/etkdg/mod.rs` names no MMFF symbol other
      than the typifier front door; and `MMFF94Typifier::new()` appears zero times inside the
      per-conformer loop body (it parses the embedded XML on every construction).
    status: pending
    last_checked:
    evidence:

  - id: ac-010
    summary: chem-perceive-14's frozen table contract regenerated from the trimmed XML
    type: code
    pass_when: |
      `tests/ff/fixtures/tables/mmff94.reference.txt` and `mmff94s.reference.txt` contain no entry line
      for styles `mmff_bond` / `mmff_angle` / `mmff_stbn` / `mmff_torsion` / `mmff_oop`, and the SHA-256
      in each header equals that of the trimmed XML and equals the pin in `tests/ff/tables_gate.rs`.
      Without this, chem-perceive-14 compiles the dead rows into committed Rust tables protected by
      tests — exactly what this chain exists to prevent.
    status: pending
    last_checked:
    evidence:

  - id: ac-011
    summary: Python surface offers only the typify -> ForceField route, wheel rebuilt
    type: runtime
    pass_when: |
      `maturin develop --release` succeeds and pytest reports 505 passed / 0 failed;
      `import molrs; assert not hasattr(molrs, 'build_mmff_potentials'); assert not
      hasattr(molrs.MMFF94Typifier(), 'build')` exits 0.
      A Python user must not be able to reach the broken door after the Rust cleanup.
    status: pending
    last_checked:
    evidence:

  - id: ac-012
    summary: Full molrs gates green, no unexplained test-count drift
    type: runtime
    pass_when: |
      All five Rust gates pass with 0 failed; the delta between the passed count and the 1914 baseline
      is itemised (deleted bespoke/classifier tests, added gate tests). "The number moved but everything
      is green" is NOT acceptable. The 15 parked chem-perceive-14 REDs stay RED and untouched.
    status: pending
    last_checked:
    evidence:

  - id: ac-013
    summary: BREAKING change recorded, molpack named
    type: code
    pass_when: |
      `CHANGELOG.md` has an unreleased BREAKING entry listing the removal of `MmffForceField`,
      `MMFF94Typifier::build` / `MMFF94STypifier::build` and `molrs.build_mmff_potentials`, showing the
      replacement snippet, and naming molpack as the downstream that must follow up (docs/interop.md
      documents `build` as "the pattern the molpack relaxer follows"). `docs/interop.md`'s MMFF snippet
      is compile-checked as a doctest and uses the new route.
    status: pending
    last_checked:
    evidence:
---

# Acceptance criteria

- **ac-001** 是排序纪律的可执行形式：**先证明，再删除**。必须在任何删除 commit 之前跑过并记录。
- **ac-002** 是本 spec 的架构交付物：不变量从"一句 grep"升级成"一条测试"。它是**双向**的——既禁止"忽略 `tp` 却假装是 Style"，也禁止"注册成 PerInstance 却真的去读 `tp`"。
- **ac-003 + ac-004** 成对存在：guard 必须**只**对 per-instance 放宽，不能顺手把所有 style 的空表检查废掉。
- **ac-005 ~ ac-007** 是删除清单的机器可查形式，其中 **ac-007 是反向保护**：那份 826 行的 RDKit-faithful 解析器必须活下来，并搬出 `energy/`。
- **ac-008** 是"删除不改数值"的证明：断言值一个都不许动。**为了让测试变绿而放宽容差，是失败，不是通过。**
- **ac-010** 阻止 `chem-perceive-14-all-tables` 把死数据编译成 Rust 表。
