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
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Ran BEFORE any deletion: 94 MMFF tests green — 11/11 RDKit parity + the frozen per-style breakdown.
      That baseline sits BEHIND the deletion, not ahead of it.

  - id: ac-002
    summary: A registered kernel constructor that ignores `tp` is not a Style
    type: runtime
    pass_when: |
      `molrs/tests/ff/potential/param_source_gate.rs` passes: for every `*_ctor` in
      `molrs/src/ff/potential/*/*.rs`, "second parameter is `_tp` (ignored)" holds IF AND ONLY IF its
      registration in `registry.rs` is `ParamSource::PerInstance`. BOTH directions asserted.
      This turns the whole bug class into a red light. Today the grep returns exactly the five MMFF
      files; every other kernel family in molrs really does resolve from `tp`.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `param_source_gate.rs` passes, both directions.
      THE SPEC'S OWN COUNT WAS WRONG AND THE GATE CAUGHT IT. The spec said six PerInstance styles, derived
      from `grep -l '_tp: &[(&str, &Params)]'` — a grep of a SPELLING. `pme_ctor` and `pair_coul_cut_ctor`
      ignore their type-params just as completely; they merely spell it `_type_params`, and both read
      per-atom charge from the Frame. EIGHT styles are PerInstance, not six. Registering only the six
      would have left the gate red.
      The tester proved the gate bites: injected a `tp`-ignoring `bond_sneaky_ctor` -> 8 violations became
      9, naming it, with no allowlist and no prior knowledge; and the reverse (tagging `pair/mmff_vdw`
      PerInstance while its ctor really reads `tp`) went red too. This is the whole bug class turned into
      a red light instead of an archaeology project.

  - id: ac-003
    summary: The six MMFF per-instance styles carry zero type-def rows
    type: runtime
    pass_when: |
      For a ForceField parsed from the MMFF94 parameter set, the styles `bond/mmff_bond`,
      `angle/mmff_angle`, `angle/mmff_stbn`, `dihedral/mmff_torsion`, `improper/mmff_oop`,
      `pair/mmff_ele` each have 0 type definitions and still compile to a Potential;
      `pair/mmff_vdw` still has exactly 95 (vdW genuinely IS a type-row lookup).
    status: verified
    last_checked: 2026-07-14
    evidence: |
      The 6 MMFF per-instance styles carry 0 type-def rows and still compile to a Potential;
      `pair/mmff_vdw` still has exactly 95 (vdW genuinely IS a type-row lookup and its ctor really reads
      `tp` — it was not swept into the deletion). The styles are still DECLARED, via the pre-existing
      generic <BondStyle>/<AngleStyle>/... elements with zero <Type> children — no new reader.

  - id: ac-004
    summary: A TypeRows style with no type definitions still errors
    type: runtime
    pass_when: |
      Constructing a `TypeRows` bonded style with an empty type-def list still returns
      Err("... has no type definitions"). The ParamSource relaxation must apply ONLY to per-instance
      styles — it must not quietly disable the empty-table check for everyone.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      A `TypeRows` style with no type-defs still errors. The implementer went further and removed the
      blanket `category != "pair"` escape hatch from the guard — `ParamSource` is exactly what that hatch
      was standing in for. No TypeRows pair style regressed.

  - id: ac-005
    summary: The 4,065 dead XML rows and their five readers are gone
    type: code
    pass_when: |
      In both MMFF XMLs and their root `data/` copies: grep -c for `<Bond `, `<Angle `,
      `<StretchBend `, `<Torsion `, `<Oop ` each returns 0, while `<VdW ` returns 95 and
      `<ElectrostaticParams` returns 1. `parse_mmff_bonds|parse_mmff_angles|parse_mmff_stbn|
      parse_mmff_torsions|parse_mmff_oop` returns 0 hits in `forcefield/xml.rs`, and
      `scripts/mmff_to_xml.py` no longer emits those five sections (or regeneration invites them back).
    status: verified
    last_checked: 2026-07-14
    evidence: |
      4,065 dead rows gone from all four XMLs (5299 -> 1134 lines each). Census: <Bond>/<Angle>/
      <StretchBend>/<Torsion>/<Oop> = 0, <VdW> = 95, <ElectrostaticParams> = 1. The five `parse_mmff_*`
      readers are gone, and `scripts/mmff_to_xml.py` no longer emits those sections — otherwise the next
      regeneration would have invited the dead data straight back.

  - id: ac-006
    summary: Bespoke energy path and the wrong classifiers are deleted repo-wide
    type: code
    pass_when: |
      `grep -rn 'MmffForceField|MmffEnergyBreakdown|build_mmff_potentials'` over molrs/src,
      molrs-python/{src,python,site-src,examples}, molrs/examples, molrs/tests and docs returns 0 hits;
      `molrs/src/ff/mmff/energy/` does not exist; `molrs/src/ff/typifier/mmff/classify.rs` does not
      exist; `MMFF94Typifier` exposes no `build` / `typify_bond` / `typify_angle` / `typify_dihedral`.
      (CHANGELOG.md excluded.)
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `MmffForceField` / `MmffEnergyBreakdown` / `build_mmff_potentials` = 0 live references (the 3
      remaining grep hits are historical PROSE in doc comments — "a bespoke MmffForceField assembly layer
      USED TO LIVE under energy/"). `ff/mmff/energy/` gone; `typifier/mmff/classify.rs` gone; no `build` /
      `typify_bond` / `typify_angle` / `typify_dihedral` on either front door.
      The three deleted classifiers were WRONG, and the RED tests pinned exactly how: benzene's aromatic
      ring bonds got type 1 where RDKit says 0 (backwards — and it labelled a six-fold-symmetric ring two
      different ways while all six resolve the same kb); cyclopropane's C-C-C angles got 0 where RDKit says
      3, and `typify_angle(bt_ij, bt_jk)` CANNOT return 3 at any input, because ring membership is not
      among its arguments. That turns "the signature cannot express the rule" from a claim into a proof.

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
    status: verified
    last_checked: 2026-07-14
    evidence: |
      REVERSE PROTECTION HELD. `molrs/src/ff/mmff/params.rs` exists (808 lines), imported by
      `frame_builder.rs`, and every bond/angle/dihedral label now derives from it. The 826-line
      RDKit-faithful resolver lived under `energy/` but is a PARAMETER RESOLVER, not an energy file — a
      deletion list that swallowed it would have destroyed the one correct implementation of the context
      rules. Only `vdw_params`/`relation`/`central_prop` were dropped from it: dead once the energy layer
      went, and `vdw_params` was a SECOND copy of combining rules that `pair/mmff.rs` already applies to
      `tp`.

  - id: ac-008
    summary: Energies are bit-for-bit unchanged after the deletion
    type: runtime
    pass_when: |
      After all deletions, the same 11 fixtures still match RDKit within 1e-3 kcal/mol and the
      per-style breakdown still matches the frozen `<name>.breakdown.json` within 1e-6, with NO
      assertion value edited — the diff of the tolerance / expected constants in
      `molrs/tests/ff/mmff/energy.rs` is empty. Loosening a tolerance to make this green is a failure,
      not a pass.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Energies bit-for-bit unchanged. `ENERGY_TOL = 1.0e-3` and `BREAKDOWN_TOL = 1.0e-6` are
      byte-identical; `git status molrs/tests/ff/mmff/fixtures/` is empty — no oracle regenerated. The only
      tolerance that disappeared is `BESPOKE_TOL`, whose subject no longer exists.

  - id: ac-009
    summary: conformer MMFF cleanup runs on the generic route, typifier hoisted
    type: runtime
    pass_when: |
      The ETKDG / conformer suites pass unchanged; `conformer/etkdg/mod.rs` names no MMFF symbol other
      than the typifier front door; and `MMFF94Typifier::new()` appears zero times inside the
      per-conformer loop body (it parses the embedded XML on every construction).
    status: verified
    last_checked: 2026-07-14
    evidence: |
      conformer's `mmff_cleanup` runs on the typify -> ForceField route, with `MMFF94Typifier::new()`
      hoisted into a `OnceLock` — it parses the embedded XML on every construction, so leaving it in the
      per-conformer loop would have been a real cost. ETKDG suite unchanged (21 -> 21).

  - id: ac-010
    summary: chem-perceive-14's frozen table contract regenerated from the trimmed XML
    type: code
    pass_when: |
      MOVED TO chem-perceive-14 (amended 2026-07-14). The original clause asked THIS spec to regenerate
      `tests/ff/fixtures/tables/{mmff94,mmff94s}.reference.txt` and re-pin their SHA-256. Wrong ordering:
      those files are currently PARKED (they belong to chem-perceive-14), and restoring them here would
      drag that spec's unrelated REDs (the `generated/` directory name, the flat-table layout, the
      $AMBERHOME gate) into this one.
      What this spec owes instead is a HANDOFF CONSTRAINT, and it is recorded in the spec body:
      chem-perceive-14 must REGENERATE that dump from the FINAL XML, never reuse the parked one. The
      parked dump was taken too early — on an XML that still had the 4,065 dead rows and no
      <ElectrostaticParams>. Reusing it would compile the dead rows into committed, test-protected Rust
      tables, which is the exact outcome this whole chain exists to prevent.
      This criterion passes when the spec body carries that constraint in writing.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      HANDOFF CONSTRAINT recorded in the spec body (this criterion was converted from a task —
      see its pass_when). chem-perceive-14 must REGENERATE the frozen table dump from the FINAL XML and
      must NOT reuse the parked one, which was taken on an XML that still had the 4,065 dead rows and no
      <ElectrostaticParams>. Reusing it would compile the dead rows into committed, test-protected Rust
      tables — the exact outcome this chain exists to prevent. The parked files were never touched.

  - id: ac-011
    summary: Python surface offers only the typify -> ForceField route, wheel rebuilt
    type: runtime
    pass_when: |
      `maturin develop --release` succeeds and pytest reports 505 passed / 0 failed;
      `import molrs; assert not hasattr(molrs, 'build_mmff_potentials'); assert not
      hasattr(molrs.MMFF94Typifier(), 'build')` exits 0.
      A Python user must not be able to reach the broken door after the Rust cleanup.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      molrs-python 510 passed / 0 failed after `maturin develop --release`. Verified live:
      `hasattr(molrs, 'build_mmff_potentials')` -> False; `hasattr(molrs.MMFF94Typifier(), 'build')` ->
      False; `typify` -> True. Before this spec a Python user had TWO adjacent exports in one namespace for
      the same job, with nothing to tell them apart, and one of them silently omitted the entire
      electrostatic term.

  - id: ac-012
    summary: Full molrs gates green, no unexplained test-count drift
    type: runtime
    pass_when: |
      All five Rust gates pass with 0 failed; the delta between the passed count and the 1914 baseline
      is itemised (deleted bespoke/classifier tests, added gate tests). "The number moved but everything
      is green" is NOT acceptable. The 15 parked chem-perceive-14 REDs stay RED and untouched.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      Rust 1934 passed / 0 failed; Python 510 / 0. All five gates exit 0. Test-count delta -19
      (1953 -> 1934), ITEMISED — no "the number moved but everything is green":
      lib -10: 7 classifier unit tests (wrong values) + 4 dead-reader tests deleted, +1 per-instance test.
      ff  -9: 4 RENAMES (not losses); 9 genuine deletions = bespoke_gate (1, it pinned the very tree being
      deleted), 3 bespoke energy tests each having a surviving generic_path_* twin, 4 classifier tests
      locking in wrong values, 1 asserting on now-deleted type rows.
      Crucially, the bespoke tests that asserted science with NO generic twin were REPOINTED, not dropped —
      `mmff94s_total_energy_matches_rdkit` above all, which is the ONLY external oracle the static variant
      has. Dropping it would have deleted MMFF94s-vs-RDKit parity outright while the suite stayed green.
      The 15 parked chem-perceive-14 REDs are untouched and still absent.

  - id: ac-013
    summary: BREAKING change recorded, molpack named
    type: code
    pass_when: |
      `CHANGELOG.md` has an unreleased BREAKING entry listing the removal of `MmffForceField`,
      `MMFF94Typifier::build` / `MMFF94STypifier::build` and `molrs.build_mmff_potentials`, showing the
      replacement snippet, and naming molpack as the downstream that must follow up (docs/interop.md
      documents `build` as "the pattern the molpack relaxer follows"). `docs/interop.md`'s MMFF snippet
      is compile-checked as a doctest and uses the new route.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      CHANGELOG BREAKING entry lists the removal of `MmffForceField`, `MMFF94Typifier::build` /
      `MMFF94STypifier::build` and `molrs.build_mmff_potentials`, shows the replacement snippet, and names
      molpack as the downstream that must follow up (`docs/interop.md` documented `build` as "the pattern
      the molpack relaxer follows"). `docs/interop.md`'s MMFF snippet is a compile-checked doctest and now
      uses the typify -> to_potentials route.
---

# Acceptance criteria

- **ac-001** 是排序纪律的可执行形式：**先证明，再删除**。必须在任何删除 commit 之前跑过并记录。
- **ac-002** 是本 spec 的架构交付物：不变量从"一句 grep"升级成"一条测试"。它是**双向**的——既禁止"忽略 `tp` 却假装是 Style"，也禁止"注册成 PerInstance 却真的去读 `tp`"。
- **ac-003 + ac-004** 成对存在：guard 必须**只**对 per-instance 放宽，不能顺手把所有 style 的空表检查废掉。
- **ac-005 ~ ac-007** 是删除清单的机器可查形式，其中 **ac-007 是反向保护**：那份 826 行的 RDKit-faithful 解析器必须活下来，并搬出 `energy/`。
- **ac-008** 是"删除不改数值"的证明：断言值一个都不许动。**为了让测试变绿而放宽容差，是失败，不是通过。**
- **ac-010** 阻止 `chem-perceive-14-all-tables` 把死数据编译成 Rust 表。
