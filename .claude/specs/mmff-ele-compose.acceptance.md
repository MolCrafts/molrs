---
slug: mmff-ele-compose
created: 2026-07-15
criteria:
  - id: ac-001
    summary: pair/mmff_ele is gone; MMFF composes from the generic Coulomb kernel
    type: code
    pass_when: |
      `grep -rn 'mmff_ele\|MMFFElectrostatic' molrs/src` returns 0 hits. The MMFF ForceField defines
      `pair/coul/cut` (not a bespoke style), and `registry.rs` no longer registers `pair/mmff_ele`.
      Owner's ruling: MMFF must not own a kernel. Its electrostatics is a BUFFERED COULOMB — a
      parameterization of the generic kernel, not a kernel of its own. `coul_cut.rs`'s own module doc
      already said its kernel was "per-atom, mirroring mmff_ele": two implementations of one physics,
      and the author knew.
    status: verified
    last_checked: 2026-07-15
    evidence: |
      `grep -rn 'mmff_ele|MMFFElectrostatic' molrs/src` -> 0. `pair/mmff_ele` is unregistered; MMFF's
      ForceField defines `pair/coul/cut`. `parse_mmff_ele` -> `parse_electrostatics` (the old name still
      matched the banned substring). MMFF no longer owns a kernel.

  - id: ac-002
    summary: The generic Coulomb kernel carries delta / dielectric / the Coulomb constant
    type: code
    pass_when: |
      `pair/coul/cut` evaluates `E = k * qi*qj / (D * (r + delta))` with `k`, `D` and `delta` taken from
      the STYLE PARAMS. `delta = 0` reproduces the pre-spec kernel BIT-FOR-BIT (regression guard: the
      generalization must not change existing behaviour).
      The Coulomb constant MUST be a style param, not a shared constant: MMFF uses Halgren's 332.0716
      while `coul/cut` used CODATA's 332.06371. The 2.4e-5 relative difference is worth 0.0036 kcal/mol
      on caffeine's E_ele = -150.48 — ABOVE the 1e-3 parity tolerance. Both values are correct; the
      force field decides. "Merge the two kernels by picking one constant" is the wrong move.
    status: verified
    last_checked: 2026-07-15
    evidence: |
      `pair/coul/cut` evaluates `E = k*qi*qj / (D*(r+delta))` with k / D / delta from the STYLE params.
      `delta_zero_reproduces_the_unbuffered_kernel_bit_for_bit` is a real assert_eq! on f64 and passes:
      k/D is hoisted (x/1.0 == x exactly), r + 0.0 == r, and the force factor associates identically.
      The Coulomb constant stayed a style param, NOT a merged constant. MMFF uses Halgren's 332.0716,
      OPLS/LAMMPS use CODATA's 332.06371 — 2.4e-5 relative, worth 0.0036 kcal/mol on caffeine's
      E_ele = -150.48, ABOVE the 1e-3 parity tolerance. `the_two_coulomb_constants_are_not_interchangeable`
      proves it from the frozen term. Both values are correct; the force field decides.

  - id: ac-003
    summary: A style missing force-field data ERRORS — it does not silently default
    type: runtime
    pass_when: |
      Constructing `pair/coul/cut` from a style whose params lack `coulomb`, `dielectric` or
      `coulomb14scale` returns `Err`. Proven by building exactly that style and asserting the error.
      THIS IS THE BACKDOOR THIS SPEC EXISTS TO CLOSE. mmff-orthogonal-02's tester found that with
      `def_pairstyle("mmff_ele", &[])` — style present, params EMPTY — 18 of 19 energy tests still
      passed, because the kernel had `sp.get("dielectric").unwrap_or(1.0)` /
      `sp.get("delta").unwrap_or(ELE_DELTA)`. The force field silently stopped being the source of
      MMFF's constants and the kernel used its own hardcoded copies: THE RIGHT NUMBERS FOR THE WRONG
      REASON. Only one gate in the whole suite could see it.
      Distinguish honestly: `cutoff = INFINITY` is a genuine SEMANTIC default ("do not truncate" is a
      real, meaningful choice). `dielectric = 1.0` is the kernel PRETENDING THE FORCE FIELD SPOKE.
      Keep the former, delete the latter.
    status: verified
    last_checked: 2026-07-15
    evidence: |
      THE BACKDOOR IS CLOSED. `coulomb` / `dielectric` / `coulomb14scale` missing -> `Err` (one `required()`
      helper; no `unwrap_or`). Proven before the fix: an EMPTY-params `coul/cut` style compiled and computed
      E = -41.507964 kcal/mol "from constants no force field ever gave it".
      The honest distinction is mechanically enforced: `cutoff = INFINITY` and `delta = 0` remain SEMANTIC
      defaults ("do not truncate" / "unbuffered" are real choices) and their two tests stayed green through
      the change; `dielectric = 1.0` was the kernel PRETENDING THE FORCE FIELD SPOKE, and is now an error.
      Blast radius the spec did not anticipate: OPLS and LAMMPS also defined `coul/cut` with EMPTY params
      (3 production sites + 1 test fixture). They only ever "worked" because the kernel's private CODATA
      copy happened to be what they wanted. All four now declare their constants.

  - id: ac-004
    summary: MMFF's electrostatic constants live in MMFF's table, not in molrs's constants
    type: code
    pass_when: |
      `grep -rn 'COULOMB_MMFF\|ELE_BUFFER' molrs/src` returns 0 hits. Halgren's 332.0716 and the 0.05 A
      buffer live in `MMFF_ELE_STYLE` in `ff/params/mmff.rs` — they are MMFF's PARAMETERS, not molrs's
      constants. A constant in `ff/constants.rs` claims to be a property of the universe; these are a
      property of one force field.
    status: verified
    last_checked: 2026-07-15
    evidence: |
      `grep -rn 'COULOMB_MMFF|ELE_BUFFER' molrs/src` -> 0. Halgren's 332.0716 and the 0.05 A buffer are a
      real `coulomb` field on `MMFF_ELE_STYLE` in `ff/params/mmff.rs` (the gate strips comments, so a doc
      mention would not count). `ff/constants.rs` claims to hold properties of the universe; these are
      properties of one force field. `VACUUM_DIELECTRIC = 1.0` was added there and survives the gate —
      vacuum permittivity is 1 BY DEFINITION, and the force field still has to CHOOSE it.

  - id: ac-005
    summary: Pure refactor — not one number moves
    type: runtime
    pass_when: |
      All 11 fixtures still match RDKit within 1e-3 kcal/mol and the 7-term frozen breakdown still
      matches within 1e-6, with NO asserted value and NO tolerance edited. Loosening a tolerance to go
      green is a failure, not a pass.
    status: verified
    last_checked: 2026-07-15
    evidence: |
      Pure refactor. 11/11 RDKit fixtures within 1e-3, the 7-term frozen breakdown within 1e-6, and NO
      asserted value or tolerance edited anywhere. 1973 passed / 0 failed (baseline 1950).

  - id: ac-006
    summary: chem-perceive-14's loose ends are closed
    type: code
    pass_when: |
      `scripts/mmff_to_xml.py` is deleted (it wrote the XMLs that no longer exist); the workspace-root
      `data/mmff94.xml` / `data/mmff94s.xml` are deleted (referenced by nothing); and `CLAUDE.md`'s
      claim that MMFF params are "embedded at compile time in core (`molrs/data/mmff94.xml`, exposed as
      `molrs::data::MMFF94_XML`)" is corrected — it is false, and a false doc misleads every later
      reader, which this chain has now paid for three times.
    status: verified
    last_checked: 2026-07-15
    evidence: |
      `scripts/mmff_to_xml.py` DELETED, and the gate that guarded it replaced rather than dropped:
      `the_xml_generator_no_longer_emits_the_dead_sections` -> `the_xml_emitter_is_gone`. Its own docstring
      explained why: it existed because the script's output "feeds the next regeneration" — that
      regeneration was chem-perceive-14, it has happened, the XMLs are gone, and a generator whose output
      nobody reads is not harmless but a TRAP (the next person runs it, gets two XMLs, and reasonably
      assumes they matter). Workspace-root `data/mmff94*.xml` deleted. `CLAUDE.md`'s false claim that MMFF
      params are `include_str!`'d from `molrs/data/mmff94.xml` is corrected, and the three architectural
      rulings this chain established are now written there.

  - id: ac-007
    summary: Full gates green
    type: runtime
    pass_when: |
      `cargo fmt --all --check`; both clippy invocations with `-D warnings`;
      `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps -p molcrafts-molrs --all-features`;
      `cargo test -p molcrafts-molrs --features full --no-fail-fast` → 0 failed;
      and molrs-python 510 passed after a wheel rebuild.
    status: verified
    last_checked: 2026-07-15
    evidence: |
      fmt / clippy x2 / rustdoc all exit 0. molrs 1973 passed / 0 failed; molrs-python 510 passed after a
      wheel rebuild.
---

# Acceptance criteria

- **ac-001 / ac-002** 是 owner 的裁决落地：MMFF 不该拥有内核。它的静电是**通用库仑的一次参数化**。而"合并两个内核=二选一"是错的——**两个常数都对，力场说了算**。
- **ac-003 是本 spec 真正的价值**。它关掉的那个后门比原缺陷更隐蔽：内核在力场没给数据时**用自己硬编码的副本**，算出**对的数、错的理由**，18/19 个测试照样绿。
- **ac-004**：`ff/constants.rs` 里的常数**声称自己是宇宙的属性**；Halgren 的 332.0716 是**一个力场的属性**。放错地方的常数，是下一个"内核持有着没人交给它的数据"。
- **ac-006**：过期的文档已经在这条链上误导过三次（假的失败原因注释、"resolver 应随表搬"、CLAUDE.md）。**假文档比没文档更坏。**
