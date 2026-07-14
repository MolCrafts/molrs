---
slug: chem-perceive-15-final-acceptance
created: 2026-07-14
criteria:
  - id: ac-001
    summary: ONE place, ONE form — and the word `generated` is gone from every identifier
    type: code
    pass_when: |
      `molrs/src/ff/params/` is the ONLY home of parameter tables: flat, no subdirectory.
      `molrs/src/ff/params/generated/` does not exist. `molrs/src/ff/mmff/tables.rs` does not exist.
      `molrs/data/` does not exist. `grep -rn 'include_str!' molrs/src` = 0 hits.
      AND: `generated` / `generator` appear in ZERO identifiers (directory, file, module, test-fn or
      type names) under `molrs/src` and `molrs/tests`. Today there are ~130 sites, including a test file
      literally named `estimate_tables_generated.rs`.
      This is not fussiness. "generated" names how the tables ARRIVED, not what they ARE — naming a thing
      after its provenance welds an implementation detail onto the public surface. It is the same mistake
      as calling an 808-line ALGORITHM `params.rs`: once a name lies, every later reader is misled by it.
      Provenance belongs in the file's header doc ("emitted by scripts/gen_param_tables.py from
      AmberTools' .DAT/.DEF"), never in the name.
      The gate must not be able to exempt itself (assemble its needle with `concat!`).
    status: pending
    last_checked:
    evidence:

  - id: ac-002
    summary: Nothing parses parameter text at runtime
    type: code
    pass_when: |
      No `serde_json::from_str` / `from_slice` / `from_reader`, no XML parse, no `include_str!` is
      reachable from the force-field path. A malformed table is a COMPILE error, not a runtime one.
    status: pending
    last_checked:
    evidence:

  - id: ac-003
    summary: ONE perception layer, ONE interpolation seam, ONE MMFF path
    type: code
    pass_when: |
      (a) `molrs::perceive` sits above `core`, below `ff`/`io`/`conformer`; the `molrs::chem` alias is 0
      hits across ALL sibling workspaces.
      (b) `ParameterInterpolator` has exactly ONE implementor, `Parmchk2Estimator`. A different force
      field is a different `TypifierParameterContext`, NOT a second estimator stack. (This chain was
      already reworked once for exactly that sin.)
      (c) No bespoke MMFF energy layer, no `build_mmff_potentials` free function, no second classifier.
      `MMFF94Typifier().typify()` gives labels and charges; potentials go through the standard ForceField
      route, like every other force field.
    status: pending
    last_checked:
    evidence:

  - id: ac-004
    summary: A registered kernel constructor that ignores `tp` is not a Style
    type: runtime
    pass_when: |
      `ParamSource::PerInstance` is a first-class concept and the bidirectional gate holds: a ctor ignores
      its type-params IF AND ONLY IF it is registered PerInstance.
      The gate must match on SEMANTICS, not spelling. My own grep judged on the spelling `_tp` and MISSED
      `pme_ctor` and `pair_coul_cut_ctor`, which spell it `_type_params` and ignore it just as completely —
      8 violators, not the 6 I claimed. A grep finds spellings; a gate finds semantics.
    status: pending
    last_checked:
    evidence:

  - id: ac-005
    summary: The full chain runs end to end against REAL external oracles
    type: runtime
    pass_when: |
      SMILES/SDF -> Perceive -> AtdTypifier -> ChargeModel -> ForceField -> Potentials -> energy + forces,
      asserted against oracles that molrs did not produce:
      - antechamber (AmberTools25), 37 molecules: atom types for all 7 ATD sets, BCC / ABCG2 / Gasteiger
        charges, and the parmchk2 estimated terms
      - RDKit MMFF, 11 molecules: total energy AND the 7-term per-style breakdown
      Not "each stage passes its own test" — one chain, green from end to end.
    status: pending
    last_checked:
    evidence:

  - id: ac-006
    summary: Python reproduces Rust BIT-FOR-BIT, not approximately
    type: runtime
    pass_when: |
      The Python bindings return the same numbers as Rust — the same numbers, not close ones. dtype
      float64, no f32 anywhere, no renormalization. Charge conservation to 1e-12 (an f32 round-trip misses
      that by four orders of magnitude, while sailing through any 1e-4 value check against a 6-decimal
      oracle).
    status: pending
    last_checked:
    evidence:

  - id: ac-007
    summary: The REVERSE gates — every one of them caught a real defect on this chain
    type: runtime
    pass_when: |
      Each of these asserts that something is ABSENT. Forward assertions ("X exists") get fooled by
      "added it, but in the wrong place":
      - zero-charge molecules get EXACTLY 0.0 electrostatic energy (not "small")
      - on molecules with no delocalized N, MMFF94 and MMFF94s are BIT-IDENTICAL (else a "they differ"
        test can pass on a difference that does not exist)
      - benzene HAS impropers (it had ZERO before this chain — silently)
      - nitrate's three oxygens, and acetate's two, carry EQUAL charges (they differed by 0.2014 e)
      - the same molecule in different conformations gets the SAME charges (the point of equivalencing)
      - a bare sulfur is REFUSED by BCC (helium is not the witness — it types fine; molrs's own docs name
        bare sulfur, the one element whose BCC rules all require a bond)
    status: pending
    last_checked:
    evidence:

  - id: ac-008
    summary: No subset assertion survives without a stated reason
    type: code
    pass_when: |
      No test asserts on a hand-picked subset of its fixtures without saying IN THE TEST why the others are
      excluded — and "not yet implemented" is not a reason to exclude, it is a reason to fail.
      This is the single pattern behind the worst defect on this chain: `generic_path_total_energy_matches_
      rdkit` asserted on `["e_ethane"]` alone — one of exactly TWO fixtures whose MMFF charges are all zero,
      i.e. the ONE input class that could not expose the missing electrostatic term. Next to it sat a
      comment blaming the wrong cause, which had misdirected every reader for a month.
      Where a list can be directory-scanned, it MUST be: a subset assertion should not be something you
      justify, it should be something you cannot write.
    status: pending
    last_checked:
    evidence:

  - id: ac-009
    summary: Every gate has been PROVEN to bite
    type: runtime
    pass_when: |
      For each gate in this spec: the defect it guards against was temporarily introduced, the gate went
      RED, and it was removed. Recorded in the evidence.
      A gate that has never been red is indistinguishable from no gate. This chain already produced one
      (`bespoke_gate`, which pinned the very tree a later spec deleted) and one grep criterion that was
      simply wrong (`_tp` by spelling). Both looked like coverage.
    status: pending
    last_checked:
    evidence:

  - id: ac-010
    summary: The acceptance fixed NOTHING — every failure it found got its own spec
    type: code
    pass_when: |
      `git log` for this spec contains no production fix. If a gate found a real defect, work STOPPED, a
      spec was written, and it was fixed there.
      An acceptance that quietly repairs what it finds is not an acceptance — it is the last place a defect
      can hide, because the thing that would have reported it is the thing that swallowed it.
    status: pending
    last_checked:
    evidence:
---

# Acceptance criteria

这条链跑了 16 个 spec，**每个只验证自己那一块，没有任何一个验证过整体**。本 spec 是唯一一次整体验收。

- **ac-001 ~ ac-004** 把五条"只此一份"的架构承诺变成门禁。没有门禁的承诺不是承诺，是意图——而这条链上的每一个严重缺陷，都是从一个"大家都以为成立"的意图底下长出来的。
- **ac-005 / ac-006** 要求真值来自**外部** oracle（antechamber、RDKit），并且 Python 与 Rust **逐位**一致。断言自己算出来的东西，是这条链最早的病（BCC 键型感知、电荷等价化两个算法阶段整个缺失，而测试全绿）。
- **ac-007** 全部是**反向**断言。正向断言会被"加了但加错地方"骗过去；`ac-006`（零电荷分子静电必须恰好为 0）和苯的 improper（之前是**零个**，静默）都是这么抓到的。
- **ac-008** 针对这条链**最贵的那个教训**：测试选了一个不可能失败的输入，然后声称覆盖了。乙烷是仅有的两个电荷全为零的分子之一——唯一一类结构上无法暴露静电缺失的输入，而它是**唯一被断言的那个**。
- **ac-009**：**一道从没红过的门禁，和没有门禁没有区别。**
- **ac-010**：验收里不许偷偷修东西。**一个会自我修复的验收，是缺陷最后的藏身之处**——因为本该报告它的那个东西，把它吞了。
