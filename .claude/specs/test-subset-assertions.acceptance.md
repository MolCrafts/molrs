---
slug: test-subset-assertions
created: 2026-07-15
criteria:
  - id: ac-001
    summary: Every computable fixture partition is COMPUTED, not hand-written
    type: code
    pass_when: |
      `architecture_gate::no_test_asserts_on_a_subset_of_its_fixtures` is green. The four hand-written
      subsets (energy.rs:781, energy.rs:979 `S_NAMES`, mmff_variant.rs:69/73) are replaced by predicates
      derived from the fixtures themselves.
      A subset assertion should not be something you justify — it should be something you CANNOT WRITE.
    status: pending
    last_checked:
    evidence:

  - id: ac-002
    summary: The new partition actually catches caffeine
    type: runtime
    pass_when: |
      MMFF94 and MMFF94s give DIFFERENT total energies on `e_caffeine`, asserted. This assertion does not
      exist today.
      That is the whole point: `e_caffeine` and `e_big` DO carry a delocalized nitrogen and appeared in
      NEITHER hand-written list. Going green by rewriting the lists without covering them would be a
      cosmetic fix to a real hole.
    status: pending
    last_checked:
    evidence:

  - id: ac-003
    summary: Any genuinely uncomputable subset states its reason IN THE TEST
    type: code
    pass_when: |
      If a predicate truly cannot be derived, the test says why the others are excluded — and
      "not yet implemented" is a reason to FAIL, not to exclude.
    status: pending
    last_checked:
    evidence:
---

# Acceptance criteria

这是这条链上**最贵的那个教训**的复发：`generic_path_total_energy_matches_rdkit` 曾只断言乙烷——**仅有的两个电荷全为零的分子之一**，唯一一类不可能暴露"缺少静电项"的输入。150 kcal/mol 的洞因此活了一个月。

**ac-002 是关键**：光把名单改成计算的还不够，必须证明**新分区真的把 caffeine 收进去了**——否则只是换了个写法，洞还在。
