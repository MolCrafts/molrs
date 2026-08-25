---
slug: release-0-14-06-docs-migration
created: 2026-08-25
criteria:
  - id: ac-001
    summary: a 0.13 to 0.14 migration guide exists and is reachable
    type: docs
    pass_when: |
      molrs-python/site-src/getting-started/migration-0-14.md covers typifier
      paths/class names, md, units (molpy.UnitPreset replacing energy_to_md /
      preset_energy_to_md / kb_md / set_energy_scale), keys, Neighbors API and
      metadata→meta, each as an old→new code pair, and is listed in
      molrs-python/zensical.toml nav.
    status: pending
  - id: ac-002
    summary: a one-page md user guide teaches the structural contract
    type: docs
    pass_when: |
      molrs-python/site-src/guides/md.md walks UnitPreset → VerletSkin → LJCut
      → VelocityVerlet → advance_n, shows a custom force that inherits from
      nothing, states that Potential is a runtime_checkable Protocol, and is
      listed in the zensical nav.
    status: pending
  - id: ac-003
    summary: no user-visible molrs spelling in site examples
    type: docs
    pass_when: |
      A directory-scanning test over molrs-python/site-src finds zero
      occurrences of "import molrs" or "molrs." inside python code blocks.
    status: pending
  - id: ac-004
    summary: analysis time is fs everywhere, with no ps dual
    type: docs
    pass_when: |
      .claude/notes/science.md states Time = fs with no ps dual, and no
      analysis-time ps claim survives in molrs-python/site-src or
      molrs-python/python/molrs/compute/.
    status: pending
  - id: ac-005
    summary: the guides teach no deleted symbol
    type: docs
    pass_when: |
      Neither new page mentions energy_to_md, preset_energy_to_md, kb_md,
      set_energy_scale, prec= or resolve_prec.
    status: pending
  - id: ac-006
    summary: doc gates are executable, not manual
    type: runtime
    pass_when: |
      molrs-python/tests/test_docs_gates.py passes under tox -e py and
      discovers its inputs by directory scan rather than a hand-written list.
    status: pending
  - id: ac-007
    summary: docs regression checklist is committed
    type: runtime
    pass_when: |
      regressions/release-0-14-06-docs-migration.md lists every grep predicate
      with its expected hit count and is reproducible without this spec.
    status: pending
out_of_scope:
  - README / examples executability and CI smoke
  - molpy docs and typifier spelling sweep
  - error-message improvements
---

# Acceptance — release-0-14-06-docs-migration

用户拿到 0.14 时有一页迁移指南与一页 md 指南可读，示例全拼 `molpy` 且自定义力不继承任何东西，分析时间单位只有 fs 一个说法，三件事都由可执行的门守着。
