---
slug: release-0-14-07-surface-hygiene
created: 2026-08-25
criteria:
  - id: ac-001
    summary: every shipped example executes
    type: runtime
    pass_when: |
      A test globbing molrs-python/examples/*.py runs each script to exit code
      0, and the same glob-based step runs in .github/workflows/ci-python.yml.
    status: pending
  - id: ac-002
    summary: README and quickstart code matches the landed API
    type: docs
    pass_when: |
      No README or quickstart snippet uses Conformer.generate as a single
      value, references an undefined nlist, or imports the symbol that made
      embed_water.py raise ImportError.
    status: pending
  - id: ac-003
    summary: three error messages carry their context
    type: runtime
    pass_when: |
      read_pdb on a missing path raises with the filename in the message; a
      PyO3 type error names the offending argument; Block column lookup failure
      lists candidate column names.
    status: pending
  - id: ac-004
    summary: no binder rustdoc claims f32
    type: docs
    pass_when: |
      A grep gate over molrs-python/src and molrs-wasm/src finds no rustdoc
      line claiming f32 precision, consistent with CLAUDE.md F = f64.
    status: pending
  - id: ac-005
    summary: every "Exposed as molrs.X" claim matches the real path
    type: docs
    pass_when: |
      A predicate-based test cross-checks each "Exposed ... as `molrs.`" doc
      comment against the class's documented Python path; zero mismatches.
    status: pending
  - id: ac-006
    summary: the wasm NeighborQuery item is deferred to 0.15 on the record
    type: docs
    pass_when: |
      The .claude/notes/notes.md binder-surface-symmetry entry records a dated
      deferral naming target version 0.15, the reason (no wasm consumer yet),
      and the four molrs consumers that keep NeighborQuery alive
      (compute/hbond/detect.rs, compute/rdf, compute/dynamics/van_hove,
      ff/potential/soft.rs); a test asserts those four imports still resolve, so
      deleting NeighborQuery turns the gate red.
    status: pending
  - id: ac-007
    summary: build artefact trees are ignored
    type: code
    pass_when: |
      `git check-ignore target-aarch64` succeeds in molrs and
      `git check-ignore benchmarks/md` succeeds in molpy.
    status: pending
  - id: ac-008
    summary: blueprint and CLAUDE.md describe the current tree
    type: docs
    pass_when: |
      .claude/notes/architecture.md lists molrs/src/md; neither it nor
      CLAUDE.md mentions a FieldSpec layer, legacy/op/embed/tool packages,
      Frame.metadata, MolRec, or a kspace style category.
    status: pending
  - id: ac-009
    summary: hygiene regression checklist is committed
    type: runtime
    pass_when: |
      regressions/release-0-14-07-surface-hygiene.md lists each gate with its
      predicate and expected result, reproducible without this spec.
    status: pending
  - id: ac-010
    summary: full gate green after the sweep
    type: runtime
    pass_when: |
      cargo lib tests, cargo doc tests and the molrs-python tox py env all pass.
    status: pending
out_of_scope:
  - new features / API expansion
  - wasm NeighborQuery symmetric gate (0.15, deferral recorded)
  - KSpace surface removal and PME as a pair style (spec 14)
  - MolRec / zarr public naming (spec 13)
  - molpy docs sweep
  - tag / publish
---

# Acceptance — release-0-14-07-surface-hygiene

0.14 发布前，用户第一眼看到的表面（README、examples、错误信息、rustdoc）不再撒谎，构建产物不进版本库，蓝图描述的是这棵树；唯一没做完的事（wasm 对称门）带着日期、理由和"它还活着"的证据躺在笔记里，而不是悬着。
