---
slug: release-0-14-08-ship-molrs
created: 2026-08-25
criteria:
  - id: ac-001
    summary: v0.14.0 is tagged on master
    type: code
    pass_when: "`git tag --points-at master` includes v0.14.0 and master contains the 0.14 chain commits."
    status: pending
  - id: ac-002
    summary: all five publish jobs are green
    type: runtime
    pass_when: |
      The publish.yml run for tag v0.14.0 shows publish-molrs, publish-wasm,
      build-python, build-python-pyodide and publish-python all succeeded.
    status: pending
  - id: ac-003
    summary: 0.14.0 resolves from all three registries
    type: runtime
    pass_when: |
      molcrafts-molrs 0.14.0 resolves on crates.io, @molcrafts/molrs 0.14.0 on
      npm, and molcrafts-molrs 0.14.0 downloads from PyPI.
    status: pending
  - id: ac-004
    summary: molnex carries the release wheel only
    type: runtime
    pass_when: |
      molnex .wheels-gh200/ contains a molcrafts_molrs-0.14.0-*.whl and no file
      matching *0.13.2.dev1*.
    status: pending
  - id: ac-005
    summary: aarch64 venv imports the released build
    type: runtime
    pass_when: |
      After gh200_bootstrap.sbatch, `python -c "import molrs, molrs.md"` in the
      aarch64 venv succeeds against the 0.14.0 wheel.
    status: pending
  - id: ac-006
    summary: release.md records the shipped tag
    type: docs
    pass_when: ".claude/notes/release.md v0.14.0 section carries the actual tag date and publish outcome."
    status: pending
  - id: ac-007
    summary: ship checklist is committed
    type: runtime
    pass_when: |
      regressions/release-0-14-08-ship-molrs.md names the tag, the five jobs,
      the wheel filename pattern and the molrs-before-molpy ordering constraint.
    status: pending
out_of_scope:
  - any code change
  - all molpy work
---

# Acceptance — release-0-14-08-ship-molrs

molrs 0.14.0 已 tag、已发布、下游 wheel 目录干净——molpy 侧工作的前置条件成立。
