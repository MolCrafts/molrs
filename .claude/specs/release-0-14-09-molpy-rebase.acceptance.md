---
slug: release-0-14-09-molpy-rebase
created: 2026-08-25
criteria:
  - id: ac-001
    summary: molpy branch is rebuilt from upstream/master v0.13.1
    type: code
    pass_when: |
      The molpy 0.14 branch's merge-base with upstream/master is the v0.13.1
      commit, and it is not descended from the 0.7.0-based dev branch.
    status: pending
  - id: ac-002
    summary: all eight dev-only commits are present
    type: code
    pass_when: |
      Every commit subject in the range 7888e335..44adb594 appears in the new
      branch's log, or is recorded as an intentional skip with a reason.
    status: pending
  - id: ac-003
    summary: molpy consumes molrs Key objects directly
    type: runtime
    pass_when: |
      A molpy test reads and writes a Frame column via a molrs Key object with
      results bit-identical to the former _key_str path; if _key_str lost every
      caller it no longer exists in molpy/src.
    status: pending
  - id: ac-004
    summary: version and pin are 0.14
    type: code
    pass_when: |
      molpy version.py reads 0.14.0 and pyproject.toml declares
      molcrafts-molrs>=0.14.0,<0.15 (the ==0.7.0 dead pin is gone).
    status: pending
  - id: ac-005
    summary: the release-order guard is executable
    type: runtime
    pass_when: |
      molpy tests/test_molrs_pin.py fails when the installed molrs is older
      than 0.14.0, and passes against the released wheel.
    status: pending
  - id: ac-006
    summary: rebase regression reproduces its goldens
    type: runtime
    pass_when: |
      `python regressions/release-0-14-09-molpy-rebase.py` in molpy exits 0,
      uses only public molpy API, imports no third-party scientific package,
      and matches its embedded column and version goldens.
    status: pending
out_of_scope:
  - full-surface sinking and compute Protocol adoption
  - docs / spelling sweep
  - joint smoke and molpy tag
  - any molrs change
---

# Acceptance — release-0-14-09-molpy-rebase

molpy 0.14 分支从 upstream v0.13.1 干净重开、8 个提交在案、keys 兼容层收敛、pin 与版本到位，且发布顺序由一条会红的测试守着。
