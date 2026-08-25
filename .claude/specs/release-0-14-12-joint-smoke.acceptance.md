---
slug: release-0-14-12-joint-smoke
created: 2026-08-25
criteria:
  - id: ac-001
    summary: every molpy submodule imports against molrs 0.14.0
    type: runtime
    pass_when: |
      A pkgutil.walk_packages-driven test imports every molpy submodule with
      zero failures in an environment holding the released molcrafts-molrs
      0.14.0 wheel.
    status: pending
  - id: ac-002
    summary: the experimental warning stays scoped
    type: runtime
    pass_when: |
      `import molpy` records zero FutureWarning; the first molpy.md attribute
      access records exactly one mentioning "experimental in 0.14".
    status: pending
  - id: ac-003
    summary: the molnex chain imports on aarch64
    type: runtime
    pass_when: |
      The molnex chain import smoke runs to exit code 0 in the aarch64 venv
      built from .wheels-gh200 with the 0.14.0 wheel.
    status: pending
  - id: ac-004
    summary: molpy 0.14.0 is tagged after molrs
    type: code
    pass_when: |
      molpy master carries tag 0.14.0 and its commit date is later than the
      molrs v0.14.0 tag date.
    status: pending
  - id: ac-005
    summary: both spec indexes record the chain outcome
    type: docs
    pass_when: |
      .claude/specs/INDEX.md in molrs and molpy each list the release-0-14
      chain with its final status; .claude/notes/notes.md md entry is no longer
      marked provisional without a stated reason.
    status: pending
  - id: ac-006
    summary: joint regression proves end-to-end usability from molpy alone
    type: runtime
    pass_when: |
      `python regressions/release-0-14-12-joint-smoke.py` in molpy exits 0
      importing only molpy, imports no third-party scientific package, and
      matches its embedded build / analysis / 5-step MD goldens.
    status: pending
out_of_scope:
  - design changes
  - 0.15 roadmap items
---

# Acceptance — release-0-14-12-joint-smoke

molpy 新分支在 molrs 0.14.0 正式件上全量导得进、molnex 链跑得通、警告不外溢，molpy 0.14.0 在 molrs 之后打出，两仓 spec 索引收口。
