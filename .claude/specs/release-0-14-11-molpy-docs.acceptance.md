---
slug: release-0-14-11-molpy-docs
created: 2026-08-25
criteria:
  - id: ac-001
    summary: every documented typifier name is real
    type: docs
    pass_when: |
      A predicate test resolves every \w+Typifier spelling in molpy docs,
      README, CHANGELOG and notes against the runtime class set; zero unknown
      spellings remain (was 98 hits across 35 files).
    status: pending
  - id: ac-002
    summary: no user-visible molrs spelling in molpy
    type: docs
    pass_when: |
      An ast-based scan of molpy/src docstrings and string literals plus a scan
      of docs/** finds zero "molrs" occurrences; engine-side import statements
      are excluded by construction, not by an allowlist of files.
    status: pending
  - id: ac-003
    summary: a molpy migration guide exists in both languages
    type: docs
    pass_when: |
      docs/getting-started/migration-0-14.md and docs/zh/getting-started/
      migration-0-14.md exist, cover typifier / md / keys / Neighbors /
      metadata→meta / compute contract conformance, and are both in zensical
      nav.
    status: pending
  - id: ac-004
    summary: the two doc trees stay in step
    type: runtime
    pass_when: "A test asserts docs/ and docs/zh/ expose identical relative page sets."
    status: pending
  - id: ac-005
    summary: molpy blueprint reflects the fused framework
    type: docs
    pass_when: |
      molpy CLAUDE.md and .claude/notes/architecture.md describe compute as
      conforming to the molrs Compute Protocol, use meta (not metadata), and
      mention the md surface.
    status: pending
  - id: ac-006
    summary: spelling regression checklist is committed
    type: runtime
    pass_when: |
      regressions/release-0-14-11-molpy-docs.md in molpy states each gate's
      predicate and its expected zero-hit result, reproducible without this spec.
    status: pending
out_of_scope:
  - code behaviour changes
  - joint smoke and molpy tag
  - any molrs change
---

# Acceptance — release-0-14-11-molpy-docs

用户读到的每一处 typifier 名字都真实存在，`molrs` 一词在用户可见面归零，中英双语各有一页 0.14 迁移指南，且三条门都是谓词式、不可靠手工名单维持。
