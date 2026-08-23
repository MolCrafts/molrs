---
slug: link-mode-static-default-02-ci-gate
criteria:
  - id: ac-001
    summary: The static link form has a pre-push hook and it passes
    type: runtime
    pass_when: |
      REVISED 2026-08-23: originally written as a runner observation, which was
      the wrong instrument — this repo's `.pre-commit-config.yaml` is the single
      source and `ci.yml` mirrors it (header line 1-2), so the gate must be
      observable locally without pushing anything.

      `prek run --all-files --hook-stage pre-push` runs a `link-static` hook
      that exits 0, and its output contains the regression's line
      `libmolrs_ffi dynamic-link entry count == 0`. The hook body passes no
      build flag and sets no environment variable of any kind — that absence is
      the proposition under test, so a hook that "helpfully" pinned anything
      would make a green run prove something weaker.
      VERIFIED 2026-08-23 via `prek ... link-static --verbose`: exit 0,
      output `libmolrs_ffi dynamic-link entry count == 0` plus the regression's
      OK line. Full `prek run --all-files --hook-stage pre-push`: 12/12 hooks
      Passed, exit 0.
    status: verified
    last_checked: 2026-08-23
  - id: ac-002
    summary: The dynamic opt-in hook still passes, cross-repo
    type: runtime
    pass_when: |
      REVISED 2026-08-23: see ac-001 — observed through prek, not a runner.

      In the same `prek run --all-files --hook-stage pre-push` sweep, the
      `verify-shared-dylib` hook exits 0: a `libmolrs_ffi` entry in BOTH
      molrs._lib and molpack.molpack, and ONE identical sha256 printed after
      the molrs wheel build and after the molpack wheel build (a path
      comparison is vacuous; the sha is the identity proof).
      VERIFIED 2026-08-23 via `prek ... verify-shared-dylib --verbose`: exit 0,
      `sha256 after molrs wheel` and `sha256 after molpack wheel` both
      9ba2c3da3613dd1fd7c173a186134ee9e8d02e30c856bd2523d5278cea50fed5,
      then `verify-shared-dylib: OK`.
    status: verified
    last_checked: 2026-08-23
  - id: ac-003
    summary: The dynamic gate demonstrably bites when its opt-in is removed
    type: runtime
    pass_when: |
      REVISED 2026-08-23: performed locally on a dirty tree instead of a
      throwaway branch — same experiment, no push, and the result is reverted
      immediately and never committed.

      With `DYN_RUSTFLAGS` removed from its two maturin invocations in
      scripts/verify-shared-dylib.sh, the gate must FAIL at
      `FAIL — no libmolrs_ffi dynamic-link entry`. Record the observed exit
      code and that failure line, then `git checkout -- scripts/…` and confirm
      the gate is green again. Without this, a passing ac-002 proves only that
      the gate ran, not that it can fail.
      VERIFIED 2026-08-23: with `RUSTFLAGS="$DYN_RUSTFLAGS"` stripped from both
      maturin calls, the hook exited 1 at `FAIL — no libmolrs_ffi dynamic-link
      entry`, printing `observed ... <none>` for BOTH extensions against
      `expected one libmolrs_ffi entry in EACH extension`. Restored with
      `git checkout --`; diff empty, both call sites back. Never committed.
    status: verified
    last_checked: 2026-08-23
  - id: ac-004
    summary: link-form is wired into the CI orchestrator, not a new top-level workflow
    type: code
    pass_when: |
      .github/workflows/ci.yml contains a fifth entry `link-form:` with a
      single `uses: ./.github/workflows/ci-link-form.yml` line and no `steps:`;
      ci-link-form.yml declares only `on: {workflow_call, workflow_dispatch}`.
    status: verified
    last_checked: 2026-08-23
  - id: ac-005
    summary: molpack gains a non-PR link-dynamic job reusing the sibling script
    type: code
    pass_when: |
      molpack/.github/workflows/ci.yml has a `link-dynamic:` job after
      `regression:` with `if: github.event_name != 'pull_request'`, checkouts
      into `molrs`/`molpack`, `workspaces: "molpack -> ../molrs/target"`,
      no CARGO_TARGET_DIR, and a step invoking
      molrs/scripts/verify-shared-dylib.sh; molpack adds no scripts/ directory.
    status: verified
    last_checked: 2026-08-23
  - id: ac-006
    summary: MOLRS_GIT_REF landing-order precondition is recorded
    type: docs
    pass_when: |
      molpack/.github/workflows/ci.yml states above its env block that
      MOLRS_GIT_REF must never point at a molrs ref predating
      link-mode-static-default-01-invert, and docs/interop.md notes the dynamic
      form is guarded by both the CI Link Form dynamic job and the pre-push
      hook.
    status: pending
    note: |
      CONTENT IS PRESENT AND CHECKED — this stays `pending` only because it is a
      `docs` criterion, which /mol:impl does not write back (that is /mol:close's
      job), and close cannot run while ac-001..003 are blocked on an
      unauthorised push. Verified 2026-08-23:
      molpack/.github/workflows/ci.yml:13 carries the "Landing-order
      precondition: MOLRS_GIT_REF must NEVER point at a molrs ref predating
      link-mode-static-default-01-invert" comment above the env block, and
      docs/interop.md:314 carries the CI-Link-Form + pre-push double-guard
      sentence. Do NOT infer from `pending` that the text is missing — an
      evaluator did exactly that on 2026-08-23 and was wrong.
---

# Acceptance criteria

ac-003 是这条链里唯一的反向门:没有它,`dynamic:` job 会是一个绿了也证明不了
任何事的 job —— 恰是本仓禁止的"不可能失败的测试"。
