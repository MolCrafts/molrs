---
slug: link-mode-static-default-02-ci-gate
criteria:
  - id: ac-001
    summary: CI Link Form static job proves the zero-argument default
    type: runtime
    pass_when: |
      A workflow_dispatch run of .github/workflows/ci-link-form.yml shows job
      `static` green, its log printing the regression's
      "libmolrs_ffi dynamic-link entry count == 0" line, and the job body
      contains no rustflags/env/--config of any kind.
    status: pending
    blocked_by: |
      Runner-observed only. Satisfying this requires pushing a branch and
      running the workflow on a GitHub runner, which was NOT authorised in
      this session (nothing in either repo has been pushed). Everything
      locally checkable about the job was verified instead — see ac-004 /
      ac-005 — but the shape of a job is not evidence that it runs green, so
      this criterion is deliberately left unverified rather than attested.
  - id: ac-002
    summary: CI Link Form dynamic job runs the cross-repo gate green
    type: runtime
    pass_when: |
      The same run shows job `dynamic` green: molrs checked out at path
      `molrs`, molpack at path `molpack`, and the log printing a libmolrs_ffi
      entry for both extensions plus one identical sha256 after the molrs and
      molpack wheel builds.
      REF REQUIREMENT (2026-08-23, grill): molrs CI triggers on `dev` and
      `master`, and molcrafts PRs land on `dev` first — so the molpack
      checkout must pin a ref that actually carries
      link-mode-static-default-01-invert (`dev` during the landing window),
      not a bare `master`. A `master` pin would silently validate a molpack
      that predates 01. The chosen ref is stated in a comment on the checkout
      step.
    status: pending
    blocked_by: |
      Runner-observed only. Satisfying this requires pushing a branch and
      running the workflow on a GitHub runner, which was NOT authorised in
      this session (nothing in either repo has been pushed). Everything
      locally checkable about the job was verified instead — see ac-004 /
      ac-005 — but the shape of a job is not evidence that it runs green, so
      this criterion is deliberately left unverified rather than attested.
  - id: ac-003
    summary: The dynamic gate demonstrably bites when the flags are removed
    type: runtime
    pass_when: |
      On a throwaway branch with the two --config flags deleted from
      scripts/verify-shared-dylib.sh, the `dynamic` job fails at
      "FAIL — no libmolrs_ffi dynamic-link entry"; the run URL is recorded and
      the branch is not merged.
    status: pending
    blocked_by: |
      Runner-observed only. Satisfying this requires pushing a branch and
      running the workflow on a GitHub runner, which was NOT authorised in
      this session (nothing in either repo has been pushed). Everything
      locally checkable about the job was verified instead — see ac-004 /
      ac-005 — but the shape of a job is not evidence that it runs green, so
      this criterion is deliberately left unverified rather than attested.
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
