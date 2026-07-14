---
slug: vacuous-green-tests
created: 2026-07-15
criteria:
  - id: ac-001
    summary: No test returns green when its input is absent
    type: code
    pass_when: |
      `architecture_gate::no_test_returns_green_when_its_input_is_absent` is green.
      `readers/opls.rs:36`'s `if !exists { println!("skipping"); return; }` is gone — replaced by a
      deletion, a vendored input, or an honest `#[ignore]`, with the reason stated in the code.
      A test that skips itself in CI is a test that never runs; it buys the APPEARANCE of coverage and
      none of the substance. This chain already ruled on exactly this shape once
      (chem-perceive-14: every $AMBERHOME-dependent test was DELETED, not skipped).
    status: pending
    last_checked:
    evidence:

  - id: ac-002
    summary: The whole test tree is swept for the same shape
    type: code
    pass_when: |
      The gate found one. The sweep confirms whether there are more, and reports the count.
    status: pending
    last_checked:
    evidence:
---

# Acceptance criteria

**一个在 CI 里 skip 掉自己的测试，就是一个从不运行的测试。** 这条链已经为同一个道理裁决过一次（`$AMBERHOME` 的测试全部**删除**，不是 skip）。这是同一个形状，换了个外部依赖。
