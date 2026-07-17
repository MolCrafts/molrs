---
title: "在 CI 里跳过自己的测试 = 从不运行的测试"
slug: vacuous-green-tests
status: approved
created: 2026-07-15
---

# 缺输入就静默变绿的测试

## Summary

`chem-perceive-15-final-acceptance` 的 `no_test_returns_green_when_its_input_is_absent` 抓到：

`molrs/tests/ff/readers/opls.rs:36` —— `reads_real_molpy_oplsaa` 在 molpy 的 `oplsaa.xml` 缺失时**打印 "skipping" 然后 `return`**。

**在 CI 里，它什么都不断言，却计入覆盖率。**

## Domain basis

这条链已经为同一个道理立过一次规矩。`chem-perceive-14` 的 owner 裁决：

> **"ci 不要和 AMBERHOME 有任何牵连，只在实施过程中验证一次！"**

当时的落地是：**依赖 `$AMBERHOME` 的测试全部删除，不是 skip。**理由写进了验收：

> **一个在 CI 里 skip 掉自己的测试，就是一个从不运行的测试；它买到的是覆盖率的样子，不是覆盖率本身。**

`reads_real_molpy_oplsaa` 是同一个形状，只是换了个外部依赖（molpy 的 XML 而不是 AmberTools）。

## Design

三选一，**必须明确选一个并说明理由**：

1. **删除** —— 如果这个测试的价值不足以支撑维护一个外部依赖。
2. **把输入 vendor 进仓库** —— 如果它真的重要，就别让它依赖别人的工作区。
3. **变成 `#[ignore]`** —— 这样它至少**诚实地报告自己没跑**，而不是伪装成绿的。

**不允许的**：保留 `if !exists { println!("skipping"); return; }`。

## Tasks

- [ ] Decide and justify: delete / vendor the input / mark `#[ignore]`. State the reason in the code.
- [ ] Sweep the test tree for the same shape — this gate found one; there may be more
- [ ] `architecture_gate::no_test_returns_green_when_its_input_is_absent` goes green

## Out of scope

- `#[ignore]`d tests that are honestly marked (they report themselves)
