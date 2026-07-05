---
slug: region-support-02-reaction-touched
criteria:
  - id: ac-001
    summary: apply 返回 touched 原子 handle 集
    type: code
    pass_when: |
      Reaction("[N;H2:1].[C:2](=O)OC>>[N:1][C:2]=O").apply(mol, binding) 返回 list[int]，
      含成键端点 N/C 与离去基团的存活邻居；被删原子自身 handle 不在集合内；结果去重。
    status: verified
  - id: ac-002
    summary: 加原子/改键级也计入 touched
    type: code
    pass_when: |
      含新增 RHS 原子的反应，touched 含新原子 handle；
      thiol-ene "[C:1]=[C:2].[S;H1:3]>>[C:1][C:2][S:3]" 的 touched 含两个碳 + 硫。
    status: verified
  - id: ac-003
    summary: 无回归（reaction-smarts-02 + molpy crosslink 不受影响）
    type: code
    pass_when: |
      apply 返回类型从 None 变 list[int]；reaction-smarts-02 测试更新后全绿；
      molpy crosslink 忽略返回值，行为不变。
    status: verified
  - id: ac-004
    summary: 质量闸：fmt/clippy/test 全绿
    type: runtime
    pass_when: |
      `cargo fmt --all --check`、`cargo clippy -- -D warnings`、`cargo check`、
      `cargo test --all-features` 全部 exit 0；Python 侧 apply 返回 list[int]。
    status: verified
---

# Acceptance criteria

- **ac-001 / ac-002**: apply 报告 touched 种子原子（成键端点/加原子/删原子存活邻居/改属性原子），供 molpy 扩成 retype-safe 球。
- **ac-003 / ac-004**: 返回类型变更无回归 + molrs 质量闸。
