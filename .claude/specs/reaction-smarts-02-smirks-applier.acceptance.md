---
slug: reaction-smarts-02-smirks-applier
criteria:
  - id: ac-001
    summary: reaction SMARTS 解析（>> 切分 + LHS 多组分 + 两侧映射）
    type: code
    pass_when: |
      Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")：reactant_patterns 为 2 个组分
      ([N;H2:1] 与 [C:2](=O)OC)，product 解析成功，两侧映射号 1/2 配对；
      "A>agent>B" 三段式解析不崩（agent 忽略）。
    status: pending

  - id: ac-002
    summary: Transform 编译符合 Daylight SMIRKS 语义
    type: code
    pass_when: |
      "[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O" 编译出 form_bonds 含 (1,2)、未映射酯 O+烷基进 delete、无虚假 add；
      "[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]" 编译出 set_order (1,2)=1.0 + form (2,3)；
      只出现在一侧的映射号（非未映射原子）触发错误（Daylight 配对要求）。
    status: pending

  - id: ac-003
    summary: 单 occurrence 就地应用正确 + 拓扑刷新（复用核心 fn）
    type: code
    pass_when: |
      Reaction(...).apply(mol, {1:n,2:c,...})：新 N–C 键存在、未映射离去原子已删、
      原子数 = 原 − 离去数；generate_topology 后新键周围 angle/dihedral 再生；
      apply 经 core add_bond/remove_bond/remove_atom/add_atom + generate_topology + perceive_aromaticity，
      未在 reaction.rs 里重造键/拓扑生成。
    status: pending

  - id: ac-004
    summary: 未映射 RHS 原子被新增（element/charge 取自模板，无坐标）
    type: code
    pass_when: |
      一条 RHS 含未映射新原子的反应：apply 后新原子按模板 element/charge 建好、连到正确保留原子、计入拓扑；
      新原子无坐标（交调用方后置几何）。
    status: pending

  - id: ac-005
    summary: PyReaction 暴露 + 不改核心匹配、无回归
    type: code
    pass_when: |
      molrs.Reaction(smirks) 在 Python 可用：reactant_patterns/forming_bonds/apply 正常；
      chain-01 的 SMARTS 匹配器与 OPLS/MMFF 分型器 parity 测试无回归。
    status: pending

  - id: ac-006
    summary: 质量闸：fmt/clippy/test 全绿
    type: runtime
    pass_when: |
      `cargo fmt --all --check`、`cargo clippy -- -D warnings`、`cargo check`、
      `cargo test --all-features` 全部 exit 0；Python import molrs; molrs.Reaction 可用。
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002**: Daylight reaction SMARTS 解析 + Transform 由映射号 diff 编译（严格 SMIRKS transform 语义）。
- **ac-003 / ac-004**: 单 occurrence 就地改图正确（成键/删离去/加新原子/拓扑刷新），且**复用**核心图编辑原语、不重造。
- **ac-005 / ac-006**: 暴露 Python、不改核心匹配、无回归 + molrs 质量闸。
