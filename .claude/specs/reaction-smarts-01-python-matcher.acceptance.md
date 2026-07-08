---
slug: reaction-smarts-01-python-matcher
criteria:
  - id: ac-001
    summary: SmartsPattern 暴露给 Python，映射号 capture 正确
    type: code
    pass_when: |
      molrs.SmartsPattern("[C:1][O:2][H:3]").find_matches_mapped(atomistic) 返回 list[dict[int,int]]，
      每个 dict 形如 {1:<C handle>,2:<O handle>,3:<H handle>}，handle 对应原子 element 为 C/O/H；
      对不含该基团的分子返回空列表。SmartsPattern("[N;H2:1]") 只匹配伯胺。
    status: verified
    last_checked: 2026-07-05

  - id: ac-002
    summary: Daylight 原子映射语义——molecule SMARTS 中映射被忽略、不改匹配
    type: code
    pass_when: |
      SmartsPattern("[C:1]").find_matches(mol) 与 SmartsPattern("[C]").find_matches(mol)
      返回相同的原子集合（映射号不加匹配约束，符合 Daylight）；map_label(0)==1、num_query_atoms 正确。
    status: verified
    last_checked: 2026-07-05

  - id: ac-003
    summary: PyAtomistic 图编辑便利方法（remove/set_bond_order/copy）
    type: code
    pass_when: |
      remove_atom(h) 删原子并级联删除其关联键/角度；remove_bond(h) 删键；
      set_bond_order(h, 2.0) 改键级；copy() 返回独立图——改副本不影响原图（原子/键计数不变）。
      均转发到已有 core fn，行为与 Rust 侧一致。
    status: verified
    last_checked: 2026-07-05

  - id: ac-004
    summary: 复用现有引擎，不改核心匹配算法，无回归
    type: code
    pass_when: |
      新增仅为 PyO3 绑定：core/chem/smarts/ 的 matcher/parser/ast 未改；
      cargo test --all-features 全绿（含 OPLS/MMFF 分型器 parity 测试无回归）。
    status: verified
    last_checked: 2026-07-05

  - id: ac-005
    summary: 质量闸：fmt/clippy/test 全绿
    type: runtime
    pass_when: |
      `cargo fmt --all --check`、`cargo clippy -- -D warnings`、`cargo check`、
      `cargo test --all-features` 全部 exit 0；Python 侧 import molrs; molrs.SmartsPattern 可用。
    status: verified
    last_checked: 2026-07-05
---

# Acceptance criteria

- **ac-001 / ac-002**: 把已有的、已支持原子映射的核心 SMARTS 引擎暴露给 Python，映射号→handle 结果正确，
  且严格遵守 Daylight "molecule SMARTS 中映射被忽略"语义。这是 molpy 交联层的匹配底座。
- **ac-003**: 补齐 Python 侧缺的图编辑便利方法（remove/set_bond_order/copy），供 SMIRKS 应用器（02）与
  molpy 交联层使用。
- **ac-004 / ac-005**: 纯绑定、不改核心算法、无回归 + molrs 质量闸。
