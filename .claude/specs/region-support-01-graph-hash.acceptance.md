---
slug: region-support-01-graph-hash
criteria:
  - id: ac-001
    summary: structural_hash 同构不变（节点顺序无关）
    type: code
    pass_when: |
      对一个图与其"节点顺序打乱的同构副本"，structural_hash() 相等；is_isomorphic()==True。
      Python：atomistic.structural_hash() 返回 u64；canonical_order() 返回 handle 列表。
    status: pending
  - id: ac-002
    summary: 相同 junction 哈希相等，不同结构哈希不同
    type: code
    pass_when: |
      两份相同局部环境（同一 repeat junction）structural_hash 相等；
      改一个原子 element / 电荷 / 芳香标志 或一根键的 order → 哈希改变；
      两个非同构图 is_isomorphic()==False。
    status: pending
  - id: ac-003
    summary: canonical_order 给出一致的节点双射
    type: code
    pass_when: |
      两个同构图的 canonical_order 诱导一致的节点对应（按序配对后 element/degree/charge 一致）。
    status: pending
  - id: ac-004
    summary: CG（bead 图）可哈希
    type: code
    pass_when: |
      CoarseGrain bead 图按 bead type + 拓扑哈希；同一 MolGraph 原语服务 AA 与 CG。
    status: pending
  - id: ac-005
    summary: 复用现有内核、无回归
    type: code
    pass_when: |
      graph_hash 建在 MolGraph 上、复用 topo/neighbor 内核，不改 SMARTS matcher/typifier；
      cargo test --all-features 全绿（含 OPLS/MMFF parity 无回归）。
    status: pending
  - id: ac-006
    summary: 质量闸：fmt/clippy/test 全绿
    type: runtime
    pass_when: |
      `cargo fmt --all --check`、`cargo clippy -- -D warnings`、`cargo check`、
      `cargo test --all-features` 全部 exit 0；molrs.Atomistic.structural_hash 在 Python 可用。
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003**: WL 结构哈希同构不变、对结构/标签变化敏感、canonical order 给一致双射——去重键的正确性核心。
- **ac-004**: AA+CG 共用同一 MolGraph 原语。
- **ac-005 / ac-006**: 复用内核、不改 matcher、无回归 + molrs 质量闸。
