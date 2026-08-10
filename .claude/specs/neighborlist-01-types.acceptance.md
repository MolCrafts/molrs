---
spec: neighborlist-01-types
created: 2026-08-10
criteria:
  - id: ac-001
    summary: 物化结果类型公开名为 Neighbors
    type: code
    pass_when: |
      molrs::spatial::neighbors 公开导出 Neighbors 与 NeighborsStorage；
      不再公开名为 NeighborList 的物化结果类型（总包名留给 02）。
    status: verified
    last_checked: 2026-08-10
    evidence: "grep: 0 hits for pub struct/type NeighborList in molrs/src; Neighbors/NeighborsStorage/NeighborPair exported in mod.rs"
  - id: ac-002
    summary: NeighborsStorage 使用 disp 字段且 FULL 表示列齐
    type: code
    pass_when: |
      NeighborsStorage 含 dist_sq: bool 与 disp: bool；
      FULL == {dist_sq:true, disp:true}；INDICES_ONLY 两者 false。
      rustdoc 写明 FULL 非双向 pair、disp 为非单位 MIC 位移。
    status: verified
    last_checked: 2026-08-10
    evidence: "mod.rs consts + rustdoc '# FULL is about columns, not about direction' + NeighborPair disp non-normalized note (documenter pass)"
  - id: ac-003
    summary: from_pairs 从 NeighborPair 流写表
    type: runtime
    pass_when: |
      Neighbors::from_pairs(已知 2 个 NeighborPair 的 iter, FULL, SelfQuery{..})
      得到 n_pairs==2，indices/dist_sq/disp 与输入一致（disp 分量误差 <= 1e-15）。
    status: verified
    last_checked: 2026-08-10
    evidence: "from_pairs_tests::from_pairs_full_roundtrip green"
  - id: ac-004
    summary: 可选列缺失时 accessor 返回 None
    type: runtime
    pass_when: |
      from_pairs(..., INDICES_ONLY, ..) 后 dist_sq() 与 disp() 均为 None；
      n_pairs()>0 时也不会返回全零假列。
    status: verified
    last_checked: 2026-08-10
    evidence: "from_pairs_tests::from_pairs_indices_only_gives_none_columns green"
  - id: ac-005
    summary: DISP 或 DIST_SQ 单列物化正确
    type: runtime
    pass_when: |
      DIST_SQ：dist_sq Some 且 len==n_pairs，disp None；
      DISP：disp Some 且 nrows==n_pairs，dist_sq None。
    status: verified
    last_checked: 2026-08-10
    evidence: "from_pairs_tests::{from_pairs_dist_sq_only,from_pairs_disp_only} green"
  - id: ac-006
    summary: Self 半壳 i<j 不变量
    type: runtime
    pass_when: |
      由 BruteForce 或 LinkCell 物化的 Self 结果，对每个 pair 有 idx_i < idx_j。
    status: verified
    last_checked: 2026-08-10
    evidence: "from_pairs_tests::bruteforce_self_half_shell_and_consistency green (4-atom PBC ortho box)"
  - id: ac-007
    summary: 双列时 d2 与 ||disp||2 一致
    type: scientific
    pass_when: |
      FULL 物化下 max_k |dist_sq[k] - ||disp[k]||^2| <= 1e-12
      （正交与至少一种 PBC 盒）。
    status: pending
  - id: ac-008
    summary: 缺列 for_each 不注入 0 物理量
    type: code
    pass_when: |
      Neighbors::for_each_pair（或等价 API）在 !storage.disp 时不向回调传入
      伪装的 [0,0,0] 作为「已存储位移」；文档写明仅完整 NeighborPair 流含全物理量。
    status: verified
    last_checked: 2026-08-10
    evidence: "Option-carrying callback FnMut(u32,u32,Option<F>,Option<[F;3]>); contract documented incl. why (zero is a legal physical value)"
  - id: ac-009
    summary: repack 仅降列，升级请求响亮失败
    type: runtime
    pass_when: |
      FULL→DIST_SQ / INDICES_ONLY 的 repack 正确复制现有列；
      任何请求源缺失列的升级（如 INDICES_ONLY→FULL）panic 且消息指明
      不可凭空造物理量，绝不填 0。
    status: verified
    last_checked: 2026-08-10
    evidence: "from_pairs_tests::{repack_downgrade_drops_columns,repack_upgrade_panics} green"
  - id: ac-010
    summary: QueryMode 携带点集计数，NeighborsMeta 不存在
    type: code
    pass_when: |
      QueryMode::SelfQuery { num_points } 与 CrossQuery { num_query_points, num_points }；
      crate 中无名为 NeighborsMeta 的类型；from_pairs 第三参即 QueryMode。
    status: verified
    last_checked: 2026-08-10
    evidence: "grep: 0 hits NeighborsMeta; payload variants live in mod.rs and are pattern-matched crate-wide"
out_of_scope:
  - NeighborList 总包 build/update/iter（02）
  - compute 缺 disp 硬错误统一化（03，可在 01 做最小编译修复）
  - Python/WASM 重命名（04）
  - skin
---

# Acceptance — neighborlist-01-types

本阶段「完成」= 物化表叫 `Neighbors`，列策略叫 `NeighborsStorage`（`disp`），
`Neighbors::from_pairs` 从 pair 流写表，缺列用 `Option` 表达，半壳与 d²–disp
一致性有测，repack 不再造零，QueryMode 自带计数（NeighborsMeta 不存在）。

沿途铁律修复：`filter_sann` 判据修为 van Meel JCP 136, 234107 (2012)（旧式仅全等距可满足 → no-op；两个旧 golden 编码了该 bug，已按文献重推导）。
