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
    status: pending
  - id: ac-002
    summary: NeighborsStorage 使用 disp 字段且 FULL 表示列齐
    type: code
    pass_when: |
      NeighborsStorage 含 dist_sq: bool 与 disp: bool；
      FULL == {dist_sq:true, disp:true}；INDICES_ONLY 两者 false。
      rustdoc 写明 FULL 非双向 pair、disp 为非单位 MIC 位移。
    status: pending
  - id: ac-003
    summary: from_pairs 从 NeighborPair 流写表
    type: runtime
    pass_when: |
      Neighbors::from_pairs(已知 2 个 NeighborPair 的 iter, FULL, SelfQuery{..})
      得到 n_pairs==2，indices/dist_sq/disp 与输入一致（disp 分量误差 <= 1e-15）。
    status: pending
  - id: ac-004
    summary: 可选列缺失时 accessor 返回 None
    type: runtime
    pass_when: |
      from_pairs(..., INDICES_ONLY, ..) 后 dist_sq() 与 disp() 均为 None；
      n_pairs()>0 时也不会返回全零假列。
    status: pending
  - id: ac-005
    summary: DISP 或 DIST_SQ 单列物化正确
    type: runtime
    pass_when: |
      DIST_SQ：dist_sq Some 且 len==n_pairs，disp None；
      DISP：disp Some 且 nrows==n_pairs，dist_sq None。
    status: pending
  - id: ac-006
    summary: Self 半壳 i<j 不变量
    type: runtime
    pass_when: |
      由 BruteForce 或 LinkCell 物化的 Self 结果，对每个 pair 有 idx_i < idx_j。
    status: pending
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
    status: pending
  - id: ac-009
    summary: repack 仅降列，升级请求响亮失败
    type: runtime
    pass_when: |
      FULL→DIST_SQ / INDICES_ONLY 的 repack 正确复制现有列；
      任何请求源缺失列的升级（如 INDICES_ONLY→FULL）panic 且消息指明
      不可凭空造物理量，绝不填 0。
    status: pending
  - id: ac-010
    summary: QueryMode 携带点集计数，NeighborsMeta 不存在
    type: code
    pass_when: |
      QueryMode::SelfQuery { num_points } 与 CrossQuery { num_query_points, num_points }；
      crate 中无名为 NeighborsMeta 的类型；from_pairs 第三参即 QueryMode。
    status: pending
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

## AC-001 — 公开名 Neighbors

结果类型不再占用 `NeighborList` 名。

## AC-002 — Storage / FULL / disp

字段与常量如上；rustdoc 澄清 FULL 与 disp 语义（位移，非方向）。

## AC-003 — from_pairs

关联构造器纯函数式写表，可单测不依赖 cell。

## AC-004 / AC-005 — 可选列

None vs Some 行为钉死。

## AC-006 — half-shell

继承现有 self-query 约定。

## AC-007 — 科学一致

MIC 一次生成 d 与 d²。

## AC-008 — 无静默零

修掉今日 `for_each_pair` 缺列填 0 的契约。

## AC-009 — repack 不造物理量

修掉今日 `repack` 升级列填 0 的契约。

## AC-010 — QueryMode payload

counts 并入 mode，self 两计数不等的非法状态不可表示。
