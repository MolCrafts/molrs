---
spec: neighborlist-02-engine
created: 2026-08-10
criteria:
  - id: ac-001
    summary: 公开类型 NeighborList 为总包且提供 build/update
    type: code
    pass_when: |
      molrs::spatial::neighbors::NeighborList 存在；
      有 build(points, &SimBox) 与 update(points) 方法签名。
    status: pending
  - id: ac-002
    summary: build/update 不强制物化 Neighbors 缓存为唯一路径
    type: code
    pass_when: |
      rustdoc 或实现表明 build/update 负责索引；物化仅通过 neighbors(storage)
      或 Neighbors::from_pairs。允许内部缓冲优化，但 for_each_pair 在未调用
      neighbors 时必须可用。
    status: pending
  - id: ac-003
    summary: for_each_pair 产出完整 NeighborPair
    type: runtime
    pass_when: |
      对 octahedron/简单立方测试系，每个 pair 的 dist_sq 与 ||disp||^2 一致
      （<=1e-12），且 self 模式 i<j。
    status: pending
  - id: ac-004
    summary: neighbors(storage) 等于 from_pairs(收集的流)
    type: runtime
    pass_when: |
      同一 NeighborList 上 neighbors(FULL) 与手动 Neighbors::from_pairs(
      收集的 pair, FULL, mode) 在 indices、dist_sq、disp 上一致。
    status: pending
  - id: ac-005
    summary: LinkCell 与 BruteForce 后端 pair 集一致
    type: runtime
    pass_when: |
      至少正交盒 + 全 PBC 一档：两后端 for_each_pair 无序 pair 多重集相同，
      dist_sq 差 <= 1e-12。
    status: pending
  - id: ac-006
    summary: 无 NeighborSearch 公开类型
    type: code
    pass_when: |
      公共 API 中不存在 NeighborSearch 标识符。
    status: pending
  - id: ac-007
    summary: Cross-query 仍可达
    type: runtime
    pass_when: |
      存在公开路径（NeighborList 方法或保留的 NeighborQuery）完成 cross-query，
      且结果 mode 为 CrossQuery、无 i<j 强制。
    status: pending
  - id: ac-008
    summary: 公开 API 无 Nb 缩写标识符
    type: code
    pass_when: |
      公共 API 不再导出 NbList 与 NbListAlgo（后者降为 pub(crate) Backend
      或等价内部名）；CLAUDE.md Trait-Based Extensibility 表已同步。
    status: pending
  - id: ac-009
    summary: update 前置条件响亮失败
    type: runtime
    pass_when: |
      未 build 先 update → panic 且消息指明需先 build；
      update 后 pair 集跟随新坐标。
    status: pending
out_of_scope:
  - skin
  - compute/binder 迁移
---

# Acceptance — neighborlist-02-engine

总包 `NeighborList` 用 `build`/`update` 管索引，用 `for_each_pair`/`neighbors`
消费 pair；不引入 `NeighborSearch`；公开表面无 `Nb` 缩写。
