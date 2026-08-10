---
title: neighborlist-02-engine — NeighborList 总包 build/update/iter/neighbors
status: approved
created: 2026-08-10
slug: neighborlist-02-engine
chain: neighborlist
phase: 02-engine
depends_on: neighborlist-01-types
grilled: true
---

# neighborlist-02-engine — NeighborList 总包

## Summary

引入 **`NeighborList` 作为总包类**（拥有 cutoff / 算法后端 / 空间索引 / 枚举），公开：

- `build(points, box)` — 建/重建 **索引**（不默认整表缓存）
- `update(points)` — 位置更新后重建索引（复用 box；见语义表）
- `for_each_pair` — 产出 **`NeighborPair`** 流（半壳 self）
- `neighbors(storage) -> Neighbors` — 糖：内部经 `pub(crate)` push 直写（01 决议），
  结果 **必须等价于** `Neighbors::from_pairs(收集的 pair 流, storage, mode)`

**禁止** 新名 `NeighborSearch`。**公开 API 禁 `Nb` 缩写**（2026-08-10 讨论决议）：

- `NbList<A>` 包装 **删除** —— 总包即其替身，不留三个近似名并存
- `NbListAlgo` **降为 `pub(crate)` trait `Backend`**（`neighbors::Backend`）；
  外部自带后端的需求目前不存在，引擎以 `brute_force()` 等显式构造覆盖选择
- 同步 CLAUDE.md Trait-Based Extensibility 表（`NbListAlgo` 行改为内部 `Backend`）

内部复用 `LinkCell` / `BruteForce`（及可选 AABB）为可插后端。
`NeighborQuery`：收编为兼容路径或 deprecate 指向 `NeighborList`（experimental：可删交叉 API 重复，保留 cross-query 能力在总包上）。

## Domain basis

- 索引构建与 pair 枚举分离：可流式（LAMMPS 热路径）或物化（freud 分析）。
- Self 枚举默认 half-shell；`NeighborPair.disp = MIC(r_j-r_i)`。
- Skin：仅文档预留，本阶段不实现（scientist §5）。

## Design

### Reuse decision

| Candidate | Tag | Decision |
|-----------|-----|----------|
| `NbListAlgo` + `NbList` | generalize | 总包 `NeighborList` 取代；`NbList` 包装 **删除**；`NbListAlgo` → `pub(crate) Backend` |
| `LinkCell` / `BruteForce` / `AabbQuery` | reuse | 后端实现 visit；可不直接作为 binder 主入口 |
| `NeighborQuery` | generalize | self → `NeighborList`；cross → `NeighborList::iter_cross` 或保留薄包装调总包 |
| `CellGrid` | reuse | 仍仅服务 LinkCell |
| `build`/`update` 名 | reuse | 语义改为 **index-only**；旧「build=物化」行为删除 |

### 公开 API（概念）

```rust
pub struct NeighborList { /* backend, cutoff, box?, … */ }

impl NeighborList {
    pub fn new(cutoff: F) -> Self; // 默认 LinkCell 后端
    pub fn brute_force(cutoff: F) -> Self; // 或 with_backend

    pub fn build(&mut self, points: FNx3View<'_>, bx: &SimBox);
    pub fn update(&mut self, points: FNx3View<'_>); // 需已 build 过；见语义表

    pub fn for_each_pair(&self, f: impl FnMut(NeighborPair));
    // 真迭代器非必需：for_each_pair 为流式 SSOT；
    // neighbors() 经 pub(crate) push 直写（from_pairs 仍是唯一公开构造入口）

    pub fn neighbors(&self, storage: NeighborsStorage) -> Neighbors;
}
```

### 语义

| 方法 | 必须 | 禁止 |
|------|------|------|
| `build` | 设置 box + 建空间索引 | 强制分配完整 `Neighbors` |
| `update` | 新 positions 更新索引（复用已 build 的 box）；build 前调用 → panic（明确消息）；box 变化（NPT 每帧）→ 文档明示用 `build` | 改名成「返回 iter」；静默容忍未 build |
| `for_each_pair` | 每个 half pair 一次完整 `NeighborPair` | 缺 d²/disp |
| `neighbors` | 按 storage 物化；结果等价 `from_pairs(流, storage, mode)` | 默认 FULL 隐式 |

### 与 01 的边界

- 01 已提供 `Neighbors` / `from_pairs` / `NeighborPair` / `QueryMode` payload
- 02 只让 **总包** 成为 pair 流的权威生产者；`Neighbors::push`（pub(crate)）
  是引擎的直写通道，公开世界只见 `from_pairs`

## Files

- `molrs/src/core/spatial/neighbors/mod.rs` — `NeighborList` 总包；`NbList` 删除；`NbListAlgo` → `Backend`（pub(crate)）
- `linkcell.rs` / `bruteforce.rs` — 后端适配
- `query.rs` — 收敛到总包或标记 deprecated
- `CLAUDE.md` — Trait-Based Extensibility 表同步
- tests + benches 路径更新（`neighbors/traversal/*` 仍测 visit）

## Tasks

1. **Add** `NeighborList` 总包类型（默认 LinkCell 后端）
2. **Implement** `build(points, box)` / `update(points)` 为 index-only；update-before-build panic
3. **Implement** `for_each_pair` → `NeighborPair`（self half）
4. **Implement** `neighbors(storage) -> Neighbors`（内部 push 直写；等价 from_pairs）
5. **Delete** `NbList<A>` 包装；**demote** `NbListAlgo` → `pub(crate) Backend`；同步 CLAUDE.md trait 表
6. **Wire** BruteForce 后端（测试/oracle）
7. **Migrate or deprecate** `NeighborQuery::query_self` 到总包
8. **Preserve** cross-query 能力（`iter_cross` 或 `NeighborQuery::query` 委托）
9. **Test** build→for_each ≡ BruteForce pair multiset；neighbors(FULL) 列完整；neighbors ≡ from_pairs(收集流)
10. **Bench** visit_pairs 路径不劣于既有 baseline 的灾难阈（沿用 cell-grid-api 精神，可选 10%）
11. **Docs** rustdoc 示例：流式 RDF 风格 vs `neighbors(DISP)` for order；NPT 用 build 每帧

## Testing

- LinkCell 总包 vs BruteForce 总包 pair 多重集一致（复用 ac 矩阵思想）
- `build` 后未 `neighbors` 时无大块 pair 缓冲（可选：内部 `nlist` 缓存为空或未用）
- `update` 后 pair 集跟随新坐标；update-before-build panic 测试
- `neighbors(FULL)` ≡ `Neighbors::from_pairs(收集的 pair, FULL, mode)`

## Out of scope

- skin / `update_if_needed`
- compute 签名统一（03）
- binders（04）
- SelfFull 模式
- 排序为 freud segments（可选后续）
