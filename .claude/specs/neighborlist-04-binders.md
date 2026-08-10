---
title: neighborlist-04-binders — Python/WASM NeighborList + Neighbors 表面
status: approved
created: 2026-08-10
slug: neighborlist-04-binders
chain: neighborlist
phase: 04-binders
depends_on: neighborlist-03-compute
grilled: true
---

# neighborlist-04-binders — 绑定层

## Summary

将 **Python** 与 **WASM** 公开表面对齐核心契约：

| 绑定名 | 含义 |
|--------|------|
| `NeighborList` | 总包：`build` / `update` / `neighbors(storage)` 或等价 |
| `Neighbors` | 物化表；只读列；`disp` / `dist_sq` 可选 |

废弃/删除「BruteForce/LinkedCell 是唯一入口且默认 lean 无 disp」导致 order 崩溃的路径；算法选择变为总包配置或显式后端构造。

## Design

### Reuse decision

| Candidate | Decision |
|-----------|----------|
| WASM `LinkedCell`/`BruteForce`/`NeighborList` | generalize → `NeighborList` 总包 + `Neighbors` 结果；旧名可短期 alias 到总包构造 |
| Python `PyNeighborList` / `PyLinkedCell` / `PyNeighborQuery` | 同上 |
| stage `storeDiff` | **out of molrs spec**；文档注明 DISP 物化；molvis 另跟 |

### WASM 草图

```ts
const nl = new NeighborList(cutoff); // or NeighborList.linkedCell / .bruteForce
nl.build(frame); // or points+box API 与现 Frame 一致
const neigh = nl.neighbors({ distSq: true, disp: true });
steinhardt.compute(frame, neigh);
```

### Python 草图

```python
nl = molrs.NeighborList(cutoff)
nl.build(points, box)
neigh = nl.neighbors(dist_sq=True, disp=True)
```

### 默认 storage

- Binder `neighbors()` 默认 **`FULL`**（安全；分析友好）。
- 显式可传 lean。
- **禁止** 默认无 disp 却让 Steinhardt 静默挂。

### Cross-query 存续

现有 binder 若暴露 cross 能力（wasm 引用 `RsNeighborQuery` 等），迁移后
**不得静默消失**：要么导出等价 cross 入口，要么在 guide/CHANGELOG 明确记录移除。

## Files

- `molrs-wasm/src/compute.rs`（及导出）
- `molrs-python/src/core/spatial/linkedcell.rs`（及 lib 导出）
- `molrs-python/tests/test_linkedcell.py`、`site-src/guides/neighbor-search.md`
- 可选：`molrs-cxxapi` 剩余清理（最小 rename 已在 01 完成）

## Tasks

1. **Expose** WASM `NeighborList` 总包 + `Neighbors`
2. **Expose** Python 同名表面
3. **Default** binder materialize FULL
4. **Migrate** 文档与 py tests
5. **Remove or alias** 旧易混导出（document breaking）
6. **Preserve or document** cross-query 出路（见 Design）
7. **Smoke** WASM 或 Python：build → neighbors(DISP) → Steinhardt 不 trap
8. **Note** freud 命名对照（Query vs List）于 guide

## Testing

- `molrs-python` tox/pytest linkedcell + 一个 order 冒烟（若已有）
- wasm 包构建 + 既有 stage 测试若在 monorepo 可选跑 structure_order（跨仓可记为 manual）

## Out of scope

- molvis `SpatialNeighborQuery` 实现（建议 follow-up；可用 FULL 默认临时对齐）
- skin
- cxx 全量
