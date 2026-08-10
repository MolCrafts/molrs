---
title: neighborlist-03-compute — compute 层统一消费 Neighbors / pair 流
status: approved
created: 2026-08-10
slug: neighborlist-03-compute
chain: neighborlist
phase: 03-compute
depends_on: neighborlist-02-engine
grilled: true
---

# neighborlist-03-compute — compute 迁移

## Summary

所有 `compute` 内核从旧「可能缺列的表 + `vectors()`」迁到：

- 参数类型 **`&Neighbors`**（或 `Vec<Neighbors>`）
- 需要位移的内核（Steinhardt / Hexatic / PMFT / bond-order…）**`disp().ok_or(BadShape)`**
- 需要距离的内核读 **`dist_sq()`**
- 仅连通性读 indices

并保证 **order 参数** 在半壳上继续用 \(Y_{\ell m}\) parity 双边更新。
RDF Self 的 **×2 归一化** 绑定 `matches!(mode, QueryMode::SelfQuery { .. })` 不变
（QueryMode 已带 payload，01 决议）。

## Domain basis

- Steinhardt half + parity（DLMF 14.30.7）— 已实现，保持。
- RDF half ×2 / cross 不 ×2 — 已实现，保持。
- 缺 `disp` 不得 OOB / WASM unreachable。

## Design

### Reuse decision

| Candidate | Decision |
|-----------|----------|
| `compute_qlm` BadShape 检查（2026-08-10 热修已入） | **reuse** 并统一到 `require_disp(&Neighbors)` 辅助 |
| `nlist_from_frame` test helper | **generalize** → 返回 `Neighbors` FULL via 总包 |
| RDF / cluster / pmft / environment / density / hbond / van_hove | **rename types + require columns** |

### 辅助

```rust
// compute 内部
fn require_disp(n: &Neighbors) -> Result<FNx3View<'_>, ComputeError>;
fn require_dist_sq(n: &Neighbors) -> Result<&[F], ComputeError>;
```

### 签名

`Compute::Args` 中 `&Vec<NeighborList>` → `&Vec<Neighbors>`（或 slice）。
mode 判等全部改 `matches!(…, SelfQuery { .. })` 形式。

## Files

- `molrs/src/compute/order/**`
- `molrs/src/compute/rdf/**`
- `molrs/src/compute/cluster/**`
- `molrs/src/compute/pmft/**`
- `molrs/src/compute/environment/**`
- `molrs/src/compute/density/**`
- `molrs/src/compute/dynamics/van_hove.rs`
- `molrs/src/compute/hbond/**`
- `molrs/src/compute/test_support.rs`
- `molrs/src/ff/potential/soft.rs`（若仍用 NeighborQuery）

## Tasks

1. **Add** `require_disp` / `require_dist_sq` in compute util
2. **Migrate** order (steinhardt, hexatic, solid_liquid, continuous_coordination)
3. **Migrate** rdf + cluster
4. **Migrate** pmft + environment + density + hbond + van_hove
5. **Update** `nlist_from_frame` → FULL `Neighbors` via `NeighborList` 总包
6. **Test** steinhardt 在 INDICES_ONLY Neighbors 上返回 BadShape（非 panic）
7. **Test** 既有 order/rdf 单元测试全绿
8. **Docs** compute 模块 rustdoc：Args 为 `Neighbors`，order 需要 DISP/FULL
9. **（02 路由）引擎并行物化决策**：`NeighborList::neighbors()` 现为串行 visit 驱动，`LinkCell::compute_pairs_parallel`（rayon，>64 occupied cells，文档记载 ~2× @N=1k）失去生产调用方，**且 cxxapi per-frame RDF 已从并行迁到串行（已落地回退）**。关闭本任务前必须跑既有 bench 对照（benches/core/neighbors/linkcell.rs 引擎 serial vs build_soa parallel）；差距为真 → 引擎物化接回 rayon，否则删除死代码。不得静默保留
10. **（架构师裁决路由）重复自查询面降级**：迁移 `compute/rdf`、`compute/test_support`、`benches/core/neighbors` 到引擎（`build_columns` 已由 02 提供）后，将 `LinkCell::{build,update,query,build_soa,with_storage,storage}`、`BruteForce::{build,update,query}`、`AabbQuery::{query}` 降为 pub(crate)。`AabbQuery::{new,cutoff,build,query_knn}` 保持公开（query_knn 是引擎无法表达的能力）。此举同时消灭 build_index→query() 空表 SelfQuery{num_points:0} 与 BruteForce::visit_pairs 三态静默选择两处公开危害
11. **（02 路由）rdf accumulator mode 判据**：`compute/rdf/accumulator.rs` 的 `std::mem::discriminant` 比较换成命名谓词（保留 01 迁移时的原语义：模式变体判等，忽略 counts）
12. **（02 路由）RDFResult.mode 冗余**：`mode` 携带 frame-0 counts 而 `n_points`/`n_query_points` 跨帧求和 —— 归一化只读变体故行为无变；03 统一契约时去冗余

## Testing

- `cargo test -p molcrafts-molrs --lib --features full,filesystem` 相关模块（与 CI gate 一致）
- 显式 BadShape 测（order）

## Out of scope

- binders
- skin
- 新科学核
