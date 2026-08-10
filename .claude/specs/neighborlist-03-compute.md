---
title: neighborlist-03-compute — compute 层统一消费 Neighbors / pair 流
status: code-complete
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

1. **Add** `require_disp` / `require_dist_sq` in compute util ✅（compute/require.rs，13 站点消费：10 disp + 3 dist_sq）
2. **Migrate** order (steinhardt, hexatic, solid_liquid, continuous_coordination) ✅
3. **Migrate** rdf + cluster ✅（**铁律加成**：修掉 rdf accumulate_into 在 release 对 indices-only 表静默返回全零 g(r) 的缺陷 → Result 传播）
4. **Migrate** pmft + environment + density + hbond + van_hove ✅（match_env expect → BadShape 传播）
5. **Update** `nlist_from_frame` → FULL `Neighbors` via `NeighborList` 总包 ✅（build_columns 路径；rdf 两条 self 流亦迁引擎，cross 每点流留 pub(crate) 后端并注明）
6. **Test** steinhardt 在 INDICES_ONLY Neighbors 上返回 BadShape（非 panic）✅（+ hexatic 同型锚点）
7. **Test** 既有 order/rdf 单元测试全绿 ✅（+ RDF ×2 因子隔离锚点，变异校验过）
8. **Docs** compute 模块 rustdoc：Args 为 `Neighbors`，order 需要 DISP/FULL（→ docs Mode A）
9. **（02 路由）引擎并行物化决策** ✅ **决议：接回 rayon**。实测（release）：串行 vs 并行 N=20k 9.6ms/3.8ms、N=100k 55ms/23ms → `Backend::materialize_into` 钩子（串行默认），LinkCell 覆写驱动 rayon fold；细胞循环 4 副本去重到 2；合并对齐升为 `Neighbors::append` 契约 + debug assert；并行路径补 512 原子/1536-pair golden 覆盖测试（破坏校验证实先前零覆盖）；引擎物化 N=100k 26.1ms（4.2×）
10. **（架构师裁决路由）重复自查询面** ✅ **as-built：删除而非降级**（降级即生产 build 死代码，clippy -D warnings 拒绝；experimental 无 shim）。删除：`LinkCell::{build,update,query,build_soa,with_storage,storage,refresh_result}` + 缓存表字段、`BruteForce::{build,query}` + 三态 visit_pairs 回退、`AabbQuery::query` + compute_pairs/AabbTree::query 级联、`Neighbors::clear`（净 −303 行）。`AabbQuery::{new,cutoff,build,query_knn}` 保持公开（k-NN 引擎无法表达）。~25 测试调用点先行迁引擎；顺带修掉一个不可能失败的测试（缓存表自比较）。两处公开危害（空表 SelfQuery{0}、三态静默）随删除消灭
11. **（02 路由）rdf accumulator mode 判据** ✅（as-built：latch 改为无计数 `Option<RdfMode>`，普通 `!=` 判等 —— 类型名承载意图，避免单调用点谓词提取）
12. **（02 路由）RDFResult.mode 冗余** ✅（as-built：`RDFResult.mode` 类型改为无计数 `pub enum RdfMode`，`From<QueryMode>` 单一转换点；撒谎的 counts 不复存在）

## Testing

- `cargo test -p molcrafts-molrs --lib --features full,filesystem` 相关模块（与 CI gate 一致）
- 显式 BadShape 测（order）

## Out of scope

- binders
- skin
- 新科学核
