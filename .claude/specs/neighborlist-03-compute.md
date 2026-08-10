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

## Testing

- `cargo test -p molcrafts-molrs --lib --features full,filesystem` 相关模块（与 CI gate 一致）
- 显式 BadShape 测（order）

## Out of scope

- binders
- skin
- 新科学核
