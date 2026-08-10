---
title: neighborlist-01-types — Neighbors storage + from_pairs + optional columns
status: approved
created: 2026-08-10
slug: neighborlist-01-types
chain: neighborlist
phase: 01-types
grilled: true
---

# neighborlist-01-types — Neighbors 物化表与 from_pairs

## Summary

把今日 **物化结果类型** `NeighborList` 改名为 **`Neighbors`**，列策略改为 **`NeighborsStorage { dist_sq, disp }`**，并提供 **`Neighbors::from_pairs(pairs, storage, mode)`**：从 `NeighborPair` 流写出半壳表，列可选。
本阶段 **不** 改搜索总包命名（仍由 `LinkCell`/`BruteForce` 产出 pair）；只钉死 **表 + 流物化契约**，为 02 总包 `NeighborList` 让路。

用户锁定名（2026-08-10 讨论修订）：`Neighbors`、`from_pairs`、`disp`（MIC 位移，**非**单位方向）、`dist_sq`。
旧议名废弃：`materialize`（DB 行话、自由函数撞反工厂规则）、`dire`（缩写且名不副实）、`NeighborsMeta`（不透明 blob，counts 并入 `QueryMode`）。
**FULL = 列齐备**，不是双向 pair。Self 默认 **半壳 `i < j`**。

## Domain basis

（scientist 摘要；完整推导见 grill 附件语境）

1. **Half-shell self** \(i < j\)：无序对只存一次；与 LAMMPS half 一致。Cross 为有向 full，禁止 `i < j`。
2. **MIC 位移** \(\mathbf{d}_{ij}=\mathrm{MIC}(\mathbf{r}_j-\mathbf{r}_i)\) 存为 `disp`；\(d^2=\|\mathbf{d}\|^2\) 同源同次 MIC。
3. **可选列**：RDF 要 \(d^2\)；order 要 \(\mathbf{d}\)；cluster 只要 indices。流式路径在枚举时 **始终算出** \(d^2,\mathbf{d}\)，只是可不落盘。
4. **禁止** 缺列时用 0 向量 / 0 距离冒充有效 MIC（静默错误源）。
5. Skin / Verlet rebuild **out of scope**（见 02 Out of scope 与 scientist §5）。

References: LAMMPS half/full neighbor lists; freud NeighborList vector convention; DLMF 14.30.7 (\(Y_{\ell m}\) parity for later compute).

## Design

### Reuse decision（librarian）

| Candidate | Tag | Decision |
|-----------|-----|----------|
| 今日 `NeighborList` 结果 struct | generalize | **改名为 `Neighbors`**；字段/mode/repack 迁移，不平行再造 |
| `NeighborListStorage` | reuse | 改名为 **`NeighborsStorage`**；`diff` → **`disp`**；常量 `INDICES_ONLY` / `DIST_SQ` / `DISP` / `FULL`（`FULL` = dist_sq+disp） |
| `QueryMode` | generalize | **携带点集计数**（见下）；`NeighborsMeta` **不引入** —— self 只有一个点集、cross 才有两个，counts 结构上属于 mode，非法状态（self 两计数不等）不可表示 |
| `PairVisitor` / `for_each_pair` | reuse | `NeighborPair` 作为统一 pair 值；`from_pairs` 建立在 pair 流上 |
| `repack` | generalize | 保留为 `Neighbors::repack(storage)`，**仅允许降列**；请求源缺失列 → panic（明确消息），禁止造零 |

### 类型

```rust
/// 枚举时的完整物理 pair（永不缺列）。
pub struct NeighborPair {
    pub i: u32,
    pub j: u32,
    pub dist_sq: F,
    pub disp: [F; 3], // MIC(r_j - r_i)，非单位向量
}

/// 列策略：FULL = 列齐，非双向。
pub struct NeighborsStorage {
    pub dist_sq: bool,
    pub disp: bool,
}

impl NeighborsStorage {
    pub const INDICES_ONLY: Self = Self { dist_sq: false, disp: false };
    pub const DIST_SQ: Self = Self { dist_sq: true, disp: false };
    pub const DISP: Self = Self { dist_sq: false, disp: true };
    pub const FULL: Self = Self { dist_sq: true, disp: true };
}

/// 查询身份：counts 并入 mode；self 模式两计数不等这类非法状态不可表示。
pub enum QueryMode {
    SelfQuery { num_points: usize },
    CrossQuery { num_query_points: usize, num_points: usize },
}

/// 半壳（Self）或 cross 物化表。
pub struct Neighbors { /* mode, idx_i, idx_j, optional cols */ }
```

`QueryMode` 带 payload 后，compute 内 `mode() == SelfQuery` 改写为
`matches!(mode, QueryMode::SelfQuery { .. })`（本阶段最小编译修复，完整契约在 03）。

### from_pairs

```rust
impl Neighbors {
    /// 唯一**公开**的从流写表构造入口（关联构造器；无自由函数，遵守反工厂规则）。
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = NeighborPair>,
        storage: NeighborsStorage,
        mode: QueryMode,
    ) -> Self;
}
```

实现：按 storage 写入列；**不**从空列填 0。
信任边界：`SelfQuery` 下对每个 pair `debug_assert!(i < j)` —— 契约违规是编程错误；
RDF ×2 归一化信任 mode，错标会静默双倍 g(r)，debug/test 构建必须抓住。
`push` 保持 `pub(crate)`：02 引擎经 `for_each` 直写、免临时缓冲；
`from_pairs` 仍是唯一**公开**入口。

### Neighbors accessors

- `query_point_indices() / point_indices() / n_pairs() / mode() / storage()` — 始终可用
- `dist_sq() -> Option<&[F]>`
- `disp() -> Option<FNx3View<'_>>` — 无列时 **None**，禁止 0×3「假有表」在 `storage.disp==true` 时出现
- `distances()`（每调 sqrt、缺列静默空）**删除**；caller 自己对 dist_sq 开方
- 若 `storage.disp && n_pairs>0` 则 `disp().unwrap().nrows() == n_pairs`（invariant）

### 迁移策略（本阶段）

1. 引入 `NeighborPair`、`Neighbors`、`NeighborsStorage`、`QueryMode` payload、`from_pairs`。
2. 一步改名 + 全 crate 同步，无 long-lived alias（stage experimental）。
3. 旧 `NeighborListStorage` / `diff` / `vectors()`：
   - storage 字段 `disp`
   - accessor `disp()` 替代 `vectors()`；`vectors`、`distances` **删除**（experimental）
4. `for_each_pair` on `Neighbors`：对缺列传 `None` 或只提供 indices 变体；**不得**把缺列当成 0.0。
5. `filter_rad` / `filter_sann`（core 层，同模块）：读列改经 Option；缺所需列 → **panic（明确消息）**，不再依赖假空表。
6. **`molrs-cxxapi` 是主 workspace 成员**（`lib.rs:413` 返回旧名）：本阶段做最小 rename 修复，不能等 04。

### Invariants

- SelfQuery：`∀k: idx_i[k] < idx_j[k]`
- CrossQuery：无 `i<j` 约束
- `dist_sq[k]` 存在时：`|dist_sq[k] - ||disp[k]||²| < ε`（当两列都在）

## Files

- `molrs/src/core/spatial/neighbors/mod.rs` — 类型改名、from_pairs、QueryMode payload、Option accessors
- `molrs/src/core/spatial/neighbors/{linkcell,bruteforce,query,filter,aabb}.rs` — 结果类型 `Neighbors`；push 路径兼容 storage.disp；filter 缺列 panic
- `molrs/src/compute/**` — **最小** 编译修复：`NeighborList` → `Neighbors`、`vectors` → `disp().expect`、mode 比较 → `matches!`（完整契约在 03）
- `molrs-cxxapi/src/lib.rs` — 最小 rename 修复（workspace 成员，01 必须编译）
- 单元测试同文件 `#[cfg(test)]`

## Tasks

1. **Add** `NeighborPair` + `NeighborsStorage`（`disp`）+ `QueryMode` payload（**不引入** NeighborsMeta）
2. **Rename** result type `NeighborList` → `Neighbors`；`NeighborListStorage` → `NeighborsStorage`；`diff` → `disp`
3. **Implement** `Neighbors::from_pairs(pairs, storage, mode)`（SelfQuery 逐 pair `debug_assert!(i<j)`）
4. **Change** accessors：`dist_sq() -> Option`、`disp() -> Option`；删除 `vectors()`、`distances()`
5. **Restrict** `repack` 为仅降列；升级请求 panic（明确消息）
6. **Harden** `filter_rad`/`filter_sann` 缺列 → panic（明确消息）
7. **Update** LinkCell/BruteForce/filter/query 内部类型到 `Neighbors`
8. **Fix** in-crate compile（compute/ff）与 `molrs-cxxapi` 最小 rename
9. **Test** from_pairs 各列旗标；半壳 `i<j`；缺列 Option；d²–disp 一致；repack 升级 panic
10. **Docs** rustdoc：FULL=columns；half-shell default；`disp` 非单位向量；freud name mapping note

## Testing

- Unit：`from_pairs` with each `NeighborsStorage` constant
- Unit：self half-shell invariant；SelfQuery `debug_assert`（debug 构建 should_panic）
- Unit：both columns ⇒ \(d^2 = \|d\|^2\)
- Unit：`disp() == None` when `!storage.disp` even if `n_pairs > 0`
- Unit：`repack` 降列 OK；升级请求 panic
- Existing neighbor equivalence tests still green after rename

## Out of scope

- `NeighborList` 作为 **总包** 类（→ 02）
- skin / `update_if_needed`
- SelfFull 模式
- binder API 重命名（→ 04）
- molvis stage（另仓）
