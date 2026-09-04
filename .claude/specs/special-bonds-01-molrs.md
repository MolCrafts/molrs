---
title: special-bonds-01-molrs — Topology 从 Frame 读图，按键距的权重表 BondDistanceWeights
status: code-complete
created: 2026-09-04
depends_on: []
chain: special-bonds（01 of 6；02–06 在 molpack 仓）
grilled: true
---

# special-bonds-01-molrs — Topology 从 Frame 读图，按键距的权重表 BondDistanceWeights

## Summary

把「模板键图」与「按拓扑键距给权重」收进 molrs `core::system`，成为 0.14 线上纯增量的公开 API：`Topology::from_frame` 从 `Frame` 的 `atoms` / `bonds` 读出连接性（邻接按键表插入顺序，永不排序），`BondDistanceWeights` 是按键距索引、末位为 1-N 尾项的权重向量，`Topology::exclusions(&BondDistanceWeights)` 给出逐原子、含自身、升序的豁免伙伴表，豁免当且仅当 `weight(distance) == 0.0`（零尾项则走通整个连通分量）。molpack 今天在 `src/topology.rs` 里维护的是同一概念的第二份实现；本 spec 落地后读图与排除表的测试归这里，molpack 只保留几何读取与生长策略错误。`ff::forcefield::SpecialBonds` 的结构体、`Default` 与访问器冻结；仅补 rustdoc，使其指向 core 表。两者之间没有 `From`/`Into`。

## Domain basis

- **按拓扑键距索引的权重向量是全行业写法。** LAMMPS `special_bonds`：amber `0 0 0.5`、charmm `0 0 0`、dreiding `0 0 1`、fene `0 1 1`（<https://docs.lammps.org/special_bonds.html>）只写 1-2 / 1-3 / 1-4，**更远键距隐式为 1**；OPLS-AA `fudgeLJ = 0.5`；TraPPE 1-4 LJ = 0（Luo et al. 2023, J. Phys. Chem. B 127, 2224, doi:10.1021/acs.jpcb.2c06993）；Cassandra `Intra_Scaling` 写作 `0 0 0.5 1.0`，**末位是显式的 1-N 尾项**。本 spec 采用 Cassandra 的尾项约定：向量第 k 项（0 基）是键距 k+1 的权重，**最后一项同时是所有更远键距的权重**。故 `[0,0,0,1]` ≡「1-2 / 1-3 / 1-4 豁免，1-5 及以上照常」，与 molpack 今天的 `exclusion_depth = 3` 逐位等价。LAMMPS 三槽必须在这里补上隐式尾项 1：charmm `0 0 0` 是 `[0,0,0,1]`，**不是** `new(vec![0.0, 0.0, 0.0])`（后者尾项为 0，余下整条连通分量都豁免）。
- **权重是无量纲的**，取值域 `[0, 1]`：`0.0` = 完全豁免，`1.0` = 全强度。键距是整数图距离。分数权重在这里合法（amber 的 0.5）；「构造算法只接受二值」是 molpack 侧的策略，由 05 在生长编译点具名拒绝。Cassandra 允许零尾项，`new` 不得拒绝。
- **豁免一类对，等于断言某个先验已经拥有那类对的距离分布**（CBMC 的 u^bond / u^ex 分解，Boon 2017, arXiv:1710.03256, doi:10.1063/1.5029566）：一阶扭转先验拥有 k ≤ 3，二阶 RIS 先验拥有 k ≤ 4，没有任何先验拥有 1-6。默认表 `[0,0,0,1]` 正是一阶先验的拥有范围。
- Packmol 的容差本就只管**分子间**（Martínez et al. 2009, J. Comput. Chem. 30, 2157, doi:10.1002/jcc.21224）。molpack 把它复用到分子内，权重表是给这次复用划边界的尺。本 spec 只提供尺，不做测量。

## Design

**实体与所有权。** 只动 always-on 的 `core::system`：既有 `Topology` 增加两个查询入口；新叶子 `BondDistanceWeights` 是调用方持有的拥有值，查询时以 `&` 传入，**不**存进 `Topology`。`exclusions` 返回拥有的 `Vec<Vec<usize>>`。`core` 不依赖 `ff`。`BondDistanceWeights` 与 `molrs::ff::forcefield::SpecialBonds` 是两个类型：后者在 `ff` 门后、按 lj/coul 各一份、固定三槽、无尾项。没有 `From`/`Into`，没有共同的 `Default`。

**`molrs/src/core/system/bond_weights.rs`（叶子）**

```
pub struct BondDistanceWeights(Vec<F>)          // #[derive(Debug, Clone, PartialEq)]
  pub fn new(weights: Vec<F>) -> Result<Self, MolRsError>   // 非空；每项 0 ≤ w ≤ 1 且有限
  pub fn from_exclusion_depth(depth: usize) -> Self         // vec![0.0; depth] ++ [1.0]
  pub fn weight(&self, distance: usize) -> F                // 0 → 0.0；>= len → 末位（尾项）
  pub fn as_slice(&self) -> &[F]
```

- **没有 `Default`。** 调用方必须写出表（`from_exclusion_depth(3)` 是一行）。
- **`weight(0) == 0.0`**：距离 0 是原子自身，永远豁免。
- **`new` 接受零尾项。** `from_exclusion_depth` 永远以 `1.0` 收尾。长度 3 的向量**不是** LAMMPS 三元组。
- 只有四个原语，不设 `is_exempt` / `tail()` / LAMMPS 三槽构造器。

**`Topology::from_frame`**

- 形状照 `Atomistic::from_frame`：`&Frame -> Result<Self, MolRsError>`，列名走 `store::keys::{ATOMI, ATOMJ}`。原子数取 `frame.get("atoms").and_then(|b| b.nrows())`，**不读坐标**。
- **错误：** 无 `atoms` → `NotFound`；`atoms` 在场但 `nrows() == None` → `Validation`（含 `atoms` 与 `nrows`）；键端点越界 → `Validation`（含两端点与 `n_atoms`）；`bonds` 在场且 `nrows() == Some(n>0)` 却缺 uint `atomi`/`atomj`（含只写 `i`/`j`）→ `Validation`，**不得**当成无键 Ok；无 `bonds` 或空 block → `Ok` 零边。
- **端点先于自环。** 先 range-check 两个端点；越界（含 `(n, n)`）是 `Validation`。只在两端都合法之后丢弃 `(a, a)`。自环丢弃是 `from_frame` 独有策略，在委托 `from_edges` **之前**过滤；不改 `from_edges`。
- 邻接永不排序。rustdoc 写明 sorting would reshape a consumer's growth tree。不把 wasm 的 `i`/`j` 列名收进 core。

**`Topology::exclusions(&weights)`**

- **唯一事实源是 `weight(distance) == 0.0`。** 返回 `Vec<Vec<usize>>`，每条含根、升序。
- 两条走法都是「先定 bound，再按 `weight(d)==0` 过滤」，不是闭球：
  - 尾项为 `0.0` → 对每个根 BFS **整个连通分量**，再过滤；
  - 否则 → 不走到超过「最大 `weight==0` 的键距」之外，再过滤。
- 洞表合法：`new([0.0, 0.5, 0.0, 1.0])` 的 1-3（权重 0.5）**不**进表，1-4（权重 0）进表。
- 零尾项不得拒绝。另一条连通分量的原子不进表。
- 不提供 `exclusions(&self, depth)`。不要求私有 `ball` 符号。
- 本方法不读、不写 `frame["exclusions"]`（PME/prmtop 对表是另一权威）。

**导出。** `core/system/mod.rs` 加 `pub mod bond_weights;`，`core/mod.rs` 再导出，于是 `molrs::BondDistanceWeights` 与 `molrs::Topology` 在 crate root 解析。不加 feature 门。

**rustdoc 方向。** always-on core **禁止** `` [`crate::ff::…`] `` 链。core 文档用代码 span 写出 `molrs::ff::forcefield::SpecialBonds`。合法 rustdoc 链接是 ff → core。`SpecialBonds` 结构体 / `Default` / 访问器一行都不改。

**Reuse decision**

- `Topology::from_edges` — **reuse**：滤自环、校验端点后就地委托。
- `Topology::n_components()` — **reuse**。
- `Topology::distances(source)` — **new**：全图 + `-1` 哨兵；测试可用它当预言机，不调用它实现 `exclusions`。
- `Atomistic::from_frame` — **pattern**。
- `ff::forcefield::SpecialBonds` — **new**：不改写成 `{lj: BondDistanceWeights, …}`，不设 `From`/`Into`。
- `ff::potential::intramolecular_pairs` — **new**：重叠写进 rustdoc，不改建。
- molpack `src/topology.rs` — **generalize**：读图与排除表升到本 spec；molpack 同名类型在 03 删除。

## Files to create or modify

- `molrs/src/core/system/bond_weights.rs` (new)
- `molrs/src/core/system/topology.rs`
- `molrs/src/core/system/mod.rs`
- `molrs/src/core/mod.rs`
- `molrs/src/ff/forcefield/mod.rs`（仅 `SpecialBonds` rustdoc）
- `docs/interop.md`（ForceField 的 `special_bonds` 子弹保留；另加 core 表子弹。不要创建 `molrs/docs/`）
- `regressions/special-bonds-01-molrs.md` (new)

## Tasks

- [x] Write failing unit tests for `BondDistanceWeights` (`molrs/src/core/system/bond_weights.rs` `#[cfg(test)]`: empty / out-of-range / non-finite rejected, `from_exclusion_depth(3).as_slice() == [0.0, 0.0, 0.0, 1.0]`, `weight(0) == 0.0`, tail applies past last entry, `new(vec![0.0])` and `new(vec![1.0, 0.0])` accept a zero tail with `weight(97) == 0.0`, type has no `Default`)
- [x] Implement `BondDistanceWeights` in `molrs/src/core/system/bond_weights.rs` and export it from `molrs/src/core/system/mod.rs` + `molrs/src/core/mod.rs`
- [x] Write failing unit tests for `Topology::from_frame` (linear / branched / ring, neighbour order follows bonds block, missing `atoms`, `atoms` with `nrows() == None`, missing `atomi`/`atomj` on a present non-empty bonds block, out-of-range including `(n,n)` is Validation not a dropped loop, in-range self-loops dropped, bondless frame is `Ok` with zero edges, `n_components` separates an isolated atom, a 12-row `atoms` block with only `id` and no `x`/`y`/`z` succeeds)
- [x] Implement `Topology::from_frame` on top of `Topology::from_edges` (range-check endpoints first; filter in-range self-loops before delegating; do not change `from_edges`) and add `#[derive(Debug, Clone)]` to `Topology`
- [x] Write failing unit tests for `Topology::exclusions(&BondDistanceWeights)` (root-inclusive and ascending, C12 depth-1/2/3 literals, ring closure, branched template, amber `[0,0,0.5,1]` leaves 1-4 out, `new(vec![0.0])` and `new(vec![1.0, 0.0])` list far partners, two-component zero-tail walk stays in-component, hole table `[0.0, 0.5, 0.0, 1.0]` lists 1-4 not 1-3, a partner is listed iff `weight(distance) == 0.0`)
- [x] Implement `Topology::exclusions` so a partner is listed iff `weight(distance) == 0.0`; if the tail is `0.0` BFS the connected component then filter, otherwise do not expand past the last zero-weight distance then filter; no `exclusions(&self, depth` entry point
- [x] Add rustdoc per rustdoc style for `BondDistanceWeights` and `Topology::{from_frame, exclusions}` (dimensionless weights, Cassandra tail, length-3 is not a LAMMPS triple, never-sorted adjacency, from_frame-only self-loop drop after range-check, no `crate::ff::` intra-doc links, disambiguate `frame["exclusions"]`, wasm `i`/`j` follow-up) plus a rustdoc-only edit of `SpecialBonds` in `molrs/src/ff/forcefield/mod.rs` and the core-table bullet in `docs/interop.md`
- [x] Add regression example `regressions/special-bonds-01-molrs.md` (public API only; hard-coded C12 goldens, no third-party runtime)
- [x] Run full check + test suite (`cargo fmt --check`, clippy `-D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem`, `cargo test --doc -p molcrafts-molrs --features full,filesystem`)

## Testing strategy

molrs 测试住在代码旁（`#[cfg(test)]`）。每个测试只打一个函数。Frame 夹具就地构造（`atoms` 只需能给出 `nrows` 的列，不必有 `x`/`y`/`z`）。无第三方。

- **Happy path：** C12 直链读出 12 原子 11 键；`neighbors(5) == [4, 6]`；`exclusions(&from_exclusion_depth(3))` 的 `[0] == [0,1,2,3]`、`[5] == [2,3,4,5,6,7,8]`、`[11] == [8,9,10,11]`。
- **Edge cases：** 无 `atoms` → `NotFound`；`Block::new()` atoms → `Validation`；`(n,n)` → `Validation`；合法自环被丢；无/空 `bonds` → `Ok` 零边；缺 `atomi`/`atomj` → `Validation`；零尾项 `new([0.0])` 使 C12 的 `exclusions[0]` 含 11；洞表 1-3 不进、1-4 进。
- **Domain validation：** `from_exclusion_depth(3)` 逐位 `[0,0,0,1]`；amber `0.5` 不是豁免。对每个根：`p` 在 `exclusions[r]` 中当且仅当 `distances(r)[p] >= 0` 且 `weight(that) == 0.0`。
- **Regression example：** `regressions/special-bonds-01-molrs.md`，公开 API + C12 三条字面量。门 `cargo test -p molcrafts-molrs --lib --features full,filesystem topology::tests::`。

## Out of scope

- 不改 `SpecialBonds` 的结构体、`Default`、访问器。允许且要求 rustdoc 互指。不设 `From`/`Into`。
- 不把 `intramolecular_pairs` 改建到 `Topology::exclusions` 上。
- 不在 molrs-python 暴露 `Topology` / `BondDistanceWeights`。
- 不做几何：`frame_positions` 留在 molpack。
- 不做版本号与 tag；molpack CI 的 molrs ref 是 operator 手动一步（03 记录）。
- 不动 `from_edges` 的自环行为。
- 不实现 wasm：`WasmTopology::from_frame` 仍读 `i`/`j`。后续必须改为委托 `Topology::from_frame`。
- 不提供 `exclusions(&self, depth)`，不要求私有 `ball` 符号。
