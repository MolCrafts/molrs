---
title: molrs-core 周期盒内自避随机行走 (SARW) 多链 path 生成器
status: code-complete
created: 2026-06-07
depends_on: []
---

# molrs-core 周期盒内自避随机行走 (SARW) 多链 path 生成器

## Summary

在 `molrs-core` 新增一个"生成类算法"子模块 `src/generate/`,用于收集"产出结构/坐标"的生成算法 —— 方向与 `compute` 的 "Frame→分析量" 相反。首个住户是一个**周期盒内、定键长、自避随机行走 (self-avoiding random walk, SARW) 的多链 path 生成器**。用户构造一个 `SelfAvoidingWalk` 配置 struct(字面量构造,泛型注入策略 struct),调用 `.generate()`,得到每条链一串 3D 点(`Vec<Vec<F3>>`)以及所用的 `SimBox`。本模块**只产出路径**:不产 `Frame`、不产 `Topology`、不产键、不带任何化学参数(质量/PE/united-atom)、不做任何 IO。下游(如 molpy 的 `PolymerBuilder`)拿到 path 后自行贴拓扑与原子类型。

## Domain basis

**来源(Why):** 本算法移植自 CAVS LAMMPS 教程的 `mc_gen.c`(Mark A. Tschopp & Don K. Ward),其原始版本用 "FCC lattice + self-avoiding random walk" 生成无定形聚乙烯 (PE) 的初始结构。本 spec 只移植其**路径生成内核**,并主动剥离三层:

1. **剥化学** —— 原始代码里的 PE 单体质量 14.02、united-atom 类型、势能参数全部下放为"调用方关心的下游参数",本模块对此一无所知。
2. **剥 IO** —— 原始代码直接写 LAMMPS data 文件;本模块不写任何文件,文件 IO 归 `molrs-io` 负责。
3. **降级 FCC** —— 把 "FCC 格点" 从"唯一做法"降级为 `GrowthStrategy` 的一种 struct 实现,与连续空间的 `OffLattice` 并列。

**归宿决策链(Why):**

- 否决"新建顶层 crate":单算法不配占一个顶层 crate。
- 否决 `molrs-compute`:那是 "Frame→分析量" 的分析方向,生成器是反方向。
- 否决 `molrs-embed`:其 distance-geometry/ETKDG 面向单分子构象嵌入,范式不符。
- **选定 `molrs-core` 子模块** `src/generate/`,作为今后生成类算法的归集点。

**自避随机行走基础:** SARW 是格点/连续统计物理中聚合物链的标准模型 —— 链段以固定键长逐步生长,且任意两个单体(经最小镜像、跨所有链)不得相互重叠。FCC 实现靠"步进落在格点上"廉价保证不重叠;off-lattice 实现靠"排除体积半径"保证不重叠。`license` 不纳入范围(算法重写,不照抄 `mc_gen.c` 源码)。

## Design

### 模块布局

- `molrs-core/src/generate/mod.rs` —— 声明子模块并 re-export 公共类型(`SelfAvoidingWalk`、`GrowthStrategy`、`FccLattice`、`OffLattice`、`WalkOutput`、错误类型)。
- `molrs-core/src/generate/walk.rs` —— `SelfAvoidingWalk<S>` 配置 struct、`GrowthStrategy` trait、`.generate()` 方法、`WalkOutput`、`WalkError`。
- `molrs-core/src/generate/strategy.rs` —— `FccLattice`、`OffLattice { excluded_radius: F }` 两个 struct 及其 `GrowthStrategy` 实现。
- `molrs-core/src/lib.rs` —— 新增 `pub mod generate;`(沿用现有 `pub mod <name>;` 模式)。

### 复用的真实 molrs-core API(精确签名)

类型(`molrs-core/src/types.rs`):`F = f64`;`F3 = Array1<F>`(一个 3D 点);`FNx3 = Array2<F>`;`F3View<'a> = ArrayView1<'a, F>`;`FNx3View<'a> = ArrayView2<'a, F>`;`Pbc3 = [bool; 3]`。crate extern 名为 `molrs_core`。

- `SimBox::cube(a: F, origin: F3, pbc: Pbc3) -> Result<Self, BoxError>` —— 构造立方盒。
- `SimBox::volume(&self) -> F`;`SimBox::lengths(&self) -> F3`。
- `SimBox::shortest_vector_impl(&self, a: [F;3], b: [F;3]) -> [F;3]` —— 最小镜像位移,PBC-aware。
- `SimBox::calc_distance2(&self, a: F3View, b: F3View) -> F` —— PBC 平方距离。
- `SimBox::wrap(&self, xyz: FNx3View) -> FNx3` —— 把点裹回盒内。
- `NeighborQuery::new(simbox: &SimBox, points: FNx3View, cutoff: F) -> Self`;`.query(&self, query_points: FNx3View) -> NeighborList`;`.query_self(&self) -> NeighborList`。
- `NeighborList::n_pairs() -> usize`、`.dist_sq() -> &[F]`、`.point_indices() -> &[u32]`、`.query_point_indices() -> &[u32]`。
- 增量、零分配的重叠检测:`LinkCell` + `PairVisitor::visit_pair(&mut self, i:u32, j:u32, dist_sq:F, diff:[F;3])`。
- 3D 几何:`molrs::math` 的 `norm3`、`cross3`。

### RNG(需新增依赖)

`rand = "0.9"` 当前仅在 `molrs-core/Cargo.toml` 的 `[dev-dependencies]`,`[dependencies]` 中**没有** `rand`。本 spec 需把 `rand = "0.9"` 提升进 `[dependencies]`。使用可种子化的确定性 RNG(`rand::rngs::StdRng` + `SeedableRng::seed_from_u64(seed)`),使同一 `seed` 给出可复现路径 —— 确定性是测试的硬要求。

### `GrowthStrategy` trait + 占用判定(occupancy,**禁距离/neighborlist**)

**核心约束(用户强约束):重叠与局部密度一律由 lattice/grid 占用决定,严禁用 neighborlist 或距离判定。** 策略只提供几何(提议候选点),占用判定由驱动统一做。trait 形状:

```rust
pub trait GrowthStrategy {
    /// 该策略用哪种占用模型判重叠。
    fn occupancy_mode(&self, bond_length: F) -> OccupancyMode;
    /// 可选:把立方盒边长放大到格点周期(默认不变)。
    fn adjust_box_edge(&self, edge: F, bond_length: F) -> F { edge }
    /// 提议首单体(已在盒内)。
    fn propose_first(&self, simbox: &SimBox, bond_length: F, rng: &mut StdRng) -> [F; 3];
    /// 提议从 `tip` 走一个键长的原始候选(边界由驱动处理)。
    fn propose_step(&self, tip: [F; 3], bond_length: F, rng: &mut StdRng) -> [F; 3];
}
```

占用网格 `OccupancyGrid`(`generate/occupancy.rs`,稀疏 `HashSet<[i64;3]>`,**零距离计算**)两种模式:

- `OccupancyMode::SameCell { cell }` —— 仅当候选自身格点已占用才拒。用于格点策略:步进几何已保证两两 ≥ bond_length,网格只需禁止"重占同一格点"。cell 取得足够小使每个格点映射唯一 cell。
- `OccupancyMode::BlockClear { cell }` —— 候选 cell 及其 26 邻域(排除键合 tip 所在 cell)非空即拒,保证两两非键合单体 ≥ cell。

两个策略实现:

- `FccLattice` —— `propose_step` 随机取 FCC 12 最近邻方向之一(按 `bond_length` 缩放);`occupancy_mode = SameCell { cell: bond_length/2 }`;`adjust_box_edge` 把盒长向上取整到 FCC 常规胞整数倍(commensurate,使周期边界处也不重叠)。
- `OffLattice { excluded_radius: F }` —— `propose_step` 在固定键长上采样随机球面方向;`occupancy_mode = BlockClear { cell: excluded_radius }`。要求 `excluded_radius <= bond_length`。

### 配置 struct 与输出契约

```rust
pub struct SelfAvoidingWalk<S: GrowthStrategy> {
    pub n_chains: usize,
    pub chain_length: usize,
    pub bond_length: F,
    pub target_density: F,   // 单体数 / 体积(见下方约定)
    pub pbc: Pbc3,           // 按轴:true=周期(wrap),false=反射
    pub seed: u64,
    pub strategy: S,
}

pub struct WalkOutput {
    pub paths: Vec<Vec<F3>>, // 每条链一串 3D 点
    pub simbox: SimBox,
}

impl<S: GrowthStrategy> SelfAvoidingWalk<S> {
    pub fn generate(&self) -> Result<WalkOutput, WalkError> { /* ... */ }
}
```

**严禁工厂函数 / 自由构造函数:** 无 `make_walk(...)`、无 `fn new_fcc(...) -> SelfAvoidingWalk` 风格的自由工厂。配置一律 struct 字面量 + 方法;策略一律泛型 struct 注入(`strategy: S`)。

### 盒子推导约定(box-from-density)

质量在范围外,故 `target_density` 的单位为**单体数 / 体积**(monomers-per-volume),需在文档中显式声明此约定。令 `n_total = n_chains * chain_length`,基准立方盒边长 `a = (n_total / target_density).cbrt()`,再经 `strategy.adjust_box_edge` 调整(`OffLattice` 不变;`FccLattice` 向上取整到格点周期),经 `SimBox::cube(edge, [0,0,0], pbc)` 构造。`OffLattice`:`simbox.volume() == n_total / target_density`(容差内);`FccLattice`:盒为 ≥ 该体积的最小 commensurate 立方盒(实际密度 ≤ 请求值)。

### 边界条件(per-axis Pbc3)与自避

- **边界:** `pbc: Pbc3` 进 config。生长时驱动对每个候选逐轴处理 —— 周期轴 `rem_euclid` wrap 到另一侧;非周期轴对**步矢量法向分量取反**做弹性反射(保键长精确)。**输出坐标恒在盒内**(周期轴已 wrap、反射轴已反射)。
- **自避:** 全由 `OccupancyGrid` 占用决定(见上),**不计算任何距离、不建 neighborlist**。周期轴的占用邻域索引按 `rem_euclid` 环绕,反射轴 clamp,故周期边界处的重叠也被正确禁止。
- 键长不变量:周期 wrap 下相邻单体最小镜像距离 = `bond_length`;反射下步矢量法向取反保 `|bond|`。护栏 `edge > 2*bond_length` 保证。

### 回退 / dead-end 策略

SARW 会自陷。定义确定性策略:每步至多尝试 `MAX_STEP_RETRIES` 次提议;耗尽后回退到上一单体并重试至 `MAX_BACKTRACK` 次;若整条链仍失败则重启该链至多 `MAX_CHAIN_RESTARTS` 次;全部耗尽则返回 `WalkError::DeadEnd { chain, monomer }`。所有重试在同一 `seed` 下完全可复现(RNG 状态单调推进,不并行化破坏顺序)。

## Files to create or modify

- `molrs-core/src/generate/mod.rs` (new) —— 子模块声明与 re-export。
- `molrs-core/src/generate/walk.rs` (new) —— `SelfAvoidingWalk`(含 `pbc`)、`GrowthStrategy`、`apply_boundary`、`WalkOutput`、`WalkError`、`generate()`。
- `molrs-core/src/generate/strategy.rs` (new) —— `FccLattice`、`OffLattice` 及其 `GrowthStrategy` 实现。
- `molrs-core/src/generate/occupancy.rs` (new) —— `OccupancyMode`(SameCell/BlockClear)+ `OccupancyGrid`(占用判定,零距离)。
- `molrs-core/src/lib.rs` (modified) —— 新增 `pub mod generate;` 及公共 re-export。
- `molrs-core/Cargo.toml` (modified) —— 把 `rand = "0.9"` 加入 `[dependencies]`。

## Tasks

- [x] Add `rand = "0.9"` to `molrs-core/Cargo.toml` `[dependencies]` and register `pub mod generate;` in `molrs-core/src/lib.rs`
- [x] Write failing tests for `GrowthStrategy` trait, `SelfAvoidingWalk`, and box-from-density convention (`molrs-core/src/generate/walk.rs` #[cfg(test)])
- [x] Implement `GrowthStrategy` trait, `GrowthContext`, `WalkOutput`, `WalkError`, and the box-from-density `SimBox::cube` derivation in `molrs-core/src/generate/walk.rs`
- [x] Write failing tests for `FccLattice` and `OffLattice` strategy structs (`molrs-core/src/generate/strategy.rs` #[cfg(test)])
- [x] Implement `FccLattice` and `OffLattice { excluded_radius }` structs with `GrowthStrategy` impls in `molrs-core/src/generate/strategy.rs`
- [x] Implement `OccupancyMode` + `OccupancyGrid` (SameCell / BlockClear, zero-distance) in `molrs-core/src/generate/occupancy.rs`
- [x] Add per-axis `pbc: Pbc3` to config and `apply_boundary` (periodic wrap / reflective step-flip) in the driver
- [x] Implement `SelfAvoidingWalk::generate()` driving multi-chain growth with deterministic seeded backtracking/restart and dead-end error
- [x] Add rustdoc per repo doc style with units and the explicit density-units convention; re-export public types from `molrs-core/src/generate/mod.rs`
- [x] Run full check + test suite (`cargo fmt --all`, `cargo clippy -- -D warnings`, `cargo test -p molcrafts-molrs-core`)

## Testing strategy

Inline `#[cfg(test)]` 单元测试(纯函数 / 不读 `tests-data/`)。覆盖:

- **Determinism(快乐路径):** 同 `seed` + 同 config 两次 `generate()` → 逐坐标 byte-identical 路径。
- **Bond-length invariant:** 同一链内每对相邻点的距离等于 `bond_length`(off-lattice 容差 `1e-9`;lattice 精确)。
- **Self-avoidance / excluded volume:** `OffLattice` 下任意两个不同单体(跨所有链、最小镜像)距离 ≥ `excluded_radius`(减容差);`FccLattice` 下无两单体占同一格点。
- **Chain count & length:** 输出恰有 `n_chains` 条链,每条恰 `chain_length` 个点。
- **Density / box:** 推导出的 `SimBox::volume()` 在约定下等于 `n_total / target_density`(容差内);所有点落在盒内或可 `wrap` 回盒内。
- **No-factory / struct-injection(代码形状):** 公共 API 暴露 `SelfAvoidingWalk { ... strategy: S }` 由 struct 字面量构造,`GrowthStrategy` impl 全是 struct(`FccLattice`、`OffLattice`);grep 显示无 `fn make_*` / 返回该 config 的自由工厂函数。
- **Purity(架构断言):** `generate` 模块不依赖 `molrs-io`,不构造 `Frame`/`Topology`;`generate()` 返回类型只含 paths + `SimBox`。
- **Edge cases:** `target_density <= 0` / `bond_length <= 0` / `chain_length == 0` → `WalkError`;dead-end 耗尽重试 → `WalkError::DeadEnd`。

## Out of scope

- 产出 `Frame` 或 `Topology`(下游 molpy `PolymerBuilder` 负责)。
- 生成键(bonds)、角、二面角等任何拓扑连接。
- 任何化学参数:质量(如 14.02)、PE 势能、united-atom 原子类型、电荷。
- 任何文件 IO,尤其写 LAMMPS data(归 `molrs-io`)。
- 与 molpy `PolymerBuilder` 的集成。
- 非立方 / 三斜(triclinic)盒子 —— 首版仅立方盒;后续可扩展 `SimBox::ortho`/通用 `h`。(注:**per-axis PBC(周期/反射)已在范围内**,只是盒形仍限立方。)
- `license` 处理(算法重写,不照抄 `mc_gen.c`)。
