# molrs-pack v2 — 组件化与插件系统 Spec

**Status**: Draft v2 (修订版 — 基于现有代码审阅)
**Author**: architect + HPC review
**Date**: 2026-04-13
**Supersedes**: v1 draft（未实现）与当前 molrs-pack 的 Packmol-flavored 单体结构

---

## Progress Checklist

> 每次 `/molrs-impl` 推进时，把对应行从 `[ ]` 改成 `[x]`，并在末尾追加一行 commit 短哈希 + 一句话总结。新一轮工作开始前先读这一节。

**Pre-Phase A — discipline updates** (foundation)
- [x] Spec §0/§9/§10/§12 改写为 extract-bench loop（放弃预冻结 monolithic baseline）
- [x] `molrs-perf` skill 加 § "Benchmarking during refactors" + Performance Budget 替换
- [x] `molrs-optimizer` agent workflow 加 step 4（hot-path 抽取检查清单）
- [x] `molrs-impl` skill 加 "Phase 3 — special case: hot-path refactors"

**Phase A — 内部重构（公开 API 不变）**
- [x] **A.1** 基础设施：`molrs-pack/Cargo.toml` 加 `criterion = "0.5"` + `[[bench]]`；`benches/pack_end_to_end.rs` 5 例灾难警报（mixture/bilayer/interface/solvprotein/spherical）— 复用 `cases::ExampleCase`
- [x] **A.2** rename：`struct Restraint{kind,params}` → `struct BuiltinConstraint`（仅内部；保留 `pub use BuiltinConstraint as Restraint` 一个 phase 的别名，下游测试无改动）
- [x] **A.3** rename：`hook.rs` 的 `Hook`/`HookRunner`/`TorsionMcHook`/`TorsionMcRunner` → `Relaxer`/`RelaxerRunner`/`TorsionMcRelaxer`/`TorsionMcRelaxerRunner`；`Target::hooks` → `relaxers`、`with_hook` → `with_relaxer`；文件 `hook.rs` → `relaxer.rs`、`tests/hook.rs` → `tests/relaxer.rs`；trait/struct 名字保 `pub use ... as Hook/HookRunner/TorsionMcHook` 兼容别名（方法名 `with_relaxer` 无别名，更新 5 个测试 + 1 example 调用点）
- [ ] **A.4** 拆 `packer.rs` → `phase/{per_type, geometric_prefit, main_loop}.rs`；**每抽一个 phase 函数都落 sentinel + 函数微基准 + 调用方微基准**（首次真正应用 extract-bench loop）
    - [x] **A.4.1** 抽 `evaluate_unscaled(sys, xwork) -> (f, fdist, frest)`（消除三处 `radiuswork` swap-evaluate-restore 重复）；落 `evaluate_unscaled_sentinel` + `benches/evaluate_unscaled.rs`（fn + caller 两组微基准）；gate：fn -3.4% vs sentinel（远过 ≤+1%），caller +1.1%（过 ≤+2%）
    - [x] **A.4.3** 抽 inner loop body → `fn run_iteration(loop_idx, ..., &mut sys, ..., &mut rng) -> IterOutcome`（Continue / Converged / EarlyStop）；落 `run_iteration_sentinel` + `benches/run_iteration.rs`（fn + caller 两组）+ 单元测试（空 context 上 fn == sentinel）；`SwapState` 提升 `pub`（bench 需要）；**先于 A.4.2 落**是因为外层 phase body 在 inner loop 抽出后变得 trivial，顺序反过来可以把复杂的一次性做完
    - [x] **A.4.2** 抽 main loop 外层 for 循环 body → `fn run_phase(phase, ntype, ntype_with_fixed, total_phases, max_loops, discale, precision, disable_movebad, &mb_cfg, &gp, &mut sys, &mut x, &mut swap, &mut relaxers, &mut handlers, &mut ws, &mut rng) -> PhaseOutcome`（Continue/Converged — 仅 all-type converged 才打破外层）；落 `run_phase_sentinel` + `benches/run_phase.rs`（fn + caller 两组）+ 单元测试（空 context trivially Converged）；外层 123 行塌缩到 13 行 match-scaffold；gate：fn +0.44%（过 ≤+1%），caller -0.86%（过 ≤+2%），e2e mixture 481→472 ms p=0.24（无变化）；113 tests pass（+1 新测试）
    - [ ] **A.4.4** 进一步拆 setup / geometric prefit（如果仍需要）
- [ ] **A.5** 把 `PackContext::evaluate` 的签名固化为 `trait Objective`；`impl Objective for PackContext`
- [ ] **A.6** `pgencan` 入参 `&mut PackContext` → `&mut dyn Objective`
- [ ] **A.7** 验收：所有现有测试通过；Packmol 等价回归 0 失败；微基准 gate 满足 §10；`pack_end_to_end.rs` 未触发 +10% 灾难警报

**Phase B — 公开新 trait，并行新旧 API**
- [ ] **B.1** 在 `molrs-pack` 公开 `Constraint` / `Region` / `Restraint` / `Selector` trait
- [ ] **B.2** opaque builder：`InsideBox::new(...)` 返回不暴露 enum 的构造器，内部填 `BuiltinConstraint`
- [ ] **B.3** `Molpack::add_constraint(impl Into<ConstraintInput>)` — 接受 enum / `Box<dyn Constraint>` / Python
- [ ] **B.4** 老 `add_restraint(Restraint)` 标 `#[deprecated(note = "use add_constraint with the new Constraint trait")]`
- [ ] **B.5** `Handler` trait 加入 §6.6 的新默认方法
- [ ] **B.6** 验收：用户能用纯 Rust 写自定义 Constraint 并跑通；基线无退化

**Phase C — Python 桥**
- [ ] **C.1** `molrs-python` 新建 `pack` 子模块
- [ ] **C.2** 暴露 `Constraint` / `Restraint` / `Handler` Python ABC
- [ ] **C.3** PyO3 桥（`PyConstraintImpl: Constraint`），`is_parallel_safe = false`
- [ ] **C.4** 编译时分区（`ConstraintSet` 三段 CSR）+ 单 GIL 批量调用
- [ ] **C.5** Python e2e：自定义 Restraint 跑通完整 packing
- [ ] **C.6** 验收：Jupyter 50 行定义并运行自定义约束；基线无退化

**Commit log** (newest last; one line per landed step)
- `pre-A` — spec/skill/agent 落地 extract-bench loop 纪律
- `A.1` — bench infra + 5-example 灾难警报（复用 `ExampleCase`，无 workload 重复）
- `A.2` — `Restraint` → `BuiltinConstraint` 重命名；`pub use ... as Restraint` 保 API；`pack_mixture` 478→482 ms（噪声内）
- `A.3` — `Hook`/`HookRunner`/`TorsionMcHook` → `Relaxer`/`RelaxerRunner`/`TorsionMcRelaxer`；`hook.rs` → `relaxer.rs`；trait 别名保 API；`pack_mixture` 472→480 ms（噪声内，p>0.05）
- `A.4.1` — 抽 `evaluate_unscaled` 去三处重复；落 sentinel + `benches/evaluate_unscaled.rs` + 单元测试（首个走完 extract-bench loop 全纪律的 commit）；fn gate -3.4%、caller gate +1.1%、e2e 467→484 ms（p=0.08, 噪声内）
- `A.4.3` — 抽 inner loop body → `run_iteration`（140 行移到纯函数）；落 `run_iteration_sentinel` + `benches/run_iteration.rs` + 空-context 单元测试；`SwapState` pub 化；fn gate -2.4%（过 ≤+1%），caller gate -0.3%（过 ≤+2%），e2e 484→481 ms p=0.57（无变化）；112 tests pass（+1 新测试）；先于 A.4.2 — phase body 在此之后变 thin wrapper
- `A.4.2` — 抽 outer phase body → `run_phase`（123 行 → 13 行 match-scaffold）；落 `run_phase_sentinel` + `benches/run_phase.rs` + 空-context 单元测试；新增 `PhaseOutcome { Continue, Converged }`；gate：fn +0.44%（过 ≤+1%），caller -0.86%（过 ≤+2%），e2e mixture 481→472 ms p=0.24（无变化）；113 tests pass（+1 新测试）；`pack()` 主循环现在是纯 scaffold

---

## 0. 修订要点（相对 v1）

v1 忽略了当前代码已经做到的部分，并引入了与现有类型重名的 trait。v2 的核心修订：

1. 承认并复用 `PackContext::evaluate(x, EvalMode, Option<&mut g>)`、CSR（`iratom_offsets`/`iratom_data`）、`init1`、`comptype`、`accumulate_pair_f_parallel` 等既有机制。
2. 命名冲突修正：现有 `Hook`（几何修改器）保留并改名为 `Relaxer`（可以是 MC / MD / 梯度下降，不限 MC）；新的观察者能力**不引入新 trait**，改为扩展现有 `Handler`。
3. 双 `scale` / `scale2` 行为写进 trait 签名；per-atom 副作用（`fdist_atom`/`frest_atom`，movebad 所需）写进输出契约。
4. Geometry cache（`matches_cached_geometry` 快速路径）作为 `Objective` trait 的显式契约。
5. 内置约束强制静态 dispatch；`dyn Constraint` 仅保留给用户扩展与 Python 桥。
6. 性能门禁量化，但**放弃**预冻结 monolithic baseline 的方案。改为 **extract-bench loop**：每次抽出一个纯函数 F，同一 commit 内落 `#[cfg(bench)] #[inline(never)] F_sentinel` + F 的微基准 + 调用方微基准，F vs. sentinel ≤ +1% 作为硬门禁。详见 `molrs-perf` skill § "Benchmarking during refactors" 与本 spec §10。
7. Phase A 不再有 "预冻结 baseline" 的 step 0；bench 与 refactor 在同一 commit 内并行进行。

---

## 1. 目标

1. 把 molrs-pack 从 Packmol 单体 port 重构为**可扩展框架**，可扩展点通过 trait 暴露。
2. 严格区分 **Constraint（硬几何约束）** 与 **Restraint（软惩罚偏置）**。
3. 支持 **Python 注入** Constraint/Restraint/Handler，无需重新编译 Rust。
4. 保持 Packmol 行为等价性（现有回归测试不退化）。
5. **性能不退化**（量化定义见 §10）。

## 2. 非目标

- 不引入 CLI 或 YAML/TOML 配置文件——所有装配在 Rust 或 Python 代码里。
- 不做 `dlopen` 动态库插件——破坏 wasm 目标，且 inventory 已能覆盖 Rust 端。
- 不重写 GENCAN 算法本身——只解耦它与 PackContext。
- 不改 Packmol Phase 0/1/2 算法语义——只重组代码结构。
- 不新建 `molrs-opt` crate。GENCAN 暂留 `molrs-pack/src/optimizer/`；若将来有第二个用户再抽。

---

## 3. 现状盘点（重要：不要再造轮子）

审阅 `molrs-pack/src/` 后，以下能力**已存在**，v2 重构只做封装/重命名/暴露：

| 能力 | 现有实现位置 | v2 动作 |
|---|---|---|
| 统一 eval API | `PackContext::evaluate(x, EvalMode, Option<&mut [F]>) -> EvalOutput`（`context/pack_context.rs:278`） | 抽到 `trait Objective`，保持签名语义 |
| EvalMode 枚举 | `constraints::{EvalMode, EvalOutput}` | 作为 Objective 输入 |
| 每原子约束 CSR | `iratom_offsets: Vec<usize>` / `iratom_data: Vec<RestraintRef>`（`pack_context.rs:120-122`） | 扩展为三段分区（见 §6.3） |
| Phase 1 跳过 pair kernel | `init1: bool` 标志（`objective.rs:36,56,173`） | 重命名 `Phase::GeometricPrefit`，语义不变 |
| Phase 0 只动一个 type | `comptype: Vec<bool>`（`objective.rs:191-194`） | 包进 `Phase::PerType` |
| cell-level rayon 并行 | `accumulate_pair_f_parallel`（`objective.rs:347-377`）+ `PARALLEL_PAIR_CELL_THRESHOLD = 64` | 保留；约束并行层不替换它 |
| Geometry cache 快速路径 | `matches_cached_geometry` + `update_cached_geometry`（`objective.rs:158,710-734`） | 写入 `Objective` trait 契约 |
| GENCAN workspace 复用 | `GencanWorkspace::ensure_len`（`gencan/mod.rs:76`） | 保留 |
| Per-atom violation 累积 | `fdist_atom` / `frest_atom` + `move_flag`（`objective.rs:136-143, 249-251`）— movebad 需要 | 作为 eval 输出契约的一部分 |
| 双邻居次序 | `neighbor_cells_f` / `neighbor_cells_g`（`pack_context.rs:159-161`）— Packmol computef/computeg 用不同次序保证梯度对称 | 保留，不合并 |
| 几何修改 hook | `trait Hook` + `trait HookRunner`（`hook.rs`）— 当前实现 `TorsionMcHook` 在 movebad 与 pgencan 之间运行 | **重命名为 `Relaxer`/`RelaxerRunner`**，职责不变 |
| 进度观察者 | `trait Handler` + `PhaseInfo`/`StepInfo`（`handler.rs`）：`on_start/on_initial/on_step/on_phase_start/on_finish/should_stop` | **扩展**（不新增 trait），见 §6.6 |
| 装配 builder | `Molpack::new().add_handler(...).pack(targets)`（`api/mod.rs`, `packer.rs`） | 保留 `Molpack` 名；新增 `add_constraint/add_restraint` |

**禁止**：v1 spec 里写的 "引入 Objective trait" / "引入 Hook 观察者 trait" / "引入 CSR" / "引入 rayon" 表述必须改成 "扩展 / 抽出 / 暴露已有的 X"。

---

## 4. 术语：四类可扩展点

| 维度 | **Constraint** | **Restraint** | **Relaxer** | **Handler** |
|---|---|---|---|---|
| 数学定义 | 区域归属 `g(x) ≤ 0`，违反必须惩罚到零 | 任意 penalty `f(x)`，加权进目标函数 | 修改分子参考坐标（构型松弛：MC / MD / 梯度下降 均可） | 只读回调 |
| 是否必须满足 | 是 | 否 | 不参与目标函数 | 不参与目标函数 |
| 影响 `nloop` 终止 | 是（`fdist`/`frest` 精度判据） | 否 | 否 | 可 (`should_stop()`) |
| 调用频率 | per atom, 每 eval | per restraint, 每 eval | 每 outer loop（movebad 之后） | 每 phase / 每 iter |
| 性能要求 | 极高（10⁵-10⁷ 次 / run） | 中（M × iter） | 低（~10² / run） | 极低 |
| 对应 Packmol 概念 | `restraint kind 1..15` + `tolerance` | （无原生） | （无原生） | 打印/进度 |
| 是否允许 Python 实现 | 允许但慢，文档告警 | 推荐 | 允许 | 推荐 |
| 现有代码中的对应 | `Restraint{kind,params[9]}` + `iratom_*` CSR | 无 | `hook.rs` 的 `Hook`/`HookRunner`（当前仅 MC 实现） | `handler.rs` 的 `Handler` |
| v2 名字 | `Constraint` | `Restraint`（新） | `Relaxer`（改名；支持 MC/MD/梯度下降等多策略） | `Handler`（扩展） |

**注意命名迁移**：当前代码的 `struct Restraint{kind,params}` 是"硬约束"，**应该叫 Constraint**。v1 spec 混淆了"Packmol 原生 restraint"（硬的）与"引入 Restraint"（软的），v2 统一如下：

- 旧 `struct Restraint{kind: u8}` → 内部重命名 `BuiltinConstraint`
- 新 trait `Constraint` 包住所有硬约束（内置 + 用户 + Python）
- 新 trait `Restraint` 引入软偏置

---

## 5. 核心分解

| # | 组件 | 类型 | 职责 | 所在位置 |
|---|---|---|---|---|
| 1 | `Region` | trait | 几何谓词 `contains / signed_distance` | `molrs-pack/src/region/` |
| 2 | `Constraint` | trait | 硬约束；与 `Restraint{kind}` 等价但支持用户扩展 | `molrs-pack/src/constraint/` |
| 3 | `BuiltinConstraint` | enum | 15 种内置约束的静态-dispatch 版本 | 私有 |
| 4 | `ConstraintSet` | struct | 编译时分区（builtin / user-rust / user-python）+ CSR | 私有 |
| 5 | `PairKernel` | 内部函数（不是 trait） | Packmol 原生 pair distance，走 cell list | `objective.rs` 现有 `fparc/gparc/fgparc`，保留 |
| 6 | `Restraint` | trait | 软偏置 `evaluate / gradient + weight + selector` | `molrs-pack/src/restraint/` |
| 7 | `Selector` | trait | 选原子子集（编译时 resolve 成 `Vec<u32>`） | `molrs-pack/src/restraint/selector.rs` |
| 8 | `Relaxer` + `RelaxerRunner` | trait（重命名现有） | 分子构型松弛（torsion MC / 内部 MD / 梯度下降等） | `hook.rs` 现有，rename |
| 9 | `Handler` | trait（扩展现有） | 观察者、早停 | `handler.rs` 现有，新增若干默认方法 |
| 10 | `Phase` | enum（非 trait） | `PerType / GeometricPrefit / MainLoop` 标识 | `molrs-pack/src/phase/` |
| 11 | `Objective` | trait | `PackContext::evaluate` 的抽象；GENCAN 依赖它而非 PackContext | `molrs-pack/src/objective/trait.rs` |
| 12 | `Optimizer` | trait | `solve(&mut dyn Objective, x, ws)`；GENCAN 是唯一内置实现 | `molrs-pack/src/optimizer/` |
| 13 | `Molpack` | struct（保留现名） | 装配 → 编译 → 运行 Phase[] → 输出 | `packer.rs` |

**Phase 作为 enum 而非 trait**：当前 Phase 0/1/2 语义固定（per-type compaction / init1 prefit / main loop），用户不扩展它们。YAGNI：不开 trait。

---

## 6. Trait 签名（关键）

### 6.1 Region

```rust
// molrs-pack/src/region/mod.rs
pub trait Region: Send + Sync {
    fn contains(&self, x: &[F; 3]) -> bool;
    fn signed_distance(&self, x: &[F; 3]) -> F;   // <0 内, >0 外
    fn bounding_box(&self) -> Option<BBox> { None }
}

// 组合子（纯类型代数，零运行时成本）
impl<A: Region, B: Region> Region for And<A, B> { /* ... */ }
impl<A: Region, B: Region> Region for Or<A, B>  { /* ... */ }
impl<A: Region>            Region for Not<A>    { /* ... */ }
```

### 6.2 Constraint（保留双 scale 与 per-atom 副作用）

```rust
// molrs-pack/src/constraint/mod.rs

/// Per-atom violation tracking for movebad heuristic.
/// Currently lives as `frest_atom[icart]` side-effect in objective.rs;
/// v2 makes it an explicit output of each Constraint call.
#[derive(Default, Clone, Copy)]
pub struct AtomViolation {
    pub frest_delta: F,   // add to frest_atom[icart]; max-in to frest
}

pub trait Constraint: Send + Sync {
    /// Function value only (used by line-search interpolation path).
    /// Maps to existing `Restraint::value(pos, scale, scale2)`.
    fn f(&self, x: &[F; 3], scale: F, scale2: F) -> F;

    /// Fused function + gradient. Hot path.
    /// Maps to existing `Restraint::value` + `Restraint::gradient` called
    /// together in `ExpandMode::FG`.
    fn fg(&self, x: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F;

    /// Parallel safety. `false` ⇒ scheduler serializes this constraint.
    /// Python-backed constraints MUST return false.
    fn is_parallel_safe(&self) -> bool { true }

    fn name(&self) -> &'static str { std::any::type_name::<Self>() }
}

/// Any Region lifts to a Constraint with quadratic exterior penalty.
/// `penalty(x) = scale2 * max(0, signed_distance(x))²`
impl<R: Region + 'static> Constraint for FromRegion<R> { /* ... */ }
```

**双 scale 契约**：Packmol 约定线性惩罚（box/plane/cube: type 2,3,6,7,10,11）用 `scale`，平方惩罚（sphere/ellipsoid/cylinder/gaussian: type 4,5,8,9,12–15）用 `scale2`。`Constraint` trait 同时接收两者，每个实现自己决定用哪个；保证与 `constraint.rs` 现有 match 分支等价。

### 6.3 ConstraintSet（静态 dispatch 内置 + dyn 扩展）

**这是 v2 最关键的性能保证点**。硬性要求：内置 15 种约束绝不走 `Box<dyn>`。

```rust
// molrs-pack/src/constraint/set.rs

/// Tagged union mirroring current `Restraint{kind:u8, params:[F;9]}`.
/// Do NOT box this; it lives inline in `ConstraintSet::builtin_*` vectors.
#[derive(Clone, Copy)]
pub(crate) struct BuiltinConstraint {
    pub kind: u8,           // 2..=15, same numbering as Packmol
    pub params: [F; 9],
}
impl BuiltinConstraint {
    #[inline(always)]
    pub fn f(&self, x: &[F;3], s: F, s2: F) -> F { /* match-on-kind, inlined */ }
    #[inline(always)]
    pub fn fg(&self, x: &[F;3], s: F, s2: F, g: &mut [F;3]) -> F { /* same */ }
}

/// Compiled constraint layout — partition done ONCE in Molpack::compile().
///
/// Each section has its own CSR. Hot path iterates `builtin` inline,
/// then `custom_rust` via vtable, then `python` serially under one GIL.
pub(crate) struct ConstraintSet {
    // Built-in: static dispatch, AoS of [F;9].
    builtin: Vec<BuiltinConstraint>,
    builtin_csr_offsets: Vec<u32>,   // len = ntotat + 1
    builtin_csr_data:    Vec<u32>,   // indices into `builtin`

    // User Rust: dyn dispatch, boxed.
    custom_rust: Vec<Box<dyn Constraint>>,
    custom_rust_csr_offsets: Vec<u32>,
    custom_rust_csr_data:    Vec<u32>,

    // Python: GIL-serialized.
    python: Vec<PyConstraintImpl>,
    python_csr_offsets: Vec<u32>,
    python_csr_data:    Vec<u32>,
}
```

**禁止项（spec 级约束）**：
- ❌ `Vec<Box<dyn Constraint>>` 作为内置约束的统一容器
- ❌ 热路径里做 `constraints.iter().partition(...)`
- ❌ 每原子一次 GIL acquire
- ✅ 编译时一次 partition；每 eval 一次 GIL acquire 包住全部 Python 约束

### 6.4 Restraint（软偏置，新能力）

```rust
// molrs-pack/src/restraint/mod.rs

pub trait Restraint: Send + Sync {
    /// Resolved atom indices — called once at compile time, NOT per eval.
    fn selector(&self) -> &dyn Selector;

    /// Evaluate over the atoms this restraint acts on.
    /// `atoms` is a borrowed view over `PackContext::xcart[selected]`.
    fn evaluate(&self, atoms: &[[F; 3]]) -> F;

    /// Accumulate gradient into `g` (length = atoms.len()).
    fn gradient(&self, atoms: &[[F; 3]], g: &mut [[F; 3]]) -> F;

    fn weight(&self) -> F { 1.0 }
    fn is_parallel_safe(&self) -> bool { true }
}

// Built-in:
pub struct DistanceRestraint    { i: u32, j: u32, target: F, k: F }
pub struct PositionRestraint    { i: u32, target: [F; 3], k: F }
pub struct OrientationRestraint { mol_id: u32, axis: [F;3], target: [F;3], k: F }
```

`Selector::select` 只在 `Molpack::compile()` 时被调用一次，结果 `Vec<u32>` 存进 compiled Restraint。不走热路径。

### 6.5 Objective（抽自 `PackContext::evaluate`）

```rust
// molrs-pack/src/objective/trait.rs

/// Abstract over what GENCAN sees. Direct generalization of the existing
/// `PackContext::evaluate(x, mode, g) -> EvalOutput` at pack_context.rs:278.
pub trait Objective {
    fn n(&self) -> usize;

    /// Value only. Maps to existing `compute_f` (objective.rs:25).
    fn f(&mut self, x: &[F]) -> F;

    /// Gradient only (assumes f was computed on current x, exploits
    /// `matches_cached_geometry` fast path at objective.rs:158).
    fn g(&mut self, x: &[F], g: &mut [F]);

    /// Fused f+g. Maps to existing `compute_fg` (objective.rs:49).
    fn fg(&mut self, x: &[F], g: &mut [F]) -> F;

    /// Bounds for projected variables (Euler angles when `constrain_rotation`
    /// is set). Maps to existing `gencan::build_bounds`.
    fn bounds(&self, l: &mut [F], u: &mut [F]);

    /// Read-only access to Packmol convergence state (fdist/frest) for
    /// `packmolprecision` check. Keeps GENCAN decoupled from PackContext
    /// internals but preserves the termination predicate.
    fn precision_check(&self, precision: F) -> bool;
}

impl Objective for PackContext { /* thin wrapper over existing methods */ }
```

**Geometry cache 契约**：`g()` 的实现**必须**沿用 `matches_cached_geometry` 快速路径。trait 文档明说：调用方保证 `g(x)` 紧跟 `f(x)` 时，`x` 不变，可跳过 Cartesian 展开。

### 6.6 Handler（扩展现有 trait，不新增）

现有 `trait Handler` 已覆盖观察者角色。v2 仅在该 trait 上增加 **默认实现** 的新方法，不破坏已有 impl：

```rust
// handler.rs —— 新增方法（全部有 default impl，向后兼容）

pub trait Handler: Send {
    // === 现有方法保留 ===
    fn on_start(&mut self, _ntotat: usize, _ntotmol: usize) {}
    fn on_initial(&mut self, _sys: &PackContext) {}
    fn on_step(&mut self, info: &StepInfo, sys: &PackContext);
    fn on_phase_start(&mut self, _info: &PhaseInfo) {}
    fn on_finish(&mut self, _sys: &PackContext) {}
    fn should_stop(&self) -> bool { false }

    // === v2 新增（全部 default impl）===

    /// Called after each inner GENCAN iteration (more granular than on_step).
    fn on_inner_iter(&mut self, _iter: u32, _f: F, _sys: &PackContext) {}

    /// Called at end of each phase with summary.
    fn on_phase_end(&mut self, _info: &PhaseInfo, _report: &PhaseReport) {}
}
```

**不新建 `Hook` 观察者 trait**。v1 spec 提议的 `Hook` trait 与现有 `handler::Handler` 职责 100% 重叠，且与现有 `hook::Hook`（几何修改器）命名冲突。

### 6.7 Relaxer（重命名现有 `Hook`）

`Relaxer` 语义 = "对分子构型做松弛"，不限定实现方式 — 可以是 MC 采样、短 MD 步、梯度下降 minimizer 等。命名迁移：

- `hook.rs` 的 `Hook` → `Relaxer`
- `HookRunner` → `RelaxerRunner`
- `TorsionMcHook` → `TorsionMcRelaxer`（保留原算法，只改名）
- `TorsionMcRunner` → `TorsionMcRelaxerRunner`
- `Target::with_hook(...)` → `Target::with_relaxer(...)`

所有调用点一起改。public API 的旧名字保留一个 Phase 窗口，标 `#[deprecated]`，下一版删除。

Trait 契约允许任何策略 — 只要在 `on_iter` 里返回"更松弛"的坐标即可，接受/拒绝由 Metropolis、能量阈值、或直接无条件接受（梯度下降时）自行决定：

```rust
pub trait Relaxer: Send + Sync + CloneRelaxer {
    fn build(&self, ref_coords: &[[F; 3]]) -> Box<dyn RelaxerRunner>;
}

pub trait RelaxerRunner: Send {
    /// Called between movebad and pgencan each outer iteration.
    /// Returns Some(new_coords) if any improvement accepted, None otherwise.
    /// Implementation is free to use MC / MD / gradient-descent internally.
    fn on_iter(
        &mut self,
        coords: &[[F; 3]],
        f_current: F,
        evaluate: &mut dyn FnMut(&[[F; 3]]) -> F,
        rng: &mut dyn RngCore,
    ) -> Option<Vec<[F; 3]>>;

    fn acceptance_rate(&self) -> F { 0.0 }    // 对梯度式实现可返回 NaN / 固定值
}
```

未来可能的内置实现（不在本 spec 范围内）：`TorsionMdRelaxer`（短 MD 热浴）、`TorsionGradientRelaxer`（内部梯度最小化）、`RingFlipRelaxer`。

### 6.8 Optimizer

```rust
// molrs-pack/src/optimizer/mod.rs （保留在 molrs-pack 内，不独立 crate）

pub trait Optimizer {
    fn solve(&mut self, obj: &mut dyn Objective, x: &mut [F]) -> OptReport;
}

pub struct GencanOptimizer {
    workspace: GencanWorkspace,    // 现有的，零变化
    params: GencanParams,
}
impl Optimizer for GencanOptimizer { /* wrap existing pgencan(x, sys, ...) */ }
```

### 6.9 装配 API

```rust
let result = Molpack::new()
    .tolerance(2.0)
    .seed(42)
    // 硬约束
    .add_constraint(InsideBox::from_simbox(&simbox))
    .add_constraint(OutsideSphere::new([0.0; 3], 5.0))
    .add_constraint(MyCustomRustConstraint::new(...))
    .add_constraint(py_constraint)
    // Region 组合自动转 Constraint
    .add_constraint(InsideSphere::new(c, r1).and(OutsideSphere::new(c, r2)))
    // 软偏置
    .add_restraint(DistanceRestraint::new(atom_a, atom_b, 1.5, k=10.0))
    .add_restraint(py_orientation_bias)
    // 观察者（扩展既有 Handler）
    .add_handler(ProgressHandler::new())
    .add_handler(XYZHandler::to_path("traj.xyz"))
    // 分子构型松弛（per-target，保留现有 API）
    .target(mol_a).with_relaxer(TorsionMcRelaxer::new(&graph))
    .pack(&targets)?;
```

---

## 7. Python 桥

### 7.1 设计原则

1. **避免假零拷贝承诺**：`PyArray1::from_slice(py, &x[..3])` 实际会拷贝 3 个 `f64`。per-atom Constraint 改用**元组传参**，比 numpy 快一个数量级。Restraint（per-selector, M 个原子）才值得用 numpy view。
2. **GIL 隔离**：`is_parallel_safe = false` 的组件走独立路径，编译时 partition（见 §6.3 `ConstraintSet`）。
3. **单次 GIL 持有**：整个 `evaluate_constraints` 对所有 Python 约束只 acquire 一次 GIL，loop 在 GIL 内。
4. **错误透传**：Python `raise` → Rust `Err(PackError::PythonCallback(msg))`，不 panic。

### 7.2 Python 端接口

```python
from molrs.pack import Constraint, Restraint, Handler, Region

class MyHelix(Constraint):
    def __init__(self, radius, pitch):
        self.r, self.p = radius, pitch

    # Per-atom: 元组传参，避免 numpy 开销
    def fg(self, x0: float, x1: float, x2: float,
           scale: float, scale2: float
          ) -> tuple[float, float, float, float]:
        # returns (f, g0, g1, g2)
        ...

class MyDistanceRestraint(Restraint):
    def selector(self):
        return Indices([10, 20])

    # Per-selector: numpy view 合理
    def evaluate(self, coords: np.ndarray) -> float: ...
    def gradient(self, coords: np.ndarray, out: np.ndarray) -> float: ...
```

### 7.3 Rust 桥接

```rust
// molrs-python/src/pack/constraint.rs
pub struct PyConstraintImpl { obj: PyObject, name: String }

impl Constraint for PyConstraintImpl {
    fn fg(&self, x: &[F;3], scale: F, scale2: F, g: &mut [F;3]) -> F {
        // Caller holds the GIL (dispatcher ensures); no per-call Python::with_gil.
        let tuple = unsafe { PY_GIL_TOKEN.assume() };   // via scheduler contract
        self.obj.call_method1(
            tuple, "fg", (x[0], x[1], x[2], scale, scale2)
        )
        .and_then(|v| v.extract::<(F, F, F, F)>(tuple))
        .map(|(f, g0, g1, g2)| { g[0]+=g0; g[1]+=g1; g[2]+=g2; f })
        .unwrap_or(F::INFINITY)    // Python 异常 → 视为巨大违反
    }

    fn is_parallel_safe(&self) -> bool { false }
}
```

### 7.4 调度器（编译时分区，非热路径分区）

```rust
// molrs-pack/src/objective/dispatch.rs

impl ConstraintSet {
    /// Hot path. Called per atom in `accumulate_constraint_value/gradient`.
    /// The three sections are built ONCE by Molpack::compile().
    #[inline]
    pub fn fg_atom(&self, icart: u32, x: &[F;3], s: F, s2: F, g: &mut [F;3]) -> F {
        let mut f = 0.0;

        // 1) Builtin: static dispatch via match-on-kind.
        let r = self.builtin_csr_offsets[icart as usize] as usize
             ..self.builtin_csr_offsets[icart as usize + 1] as usize;
        for &idx in &self.builtin_csr_data[r] {
            f += self.builtin[idx as usize].fg(x, s, s2, g);
        }

        // 2) Custom Rust: vtable, but cold unless user extended.
        let r = self.custom_rust_csr_offsets[icart as usize] as usize
             ..self.custom_rust_csr_offsets[icart as usize + 1] as usize;
        for &idx in &self.custom_rust_csr_data[r] {
            f += self.custom_rust[idx as usize].fg(x, s, s2, g);
        }

        // 3) Python: caller holds GIL for the entire outer loop (see below).
        let r = self.python_csr_offsets[icart as usize] as usize
             ..self.python_csr_offsets[icart as usize + 1] as usize;
        for &idx in &self.python_csr_data[r] {
            f += self.python[idx as usize].fg(x, s, s2, g);
        }

        f
    }
}
```

Python 路径在外层包一次 `Python::with_gil`；若 `python` section 为空则完全绕开 GIL 代码。

### 7.5 Python 性能预算（atom 数从 1000 起）

| 场景 | 调用次数 / pack run | 单次开销 | 预期总时间 | 可接受 |
|---|---|---|---|---|
| 1 个 Python Constraint，N=1000 | ~10⁶ | ~1µs（元组调用） | ~1s | ⚠️ 原型 ok，生产慢 |
| 1 个 Python Constraint，N=10000 | ~10⁷ | ~1µs | ~10s | ❌ 改 Region |
| 1 个 Python Constraint，N=100000 | ~10⁸ | ~1µs | ~100s | ❌ 禁止 |
| 10 个 Python Restraint，每个作用 ≤ 100 原子 | ~10⁴ | ~10µs | ~0.1s | ✅ |
| Python Handler，每 100 iter 一次 | ~10² | ~50µs | ~5ms | ✅ 完全可忽略 |

**经验法则（写进 user docs）**：
- Python 适合做 **Restraint** 和 **Handler**。
- Python 写 **Constraint** 仅用于原型/调试；N ≥ 10000 必须用 Rust 或 Region 组合子表达。
- Region 组合子是纯类型代数，Region 组合出来的 Constraint **走静态 dispatch**，零运行时成本。

---

## 8. 流程图（一次 pack() 的完整生命周期）

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. USER ASSEMBLY                                                     │
│   Molpack::new()                                                     │
│     .add_constraint(c1)  ──► Vec<Box<dyn Constraint>>  [user API]    │
│     .add_restraint(r1)   ──► Vec<Box<dyn Restraint>>                 │
│     .add_handler(h1)     ──► Vec<Box<dyn Handler>>                   │
│     .pack(targets)                                                   │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. COMPILE (Molpack → Packer)                                        │
│   - Resolve Selector → Vec<u32> atom indices (freeze per-Restraint)  │
│   - Build ConstraintSet:                                             │
│       · Partition by {Builtin | CustomRust | Python}                 │
│       · Move Builtin trait-impls back into BuiltinConstraint enum    │
│       · Build 3 CSRs (one per section)                               │
│   - Reuse existing PackContext allocation & cell-list setup          │
│   - Validate: ≥1 constraint? bounding box well-defined?              │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. PHASE 0: Per-type sequential placement (uses comptype[])          │
│   for each free molecule type T:                                     │
│     set comptype[*] = (== T)                                         │
│     handlers.on_phase_start(PerType{T})                              │
│     optimizer.solve(objective_only_T, x_T)                           │
│     handlers.on_phase_end(PerType{T}, report)                        │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. PHASE 1: Geometric pre-fit (init1 = true)                         │
│   sys.init1 = true  ⇒ existing code path skips accumulate_pair_*     │
│   optimizer.solve(objective_constraints_only, x)                     │
│   ──► 保证每个原子先进入自己的 region，再做 pair check               │
│   sys.init1 = false                                                  │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 5. PHASE 2: Main loop (× nloop)                                      │
│   for outer in 0..nloop:                                             │
│     discale = inflate_tolerance(outer, fimp, flast)                  │
│     handlers.on_phase_start(MainLoop)                                │
│                                                                       │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │  relaxers.on_iter(...)         [per target, existing path]  │  │
│     │  movebad(...)                  [existing]                   │  │
│     │  GENCAN inner loop (× maxit)                                │  │
│     │    Objective::fg(x, &mut g):                                │  │
│     │      ┌─ PairKernel (cell-list)  [existing rayon @ cell]     │  │
│     │      ├─ ConstraintSet.fg_atom per-atom:                     │  │
│     │      │    ├─ Builtin [inline match-on-kind]                 │  │
│     │      │    ├─ CustomRust [vtable, cold]                      │  │
│     │      │    └─ Python [single GIL acquire per outer batch]    │  │
│     │      ├─ Restraint dispatcher    [serial; M ≪ N]             │  │
│     │      └─ project_cartesian_gradient (existing)               │  │
│     │    line search → update x                                   │  │
│     │    handlers.on_inner_iter(iter, f)                          │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                                                                       │
│     handlers.on_phase_end(MainLoop, report)                          │
│     if handler.should_stop() or all_atoms_satisfied: break           │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 6. RESULT                                                            │
│   PackResult {                                                       │
│     frame: Frame,                                                    │
│     unsatisfied: Vec<(atom_id, constraint_id, violation)>,           │
│     phase_reports: [PhaseReport; 3],                                 │
│     total_iterations: u32,                                           │
│   }                                                                  │
│   handlers.on_finish(&ctx)                                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. 迁移计划（3 个不破坏的增量阶段）

### Phase A：内部重构，公开 API 不变（1-2 周）

**Phase A 基准原则（取代原 "step 0 预冻结 baseline" 方案）**：不再预先冻结 monolithic baseline。采用 **extract-bench loop** —— 抽象一个函数，就给它（和调用方）落一个微基准，作为同一 commit 的组成部分。理由：monolithic end-to-end 基准的 noise floor（±3% 量级）与 §10 的原硬红线同量级，gate 对增量抽取无信号。详细规则见 `molrs-perf` skill § "Benchmarking during refactors"。

具体做法（每次从 `objective.rs` / `packer.rs` 抽出一个纯函数 F 时）：
- 抽出前：在原文件留一份 `#[cfg(bench)] #[inline(never)] fn F_sentinel(...)`，保存抽出前的函数体作为对照基线。
- 同一 commit 内落：F 的 unit test + F 的 criterion 微基准 + F 的调用方 criterion 微基准。
- 微基准 gate：F vs. `F_sentinel` ≤ +1%；调用方 vs. 抽出前的快照 ≤ +2%。
- 下一次重构周期后删除 `F_sentinel`。

**唯一的 monolithic 基准 `molrs-pack/benches/pack_end_to_end.rs`** 在 Phase A 开始时一次性建立，仅作为**灾难级回退警报**（> 10% block），不作为每次抽取的 gate。`molrs-pack/benches/` 下所有微基准永久保留，作为未来重构的回归守卫（不是一次性脚手架）。

**Phase A 本体**（按以下顺序，每步都是 test + 微基准 + 抽取 的单 commit）：
1. 落基础设施：`molrs-pack/Cargo.toml` 加 `criterion = "0.5"` dev-dep 与 `[[bench]]` 条目；建立 `pack_end_to_end.rs` 作为灾难警报。
2. 将 `struct Restraint{kind,params}` rename → `struct BuiltinConstraint`（仅内部）
3. `hook.rs` 中 `Hook`/`HookRunner`/`TorsionMcHook`/`TorsionMcRunner` → `Relaxer`/`RelaxerRunner`/`TorsionMcRelaxer`/`TorsionMcRelaxerRunner`；文件重命名为 `relaxer.rs`
4. 拆 `packer.rs` → `phase/{per_type, geometric_prefit, main_loop}.rs`；**每抽一个 phase 函数都落一次微基准**
5. 把 `PackContext::evaluate` 的签名固化为 `trait Objective`；`impl Objective for PackContext`
6. `pgencan` 入参 `&mut PackContext` → `&mut dyn Objective`
7. **验收**：所有现有测试通过；Packmol 等价回归测试 0 失败；每次抽取的微基准 gate 满足 §10；`pack_end_to_end.rs` 未触发灾难警报

### Phase B：公开新 trait，并行新旧 API（1 周）

1. 在 `molrs-pack` 公开 `Constraint` / `Region` / `Restraint` / `Selector` trait
2. 内置约束保持 `BuiltinConstraint` enum；向用户暴露 `InsideBox::new(...)` 等构造器返回 **opaque builder** 类型，内部填 `BuiltinConstraint`（不暴露 enum 本身）
3. `Molpack::add_constraint(impl Into<ConstraintInput>)` — 接受三路（enum / `Box<dyn Constraint>` / Python）
4. 老的 `add_restraint(Restraint)` 保留但 `#[deprecated(note = "use add_constraint with the new Constraint trait")]`
5. `Handler` trait 加入 §6.6 的新默认方法
6. **验收**：用户能用纯 Rust 写自定义 Constraint 并跑通；基线无退化

### Phase C：Python 桥（1-2 周）

1. 在 `molrs-python` 新建 `pack` 子模块
2. 暴露 `Constraint` / `Restraint` / `Handler` Python 抽象基类
3. 实现 PyO3 桥（`PyConstraintImpl: Constraint`），`is_parallel_safe = false`
4. 实现编译时分区（`ConstraintSet` 的三段 CSR）与单 GIL 批量调用
5. 写 Python 端 e2e 测试：自定义 Restraint 跑通完整 packing
6. **验收**：Jupyter 50 行定义并运行自定义约束；基线无退化

---

## 10. 性能门禁（量化）

**强制**：每次抽取（即每个 commit）必须跑 `cargo bench -p molrs-pack --bench <相关微基准>` 对比当前 commit 的 `#[inline(never)] F_sentinel`。Phase 结束时额外跑 `pack_end_to_end.rs` 作灾难警报。

| 粒度 | 指标 | 硬红线 | 软红线 |
|---|---|---|---|
| 每次抽取 | 抽出函数 vs. `#[cfg(bench)] #[inline(never)] F_sentinel` | ≤ **+1%** | 0% 或改善 |
| 每次抽取 | 调用方微基准（捕捉 indirection / vtable 开销） | ≤ **+2%** | ≤ +1% |
| Phase 结束 | `pack_end_to_end.rs` 灾难警报 | ≤ **+10%** | ≤ +5% |

- 超过硬红线 → 该抽取不准合并；必须拆更细、或保留 `#[inline(always)]`、或回退方案。
- 软硬红线之间 → 附 profiler 报告解释来源，经 HPC review 批准。
- **不再做 geomean across monolithic benches** —— 对增量抽取无信号（noise floor 与原 ±1.5% geomean 门禁同量级）。

**基准组织**：
- 微基准按抽出函数组织（`molrs-pack/benches/<module>_<fn>.rs` 或统一的 `bench_hot_path.rs`），是抽取的一部分，永久保留。
- `pack_end_to_end.rs` 只跑一个代表性工作量（1 种 ~10k 原子配置），作为 Phase 收尾灾难警报。**不再** 跑 {1k, 10k, 100k} × {1, 4, 16} 矩阵作为每次抽取的 gate —— 此矩阵只在 end-to-end 警报内部可选使用。
- 机器基线：开发者自己机器（记录 CPU / RAM / OS）+ CI runner。微基准对机器敏感度低（相对 sentinel，不是绝对值），CI 可直接 gate。

---

## 11. 测试与验收标准

### 11.1 不退化
- 现有 `molrs-pack/tests/` 全部通过（constraint, euler, gradient, hook, packer, target, examples_batch）
- `learn-packmol` skill 的 Packmol 等价回归测试 0 失败

### 11.2 新增组件单元测试
- `Region` 组合子：`(A and B).contains(x) == A.contains(x) && B.contains(x)`
- `Constraint::fg` vs `f + numerical grad`：所有内置约束有限差分验证（ε=1e-5，tol=1e-3）
- 双 scale 分支：每个内置 kind 验证用 `scale` 还是 `scale2` 与现有 `Restraint::value` 一致
- `Restraint` 对 `Selector` 的过滤：`DistanceRestraint(a, b)` 只对 a, b 两原子加梯度
- `Optimizer` 与 `Objective` 解耦：用 Rosenbrock / Booth / Beale 标准函数验证 GencanOptimizer 收敛

### 11.3 Python 集成测试
- 自定义 `MyConstraint(Constraint)` 跑通 1000 原子 packing（元组 API）
- 自定义 `MyRestraint(Restraint)` 跑通 10000 原子 packing（numpy API）
- Python `raise` 正确传回 Rust 并 `Err(PackError::PythonCallback)`，无 panic
- 性能基线：1 个 Python Constraint vs 等价 Rust，在 1000 原子下 slowdown ≤ 100×（目标 50×，硬红线 100×）

### 11.4 movebad 等价性
- 打开 `move_flag`，比对 `fdist_atom` / `frest_atom` 数组前后逐原子一致（tol 1e-12）
- 验证 `AtomViolation::frest_delta` 的累加等价于现有 `accumulate_constraint_value` 的副作用

### 11.5 Geometry cache 等价性
- 构造 `f(x); g(x)` 连续调用场景，验证 `matches_cached_geometry` 命中
- 构造 `f(x1); g(x2)` 场景，验证缓存失效并重算

---

## 12. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| `dyn Constraint` 热路径退化 | **高** | 硬性要求内置 15 类走 `BuiltinConstraint` enum + match；dyn 仅用户扩展。`ConstraintSet` 三段 CSR 在 compile 时分区 |
| Python GIL 在 rayon 死锁 | 高 | Python 约束强制单线程；由 `ConstraintSet::python` section 的串行调用保证；**禁止** rayon 任务内部调 Python |
| Geometry cache 失效导致性能回退 | 中 | trait 文档明示契约；`cache_hit_rate` 作为 benchmark 报告项 |
| movebad 副作用丢失 | 中 | `AtomViolation` 作为 eval 输出；加 11.4 等价性测试 |
| 双 scale 语义偏差 | 中 | `Constraint::f/fg` 同时吃 `(scale, scale2)`；内置约束单元测试枚举每个 kind 验证 |
| 新旧 API 双轨期混乱 | 低 | Phase B ≤ 2 周；旧 API `#[deprecated]` 强提示 |
| Python user 写错梯度（与 f 不一致） | 中 | 提供 `assert_gradient_consistent(c)` Python 工具；文档强示例 |
| `Relaxer` 重命名破坏下游代码 | 低 | 保留 `pub use Relaxer as Hook` 与 `pub use TorsionMcRelaxer as TorsionMcHook` 一个 Phase 的别名并 `#[deprecated]` |
| `#[inline(never)] F_sentinel` 泄漏到 release 二进制 | 低 | 仅在 `#[cfg(bench)]` 下编译；下一次重构周期后删除；`molrs-optimizer` 代码审查强制检查 |

---

**结束**。本 spec 完成后，molrs-pack 从"Packmol 的 Rust port"升级为"通用分子堆积框架"，且**性能不退化**（§10 量化保证），Python 可扩展三个弱耦合扩展点（Constraint / Restraint / Handler）。
