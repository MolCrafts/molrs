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
- [x] **A.4** 拆 `packer.rs` → 抽 `evaluate_unscaled` + `run_iteration` + `run_phase` 共三个纯函数；**每抽一个 phase 函数都落 sentinel + 函数微基准 + 调用方微基准**（首次真正应用 extract-bench loop）；A.4.4（进一步拆 setup / geometric prefit）不再需要 — 外层已经是 13 行 scaffold，setup block 是直线代码（非热路径），不拆
    - [x] **A.4.1** 抽 `evaluate_unscaled(sys, xwork) -> (f, fdist, frest)`（消除三处 `radiuswork` swap-evaluate-restore 重复）；落 `evaluate_unscaled_sentinel` + `benches/evaluate_unscaled.rs`（fn + caller 两组微基准）；gate：fn -3.4% vs sentinel（远过 ≤+1%），caller +1.1%（过 ≤+2%）
    - [x] **A.4.3** 抽 inner loop body → `fn run_iteration(loop_idx, ..., &mut sys, ..., &mut rng) -> IterOutcome`（Continue / Converged / EarlyStop）；落 `run_iteration_sentinel` + `benches/run_iteration.rs`（fn + caller 两组）+ 单元测试（空 context 上 fn == sentinel）；`SwapState` 提升 `pub`（bench 需要）；**先于 A.4.2 落**是因为外层 phase body 在 inner loop 抽出后变得 trivial，顺序反过来可以把复杂的一次性做完
    - [x] **A.4.2** 抽 main loop 外层 for 循环 body → `fn run_phase(phase, ntype, ntype_with_fixed, total_phases, max_loops, discale, precision, disable_movebad, &mb_cfg, &gp, &mut sys, &mut x, &mut swap, &mut relaxers, &mut handlers, &mut ws, &mut rng) -> PhaseOutcome`（Continue/Converged — 仅 all-type converged 才打破外层）；落 `run_phase_sentinel` + `benches/run_phase.rs`（fn + caller 两组）+ 单元测试（空 context trivially Converged）；外层 123 行塌缩到 13 行 match-scaffold；gate：fn +0.44%（过 ≤+1%），caller -0.86%（过 ≤+2%），e2e mixture 481→472 ms p=0.24（无变化）；113 tests pass（+1 新测试）
    - [~] **A.4.4** 进一步拆 setup / geometric prefit — **不做**：A.4.2 之后 `pack()` 主循环已经是 13 行 scaffold，剩下的 setup 是 ~240 行直线 bookkeeping（target / atom / restraint 映射 + relaxer runner build），非热路径，不拆
- [x] **A.5** 把 `PackContext::evaluate` 的签名固化为 `pub trait Objective { fn evaluate(&mut self, x, mode, grad) -> EvalOutput; fn fdist(&self); fn frest(&self); fn ncf(&self); fn ncg(&self); fn reset_eval_counters(&mut self); }`；`impl Objective for PackContext` 纯转发；单测 `dyn_objective_matches_inherent_evaluate`（trait dispatch 与直接调用 byte-identical）。纯加性改动，不动 call site；A.6 里才把 `pgencan` 入参换成 `&mut dyn Objective`（那一步需要完整 sentinel + 微基准）
- [x] **A.6** `pgencan` / `gencan` / `tn_ls` / `spg::spgls` / `cg::cg_solve` / `cg::hessian_times_vec_diff` / `packmolprecision` 全部 `&mut PackContext` → `&mut dyn Objective`；`Objective` trait 新增 `fn bounds(&self, l, u)`（default 全 `±1e20`；`PackContext` 覆盖搬 `build_bounds` 的 Euler 约束逻辑）；`build_bounds` 删除；14 处 `compute_f`/`compute_g`/`sys.evaluate` 调用点换成 `obj.evaluate(x, mode, grad)`；落 `benches/objective_dispatch.rs`（via_inherent vs via_dyn 单 call + caller 4-call burst 两组）；gate：fn via_dyn **-0.95%** vs via_inherent（过 ≤+1%），caller_dyn **-0.70%** vs caller_inherent（过 ≤+2%）；e2e mixture 472→498 ms ≈ **+5.6%**（过 ≤+10% 硬门禁，略过 ≤+5% 软门禁 ~0.6pp；per-call 微基准清洁说明根源是 dyn dispatch 阻止了 `evaluate` 跨 boundary inline，~10k calls/pack 的复合 inlining loss，非 dyn itself 的成本；Phase D 的 GEMM / SIMD 可回收）；114 tests pass
- [x] **A.7** 验收：114 tests pass（unit + integration，`--all-features`）；Packmol 等价回归 `tests/examples_batch.rs --ignored` 全部 5 例通过（release 模式 27s，0 failures）；微基准 gate A.4.1/A.4.2/A.4.3 fn ≤+1%、caller ≤+2% 全过；A.6 fn via_dyn -0.95%、caller_dyn -0.70% 过；`pack_end_to_end/mixture` 472→498 ms ≈ +5.6%（过 ≤+10% 硬门禁；略过软门禁由 Phase D 回收）；workspace `cargo build --all-features` 0 errors；`cargo clippy -p molcrafts-molrs-pack --all-features --tests --benches` 0 molrs-pack warnings；`cargo fmt --all --check` 通过

**Phase B — 公开新 trait，并行新旧 API**（v2-r4 修订：direction 3 扩展点通用模式；B.0 从 rename 升级为结构重构；B.1 并入 B.0）
- [ ] **B.0** **结构重构（direction 3）**：
  - 删除 `BuiltinConstraint` / `MoleculeConstraint` / `AtomConstraint` struct 与 `RegionConstraint` trait（composability-only，不是 eval trait）
  - 定义 `pub trait Restraint: Send + Sync { f; fg; is_parallel_safe; name; }`（§6.2 签名）
  - 创建 15 个具体 `pub struct *Restraint`，每个持自己几何字段（`InsideBoxRestraint { min, max }`、`InsideSphereRestraint { center, radius }`、...），各自 `impl Restraint`
  - `Target::with_constraint` → `with_restraint(impl Restraint + Clone + 'static)`；字段 `constraints` → `restraints`
  - `Target::with_constraint_for_atoms(&[idx], ...)` → `with_restraint_for_atoms(&[idx], impl Restraint + Clone + 'static)`（替代原 `AtomConstraint` wrapper）
  - 文件重命名 `src/constraint.rs` → `src/restraint.rs`（或 `src/restraint/mod.rs` + 每 kind 一个 `src/restraint/<kind>.rs`）；`src/constraints/` → `src/restraints/`；`tests/constraint.rs` → `tests/restraint.rs`
  - `lib.rs` 删除 `pub use constraint::BuiltinConstraint as Restraint` 与所有 `*Constraint` re-exports；改为 15 个 `*Restraint` + `Restraint` trait re-export
  - 内部存储首轮统一 `Vec<Box<dyn Restraint>>`（fast-path 作 follow-up，仅当 e2e 超过 +5% 软门禁时再做）
  - 更新 5 个 examples + README + doctests
  - 保留 `constrain_rot` / `rot_bound` / gencan 投影 bounds（唯一真正的硬约束语义）
  - 单元测试：15 struct 各自 `Restraint::fg` vs 数值梯度（ε=1e-5，tol=1e-3）；与 rewrite 前 inherent method 在相同参数下数值等价
  - 验收：所有现有 tests + `examples_batch` Packmol 等价回归通过；`pack_end_to_end/mixture` 不超过 +5%（若超过且 ≤ +10% 硬门禁，落 `PackedRestraint` fast-path follow-up）
- [ ] **B.0b** Region trait（§6.1）：`pub trait Region { contains; signed_distance; }` + `And`/`Or`/`Not` 组合子 + blanket `impl<R: Region + 'static> Restraint for FromRegion<R>`；用于 `InsideSphereRegion.and(OutsideSphereRegion)` 形式的几何交并运算
- [ ] **B.2** 便捷静态方法（**不是** builder pattern）：如 `InsideBoxRestraint::from_simbox(&simbox)` / `InsideCubeRestraint::from_origin(origin, side)` 等纯构造 helper
- [ ] **B.3** `Molpack::add_restraint(impl Restraint + Clone + 'static)` — 内部实现 = 遍历 `targets` 各调一次 `target.with_restraint(r.clone())`，不开新存储路径
- [~] **B.4** ~~deprecate 老 `add_restraint`~~ — **取消**：`Molpack` 上从未有过 `add_restraint` 公开方法，无要 deprecate 的老 API
- [ ] **B.5** `Handler` trait 加入 §6.6 的新默认方法（`on_inner_iter` / `on_phase_end` 带 default impl）
- [ ] **B.6** 验收：用户能用纯 Rust 写自定义 `Restraint`（global 或 per-target 两种挂法都跑通，与内置 15 struct 在类型维度完全平权）；基线无退化

**未来（不在当前 spec 范围）**：`pub trait Constraint`（硬约束 / Lagrange / SHAKE / RATTLE / LINCS）。molrs-pack 目前没有任何硬约束实现，定义 trait 也无对应实例；留作后续 phase 的 placeholder，不与 Phase B 混淆。

**Phase C — Python 桥**
- [ ] **C.1** `molrs-python` 新建 `pack` 子模块
- [ ] **C.2** 暴露 `Restraint` / `Handler` / `Relaxer` Python ABC（**无 `Constraint` ABC** —— 硬约束未实现）
- [ ] **C.3** PyO3 桥（`PyRestraintImpl: Restraint`），`is_parallel_safe = false`
- [ ] **C.4** 编译时分区（`RestraintSet` 三段 CSR）+ 单 GIL 批量调用
- [ ] **C.5** Python e2e：自定义 Python `Restraint`（global + per-target 两种 scope 各跑一次）跑通完整 packing
- [ ] **C.6** 验收：Jupyter 50 行定义并运行自定义 Restraint；基线无退化

**Phase D — 性能优化（trait 稳定后再做；先 profile 再动手）**

> 这三条是 `objective.rs` 里 `expand_molecules` / `project_cartesian_gradient` 两处 Euler 旋转双循环的候选加速路径。**不先 profile 不动手** — Packmol 的 hot path 历来是 `fparc` / `gparc`（pair-list 距离计算），Euler 展开可能只占 < 5%，那样这三条都是 premature optimization。先在 A.7 之后跑一次 `perf record` / `samply` 看真实热点分布；≥ 20% 才值得做 D.1 / D.2，< 5% 直接关 Phase D。

- [ ] **D.0** Profile baseline：`cargo bench --bench pack_end_to_end -- mixture` + `samply record`，记录 `expand_molecules` / `project_cartesian_gradient` / `fparc` / `gparc` 各自的 CPU % (on `pack_mixture` 和 `pack_spherical` 两个 workload)
- [ ] **D.1** **批量化 GEMM**：当前 per-molecule `compcart` 调用是 3×3 × 3×N 的小矩阵乘（N≈3–20），BLAS 调用开销在这个尺度上 >> 计算本身（负优化）。正确做法：**同类型所有分子堆成一个 `(Nmol × Natoms, 3)` 大矩阵**，一次 `R_stack · coords_ref^T`（或展成 blocked GEMM）。需要重排 `xcart` / `gxcar` memory layout 与 `expand_molecules` 的遍历顺序。Gate：mixture workload ≤ -10%（目标是改善，不是持平）
- [ ] **D.2** **SIMD**（`std::simd` / `wide`）：对 D.1 之后仍然耗时的 per-atom 算术（`compcart` 内循环、`project_cartesian_gradient` 的 9+9+9 元素积、`fparc` 的距离计算），展 4×f64 lane。对 loop 结构改动最小。Gate：hot fn 微基准 ≤ -20%
- [ ] **D.3** **验证 Rust 自动向量化**：在 D.1/D.2 动手前，先 `cargo rustc --release -- --emit=asm -C target-cpu=native` 看 `compcart` / `eulerrmat` / `fparc` 是否已经被 LLVM 展开成 AVX2/AVX512（大概率是）。已经向量化的话就跳过 D.2
- [ ] **D.4** 验收：mixture / spherical workload 较 A.7 baseline 至少 20% 改善，且所有现有测试 + Packmol 等价回归仍通过

**Commit log** (newest last; one line per landed step)
- `pre-A` — spec/skill/agent 落地 extract-bench loop 纪律
- `A.1` — bench infra + 5-example 灾难警报（复用 `ExampleCase`，无 workload 重复）
- `A.2` — `Restraint` → `BuiltinConstraint` 重命名；`pub use ... as Restraint` 保 API；`pack_mixture` 478→482 ms（噪声内）
- `A.3` — `Hook`/`HookRunner`/`TorsionMcHook` → `Relaxer`/`RelaxerRunner`/`TorsionMcRelaxer`；`hook.rs` → `relaxer.rs`；trait 别名保 API；`pack_mixture` 472→480 ms（噪声内，p>0.05）
- `A.4.1` — 抽 `evaluate_unscaled` 去三处重复；落 sentinel + `benches/evaluate_unscaled.rs` + 单元测试（首个走完 extract-bench loop 全纪律的 commit）；fn gate -3.4%、caller gate +1.1%、e2e 467→484 ms（p=0.08, 噪声内）
- `A.4.3` — 抽 inner loop body → `run_iteration`（140 行移到纯函数）；落 `run_iteration_sentinel` + `benches/run_iteration.rs` + 空-context 单元测试；`SwapState` pub 化；fn gate -2.4%（过 ≤+1%），caller gate -0.3%（过 ≤+2%），e2e 484→481 ms p=0.57（无变化）；112 tests pass（+1 新测试）；先于 A.4.2 — phase body 在此之后变 thin wrapper
- `A.4.2` — 抽 outer phase body → `run_phase`（123 行 → 13 行 match-scaffold）；落 `run_phase_sentinel` + `benches/run_phase.rs` + 空-context 单元测试；新增 `PhaseOutcome { Continue, Converged }`；gate：fn +0.44%（过 ≤+1%），caller -0.86%（过 ≤+2%），e2e mixture 481→472 ms p=0.24（无变化）；113 tests pass（+1 新测试）；`pack()` 主循环现在是纯 scaffold
- `A.5` — 加 `pub trait Objective` + `impl Objective for PackContext`（纯 inline 转发到既有方法）；单测 `dyn_objective_matches_inherent_evaluate` 固定 dyn dispatch 与直接调用等价；114 tests pass（+1）；**非抽取**（无代码移动）故不带 sentinel / 微基准；A.6 rewire `pgencan` 时再落完整 extract-bench
- `A.6` — `pgencan`/`gencan`/`tn_ls`/`spg`/`cg` 全链 `&mut PackContext` → `&mut dyn Objective`；`Objective::bounds` trait 方法替代 `gencan::build_bounds`；14 call site 切到 `obj.evaluate`；落 `benches/objective_dispatch.rs`；gate：fn via_dyn -0.95%（过 ≤+1%），caller_dyn -0.70%（过 ≤+2%），e2e mixture 472→498 ms ≈ +5.6%（过硬 ≤+10%，略过软 ≤+5% ~0.6pp，根源是 ~10k evaluate/pack 的 inlining loss 而非 dyn cost 本身 — Phase D 回收）；114 tests pass
- `A.7` — Phase A 验收：114 unit/integration tests 通过；`examples_batch.rs` 5-case Packmol 等价回归 `--ignored` release 27.36s 全过（mixture/interface/bilayer/solvprotein/spherical 全部 `validate_from_targets` 通过）；所有 extract-bench 微基准 gate 满足；e2e +5.6% 过硬门禁；workspace build + clippy + fmt 全清。**Phase A 完成** —— 可进入 Phase B（公开 Constraint/Region/Selector trait）；A.4.4 标 ~（skip，非热路径不拆）

---

## 0. 修订要点（相对 v1）

v1 忽略了当前代码已经做到的部分，并引入了与现有类型重名的 trait。v2 的核心修订：

1. 承认并复用 `PackContext::evaluate(x, EvalMode, Option<&mut g>)`、CSR（`iratom_offsets`/`iratom_data`）、`init1`、`comptype`、`accumulate_pair_f_parallel` 等既有机制。
2. 命名冲突修正：现有 `Hook`（几何修改器）保留并改名为 `Relaxer`（可以是 MC / MD / 梯度下降，不限 MC）；新的观察者能力**不引入新 trait**，改为扩展现有 `Handler`。
3. 双 `scale` / `scale2` 行为写进 trait 签名；per-atom 副作用（`fdist_atom`/`frest_atom`，movebad 所需）写进输出契约。
4. Geometry cache（`matches_cached_geometry` 快速路径）作为 `Objective` trait 的显式契约。
5. 内置 restraint 强制静态 dispatch；`dyn Restraint` 仅保留给用户扩展与 Python 桥。
6. 性能门禁量化，但**放弃**预冻结 monolithic baseline 的方案。改为 **extract-bench loop**：每次抽出一个纯函数 F，同一 commit 内落 `#[cfg(bench)] #[inline(never)] F_sentinel` + F 的微基准 + 调用方微基准，F vs. sentinel ≤ +1% 作为硬门禁。详见 `molrs-perf` skill § "Benchmarking during refactors" 与本 spec §10。
7. Phase A 不再有 "预冻结 baseline" 的 step 0；bench 与 refactor 在同一 commit 内并行进行。
8. **v2-r3 修订（2026-04-14，撤回 v2-r2 合并）**：`Constraint` 与 `Restraint` **语义不同，不合一**。依据 molecular-simulation 惯例与 Packmol 实现现实：
   - **Constraint** = 硬约束（g(x)=0 必须满足；Lagrange 乘子 / SHAKE / RATTLE / LINCS 型机制）。molrs-pack **目前没有** 硬约束实现，本 spec 不在 Phase A-C 引入 `Constraint` trait；留作未来 phase placeholder。
   - **Restraint** = 软惩罚（penalty function with 梯度；可违反，付能量代价）。当前 `BuiltinConstraint` 及 15 类"约束"（InsideBox / InsideSphere / OutsideSphere / AbovePlane / BelowPlane / Ellipsoid / Cylinder / Gaussian 等）实际上都是 `scale * max(0, d)` 线性或 `scale2 * max(0, d)²` 平方 penalty —— **是 Restraint 不是 Constraint**。
   - **统一命名（B.0 sweep）**：把所有 `*Constraint` code identifier 与 spec 文字 rename 为 `*Restraint`。不保留别名（API 未发版，无反向兼容负担）。
   - **Scope 分层仍保留**：global（`Molpack::add_restraint`）与 per-target（`Target::with_restraint`），前者实现为对每个 target 广播 `with_restraint`，无独立存储路径。
   - **历史记录**：v2-r2 (commit `d131a49`) 曾依据 "Constraint/Restraint 本质无区别" 合并两者，由 v2-r3 撤回 —— 合并是错的，因为 Packmol 所有几何约束都是 penalty 形式，名字应反映实现（Restraint），而非反映用户意图（Constraint）。

9. **v2-r4 修订（2026-04-14，扩展点通用模式 "direction 3"）**：molrs-pack 所有扩展点（`Restraint` / `Relaxer` / `Handler` / 未来新增的其他 trait）统一采用同一模式：
   - **公共 API**：`pub trait X` + N 个具体 `pub struct`，每个 struct `impl X`。每个 struct **持有自己语义命名的字段**（如 `InsideBoxRestraint { min, max }`、`InsideSphereRestraint { center, radius }`），不共享 `{kind, params[9]}` 型的 generic blob。
   - **用户扩展**：用户的自定义 `pub struct MyFoo` `impl X for MyFoo`，与内置任何一个 struct 在类型维度上**完全平权**。
   - **禁止项**（硬性）：
     - ❌ 公共 API 中出现 `Builtin*` / `Native*` / `Packmol*` 前缀的 wrapper 类型
     - ❌ 公共 tagged-union / enum 把 N 种内置打包到一起
     - ❌ Builder pattern（用户不必 `X::new().add(...).add(...)` 构造）
     - ❌ 组合子（如 `.and()`）塞进主 trait —— 组合性放到单独的 `Region` trait
     - ❌ 为 per-atom-subset / scope 分发另开一个 wrapper 类型（应作为容器方法参数：`Target::with_X_for_atoms(&[idx], x)`）
   - **性能（内部 AoS fast-path）**：若 hot path 需要静态 dispatch，内部可维护 `pub(crate) enum PackedX` / tagged-union，内置类型通过 crate-private `fn try_pack(&self) -> Option<PackedX>` 或 `Any` downcast 在注册时转进；此 fast-path 对用户不可见。**首轮实现允许统一走 `Vec<Box<dyn X>>`**，若 `pack_end_to_end` 灾难警报 > +5% 软门禁则落 fast-path follow-up。
   - **当前状态**：`Handler` ✅ 合规；`Relaxer` ✅ 合规（A.3 完成）；`Restraint` ❌ 不合规（B.0 结构重构解决）。

---

## 1. 目标

1. 把 molrs-pack 从 Packmol 单体 port 重构为**可扩展框架**，可扩展点通过 trait 暴露。
2. `Restraint`（soft penalty；对应 Packmol 所有几何 "constraint" 类型的本质实现）既可 **global**（`Molpack::add_restraint`，语义 = 给每个 target 都挂一份）也可 **per-target**（`Target::with_restraint`）。真正的硬 `Constraint` trait 留给未来 phase。
3. 支持 **Python 注入** Restraint / Handler / Relaxer，无需重新编译 Rust。
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
| 装配 builder | `Molpack::new().add_handler(...).pack(targets)`（`api/mod.rs`, `packer.rs`） | 保留 `Molpack` 名；新增 `add_restraint` |

**禁止**：v1 spec 里写的 "引入 Objective trait" / "引入 Hook 观察者 trait" / "引入 CSR" / "引入 rayon" 表述必须改成 "扩展 / 抽出 / 暴露已有的 X"。

---

## 4. 术语：三类可扩展点（v2-r3 修订）

| 维度 | **Restraint**（当前 spec 唯一 penalty trait） | **Relaxer** | **Handler** |
|---|---|---|---|
| 数学定义 | penalty `f(x)` + 梯度；进目标函数 | 修改分子参考坐标（构型松弛：MC / MD / 梯度下降 均可） | 只读回调 |
| 作用域 (scope) | **global**（Molpack 级）或 **per-target**（Target 级，可进一步限到 atom 子集） | per-target | global |
| 影响 `nloop` 终止 | 进 `frest` 则是（该实例配置决定）；否则仅影响 `f` 而不决定精度判据 | 否 | 可 (`should_stop()`) |
| 调用频率 | per atom, 每 eval | 每 outer loop（movebad 之后） | 每 phase / 每 iter |
| 性能要求 | 极高（10⁵-10⁷ 次 / run） | 低（~10² / run） | 极低 |
| 对应 Packmol 概念 | `restraint kind 1..15` + `tolerance`（Packmol 代码中 "constraint" 实为 soft penalty） | （无原生） | 打印/进度 |
| 是否允许 Python 实现 | 允许但慢，文档告警（`is_parallel_safe=false` 强制单线程） | 允许 | 推荐 |
| 现有代码中的对应 | pre-B.0: `BuiltinConstraint{kind,params[9]}` + 15 newtype shells + `iratom_*` CSR。post-B.0 (direction 3): 15 个 `pub struct *Restraint` 各持自己字段 + `impl Restraint` | `Relaxer` / `RelaxerRunner`（A.3 rename，原名 `Hook`） | `handler.rs` 的 `Handler` |
| v2-r4 名字 | `Restraint` trait + 15 concrete pub structs（无 `Builtin*` wrapper，无 builder） | `Relaxer`（支持 MC/MD/梯度下降等多策略） | `Handler`（扩展） |

**Restraint 为当前 spec 唯一 penalty trait**（v2-r3 修订；撤回 v2-r2 的 Constraint/Restraint 合并）：

- Packmol 所有 "constraint" 类型（InsideBox / Sphere / Plane / Ellipsoid / Cylinder / Gaussian）实际上都是 `scale / scale2` 加权的 penalty，不是硬约束机制，命名应反映实现本质 → 统一为 `Restraint`。
- B.0 sweep 把 code 中所有 `*Constraint` rename 为 `*Restraint`；`pub use ... as Restraint` 别名删除。
- 真正的 `Constraint` trait（硬约束：Lagrange / SHAKE / RATTLE / LINCS）**本 spec 不引入**；molrs-pack 目前零硬约束实例，留作未来扩展 placeholder。
- "区分硬/软" 在实例配置层面仍可做（是否进 `frest` / 权重 / `scale` vs `scale2`），但类型层面仅一个 `Restraint` trait。

**Scope 等价律（仍保留）**：

```rust
// 这两者语义完全等价
molpack.add_restraint(r);
for t in &mut all_targets { t.with_restraint(r.clone()); }
```

`Molpack::add_restraint(r)` 内部就是遍历 `targets` 并 `target.with_restraint(r.clone())`；没有单独的 "global restraint" 路径，也不会在 `PackContext` 里另开一个全局 slot。Atom 级同理：`Target::with_restraint_for_atoms(idx, r)` 是已有能力，scope 最细到 atom 子集。

---

## 5. 核心分解

| # | 组件 | 类型 | 职责 | 所在位置 |
|---|---|---|---|---|
| 1 | `Region` | trait | 几何谓词 `contains / signed_distance`；组合子（`.and()` / `.or()` / `.not()`） | `molrs-pack/src/region/` |
| 2 | `Restraint` | trait | 唯一的 penalty+gradient 抽象；global 或 per-target 均可挂；内置 + 用户 Rust + Python 同型 | `molrs-pack/src/restraint/mod.rs` |
| 3 | 15 个 `*Restraint` 具体 struct | struct + `impl Restraint` | 每个 Packmol kind 一个独立 struct（`InsideBoxRestraint`、`InsideSphereRestraint` 等），各自持几何字段 | `molrs-pack/src/restraint/*.rs` |
| 4 | `RestraintSet` | `pub(crate)` struct | 编译时分区（packed fast-path / user-rust / user-python）+ CSR；对用户不可见 | 私有 |
| 5 | `PairKernel` | 内部函数（不是 trait） | Packmol 原生 pair distance，走 cell list | `objective.rs` 现有 `fparc/gparc/fgparc`，保留 |
| 6 | ~~`Selector`~~ | — | **不引入**（§6.4）—— selector 数据放在 Restraint 实例内部 | — |
| 7 | `Relaxer` + `RelaxerRunner` | trait（重命名现有） | 分子构型松弛（torsion MC / 内部 MD / 梯度下降等） | `relaxer.rs` 现有 |
| 8 | `Handler` | trait（扩展现有） | 观察者、早停 | `handler.rs` 现有，新增若干默认方法 |
| 9 | `Phase` | enum（非 trait） | `PerType / GeometricPrefit / MainLoop` 标识 | `molrs-pack/src/phase/` |
| 10 | `Objective` | trait | `PackContext::evaluate` 的抽象；GENCAN 依赖它而非 PackContext（A.5/A.6 已落地） | `molrs-pack/src/objective.rs` |
| 11 | `Optimizer` | trait | `solve(&mut dyn Objective, x, ws)`；GENCAN 是唯一内置实现 | `molrs-pack/src/optimizer/` |
| 12 | `Molpack` | struct（保留现名） | 装配 → 编译 → 运行 Phase[] → 输出 | `packer.rs` |

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

### 6.2 Restraint trait（保留双 scale 与 per-atom 副作用）

```rust
// molrs-pack/src/restraint/mod.rs

/// Per-atom violation tracking for movebad heuristic.
/// Currently lives as `frest_atom[icart]` side-effect in objective.rs;
/// v2 makes it an explicit output of each Restraint call.
#[derive(Default, Clone, Copy)]
pub struct AtomViolation {
    pub frest_delta: F,   // add to frest_atom[icart]; max-in to frest
}

pub trait Restraint: Send + Sync {
    /// Function value only (used by line-search interpolation path).
    fn f(&self, x: &[F; 3], scale: F, scale2: F) -> F;

    /// Fused function + gradient. Hot path.
    /// Accumulate gradient INTO `g` with `+=` (do not overwrite).
    fn fg(&self, x: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F;

    /// Parallel safety. `false` ⇒ scheduler serializes this restraint.
    /// Python-backed restraints MUST return false.
    fn is_parallel_safe(&self) -> bool { true }

    fn name(&self) -> &'static str { std::any::type_name::<Self>() }
}

/// Any Region lifts to a Restraint with quadratic exterior penalty.
/// `penalty(x) = scale2 * max(0, signed_distance(x))²`
impl<R: Region + 'static> Restraint for FromRegion<R> { /* ... */ }
```

**双 scale 契约**：Packmol 约定线性惩罚（box/plane/cube: type 2,3,6,7,10,11）用 `scale`，平方惩罚（sphere/ellipsoid/cylinder/gaussian: type 4,5,8,9,12–15）用 `scale2`。`Restraint` trait 同时接收两者，每个实现自己决定用哪个；保证与现有 match 分支等价。

### 6.3 15 个具体 `*Restraint` struct + 内部 `RestraintSet`（direction 3）

**公共 API**：不存在 `BuiltinRestraint` / tagged-union / builder。15 个 Packmol kind 各自是独立的 `pub struct`，持自己语义命名的几何字段：

```rust
// molrs-pack/src/restraint/inside_box.rs
pub struct InsideBoxRestraint {
    pub min: [F; 3],
    pub max: [F; 3],
}
impl Restraint for InsideBoxRestraint {
    #[inline]
    fn f(&self, x: &[F; 3], scale: F, _scale2: F) -> F {
        // 6 faces: linear penalty scale * sum_k max(0, d_k)
        let mut sum = 0.0;
        for k in 0..3 {
            if x[k] < self.min[k] { sum += self.min[k] - x[k]; }
            if x[k] > self.max[k] { sum += x[k] - self.max[k]; }
        }
        scale * sum
    }
    #[inline]
    fn fg(&self, x: &[F; 3], scale: F, _scale2: F, g: &mut [F; 3]) -> F {
        /* same + accumulate grad into g with += */
    }
}

// molrs-pack/src/restraint/inside_sphere.rs
pub struct InsideSphereRestraint {
    pub center: [F; 3],
    pub radius: F,
}
impl Restraint for InsideSphereRestraint { /* scale2 * max(0, d)² */ }

// ... 13 more (OutsideBox / OutsideSphere / AbovePlane / BelowPlane /
//  InsideCube / OutsideCube / InsideEllipsoid / OutsideEllipsoid /
//  InsideCylinder / OutsideCylinder / Gaussian / ...)
```

用户扩展在类型维度上与内置 struct **完全平权**：

```rust
pub struct MyHelixRestraint { pub radius: F, pub pitch: F }
impl Restraint for MyHelixRestraint { /* ... */ }

let target = Target::from_coords(...)
    .with_restraint(InsideBoxRestraint { min, max })
    .with_restraint(MyHelixRestraint { radius: 10.0, pitch: 2.5 });
```

**内部存储（crate-private，用户不可见）**：

首轮实现统一走 `Vec<Box<dyn Restraint>>`。hot path `evaluate` 对每个 restraint 做 virtual call；e2e 若在 `pack_end_to_end` 超过 +5% 软门禁，再落 fast-path follow-up commit：

```rust
// pub(crate) 内部接口，不在公共 API
// 仅当首轮 e2e 超过 +5% 软门禁时才添加
pub(crate) enum PackedRestraint {
    Kind02(/* inside_cube params */),
    Kind03(/* inside_box params */),
    // ... 15 kinds
}

// 每个内置 struct 提供 crate-private 转换
impl InsideBoxRestraint {
    pub(crate) fn try_pack(&self) -> PackedRestraint {
        PackedRestraint::Kind03(/* ... */)
    }
}

// 注册时把已知内置转进 PackedRestraint 走静态 dispatch；
// 用户自定义走 Box<dyn Restraint> 慢路径
pub(crate) struct RestraintSet {
    packed: Vec<PackedRestraint>,          // fast path (Packmol kinds)
    custom_rust: Vec<Box<dyn Restraint>>,  // user Rust extensions
    python: Vec<PyRestraintImpl>,          // Python (GIL-serialized)

    // CSR per section (packed / custom_rust / python)
    packed_csr_offsets: Vec<u32>,
    packed_csr_data:    Vec<u32>,
    // ... etc.
}
```

**禁止项（spec 级）**：
- ❌ 公共 API 中的 `BuiltinRestraint` / `PackedRestraint` / 任何 `Builtin*` wrapper 类型
- ❌ 公共 API 的 builder pattern（如 `Restraint::new().add(...)`）
- ❌ 热路径里做 `restraints.iter().partition(...)`
- ❌ 每原子一次 GIL acquire
- ✅ 编译时（`Target` 注册 restraint 时）一次 partition；每 eval 一次 GIL acquire 包住全部 Python restraints
- ✅ Fast-path 走内部 `PackedRestraint` match；用户自定义走 `Box<dyn>`；两条路径对用户透明

### 6.4 Restraint trait（当前 spec 唯一惩罚 trait；Constraint 为未来扩展）

**v2-r3 修订**：§6.2 的 trait 正式命名为 `Restraint`（不是 `Constraint`）。理由见 §0 bullet 8 —— Packmol 15 类几何约束全部通过 `scale / scale2` penalty 实现，不是硬约束机制，名字应反映实现本质。

- `Restraint` trait 已经覆盖 per-atom penalty + 梯度。需要多原子联合 penalty（例如 `DistanceRestraint(a, b, target, k)`）时，实现方把 selector 放到自己结构体内部（`self.atoms: Vec<u32>`），在 `f` / `fg` 里自己遍历；**不**把 `selector()` 提到 trait。
- `weight` 同理，作为实例字段（`k` 系数）由实现方自持，**不**作为 trait 方法。
- 多原子累加契约：`fg` 签名保持单原子 `g: &mut [F;3]`——多原子展开由外层 dispatcher（`pub(crate) RestraintSet`）调 `fg` 多次完成，避免 trait 签名特化到多原子形态。
- **不引入 `Selector` trait**（v1/v2-r1 计划的产物）：selector 是实现方内部数据，不需要 trait 抽象。

**硬约束（`Constraint` trait）未来再引**：SHAKE / RATTLE / LINCS / Lagrange 乘子在 molrs-pack 现无实例；Phase A-C 不引入 `Constraint` trait，留作后续 phase placeholder。

**Built-in 示例**（全部 `impl Restraint`）：

```rust
pub struct DistanceRestraint    { i: u32, j: u32, target: F, k: F }   // 挂到 atom i 时用 j 的位置
pub struct PositionRestraint    { i: u32, target: [F; 3], k: F }
pub struct OrientationRestraint { mol_id: u32, axis: [F;3], target: [F;3], k: F }
```

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

Restraint 有两条挂点，语义等价：

- **Global**: `Molpack::add_restraint(r)` — 内部等同于对每个 `Target` 各调一次 `with_restraint(r.clone())`
- **Per-target**: `Target::with_restraint(r)` / `Target::with_restraint_for_atoms(idx, r)` — 仅挂到该 target（可进一步限到 atom 子集）

本 spec 只有 `Restraint` 一条 API —— `Constraint`（硬约束）未来再引，目前零实例不开 API。

```rust
let result = Molpack::new()
    .tolerance(2.0)
    .seed(42)

    // Global scope: 挂到所有 target 的所有 atom
    .add_restraint(InsideBoxRestraint { min: [0.0; 3], max: [40.0; 3] })
    .add_restraint(OutsideSphereRestraint { center: [0.0; 3], radius: 5.0 })
    // 用户自定义 struct —— 与内置在类型维度完全平权
    .add_restraint(MyHelixRestraint { radius: 10.0, pitch: 2.5 })
    .add_restraint(py_restraint)
    // Region 组合自动转 Restraint（经 §6.1 的 Region trait blanket impl）
    .add_restraint(
        (InsideSphereRegion { center: c, radius: r1 })
            .and(OutsideSphereRegion { center: c, radius: r2 })
    )
    // 多原子 penalty 也走同一个 add_restraint；实例字段（k / weight / 是否进 frest）自决定
    .add_restraint(DistanceRestraint { i: atom_a, j: atom_b, target: 1.5, k: 10.0 })
    .add_restraint(py_orientation_bias)

    // 观察者（扩展既有 Handler）
    .add_handler(ProgressHandler::new())
    .add_handler(XYZHandler::to_path("traj.xyz"))

    // Per-target scope: 只挂到特定 target / atom 子集
    .target(mol_a).with_relaxer(TorsionMcRelaxer::new(&graph))
    .target(mol_b).with_restraint_for_atoms(&[0, 1, 2], LocalBiasRestraint { ... })

    .pack(&targets)?;
```

所有 struct 直接字段构造；没有 `.new()` 链 + `.add()` builder —— 内置与用户自定义写法一致。

---

## 7. Python 桥

### 7.1 设计原则

1. **避免假零拷贝承诺**：`PyArray1::from_slice(py, &x[..3])` 实际会拷贝 3 个 `f64`。per-atom `Restraint` 改用**元组传参**，比 numpy 快一个数量级。多原子 `Restraint`（per-selector, M 个原子）才值得用 numpy view。
2. **GIL 隔离**：`is_parallel_safe = false` 的组件走独立路径，编译时 partition（见 §6.3 `RestraintSet`）。
3. **单次 GIL 持有**：整个 `evaluate_restraints` 对所有 Python `Restraint` 只 acquire 一次 GIL，loop 在 GIL 内。
4. **错误透传**：Python `raise` → Rust `Err(PackError::PythonCallback(msg))`，不 panic。

### 7.2 Python 端接口

```python
from molrs.pack import Restraint, Handler, Region

class MyHelix(Restraint):
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
// molrs-python/src/pack/restraint.rs
pub struct PyRestraintImpl { obj: PyObject, name: String }

impl Restraint for PyRestraintImpl {
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

impl RestraintSet {
    /// Hot path. Called per atom in `accumulate_restraint_value/gradient`.
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
| 1 个 per-atom Python `Restraint`，N=1000 | ~10⁶ | ~1µs（元组调用） | ~1s | ⚠️ 原型 ok，生产慢 |
| 1 个 per-atom Python `Restraint`，N=10000 | ~10⁷ | ~1µs | ~10s | ❌ 改 Region |
| 1 个 per-atom Python `Restraint`，N=100000 | ~10⁸ | ~1µs | ~100s | ❌ 禁止 |
| 10 个 per-selector Python `Restraint`，每个作用 ≤ 100 原子 | ~10⁴ | ~10µs | ~0.1s | ✅ |
| Python Handler，每 100 iter 一次 | ~10² | ~50µs | ~5ms | ✅ 完全可忽略 |

**经验法则（写进 user docs）**：
- Python 适合做 per-selector 的 `Restraint` 和 **Handler**。
- Python 写 per-atom `Restraint` 仅用于原型/调试；N ≥ 10000 必须用 Rust 或 Region 组合子表达。
- Region 组合子是纯类型代数，Region 组合出来的 `Restraint` **走静态 dispatch**，零运行时成本。

---

## 8. 流程图（一次 pack() 的完整生命周期）

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. USER ASSEMBLY                                                     │
│   Molpack::new()                                                     │
│     .add_restraint(r1)   ──► Vec<Box<dyn Restraint>>  [user API]     │
│     .add_handler(h1)     ──► Vec<Box<dyn Handler>>                   │
│     .pack(targets)                                                   │
└──────────────────────────────────────┬───────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. COMPILE (Molpack → Packer)                                        │
│   - Resolve per-Restraint selector data (frozen Vec<u32>)            │
│   - Build pub(crate) RestraintSet:                                   │
│       · Partition by {Packed fast-path | CustomRust | Python}        │
│       · 15 known concrete *Restraint structs opt into fast-path via  │
│         crate-private try_pack() → PackedRestraint                   │
│       · Build 3 CSRs (one per section)                               │
│   - Reuse existing PackContext allocation & cell-list setup          │
│   - Validate: ≥1 restraint? bounding box well-defined?               │
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
│   optimizer.solve(objective_restraints_only, x)                      │
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
│     │      ├─ RestraintSet.fg_atom per-atom:                      │  │
│     │      │    ├─ Packed [inline match-on-kind; Packmol originals]│  │
│     │      │    ├─ CustomRust [vtable; user types]                │  │
│     │      │    └─ Python [single GIL acquire per outer batch]    │  │
│     │      ├─ per-selector Restraint  [serial; M ≪ N]             │  │
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
│     unsatisfied: Vec<(atom_id, restraint_id, violation)>,            │
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

0. **B.0 Rename sweep**：`BuiltinConstraint` → `BuiltinRestraint`；15 内置 `*Constraint` → `*Restraint`；`Target::with_constraint` → `with_restraint`、字段 `constraints` → `restraints`；文件 `src/constraint.rs` → `src/restraint.rs`、`src/constraints/` → `src/restraints/`、`tests/constraint.rs` → `tests/restraint.rs`；**删除** `pub use BuiltinConstraint as Restraint`；examples + README + doctests 同步。保留 `constrain_rot` / gencan bounds（真正的硬约束语义）
1. 在 `molrs-pack` 公开 `pub trait Restraint: Send + Sync`（§6.2 签名）；`impl Restraint for BuiltinRestraint`；15 kind × trait-path vs inherent-path 数值等价单测。不引入 `Constraint` trait（未来硬约束机制的 placeholder，本 spec 不做）；不引入 `Selector` trait（§6.4：selector 放实例内部）
2. 内置约束保持 `BuiltinRestraint` struct；向用户暴露 `InsideBoxRestraint::new(...)` 等构造器返回 **opaque builder** 类型，内部填 `BuiltinRestraint`
3. `Molpack::add_restraint(impl Into<RestraintInput>)` — 接受三路（builder / `Box<dyn Restraint>` / Python）；**内部实现 = 遍历 targets 各调一次 `with_restraint`**（不开新存储路径）
4. ~~deprecate 老 `add_restraint`~~ — **不做**：`Molpack` 上从未有过 `add_restraint` 公开方法
5. `Handler` trait 加入 §6.6 的新默认方法
6. **验收**：用户能用纯 Rust 写自定义 `Restraint`（既可 global 挂也可 per-target 挂）并跑通；基线无退化

### Phase C：Python 桥（1-2 周）

1. 在 `molrs-python` 新建 `pack` 子模块
2. 暴露 `Restraint` / `Handler` / `Relaxer` Python 抽象基类（**无 `Constraint` ABC** — 硬约束未实现）
3. 实现 PyO3 桥（`PyRestraintImpl: Restraint`），`is_parallel_safe = false`
4. 实现编译时分区（`RestraintSet` 的三段 CSR）与单 GIL 批量调用
5. 写 Python 端 e2e 测试：自定义 Python `Restraint`（global 或 per-target 挂）跑通完整 packing
6. **验收**：Jupyter 50 行定义并运行自定义 Restraint；基线无退化

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
- 现有 `molrs-pack/tests/` 全部通过（restraint, euler, gradient, relaxer, packer, target, examples_batch）—— B.0 之后 `constraint.rs` 已 rename 为 `restraint.rs`
- `learn-packmol` skill 的 Packmol 等价回归测试 0 失败

### 11.2 新增组件单元测试
- `Region` 组合子：`(A and B).contains(x) == A.contains(x) && B.contains(x)`
- `Restraint::fg` vs `f + numerical grad`：所有内置 restraint 有限差分验证（ε=1e-5，tol=1e-3）
- 双 scale 分支：每个内置 kind 验证用 `scale` 还是 `scale2` 与原 inherent-path 一致
- 多原子 `Restraint`（如 `DistanceRestraint(a, b)`）只对 a, b 两原子加梯度
- `Optimizer` 与 `Objective` 解耦：用 Rosenbrock / Booth / Beale 标准函数验证 GencanOptimizer 收敛

### 11.3 Python 集成测试
- 自定义 per-atom `MyRestraint(Restraint)` 跑通 1000 原子 packing（元组 API）
- 自定义 per-selector `MyDistanceRestraint(Restraint)` 跑通 10000 原子 packing（numpy API）
- Python `raise` 正确传回 Rust 并 `Err(PackError::PythonCallback)`，无 panic
- 性能基线：1 个 per-atom Python `Restraint` vs 等价 Rust，在 1000 原子下 slowdown ≤ 100×（目标 50×，硬红线 100×）

### 11.4 movebad 等价性
- 打开 `move_flag`，比对 `fdist_atom` / `frest_atom` 数组前后逐原子一致（tol 1e-12）
- 验证 `AtomViolation::frest_delta` 的累加等价于现有 per-atom penalty 副作用

### 11.5 Geometry cache 等价性
- 构造 `f(x); g(x)` 连续调用场景，验证 `matches_cached_geometry` 命中
- 构造 `f(x1); g(x2)` 场景，验证缓存失效并重算

---

## 12. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| `dyn Restraint` 热路径退化 | **高** | 硬性要求内置 15 类走 `BuiltinRestraint` + match；dyn 仅用户扩展。`RestraintSet` 三段 CSR 在 compile 时分区 |
| Python GIL 在 rayon 死锁 | 高 | Python restraint 强制单线程；由 `RestraintSet::python` section 的串行调用保证；**禁止** rayon 任务内部调 Python |
| Geometry cache 失效导致性能回退 | 中 | trait 文档明示契约；`cache_hit_rate` 作为 benchmark 报告项 |
| movebad 副作用丢失 | 中 | `AtomViolation` 作为 eval 输出；加 11.4 等价性测试 |
| 双 scale 语义偏差 | 中 | `Restraint::f/fg` 同时吃 `(scale, scale2)`；内置 restraint 单元测试枚举每个 kind 验证 |
| 新旧 API 双轨期混乱 | 低 | B.0 一次性 rename（无别名）；下游未发版，反向兼容不是目标 |
| Python user 写错梯度（与 f 不一致） | 中 | 提供 `assert_gradient_consistent(c)` Python 工具；文档强示例 |
| `Relaxer` 重命名破坏下游代码 | 低 | 保留 `pub use Relaxer as Hook` 与 `pub use TorsionMcRelaxer as TorsionMcHook` 一个 Phase 的别名并 `#[deprecated]` |
| `#[inline(never)] F_sentinel` 泄漏到 release 二进制 | 低 | 仅在 `#[cfg(bench)]` 下编译；下一次重构周期后删除；`molrs-optimizer` 代码审查强制检查 |

---

**结束**。本 spec 完成后，molrs-pack 从"Packmol 的 Rust port"升级为"通用分子堆积框架"，且**性能不退化**（§10 量化保证），Python 可扩展三个弱耦合扩展点（`Restraint` / `Handler` / `Relaxer`）；`Restraint` 同时支持 global (`Molpack::add_restraint`) 与 per-target (`Target::with_restraint`) 两种 scope，两者语义等价。硬约束 (`Constraint` trait) 留作未来扩展，当前 spec 不在 Phase A-C 引入。
