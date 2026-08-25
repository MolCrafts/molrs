---
slug: release-0-14-14-pair-kernel-merge
title: release-0-14-14-pair-kernel-merge — 单一 LJ pair kernel、每步 pair 数据直通、kspace 表面归位 pair
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-03-potential-protocol
---

# release-0-14-14-pair-kernel-merge — 一个 pair 概念，一份 pair 数据

## Summary

三件事，同一个概念：(1) `md::LJCut` 与 `ff::potential::pair::lj_cut::PairLJCut` 合并为**唯一** LJ pair kernel——这是**纯 API 统一**，不新增任何 NVE / 物理测试，验收就是同输入同输出逐位相同；(2) 用**每步 pair 数据直通**替掉 `set_pairs` 快照：indices / disp / dist_sq 由循环侧的邻居机制**每步算一次**，直接交给所有 pair 势共享（lj + coul + …），任何势内部不再做 MIC、不再存快照；(3) `kspace` 这个**名字**从 ForceField 表面消失（`StyleDefs::KSpace` 变体、`"kspace"` 类别串、`def_kspacestyle` 全删），PME 作为**静电 pair 势**按 LAMMPS pair 词汇挂进 pair 类别，`ff/potential/kspace` **模块保留**，纯粹作为让 FFT 依赖可被 feature 门控的编译单元。ONE `Potential` 概念自始至终不变。

## Domain basis

**本规范不引入新物理，也不新增守恒测试**（维护者裁决："你不需要测试 nve 守恒，就是 api 的变动而已"）。数值面唯一的事实是"合并前后，同输入同输出"，以逐位断言证明。

- LJ 12-6 / Mie 形式不变：`E = C·ε[(σ/r)ⁿ − (σ/r)ᵐ]`，`C = (n/(n−m))·(n/m)^{m/(n−m)}`，12-6 时 `C = 4` 精确成立。Mie, *Ann. Phys.* **316** (1903) 657, DOI 10.1002/andp.19033161002；截断与能量平移对齐 LAMMPS `pair_style lj/cut` + `pair_modify shift yes`（Thompson et al., *Comput. Phys. Commun.* **271** (2022) 108171, DOI 10.1016/j.cpc.2021.108171）。
- **md 没有单位这个概念**：kernel 是对给定数字做的无量纲数学，`epsilon` / `sigma` 的单位是调用者的事。本规范任何地方都不得把单位写成合并的障碍或前置条件（02 号已据此删去该说法）。
- 最小镜像只在**邻居机制**里做一次：`VerletSkin::for_each_pair_at`（`verlet_skin.rs:380`）的 `mic.apply` 与 `r2` 算术，与今天 `LJCut::calc_energy_forces`（`md/lj_cut.rs:337-342`）内联的那份**逐字相同**，且遍历顺序相同——这是 md 路径可以要求逐位相等的依据。Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §3.2。
- PME 作为静电势本身不改一行：Darden et al., *J. Chem. Phys.* **98** (1993) 10089, DOI 10.1063/1.464397；Essmann et al., *J. Chem. Phys.* **103** (1995) 8577, DOI 10.1063/1.470117。本规范只改它**挂在哪个类别、叫什么名字**。
- **一处逐位不可能成立，必须明说**（见 Design"算术形式的裁定"）：两份实现在 `1/r²` 上一个用乘倒数、一个用除法，差异上界 1 ulp。它是**报告项**，不是容差项。

## Design

**为什么是一个规范而不是三个（跨层范围说明）。** 三件事是同一条链：pair 势要能"每步收到一份共享的 pair 数据"，才谈得上两份 LJ 实现合并成一个 kernel；PME 要能"作为 pair 势和 lj 共用那份数据"，`kspace` 类别才有理由消失。拆开会在两个规范之间留下两套 pair 数据通路或两份 LJ 实现并存——正是本次裁决要消灭的腐坏。触及 core/spatial、ff、md 与三个 binder 面，其中 binder 是引擎面的**镜像**而非独立设计（CLAUDE.md：binders 只测 FFI 接缝），必须同批落地。

### 1. 每步 pair 数据直通（替掉 `set_pairs` 快照）

**今天的形状（要删的）：** `Potential::set_pairs(&[SkinPair], &SimBox)`（`ff/potential/mod.rs:147`）把 pairs 与 box **存进势里**（`md/lj_cut.rs:198-201` 的 `pairs` / `simbox` 字段），势在 `calc_energy_forces` 里对每个 pair **自己做 MIC**（`:337`）。两个后果：每多一个 pair 势就把同一份几何**重算一遍**；势里住着一份会过期的邻居快照。

**新形状（信封即数据）：** 循环每步算一次，直接传：

- `VerletSkin::pairs_at(&mut self, positions) -> Result<&Neighbors, SkinError>`（new）：跑完 `update` 的策略后，把当前几何下的 `(i, j, disp, dist_sq)` 物化进 skin 自己持有的**可复用缓冲**（`NeighborsStorage::FULL`），返回共享借用。实现直接折叠既有的 `for_each_pair_at`——不新写一份 MIC 算术。**全流程中唯一计算 MIC 的地方**。
- `Potential::calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>)`：默认实现忽略 `pairs`、转调 `calc_energy_forces(coords)`——bonded 与已编译 pairs 块的 kernel 一行不改。`Potentials` 同名方法把**同一个** `&Neighbors` 转发给每个成员：一份数据，lj / coul / … 共享。
- `set_pairs`（trait 方法、`Box<dyn Potential>` 转发、`Potentials::set_pairs`、`LJCut::set_pairs` 与其两个字段）**全删**。

信封为什么不是"god blob"：`Neighbors` 是**一张表的四列**（indices / disp / dist_sq + mode），一个职责，且它**已经在树上**（`core/spatial/neighbors/mod.rs:795`，`disp()` / `dist_sq()` 列俱全）——本规范不新建信封类型。形状对齐 molnex 的 data-envelope（`edges` 命名空间：`edge_index` / `edge_diff` / `edge_dist` 由邻表算一次、下游只读），跨仓一致。

**pair 来源在构造时定死。** 合并后的 kernel 有两种 pair 来源，二选一，构造时确定，私有枚举承载：
- `Compiled { atom_i, atom_j }` —— ff 路径（`pairs` 块，静态拓扑，无 PBC）：**不覆盖** `calc_energy_forces_with_pairs`，因此循环即使传了每步数据它也只用自己那份，不会重复计入。
- `Loop` —— md 路径（uniform ε/σ）：`calc_energy_forces(coords)` 在没有 pair 数据时返回**显式的 0**（与今天"未喂 pairs 即 0"一致，不是缺项），`calc_energy_forces_with_pairs` 折叠传入的数据集。

**被删掉的 skin-owning 驱动。** `PairPotential::eval` / `eval_into`（`md/lj_cut.rs:48,60`）会在势里调 `neighbors.update(pos)` ——邻居策略是循环的事，与本裁决直接冲突，删（Python 面 `LJCut.eval` 一并删）。等价组合由调用者拼：`skin.update(pos)` + `lj.eval_table(n, skin.pairs_at(pos))`，两个原语（CLAUDE.md：组合是调用者的事）。为此 `VerletSkin` 的 PyO3 面补出 `pairs_at`——净公开符号 −1。`eval_pairs` / `eval_table` **保留**（它们本来就是直通折叠，且已是公开 Python API）。

### 2. 单一 LJ pair kernel

`PairPotential` trait 迁到它的家 `ff::potential::pair`（pair kernel 家族所在），只留纯 kernel 三件套 `pair_energy` / `pair_force` / `pair_eval` 与两个直通折叠 `eval_pairs` / `eval_table`（错误类型随 ff 约定改为 `String`；PyO3 侧异常类型不变，由测试钉死）。合并后的唯一类型 `LJCut` 落在 `ff/potential/pair/lj_cut.rs`，`md` **re-export** 它（`molrs::md::{LJCut, PairPotential}` 与 Python `molrs.md.LJCut` 拼写全部不变——这是纯 API 统一，用户可见名字不动）。`md/lj_cut.rs` 整个文件删除。`pair_lj_cut_ctor` 改为构造 `Compiled` 形态（LB 混合 + 1-4 权重的构造逻辑原样搬运，一个数不改）。

**算术形式的裁定（必须一次说清）。** 两份实现在同一处不同：md 版算 `inv_r2 = 1.0/r2` 后乘（`md/lj_cut.rs:281,286`），ff 版两次用除法（`ff/potential/pair/lj_cut.rs:54,59`）。二者代数相等、浮点不等（上界 1 ulp）。裁定：**保留 `inv_r2` 形式**——每对少一次除法，而 MD 循环是热路径。因此：
- md 路径（uniform）**逐位相同**，`assert_eq!` 断言，不用容差；
- ff 路径（compiled）在两式一致处逐位相同，不一致处**差异必须 ≤ 1 ulp 且能由"乘倒数 vs 除法"解释**——测试体内当场把两式都算一遍来证明，而不是把断言放宽成相对容差（有容差就是在藏东西）。超过 1 ulp、或差异出现在截断遮罩 / 1-4 缩放 / LB 混合任何其他量上，都是硬失败，按铁律上报。
- **considered and dropped**：保留除法形式让 ff 路径逐位不变、由实验性的 md 路径吸收 1 ulp。弃的理由是热路径每对多一次除法；若维护者更看重 ff 侧既有黄金值的字节稳定，此处一行即可翻转。
- **退化对守卫统一**：ff 版跳过 `r2 < 1e-24`，md 版跳过 `r2 <= 0.0`（严格更弱）。统一取 `r2 < 1e-24`——受影响的只有间距 < 1e-12 Å 的对，那种几何本来就产出 inf。此变化写进文档与测试。

### 3. kspace 名字退出 ForceField 表面

- **删名字**：`StyleDefs::KSpace` 变体与 `category()` 的 `"kspace"` 串（`ff/forcefield/mod.rs:145,158`）、`collect_type_params` 的 dummy 行（`:190`）、各 match 臂（`:189,448,556,578,601,625,645,806,959,1022,1032`）、`def_kspacestyle`（`:782`）及其三个镜像面：`molrs-python/src/ff/mod.rs:1055`、`molrs-python/python/molrs/ff/forcefield.py:549,715,778`、`molrs-capi/src/forcefield.rs:830` 与其文档行 `:4`。**不留任何 `def_kspace` 形状的东西。**
- **留模块**：`ff/potential/kspace/`（`mod.rs` + `pme.rs`）**保留为编译单元**。理由写进模块头文档：`ff = ["dep:rustfft", ...]`（`molrs/Cargo.toml:51`）今天把 FFT 依赖绑在整个 `ff` 上；保住这个模块边界，未来才可能用一个 feature 把 FFT 从 `ff` 里门控出去（0.15，不在本规范）。**边界删了就再也接不回来**——这与 13 号保留 `io::store::zarr` 适配模块是同一条规则：技术名活在编译单元，死在公开面。
- **PME 归位 pair**：注册键 `("kspace", "pme")` → `("pair", "coul/long/pme")`（`ff/potential/registry.rs:254`），`ParamSource::PerInstance` 不变，kernel 一行不改。名字取 LAMMPS pair 词汇的斜杠形状（既有 `lj/cut`、`coul/cut` 同源，LAMMPS 自己的 `lj/cut/coul/long`）：`coul/long` 是"带长程处理的库仑"，末段点名求解器，为 `coul/long/ewald` / `coul/long/pppm` 留位。**considered and dropped**：`("pair", "pme")`——求解器名单独成 style 名会让 Ewald/PPPM 进来时没有共同前缀。
- **LAMMPS reader 的 `kspace_style` 跳过保留**（`ff/forcefield/readers/lammps.rs:208`）：输入文件里仍有这行，跳过它是 reader 侧的外部词汇处理，不是我们的类别；把它映射成 pair style 是新能力，不在本规范。
- 连带：`molrs/tests/architecture_gate.rs:392,456` 两张表、`molrs-python/tests/test_forcefield_builder.py:64` 的参数化、CLAUDE.md § Potential System 的 "Categories: bonds, angles, dihedrals, impropers, pairs, kspace" 行。

### Reuse decision

- `reuse` `core::spatial::neighbors::Neighbors`（`mod.rs:795`，`disp()` / `dist_sq()` 列已在）作为每步 pair 数据集**本体**——不新建信封类型。
- `reuse` `VerletSkin::for_each_pair_at`（`verlet_skin.rs:380`）作为 `pairs_at` 的实现体：MIC 算术只有一份，md 路径的逐位相等靠的就是它。
- `reuse` `NeighborsStorage::FULL`（`mod.rs:714`）作为物化策略，不新定义列策略。
- `reuse` `PairPotential::eval_pairs` / `eval_table`（`md/lj_cut.rs:95,149`）作为直通折叠——它们本来就是"从已归约的 pair 列求和"，正是新接缝的形状；随 trait 迁到 ff。
- `reuse` `LJCut::pair_kernel`（`md/lj_cut.rs:275`）的 12-6 快路径与 Mie 通路作为合并后的**唯一**算术。
- `reuse` `pair_lj_cut_ctor`（`ff/potential/pair/lj_cut.rs:87`）的 LB 混合 + 1-4 权重构造逻辑，原样搬进合并后的 `Compiled` 构造器。
- `reuse` `kspace::pme::pme_ctor` 与 `PmePotential` 原样（只换注册键与文档），并 `reuse` 03 号的 ONE `Potential` 裁决——PME 从来就是一个 `Potential` 实现，本规范只改它的类别标签。
- `generalize` `Potential` trait：新增 `calc_energy_forces_with_pairs`（默认转调 `calc_energy_forces`），把"接收循环侧 pair 数据"从 md 的 `LJCut` 一处扩展到任意 pair 势；`set_pairs` 同时删除，**不是**两条并存的路。
- `generalize` `VerletSkin`：从"只交出陈旧 edges"扩展为"交出当前几何下的完整 pair 数据集"，服务循环与所有 pair 势两类调用方。
- `reuse` molnex 的 data-envelope 形状（`edges`: `edge_index` / `edge_diff` / `edge_dist` 由邻表算一次、下游只读；molnex CLAUDE.md § Post-collate batch schema）作为跨仓一致的接缝范式。
- `new` — none（无新概念；`pairs_at` 是既有 `for_each_pair_at` 的物化形态，`coul/long/pme` 是既有 kernel 的新注册键）。

## Files to create or modify

- `molrs/src/core/spatial/neighbors/verlet_skin.rs`
- `molrs/src/core/spatial/neighbors/mod.rs`
- `molrs/src/ff/potential/mod.rs`
- `molrs/src/ff/potential/pair/mod.rs`
- `molrs/src/ff/potential/pair/lj_cut.rs`
- `molrs/src/ff/potential/registry.rs`
- `molrs/src/ff/potential/kspace/mod.rs`
- `molrs/src/ff/potential/kspace/pme.rs`
- `molrs/src/ff/forcefield/mod.rs`
- `molrs/src/ff/forcefield/readers/lammps.rs`
- `molrs/src/ff/typifier/estimate/candidate.rs`
- `molrs/src/md/lj_cut.rs` (delete)
- `molrs/src/md/mod.rs`
- `molrs/src/md/integrators.rs`
- `molrs/tests/architecture_gate.rs`
- `molrs-python/src/md.rs`
- `molrs-python/src/core/spatial/neighborlist.rs`
- `molrs-python/src/ff/mod.rs`
- `molrs-python/python/molrs/ff/forcefield.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_md.py`
- `molrs-python/tests/test_forcefield_builder.py`
- `molrs-capi/src/forcefield.rs`
- `CLAUDE.md`
- `.claude/notes/notes.md`
- `regressions/release-0-14-14-pair-kernel-merge.py` (new)

## Tasks

- [ ] Write failing bit-identity tests capturing pre-merge LJ energies and forces for both call paths — uniform `md::LJCut` over loop-fed pairs and compiled `PairLJCut` over a `pairs` block (`molrs/src/ff/potential/pair/lj_cut.rs` `#[cfg(test)]`)
- [ ] Write failing seam tests: no `set_pairs` symbol survives, two pair potentials in one `Potentials` see the same per-step dataset, a compiled pair kernel ignores it, and no potential computes MIC (`molrs/src/ff/potential/mod.rs` `#[cfg(test)]`, `molrs-python/tests/test_md.py`)
- [ ] Write failing kspace-absence tests: `StyleDefs::KSpace` and the `"kspace"` category are gone, `def_kspacestyle` exists on no surface, and `lookup_kernel("pair", "coul/long/pme")` resolves (`molrs/src/ff/forcefield/mod.rs` `#[cfg(test)]`, `molrs-python/tests/test_forcefield_builder.py`)
- [ ] Implement `VerletSkin::pairs_at` on top of `for_each_pair_at` with a reusable `Neighbors` buffer, and mirror it on `molrs-python/src/core/spatial/neighborlist.rs`
- [ ] Replace `Potential::set_pairs` with `calc_energy_forces_with_pairs(coords, &Neighbors)` in `molrs/src/ff/potential/mod.rs`, forward it from `Potentials`, and rewire `eval_potential` in `molrs/src/md/integrators.rs` plus the PyO3 md seam (deleting `LJCut.eval` / `PairPotential::eval*` skin drivers)
- [ ] Merge `md::LJCut` and `PairLJCut` into the single kernel in `molrs/src/ff/potential/pair/lj_cut.rs` with a construction-fixed pair source, move `PairPotential` to `ff::potential::pair`, re-export both from `md`, and delete `molrs/src/md/lj_cut.rs`
- [ ] Delete the KSpace surface across `molrs/src/ff/forcefield/mod.rs`, `molrs-python/src/ff/mod.rs`, `python/molrs/ff/forcefield.py` and `molrs-capi/src/forcefield.rs`, and register PME as `("pair", "coul/long/pme")` while keeping `ff/potential/kspace` as the FFT compilation unit
- [ ] Update `molrs/tests/architecture_gate.rs` style tables, the CLAUDE.md category line, and close the 2026-08-25 merge-debt entry in `.claude/notes/notes.md`
- [ ] Add regression example `regressions/release-0-14-14-pair-kernel-merge.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Rust 单测就地 `#[cfg(test)]`（本仓无 `molrs/tests/` 单测树）；Python 单测平铺，每条只测一个不变式。**本规范不新增任何守恒 / 长跑物理测试**——04 号的 NVE 判据（1000 步、相对漂移 < 5e-5）保持原样跑绿即可，阈值一个字不许改。

- 逐位相等（md 路径，最重要的一条）：同一个盒子、同一组坐标、同一条 skin，合并前后 `LJCut` 的能量与力数组 `assert_eq!` **逐位相同**，不用容差。依据是 `pairs_at` 复用 `for_each_pair_at`，MIC 算术与遍历顺序都与旧内联版逐字相同。
- 逐位相等（ff 路径）：同一 `pairs` 块与坐标，合并前后能量/力在两式一致处逐位相同；不一致处断言差异 ≤ 1 ulp，且测试体内**当场把 `x/r2` 与 `x*(1/r2)` 两式都算一遍**证明差异来源。任何更大的差异是硬失败。
- 一份数据、多方共享：把两个 pair 势（`LJCut` + 一个计数用的测试势）放进同一个 `Potentials`，断言两者收到**同一个** `&Neighbors`（指针/长度与内容一致），且每步 skin 只物化一次（物化计数器 == 1）。
- pair 来源固定：编译形态的 kernel 放进带 skin 的循环里，能量等于它单独用编译 pairs 算的值——证明它没把每步数据集也计入（防重复计数）。
- 缺席门：全仓 `set_pairs` 命中为零（历史笔记除外）；任何 `ff::potential` / `md` 下的势中不出现 `mic(` / `minimum_image`——MIC 只许出现在 `core::spatial`。
- 边界：`Loop` 形态在无 pair 数据时返回**显式 0** 而非报错；退化对 `r2 < 1e-24` 被跳过（新统一守卫）；`eval_pairs` 列长度不一致仍报同样形状的错误。
- kspace 表面：`StyleDefs` 上不存在 KSpace 变体（编译期即证）；`ff.def_style("kspace", …)` 在三个绑定面均不可达；`lookup_kernel("pair", "coul/long/pme")` 有值而 `lookup_kernel("kspace", "pme")` 为 `None`；同一 frame 上 PME 的能量/力在改注册键前后 **`assert_eq!` 逐位相同**（换标签不换数）。
- 编译单元保留：`molrs/src/ff/potential/kspace/mod.rs` 仍存在且被 `ff` 声明，模块头文档写明它作为 FFT 依赖门控边界的理由（一条 grep 门断言该理由行存在——没有理由的边界会被下一个人顺手删掉）。
- 回归样例 `regressions/release-0-14-14-pair-kernel-merge.py`：纯公开 API，两条路径一个数字——(a) `molrs.md.LJCut(eps, sigma, cutoff, shifted=False)` 走 `eval_pairs` 手喂一对；(b) 同参数的 `pair:lj/cut` ForceField 经 `to_potentials(frame).calc_energy_forces(coords)`；断言两者与写死的解析黄金值（`4ε[(σ/r)¹²−(σ/r)⁶]`，本仓 Rust 单测取得，非第三方）在 1e-15 相对误差内，并断言两者互相之差 ≤ 1 ulp。
- 域验证：由上述逐位断言承担；无新物理，故无新物理判据。

## Open questions (maintainer ruling required)

- 无。

## Out of scope

- 任何新的 NVE / 守恒 / 长跑物理测试（裁决：本规范只是 API 变动）
- 把 FFT 依赖真正从 `ff` feature 里门控出去（0.15；本规范只保住模块边界）
- LAMMPS reader 把 `kspace_style` 映射成 pair style（新能力）
- 驱动 pair 路径的 per-type mixing 与 special_bonds 排除（0.15，04 已登记）
- `MD` 驱动形状与 `_pair_kernels` 的收敛（04）
- Ewald / PPPM kernel 的实现（只留出 `coul/long/*` 名字位）
- wasm / capi 的 md 绑定面（0.15）
