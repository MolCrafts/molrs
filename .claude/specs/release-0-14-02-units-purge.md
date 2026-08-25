---
slug: release-0-14-02-units-purge
title: release-0-14-02-units-purge — 单位 preset 上提 core、三套词汇归一、MD 内部换算清零
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-01-baseline
---

# release-0-14-02-units-purge — MD 零单位知识，单位有唯一的家

## Summary

把单位制 preset 机制上提到 `molrs::core::units`，以引擎中立的类型名承载（preset 的**名字**保留 `"real"` / `"metal"` / `"lj"` 这类熟悉字串，**类型名与模块名不带 LAMMPS**）；`ff` 的 `LammpsUnits` / `LammpsUnitSystem` 与 zarr 的 `UnitSystem` 降为它的消费者。同时把 MD 内部所有隐式换算连根删掉——包括 `md::units` 模块本身——`MaxwellBoltzmann::new(temperature, seed)` 改为 `new(kbt, seed)`。

## Domain basis

本规范搬运常数、不改变数值。

- 常数早已在 core：`molrs/src/core/units/constants.rs` 的 `COULOMB_REAL`(:40) 与 `BOLTZMANN_REAL`(:43)。`md::units::kb_md()` 一直是它们的**第二份副本**——这既是删除它的理由，也是 preset 上提到 core 的独立证据：数据本来就在那儿，只有词汇散在三处。
- 单位制语义沿用 LAMMPS `units`（Thompson et al., *Comput. Phys. Commun.* **271** (2022) 108171, DOI 10.1016/j.cpc.2021.108171；https://docs.lammps.org/units.html）。项目分析时间单位为 **fs**（`.claude/notes/science.md`）。
- `MaxwellBoltzmann` 采样物理不变：旧式 `kb = kb_md(); scale = sqrt(kb·T/m)`，新式 `scale = sqrt(kbt/m)`。传 `kbt = BOLTZMANN_REAL·T` 时采样**逐位一致**——本规范唯一触及物理之处，必须以逐位相等断言证明，不用容差。
- Maxwell–Boltzmann 分布：Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.2。Langevin `kbt` 形状：Bussi & Parrinello, *Phys. Rev. E* **75** (2007) 056707, DOI 10.1103/PhysRevE.75.056707。

## Design

**上提目标已存在。** `molrs/src/core/units/` 已有 `registry.rs`（`UnitRegistry:120`、`define_lj_units:250`）、`constants.rs`、`unit.rs`、`dimension.rs`、`quantity.rs`，由 `molrs/src/core/mod.rs:51` 声明。preset 机制落在 `molrs/src/core/units/preset.rs`（new），与它们同处——这是**上提到既有的家**，不是新建一层。

**类型形状（OOP、引擎中立命名）：**

- `UnitPreset`：拥有型只读值，命名构造器 `UnitPreset::real()` / `::metal()` / `::si()` / `::lj()` / …（**禁止** `make_*` / `build_*` / `create_*` 工厂）。只读访问器 `boltzmann()` / `coulomb()` / `length()` / `energy()` / `time()` ……，每个方法只做一件事，返回该单位制下的数值。**不提供** `convert(value, from, to)` 式门面——组合是调用者的事。
- `UnitPresetRegistry`：`register(name, data)` 扩展点（形状镜像 molpy `UnitSystem.register_preset`，`<molpy>/src/molpy/core/unit.py:244`）与 `get(name)`。preset **名字**保留 lammps 风格字串（用户认得），**类型名与模块名一律不含 "Lammps"**。
- 覆盖 7 个 style × 10 个维度，参考形状取 molpy `_LAMMPS_PRESETS`（`core/unit.py:57-141`）。**新增的数值常数一律进 `core/units/constants.rs`，不进 `preset.rs`** ——常数只有一个家，preset 只是它们的按单位制视图。`lj` style 经 `UnitRegistry::define_lj_units`（`registry.rs:250`）取得，不另写一份。

**三套词汇归一：**

- **ff（0.14 内完成）**：`LammpsUnits` / `LammpsUnitSystem` 两个**类型删除**。`molrs/src/ff/forcefield/lammps_units.rs` 保留为 LAMMPS reader 的**薄适配器**——只做 "LAMMPS style token → core preset 名" 的映射，**不再持有任何单位数据、不再定义任何 `*UnitSystem` 类型**。（文件名含 lammps 是正确的：它是 LAMMPS 读写器的适配层；R2 禁止的是 preset **设施本身**的类型/模块名带 LAMMPS。）消费者 `readers/lammps.rs`、`writers/lammps.rs`、`molrs-python/src/ff/mod.rs` 改经 core。它已在 `:92` 调用 `reg.define_lj_units`，此举只是把数据挪回它本来所在的层。
- **zarr（0.14 内降为消费者，完全删除记为跟进）**：`molrs/src/io/store/zarr/mod.rs` 的第三套 `UnitSystem` 枚举改为把自身变体映射到 core preset 名，不再独立持有单位语义；**枚举的彻底删除**涉及已落盘 Zarr 元数据的兼容性，作为跟进项登记进 `.claude/notes/notes.md`（带目标版本），不在发布窗口里赌存量数据的读取。
- 此前"三套单位词汇哪个 canonical"的开放项已由维护者裁定——canonical 家是上提后的 `core::units` preset 系统，该开放项已从本规范移除。

**MD 内部换算清零：**

- 删除 `md::units` **整个模块**（`molrs/src/md/units.rs` 与 `md/mod.rs:34` 的 `pub mod units;`）。MD 零单位知识，一个叫 `md::units` 的模块本身就是自相矛盾；用户改用 `core::units::UnitPreset`（Python 侧 `molrs.UnitPreset`，用户可见拼写 `molpy.UnitPreset`）。
- 删除 `Potentials::set_energy_scale` 与 `energy_scale` 字段及其缩放乘法（`molrs/src/ff/potential/mod.rs:179,235,241,267-272`）。
- 调用点连带修：`md/maxwell.rs:10,86`、`md/integrators.rs:506,596`（测试）、`ff/potential/mod.rs:609-626`（测试）、`md/lj_cut.rs:183`（doc）。
- 绑定面镜像删除：`molrs-python/src/md.rs:26-27,700-725`、`molrs-python/src/ff/mod.rs:448-464`。绑定面是引擎面的**镜像**而非独立设计（CLAUDE.md：binders 只测 FFI 接缝），必须同批落地，否则留下编译不过的 binder 树——正是铁律禁止的静默债。
- Python 面：`md/__init__.py`（导出与单位段 docstring 重写）、`md/driver.py`（`set_forcefield` 去掉 `energy_scale`；`MD.run(..., thermo=N)` 要求**显式 `kb=`**，缺失时 raise 指名 `kb=` 的 `ValueError`）、`_lib.pyi`。
- `MaxwellBoltzmann::new(kbt, seed)`（`molrs/src/md/maxwell.rs:29,65,86,90`）。`kinetic_energy` / `scalar_mass` / `removed_dof` 不动（本就无单位知识）。

**跨层范围说明。** 本规范触及 core / ff / io / md / binder / python 六个面。这是**一个概念**（单位词汇）的统一：任何进一步切分都会在两个规范之间留下两套并存的单位词汇，正是本次裁决要消灭的腐坏。若仍要切，唯一无腐坏的切点是「core 上提 + 消费者降级」与「MD 换算删除」两段，但两段之间 `md::units` 会短暂重复 core 的常数。

### Reuse decision

- `reuse` `core::units::registry.rs:120 UnitRegistry` 与 `define_lj_units:250` 作为 preset 的底座与 `lj` style 数据源。
- `reuse` `core::units::constants.rs:40,43` 的 `COULOMB_REAL` / `BOLTZMANN_REAL` 作为唯一常数来源（**不复制**）；新增常数也进该文件。
- `generalize` `ff::forcefield::lammps_units` 的 3 style × 4 维度表 → 7 style × 10 维度并**上提** core；原类型删除，文件降为 reader 侧薄适配器。
- `generalize` zarr `UnitSystem`（`io/store/zarr/mod.rs`）→ 降为 core preset 的消费者；完全删除记为跟进项。
- `reuse` `Langevin::new` 的 `kbt` 形状（`integrators.rs:271`）作为 `MaxwellBoltzmann` 的范本。
- `reuse` molpy `_LAMMPS_PRESETS` / `register_preset`（`<molpy>/src/molpy/core/unit.py:57-141,244`）作为数据形状与扩展点范式；参照的是 `core/unit.py`，molpy **没有** `md/units.py`。
- `new` — `UnitPreset` / `UnitPresetRegistry`：`UnitRegistry` 是量纲/解析设施，不是"某单位制下的常数视图"；职责不同，故新增一层而非把 `UnitRegistry` 直接塞给用户。

## Files to create or modify

- `molrs/src/core/units/preset.rs` (new)
- `molrs/src/core/units/mod.rs`
- `molrs/src/core/units/constants.rs`
- `molrs/src/ff/forcefield/lammps_units.rs`
- `molrs/src/ff/forcefield/readers/lammps.rs`
- `molrs/src/ff/forcefield/writers/lammps.rs`
- `molrs/src/io/store/zarr/mod.rs`
- `molrs/src/ff/potential/mod.rs`
- `molrs/src/md/units.rs` (delete)
- `molrs/src/md/mod.rs`
- `molrs/src/md/maxwell.rs`
- `molrs/src/md/integrators.rs`
- `molrs/src/md/lj_cut.rs`
- `molrs-python/src/core/units.rs`
- `molrs-python/src/md.rs`
- `molrs-python/src/ff/mod.rs`
- `molrs-python/python/molrs/md/__init__.py`
- `molrs-python/python/molrs/md/driver.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_md.py`
- `molrs-python/tests/test_units.py`
- `.claude/notes/notes.md`
- `regressions/release-0-14-02-units-purge.py` (new)

## Tasks

- [ ] Write failing absence tests for `MD_ENERGY` / `energy_to_md` / `preset_energy_to_md` / `kb_md` / `set_energy_scale` / `energy_scale` and for the `md::units` module itself (`molrs-python/tests/test_md.py`, `molrs/src/md/mod.rs` `#[cfg(test)]`)
- [ ] Write failing unit tests for `UnitPreset` / `UnitPresetRegistry` presets, the register extension point, and bit-identity of `MaxwellBoltzmann::new(kbt, seed)` (`molrs/src/core/units/preset.rs` `#[cfg(test)]`, `molrs/src/md/maxwell.rs` `#[cfg(test)]`, `molrs-python/tests/test_units.py`)
- [ ] Implement `UnitPreset` and `UnitPresetRegistry` in `molrs/src/core/units/preset.rs` on `UnitRegistry` and `constants.rs`, with engine-neutral type names and 7 preset name strings
- [ ] Reduce `molrs/src/ff/forcefield/lammps_units.rs` to a reader-side token→preset adapter and repoint `readers/lammps.rs`, `writers/lammps.rs` and `molrs-python/src/ff/mod.rs` onto the core facility
- [ ] Make the zarr `UnitSystem` enum a consumer of core presets in `molrs/src/io/store/zarr/mod.rs` and record the full-collapse follow-up in `.claude/notes/notes.md`
- [ ] Delete `molrs/src/md/units.rs` and its module declaration plus `Potentials::set_energy_scale` / `energy_scale`, fixing every Rust call site
- [ ] Implement `MaxwellBoltzmann::new(kbt, seed)` in `molrs/src/md/maxwell.rs`
- [ ] Delete the mirrored PyO3 conversion surface in `molrs-python/src/md.rs` / `src/ff/mod.rs`, expose `UnitPreset` via `molrs-python/src/core/units.rs`, and update `md/__init__.py`, `md/driver.py` (`run(kb=)`) and `_lib.pyi`
- [ ] Add regression example `regressions/release-0-14-02-units-purge.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Rust 单测就地 `#[cfg(test)]`（本仓无 `molrs/tests/` 树）；Python 单测平铺 `molrs-python/tests/test_*.py`。

- 缺席门：六个符号在 `molrs.md` / `molrs._lib.md` / `molrs.ff` 的 `dir()` 中零命中；`md::units` 不再是模块；全仓 grep `LammpsUnitSystem` / `LammpsUnits` 类型零命中（历史笔记除外）。
- `UnitPreset` 单测（域验证，硬编码期望值）：`UnitPreset::real().boltzmann()` 与 `core::units::constants::BOLTZMANN_REAL` **逐位相等**；`real()` 能量为 kcal/mol、时间为 fs。7 个 preset 以**遍历注册表**方式断言各返回 10 个维度——不写死名单（手写名单会被手工缩短，本仓已付过学费）。
- 收编等价性：LAMMPS force-field reader 在改动前后对同一 committed `.ff` fixture 产出**逐位相同**的参数；zarr 存量元数据在改动后仍可读（这两处是收编唯一可能悄悄改数/改行为的地方）。
- `MaxwellBoltzmann`（域验证）：同 seed、同质量下 `new(BOLTZMANN_REAL*300.0, seed)` 与旧 `new(300.0, seed)` 的速度数组 `assert_eq!` **逐位相等**，不用容差——有容差就是在藏东西。
- 边界：`UnitPresetRegistry::register` 重名 raise；`MD.run(thermo=N)` 缺 `kb=` 时 raise 且消息含 `kb=`。
- 回归样例 `regressions/release-0-14-02-units-purge.py`：纯公开 API——`UnitPreset("real")` 取 k_B，用户自行把 ε 换算到 MD 单位，构造 `LJCut` 跑 10 步，断言总能量与写死黄金值（本仓 Rust 单测取得，非第三方）在 1e-12 相对误差内。

## Open questions (maintainer ruling required)

- 无。（`md::LJCut` 与 `ff::potential::pair::lj_cut::PairLJCut` 的合并已裁定：0.14 内做，由 14 号规范承担，且它是**纯 API 统一**——md 没有单位这个概念，合并从来不需要任何单位手术做前置。）

## Out of scope

- `MD(dtype=)` 与驱动形状（04）
- `Potential` Protocol（03）
- zarr `UnitSystem` 枚举的彻底删除（跟进项，涉及存量元数据兼容）
- `LJCut` / `PairLJCut` 内核合并（14）
