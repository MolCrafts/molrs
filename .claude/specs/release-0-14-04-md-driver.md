---
slug: release-0-14-04-md-driver
title: release-0-14-04-md-driver — MD 驱动定形：dtype= 精度入口、唯一 MDState、测试归位
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-03-potential-protocol
  - release-0-14-14-pair-kernel-merge
---

# release-0-14-04-md-driver — 驱动定形

## Summary

把 MD 驱动收到最终形状：精度接口仿 numpy 改为 `MD(dtype=)`（`PRECISIONS` / `resolve_prec` / `prec=` 全删），唯一 `MDState`（PyO3 带 setter），`tests/` 只留接缝短测试而驱动级 NVE 长跑迁进 `regressions/`，`_lib.pyi` 与运行时逐名一致，并验证 `.so` 重建后 7 个预存失败全绿。

## Domain basis

NVE 守恒是本规范唯一的物理断言，权威在 Rust 单元测试：`molrs/src/md/integrators.rs` 的守恒测试在 64 个 Ar-like 原子、`dt = 1 fs`、1000 步下实测**相对漂移 1.71e-5**，断言阈值 **5e-5**。驱动级长跑（1200 步）在 `regressions/` 复现同一判据。

- 速度 Verlet 与其守恒性质：Verlet, *Phys. Rev.* **159** (1967) 98, DOI 10.1103/PhysRev.159.98；Swope et al., *J. Chem. Phys.* **76** (1982) 637, DOI 10.1063/1.442716。
- 截断 LJ 的能量平移等价于 LAMMPS `pair_modify shift yes`（LAMMPS: DOI 10.1016/j.cpc.2021.108171）；平移只改能量零点、不改力，不影响守恒判据量级。
- 时间单位 fs，与 `.claude/notes/science.md` 单位表一致。

## Design

**已在树上、本规范只加锁不重做的部分（delta 纪律）：** `MD` 与 `MDRunner` 的合并已完成（`md/driver.py` 只剩一个 `class MD:`，:47），`FrameVelocityVerlet` 与 NamedTuple 版 `MDState` 已删除（`MDState` 唯一来自 `_lib.md`，PyO3 带 setter），experimental `FutureWarning` 与顶层 PEP 562 惰性加载已就位（`md/__init__.py:66-71`）。本规范**不重做这些**，只补缺席门与行为门把它们钉死。

**精度接口仿 numpy。** `MD(dtype=np.float64)` 是唯一入口（默认 `np.float64`）；`np.float32` / mixed 一律 raise，错误信息声明它们将落在 Rust 积分器。`PRECISIONS`、`resolve_prec`、`prec=` 从 `md/__init__.py:86-98` 与 `driver.py:16,64-83` 全删。`dtype` 是 numpy 词汇，用户不需要学第二套精度名。

**thermo 的 kB。** 02 已把 `MD.run(..., kb=)` 变成显式参数；本规范把 docstring 示范接到上提后的 core preset：`kb=molpy.UnitPreset("real").boltzmann()`（引擎侧路径为 `molrs.UnitPreset`，用户可见处一律拼 `molpy`）。绝不隐式取常数。

**`.pyi` 逐名一致与 Potential 注解。** 加一条按 `dir(molrs._lib.md)` 与 `.pyi` 声明名对拍的测试——用**运行时枚举**而非手写名单（手写名单会被手工缩短，本仓已付过学费）。凡接受力的参数，注解一律单一名字 `md.Potential`（03 落地的 `runtime_checkable` Protocol，即那唯一的类型），`Union` 形状依旧禁止。`set_potential` / `set_forcefield` 的入参契约是**结构化**的：任何具备 `calc_energy_forces` 的对象都被接受，驱动**不做** `isinstance` 守卫（`isinstance` 只查方法存在、不查签名，且不适合热路径）。

**测试归位。** `molrs-python/tests/test_md.py` 只保留**接缝级**短测试（构造、类型、move 语义、异常穿透、数步级推进）。驱动级 NVE 长跑迁到 `regressions/release-0-14-04-md-driver-nve.py`；引用了已删名字的旧文件 `regressions/release-0-14-01-md-driver-nve.py` **删除**（它当前是断裂的，属于必须当场清掉的已知腐坏）。

**预存失败转绿。** `.so` 重建后的 7 个预存失败（`Frame.meta` MetaValue × 6，keys-tuple × 1）在本规范收口处逐条验证为绿；任何一条仍红即为阻断项，按铁律上报而非跳过。

### Reuse decision

- `reuse` 已合并的 `MD` 驱动与 PyO3 `MDState`（含 setter）——不重新设计。
- `reuse` `md/__init__.py` 现有的 `FutureWarning` + 顶层惰性加载实现。
- `reuse` 02 落地的 `core::units::UnitPreset`（Python 侧 `molrs.UnitPreset`）作为 `kb=` 的取值来源。
- `reuse` 03 落地的 `md.Potential` Protocol 作为 `set_potential` / `set_forcefield` 的唯一注解类型与结构化入参契约。
- `new` — none（`dtype=` 是既有 `prec=` 参数的改形，不是新概念）。

## Files to create or modify

- `molrs-python/python/molrs/md/__init__.py`
- `molrs-python/python/molrs/md/driver.py`
- `molrs-python/python/molrs/__init__.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_md.py`
- `regressions/release-0-14-04-md-driver-nve.py` (new)
- `regressions/release-0-14-01-md-driver-nve.py` (delete)

## Tasks

- [ ] Write failing tests for `MD(dtype=np.float64)` acceptance and `np.float32` rejection with a Rust-integrator message (`molrs-python/tests/test_md.py`, `TestMDDtype`)
- [ ] Write failing absence tests for `PRECISIONS`, `resolve_prec`, `prec=`, `FrameVelocityVerlet`, and any second `MDState` (`molrs-python/tests/test_md.py`)
- [ ] Write failing `.pyi`-vs-runtime name-parity test enumerated from `dir(molrs._lib.md)`, plus a gate that force parameters are annotated `md.Potential` with no Union (`molrs-python/tests/test_md.py`)
- [ ] Write failing warning tests: `import molrs` silent, `import molrs.md` emits FutureWarning (`molrs-python/tests/test_md.py`)
- [ ] Implement the `dtype=` entry in `molrs-python/python/molrs/md/driver.py` and delete `PRECISIONS` / `resolve_prec` from `molrs-python/python/molrs/md/__init__.py`
- [ ] Wire the `MD.run` thermo `temp` column to a `UnitPreset`-sourced `kb=`, with the docstring example spelled `molpy`
- [ ] Trim `molrs-python/tests/test_md.py` to seam-level tests and align `molrs-python/python/molrs/_lib.pyi` name-by-name
- [ ] Add regression example `regressions/release-0-14-04-md-driver-nve.py` (public API only; hard-coded goldens, no third-party runtime) and delete the stale `regressions/release-0-14-01-md-driver-nve.py`
- [ ] Verify the 7 pre-existing pytest failures (6 × `Frame.meta` MetaValue, 1 × keys tuple) are green against the rebuilt extension
- [ ] Run full check + test suite

## Testing strategy

- 接缝单测（`tests/`，每条只测一件事）：`dtype=np.float64` 接受；`np.float32` raise 且消息含 "Rust"；`MDState` setter 写回；`set_neighbors` prebuilt 被消费后二次 run 报指名 `set_neighbors` 的 `ValueError`；数步级 `advance_n` 返回有限能量。**不在 `tests/` 里跑千步物理**。
- 结构化入参：一个只定义 `calc_energy_forces`、不继承任何东西的对象可直接传给 `set_potential`——证明驱动没有偷偷加 `isinstance` 守卫。
- 缺席门：`PRECISIONS` / `resolve_prec` / `prec` / `FrameVelocityVerlet` 在 `molrs.md` 与 `driver.py` 源码中零命中。
- `.pyi` 对拍：以 `dir(molrs._lib.md)` 为真值枚举，逐名要求 `.pyi` 有对应声明，反向亦然；并断言 `.pyi` 中 `Union[` 与 `Potential` 不同现。
- 警告门：`import molrs` 在 `catch_warnings(record=True)` 下零 `FutureWarning`；`import molrs.md` 恰好一条。
- 域验证（NVE，硬编码期望值）：Rust 单测为权威——64 原子 Ar-like、`dt = 1 fs`、1000 步，相对漂移断言 `< 5e-5`（实测 1.71e-5）。驱动级回归 `regressions/release-0-14-04-md-driver-nve.py`：同体系 1200 步，断言相对漂移 `< 5e-5` 且 `rebuild_count > 0`（证明邻表确实重建过，不是空转）；黄金值写死在脚本里，不调用任何第三方。
- 预存失败：以 pytest 全量运行的失败计数为门——期望 0。

## Open questions (maintainer ruling required)

- 无。`dtype=` 只收 `float64` 是已裁决项；float32/mixed 的 Rust 积分器落地属 0.15+。

## Out of scope

- Rust 积分器的 float32 / mixed 实现（未来）
- 驱动 pair 路径的 per-type mixing 与 special_bonds 排除（0.15）
- wasm / capi 的 md 绑定面（0.15）
