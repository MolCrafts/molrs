---
title: "chem-perceive 13/14 — molrs-python：Perceive + AtdTypifier + 三个电荷模型上 Python"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-01-layer; chem-perceive-05-atd-typifier; chem-perceive-07-charge-trait; chem-perceive-08-gasteiger"
blocks: ""
---

# Python 首次可达原生 AM1-BCC

> Chain **chem-perceive** 13/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

把 `Perceive`、`AtdTypifier` 和三个电荷模型（`BccModel` / `MullikenModel` / `GasteigerModel`）
暴露到 `molrs-python`，并把 `molrs::chem::` 迁到 `molrs::perceive::`、**删掉 01 期引入的 compat alias**。

这是 **Python 第一次能碰到原生 AM1-BCC**。今天 molrs-python 只绑定了 `MMFFTypifier` /
`OPLSAATypifier` / `compute_gasteiger_charges`，而 molpy 唯一的 AM1-BCC 是
`antechamber -c bcc`（外部 AmberTools 二进制）。绑定落地后，molpy 才有可能拿原生结果去和
antechamber 对账——那正是目前完全缺失的验证。

## Domain basis

01 期引入的 `pub use crate::perceive as chem;` 是为了让 01 可以**独立合并**而不打断
`molrs-python`（它是独立 workspace，通过模块路径 `molrs::chem::…` 导入）。
到本期，binder 迁移完成，alias 的使命结束，必须删掉——否则它会永久固化成一个
"两个名字指同一个东西"的债。

本期合法地跨两个 crate（binder 迁移 + `lib.rs` 里删一行 alias），
因为**先删 alias 再迁 binder** 会打断构建，两者不可分开合并。

### 证据链与出处（所有子 spec 共用）

**绑定真值 = 实测的 antechamber oracle**，不是文献：
- `molrs/tests/ff/typifier/antechamber_oracle.rs` — 37 个分子，由 AmberTools25 的
  `antechamber` 真实跑出来后 hardcode，覆盖芳香/杂芳/羧酸根/硝基/酰胺/S/P/卤素，
  以及 molcrafts 实际在用的 EC、DMC、DME(PEG 片段)、甲基丙烯酸甲酯、咪唑鎓。
- `scripts/gen_am1bcc_oracle.py` — 重新生成该 fixture（tempdir，不落仓库）。

**支撑证据**（可自由获取，非付费墙）：
- Antechamber 论文 <https://ambermd.org/antechamber/antechamber.pdf>（§2.1 式 (I)、§2.2 七种键型）
- antechamber C 源码 `bondtype.c` / `equatom.c` / `am1bcc.c` / `charge.c` / `mol2.c`
- 本地 AmberTools25 二进制与 `.DAT`/`.DEF` 数据表（可执行 oracle）

**未读到的一次文献（付费墙，任何结论都不建立在其上）**：
Jakalian, Bush, Jack & Bayly, *J Comput Chem* 2000, **21**:132;
Jakalian, Jack & Bayly, *J Comput Chem* 2002, **23**:1623;
Gasteiger & Marsili, *Tetrahedron* 1980, **36**:3219。
不得杜撰这三篇论文的公式细节。

### 风险：GPL / clean-room

本链条重新实现了 antechamber 的 `bondtype.c`(03)、`equatom.c`(04)、`atomtype.c`(05/06)。
`.claude/notes/notes.md`（2026-06-19）当初否决原生 GAFF typifier 的理由之一正是
*"Trustworthy native GAFF typing would need a clean-room reimplementation of antechamber's
GPL atomtype.c"*——现在这个顾虑以更大规模重现。参数**表**（`.DAT`/`.DEF`）是数据，
但**算法**是 GPL C 源码。合并前必须明确并记录 clean-room / 授权姿态，不得默默推进。

## Files to create or modify

- 修改 `molrs/molrs-python/src/core/system/molgraph.rs`、`src/lib.rs`
- 新增 `molrs/molrs-python/src/ff/charge.rs`（三个模型的绑定）
- 修改 `molrs/molrs-python/python/molrs/{__init__.py,molrs.pyi}`
- 修改 `molrs/molrs/src/lib.rs`（**删除** compat alias）

## Tasks

- [ ] Expose `Perceive` with its `find_*` methods to Python
- [ ] Expose `AtdTypifier` (parameter-set selectable) to Python
- [ ] Expose `BccModel(...).correct(mol, am1)`, `MullikenModel`, `GasteigerModel().assign(mol)`
- [ ] Migrate `molrs::chem::` → `molrs::perceive::` in molrs-python/src/core/system/molgraph.rs:27-28
- [ ] Remove the `pub use crate::perceive as chem;` compat alias from molrs/src/lib.rs
- [ ] Update the .pyi stubs and __init__.py exports

## Testing strategy

- Python 侧 smoke：`molrs.BccModel(...).correct(mol, am1)` 返回 ndarray，
  在同一个 37 分子集上与 Rust 侧结果逐位一致。
- `molrs.Perceive().find_rings(g)` 返回图。
- grep gate：全仓 `molrs::chem` **0 命中**；alias 已从 `lib.rs` 移除。
- 已有的 Python 测试（`perceive_aromaticity` / `add_hydrogens` / `find_rings` /
  `compute_gasteiger_charges`）全绿（可以改导入路径，不改断言值）。

## Out of scope

- **molpy 侧的接入**（`molpy.charge.bcc(...)`）与 molpy↔antechamber 对账 harness：
  属于 molpy 仓库，另立 spec。
- 保留 `compute_gasteiger_charges` 这个自由函数：可以留作兼容层，
  但它必须转调 `GasteigerModel`（不能有第二份实现）。
