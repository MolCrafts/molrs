---
title: "chem-perceive 11/14 — 原生缺参估计（parmchk2 算法）——彻底不碰 frcmod 文件"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-09-gaff-params; chem-perceive-10-parmchk-tables"
blocks: ""
---

# 缺参估计：在既有 ParameterEstimator 上做 parmchk2，不做文件 I/O

> Chain **chem-perceive** 11/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

**不读写 frcmod，不依赖任何外部文件。** 删掉 `molrs/src/ff/frcmod.rs:33,64` 的
`Frcmod::parse_str` / `write_string` 及其调用方。

改为在**既有的参数估计器架构**上原生实现 parmchk2 式的缺参估计——
`ParameterEstimator` / `ParameterInterpolator` / `TypifierParameterContext`
（`ff/typifier/estimate/`，861 + 211 行）已经提供了
`estimate_bond` / `estimate_angle` / `estimate_dihedral`、类比 + 序列/取向/端位/替代惩罚、
`PenaltyTier`、`EstimateMethod`，以及经验键（Badger）/键角公式。

本期要加的是：由（现已 `.rs` 化的）`gaff_equiv` corr / weights / defaults 表驱动的
**通配符 + 原子类型等价回退**，以及一个 GAFF/GAFF2 的参数上下文（与既有的 OPLS 上下文并列）。

## Domain basis

parmchk2 的回退层级：**精确命中 → 通配符（`X`）行 → 原子类型等价类替代（`corr` 行，带惩罚权重）
→ 经验公式**。molrs 的 `PenaltyTier` / `EstimateMethod` 已经是这套结构的抽象，
`gaff_equiv.json` 的 `weights` / `defaults` / `types` 就是它的数据，
`gaff_empirical.json` 的 `bond_power_m` / `bond_lnk` / `angle_zc` 就是经验公式的系数。

⇒ 本期**不是从零造 parmchk2**，而是把既有估计器接到（已 `.rs` 化的）GAFF 表上，
并补上通配符/等价回退这一段。用户原话："我们已经有了参数估计器的架构，实现 frcmod 等插值算法"。

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

- 修改 `molrs/src/ff/typifier/estimate/mod.rs`（GAFF 上下文 + 通配符/等价回退）
- **删除** `molrs/src/ff/frcmod.rs` 的 `parse_str` / `write_string`（整个文件很可能可删）
- 修改 `molrs/src/ff/mod.rs`（移除 frcmod 的 re-export）

## Tasks

- [ ] Add a GAFF/GAFF2 `TypifierParameterContext` alongside the existing OPLS one
- [ ] Implement wildcard + atom-type-equivalence fallback driven by the `.rs` gaff_equiv corr/weights/defaults tables
- [ ] Wire the empirical bond (Badger) / angle formulas from gaff_empirical for terms with no analogy
- [ ] Delete `Frcmod::parse_str` / `write_string` and every caller; molrs must not depend on an external frcmod file
- [ ] Turn the parmchk2 oracle (10) green

## Testing strategy

- 对 37 分子集，**parmchk2 写进 frcmod 的每一个 term 都被原生复现**：
  同样的回退层级（`PenaltyTier`）+ 同样的数值（在表精度内）。
- grep gate：`Frcmod::parse_str|write_string` **0 命中**。
- **没有任何测试读取外部 `.frcmod` 文件。**

## Out of scope

- 非 GAFF 力场的缺参估计（OPLS 上下文已存在，本期不动）。
- RESP / ESP 相关参数。
