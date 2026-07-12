---
title: "chem-perceive 9/14 — gaff.dat / gaff2.dat → committed .rs + ForceField 填充"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-02-param-rs"
blocks: "chem-perceive-10-parmchk-tables; chem-perceive-11-param-estimate"
---

# GAFF / GAFF2 力场参数表编译期化

> Chain **chem-perceive** 9/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

把 `gaff.dat`(7312 行) 与 `gaff2.dat`(13181 行) 也翻成**提交的 Rust 静态表**
（MASS / BOND / ANGLE / DIHE / IMPROPER / NONBON），并用 06 产出的 GAFF/GAFF2 原子类型
去填充 `ForceField`。运行期依然**不解析任何文本**。

## Domain basis

`gaff.dat` / `gaff2.dat` 是 AMBER 的 `parm` 格式：分节的 MASS / BOND / ANGLE / DIHE /
IMPROPER / NONBON。生成的 Rust 结构**必须保留通配符（`X`）行**，因为 11 期的回退匹配要用。

### 体积不是问题——已实测

实测把 gaff.dat + gaff2.dat + BCCPARM（共 **15,474** 行 static）编进一个真实消费它们的
release 二进制：**stripped 增量 +1071 KB，编译 0.37 s**。而 molrs 今天已经用 `include_str!`
编进去 **3974 KB** 的原始文本。所以 13k 行不是风险，**不需要** phf/二分的特殊布局，
直接扁平 static slice 即可；真的需要时再优化，但**绝不退回运行期文本解析**。

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

- 修改 `scripts/gen_param_tables.py`（从 `$AMBERHOME` 读 `gaff.dat`/`gaff2.dat`）
- 新增 `molrs/src/ff/params/generated/{gaff,gaff2}.rs`
- 修改 `molrs/src/ff/forcefield/**`（GAFF/GAFF2 填充路径）

## Tasks

- [ ] Extend `scripts/gen_param_tables.py` to emit `.rs` for gaff.dat and gaff2.dat (MASS/BOND/ANGLE/DIHE/IMPROPER/NONBON), preserving wildcard (X) rows
- [ ] Populate `ForceField` from GAFF/GAFF2 atom types (exact-match terms only; fallback is 11)
- [ ] Confirm the measured budget holds in-tree (baseline: +1071 KB / 0.37 s for 15,474 rows)

## Testing strategy

- Drift guard：生成器逐字节复现两张 `.rs`（`$AMBERHOME` 未设置时跳过）。
- 37 分子集用 GAFF 类型填 `ForceField`，所有**精确命中**的 term 都被找到；
  缺参此时**直接报错**（回退在 11 期实现）。
- 二进制/编译时间不超出实测预算。

## Out of scope

- 缺参回退 / parmchk2 估计（11）。
- frcmod 文件读写（本链一律不做，见 11）。
