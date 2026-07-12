---
title: "chem-perceive 10/14 — gaff_equiv / gaff_empirical → .rs + parmchk2 frcmod oracle（RED）"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-02-param-rs; chem-perceive-09-gaff-params"
blocks: "chem-perceive-11-param-estimate"
---

# 最后一处运行期文本解析归零 + parmchk2 oracle 落地

> Chain **chem-perceive** 10/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

`molrs/src/ff/typifier/estimate/tables.rs:28-36` 目前用 `include_str!` + **运行期 `serde_json` 解析**
读 `gaff_empirical.json`(87 行，Badger 类经验公式) 和 `gaff_equiv.json`(6159 行，parmchk2 的
原子类型等价 + 替代 `corr` 行 + 权重 + 默认值)——而且是 `.expect()` 在嵌入数据上。
这是 FF 路径上**最后一处运行期文本解析**，本期一并归零。

同时把 **parmchk2 的 frcmod oracle** 落地成 RED fixture：对同样 37 个分子跑
`antechamber` + `parmchk2`，把生成的 frcmod hardcode 进来，供 11 期转绿。

## Domain basis

实地核实：`gaff_equiv.json` 的顶层键是 `_source` / `weights` / `defaults` / `types`；
`gaff_empirical.json` 的顶层键是 `_source` / `bond_power_m` / `_bond_formula` / `bond_lnk` /
`_angle_formula` / `angle_zc`。**这就是 parmchk2 的等价表 + 惩罚权重 + 经验公式数据**——
也就是说 parmchk2 的数据侧在 molrs 里**已经有了**，11 期主要是把它接到 gaff.dat 上，
而不是从零造。

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

- 修改 `scripts/gen_param_tables.py`、`scripts/gen_am1bcc_oracle.py`
- 新增 `molrs/src/ff/params/generated/{gaff_equiv,gaff_empirical}.rs`
- 修改 `molrs/src/ff/typifier/estimate/tables.rs`（删除 include_str! + serde_json）
- 新增 `molrs/tests/ff/typifier/parmchk2_oracle.rs`（RED fixture）

## Tasks

- [ ] Generate `.rs` for gaff_equiv.json and gaff_empirical.json (equivalence/corr rows, weights, defaults; bond_power_m, bond_lnk, angle_zc + formulas)
- [ ] Delete the `include_str!` + `serde_json::from_str(...).expect(...)` loads at estimate/tables.rs:28-36,74-76,165-166
- [ ] Extend `scripts/gen_am1bcc_oracle.py` to run `parmchk2` over the 37 molecules and hardcode the resulting frcmod terms
- [ ] Land the parmchk2 oracle as a RED fixture (turns green in 11)

## Testing strategy

- Drift guard：生成器逐字节复现两张 `.rs`。
- grep gate：typifier 路径上 `serde_json` **0 命中**（顺带把两个 `.expect()` on embedded data
  的 fail-fast 违规一起清掉）。
- parmchk2 oracle fixture 已提交且是 **RED**（11 期转绿）。

## Out of scope

- 估计算法本身（11）。
- frcmod 的读写（本链一律不做）。
