---
title: "chem-perceive 2/14 — 参数表 → committed .rs（molrs 不再解析文本）"
status: done
created: 2026-07-12
chain: chem-perceive
depends_on: ""
blocks: "chem-perceive-03-bondtype; chem-perceive-05-atd-typifier; chem-perceive-08-gasteiger; chem-perceive-09-gaff-params"
---

# 参数表编译期化：Python 生成器 → 提交的 Rust 数据结构

> Chain **chem-perceive** 2/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

用 Python 脚本把 antechamber 的参数表**直接翻译成 Rust 语法和数据结构**（typed const /
static slice / phf-style map），提交进仓库、直接 `use`。molrs **不再在运行期解析任何文本**——
解析错误从 runtime error 变成 **compile error**，且参数表可以直接 grep、直接断点、直接 review。

这一期同时消灭 `BCCAtomTypeRules::parse_str`、`BCCCorrectionTable::parse_str` 和
`core/data.rs:29-50` 的 `include_str!` 常量。

## Domain basis

七个 `ATOMTYPE_*.DEF` 使用**完全相同**的 ATD / WILDATOM 语法——正是 molrs 现有
`BCCAtomTypeRules` 已经能解析的那套。已实地核对 `ATOMTYPE_BCC.DEF` 与 `ATOMTYPE_GFF2.DEF`
的 `WILDATOM` / `ATD` 行同构。因此**一个生成器、七份输出**。

`GASPARM.DAT` 的列是 `a b c d formal_charge`：`a/b/c` 是电负性多项式系数，
`d` 是 χ⁺（归一化分母，**不是四次项系数**），`formal_charge` 是种子电荷 q⁰（详见 08）。
生成的 Rust 结构必须把这三者的语义分开命名，不能糊成一个 `[f64; 5]`。

### 体积与编译时间：已实测，不是估计

把 **gaff.dat + gaff2.dat + BCCPARM 共 15,474 行**编成 typed Rust static 表，
在一个真实消费这些表的 release 二进制里实测：

| 项 | 实测值 |
|---|---|
| 最终 stripped 二进制增量 | **+1071 KB** |
| 编译时间（15,474 行 static） | **0.37 s** |
| molrs **今天已经**用 `include_str!` 编进去的原始文本 | **3974 KB** |

对照：单是 `gen3d/rigid-fragments.txt`（1891 KB）就比整个 GAFF+GAFF2+BCC 的 typed 表还大。
⇒ **体积和编译时间都不构成理由。** 直接一步写成 `.rs`，不留 `.DAT` 中间态。

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

- 新增 `scripts/gen_param_tables.py`（从 `$AMBERHOME` 读源表）
- 新增 `molrs/src/ff/params/generated/{bccparm,bccparm_abcg2,gasparm,atomtype_*}.rs`（提交的生成物 = 仓库内唯一真值）
- **删除** `molrs/data/antechamber/`（不再保留 `.DAT`/`.DEF` 中间态）
- 修改 `molrs/src/core/data.rs`（删除 antechamber `include_str!` 常量）
- 修改 `molrs/src/ff/typifier/am1bcc.rs`（删除两个 `parse_str`）

## Tasks

- [x] Write `scripts/gen_param_tables.py`: reads the source tables from `$AMBERHOME` and emits committed `.rs` (typed consts / static slices)
- [x] Generate `.rs` for BCCPARM.DAT (405), BCCPARM_ABCG2.DAT (475), GASPARM.DAT (40), and all 7 ATOMTYPE_*.DEF (BCC/ABCG2/GAS/GFF/GFF2/AMBER/SYBYL)
- [x] Delete `BCCAtomTypeRules::parse_str`, `BCCCorrectionTable::parse_str`, and the antechamber `include_str!` consts in `core/data.rs:29-50`
- [x] DELETE the raw `.DAT`/`.DEF` files from `molrs/data/antechamber/` — the committed `.rs` is the single in-repo source of truth
- [x] Add a drift-guard test: with `$AMBERHOME` set, the generator byte-reproduces every committed `.rs`; skip cleanly when it is not set

## Testing strategy

- **Drift guard**：CI（或本地装了 AmberTools 时）跑生成器，输出必须与已提交的 `.rs`
  **逐字节相同**（防止手改表 / 上游漂移）。`$AMBERHOME` 未设置时**干净跳过**——
  因为提交的 `.rs` 才是仓库内唯一真值，AmberTools 只是上游。
- 已核实 molrs 现有的 4 个 antechamber 文件与 AmberTools25 **逐字节相同**，
  所以生成的 `.rs` 是可证明忠实的。
- grep gate：`core/data.rs` 中不再有 antechamber 表的 `include_str!`；BCC 路径上 `parse_str` 归零；
  `molrs/data/antechamber/` 目录已删除。

## Out of scope

`mmff94.xml` / `mmff94s.xml` / `oplsaa.xml` / `gen3d` 的两个 fragment 库 —— **不是不做，是挪到 14 期**。
它们属于其他子系统（ForceField XML reader / conformer 流水线），blast radius 与本链无关，
单独一期更干净。**注意：早先"转成 .rs 会让编译时间爆炸"的说法已被实测推翻**（见 Domain basis），
所以它们没有被豁免的理由，只是排期靠后。
`gaff_equiv.json` / `gaff_empirical.json` 在 10 期；`gaff.dat` / `gaff2.dat` 在 9 期。
