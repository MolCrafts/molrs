---
title: "chem-perceive 6/14 — GAFF / GAFF2 / AMBER / SYBYL 原子类型（推翻 2026-06-19 决定）"
status: done
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-05-atd-typifier"
blocks: ""
---

# 同一引擎再挂四张表：GAFF / GAFF2 / AMBER / SYBYL

> Chain **chem-perceive** 6/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

在 05 的 `AtdTypifier` 上再挂四张表：`ATOMTYPE_GFF.DEF`(GAFF)、`ATOMTYPE_GFF2.DEF`(GAFF2)、
`ATOMTYPE_AMBER.DEF`、`ATOMTYPE_SYBYL.DEF`。引擎不动，只加表。

**本期显式推翻 2026-06-19 的决定。** `.claude/notes/notes.md` 当时记录
"GAFF = AmberTools(antechamber) 委托，不做原生 molrs GAFF typifier"，相关的
`gaff-typifier-*` spec 链在 commit `dc2f1fb` 被丢弃。推翻的理由是**情况变了**：
ATD 引擎现在已经存在且已验证，GAFF/GAFF2 原子类型只差一张 `.DEF` 表。
该推翻已获用户授权（"纳入：ATD 引擎泛化到 gaff/gaff2/amber/sybyl"）。

## Domain basis

七个 `ATOMTYPE_*.DEF` 的 ATD / WILDATOM 语法完全一致（已实地核对 BCC 与 GFF2 的
`WILDATOM` / `ATD` 行同构），所以本期是**纯加表**，引擎零改动。

行数：GFF 406、GFF2 429、AMBER 150、SYBYL 159。

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

## Tasks

- [x] Generate `.rs` tables for ATOMTYPE_GFF / GFF2 / AMBER / SYBYL (mechanically, via 02's generator)
- [x] Wire them through `AtdTypifier` — engine unchanged, tables only
- [x] Extend `scripts/gen_am1bcc_oracle.py` with `-at {gff,gff2,amber,sybyl}` oracle columns
- [x] Update `.claude/notes/notes.md` via `/mol:note`: the 2026-06-19 "GAFF = AmberTools-only" decision is REVERSED, with the reason (the ATD engine now exists and is validated)
- [x] Establish and document the clean-room / licensing posture (shared with 05)

## Testing strategy

每张表各跑一遍 37 分子回归：`antechamber -at gff` / `-at gff2` / `-at amber` / `-at sybyl`，
各 37/37。这是 **typifier 层的泛化证明**——如果一个引擎能同时喂出七套原子类型，
这个抽象就是诚实的。

## Out of scope

- 力场**参数**匹配（gaff.dat / gaff2.dat）——那是 09 / 11。
- 本期只做**原子类型**，不产出任何 bonded/vdw 参数。
