---
title: "chem-perceive 5/14 — AtdTypifier：把 ATD 规则引擎参数化抽出（一个引擎 + N 张表）"
status: done
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-02-param-rs; chem-perceive-03-bondtype"
blocks: "chem-perceive-06-gaff-types; chem-perceive-07-charge-trait"
---

# ATD 规则引擎参数化

> Chain **chem-perceive** 5/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

把 ATD / WILDATOM 规则引擎从 `ff/typifier/am1bcc.rs` 里**抽出来参数化**成 `AtdTypifier`：
一个引擎 + N 张生成的表。本期先让 BCC / ABCG2 / GAS 三套原子类型跑绿；GAFF/GAFF2/AMBER/SYBYL 在 06。

**硬依赖 03**：ATD 规则是按 `sb`/`db`/`ab`/`DL` 计数匹配的，而这些计数由**键型**导出。
pyridine / imidazole 的芳香 N 现在被判成 `"25"`（应为 `"24"`），进而使修正表的查找键变成
`17|25|10` 而查不到——那个 *"missing BCC correction"* 报错**本质是原子类型 bug 以查表失败的形式冒出来**，
不是修正表缺行。

## Domain basis

七个 `ATOMTYPE_*.DEF` 共用同一套 ATD / WILDATOM 语法，而 molrs 现有的规则引擎**已经能解析它**
并且已经能把 35/37 个分子的 BCC 原子类型判对——剩下 2 个的错误是从键型级联下来的（见 03）。
因此"一个引擎 + N 张表"不是设想，而是既有能力的直接参数化。

这与 07 的 `ChargeModel` 是对称的：**typifier 层的泛化证明**。

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

- 新增 `molrs/src/ff/typifier/atd/{mod,rules,facts,pattern}.rs`（从 `am1bcc.rs` 抽出）
- 修改 `molrs/src/ff/typifier/am1bcc.rs`（瘦身：只留 BCC 修正逻辑）
- 修改 `molrs/src/ff/typifier/mod.rs`（导出 `AtdTypifier`）

## Tasks

- [x] Extract the ATD/WILDATOM rule engine from `ff/typifier/am1bcc.rs` into a parameterized `AtdTypifier`
- [x] Drive it from the generated `.rs` tables (02) — one engine, N tables
- [x] Wire BCC / ABCG2 / GAS atom typing through it
- [x] Assert the per-typify() table re-parse (am1bcc.rs:164) is gone (moot after 02, but assert it)
- [x] Remove the `AM1BCCTypifier::new()` empty-table footgun (am1bcc.rs:1190) — a constructible but non-functional object, forbidden by molrs's no-fallback-values rule
- [x] Establish and document the clean-room / licensing posture for reimplementing antechamber's atomtype.c before merging

## Testing strategy

- `bcc_atom_types_match_antechamber` → 37/37（当前 2/37 挂：pyridine、imidazole）。
- ABCG2 原子类型 37/37 vs `antechamber -at abcg2`。
- GAS 原子类型 37/37 vs `antechamber -at gas`。
- grep gate：typify 路径上没有任何运行期表解析。

## Out of scope

- GAFF / GAFF2 / AMBER / SYBYL 表（06）。
- 电荷模型（07）。
