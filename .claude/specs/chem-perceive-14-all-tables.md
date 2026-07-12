---
title: "chem-perceive 14/14 — 剩余参数表 → .rs：mmff94 / mmff94s / oplsaa / gen3d fragments"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-02-param-rs"
blocks: ""
---

# 把「所有参数表编译期化」做完

> Chain **chem-perceive** 14/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

收尾："所有的参数表应该写成 .rs"——把 02/09/10 之外**剩下的全部**也转掉：
`mmff94.xml`(427 KB)、`mmff94s.xml`(427 KB)、`oplsaa.xml`(346 KB)、
`gen3d/rigid-fragments.txt`(1891 KB)、`gen3d/ring-fragments.txt`(740 KB)。

这一期存在的理由本身就是一次纠错：早前把它们排除的理由是"转成 `.rs` 会让编译时间爆炸"，
而**实测推翻了这个说法**（见 Domain basis）。既然理由不成立，就不该有豁免。

## Domain basis

### 排除理由已被实测推翻

早前的说法是"6 万/3 万行的 fragment 库转 `.rs` 会让编译时间爆炸"。实测：

| 项 | 实测值 |
|---|---|
| 15,474 行 static Rust 表（gaff+gaff2+BCCPARM） | 二进制 **+1071 KB**，编译 **0.37 s** |
| molrs 今天已经 `include_str!` 进去的原始文本总量 | **3974 KB** |
| 其中 `gen3d/rigid-fragments.txt` 单个 | **1891 KB** |

也就是说：**这些数据早就在二进制里了**，只不过是以"未解析的原始文本"形式躺着，
每次用还要在运行期解析一遍。转成 typed 表之后，文本副本消失、解析消失，
净增量远小于直觉。fragment 库是**坐标数据**，转成扁平 `f64` static slice 甚至可能比
原始文本**更小**（省掉了数字的十进制文本表示）。

⇒ 没有豁免的理由。做完它。

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

- 修改 `scripts/gen_param_tables.py`
- 新增 `molrs/src/ff/params/generated/{mmff94,mmff94s,oplsaa}.rs`
- 新增 `molrs/src/conformer/data/generated/{rigid_fragments,ring_fragments}.rs`
- 修改 `molrs/src/core/data.rs`（清空）
- **删除** `molrs/data/`

## Tasks

- [ ] Extend the generator to emit `.rs` for mmff94.xml, mmff94s.xml, oplsaa.xml
- [ ] Extend the generator to emit `.rs` for gen3d/rigid-fragments.txt and ring-fragments.txt (coordinate libraries — emit as flat f64 static slices, not strings)
- [ ] Delete the corresponding `include_str!` consts in `core/data.rs` and the runtime XML/text parsers on those paths
- [ ] Delete `molrs/data/` entirely once every table is generated
- [ ] Re-measure binary size and clean-build time; record the totals

## Testing strategy

- MMFF94 / OPLS 的现有 typifier + potential 测试**全绿且断言值不变**（纯数据表示变更）。
- conformer / ETKDG 的 fragment 相关测试全绿。
- Drift guard：生成器逐字节复现全部 `.rs`。
- grep gate：`molrs/src` 中 `include_str!` **0 命中**；`molrs/data/` 目录不再存在。
- **实测并记录**最终的二进制体积与 clean build 时间。

## Out of scope

- 力场语义变更（本期是**纯表示层**变更：文本 → typed Rust，数值一个不动）。
- `tests-data/`（那是测试输入文件，不是参数表，继续留在外部仓库）。
