---
title: "chem-perceive 4/14 — Perceive::find_equivalence_classes — 电荷等价化（缺失算法之二）"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-01-layer"
blocks: "chem-perceive-07-charge-trait"
---

# 电荷等价化：antechamber 的 path-score 算法（不是自同构轨道）

> Chain **chem-perceive** 4/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

molrs **完全没有电荷等价化**。antechamber 默认 `-eq 1`，在 BCC **之前**把拓扑等价原子的
AM1 Mulliken 电荷平均掉。实测甲醇三个甲基 H：sqm 原始 `(0.053, 0.098, 0.053)` →
antechamber `(0.068, 0.068, 0.068)`（正是均值）。

而 `Atomiverse/src/cpu/semiempirical/am1_bcc_charge_assigner.cpp:82` 喂给 molrs 的正是
**未平均的原始 Mulliken**。37 个分子里 **20 个（54%）** 受影响，最大偏差 **0.036 e**。
后果不只是与 antechamber 对不上——是产出**构象依赖、对称性破缺**的力场电荷
（同一分子换个 rotamer，甲基 H 电荷就变）。

等价化归属 **molrs**（纯拓扑操作）；Atomiverse 继续喂原始 Mulliken 是对的，**本期不需要改 Atomiverse**。

## Domain basis

**算法 = antechamber `equatom.c` 的 path-score，即 Antechamber 论文 §2.1 式 (I)：**

> `Score = Σ_j [ (j+1)·0.11 + Z_j·0.08 ]`  沿一条路径求和，j 为路径中的位置下标，Z_j 为该位原子序数。

对每个原子，枚举它到**每一个端位原子**的**所有简单路径**，各算一个 score，升序排序。
两个原子等价 ⟺ 路径**条数相同** **且** 排序后的 score 数组**逐元素精确相等**（f64 `==`，**没有容差**）。
出处：`equatom.c` L205-206、L285-300。

**🚨 为什么自同构轨道（WL / Morgan / `graph_hash`）是错的——这条推翻了初版设计：**
把 score 展开：

> `Score = 0.11·L(L+1)/2 + 0.08·(ΣZ along the path)`

它只通过 **(路径长度 L, 路径上原子序数之和 ΣZ)** 依赖于路径——**对路径上原子的顺序完全盲视**
（C–N–O 与 C–O–N 得分相同）。因此：
- 自同构轨道 ⊆ path-score 类，且**是真子集**：antechamber **从不拆开**一个轨道，
  但它会**合并**处于不同轨道的原子。
- 用轨道做平均 ⇒ 得到一个**比 antechamber 更细**的划分 ⇒ **必然偏离 oracle**。

⇒ 为了对齐 antechamber，**必须实现 path-score**。`core/system/graph_hash.rs` 只可作为可选的
快速预筛（轨道内必然同类），**不得**充当分类引擎。

**`-eq` 语义（初版理解是反的）**，来自本地 AmberTools25 `antechamber -h`：
- `0` = 关闭
- `1` = "by atomic paths"，**是 `-c bcc` / `-c abcg2` / `-c resp` 的默认值；其他所有电荷方法默认是 0**
- `2` = 路径 + E/Z 结构信息（每位系数 1.01 反式 / 0.99 顺式），因此 **`-eq 2` 比 `-eq 1` 更细，不是更粗**
- `-pl` 限制路径长度，默认 −1 = 不限

这张默认值表正是为什么 `needs_equivalencing` 必须是 **per-model 声明**（07），而不是一个全局阶段。

平均 = 类内**算术平均**后广播回成员（`charge.c` L485-512），在 BCC **之前**执行，且精确保总电荷。

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

- 新增 `molrs/src/perceive/equivalence.rs`（path-score + 类划分 + 类内均值）
- 修改 `molrs/src/perceive/mod.rs`

## Tasks

- [ ] Implement `Perceive::find_equivalence_classes(&MolGraph) -> MolGraph` using antechamber's PATH-SCORE algorithm
- [ ] Enumerate all simple paths from each atom to every terminal atom; score, sort, compare exactly
- [ ] Support the `-eq` levels: 0 (off), 1 (paths, default for bcc/abcg2), 2 (paths + E/Z, strictly FINER)
- [ ] Add class-mean averaging as a separate, explicit step consumed by ChargeModel (07)
- [ ] Establish and document the clean-room / licensing posture for reimplementing antechamber's equatom.c before merging

## Testing strategy

- 用 oracle 的 `am1_charges_raw`（原始 sqm Mulliken）做输入，平均后必须复现 antechamber 的
  `am1_charges`（已等价化），37/37 在 1e-4 内；其中 20 个分子实际发生了变化。
- 甲醇：三个甲基 H 得到三个**完全相同**的值。
- **必须有一个专门的测试证明比较是精确相等而非容差**：构造一个用容差比较会误合并的用例，
  断言它不被合并。
- 平均**精确保总电荷**：`Σq` 在等价化前后逐位相同。

## Out of scope

- `-eq 2` 的 E/Z 判定可以留到后续（本期先把 0/1 做对，2 至少要在 API 上留位）。
- 把 `graph_hash` 当作等价类引擎（见 Domain basis，它是**错**的）。
