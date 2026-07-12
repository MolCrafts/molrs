---
title: "chem-perceive 3/14 — Perceive::find_bond_types — BCC 键型感知（缺失算法之一）"
status: done
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-01-layer; chem-perceive-02-param-rs"
blocks: "chem-perceive-05-atd-typifier; chem-perceive-07-charge-trait"
---

# BCC 键型感知：芳香 7/8 边界 + 离域 9

> Chain **chem-perceive** 3/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

molrs **根本没有 BCC 键型感知**。`ff/typifier/am1bcc.rs:1298 infer_bcc_bond_type` 只有 20 行：
键级 1/2/3 → 类型 1/2/3，芳香 → 类型 10。antechamber 的 `bondtype.c` 做的是真正的
共振/Kekulé 判定，产出 `{1,2,3,6,7,8,9,10,11}`。

这是 acetate 两个羧酸氧拿到**不同电荷**（-0.8351 / -0.6861，应同为 -0.8613，偏差 0.20 e）、
以及 nitromethane / pyridine / imidazole 直接报错的根因。

## Domain basis

**类型 10 是 7/8 的"未解析前身"，不是同级兄弟。** `mol2.c` 把 SYBYL 的 `"ar"` 映射为 10；
`bondtype.c::finalize()` 随后把 10 **解析**成 7（芳香单键）或 8（芳香双键），判据是相邻键是否已经是 2/8。
`-j full` 下键级从头重感知，10 根本活不到 am1bcc。

**Kekulé 选择对电荷不可见（决定性，且推翻了初版判断）。** 实测 dump `BCCPARM.DAT` 按键型分组比较：
7 vs 10 → 25/25 个 key，25 个值完全相同；8 vs 10 → 15 个公共 key，15 个相同；7 vs 8 → 15 个公共，15 个相同。
**7/8/10 是同一组数字复制了三份。** 这是刻意设计：芳香键的 BCC 增量不能依赖于感知器碰巧选了哪个
Kekulé 结构，否则电荷就是共振任意的。
⇒ 环内 Kekulé 交替错了，**在电荷层面是静默的**，不要在那里花正确性预算。

**真正要花预算的是"芳香/非芳香"边界**：只有当两个端点都芳香、**且**它们共处一个
**每个环原子都芳香的 5 元或 6 元环**时，才提升为 7/8。因此 biphenyl 的环间键、以及任何
7 元及以上芳环里的键，**保持 1/2**。

**类型 9（离域）不是共振计算，是三条局部连接规则**（`bondtype.c::finalize()`，已在 acetate 上实跑验证
两根 C–O 都是 9）：
- part2：type-2（双）键，一端是共轭原子，另一端是**端位**（connum==1）O 或 S → 9
- part4：type-1（单）键，**任一**端是端位 O/S → 9
- part5：S 且 connum==3 且带正好 2 个端位 O/S → 那些 S–O 键 → 9；
  P 且 connum==4 且带 ≥2 个端位 O/S → 那些 P–O 键 → 9

即"离域" ≡ *共轭/超价中心上连到端位硫族原子的键*。羧酸根与硝基都落在这里。

### 类型 6：已实测查清，并由 owner 定调（不再是 OPEN）

**可达性（实测，33 个探针分子跑真 AmberTools25）：** 发射的键型是 {1,2,3,6,7,8,9}，
**从不发射 10 或 11**。类型 6 **可达**——nitrate、nitrite、pyridine-N-oxide。
类型 11 **任何探针都没有触发**（BCCPARM 里 26 行全是同类型对角线、值全 0.0000，
源码里也找不到发射点）⇒ 断言不可达，**不要实现死分支**。

**类型 6 不是电荷不变的**（与 7/8/10 相反）：BCCPARM 对 `23|31` 在 type 6 是 `+0.1317`、
在 type 9 是 `-0.1500`；而且 6 个 type-6 对里有 **5 个根本没有 type-9 行**——判错就是硬报错。
所以必须实现。

**antechamber 的 part3 有两个缺陷**（`bondtype.c:897-933`）：
- **branch A**（`bondi` 是 N）：`break` **没有加括号**，邻居扫描在**第一个**非配对邻居就停 ⇒
  结果依赖 `con[]` 顺序，也就是**依赖输入文件里键的书写顺序**。
- **branch B**（`bondj` 是 N）：`bond[i].type = 6;` 在循环**外面无条件执行** ⇒ 完全不检查
  第二个硫族原子。

**实测后果：** nitrate 的三根 N–O 拿到 **6, 9, 9**——两个**拓扑完全等价**的单键 O⁻ 拿到了
不同键型 ⇒ 最终 AM1-BCC 电荷 `-0.6997 / -0.4180 / -0.4180`，三个等价氧相差 **0.28 e**。
（等价化救不了它：`-eq` 平均的是 BCC **之前**的 AM1 电荷，而损坏发生在 BCC 的键型上。）

**Owner 决定（2026-07-12，权威）：只修对称性/顺序依赖这个 bug，其余一律跟 antechamber。**
- **确定性**：键型不得依赖输入里键的顺序。置换键顺序必须得到完全相同的结果（硬性，要有测试）。
- **对称性**：同一个原子连到**拓扑等价**的末端硫族原子的键，必须拿到**同一个**键型。
  nitrate 那两个单键 O⁻ 必须一致。
- **其余照抄 antechamber**：nitrite 仍是 6/6，nitromethane 仍是 9/9，37 个 oracle 分子的键型
  一个都不许动。
- **不发明新的化学规则。** 若任何去偏方案会改动 nitrite 或 nitromethane ⇒ 停下来报 blocked。

**oracle 里没有任何 type-6 键**（只有 1,2,3,7,8,9），所以上面这个决定不会影响 ac-001。

**类型 6 与 11 是 OPEN，不要猜。** 6 的规则存在（N，connum 2–3，连到端位 O/S，且自身另带一个端位 O/S），
但 part2 会先把共轭 N=O 判成 9，所以 6 的可达性未确认。11 的 26 行全是同原子类型对角线、值全为 0.0000，
**没有找到任何发射它的代码路径**。

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

- 新增 `molrs/src/perceive/bond_type.rs`（`Perceive::find_bond_types`）
- 修改 `molrs/src/perceive/mod.rs`
- 删除 `molrs/src/ff/typifier/am1bcc.rs::infer_bcc_bond_type`（及其 20 行启发式）

## Tasks

- [x] Implement `Perceive::find_bond_types(&MolGraph) -> MolGraph` writing a BCC bond-type component
- [x] Implement aromatic 7/8 promotion with the correct boundary (see Domain basis)
- [x] Implement type-9 (delocalized) via the three local connectivity rules
- [x] Handle type 10 as the unresolved precursor (SYBYL `ar`), resolving it to 7/8
- [x] Investigate BCC bond types 6 and 11 reachability against AmberTools25; if unreachable, assert unreachable in a test rather than implementing dead branches
- [x] Establish and document the clean-room / licensing posture for reimplementing antechamber algorithms (bondtype.c) before merging

## Testing strategy

- `bcc_bond_types_match_antechamber` → 323/323 根键。
- 边界用例：biphenyl 环间键 ∉ {7,8}；acetate 两根 C–O 均为 9；nitromethane 两根 N–O 均为 9。
- **表恒等式测试**（极便宜、极锋利，且它是下面那个 fallback 的许可证）：
  ∀ BCCPARM 中存在的 (i,j)：`bcc(i,j,7) == bcc(i,j,8) == bcc(i,j,10)` 精确相等。

**验收口径的一个岔路，spec 必须显式锁定**：RED 测试 `bcc_bond_types_match_antechamber` 是
**逐字面**比较键型的，所以即使电荷可证明相同，一个 Kekulé 交替选择不同也会挂。
- **主口径**：忠实复刻 `bondtype.c::finalize()` 的判定顺序 → 323/323。
- **被许可的退路**（仅当主口径证明过于脆弱）：把键型断言放宽为把 {7,8,10} 视作一个等价类，
  **理由正是上面的表恒等式**；此时端到端电荷（07）是硬杠。**表恒等式测试未绿之前不得放宽。**

## Out of scope

- 类型 6 / 11 的**实现**：先调查可达性，不可达就写断言，不要实现死分支。
- 原子类型（05）与电荷（07）。
- `-j full` 的整体键级重感知（本期只做 BCC 键型，不重排键级）。
