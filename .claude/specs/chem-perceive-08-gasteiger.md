---
title: "chem-perceive 8/14 — GasteigerModel — antechamber `-c gas` 对齐（无 QM 的泛化证明）"
status: done
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-02-param-rs; chem-perceive-05-atd-typifier; chem-perceive-07-charge-trait"
blocks: "chem-perceive-13-python-bind"
---

# Gasteiger / PEOE 到 antechamber 对齐

> Chain **chem-perceive** 8/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

在 `ChargeModel` trait 上实现 Gasteiger/PEOE，对齐 `antechamber -c gas`。
这是**架构泛化的关键一环**：它**完全不需要 QM 输入**，纯拓扑迭代。
如果同一个 trait 能同时托起它和 BCC，就证明这个抽象没有偷偷假设 QM 基电荷。

顺带把现有的 `core/chem/gasteiger.rs`（655 行，**不是** antechamber 对齐的——它没有用
GASPARM.DAT + ATOMTYPE_GAS.DEF）折叠进来。

实测 antechamber `-c gas` 在甲醇上的三个甲基 H 是**完全相同**的（0.052691 ×3）——
纯拓扑模型天然对称，所以它正确地跑在 `needs_equivalencing = false` 下。

## Domain basis

**电负性多项式**：`χ_i = a + b·q_i + c·q_i²`。

**`d` 列是 χ⁺，即归一化分母——不是四次项系数。** 把它当成 `+d·q³` 编码是**灾难性错误**。
已在 `GASPARM.DAT` 上实地验证：重原子严格满足 `d == a+b+c`
（c1: 10.39+9.45+0.73 = 20.57 = d ✓；c3: 7.98+9.18+1.88 = 19.04 = d ✓）。
**氢是唯一例外**：`d_H = 20.02`，而 `a+b+c = 7.17+6.24−0.56 = 12.85`。
这正是那个有文献记载的氢特例——H⁺ 是裸质子，它的多项式 χ⁺ 无意义，因此代入一个固定的
χ⁺(H) = 20.02 eV。**这个值在随附的数据文件里就能确认，不依赖付费墙论文。**

**电荷转移**（`charge.c` L864-900）：对每条键，

> `Δq = (χ_high − χ_low) / χ⁺_donor × (1/2)^n`

分母取自**电负性较低的（给电子的）**那个原子——即将带正电的那个；`n = iteration + 1`，从 1 开始。
电荷流向：**+ 给低 χ 原子，− 给高 χ 原子**。逐键反对称 ⇒ **PEOE 精确保总电荷**。

**它是收敛循环，不是经典的固定 6 次迭代。** antechamber 的常量：
`CONVERG 0.00001`、`GASMAXITER 500`、`DAMPFACTOR 0.5`，循环条件 `while rmsd > 1e-5 && iter < 500`。
**不要把迭代数写死成 6。**

`formal_charge` 列是**种子电荷 q⁰**（不是查表键）；多数行是 0.00，带电类型带种子（如 `cg` = 0.04）。

**OPEN（已列为 Task）**：antechamber 的 PEOE 循环用上一轮的迭代值构造 χ、累加进第二个缓冲区，
而缓冲区滚动发生在 `rmscal()` 内部——**Jacobi 还是 Gauss-Seidel 未经验证**。编码前必须确认，
搞混会改变收敛轨迹。

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

- 新增 `molrs/src/ff/charge/gasteiger.rs`
- 删除 / 折叠 `molrs/src/perceive/gasteiger.rs`（01 之后的路径）
- 修改 `molrs-python`：`compute_gasteiger_charges` 改指向新模型（13 期做绑定）

## Tasks

- [x] Verify antechamber `rmscal()` buffer discipline (Jacobi vs Gauss-Seidel) BEFORE encoding the PEOE update loop
- [x] Implement `GasteigerModel` on the ChargeModel trait with `needs_equivalencing = false`
- [x] Use the generated GASPARM table: chi = a + b*q + c*q^2; `d` is the chi-plus DIVISOR; `formal_charge` is the seed q0
- [x] Implement the H special case: chi_plus(H) = 20.02 (NOT a+b+c = 12.85)
- [x] Implement the damped convergence loop (CONVERG 1e-5, GASMAXITER 500, DAMPFACTOR 0.5) — do NOT hardcode 6 iterations
- [x] Fold / delete `core/chem/gasteiger.rs` (or `perceive/gasteiger.rs` after 01)

## Testing strategy

- 37/37 vs `antechamber -c gas` @1e-4。甲醇参考值：
  `0.031933, -0.399641, 0.052691, 0.052691, 0.052691, 0.209634`。
- `Σq` 精确守恒（PEOE 逐键反对称）。
- **H 的 χ⁺ = 20.02 必须被断言**（用 `a+b+c = 12.85` 是错的）。
- 收敛循环测试：构造一个需要 > 6 次迭代的分子，证明没有把迭代数写死成 6。

## Out of scope

- `-c gas` 之外的 Gasteiger 变体。
- 把 Gasteiger 用于力场参数估计。
