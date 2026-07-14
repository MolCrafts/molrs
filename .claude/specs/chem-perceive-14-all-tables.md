---
title: "chem-perceive 14/14 — 剩余力场参数表 → .rs"
status: in-progress
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-02-param-rs"
blocks: ""
---

# 把「所有参数表编译期化」做完

> Chain **chem-perceive** 14/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

收尾：把 02/09/10 之外剩余的力场参数表转为 typed Rust：
`mmff94.xml`(427 KB)、`mmff94s.xml`(427 KB)、`oplsaa.xml`(346 KB)。

`gen3d/rigid-fragments.txt` 和 `gen3d/ring-fragments.txt` 是 Open Babel 的坐标模板库，
不是 RDKit ETKDG 参数，且 molrs 从未加载它们。本期直接删除，不转换为 Rust 表。

这一期同时纠正数据边界：只有实际被运行时消费的参数表才转为 `.rs`；
从未接入当前算法的坐标库直接删除。

## Domain basis

### 力场表转换已被实测验证

对实际有消费者的力场表，typed Rust 的编译成本可接受：

| 项 | 实测值 |
|---|---|
| 15,474 行 static Rust 表（gaff+gaff2+BCCPARM） | 二进制 **+1071 KB**，编译 **0.37 s** |
| 需要替换的三个 XML 原始文本 | **1204 KB** |

这三个 XML 已经被 `include_str!` 带入二进制，转成 typed 表可以移除运行时解析。
Open Babel fragment 文件没有任何消费者，不符合这个转换前提。

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

## 统一到一个地方，一种形式（owner 定案）

> "请你将所有的 data 统一成一个形式，放在统一的地方"
> "如果是原位的，那就不要叫 generated 这种傻逼的名字！"
> "ci 不要和 AMBERHOME 有任何牵连，只在实施过程中验证一次！"

今天参数数据散落在**四种形式、三个地方**：

| 在哪 | 形式 | 行/大小 |
|---|---|---|
| `molrs/src/ff/params/generated/` ×15 | 已提交的 typed Rust | 49,977 行 / 2.9 MB |
| `molrs/src/ff/mmff/tables.rs` | 已提交的 typed Rust（**从 RDKit `Params.cpp` 移植**，BSD-3；17 张 static 表 + 17 个二分查找访问器） | **51,621 行** |
| `molrs/data/*.xml` ×3 | 原始 XML，`include_str!` + **运行期解析** | 1,204 KB |
| `molrs/data/gen3d/` | 死数据（零引用） | 已由 owner 删除 |

⇒ 归位到**唯一**的 `molrs/src/ff/params/`：

```
molrs/src/ff/params/
    mod.rs            ← 只放行类型（今天就是）
    gaff.rs  gaff2.rs  bccparm.rs  bccparm_abcg2.rs  gasparm.rs
    gaff_equiv.rs  gaff_empirical.rs
    atomtype_{gff,gff2,bcc,abcg2,amber,sybyl,gas}.rs
    oplsaa.rs         ← 新增（XML → typed Rust）
    mmff.rs           ← MMFF 的全部参数数据，合成一份：
                        (a) ff/mmff/tables.rs 的 RDKit Params.cpp 移植（51,621 行，17 张 static + 17 个访问器）
                        (b) mmff94.xml 剩下的 199 条（95 vdW + 95 原子属性 + style 骨架 + special_bonds）

molrs/src/ff/mmff/
    resolve.rs        ← 808 行的解析算法，从 params.rs 改名
    atomtype.rs  charges.rs  aromaticity.rs  hybrid.rs  topo.rs
```

### 命名（owner 定案，已重申多次）

> "我跟你说了一万遍了，不要叫 generated 这个名字！"

**`generated` / `generator` 不得出现在任何标识符里**——目录名、文件名、模块名、测试函数名、类型名，一个都不许。今天它有 **~130 处**（`tables_gate.rs` 26、`params.rs` 26、`params/mod.rs` 22、`tables_equivalence.rs` 18、`gen_param_tables.py` 15，外加一个叫 `estimate_tables_generated.rs` 的测试文件）。

理由不是洁癖：**"generated" 描述的是它们*怎么来的*，不是*是什么*。** 用来源给东西命名，等于把一个实现细节焊死在公共表面上——就像把 `ff/mmff/params.rs` 叫 params（它其实是算法）一样，名字一旦说错，读代码的人就会一直被误导。出处属于文件头的文档注释（"emitted by `scripts/gen_param_tables.py` from AmberTools' `.DAT`/`.DEF`"），不属于名字。

- **`params/` 底下的东西全是表**，所以文件名里不许再出现 `_tables` 这种后缀——等于说"表表"。
- 加一道门禁：`grep -rniE '\bgenerated?\b|\bgenerator\b' molrs/src/ff/params molrs/tests` 在**标识符位置** 0 命中（散文里说明出处允许）。这道门禁必须自己不能豁免自己（needle 用 `concat!` 拼）。
- **`ff/mmff/params.rs` → `ff/mmff/resolve.rs`。** 那 808 行**根本不是参数**，是算法（`bond_type`/`angle_type`/`torsion_type`、四级等价降级、经验规则）。叫它 `params.rs` 是 spec 02 里**我起错的名字**，在这里改正。它不进 `params/`——它属于 typifier 前端。

### mmff94s.xml 删除（前两期的后果，不是前提）

`mmff-orthogonal-02` 删掉 MMFF XML 里全部 bond/angle/stbn/torsion/oop type-def（它们是死的：没有代码读，内核读的是 frame 列）之后，`mmff94.xml` 与 `mmff94s.xml` **只差 2 行**——`<ForceField name=...>`。

真正的 94 vs 94s 差异（11 行 Oop + 42 行 Torsion）**全部住在 `tables.rs`** 里（`MMFF_OOP` vs `MMFF_OOP_S`），由 `MmffVariant` 选择。

⇒ **一张共享表，两个名字**。`MMFF94Typifier` / `MMFF94STypifier` 各自提供 ForceField 名字 + `MmffVariant`，这是它们唯一的差别。`molrs/data/mmff94s.xml` 删除。

> 诚实注脚：本链早期曾断言"mmff94s.xml 是纯冗余、该删"，被 owner 否决——**那次断言是错的**，当时 XML 确实带着 11+42 行差异并喂给 ForceField 树。它**现在**成真了，是我们删掉死数据的**后果**，不是当时的事实。

**没有 `generated/` 这个目录名。** 这些表是仓库里的一等源码，不是构建产物——"generated" 是个描述它们怎么*来的*的名字，不是描述它们*是什么*的名字。"怎么来的"写在每个文件的头部文档注释里（`scripts/gen_param_tables.py` / RDKit `Params.cpp`）就够了。

**CI 与 `$AMBERHOME` 零牵连。** 任何依赖 `$AMBERHOME` 的测试**从套件里删除**（不是 skip，是删）。逐字节复现**只在实施时本地验证一次**，结果记进本 spec。

### 前置依赖

本 spec 必须在 `mmff-typifier-split` **之后**落地。在那之前 `mmff94s.xml` 的消费者是**零**，先转它会产出一张没有读者的死表——正是 owner 已两次否决的死重量。`mmff-typifier-split` 给了它第一个真实读者（`MMFF94STypifier`），三张表才都有人读。

## Files to create or modify

- 修改 `scripts/gen_param_tables.py`（新增三个 XML 的 emitter）
- 新增 `molrs/src/ff/params/{mmff94,mmff94s,oplsaa}.rs`
- **搬迁** `molrs/src/ff/mmff/tables.rs` → `molrs/src/ff/params/mmff.rs`（连 17 个访问器；`git mv` + 改路径，不改一行逻辑）
- **重命名** `molrs/src/ff/params/generated/` → 平铺进 `molrs/src/ff/params/`
- 修改 `molrs/src/core/data.rs`（清空 `include_str!`）
- **删除** `molrs/data/*.xml`（三份）
- 删除依赖 `$AMBERHOME` 的漂移测试

## Tasks

- [ ] Extend the generator to emit `.rs` for mmff94.xml, mmff94s.xml, oplsaa.xml
- [ ] Flatten `ff/params/generated/` into `ff/params/` — the tables are first-class source, not a build artefact; no directory named `generated`
- [ ] Move `ff/mmff/tables.rs` (51,621 lines, RDKit port) to `ff/params/mmff.rs` with its 17 accessors; fix import paths, change no logic
- [ ] Delete the `include_str!` consts in `core/data.rs` and the runtime XML parsers on those paths
- [ ] Delete `molrs/data/*.xml`
- [ ] Remove every `$AMBERHOME`-dependent test from the suite (delete, do not skip) — CI must have zero coupling to AmberTools
- [ ] Validate byte-for-byte regeneration ONCE, locally, with $AMBERHOME; record the result in this spec
- [ ] Re-measure binary size and clean-build time; record the totals

## Testing strategy

- MMFF94 / OPLS 的现有 typifier + potential 测试**全绿且断言值不变**（纯数据表示变更）。
- conformer / ETKDG 测试全绿，删除死数据不影响数值。
- Drift guard：生成器逐字节复现全部 `.rs`。
- grep gate：`molrs/src` 中 `include_str!` **0 命中**；`molrs/data/` 目录不再存在。
- **实测并记录**最终的二进制体积与 clean build 时间。

## Out of scope

- 力场语义变更（本期是**纯表示层**变更：文本 → typed Rust，数值一个不动）。
- `tests-data/`（那是测试输入文件，不是参数表，继续留在外部仓库）。
