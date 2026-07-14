---
title: "MMFF 去重复 — 删除 bespoke 实现与死数据，per-instance style 成为一等概念"
slug: mmff-orthogonal-02-delete-bespoke
status: in-progress
created: 2026-07-14
chain: mmff-orthogonal 2/2
depends_on: "mmff-orthogonal-01-fix-generic"
blocks: "chem-perceive-14-all-tables"
---

# MMFF 去重复 — 删除 bespoke 实现与死数据

## Summary

`mmff-orthogonal-01-fix-generic` 证明 generic 路径在 11/11 fixture 上与 RDKit 一致之后，本 spec 删除重复实现与死数据：bespoke 的 `MmffForceField` 能量装配层、函数式入口 `molrs.build_mmff_potentials`、便捷方法 `MMFF94Typifier::build` / `MMFF94STypifier::build`（连同 `.pyi` / `__init__.py` / 文档 / 示例里的每一个门）、`typifier/mmff/classify.rs` 里三个**实现错误**的分类器，以及 4,065 行**没有任何代码读取**的 XML type-def。

同时把"参数逐实例、不来自 type-row"从"五个文件偷偷忽略 `tp`"升级成**有名字的一等概念**（`ParamSource::PerInstance`），并用一条 grep 级不变量把它钉成测试。

终态是 owner 的裁决：`MMFF94Typifier().typify(mol)` 给标签与电荷，剩下的走标准 ForceField 路线——**MMFF 不再是特例**。

## HARD ORDERING（本 spec 的第一约束）

**绝不在替代实现被证明之前删除唯一正确的实现。**

- 本 spec **只在 01 绿之后开始**。Task 1 就是**在任何删除动作之前**重新断言 11/11 RDKit parity + 逐项分解——不是走过场：删除会改动 `frame_builder` 的标签来源、conformer 的调用点、XML 的行数，任何一处回归都必须在删除**之前**就有一条已知绿的基线。
- 01 + 02 **必须都在 `chem-perceive-14-all-tables` 之前落地**。后者会把这 4,065 行死 XML 忠实编译成 committed Rust 表；一旦顺序颠倒，死数据就从"可以删的 XML"变成"有测试保护的 Rust 事实"，并且它冻结的 reference dump 会把死行写死成 conversion contract。

## Domain basis

物理与常量与 01 完全相同（Halgren 1996/1999；RDKit oracle：Tosco et al. 2014）。本 spec **不改任何数值**，域断言就是 01 的断言在删除前后**逐位重放**。

### 为什么 MMFF 的参数天生不是 type-tuple 表

MMFF 的键 / 角 / 二面角 / 面外参数依赖芳香性、环大小、四级等价降级，以及在查表失败时**由共价半径现场发明数值**的经验规则。这不是 `(type_i, type_j, …) → params` 能表达的。typifier 已经正确地逐实例解析它们并烘焙进 Frame 列，kernel 也正确地读那些列——**这不是 bug**。

bug 是：MMFF **把自己注册成了一个用 type-row 的 style**，于是每次构造 ForceField 都要从 XML 解析 4,160 行 type-def（438 KB，5.8 ms），而其中：

```
bond/mmff_bond         493 行   <- 没有任何代码读
angle/mmff_angle      2342 行   <- 没有任何代码读
angle/mmff_stbn        282 行   <- 没有任何代码读
improper/mmff_oop      117 行   <- 没有任何代码读
dihedral/mmff_torsion  926 行   <- 没有任何代码读
pair/mmff_vdw           95 行   <- 真的在读（vdW 确实按原子类型查表）→ 保留
```

4,065 行的唯一存在理由，是满足 `potential/mod.rs:265-270` 的 `type_params.is_empty()` guard。

### 可 grep 的不变量

```
grep -l '_tp: &\[(&str, &Params)\]' molrs/src/ff/potential/*/*.rs
```

今天恰好返回五个 MMFF 文件。molrs 里**其它每一个** kernel family（harmonic bond/angle、opls/periodic dihedral、lj/cut）都真的从 `tp` 解析。

> **"注册了却忽略 `tp` 的 kernel 构造器不是一个 Style。"**

这句话就是本 spec 要立的不变量，并且它必须**变成一个测试**——否则下一个 150 kcal/mol 的洞会以同样的方式长出来。

### 第二个（错误的）分类器

`typifier/mmff/classify.rs` 重新实现了 `ff/mmff/energy/params.rs` 已经正确实现的上下文规则，并且**做错了**：

- `classify.rs:10-12` `typify_bond`：芳香键 → type **1**。RDKit（`ff/mmff/charges.rs:32-38`）里芳香键是 `AROMATIC`、永远不是 `SINGLE`，所以是 type **0**。**反了**。
- `classify.rs:27` `typify_angle(bt_ij, bt_jk)`：**签名本身表达不了规则**——RDKit 的 `angle_type` 需要拓扑，因为 3-/4-元环成员资格会把角类型提升到 3–8。
- `classify.rs:39` `typify_dihedral`：同样的病；没有 4-/5-元环扭转类型。

三个都是 **public API**（`typifier/mmff/mod.rs:127-139`），而 `tests/ff/typifier/mmff_variant.rs:382-391` 只断言"两扇前门彼此一致"，**等于把错值锁死**。

## Design

### 删除清单（owner 裁决）

> "删除 molrs.build_mmff_potentials 这种函数式的，只保留 MMFF94Typifier.typify()"

终态：

```python
typed = molrs.MMFF94Typifier().typify(mol)     # 标签 + 电荷 —— Typifier 的契约，仅此而已
frame = typed.to_frame()
frame["pairs"] = molrs.intramolecular_pairs(frame)
pots  = molrs.MMFF94Typifier().forcefield().to_potentials(frame)   # 标准 ForceField 路线
```

删除：

1. `MmffForceField` 及能量装配层：`ff/mmff/energy/{mod,bond,angle,stretchbend,torsion,oop,nonbonded,geom}.rs`
2. `MMFF94Typifier::build` / `MMFF94STypifier::build`（含 `MmffEngine::build`）
3. `molrs.build_mmff_potentials` + `build_mmff_potentials_py` + Python `build` 方法 + `.pyi` / `__init__.py` / `site-src` / `README` / `examples` 里的每一处 —— **否则 Rust 清理完了，坏掉的门还在 Python 那边开着**
4. `typifier/mmff/classify.rs` **整个文件**（三个错分类器 + 两个 `resolve_*_label` —— 后者是为已删除的 type-row 查表服务的第二套等价降级循环）
5. `forcefield/xml.rs` 的五个 `parse_mmff_*`（`:345-479`）及 dispatch 分支，以及 XML 里对应的 4,065 行

**保留**：

- `ff/mmff/{atomtype,charges,aromaticity,hybrid,topo,tables}.rs` —— typifier 的前端
- **`ff/mmff/energy/params.rs`（826 行）** —— 尽管住在 `energy/` 下，它是 MMFF 的**参数解析器**（`bond_type` / `angle_type` / `torsion_type`、四级等价降级、经验规则），`frame_builder.rs:26` 以 `eparams` 导入它，即 **typifier 依赖它**。它是那条正确的、RDKit-faithful 的解析层，必须活下来。
  → 本 spec 把它**搬出 `energy/`**：→ `molrs/src/ff/mmff/params.rs`（它不是能量文件）。并显式标注交接：`chem-perceive-14-all-tables` 计划把 `ff/mmff/tables.rs` 搬到 `ff/params/mmff.rs`，**解析器应当随表一起搬**。
- `pair/mmff_vdw` 的 95 行 type-row —— vdW **确实**按原子类型查表，是货真价实的 type-row style

### per-instance style 成为一等概念

```rust
pub enum ParamSource {
    TypeRows,     // 参数来自 ForceField 的 type-def 行（tp）
    PerInstance,  // 参数由 typifier 逐实例烘焙进 Frame 列；tp 恒为空
}
```

- 注册接口带上 `ParamSource`（旧 `register` 保留为 `TypeRows` 的薄包装，避免改动其它 20+ 个 kernel）。
- `Style::to_potential` 的空 type-params guard 改为查 `ParamSource`：`TypeRows` 且无 type-def → 报错（今天的行为）；`PerInstance` → 允许零 type-def（**今天必须靠 4,065 行假数据骗过去的那个 guard**）。
- 六个 MMFF style 注册为 `PerInstance`；`pair/mmff_vdw` 保持 `TypeRows`。

### 不变量测试（核心交付物之一）

新增 `molrs/tests/ff/potential/param_source_gate.rs`（源码级静态检查，仓库已有先例）：

1. 扫描 `potential/*/*.rs`，收集所有 `pub fn *_ctor`，判定第二参数是 `_tp`（忽略）还是 `tp`（使用）。
2. 扫描 `registry.rs`，解析每条注册语句 → `(category, style, ctor, ParamSource)`。
3. 断言**双向**等价：`ctor 忽略 tp` ⟺ `注册为 PerInstance`。

### 标签的来源改为唯一正确的解析器

`classify.rs` 删除后，`frame_builder.rs` 的 bond / angle / dihedral 标签改由 `eparams` 产生——即**那份正确的、带拓扑与环规则的实现**。标签此后只是可读性 / 溯源信息（per-instance kernel 不读标签），但它至少不再是**错的**。

### conformer 与跨 crate 的一次性落地

- `conformer/etkdg/mod.rs:338-354` 的 `mmff_cleanup` 是 bespoke 的生产调用者，改走 generic 路线。
  ⚠️ **性能**：`MMFF94Typifier::new()` 每次都会解析内嵌 XML，必须**提到每次 `generate` 之外构造一次**，不能放进 per-conformer 循环。
- `molrs-python` 与 `molrs` 在**同一个 git 仓库**。删除一个 Rust 公开 API 的同时必须删掉它的 binder，否则树不编译——所以 02 是**一次跨 crate 落地**。

### chem-perceive-14 的冻结 fixture：不在这里碰，但必须重新抓

`tests/ff/fixtures/tables/{mmff94,mmff94s}.reference.txt` 是从 XML 冻结的 conversion contract，`tables_gate.rs` 还 pin 了 XML 的 SHA-256。01 加了 `<ElectrostaticParams>`、02 删掉 4,065 行——两者都让它们过期。

**修正（2026-07-14）**：初稿要求 02 重新生成它们。这是错的排序——那两个文件当前是 **parked**（属于 chem-perceive-14），把它们 restore 回来只会把一堆与 02 无关的红（`generated/` 目录名、表平铺、`$AMBERHOME` 门）拖进这一期。

正确做法：**02 不碰它们**。02 落地后死行已经不在 XML 里了，chem-perceive-14 的 tester 到时候从**最终的 XML** 重新抓 dump——它本来就该在转换发生的那一期抓。那份 parked 的 dump 是**抓早了**的产物，必须**重新生成而非恢复**。

⚠️ 交接给 chem-perceive-14 的硬约束：**不得复用 parked 的 `.reference.txt`**。它是在 4,065 行死数据还在、`<ElectrostaticParams>` 还没有的 XML 上抓的；直接用它，就等于把死行当成 conversion contract 编译成受测试保护的 Rust 表——**正是这整条链存在的理由**。

## Files to create or modify

**架构 / kernel**：`potential/registry.rs`（`ParamSource`）、`potential/mod.rs`（guard）、五个 MMFF kernel 文件的 rustdoc

**删除 / 搬迁**：`ff/mmff/energy/{mod,bond,angle,stretchbend,torsion,oop,nonbonded,geom}.rs`（删）、`ff/mmff/energy/params.rs` → `ff/mmff/params.rs`（搬）、`ff/mmff/mod.rs`、`typifier/mmff/classify.rs`（整删）、`typifier/mmff/{mod,engine,frame_builder,tests}.rs`、`forcefield/xml.rs`、`conformer/etkdg/mod.rs`、`examples/typify_molecule.rs`

**数据**：`molrs/data/mmff94{,s}.xml`、`data/mmff94{,s}.xml`（各删 4,065 行）、`scripts/mmff_to_xml.py`

**测试**：`tests/ff/potential/param_source_gate.rs`(new)、`tests/ff/potential/mod.rs`、`tests/ff/mmff/energy.rs`、`tests/ff/typifier/mmff.rs`、`tests/ff/typifier/mmff_variant.rs`、`tests/ff/potential/mmff.rs`、`tests/embed/etkdg.rs`、`tests/ff/fixtures/tables/mmff94{,s}.reference.txt`（重生成）、`tests/ff/tables_gate.rs`（SHA pin）

**Python 与文档**：`molrs-python/src/{ff/mod.rs,lib.rs}`、`python/molrs/{__init__.py,molrs.pyi}`、`tests/test_mmff{,94s}_typifier.py`、`examples/*.py`、`README.md`、`site-src/*`、`docs/interop.md`、`CHANGELOG.md`（BREAKING）

## Tasks

- [ ] Verify 11/11 RDKit parity and the frozen per-style breakdown are green BEFORE any deletion, and record the baseline
- [ ] Write failing param-source gate test: every ctor that ignores `tp` must be registered `PerInstance`, and vice versa (both directions)
- [ ] Implement `ParamSource` in `registry.rs`, make the empty-type-params guard consult it, register the six MMFF per-instance styles
- [ ] Delete the five dead XML readers in `forcefield/xml.rs` and the 4,065 dead rows from both XMLs (+ root copies + `scripts/mmff_to_xml.py`)
- [ ] Repoint `mmff_cleanup` in `conformer/etkdg/mod.rs` onto the typifier -> ForceField route, hoisting the typifier out of the per-conformer loop
- [ ] Delete `MmffForceField` and `ff/mmff/energy/*`; move `energy/params.rs` to `ff/mmff/params.rs`
- [ ] Delete `typifier/mmff/classify.rs` and the four front-door methods; derive labels in `frame_builder.rs` from the `eparams` resolver
- [ ] Split `annotate_mmff` (`typifier/mmff/frame_builder.rs`, now **221 lines**) into the six private helpers its own numbered comments already imply (`// 1. Atoms` … `// 6. Out-of-plane`) — inherited debt that **two consecutive MMFF specs have now grown**, and this spec rewrites its label source anyway, so it is the natural place. House limit is 50-80 lines/fn.
- [ ] Delete `build_mmff_potentials` and the Python `build` door across molrs-python (src, pyi, __init__, tests, examples, README, site-src); rewrite `docs/interop.md`
- [ ] Add the BREAKING CHANGELOG entry naming molpack; rewrite docs/interop.md onto the typify -> to_potentials route (its MMFF snippet is a compile-checked doctest)
- [ ] Run full check + test suite (Rust gates; then rebuild the maturin wheel and run pytest — the wheel MUST be rebuilt or pytest lies)

## Testing strategy

**删除前的门（Task 1，非仪式）**：generic 11/11 parity + 逐项分解全绿。任何一项红 → **停止**，回到 01。

**Happy path（删除后逐位重放）**：同一套 11 fixture、同一套断言，anchor 换成 RDKit `.energy.json` + 01 冻结的 `.breakdown.json`。**数值一个都不许动**。

**Edge cases**
- **不变量 gate**：故意加一个忽略 `tp` 却注册成 `TypeRows` 的 kernel，gate 必须红（手工验证一次，不提交）。
- 零 type-row 的 bonded style 能构造；**`TypeRows` style 无 type-def 仍必须报错**（`ParamSource` 的放宽不能顺手把旧守卫废掉）。
- XML 行数：`<Bond `/`<Angle `/`<StretchBend `/`<Torsion `/`<Oop ` 计数为 **0**；`<VdW ` 仍为 **95**；`<ElectrostaticParams` 为 **1**。
- 死符号清零：`MmffForceField` / `build_mmff_potentials` / MMFF 前门的 `typify_bond|angle|dihedral` 在 src / python / site-src / examples / docs 下 grep 命中 **0**（CHANGELOG 除外）。

**基线**：molrs 0 failed；passed 与 1914 的差额必须**逐条说明来源**（删掉的 bespoke/classifier 单测数、新增的 gate 测试数）——**不许"数字对不上但都绿"**。molrs-python 505 passed（wheel 重建后）。15 个 chem-perceive-14 parked RED 仍 RED。

## Out of scope

- **`chem-perceive-14-all-tables` 本身**。本 spec 只把它的输入清干净并重生成它的冻结 reference。
- **`ff/mmff/tables.rs` → `ff/params/mmff.rs` 的搬迁**。本 spec 只把解析器移出 `energy/`，并标注"解析器应随表一起搬"。
- **`OPLSAATypifier::build`**。owner 的裁决只点名了 MMFF 的三个门；02 落地后两个 typifier 表面暂时不对称。这是**已知且刻意**的——是否一并收敛，另开决策，不在本链内自作主张。
- **molpack 的迁移**（独立仓库）。只在 CHANGELOG 以 BREAKING 记录并点名。
- **性能优化**。省下的 ~5.8 ms / 438 KB 是删除的副产品，不是目标，不设性能门。
