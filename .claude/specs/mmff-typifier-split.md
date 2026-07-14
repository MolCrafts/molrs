---
title: "MMFF typifier — one engine, two front doors (MMFF94Typifier / MMFF94STypifier)"
slug: mmff-typifier-split
status: in-progress
created: 2026-07-14
scope_layer: "molrs/src/ff/typifier/mmff/, molrs/src/ff/mmff/, molrs-python/src/ff/"
sequencing: "must land BEFORE chem-perceive-14-all-tables"
---

# MMFF typifier — one engine, two front doors

## Summary

molrs 今天可以算 MMFF94s 的**能量**，却无法用 MMFF94s **打类型**：`MmffVariant::{Mmff94, Mmff94s}` 存在，`_S` 表存在，`energy/params.rs` 的分派也正确，但 `ff/typifier/mmff/frame_builder.rs` 把 variant 硬编码成 `MmffVariant::Mmff94`，于是 typifier / ForceField 这条路径永远只产出 MMFF94 参数——没有任何测试覆盖这个静默缺口。

本 spec 把 variant 贯通到 typifier 的**两条**参数路径（Frame 标注 + `to_potentials` 的 ForceField 树），并按 owner 的定案重塑公开面：内部只有**一个** typifier 引擎，对外只有**两个具名门面** `MMFF94Typifier` 与 `MMFF94STypifier`——用户永远不传 variant 旗标。

这同时给 `mmff94s.xml` 装上它的**第一个真实消费者**，是 `chem-perceive-14-all-tables` 的前置条件。

## Domain basis

### MMFF94 与 MMFF94s 的实测差异

`diff molrs/data/mmff94.xml molrs/data/mmff94s.xml`，两文件均 5298 行，**仅 108 行不同**：

| 区段 | 差异行数 | = 参数行 | 说明 |
|---|---|---|---|
| `<ForceField>` name | 2 | — | `MMFF94` vs `MMFF94s` |
| `<Oop>` | 22 | 11 行 | 全部以两种氮为中心 |
| `<Torsion>` | 84 | 42 行 | 同一邻域 |
| `<Type>`/`<Def>`/`<Prop>`（95 行）、`BCI`/`PBCI`、Bond、Angle、StretchBend、vdW | **0** | — | **逐字节相同** |

11 行 Oop 差异全部集中在 XML 自己命名的两个氮类型：**type 10 = `NC=O`**（酰胺氮，6 行）与 **type 40 = `NC=C`**（烯胺型氮，5 行）。差异的内容是面外力常数 `koop` 被**统一提到一个显著为正的值**：

- MMFF94：`koop` ∈ {−0.033, −0.030, −0.020, −0.019, −0.007, −0.006, −0.005, **+0.004**}
- MMFF94s：`koop` 一致为正（type 10 = **+0.015**，type 40 = **+0.030**）⇒ 平面构型是能量**极小** ⇒ 该氮被压平。

> **不是"符号翻转"。** 本 spec 起草时曾如此概括，**是错的**：MMFF94 侧有一行本来就是正的（`+0.004`），实测落在 `s_aniline` 的 type-40 氮上——MMFF94s 把它放大 7.5 倍到 `+0.030`，并没有翻转符号，只是把平面性的势阱**挖深**。多数行确实由负转正（平面从能量**极大**变**极小**），但断言必须写成"**值改变、且 94s 侧为 +0.015 / +0.030**"，**不得**写成"符号翻转"。实测值见下表，已钉进 `mmff_variant.rs` 的 `N_CENTRE_ORACLE`。

| fixture | 中心 | MMFF type | Oop 行 | **MMFF94（实测）** | MMFF94s |
|---|---|---|---|---|---|
| `s_acetamide` | N | 10 | `3_10_28_28` | −0.019 | +0.015 |
| `s_nmethylacetamide` | N | 10 | `1_10_3_28` | −0.020 | +0.015 |
| `s_urea` | N ×2 | 10 | `3_10_28_28` | −0.019 | +0.015 |
| `s_aniline` | N | 40 | `28_40_28_37` | **+0.004** | +0.030 |

这就是 "s" = **static**：MMFF94s 在**静态**能量极小点上把离域三价氮平面化（使最小化后的几何与晶体结构中的平面氮一致），MMFF94 复现的是动态/时间平均的图像。

### 面外项的能量形式

取自内核源码 `molrs/src/ff/potential/improper/mmff.rs:1,46`（非二手转述）：

```
E_oop = 0.5 · 143.9325 · koop · χ²
```

- χ = Wilson 面外角，**弧度**（house 约定：内部一律弧度）；
- `koop` 单位 md·Å·rad⁻²，143.9325 是 md·Å → kcal·mol⁻¹ 换算常数；
- 故 `koop > 0` ⇔ χ=0（平面）是极小，`koop < 0` ⇔ χ=0 是极大。上表的符号翻转与这个式子是同一件事。

### 参考文献

molrs 在 `molrs/src/ff/mmff/mod.rs:41` 引用 Halgren 1999：

- T. A. Halgren, *MMFF VI. MMFF94s option for energy minimization studies*, J. Comput. Chem. **20**, 720–729 (1999).
- T. A. Halgren, *Merck molecular force field. I.*, J. Comput. Chem. **17**, 490–519 (1996).
- 表格出处：RDKit `Code/ForceField/MMFF/Params.cpp`（BSD-3）。

> **诚实边界**：除引文条目本身外，上面每一条论断都来自**实测的 XML diff** 与 molrs 自己的内核/文档注释。Halgren 1999 原文**未阅读**，本 spec 不得据其正文杜撰任何公式或数值。

### 校验锚点已经存在

`molrs/tests/ff/mmff/fixtures/` 里已有 RDKit 生成的 MMFF94s 参照：`s_acetamide` / `s_nmethylacetamide` / `s_aniline` / `s_urea`，每个 `.energy.json` 同时带 `mmff94_total_energy` 与 `mmff94s_total_energy`（例：`s_nmethylacetamide` = −18.149016 vs −18.893930 kcal/mol，差 0.745）。

这些 fixture 目前**只被 bespoke 能量路径**（`MmffForceField`）消费，**typifier 路径一个都没用上**。

## Design

### 一个引擎，两扇具名门面

owner 定案（合同，非建议）：**内部抽象出一个 typifier 引擎，用户永远只用 `MMFF94Typifier` 和 `MMFF94STypifier`**。形状对齐树里已有的 `AtdTypifier`（一个引擎 + 七张表），区别是 MMFF 只有两个变体、且在文献中**有名字**，所以暴露成两个**具名类型**而非一个 `parameter_set` 参数。

```
molrs::ff::typifier::mmff
├── (crate-private) 引擎        ← 唯一实现：variant + MMFFParams + ForceField
├── pub struct MMFF94Typifier   ← 门面，variant = Mmff94,  XML = MMFF94_XML
└── pub struct MMFF94STypifier  ← 门面，variant = Mmff94s, XML = MMFF94S_XML
```

硬约束：

- 公开 API **不得**出现 `MMFFTypifier::new(variant)` 这类旗标构造器；`molrs/src/ff/typifier/mmff/` 下任何 `pub fn` 的签名里**不得**出现 `MmffVariant`。variant 是引擎的私有字段。
- **不得**建第二套并行的 typifier 栈（这条链此前已因"复制一个估计器"返工过一次）。两个门面是同一引擎的 newtype，方法逐条转发。
- 两个门面都实现 `Typifier` trait，都提供 `new()`、`from_xml_str(xml)`（variant 由门面钉死）、`typify` / `build` / `params()` / `ff()`，以及 `typify_bond` / `typify_angle` / `typify_dihedral` 转发。

### variant 必须贯穿**两条**参数路径

只修一条会让 `MMFF94STypifier` **静默产出 MMFF94 势能**。

**路径 1 — Frame 标注（`frame_builder::annotate_mmff`）**，硬编码所在：

- `frame_builder.rs:41` `MmffMolProperties::compute(mol, MmffVariant::Mmff94)`
- `frame_builder.rs:176` `eparams::torsion_params(MmffVariant::Mmff94, …)`
- `frame_builder.rs:224` `eparams::oop_koop(MmffVariant::Mmff94, …)`

`annotate_mmff` 签名扩为 `(mol, &MMFFParams, MmffVariant)`。这条路径把 `v1/v2/v3`（dihedral）与 `koop`（improper）**烘焙成 Frame 列**，而 `mmff_torsion_ctor` / `mmff_oop_ctor` 正是**读这些列**——94 vs 94s 的数值差异在通用路径上就是从这里流出的。

**路径 2 — ForceField 树（`self.ff.to_potentials(&frame)`，`typifier/mmff/mod.rs:93`）**。`ff` 由 `read_forcefield_xml_str(<XML>)` 解析而来：`MMFF94STypifier` 必须从 `molrs::data::MMFF94S_XML` 构树。它决定 `ff.name`（`MMFF94s`）以及 Oop/Torsion 风格的 type 行；`Style::to_potential` 在 `type_params.is_empty()` 时对非 pair 类别直接报错，所以这棵树在结构上是必需的，不是装饰。

`MMFFParams`（typing 元数据）在两份 XML 中逐字节相同，但仍从**各自的 XML** 解析——单一真相源，不做特判。

### 破坏性重命名：干净断裂，不留兼容别名

`MMFFTypifier` 是 Rust 与 Python 双向的公开符号（75 处引用 / 25 个文件）。方案：**直接重命名为 `MMFF94Typifier`，不保留别名、不留 `#[deprecated]` shim**。

理由：仓库处于 `stage: experimental`（pre-1.0）；owner 明确不喜欢兼容垫片；下游（molpy / molpack）以精确版本 pin molrs；别名会让"用户永远只用两个具名门面"这条定案在第一天就被绕过。CHANGELOG 记 BREAKING。

Python 侧同构：`molrs.MMFFTypifier` 消失，替换为 `molrs.MMFF94Typifier` 与 `molrs.MMFF94STypifier`。

### 为什么这是**一个** spec 而不是一条链

molrs-python 的 `PyMMFFTypifier` 按**名字**包住 `MMFFTypifier`。若把 PyO3 重命名切到后一个子 spec，前一个子 spec 落地时 molrs-python **无法编译**——即"不能独立合并"的子 spec。binder 不是 crate-graph 里的一层架构，是同一公开面的薄绑定；重命名必须原子完成。

### 顺序：本 spec 必须先于 `chem-perceive-14-all-tables`

spec 14 会把三份 XML 转成提交进仓库的 Rust 表。**今天 `mmff94s.xml` 的消费者是零**——先做 14 会生成一张**没有读者的死表**，正是 owner 已两次否决的死重量（`data/gen3d/` 前车之鉴）。本 spec 落地后 `mmff94s.xml` 有了真实读者，14 才能把三份都转掉且每张表都有读者。

对 14 的接口承诺：`MMFF94STypifier::new()` 是**稳定接缝**——今天它的函数体读 `MMFF94S_XML`，14 之后读生成表。

### 顺带修掉的陈旧文档（在范围内）

`molrs/src/ff/mmff/mod.rs:43-44` 声称 Oop/Tor 表 "not yet present in [`tables`]"。它们**在**（`MMFF_OOP_S` 117 行、`MMFF_TOR_S` 926 行），且能量层早已在用。该注释先于表存在，**已经骗过一轮读者**（本 spec 起草时据它误判"MMFF94s 是缺口"），必须改正。

## Files to create or modify

**Rust 核心**

- `molrs/src/ff/typifier/mmff/mod.rs` — 私有引擎 + 两个门面
- `molrs/src/ff/typifier/mmff/frame_builder.rs` — `annotate_mmff` 接收 variant；拆掉三处硬编码
- `molrs/src/ff/typifier/mmff/tests.rs`、`molrs/src/ff/typifier/mod.rs`
- `molrs/src/ff/mmff/mod.rs` — 修正 `MmffVariant` 陈旧注释
- `molrs/src/ff/typifier/opls/mod.rs` — 文档交叉引用改名

**Rust 测试**

- `molrs/tests/ff/typifier/mmff_variant.rs`（new）— 区分性 + 同一性护栏 + 源码门禁
- `molrs/tests/ff/typifier/mod.rs`、`molrs/tests/ff/typifier/mmff.rs`
- `molrs/tests/ff/potential/mmff.rs`、`molrs/tests/ff/mmff/energy.rs`
- `molrs/tests/ff/tables_equivalence.rs` — `shipped_mmff94s()` 改指 `MMFF94STypifier::new()`
- `molrs/tests/ff/tables_gate.rs` — 更新"mmff94s 无消费者"的陈旧注释

**示例与文档**

- `molrs/examples/typify_molecule.rs`、`molrs/examples/typify_litfsi.rs`
- `docs/interop.md`、`CHANGELOG.md`（BREAKING）、`CLAUDE.md`

**Python 绑定**

- `molrs-python/src/ff/mod.rs`、`molrs-python/src/lib.rs`
- `molrs-python/python/molrs/{__init__.py,typifier.py,molrs.pyi}`
- `molrs-python/tests/test_mmff_typifier.py`（改名）、`molrs-python/tests/test_mmff94s_typifier.py`（new）
- `molrs-python/examples/{forcefield_ethane.py,full_pipeline.py}`、README、site-src ×3

## Tasks

- [ ] Write failing tests for the two front doors (`molrs/tests/ff/typifier/mmff_variant.rs`): koop sign flip on type-10/40 N centres of `s_nmethylacetamide` / `s_acetamide` / `s_aniline`; byte-identical typed output on `e_ethane` / `e_benzene`; ForceField-tree diff confined to name + 11 Oop + 42 Torsion rows; source gate asserting `MmffVariant::Mmff94` appears zero times in `frame_builder.rs`
- [ ] Implement the crate-private MMFF typifier engine plus the `MMFF94Typifier` / `MMFF94STypifier` newtype front doors (no `MmffVariant` in any public signature)
- [ ] Thread `MmffVariant` through path 1 in `frame_builder.rs` — extend `annotate_mmff`, replace the hardcoded variant at `MmffMolProperties::compute`, `eparams::torsion_params`, `eparams::oop_koop`
- [ ] Rename the Rust call sites (identifier changes only — no asserted value moves)
- [ ] Point `shipped_mmff94s()` at `MMFF94STypifier::new()` and refresh the stale gate docs in `tables_gate.rs`
- [ ] Add rustdoc with units (koop in md·Å·rad⁻², χ in radians); fix the stale `MmffVariant` comment in `ff/mmff/mod.rs`
- [ ] Write failing Python tests for both typifiers; rename the existing ones
- [ ] Implement `PyMMFF94Typifier` / `PyMMFF94STypifier`, register both, update `__init__.py` / `typifier.py` / `.pyi`
- [ ] Update docs and examples for the renamed surface (`docs/interop.md`, `CHANGELOG.md` BREAKING, `CLAUDE.md`, README, site-src)
- [ ] Run the full gate set (fmt; both clippys; `cargo test -p molcrafts-molrs --features full`; `RUSTDOCFLAGS='-D warnings' cargo doc`; then `cd molrs-python && source .venv/bin/activate && maturin develop --release && python -m pytest -q`)

## Testing strategy

**区分性（这是本 spec 挣钱的地方）**

**两条路径都必须覆盖。只查原子类型的测试会空洞通过——原子类型按设计就是相同的。**

- 路径 1（koop）：用已有的 `s_nmethylacetamide` / `s_acetamide` / `s_aniline` fixture，断言以 type 10 / type 40 氮为中心的 improper 上烘焙的 `koop`：94s 侧 = **+0.015**（type 10）/ **+0.030**（type 40）且 > 0；94 侧 ≠ 该值。若某中心实测非负，**记录实测值，不得为了让断言变绿而改实现**。
- 路径 1（torsion）：至少一个 dihedral 的 `(v1, v2, v3)` 在两个 typifier 间不同（> 1e-12）。
- 路径 2（ForceField 树）：两棵树的差异**只**出现在 name、11 行 Oop、42 行 Torsion；Atom/Bond/Angle/StretchBend/vdW/charge 行逐条相同。
- 端到端：两个 typifier 的 `build()` 总能量在每个 `s_*` fixture 上相差 > 1e-3 kcal/mol。

**同一性护栏（防止测试因"根本不存在的差异"而侥幸通过）**

- `e_ethane`（无三价氮）与 `e_benzene`（**有 oop 中心但无 type 10/40 氮** —— 所以这条护栏非空洞）：两个 typifier 的 typed Frame 全部列逐值相同，`build()` 总能量逐位相同。

**结构门禁**

- `MmffVariant::Mmff94` 字面量在 `frame_builder.rs` 中出现 **0** 次；`typifier/mmff/` 下无 `pub fn` 签名含 `MmffVariant`；`molrs/src` 与 `molrs-python/src` 中 `MMFFTypifier` 出现 0 次。
- 可达性：`molrs::data::MMFF94S_XML` 至少被 `molrs/src/` 命名一次（spec 14 的解锁条件）。

**回归**

- 既有 MMFF94 测试全绿，且 **diff 中不得出现任何被修改的数值字面量**——本 spec 重排的是管线，不是 MMFF94 的参数化。
- 基线：molrs `1890 passed / 0 failed`；molrs-python `497 passed / 0 failed`。**Python 轮子必须重建**（`maturin develop --release`），否则 pytest 会对着旧 `.so` 撒谎。

**科学校验的边界（明确不做）**

通用路径（`to_potentials`）在芳香/sp2 分子上仍有 stretch-bend / torsion 标签的等价回退缺口（`tests/ff/mmff/energy.rs:443-449`）。因此本 spec **不**新增"通用路径 MMFF94s 能量对齐 RDKit `s_*` fixture"这类断言——那会因与本 spec 无关的既有缺口而失败。RDKit 锚定的 MMFF94s 能量校验继续留在已验证的 bespoke 路径。

## Out of scope

- **不**重新参数化 MMFF94：任何既有断言数值都不许动。
- **不**把 `mmff94s.xml` 转成 Rust 表——那是 `chem-perceive-14-all-tables` 的活；本 spec 只负责给它装上读者。
- **不**修通用路径在芳香/sp2 分子上的 stbn/torsion 标签等价回退缺口。
- **不**引入 `MmffVariant` 旗标式公开构造器，**不**保留 `MMFFTypifier` 兼容别名或 `#[deprecated]` shim。
- **不**新建第二套并行 typifier 栈。
- **不**改动 bespoke 能量路径（`MmffForceField` / `MmffMolProperties`）已有的 RDKit 校验与 API。
- **不**把 `conformer` 的 MMFF94 cleanup 切到 94s。
- **不**在本仓库内修 molpy / molpack 下游调用点（仅在 CHANGELOG 记 BREAKING）。
