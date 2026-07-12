---
title: "chem-perceive 1/14 — perceive 层：core 之上、ff 之下的化学感知层"
status: in-progress
created: 2026-07-12
chain: chem-perceive
depends_on: ""
blocks: "chem-perceive-03-bondtype; chem-perceive-04-equivalence; chem-perceive-05-atd-typifier; chem-perceive-13-python-bind"
---

# 化学感知层 `perceive`（core↑ / ff↓）

> Chain **chem-perceive** 1/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

把散落在 `core/chem/` 的化学感知代码整体上提为**独立一层** `molrs::perceive`，位置严格在
`core` 之上、`ff/typifier` 与 `ff/forcefield` 之下，供 `io` / `ff` / `conformer` / binder 复用。
对外形态是 builder：`Perceive::new()...find_*(&MolGraph) -> MolGraph`（图进图出，写回组件，
不返回旁路表），与既有的 `Conformer::new(opts).generate(mol)` 和 `Typifier::typify(&Mol) -> Mol` 同构。

本期是**纯搬迁 + API 皮肤**：不改任何算法，行为零差异。`find_bond_types`（03）与
`find_equivalence_classes`（04）在本期**不落桩**，后续再加。

## Domain basis

### 分层：已 grep 验证，`perceive` 只坐在 `core::{system, error}` 上

这是唯一可能推翻整个设计的一条，**已验证通过**。搬迁的 5432 行对 `crate::ff` / `crate::io` /
`molrs::ff` / `molrs::io` 的 grep 是 **0 命中**。整个搬迁集的**全部**入向依赖只有：
`core::system::{atomistic, molgraph, element, topology}` 与 `core::error::MolRsError`，
外加集合内部的 `chem::rings` / `chem::hydrogens`。

- `smarts/` **不**依赖 `ff`：`smarts/mod.rs:142`、`smarts/ast.rs:28,209` 里提到 "iterative typifiers"
  只是**散文描述消费方**，零代码耦合。依赖方向是 `ff/typifier → smarts`，正是设计想要的。
- `smarts/` **不**依赖 `io::smiles`：对 RDKit `SmartsParse.cpp` 的引用是**语法出处标注**，不是 import。
  方向反过来才对——`io/smiles/mod.rs:5,24` 明确写着 `SMARTS → … → crate::chem::smarts::SmartsPattern`，
  即 **`io` 依赖 `chem`**，佐证 `perceive` 在 `io` 之下。
- `gasteiger.rs` **不**依赖 `ff`：唯一的非 `core::system` import 是 `chem::hydrogens::implicit_h_count`。

`core` 对 `core::chem` 也是**零反向依赖**（只有 `core/mod.rs` 的 re-export 与
`core/system/atomistic.rs:641` 的一句注释），因此上提无环。

### `perceive` 常开 —— 这**不是**对 CLAUDE.md 的偏离

`molrs/src/lib.rs:60` 的 `pub mod optimize;` **本来就没有 cfg**，且 `optimize` 根本不是一个 feature
（`Cargo.toml` 里 0 命中）。"core 常开、其余 feature-gate" 这条规则在代码里**早已不成立**，
`perceive` 只是加入既有先例。

更强的理由：**常开才是零差异的选择**。`core` 常开、`chem` 在其中，所以今天**每一种**编译配置
（含 `default-features = false` 的 `molrs-ffi` 与 `molrs-wasm`）都在编这 5432 行。常开等于逐字节
复现当前依赖图；feature-gate 反而会**改变**依赖图，把一个真实的行为变更塞进一个验收标准写着
"行为零差异" 的 spec。（gate 掉能让 wasm bundle 少 5.4k 行——那是**另一个可测量的 follow-up**，
不是 01 的范围。）

订正：早先"`core/chem` 不引入 petgraph（它只在 `compute/hbond/`）"的说法**是错的**——
petgraph 是一个**死依赖**，全 `molrs/src/` 里 `use petgraph` **0 命中**，
而 `Cargo.toml:43,83` 仍把它挂在 `smiles` feature 下。这不影响本设计，反而**加强**它：
搬迁集的可选依赖不是"只有别处也有的那些"，而是**一个都没有**。
（清理 petgraph 是**独立的一行 PR**，不并进 01。）

### 🚨 crate 根的扁平 re-export 面必须**搬迁**，不是删除

`molrs/src/lib.rs:49` 是 `pub use crate::core::*;`——正是这个 glob 把 `core/mod.rs:53-61`
提升成 **18 个扁平根名**：`molrs::find_rings`、`molrs::SmartsPattern`、`molrs::MatchOptions`、
`molrs::add_hydrogens`、`molrs::perceive_aromaticity`、`molrs::compute_gasteiger_charges`、
`molrs::RingInfo`、`molrs::Reaction`、`molrs::BondStereo` …

若按初稿"移除 re-export"，这 18 个名字全部消失，**13 个调用点当场断**，其中
**4 个在 `ff/` 里**（`ff/typifier/am1bcc.rs:13`、`opls/deps.rs:25`、`opls/layered.rs:27`、
`opls/typing.rs:14-15`），而且 `typing.rs` 那两个是 **intra-doc link**——它们不报类型错，
只在 `cargo clippy -D warnings` 下作为 broken intra-doc link 挂掉，本地极易漏、CI 必红。

⇒ 正解是把 `core/mod.rs:53-61` 那个块**原样搬到 `lib.rs`**、重指向 `perceive`。
这样 13 个调用点**一行都不用改**，diff 大幅缩小。

`pub use crate::perceive as chem;` 仍然必须保留（它恢复的是**模块路径** `molrs::chem::…`，
**不是**扁平名），因为 `molrs-python` 是独立 workspace 且是 path 依赖，主 workspace 的
`cargo check` 抓不到它的断裂。alias 在 spec 13 删除。

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

- 新增 `molrs/src/perceive/**`（由 `molrs/src/core/chem/**` 整体移动而来）
- 修改 `molrs/src/lib.rs`（挂 `perceive` 模块 + compat alias）
- 修改 `molrs/src/core/mod.rs`（移除 `chem` 子模块与 `:53-58` 的 re-export）
- 修改上述所有 in-crate 调用点 + `CLAUDE.md` 模块表

## Tasks

- [ ] Write failing tests for the `Perceive` builder (graph-in / graph-out, non-mutating) — RED before GREEN
- [ ] Create `molrs/src/perceive/{mod,aromaticity,gasteiger,hydrogens,rings,rotatable,stereo}.rs` by moving `core/chem/*` verbatim; move `core/chem/smarts/{mod,ast,parser,matcher,reaction}.rs` to `perceive/smarts/`
- [ ] Add the `Perceive` builder with `find_rings` / `find_aromaticity` / `find_hydrogens` / `find_stereo` / `find_rotatable`
- [ ] RELOCATE the `core/mod.rs:53-61` re-export block into `lib.rs`, retargeted at `perceive` — preserves all 18 flat crate-root names (`molrs::find_rings`, `molrs::SmartsPattern`, …). DO NOT delete it: 13 call sites depend on those names, 4 of them in `ff/`, 2 as clippy-only broken intra-doc links.
- [ ] Add `pub use crate::perceive as chem;` compat alias in `lib.rs` so the separate `molrs-python` workspace keeps building (deleted in spec 13)
- [ ] Rename `conformer/distgeom/perceive.rs` → `conformer/distgeom/mol_features.rs` to avoid the module-name collision with the new top-level `perceive` layer (it is `mod`-private to `distgeom`; the rename is contained)
- [ ] Update the module table in `CLAUDE.md`: add the `perceive` row (ALWAYS compiled); also fix the 6 pre-existing staleness items the architecture check found (missing `optimize` + `stream` rows, missing `voronoi` feature, SMARTS located in `core/chem/` not `io/smiles/`, the false petgraph-VF2 claim, and the `core` row still listing rings/stereo/Gasteiger/hydrogens)

## Testing strategy

纯搬迁，验收即**行为零差异**：现有整套测试（`core/{aromaticity,rings,hydrogens}`、
`embed/{pipeline,etkdg,success_rate}`、examples、benches）全绿且断言值一字不改。
额外加一个 `Perceive` builder 的 smoke 测试（`find_rings` 返回的图带 ring 组件）。
`molrs-python` 是独立 workspace，必须仍能靠 compat alias 编过。

## Out of scope

- 任何算法改动（芳香性判定、成环、加氢逻辑一律原样搬）。
- `core/system/graph_hash.rs` **留在 core**：它是通用图算法（不吃 `Element`、不吃键级、不吃芳香性），
  且 `region-support-01-graph-hash` 正在 in-flight 改它；04 只是消费它（且仅作为可选预筛，见 04）。
- 删除 compat alias（留到 13）。

**架构检查发现的、明确不并入 01 的 follow-up：**
- **petgraph 死依赖清理**：全 `molrs/src/` `use petgraph` **0 命中**，但 `Cargo.toml:43,83`
  仍把它挂在 `smiles` feature 下（VF2 matcher 是手写的）。独立的一行 PR，并同步订正 CLAUDE.md:147。
- **重复的 SSSR 实现**：`perceive::rings::find_rings(&Atomistic) -> RingInfo` 与
  `core::system::topology::Topology::find_rings() -> TopologyRingInfo` 都是 SSSR，都在 crate 根导出，
  搬迁后会**跨两层**各留一份（wasm 绑一个、python 绑另一个）。记入 `.claude/notes/notes.md` 作为
  follow-up：要么让前者委托后者，要么把后者明确文档化为纯图原语、前者是它的化学装饰。
- **`io/smiles/chem/` 改名**：spec 13 删掉 `chem` alias 之后，`chem` 在树里将只剩下 SMILES AST 那一个，
  语义很差。建议 13 顺手把 `io/smiles/chem/` 改名为 `io/smiles/ir/`。

**架构检查纠正的两个误判（不要照初稿去改这两处）：**
- `io/smiles/smiles/to_atomistic.rs` **不是** `core::chem` 的消费者——它 import 的是 io 本地的
  `crate::io::smiles::chem::ast`（SMILES IR），与 `core::chem` 零关系。**改它是错的。**
- `tests/embed/*` 里的 `add_hydrogens: false` 是 `ConformerOptions` 的**结构体字段**，不是那个函数。
  这些文件**不需要任何改动**，但要留在回归运行里——它们经 `Conformer` 间接跑过搬迁的代码，
  正是我们想要的零差异信号。
