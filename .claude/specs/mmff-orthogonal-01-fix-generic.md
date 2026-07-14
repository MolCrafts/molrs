---
title: "MMFF 通用路径修复 — generic path 全 fixture 对齐 RDKit"
slug: mmff-orthogonal-01-fix-generic
status: in-progress
created: 2026-07-14
chain: mmff-orthogonal 1/2
blocks: "chem-perceive-14-all-tables"
---

# MMFF 通用路径修复 — generic path 全 fixture 对齐 RDKit

## Summary

molrs 目前有**两条** MMFF 实现：`ff/mmff/`（bespoke，`MmffForceField`，10/10 fixture 与 RDKit 最大 |Δ| = 0.00000）和 typifier → `ForceField` → `KernelRegistry`（generic，**无 Rust 生产调用者，却是文档化的公开 API**，caffeine 上偏差 150 kcal/mol）。

本 spec 只做一件事：把 generic 路径修到与 RDKit 在**全部 fixture** 上一致（tol 1e-3 kcal/mol），并把 bespoke 的逐项能量分解冻结成 fixture，作为删除 bespoke 之前的参照物。**bespoke 路径在本 spec 内一行不改、全程绿**——它是衡量修复是否成立的参照系。删除工作全部留给 `mmff-orthogonal-02-delete-bespoke`。

## HARD ORDERING（本 spec 的第一约束）

**绝不在替代实现被证明之前删除唯一正确的实现。**

- 本 spec（01）**只修不删**：generic 路径达到 11/11 RDKit parity（10 个现有 fixture + 新增乙腈线性角 fixture）。
- `mmff-orthogonal-02-delete-bespoke` 在 01 绿之后才开始，且 02 的第一个 task 是**在任何删除动作之前重新断言 11/11**。
- 01 + 02 **必须都在 `chem-perceive-14-all-tables` 之前落地**：后者会把 4,065 行死 XML type-def 忠实地编译成 Rust 表，一旦先落地，死数据就被固化成"事实"。

## Domain basis

MMFF94 / MMFF94s（Halgren 1996/1999），oracle 是 RDKit 的 MMFF 实现（Tosco et al. 2014）。

- T. A. Halgren, *Merck molecular force field. I.*, J. Comput. Chem. **17**, 490–519 (1996).
- T. A. Halgren, *MMFF VI. MMFF94s option for energy minimization studies*, J. Comput. Chem. **20**, 720–729 (1999).
- P. Tosco, N. Stiefl, G. Landrum, *Bringing the MMFF force field to the RDKit*, J. Cheminform. **6**, 37 (2014). DOI `10.1186/s13321-014-0037-3`

### 单位

能量 kcal·mol⁻¹；长度 Å；`ka` / `koop` md·Å·rad⁻²；电荷 e。
`143.9325 kcal·mol⁻¹ / (md·Å)` 是换算常数；`0.043844 = 143.9325 · (π/180)²`。
**角度在 molrs 内部一律弧度**；degrees 只在 reader 边界归一化（房规）。

### 相关能量项（RDKit 权威形式）

角弯曲（三次项）：

```
EA = 0.043844 · (ka/2) · Δθ_deg² · (1 + cb_deg · Δθ_deg),   cb_deg = −0.006981317 deg⁻¹
   ≡ 143.9325 · (ka/2) · Δθ_rad² · (1 + cb_rad · Δθ_rad),   cb_rad = cb_deg · 180/π = −0.4 rad⁻¹（精确）
```

线性中心（`linh ≠ 0`，炔/腈/累积双烯）：

```
EA = 143.9325 · ka · (1 + cos θ)          （并且该中心的 stretch-bend 项整体跳过）
```

静电（buffered Coulomb）：

```
E = 332.0716 · q_i q_j / (D · (R + δ)),   δ = 0.05 Å,  D = 1
1-4 对 ×0.75；1-2 / 1-3 排除。1-4 的 vdW 不缩放（RDKit 语义）。
```

vdW（buffered 14-7）：donor 抑制 B 项；donor–acceptor 对 `R*_ij ×= DARAD(0.8)`、`ε_ij ×= DAEPS(0.5)`。

### 实测（release，同 fixture、同坐标，kcal/mol）

| fixture | RDKit | bespoke | generic | error | Σ\|q\| |
|---|---|---|---|---|---|
| e_ethane | −0.33037 | −0.33037 | −0.33035 | 0.00002 | **0.00** |
| e_butane | 6.13899 | 6.13899 | 6.13901 | 0.00002 | **0.00** |
| e_ethylene | 8.58199 | 8.58199 | 0.47206 | **8.11** | 1.20 |
| e_benzene | 18.02361 | 18.02361 | 14.94116 | **3.08** | 1.80 |
| e_caffeine | −109.36051 | −109.36051 | 41.11497 | **150.48** | 5.57 |
| e_big | 108.69662 | 108.69662 | 112.01202 | **3.32** | 7.32 |
| s_nmethylacetamide | −18.14902 | −18.14902 | 10.13324 | **28.28** | 2.60 |
| s_urea | −93.75609 | −93.75609 | 22.84335 | **116.60** | 4.34 |

bespoke 全 10 fixture 最大 |Δ| = **0.00000**。

**唯一被断言的 fixture 是 `e_ethane`**（`molrs/tests/ff/mmff/energy.rs:449`，`let names = ["e_ethane"];`）——而它正是**两个 MMFF 电荷全为零**的分子之一，即唯一一类**结构上无法暴露主缺陷**的输入。

`energy.rs:443-449` 的注释把原因归给 "stretch-bend + torsion eq-fallback label resolution"。**该注释是错的**：stbn 与 torsion 在每个 fixture 上都与 bespoke 五位小数一致（NMA：−0.31596 / −1.18237，两条路径相同）。它自 commit `8edfc37`（2026-06-19，kernel 迁到 per-instance 参数）起就在误导每一个读者。本 spec 删除该注释。

### 四个真实缺陷（generic 路径）

1. **CRITICAL — 完全没有静电项。** `mmff_ele_ctor` 已在 `potential/registry.rs:120` 注册，但**没有任何 ForceField 定义过 `pair/mmff_ele` style**（`grep def_pairstyle("mmff_ele")` 仓库内 0 命中）。generic 的误差 ≈ −E_ele。电荷**已经**在 frame 上（typifier 烘焙了 `atoms.charge`），kernel 也在——只是没人把线接上。
2. **HIGH — vdW 丢了 donor/acceptor 规则。** XML 的 `<VdW>` 行带 `da="-"`，section 带 `DARAD="0.8" DAEPS="0.5"`，但 `parse_mmff_vdw`（`forcefield/xml.rs:494-511`）从不读 `da`，`vdw_combining`（`potential/pair/mmff.rs:70-79`）既没有 donor 的 R\* 抑制、也没有 D/A 缩放（`ff/mmff/energy/params.rs:773-795` 两者都有），并且丢弃了传进来的 `_sp`。代价：urea +0.588，e_big +1.261，NMA +0.116。
3. **HIGH — 三次项常数写错。** `potential/angle/mmff.rs:17`：`const CB_RAD: f64 = -0.40107; // = -0.007 * 180/pi` —— 作者**先四舍五入再换算**。正确值恰为 **−0.4 rad⁻¹（精确）**。0.27 % 的非谐项误差 → e_big 角项 Δ 0.0073，**7 倍于 1e-3 tol**。即使静电与 vdW 全修好，e_big 仍会因此单独 fail。
4. **HIGH — 没有线性角分支。** bespoke 从 `linh` 取 `linear` 并切换到 `E = 143.9325·ka·(1 + cos θ)`，且**对线性中心整体跳过 stretch-bend**；`MMFFAngleBend` 只有三次型。实测弯曲乙腈：generic 2.60691 vs bespoke 2.36706（10 %）。所有炔 / 腈 / 累积双烯都是错的，而**当前 10 个 fixture 一个都覆盖不到**。

## Design

### 修复的边界

本 spec 只动 generic 路径的四处及其供数层。`ff/mmff/**`（bespoke，含 `energy/params.rs` 这个 RDKit-faithful 解析器）**只读不改**。

**1. 接上 `pair/mmff_ele`**

MMFF 静电没有任何 type-row：电荷是逐原子的、已在 frame 上，style 只需标量参数。在两个 MMFF XML 中新增显式 section（与 `<VdWParams>` 对称，而不是在 reader 里凭空 `def_pairstyle`）：

```xml
<ElectrostaticParams dielectric="1.0" delta="0.05" scale14="0.75"/>
```

`read_forcefield_xml_str` 新增 dispatch 分支 → `ff.def_pairstyle("mmff_ele", …)`。pair style 允许零 type-row（guard 对 `category == "pair"` 已豁免），故本 spec 不动 guard——per-instance style 的一般化留给 02。

1-4 缩放走**房规**而非 kernel 硬编码：reader 同时 `set_special_bonds`（lj 1-4 = **1.0**，coul 1-4 = **0.75**）。`mmff_ele_ctor` 改读 `sp` 的 `coulomb14scale` / `dielectric` / `delta`，不再硬编码 `0.75`。`mmff_vdw_ctor` 读 `lj14scale` 并**在 rustdoc 写明"MMFF 的 1-4 vdW 不缩放"**，防止后人"顺手修好"。

**2. vdW donor/acceptor**：`parse_mmff_vdw` 读 `da` 写进 `PairType.params`；`vdw_combining` 增加 donor 抑制与 D/A 缩放，语义逐行对齐 `ff/mmff/energy/params.rs:773-795`。**不复制那份代码**——重写一次同一公式，由逐项分解测试证明二者等价。

**3. 三次常数**：`CB_RAD = -0.4`（精确），注释写清推导，并注明"先取整再换算"是原缺陷。

**4. 线性角**：`annotate_mmff` 在每条 angle 上烘焙 `linear`（bool 列，源自 `MmffProp.linh`）。`MMFFAngleBend` 读该列 → 线性分支（含解析梯度）；`mmff_stbn_ctor` 读该列 → **跳过**线性中心的 stretch-bend 行。

### 为什么"逐项分解"是本 spec 的核心资产

总能量在 caffeine 上把 150 kcal/mol 的静电空洞藏在了部分抵消的其它项后面；e_ethane 的总能量甚至掩盖了**整整一个缺失的能量项**。

因此引入 **per-style 分解断言**（bond / angle / stbn / torsion / oop / vdw / ele 七项），并把 bespoke 的 `MmffForceField::energy_terms()` 输出**冻结成 fixture**（`<name>.breakdown.json`）。两个后果：

- generic 的每一项都必须逐项对上，而不是"总和碰巧对上"；
- **spec 02 删掉 bespoke 之后，参照物仍在**——冻结的 fixture 是 02 的安全网。合法性由外部 oracle 自证：每个 breakdown 必须满足 `Σ(七项) == mmff94_total_energy`（RDKit 值）到 1e-6。

### 新 fixture：乙腈（线性角）

缺陷 4 在现有 10 个 fixture 上**不可见**。新增 `e_acetonitrile`（CH₃C≡N）。

⚠️ **几乎线性时两种角形式都趋近 0，缺陷自动隐身。** 这不是理论顾虑——实测（每单位 `ka`）：

| C–C≡N 角 | RDKit 能量 | 线性式 `143.9325·ka·(1+cos θ)` | 三次式 | 两式之差 |
|---|---|---|---|---|
| **179.14°**（ETKDG 直接产出） | 0.49055 | 0.01621 | 0.01631 | **~1e-4·ka —— 低于 1e-3 容差** |
| 175° | 0.73922 | 0.54771 | 0.56719 | 0.02·ka |
| **170°（采用）** | **1.50630** | 2.18666 | 2.34526 | **0.16·ka —— 远高于容差** |
| 160° | 4.54993 | 8.68019 | 9.99323 | 1.3·ka |

**修正配方**（初稿的"ETKDG embed，不最小化"是错的——它给出 179.14°，过不了自己那道 ≥2° 的门，会产出一个假 fixture）：

1. `Chem.AddHs(MolFromSmiles("CC#N"))` → `EmbedMolecule(randomSeed=42)`
2. `rdMolTransforms.SetAngleDeg(conf, 0, 1, 2, 170.0)` —— **刻意弯曲**
3. 在**该几何**上取 `MMFFGetMoleculeForceField(...).CalcEnergy()` 作为 oracle（RDKit 在任意坐标上算能量，这是合法 oracle）
4. 写出 SDF（坐标）+ `.energy.json`（沿用现有 schema）

测试仍断言该 fixture 的 C–C≡N 角**偏离 180° ≥ 2°**，作为"fixture 没退化成假 fixture"的自检。

生成环境：`molrs-python/.venv` 已装 `rdkit==2026.3.3`（本 spec 实施时装入，仅用于生成 fixture，非运行期依赖）。

### 与 chem-perceive-14 的交接

本 spec 修改两个 MMFF XML（新增 `<ElectrostaticParams>`），这会让 `molrs/tests/ff/fixtures/tables/*.reference.txt` 的冻结 dump 与 `tables_gate.rs` 的 SHA-256 pin 过期。它们是 **chem-perceive-14 的 parked RED**，本 spec **不去 re-bless**——在 02 里连同 4,065 行死数据的删除**一次性重新生成**。禁止在 01 里单独刷新（会白刷一遍）。

## Files to create or modify

- `molrs/src/ff/potential/angle/mmff.rs` — `CB_RAD` = −0.4；线性分支（能量 + 梯度）；`mmff_stbn_ctor` 跳过线性中心
- `molrs/src/ff/potential/pair/mmff.rs` — `vdw_combining` 加 donor 抑制 + D/A 缩放；`mmff_vdw_ctor` / `mmff_ele_ctor` 改读 `sp`
- `molrs/src/ff/forcefield/xml.rs` — `parse_mmff_ele` + dispatch；`parse_mmff_vdw` 读 `da`；`set_special_bonds`
- `molrs/src/ff/typifier/mmff/frame_builder.rs` — 烘焙 `linear` 列
- `molrs/data/mmff94.xml`、`molrs/data/mmff94s.xml`、`data/mmff94.xml`、`data/mmff94s.xml` — 新增 `<ElectrostaticParams>`
- `scripts/mmff_to_xml.py` — 同步 emit（否则下次重生成会抹掉）
- `molrs/tests/ff/mmff/energy.rs` — 目录扫描式 fixture 列表；删除错误注释；per-style 分解断言；FD 梯度扩到全 fixture
- `molrs/tests/ff/mmff/fixtures/e_acetonitrile.{sdf,energy.json}` (new)
- `molrs/tests/ff/mmff/fixtures/<name>.breakdown.json` × 11 (new)

## Tasks

- [ ] Write failing full-fixture parity test in `molrs/tests/ff/mmff/energy.rs` — replace `let names = ["e_ethane"]` with a scan of `fixtures/*.energy.json`, delete the false stbn/torsion comment at `:443-449`
- [ ] Add the acetonitrile linear-angle fixture (RDKit oracle) with an in-test guard that its C–C≡N angle deviates >= 2 degrees from 180
- [ ] Write failing per-style breakdown test and freeze the bespoke reference (`<name>.breakdown.json` x 11, each self-checked by `sum(terms) == mmff94_total_energy` to 1e-6)
- [ ] Wire `pair/mmff_ele`: add `<ElectrostaticParams>` to both XMLs (+ root copies + `scripts/mmff_to_xml.py`), add `parse_mmff_ele` + `set_special_bonds`, make `mmff_ele_ctor` read `sp`
- [ ] Implement the MMFF vdW donor/acceptor rule (`da` from type rows, B/Beta/DARAD/DAEPS from `sp`, donor R* suppression + D/A scaling)
- [ ] Fix `CB_RAD` to −0.4 rad^-1 exactly, with the deg->rad derivation in the comment
- [ ] Implement the linear-angle branch: bake `linear` in `frame_builder.rs`, branch in `MMFFAngleBend`, skip linear centres in `mmff_stbn_ctor`
- [ ] Extend the finite-difference gradient test to every fixture (max_err < 1e-5)
- [ ] Add rustdoc with units on every changed kernel / reader symbol, including "MMFF does not scale 1-4 vdW"
- [ ] Run full check + test suite (fmt; both clippys; `cargo test -p molcrafts-molrs --features full`; `RUSTDOCFLAGS='-D warnings' cargo doc`)

## Testing strategy

**Happy path**
- `generic_path_total_energy_matches_rdkit`：fixture 列表由**目录扫描**产生，11/11 全部断言，|Δ| < 1e-3。任何"只断言子集"的写法被**结构性禁止**——测试拿不到硬编码列表。
- `generic_path_per_style_breakdown_matches_frozen`：逐 style 七项对上冻结的 bespoke 分解，|Δ| < 1e-6。
- bespoke parity 保持绿，最大 |Δ| = 0.00000 —— 本 spec 未触碰 bespoke 的证据。

**Edge cases**
- `e_acetonitrile`：先断言几何弯曲 ≥ 2°（否则 fixture 无效），再断言能量 parity；同时断言 stbn 项**恰好为 0**。
- `e_ethane` / `e_butane`：Σ|q| = 0 —— 静电项必须**恰好为 0**（而非"碰巧很小"），证明接上静电没有引入伪能量。
- `e_butane`：唯一带 1-4 对且电荷为零的分子 —— 证明 `lj14scale = 1.0` 的 no-op 语义没有改变 vdW。
- 梯度：全 fixture 有限差分（h = 1e-5），max_err < 1e-5。

**反模式禁令**
任何"只断言子集"的测试必须在测试内**写明被排除分子的原因**；**"尚未实现"不是排除理由，而是 fail 的理由**。

## Out of scope

- **任何删除**。`MmffForceField`、`build_mmff_potentials`、`MMFF94Typifier::build`、`classify.rs`、4,065 行死 XML —— 全部留给 02。在替代品被证明前拆掉参照系是本链最不该犯的错。
- **per-instance style 的一般化**（`ParamSource` / grep gate）—— 02 的架构工作；01 靠 pair style 对零 type-row 的既有豁免绕过去。
- **`classify.rs` 的错误分类器** —— 它们影响的是**标签字符串**，per-instance kernel 不读标签，故不影响本 spec 的能量 parity。02 删除。
- **chem-perceive-14 冻结 fixture / SHA pin 的刷新** —— 02 一次性重生成。
- **conformer / molrs-python 的调用点迁移** —— 它们仍指向 bespoke，01 不动。
