---
title: "删除 pair/mmff_ele —— MMFF 的静电由通用 LAMMPS 风格内核组合而成"
slug: mmff-ele-compose
status: approved
created: 2026-07-15
blocks: "chem-perceive-15-final-acceptance"
---

# 删除 `pair/mmff_ele`

## Summary

> owner：**"不要写 mmff_ele，而是用其他的 lammps 风格的 potential 组合，比如说 coul/long coul/cut 等等"**

`mmff-orthogonal-02` 把 MMFF 从 ForceField 层的特例变成了普通力场。本 spec 把同一条原则**再往下推一层**：**内核层也不该有 MMFF 专属内核**。

MMFF 的静电就是一个**带缓冲的库仑**。它不该拥有自己的内核，它该是**通用库仑内核的一次参数化**。

## Domain basis

### 两份实现，同一个物理，只差三个常数

| | `pair/coul/cut`（通用） | `pair/mmff_ele`（专属） |
|---|---|---|
| 能量式 | `k · qᵢqⱼ / r` | `k · qᵢqⱼ / (D · (r + δ))` |
| 库仑常数 k | `COULOMB_REAL` = **332.06371**（CODATA） | `COULOMB_MMFF` = **332.0716**（Halgren） |
| 缓冲 δ | — | **0.05 Å** |
| 介电 D | — | 1.0 |
| 电荷来源 | frame 上的逐原子 `charge` | 同 |
| type-row | 不读（`_type_params`） | 不读（`_tp`） |

`coul_cut.rs` 的模块注释自己就写着：*"this kernel is per-atom, **mirroring `mmff_ele`**"*。**它们本来就是近亲，而且写注释的人知道。**

### 库仑常数的差异是 load-bearing，不能"统一成一个"

`332.0716` vs `332.06371`，相对差 2.4e-5。咖啡因的 `E_ele = −150.48 kcal/mol`：

```
|ΔE| = 150.48 × 2.4e-5 = 0.0036 kcal/mol   >   1e-3（RDKit parity 容差）
```

⇒ **常数必须是 style 参数**。把它硬编码成"通用值"会让 MMFF 掉出 parity。这也是为什么"合并两个内核 = 二选一"是错的：**两个都对，各自的力场说了算**。

### 现有的后门（本 spec 一并堵死）

`mmff-orthogonal-02` 的 tester 在证明门禁"咬得住"时发现：把 style 改成 `def_pairstyle("mmff_ele", &[])`（**风格在，参数空**），**19 个能量测试里 18 个照样通过**。因为：

```rust
let dielectric = sp.get("dielectric").unwrap_or(1.0) as F;
let delta      = sp.get("delta").unwrap_or(ELE_DELTA) as F;
```

**力场悄悄不再是常数的来源，内核用起了自己硬编码的副本——算出对的数，但理由是错的。**

`coul/cut` 有同样的病：`style_params.get("coulomb14scale").unwrap_or(1.0)`。

⇒ 本 spec 的硬规则：**力场必须提供的参数，内核不得有 `unwrap_or` 兜底。缺参数是 `Err`，不是默认值。**
（区分：`cutoff` 缺省为 `INFINITY` 是**语义默认**——"不截断"是一个合法的、有意义的选择；而 `dielectric` 缺省为 1.0 是**在假装力场说了话**。前者留，后者删。）

## Design

### 一个通用的缓冲库仑内核

`pair/coul/cut` 泛化，参数全部来自 style：

```
E(r) = k · qᵢqⱼ / (D · (r + δ))      r < r_cut
```

| style 参数 | 语义 | 缺省 |
|---|---|---|
| `coulomb` | 库仑常数 k | **无缺省——必须由力场提供** |
| `delta` | 缓冲距离 δ (Å) | `0.0`（= 无缓冲，退化成教科书库仑）——这是**语义默认**，允许 |
| `dielectric` | 介电常数 D | **无缺省——必须由力场提供** |
| `cutoff` | 截断 (Å) | `INFINITY`（语义默认，允许） |
| `coulomb14scale` | 1-4 缩放 | **无缺省**（房规：由 `SpecialBonds` 投影而来，力场必须声明） |

`δ = 0` 时它**逐位退化**成今天的 `coul/cut`。MMFF 只是 `δ = 0.05, k = 332.0716, D = 1.0` 的一次参数化。

### 删除

- `pair/mmff_ele` 内核（`MMFFElectrostatic`、`mmff_ele_ctor`）及其注册
- `ff/constants.rs` 的 `COULOMB_MMFF` / `ELE_BUFFER` —— 它们搬进 `ff/params/mmff.rs` 的 `MMFF_ELE_STYLE`，因为它们是**MMFF 的参数**，不是 molrs 的常数
- 所有假装力场说过话的 `unwrap_or`

### `coul/long` 不在本期

LAMMPS 的 `coul/long` 是 kspace（Ewald/PME）的实空间伴生项。molrs 已有 `kspace/pme`，但**没有配套的 `pair/coul/long`**——这是一个真实的缺口，但它**与本 spec 正交**（MMFF 是非周期的分子力场，走 `coul/cut`）。记为 follow-up，不在本期塞进来。

## Files to create or modify

- `molrs/src/ff/potential/pair/coul_cut.rs` — 泛化（`delta` / `dielectric` / `coulomb`）；删掉假装默认的 `unwrap_or`
- `molrs/src/ff/potential/pair/mmff.rs` — **删除** `MMFFElectrostatic` + `mmff_ele_ctor`（`MMFFVdW` 保留：vdW 是真的 type-row 查表）
- `molrs/src/ff/potential/registry.rs` — 注销 `pair/mmff_ele`
- `molrs/src/ff/constants.rs` — 删 `COULOMB_MMFF` / `ELE_BUFFER`
- `molrs/src/ff/params/mmff.rs` — `MMFF_ELE_STYLE` 携带 k / δ / D / scale14
- `molrs/src/ff/typifier/mmff/embedded.rs` — MMFF 的 ForceField 改 def `pair/coul/cut`
- `molrs/tests/ff/mmff/energy.rs` — 门禁改盯 `pair/coul/cut`（**意图不变**：静电项必须由力场供数，不是内核硬编码）

## Tasks

- [ ] Write the failing tests: MMFF's electrostatics come from `pair/coul/cut`; `pair/mmff_ele` is not registered; a style missing `coulomb` / `dielectric` / `coulomb14scale` is an `Err`, not a silent default
- [ ] Generalize `pair/coul/cut` with `delta` / `dielectric` / `coulomb` style params; `delta = 0` reproduces today's kernel bit-for-bit
- [ ] Delete `MMFFElectrostatic` / `mmff_ele_ctor` / the `pair/mmff_ele` registration / `COULOMB_MMFF` / `ELE_BUFFER`
- [ ] Move MMFF's electrostatic constants into `MMFF_ELE_STYLE` in `ff/params/mmff.rs` — they are MMFF's parameters, not molrs's constants
- [ ] Remove every `unwrap_or` that fakes a force-field-supplied value (keep the ones that are genuine semantic defaults, e.g. `cutoff = INFINITY`)
- [ ] Wrap up chem-perceive-14's loose ends: `scripts/mmff_to_xml.py` is dead (it wrote the deleted XMLs); workspace-root `data/mmff94*.xml` are unreferenced; `CLAUDE.md:175` still claims MMFF params are `include_str!`'d from `molrs/data/mmff94.xml`
- [ ] Run the full gate set; 11/11 RDKit parity and the frozen per-style breakdown must hold BIT-FOR-BIT

## Testing strategy

**这是纯重构：一个数都不许动。** 11 个 fixture 的 RDKit parity（tol 1e-3）与七项冻结分解（tol 1e-6）逐位重放。为了让测试变绿而放宽容差，是失败，不是通过。

**反向门禁（本 spec 的核心）**
- `pair/mmff_ele` 在 registry 中 **0 命中**
- 一个缺 `coulomb` / `dielectric` / `coulomb14scale` 的 style **必须 `Err`**，不得静默取默认值 —— 这是那个"算出对的数，但理由是错的"后门的正面证明。**故意造一个空 params 的 style，看它报错**。
- `delta = 0` 时 `coul/cut` 与本 spec 之前的实现**逐位相同**（回归守卫：泛化不得改变既有行为）
- `COULOMB_MMFF` / `ELE_BUFFER` 在 `molrs/src` 中 **0 命中**（它们是 MMFF 的参数，该住在 MMFF 的表里）

## Out of scope

- **`pair/coul/long`**（PME 的实空间伴生项）—— 真实缺口，与本 spec 正交，另立。
- `coul/tt` / `thole` 的去重 —— 未查证是否重复，不在本期猜测。
- MMFF 的 vdW（`MMFFVdW`）—— 它**真的**按原子类型查表，是货真价实的 type-row style，保留。
