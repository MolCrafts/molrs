---
title: "chem-perceive 7/14 — ChargeModel trait + BccModel push API + Mulliken（删掉假 backend）"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-03-bondtype; chem-perceive-04-equivalence; chem-perceive-05-atd-typifier"
blocks: "chem-perceive-08-gasteiger; chem-perceive-12-cxx-bridge; chem-perceive-13-python-bind"
---

# ChargeModel：2×2 泛化证明 + push API

> Chain **chem-perceive** 7/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

重塑电荷 API。现在的 `AM1ChargeBackend` 是个**拉模型** trait，但全 workspace 三个实现
**没有一个真的从分子算 AM1**，`mol` 参数在每个实现里都未使用：`UnavailableAM1Backend`（永远报错）、
`ProvidedAM1ChargeBackend`（`molrs-cxxapi/src/lib.rs:1172`，忽略 `_mol`，返回预先算好的 Vec）、
`FakeAM1Backend`（测试）。生产路径要**伪造一个假后端**才能把一个 `Vec<f64>` 偷渡进去——接缝装反了。

换成 **push API**：`correct(&mol, &am1_charges) -> Result<Vec<f64>>`，纯函数，无 trait 对象，不改分子。

同时用一个 `ChargeModel` trait 托起 **2×2 泛化证明**：

| 模型 | 需要 QM 输入？ | 有拓扑修正？ | `needs_equivalencing` 默认 |
|---|---|---|---|
| Mulliken | 是 | 否（直通） | 0（关） |
| AM1-BCC / ABCG2 | 是 | 是（键增量） | 1（开） |
| Gasteiger/PEOE（08） | **否** | 是（迭代均衡） | 0（关） |

如果一个 trait 能同时托起这三个而没有任何一个是特例，就证明它没有偷偷假设"QM 基电荷 + 修正"。

## Domain basis

**A5 — 符号约定：molrs 现在就是对的，不要"修"。** BCC 增量是加到 **BCC 原子类型编号较小**的那个原子上
（`am1bcc.c` 按 bcctype 排序，**不是**按键的端点顺序）；`BCCPARM.DAT` 只存 r≤s 方向，反向取负。
molrs 的 `BCCCorrectionTable::direct_correction` 已经正好这么做，而
`bcc_corrections_match_antechamber_given_reference_types` **37/37 全绿**就是证明。
这是一个经典的静默符号翻转陷阱：若改成按端点顺序查表，会翻转约一半增量的符号，
**而且总电荷仍然守恒**，结果看起来对称、实则完全错误。

**A6 — 不做归一化。** `am1bcc.c::charge()` 在增量循环处就结束，之后没有任何 rescale / shift / round。
antechamber 把 AM1 的 3 位小数舍入残差**原样带到最终电荷**（实测残差最大 0.004 e）。
因此 molrs 现有的 `normalize_total_charge`（把 `(target−sum)/N` 均摊到每个原子）不但没用，
还会让 molrs **偏离** antechamber；更糟的是它会**掩盖 bug**——若 AM1 不收敛、在中性分子上求和为 +0.7，
它会静默地给每个原子减去一点，交回一份看起来合理的垃圾。这违反 molrs 自己的 "no fallback values" 规则。

**BCC 修正引擎本身是正确的，不要重写。** L3 测试
（`bcc_corrections_match_antechamber_given_reference_types`）在喂入 antechamber 自己的类型时
**37/37 全过**——查表（含 `CORR` 别名、反向取负）与应用算术都对。本期只换外壳。

**`keys::TYPE` 污染必须终结。** 现在 `BCCAtomTypifier::typify` 把 BCC 码（`"11"`/`"91"`）写进
`atoms.type`，而 `ff/charge.rs:52` 的 `ChargeAssigner::assign_atomistic` 执行 `*mol = typed`——
于是"文档推荐的门"会**静默地把用户的 GAFF/OPLS 力场原子类型替换成 BCC 码**。
而标准 AM1-BCC 工作流恰恰需要两者**同时存在**（GAFF 管 LJ/bonded，BCC 管电荷）。
C++ 路径至今没被咬到，纯粹因为它在一个用完即弃的 `Atomistic` 上跑、只把 charge 列拷回去——
那是在绕过一个坏契约。键类型同理：绝不能把已有的 LAMMPS 键类型 id 重新解释成 BCC 码。

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

- 新增 `molrs/src/ff/charge/{mod,model,bcc,mulliken,error}.rs`
- 修改 `molrs/src/ff/charge.rs` → 拆成上面的模块；删除 `ChargeAssigner` 的 `*mol = typed` 语义
- 修改 `molrs/src/ff/typifier/am1bcc.rs` → 删除 `AM1ChargeBackend` / `UnavailableAM1Backend` / `AM1BCCTypifier` / `AM1ChargeTypifier` / `normalize_total_charge`
- 修改 `molrs/tests/ff/typifier/am1bcc.rs`（老测试里的 `FakeAM1Backend` 随 trait 一起删）

## Tasks

- [ ] Define the `ChargeModel` trait with a per-model `needs_equivalencing` declaration
- [ ] Implement `BccModel::correct(&mol, &am1) -> Result<Vec<f64>>` (push API, pure, no mutation)
- [ ] Implement `MullikenModel` (QM in, pass-through, no topology correction)
- [ ] Wire ABCG2 as a second parameter set through the same model
- [ ] Delete `AM1ChargeBackend`, `UnavailableAM1Backend`, `AM1BCCTypifier<B>`, `AM1ChargeTypifier<B>`, `normalize_total_charge`
- [ ] Ensure no charge model ever writes `keys::TYPE` (atom or bond)
- [ ] Add a typed `BccError` so C++/Python can discriminate missing-parameter from malformed-input

## Testing strategy

- `am1bcc_charges_match_antechamber_end_to_end` → 37/37 @1e-4（当前 4/37 挂）。
- ABCG2 端到端 37/37 vs `antechamber -c abcg2`。
- Mulliken 直通 == oracle 的 `am1_charges`，精确相等。
- **总电荷**：`Σq_final − Σq_after_equivalencing == 0` **精确**成立；
  `|Σq_final − net_charge| ≤ 0.005` 作为**容差断言，绝不是修正**。
- grep gate：`AM1ChargeBackend|UnavailableAM1Backend|normalize_total_charge` 在 `molrs/src` 中 0 命中。
- **类型不被污染**：一个已经带 GAFF 原子类型的分子，跑完电荷赋值后 `keys::TYPE` 原封不动。

## Out of scope

- Gasteiger（08）。
- cxx bridge（12）与 Python 绑定（13）。
- RESP / ESP / CM1 / CM2：依赖外部 QM（Gaussian/MOPAC ESP），本链不做（用户明确排除）。
