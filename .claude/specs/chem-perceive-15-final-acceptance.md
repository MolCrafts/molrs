---
title: "chem-perceive 15/15 — 整体验收：这条链到底收敛了没有"
slug: chem-perceive-15-final-acceptance
status: in-progress
created: 2026-07-14
chain: chem-perceive
depends_on: "chem-perceive-14-all-tables"
blocks: ""
---

# 整体验收 — 这条链到底收敛了没有

## Summary

这条链跑了 16 个 spec（chem-perceive 01–14、mmff-typifier-split、mmff-orthogonal 01–02）。**每个 spec 只验证自己那一块。没有任何一个验证过整体。**

本 spec 只做一件事：**端到端地证明整个系统真的收敛到了它声称的形状**，并把每一条"只此一份"的架构承诺变成**机器可查的门禁**——否则它们会以完全相同的方式重新长出来。这不是走过场：这条链上被抓到的**每一个**严重缺陷，都是"局部全绿、整体撒谎"。

## Domain basis：这条链抓到过的东西

本 spec 的每一条门禁，都对应一个**真实发生过**的失败。它们不是假想的风险。

| 缺陷 | 为什么局部测试没抓到 |
|---|---|
| 通用 MMFF 路径**完全没有静电项**（咖啡因差 150 kcal/mol） | 唯一被断言的 fixture 是乙烷——**仅有的两个电荷全为零的分子之一**，即唯一一类结构上不可能暴露它的输入 |
| BCC 键型感知、电荷等价化**两个算法阶段整个缺失** | 没有 oracle。测试断言的是代码自己算出来的东西 |
| 乙酸的两个羧基氧差 **0.2014 e** | 对称性从没被断言 |
| 硝酸根三个等价氧被判成 6/9/9 | 同上 |
| `MMFF94STypifier` 可以**静默吐出 MMFF94 的势函数** | 只修了两条参数路径中的一条；测试只查原子类型，而原子类型两个变体本来就相同 |
| cxx bridge 让用户的化学错误**直接 abort 进程** | cxx 给非 `Result` 函数生成的 C++ 声明带 `noexcept`——catch **结构上不可达** |
| `keys::TYPE` 键碰撞导致写入**静默失败** | 被 `let _ =` 吞掉 |
| `linear` 列若烘焙成 `bool` 会被 **`to_frame` 静默丢弃** | 每个角都读成"非线性"，缺陷会假装修好了 |
| 4,065 行**没有任何代码读**的 XML 参数 | 它们活着只为骗过一个 `is_empty()` 守卫 |
| `pme_ctor` / `pair_coul_cut_ctor` 也忽略 `tp` | 我的 grep 找的是**拼写**（`_tp`），它们拼的是 `_type_params` |
| `classify.rs` 把苯的芳香键判成 type 1（RDKit 说 0，**反的**） | 测试只断言"两扇门彼此一致"——**两扇门一致地错着** |

**共同的病：测试选了一个不可能失败的输入，然后声称覆盖了。** 本 spec 的门禁全部是**反向的**——它们断言的不是"某个东西存在"，而是"**某个东西不存在**"。

## Design

### 五条"只此一份"的承诺，逐条钉死

这条链的全部架构主张可以压缩成五句话。每一句都必须有一道自己不能豁免自己的门禁。

**1. 参数只有一个地方、一种形式。**
- `molrs/src/ff/params/` 是**唯一**的参数表所在地，平铺，无子目录。
- `molrs/data/` 不存在。`grep -rn 'include_str!' molrs/src` = 0。
- **`generated` / `generator` 不出现在任何标识符里**（目录/文件/模块/测试函数/类型名）。出处写在文件头文档注释里——**用来源给东西命名，等于把实现细节焊死在公共表面上**。
- 运行期不解析任何参数文本：无 `serde_json::from_*`、无 XML 解析、无 `include_str!`。

**2. 感知只有一个层。** `molrs::perceive` 在 `core` 之上、`ff`/`io`/`conformer` 之下。`molrs::chem` 别名 0 命中（含所有兄弟 workspace）。

**3. 插值只有一个 seam。** `ParameterInterpolator`（trait）+ `Parmchk2Estimator`（唯一实现）。没有第二套估计器栈。换力场是换 `TypifierParameterContext`，不是换实现。

**4. MMFF 只有一条路。** 没有 bespoke 能量层、没有 `build_mmff_potentials` 自由函数、没有第二份分类器。`MMFF94Typifier().typify()` 给标签与电荷，势函数走标准 ForceField 路线。

**5. 忽略 `tp` 的内核构造器不是 Style。** `ParamSource::PerInstance` 是一等概念，双向门禁盯着（按**语义**，不按拼写——这条一落地就多抓到两个我 grep 漏掉的）。

### 端到端的真值链，必须一路走通

不是分段绿，是**一条链从头绿到尾**：

```
SMILES / SDF
   → Perceive（环、芳香性、键型、等价类）
   → AtdTypifier（七套表之一）
   → ChargeModel（BCC / ABCG2 / Gasteiger / Mulliken）
   → ForceField（GAFF / GAFF2 / MMFF94 / MMFF94s / OPLS）
   → Potentials → 能量 + 力
```

**每一段都对着真实的外部 oracle**：37 分子的 antechamber oracle（原子类型 ×7 套、BCC/ABCG2/Gasteiger 电荷、parmchk2 估计项），11 分子的 RDKit MMFF oracle（总能量 + 七项分解）。

**Python 侧必须逐位复现 Rust 侧**——不是"差不多"，是同一个数。

### 反向门禁（本 spec 的核心）

正向断言（"X 存在"）会被"加了但加错地方"骗过去。本 spec 的门禁全部是反向的：

- 零电荷分子的静电能量必须**恰好为 0**（不是"很小"）
- 无芳香氮的分子上，MMFF94 与 MMFF94s 必须**逐位相同**（否则"两者不同"的测试可能靠一个根本不存在的差异侥幸通过）
- 苯必须**有** improper（这条链之前它有零个，静默）
- 硝酸根的三个氧、乙酸的两个氧必须**电荷相同**
- 同一分子的不同构象必须给出**相同**电荷（等价化的意义）
- 任何"只断言子集"的测试，**必须在测试里写明为什么排除了其余的**——"还没实现"不是排除理由，是失败的理由

### 不做的事

- **不新增功能。** 本 spec 一行生产代码都不写（除非门禁揭发了真实缺陷——那时停下来立 spec，不要在验收里偷偷修）。
- **不放宽任何容差。** 为了让整体验收变绿而动一个数，是失败，不是通过。

## Files to create or modify

- `molrs/tests/architecture_gate.rs` (new) — 五条承诺的门禁，自己不能豁免自己
- `molrs/tests/end_to_end.rs` (new) — 全链路 oracle 对账（antechamber 37 + RDKit 11）
- `molrs-python/tests/test_parity.py` (new) — Python 逐位复现 Rust
- git log / tags — 这条链的完整 BREAKING 面（不再维护手写 CHANGELOG）
- `.claude/notes/notes.md` — 把这条链学到的**教训**（不是结论）写下来

## Tasks

- [ ] Write the five architecture gates (one place / one perceive layer / one interpolation seam / one MMFF path / no tp-ignoring Style), each self-non-exempting
- [ ] Write the `generated`-purge gate: zero `generated` / `generator` in any identifier under `molrs/src` and `molrs/tests`
- [ ] Write the end-to-end oracle test: SMILES -> Perceive -> AtdTypifier -> ChargeModel -> ForceField -> Potentials, against the antechamber (37) and RDKit MMFF (11) oracles
- [ ] Write the Python-vs-Rust bit-parity test (not "close" — the same number)
- [ ] Write the reverse gates (exactly-zero electrostatics on neutral molecules; bit-identical 94/94s without aromatic N; benzene HAS impropers; symmetric charges on nitrate/acetate; conformer-independent charges)
- [ ] Write the "no subset assertion without a stated reason" gate over the test tree
- [ ] Record LESSONS (not the conclusions) in .claude/notes/notes.md; BREAKING surface is git history, not a hand-written CHANGELOG
- [ ] Run every gate; any failure STOPS and gets its own spec — do not fix it inside the acceptance

## Testing strategy

本 spec **就是**测试。它没有"测试策略"，它有**验收本身**。

唯一的元规则：**每一道门禁必须先被证明会红。** 对每一条，临时制造它要防的那个缺陷，看它变红，再撤掉。一道从没红过的门禁，和没有门禁没有区别——这条链上已经有过一道（`bespoke_gate`）和一个 grep 判据（`_tp` 拼写）证明了这一点。

## Out of scope

- 性能优化
- 任何新功能
- molpack / molpy 的下游迁移（另仓）
