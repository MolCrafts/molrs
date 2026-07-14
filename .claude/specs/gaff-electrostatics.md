---
title: "GAFF 的力场没有静电项 —— 整条链对全部 37 个分子静默丢掉库仑能"
slug: gaff-electrostatics
status: approved
created: 2026-07-15
severity: HIGH
---

# GAFF 的力场没有静电项

## Summary

`chem-perceive-15-final-acceptance` 的整体验收抓到的第一个缺陷，**也是这条链一直在打的同一个洞，只是换了个力场**。

`gaff_forcefield` 构建的 style：

```
atom/full · pair/lj/cut · bond/harmonic · angle/harmonic · dihedral/periodic · improper/periodic
```

**没有任何库仑 style。**

所以整条链 `Perceive → AtdTypifier → BccModel → ForceField → Potentials` 对**全部 37 个分子**产出的能量里，**静电项是零，而且是静默的**——包括那些离子：

| 分子 | 净电荷 | Σ\|q\| |
|---|---|---|
| 乙酸根 | **−1** | 2.85 e |
| 甲铵 | **+1** | — |
| 咪唑鎓 | **+1** | — |

一个净电荷 −1 的离子，算出来的能量里**一点静电都没有**。

## Domain basis

### 它为什么能藏住

**每一段都被测过。组合从来没有被跑到能量。**

- 感知层有测试 ✓
- ATD 分型对着 antechamber oracle 37/37 ✓
- BCC 电荷对着 antechamber oracle 37/37 ✓
- GAFF 参数表对着 gaff.dat ✓
- parmchk2 估计对着 oracle ✓
- **`gaff_forcefield(...).to_potentials(frame).calc_energy_forces(...)` —— 从来没有任何测试跑过这一步**

这与 `mmff-orthogonal-01` 抓到的 150 kcal/mol 缺陷是**同一个形状**：通用 MMFF 路径也曾完全没有静电项，也曾在测试全绿的情况下活着。

### 证据一直摆在代码里

`molrs/src/ff/forcefield/gaff.rs:745`：

```rust
ff.set_special_bonds(SpecialBonds {
    lj:   [0.0, 0.0, AMBER_LJ_14],
    coul: [0.0, 0.0, AMBER_COUL_14],   // SCEE = 1.2 —— 一个 1-4 库仑缩放因子
});                                     // 而那个库仑项并不存在
```

`AMBER_COUL_14 = 1.0/1.2`（AMBER 的 SCEE）**被声明了，然后没有任何东西消费它**。

> **一个没人消费的常数，和 4,065 行没人读的 XML 参数，是同一种气味。**

### AMBER/GAFF 的静电形式

AMBER 的非键静电是**未缓冲的库仑**（不是 MMFF 那种缓冲形式）：

```
E = k · qᵢqⱼ / (D · r)          1-2 / 1-3 排除；1-4 缩放 1/1.2（SCEE）
```

⇒ 它正好是 `mmff-ele-compose` 刚泛化好的 `pair/coul/cut`，取 **δ = 0**（无缓冲）。

**不需要新内核。** 需要的只是**让力场把它声明出来**。

## Design

`build_forcefield`（`ff/forcefield/gaff.rs`）新增：

```rust
ff.def_pairstyle("coul/cut", &[
    ("coulomb",    COULOMB_REAL),       // AMBER 用 CODATA 值，不是 Halgren 的
    ("dielectric", VACUUM_DIELECTRIC),
    // delta 缺省 0 —— AMBER 的库仑不带缓冲。这是语义默认，合法。
]);
```

1-4 缩放已经通过 `special_bonds.coul[2] = 1/1.2` 走房规投影成 `coulomb14scale`——**那个常数终于有人消费了**。

### 库仑常数用哪个？

**必须实测决定，不许猜。** AMBER 的 `parm` 文件用的是 18.2223² = 332.0522…，而 CODATA 是 332.06371。这两个数**不一样**，而 `mmff-ele-compose` 已经证明了：**2.4e-5 的相对差在 −150 kcal/mol 的静电能上就是 0.0036 kcal/mol，高于容差**。

⇒ **实施的第一步是拿 antechamber/sander 跑一个已知分子，反解出它实际用的常数。** 不要从任何文档里抄。

## Files to create or modify

- `molrs/src/ff/forcefield/gaff.rs` — `build_forcefield` 声明 `pair/coul/cut`
- `molrs/tests/end_to_end.rs` — 那道红着的门禁（`the_force_field_the_chain_builds_declares_its_electrostatics`）转绿
- 新增：GAFF 链路的能量 oracle（见下）

## Tasks

- [ ] Determine AMBER's Coulomb constant EMPIRICALLY (run sander/antechamber on a known molecule and back it out) — do NOT copy it from a doc
- [ ] Declare `pair/coul/cut` in `gaff.rs`'s `build_forcefield` with that constant, `dielectric = 1`, `delta = 0`
- [ ] Add an ENERGY oracle for the GAFF chain — this is the real gap: nothing has ever run this chain to an energy. Anchor it on an external tool (sander single-point on the 37-molecule set, or a subset if sander is unavailable — but say which and why IN THE TEST)
- [ ] Assert the ions specifically: acetate (net −1), methylammonium (+1), imidazolium (+1). A neutral-only test would be the `["e_ethane"]` mistake a third time
- [ ] Assert `special_bonds.coul[2]` is actually CONSUMED — the 1-4 Coulomb scale must change the energy of a molecule with a 1-4 pair

## Testing strategy

**反向断言**：一个净电荷非零的分子，静电能**不得为 0**。这条断言在今天会红——那正是缺陷。

**外部 oracle 是硬要求。** 这条链上"没有 oracle 的测试"已经三次让缺陷活下来（BCC 键型感知、电荷等价化、通用 MMFF 静电）。断言自己算出来的东西，就是在给下一个洞挖坑。

**离子必须单独断言。** 中性分子上，静电项漏掉的影响比离子小得多；只测中性分子就是重犯 `["e_ethane"]` 的错误——**选一个不可能失败的输入，然后声称覆盖了**。

## Out of scope

- MMFF（已由 `mmff-ele-compose` 解决）
- 周期性体系的 `coul/long` / PME 实空间伴生项（真实缺口，另立）
