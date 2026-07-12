---
title: "chem-perceive 12/14 — molrs-cxxapi：bridge 返回 Result + parameter-set 选择器"
status: approved
created: 2026-07-12
chain: chem-perceive
depends_on: "chem-perceive-07-charge-trait"
blocks: ""
---

# cxx bridge：不再 abort 进程；打通 ABCG2

> Chain **chem-perceive** 12/14.
> > 说明：`.claude/notes/architecture-rules.md` 已过时（仍按单 crate 合并前的 8-crate workspace 描述），本 spec 不引用它；blueprint refresh deferred（`/mol:map` 需用户确认门）。


## Summary

`molrs-cxxapi/src/bridge.rs:39-43` 现在声明的是
`fn am1_bcc_assign_frame_from_base(…, am1_charges: &[f64], total_charge: f64, normalize_total_charge: bool) -> Vec<f64>`
——**没有 `Result`**，而函数体在 `lib.rs:1198` 直接 `.expect()`。

这意味着 Rust 一 panic 就**不会**变成可捕获的 C++ 异常，而是**直接 abort 进程**。
而它 panic 的那些错误根本不是程序员 bug，是**用户的化学**：缺 BCC 修正行、缺原子类型、缺键级。
Atomiverse 那边小心翼翼地 `throw std::runtime_error`，然后调用一个会 abort 的函数。

同时打通 ABCG2：molrs 有 `BccParameterSet::{Bcc,Abcg2}` + 两张表，但 bridge 硬编码
`AM1BCCTypifier::bcc(...)`；在 Atomiverse + molrs-cxxapi 全文搜 `abcg2` / `parameter_set` 是 **0 命中**。

## Domain basis

这是本链**唯一**伸出 molrs 仓库之外的一处。bridge 的签名变更是 ABI 变更，
Atomiverse 的 `AM1BCCChargeAssigner::assign` 必须同步更新。

顺带纠正一个架构错觉：Atomiverse 的 `AM1BCCChargeAssigner` 既不是 Driver 也不是 Component，
而 Atomiverse 自己的 CLAUDE.md 明写
*"Properties (charges, dipoles, Mulliken populations, **AM1-BCC charges**, …) are component
Capabilities … never appear at the Driver level"*。实现与文档不一致——
我倾向认为 facade 是**对的**（AM1-BCC 是一次性制备步骤，不是每步 `launch()`），
但那句 CLAUDE.md 该修。列为 Atomiverse 侧的 follow-up，不在本 spec 的验收内。

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

- 修改 `molrs/molrs-cxxapi/src/bridge.rs`（签名 → `Result`，加 selector，去 normalize 参数）
- 修改 `molrs/molrs-cxxapi/src/lib.rs`（删 `ProvidedAM1ChargeBackend` 与 `.expect()`）

## Tasks

- [ ] Change the bridge fn to return `Result<Vec<f64>>` so cxx surfaces a catchable `rust::Error`
- [ ] Drop the `normalize_total_charge` argument (deleted in 07)
- [ ] Add a parameter-set selector so ABCG2 is reachable from C++
- [ ] Delete `ProvidedAM1ChargeBackend` (molrs-cxxapi/src/lib.rs:1172) — the fake backend dies with the trait
- [ ] Declare the Atomiverse companion change (am1_bcc_charge_assigner.cpp + AM1BCCChargeConfig) as a cross-repo dependency

## Testing strategy

- 一个化学不受支持的分子（如含硼）经 bridge 调用 → C++ 侧能 `catch` 到 `rust::Error`，
  **进程不 abort**。这是本期的核心测试。
- ABCG2 经 selector 可达。
- grep gate：`molrs-cxxapi` 中 `normalize_total_charge` 0 命中。

## Out of scope

- **Atomiverse 侧的配套改动**（`src/cpu/semiempirical/am1_bcc_charge_assigner.cpp` 与
  `AM1BCCChargeConfig`）不在本仓库，列为 `blocks:` 的跨仓依赖。
- 等价化**不需要** Atomiverse 改动：它继续喂原始 Mulliken 是**对的**，
  平均是 molrs 的活（纯拓扑）。
