---
slug: release-0-14-05-compute-protocol
title: release-0-14-05-compute-protocol — Compute 作为 runtime_checkable Protocol
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-01-baseline
---

# release-0-14-05-compute-protocol — 唯一 Compute 契约

## Summary

把 molrs 暴露给下游的 `Compute` 定为 **`@runtime_checkable` Protocol**，契约只有一条：`compute(...)`。这让非 object-safe 的 Rust `Compute` trait 不再构成障碍，也免掉给 45 个 pyclass 内核做 `extends=` 手术的成本——它们**一行不改**就已满足契约。

*（slug 由 `release-0-14-05-compute-base` 改名而来——"base" 已与内容不符。）*

## Domain basis

N/A — 契约层，不含物理。各分析内核的科学正确性由既有内核测试守；本规范不动内核。

## Design

**object-safety 的死结被 Protocol 彻底解开。** `molrs::compute::Compute` trait（`molrs/src/compute/traits.rs:49`）带 GAT 与泛型 FA 方法，**非 object-safe**，做不成 `Box<dyn Compute>`，也就做不成 `#[pyclass]` 基类。名义基类方案在此是**被类型系统否决**的；Protocol 是结构化的，不需要任何可对象化的 Rust 载体，死结自然消失。R1 的裁决在这里比在 Potential 处更直接。

**契约形状（只有一条方法）。** `molrs-python/python/molrs/compute/protocol.py`（new）：

```python
@runtime_checkable
class Compute(Protocol):
    def compute(self, *args, **kwargs): ...
```

- canonical 动词是 `compute`——依据是既有事实：molrs-python 约 45 个 pyclass 内核**全部**拼 `compute`，`__call__` / `dump` 命中为零。契约迁就已落地的实现，而不是让 45 个内核改名。
- **`__call__` 别名与 `dump()` 不是 molrs 契约的一部分。** molpy 基类拼 `__call__`（`<molpy>/src/molpy/compute/base.py:21`）并提供 `dump()`；这两者的去留**绑定到 molpy 薄壳的开放项**（见 Open questions 与 10 号规范）——若薄壳保留，它们由 molpy 侧自行提供；若薄壳塌缩为直接使用 molrs 内核，它们随薄壳一并消失。molrs 不预先承诺自己不需要的成员。
- docstring 必须写明 `runtime_checkable` 只检查方法存在、不检查签名（PEP 544），且不适合放在热路径。
- **不加** `run_all` / `pipeline` 门面；组合是调用者的事。

**覆盖哪一段流水线。** molrs 是四段（Compute → Fit → Check/Verdict），molpy 是扁平 Compute。0.14 的契约**只覆盖 Compute 段**；Fit / Check 不进契约，避免把未收敛的分段固化成公开承诺。`ComputeResult` / `DescriptorRow`（`molrs/src/compute/result.rs:28,40`）原样保留，不包装、不镜像。

**Protocol 也需要一个家，例外照记。** `molrs-python/python/molrs/compute/__init__.py:3-4` 声明"one subpackage per `molrs::compute` module"，`protocol.py` 打破它（不对应任何 Rust module）。按该 docstring 自己的先例（:9-15 已就地记录 `rdf` / `shape` 两个例外）**就地补记**——例外写在不变式旁边，不散落在提交信息里。纯 Python 文件随 molrs wheel 发布的先例已在树上：`molrs-python/python/molrs/ff/potential/soft.py`。

### Reuse decision

- `reuse` `ComputeResult` / `DescriptorRow`（`compute/result.rs:28,40`）原样。
- `reuse` `molrs-python/python/molrs/views.py:26 RefLike(Protocol)` 作为本仓 Protocol 风格范本。
- `reuse` molpy `@runtime_checkable` 先例（`TypifierProtocol` `builder/polymer/core.py:79-80`、`RegionTypifier` `typifier/region.py:35-36`、`PotentialLike` `optimize/base.py:20`）保持跨仓一致。
- `reuse` 既有 45 个 pyclass 内核的 `compute` 拼写作为 canonical 动词的**事实依据**。
- `reuse` `ff/potential/soft.py` 作为"纯 Python 文件进 molrs wheel"的先例；`compute/__init__.py:9-15` 作为例外记录格式。
- `generalize` molpy `compute/base.py:21` 的抽象调用契约 → 提升为 molrs 侧唯一结构化契约（`__init__(**config)` / `__call__` / `dump()` 三者中，只有"可调用的分析动作"被提升，其余留给 molpy 侧的薄壳裁定）。
- **considered and dropped**：给 45 个 pyclass 加 `extends=`（基类不持状态，45 个二元组构造器只为一次 `isinstance`）；ABC 名义基类（R1 裁定 Protocol 优先，且 object-safety 本就否决了 Rust 侧基类）。
- `new` — `molrs-python/python/molrs/compute/protocol.py`。

## Files to create or modify

- `molrs-python/python/molrs/compute/protocol.py` (new)
- `molrs-python/python/molrs/compute/__init__.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_compute_protocol.py` (new)
- `regressions/release-0-14-05-compute-protocol.py` (new)

## Tasks

- [ ] Write failing structural-typing tests: a class defining only `compute` satisfies `isinstance(obj, Compute)`, and the existing pyclass kernels do too without modification (`molrs-python/tests/test_compute_protocol.py`, `TestComputeProtocol`)
- [ ] Write failing contract-scope test asserting `Compute` declares only `compute` — no `__call__` and no `dump` member on the molrs contract
- [ ] Implement the `runtime_checkable` `Compute` Protocol in `molrs-python/python/molrs/compute/protocol.py`, documenting the PEP 544 presence-only semantics
- [ ] Export `Compute` from `molrs-python/python/molrs/compute/__init__.py` and record the layout-invariant exception inline in its module docstring
- [ ] Add `Compute` to the `molrs-python/python/molrs/_lib.pyi` type surface where the analysis kernels are declared
- [ ] Add regression example `regressions/release-0-14-05-compute-protocol.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

- 结构化（happy path）：只定义 `compute` 的普通类 `isinstance` 为真；**扫描** `molrs.compute` 下的内核类断言它们未经修改即满足契约（用目录/属性扫描，不写死名单——手写名单会被手工缩短）。
- 契约边界：`Compute` 的成员集合恰为 `{compute}`；断言 `hasattr(Compute, "__call__")` 不构成契约成员、`dump` 不在协议成员中——防止有人把 molpy 的形状悄悄塞回 molrs。
- 边界（`runtime_checkable` 弱点）：签名不符但方法名相符的对象仍通过 `isinstance`（钉死 PEP 544 语义），实际调用时给出清晰错误。
- 布局门：`molrs.compute.__doc__` 中出现 `protocol` 例外说明（例外与不变式同处）。
- 回归样例 `regressions/release-0-14-05-compute-protocol.py`：用公开 API 定义一个**不继承任何东西**、只实现 `compute` 的 RDF 薄壳，跑一个 8 原子固定构型，断言首个非零 bin 与写死黄金值相等。
- 域验证：不适用（契约层无物理）。

## Open questions (maintainer ruling required)

1. **molpy 24 个薄壳保留还是塌缩？（决定 `__call__` / `dump()` 的归属）** 保留 → 两者由 molpy 侧薄壳自行提供，molrs 契约不变；塌缩为直接使用 molrs 内核 → 两者随薄壳一并删除，同样不影响 molrs 契约。无论哪种裁定，molrs 侧都不承诺这两个成员——本规范据此落地，10 号规范执行裁定。*（grill 已裁定：保留。）*
2. **契约是否在 0.15 覆盖 Fit / Check 段？** 0.14 只覆盖 Compute 段。

## Out of scope

- molpy 侧 24 个薄壳的改造（10）
- `__call__` / `dump()` 的实现（不属 molrs 契约）
- Fit / Check / Verdict 段（0.15）
- 任何分析内核的算法改动
