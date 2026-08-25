---
slug: release-0-14-03-potential-protocol
title: release-0-14-03-potential-protocol — Potential 是 runtime_checkable Protocol，分派保持鸭子类型
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-02-units-purge
---

# release-0-14-03-potential-protocol — 结构化的唯一 Potential

## Summary

把导出的 `Potential` 定为 **`@runtime_checkable` Protocol**，契约是 `calc_energy_forces(pos) -> (energy, forces)`：`LJCut`、`Potentials`、用户自定义类**在结构上**满足它（`isinstance` 经 `runtime_checkable` 生效），显式继承仍然可行但可选。PyO3 分派保持鸭子类型——具体 Rust 快路径臂在前，鸭子类型回退臂在最后。"接口有且仅有 Potential" 依旧绑定每一处签名与 `.pyi` 注解：Protocol 就是那唯一的类型，Union 依旧禁止。

*（slug 由 `release-0-14-03-potential-base` 改名而来——"base" 已与内容不符。）*

## Domain basis

N/A — 类型契约改造。数值路径不变；唯一与数值相关的风险是分派塌陷（具体类型误入 Python 回退臂 → 性能崩塌 + move 语义丢失），由分派测试而非物理断言把关。

## Design

**Rust trait 层：no-op。** `impl Potential for LJCut`（`molrs/src/md/lj_cut.rs:325`）、`for Potentials`（`molrs/src/ff/potential/mod.rs:297`）、`for Box<dyn Potential>`（`:152`）已在树上。本规范不重新设计，只补一条不变式测试防回归。

**唯一身份的裁决（本规范必须一次说清）：** 树上现有的 PyO3 可继承基类 `PyPotential`（`molrs-python/src/md.rs:356`）**从导出面退役**——它不再以任何 Python 名字对外可见。导出的 `Potential` 只有一个，就是 Protocol。其 catch-all `#[new]`、`SubclassPotential` 适配器与 `ErrSlot` 异常中继**保留为内部实现**，成为鸭子类型回退臂的机制。这样"显式基类"与"Protocol"不会形成两个互相竞争的 `Potential` 身份：想显式表意的用户直接继承 Protocol（Protocol 可被继承），想极简的用户什么都不继承。

**Protocol 归位。** `molrs-python/python/molrs/ff/potential/protocol.py`（new），由 `molrs.ff` 与 `molrs.md` re-export 为**同一对象**。位置理由沿用层序规则：Rust 侧 `md` 从 `ff` re-export，力的接缝概念归 `ff::potential`，Python 侧镜像之。

```python
@runtime_checkable
class Potential(Protocol):
    def calc_energy_forces(self, pos): ...
```

- **结构即契约**：任何具备该方法的对象 `isinstance` 为真，无需继承。
- **显式继承可选**：Protocol 类体可带默认实现供显式子类继承（`calc_energy_forces` 默认 `raise NotImplementedError`）；纯结构实现者不受影响。
- **已知边界必须写进 docstring**：`runtime_checkable` 的 `isinstance` **只检查方法是否存在，不检查签名**（PEP 544），且不适合放在热路径。因此 `isinstance` 是友好的类型注解手段，**不是**分派正确性的依据——快路径靠具体 Rust 类型抽取。

**Python 契约只有一条方法，Rust 侧的每步 pair 数据接缝不改变这一点（与 14 号的边界）。** 14 号规范用**每步 pair 数据直通**替掉 `set_pairs` 快照：循环侧的邻居机制每步算一次 `(indices, disp, dist_sq)`，直接交给所有 pair 势共享，势内部不再存快照、不再做 MIC。那是 **Rust 侧的接缝**，机制细节全部归 14；对本规范只有一条影响，必须写死在这里：**导出的 Python `Potential` 契约仍然只有 `calc_energy_forces(pos)`**。鸭子类型适配器 `SubclassPotential`（`md.rs:326`）今天就不实现 `set_pairs`（走 trait 默认 no-op），14 之后同样不实现新的 pair 数据方法——Python 自定义力（NN / Torch 接缝）只收到坐标，自己的邻居正确性自己负责。这条边界现在就可测（自定义力的 `calc_energy_forces` 只收到一个位置参数），并在 14 落地后保持为真。

**PyO3 分派：鸭子类型，臂序是硬要求。** `take_potential`（`molrs-python/src/md.rs:376`）改为：

1. `PyLJCut` 抽取（具体 Rust 快路径）
2. `PyPotentials` 抽取（具体 Rust 快路径）
3. **最后**：鸭子类型回退——对象具备可调用 `calc_energy_forces` 则包成 `SubclassPotential`（实例共享引用、GIL 下调用 override、异常经 `ErrSlot` 原样上抛），否则报错并在消息中点名所需方法。

臂序是最容易被静默破坏的不变式：把回退臂提前，每个 `Potentials` 都会被包成 Python 分派对象——性能崩塌 + 丢失 move 语义，而所有功能测试**仍然全绿**。因此必须有一条**会咬**的分派测试（按错误臂序应变红，验证后还原）。

**被裁定为不必要而放弃的方案：** 把 `PyPotential` 搬到 `ff` 并给 `PyLJCut` / `PyPotentials` 加 `extends=` 与二元组 `#[new]`。R1 裁定 Protocol 优先于名义基类，这笔手术的成本不必付。记录于此以免后续重新提案。

**签名塌缩。** `_lib.pyi` 三处 Union（:1315、:2366、:2397）塌缩为单一名字 `md.Potential`。全仓禁止 `Union[LJCut, Potential, Potentials]` 形状。

**导出补齐。** `molrs.md` 补出 `Potentials`（`md/__init__.py:78-118` 现无），否则"只有 Potential"在 md 命名空间里拼不出来。

### Reuse decision

- `reuse` Rust `ff::potential::Potential` trait 与三处 `impl` —— 实现层 **no-op**，只加不变式测试。
- `reuse` `molrs-python/python/molrs/views.py:26 RefLike(Protocol)` 作为本仓 Protocol 书写风格范本（命名、docstring、作为类型界的用法 `Refs[R: RefLike]` :250）。
- `reuse` molpy 的 Protocol 先例——`PotentialLike`（`<molpy>/src/molpy/optimize/base.py:20`）以及两处 `@runtime_checkable`：`TypifierProtocol`（`builder/polymer/core.py:79-80`）、`RegionTypifier`（`typifier/region.py:35-36`）——保持跨仓一致的结构化类型约定。
- `generalize` `PyPotential` 的 `SubclassPotential` / `ErrSlot` 机制（`molrs-python/src/md.rs:356`）：从"基类实例的分派"扩展为"任意鸭子类型对象的分派"，同时该 pyclass 退出导出面。
- `generalize` `take_potential`（`md.rs:376`）：单一分派点覆盖具体快路径与鸭子回退两类调用方。
- **considered and dropped**：`PyGraph → PyAtomistic` 的 `extends=` 范式（`molrs-python/src/core/system/molgraph.rs:481-545`）——R1 使其不再必要。
- `new` — `molrs-python/python/molrs/ff/potential/protocol.py`：本仓尚无力接缝的结构化类型声明。

## Files to create or modify

- `molrs-python/python/molrs/ff/potential/protocol.py` (new)
- `molrs-python/python/molrs/ff/potential/__init__.py`
- `molrs-python/python/molrs/ff/__init__.py`
- `molrs-python/python/molrs/md/__init__.py`
- `molrs-python/src/md.rs`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_md.py`
- `molrs-python/tests/test_subclass.py`
- `molrs/src/ff/potential/mod.rs` (invariant test only)
- `regressions/release-0-14-03-potential-protocol.py` (new)

## Tasks

- [ ] Write failing structural-typing tests: a class defining only `calc_energy_forces` and inheriting nothing satisfies `isinstance(obj, md.Potential)`, and `LJCut` / `Potentials` do too (`molrs-python/tests/test_md.py`, `TestPotentialProtocol`)
- [ ] Write failing single-identity test: `molrs.md.Potential is molrs.ff.Potential`, it is a runtime_checkable Protocol, and no PyO3 class is exported under any `Potential` name (`molrs-python/tests/test_subclass.py`)
- [ ] Write failing dispatch-order test proving a `Potentials` handed to `VelocityVerlet` takes the native fast path, not the duck-typed fallback (`molrs-python/tests/test_md.py`)
- [ ] Write failing boundary test that a duck-typed potential's `calc_energy_forces` is called with coordinates only — no neighbour/pair argument ever reaches Python (`molrs-python/tests/test_subclass.py`)
- [ ] Write failing Rust invariant test that `LJCut`, `Potentials` and `Box<dyn Potential>` all implement `Potential` (`molrs/src/ff/potential/mod.rs` `#[cfg(test)]`)
- [ ] Implement the `runtime_checkable` `Potential` Protocol in `molrs-python/python/molrs/ff/potential/protocol.py`, documenting the PEP 544 presence-only semantics, its unsuitability for hot paths, and that the contract is coordinates-only
- [ ] Retire the exported `PyPotential` class in `molrs-python/src/md.rs`, keeping its `SubclassPotential` / `ErrSlot` machinery as the internal duck-typed adapter, and re-export the Protocol as `molrs.ff.Potential` / `molrs.md.Potential`
- [ ] Reorder `take_potential` to concrete Rust arms first and the duck-typed fallback last, and prove the dispatch test bites under the wrong order
- [ ] Export `Potentials` from `molrs-python/python/molrs/md/__init__.py` and collapse the 3 `.pyi` Union sites to `md.Potential`
- [ ] Add regression example `regressions/release-0-14-03-potential-protocol.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Rust 不变式测试就地 `#[cfg(test)]`；Python 单测平铺，每条只测一个不变式。

- 结构化类型（happy path）：只定义 `calc_energy_forces` 的普通类实例 `isinstance` 为真且可直接交给积分器；`LJCut`、`Potentials` 同样为真；显式继承 Protocol 的子类同样可用。
- 唯一身份：`md.Potential is ff.Potential`；`isinstance(md.Potential, type)` 下它是 Protocol 而非 pyclass；`molrs._lib.md` 中不存在任何导出的 `Potential` 类。
- **分派（本规范最重要的一条）**：`Potentials` 交给 `VelocityVerlet` 后原对象被 move 消费（再次使用 raise `ValueError`），证明走的是具体臂。**bite-proof**：把回退臂提前，该测试应变红，验证后还原。
- 契约边界（对 14 号的接缝）：自定义力的 `calc_energy_forces` 被调用时**只收到坐标**——记录实参个数与形状，断言没有邻表/pair 参数越过 FFI 进入 Python；该断言在 14 号把 Rust 侧接缝换成每步 pair 数据直通之后必须仍然为真。
- 边界（`runtime_checkable` 的已知弱点）：签名不符但方法名相符的对象**仍**通过 `isinstance`（钉死 PEP 544 语义），并断言它在实际调用时给出清晰的 TypeError——防止把 `isinstance` 误当作正确性保证。
- 异常穿透：自定义力的 `calc_energy_forces` 抛出的异常类型与消息原样到达 `advance_n` 调用点（`ErrSlot` 行为加锁）。
- 缺席门：`.pyi` 全文中 `Union[` 与 `Potential` 不同现。
- 回归样例 `regressions/release-0-14-03-potential-protocol.py`：定义一个**不继承任何东西**的谐振子对象，与 `LJCut` 一起 push 进 `Potentials` 跑 5 步，断言总能量与写死黄金值（解析解）在 1e-12 内——同时证明"结构足够"。
- 域验证：不适用（无新物理）。

## Open questions (maintainer ruling required)

- 无。

## Out of scope

- Rust trait 层重设计（已满足）
- Rust 侧每步 pair 数据接缝的机制（14；本规范只钉住"Python 契约仍是坐标单参"这条边界）
- `extends=` 基类手术（R1 裁定不必要）
- 驱动形状与 `dtype=`（04）
- wasm / capi 的 Potential 对称面（0.15）
