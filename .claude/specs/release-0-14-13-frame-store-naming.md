---
slug: release-0-14-13-frame-store-naming
title: release-0-14-13-frame-store-naming — 公开面按对象命名：MolRec / zarr 退出，内部机制不动
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-05-compute-protocol
---

# release-0-14-13-frame-store-naming — 名字说的是对象，不是技术

## Summary

把 `MolRec` 与 `zarr` 从公开面上清掉：公开 API 一律以它作用的**对象**命名（record / frame），不以实现它的**技术**命名——后端尚未裁定（"write_frame 就是用 molrec 的定义写 frame，后端未定"），名字不能替维护者先把这个决定做了。`molrs.MolRec` → `molrs.Record`，`read_zarr` / `write_zarr` → `read` / `write`（record 与 trajectory 两处），cxxapi `write_frame_zarr` / `read_frame_zarr_first` → `write_frame` / `read_first_frame`。`molrs/src/core/store/record.rs` 的内部机制**一行不动**（molrec 测试套件在另处建设中，引擎内部标识符豁免）。发现用**谓词**，不用手写名单。

## Domain basis

N/A — 命名与公开面治理，不含物理。唯一与数据正确性相关的一条：`RECORD_FORMAT_NAME = "molrec"`（`molrs/src/core/store/record.rs:23`）是**写进落盘 meta 的值**，不是 API 名字；改它会让存量 record 读不出来。落盘值与公开名字是两回事，本规范只动后者，并在门里显式豁免前者——豁免写在门旁边，不散落在提交信息里。

## Design

**命名原则（本规范确立；随后由 `/mol:note` 提升进 CLAUDE.md § Naming）：**

> 公开 API 以**它作用的对象**命名（`Record`、`Frame`、`Trajectory`；`write_frame`、`read_first_frame`），不以**实现它的技术**命名。名字里的技术词只有一种合法情形：它是**调用者自己选定的格式**，因而是对象的一部分（`read_pdb` / `write_xyz` —— 用户要的就是那个格式）。当技术词指的是**尚未裁定的后端或线上契约**（molrec 定义、zarr 存储），它必须从名字里消失——否则名字替维护者承诺了一个还没做的决定。

**三条边界（哪些不改，为什么）：**

- **引擎内部标识符豁免。** `core/store/record.rs` 的 `MolRec` 类型、`RECORD_FORMAT_NAME` / `RECORD_SCHEMA_VERSION` 常数及其值原样保留。它是 molrec 契约的实现机制，不是用户键入的名字。
- **适配层模块可以叫技术名。** `molrs::io::store::zarr` 保留——它**就是** Zarr 绑定，是一个编译单元/适配层。先例在本链上已有两处：02 号裁定 `ff/forcefield/lammps_units.rs` 的文件名含 lammps 是正确的（它是 LAMMPS 读写器的适配层），14 号同样保留 `ff/potential/kspace` 模块作为 FFT 编译单元。技术名活在适配层，死在公开面——同一条规则。
- **外部契约的出处可以被引用。** 模块头文档里指向 `https://github.com/MolCrafts/molrec` 的出处链接保留，与参数表在头文档里写明来自 AmberTools `.DAT` 是同一条约定（CLAUDE.md § Potential System："How a table *arrived* lives in its header doc — never in its name"）。出处是引用，不是命名。

**改名表（公开面，逐条）：**

| 现名（公开） | 新名 | 位置 |
|---|---|---|
| `molrs.MolRec` | `molrs.Record` | `molrs-python/src/core/store/record.rs:15`（pyclass `name`）、`src/lib.rs:218`、`python/molrs/__init__.py:53,166` |
| `MolRec.read_zarr(path)` | `Record.read(path)` | `molrs-python/src/core/store/record.rs:162` |
| `MolRec.write_zarr(path)` | `Record.write(path)` | `molrs-python/src/core/store/record.rs:171` |
| `Trajectory.read_zarr` / `.write_zarr` | `Trajectory.read` / `.write` | `molrs-python/src/core/store/trajectory.rs:116,126` |
| `molrs::MolRec`（crate 根 re-export） | `molrs::Record`（`pub use store::record::MolRec as Record`） | `molrs/src/core/mod.rs:67-69` |
| cxxapi `write_frame_zarr` | `write_frame` | `molrs-cxxapi/src/bridge.rs:283`、`src/lib.rs:651`、`build.rs:173` |
| cxxapi `read_frame_zarr_first` | `read_first_frame` | `molrs-cxxapi/src/bridge.rs:293`、`src/lib.rs:688`、`build.rs:183` |
| `MolRecReader`（wasm README:43） | 删除 | 该符号在 `molrs-wasm/src` 中**不存在**——是文档里的幽灵符号，与 07 号要清的 f32 谎言同类；发现即删，不留 |

`Record` 这个名字：对象就是"一条记录"（`meta` 加上 frame / system / status 之一）。同一命名空间里已有 `ObservableRecord`（可观测量的一行），两者层级不同、不冲突。考虑过 `Dataset` / `Archive`，都比对象本身更抽象，弃。Rust 侧 `read_record_file` / `write_record_file`（`io/store/zarr/record_io.rs:70,232`）**本来就合规**，本规范不动它们——只有模块路径带 zarr，而模块是适配层。

**外部消费者必须同批告知。** `read_frame_zarr_first` 的唯一已知消费者是 Atomiverse 的 checkpoint 重载（`cpu::ZarrReader`，见 `molrs-cxxapi/src/lib.rs:682` 的文档注释）；`write_frame_zarr` 的消费者是同一条链上的 polyethylene checkpoint 写出（`:648`）。0.14 是破坏窗口，**不留 deprecated 别名**——留别名等于把技术名继续挂在公开面上，正是本规范要消灭的东西。旧名→新名映射进 `.claude/notes/notes.md`（跨仓破坏项，带日期），用户可见的迁移条目由 06 承担（本规范新增的依赖边就是为此）。

**发现用谓词。** 扫描六个 crate 树（`molrs`、`molrs-cxxapi`、`molrs-python`、`molrs-ffi`、`molrs-wasm`、`molrs-capi`）与站点/README 文档，命中 `MolRec` / `molrec` / `zarr` 的**公开标识符与用户可见文档行**即红；豁免清单只有三项（内部 `record.rs`、`io::store::zarr` 适配模块路径、出处 URL），且豁免必须写在门的断言体里。手写名单会被手工缩短，本仓已付过学费。

**门放在被 CI 跑的套件里。** 谓词门写进 `molrs-python/tests/test_record.py`（由 tox / `ci-python.yml` 跑），**不放** `molrs/tests/architecture_gate.rs`——后者是 `[[test]]` 目标，默认门 `cargo test --lib` 跑不到它，没被跑过的门等于没有门。

### Reuse decision

- `reuse` `molrs-python/tests/test_molrec.py` 的既有 round-trip 断言作为改名后测试的内容来源（文件随符号改名为 `test_record.py`，断言不重写）。
- `reuse` `molrs/src/core/store/record.rs` 的 `MolRec` 类型与 `io/store/zarr/record_io.rs` 的 `read_record_file` / `write_record_file` —— 实现层 **no-op**，只换公开面的名字。
- `reuse` 02 号确立的 R2 边界（"设施的类型/模块名不带技术名，适配层文件名可以带"）作为本规范三条边界的依据，不另立一套规则。
- `reuse` `molrs-python/tests/` 平铺测试布局与 07 号的 grep 门写法（目录扫描 + 谓词）。
- `generalize` crate 根 re-export（`core/mod.rs:67`）：从"暴露内部类型名"变为"暴露对象名"（`MolRec as Record`），内部实现文件不动。
- `new` — none（本规范不新增任何概念，只改名字与删一条幽灵文档）。

## Files to create or modify

- `molrs/src/core/mod.rs`
- `molrs-python/src/core/store/record.rs`
- `molrs-python/src/core/store/trajectory.rs`
- `molrs-python/src/lib.rs`
- `molrs-python/python/molrs/__init__.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_record.py` (new)
- `molrs-python/tests/test_molrec.py` (delete)
- `molrs-cxxapi/src/bridge.rs`
- `molrs-cxxapi/src/lib.rs`
- `molrs-cxxapi/build.rs`
- `molrs-wasm/README.md`
- `CLAUDE.md`
- `.claude/notes/architecture.md`
- `.claude/notes/architecture-rules.md`
- `.claude/notes/notes.md`
- `regressions/release-0-14-13-frame-store-naming.py` (new)

## Tasks

- [ ] Write failing naming-gate tests scanning the six crate trees and the user-visible docs for `MolRec` / `molrec` / `zarr` in public identifiers, with the three exemptions asserted inline (`molrs-python/tests/test_record.py`)
- [ ] Write failing surface tests: `molrs.Record` exists with `read` / `write`, `molrs.MolRec` and `read_zarr` / `write_zarr` do not, and a record round-trips (`molrs-python/tests/test_record.py`, ported from `test_molrec.py`)
- [ ] Rename the PyO3 record class to `Record` with `read` / `write` in `molrs-python/src/core/store/record.rs`
- [ ] Rename `Trajectory.read_zarr` / `write_zarr` to `read` / `write` in `molrs-python/src/core/store/trajectory.rs`
- [ ] Repoint the Python export surface in `molrs-python/src/lib.rs`, `python/molrs/__init__.py` and `_lib.pyi`, and delete `molrs-python/tests/test_molrec.py`
- [ ] Re-export the record aggregate as `molrs::Record` in `molrs/src/core/mod.rs`, leaving `molrs/src/core/store/record.rs` untouched
- [ ] Rename the cxxapi bridge entries to `write_frame` / `read_first_frame` in `molrs-cxxapi/src/bridge.rs`, `src/lib.rs` and `build.rs`
- [ ] Record the naming principle and the cross-repo breaking rename (Atomiverse `cpu::ZarrReader`) with old→new mapping in `.claude/notes/notes.md`
- [ ] Correct the naming lines in `CLAUDE.md`, `.claude/notes/architecture.md` and `architecture-rules.md`, and delete the phantom `MolRecReader` bullet from `molrs-wasm/README.md`
- [ ] Add regression example `regressions/release-0-14-13-frame-store-naming.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Python 单测平铺 `molrs-python/tests/test_*.py`，每条只测一件事；Rust 侧无新逻辑，故无新 `#[cfg(test)]`。

- 谓词门（本规范最重要的一条）：扫描六个 crate 的 `src/` 与 README / site-src，断言公开标识符与用户可见文档行中 `MolRec` / `molrec` / `zarr` 命中为零；三条豁免（`molrs/src/core/store/record.rs`、`io::store::zarr` 模块路径、出处 URL）以显式白名单写在断言体里，附一行理由。**用目录扫描，不写死文件名单。**
- 缺席门：`hasattr(molrs, "MolRec")` 为假；`Record` 上无 `read_zarr` / `write_zarr`；`Trajectory` 同理；cxxapi 头文件中 `write_frame_zarr` / `read_frame_zarr_first` 命中为零。
- happy path：`Record()` 设 frame + meta → `write(tmp)` → `Record.read(tmp)`，坐标数组与 meta **逐位相等**（`assert_array_equal`，不用容差）；`Trajectory` 同形一条。
- 边界：空 `Record` 写出仍报错且消息不变（改名不得顺手改语义）；`Record.read` 读不存在的路径时异常类型与消息与改名前一致。
- 落盘值不变（防误伤）：改名后写出的 record，其 `meta["format_name"]` 仍为 `"molrec"`，且改名前写出的存量 record 仍可被 `Record.read` 读出——落盘契约与 API 名字解耦的证据。
- 幽灵符号：`molrs-wasm/README.md` 中不再出现 `MolRecReader`（该符号在 `molrs-wasm/src` 中从未存在）。
- 回归样例 `regressions/release-0-14-13-frame-store-naming.py`：纯公开 API——构造 3 原子 Record，写出、读回，断言坐标与写死黄金值逐位相等，并断言旧名字在 `dir(molrs)` 中为零命中。
- 域验证：不适用（无物理）。

## Open questions (maintainer ruling required)

- 无。

## Out of scope

- `core/store/record.rs` 内部机制与 molrec 测试套件（另处建设中）
- 迁移指南中的旧名→新名条目（06，经本规范新增的依赖边）
- Atomiverse 侧 `cpu::ZarrReader` 的跟进改动（另仓；本规范只负责告知与记录）
- zarr `UnitSystem` 枚举的彻底删除（02 已登记的跟进项）
- `io::store::zarr` 模块路径本身（适配层，按边界保留）
