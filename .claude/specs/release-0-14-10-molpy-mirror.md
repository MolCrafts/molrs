---
slug: release-0-14-10-molpy-mirror
title: release-0-14-10-molpy-mirror — 共享面一律下沉 molrs；能力判据定去留，molrs 没有的改称 molpy 原生扩展
status: approved
grilled: true
created: 2026-08-25
depends_on:
  - release-0-14-09-molpy-rebase
---

# release-0-14-10-molpy-mirror — 能力判据，不是排期表

## Summary

冻结治理原则：**凡 molpy 与 molrs 共有的东西，一律下沉/被吸收进 molrs**；molpy 收敛为纯 re-export 加面向 numpy 的扩展层。去留不再按"排期"讨论，而由**能力判据**一刀切：**molrs 已经实现的格式/能力，其 molpy 重复实现在 0.14 内下沉，验收是逐格式在 committed 语料上逐位对拍**；molrs 没有的，从"债"改称 **molpy 原生扩展**（numpy 面），不再被当作待清理项。防扩张门随之从"临时冻结"改为**永久反重复门**：任何新增的、molrs 已覆盖能力的 molpy 实现即红。

## Domain basis

N/A — 命名空间与所有权治理。下沉与改造都不得改变任何内核的数值行为；这一点由"改动前后结果逐位相同"的测试守，而不是靠断言"应该不变"。跨实现替换（molpy numpy 实现 → molrs Rust 内核）若出现逐位不同，按铁律是**报告项**：量出最大 ULP、指名符号、上报维护者，**不得**就地放宽成相对容差。

## Design

**治理原则（适用于本规范及后续所有 molpy 工作）：**

> molpy 与 molrs 共有的任何实现，方向是**下沉进 molrs**。molpy 保留的只有两类：(a) 对 molrs 的纯 re-export；(b) 面向 numpy 的扩展层（molrs 不提供、且天然属于 Python/numpy 生态的便利）。任何"molpy 自己再实现一份"的新增，默认视为回归，需维护者显式豁免。

**能力判据（维护者裁定，取代旧的分项排期表）：**

> 一个 molpy 模块的去留，只看一个谓词：**它实现的能力，molrs 是否已经在 molpy 消费得到的表面上提供**（`molrs.io.*` / `molrs.ff.*` / `molrs.Box` 等）。
> - **是** → 0.14 内下沉，molpy 侧只保留面向 numpy 的 formatter / 便利层，验收是**逐格式在 committed 语料上逐位对拍**。
> - **否** → 它不是债，是 **molpy 原生扩展**（numpy 面），照此命名、照此存续，不再列入清理清单。

**能力盘点（对着 molrs 树核对；用谓词枚举，下表是本轮的快照）：**

| molpy 模块 | molrs 对应 | 判据结果 |
|---|---|---|
| `io/data/pdb.py` | `molrs.io.read_pdb` / `write_pdb` / `read_pdb_trajectory` | **0.14 下沉** |
| `io/data/top.py` | `molrs.io.read_top` / `write_top` | **0.14 下沉** |
| `io/data/amber.py`（prmtop / inpcrd） | `molrs.io.read_prmtop` / `read_inpcrd` | **0.14 下沉** |
| `io/log/lammps.py` | `molrs.io.raw.read_lammps_log` / `parse_lammps_log_text` | **0.14 下沉** |
| `io/data/lammps.py`（DATA 读写） | `molrs.io.read_lammps_data` / `write_lammps_data` | **0.14 下沉** |
| `io/data/lammps_molecule.py` | `molrs.io.read_lammps_molecule` / `write_lammps_molecule` | **0.14 下沉** |
| `io/forcefield/xml.py`（+ OPLSAA 子类） | `molrs.ff.read_forcefield_xml` / `read_opls_xml` / `write_forcefield_xml` | **0.14 下沉** |
| `core/box.py` 几何 override（`wrap*` / `diff*` / `dist*` / `get_images` / `unwrap` / `make_fractional` / `make_absolute` / `get_distance_between_faces`） | `molrs.Box.wrap` / `delta` / `images` / `unwrap` / `to_frac` / `to_cart` / `nearest_plane_distance` | **0.14 下沉**（能力已齐；`wrap` / `diff_dr` / `make_fractional` 今天已经在委托，剩下几个补齐） |
| `io/data/h5.py`、`io/trajectory/h5.py`、`io/store/_h5.py` | 无 | **molpy 原生扩展** |
| `io/data/ac.py`（`AcFieldFormatter` / `AcReader`） | molrs 有 Rust `read_ac`，**未绑定到 Python 面** | **molpy 原生扩展**（判据看的是 molpy 消费得到的表面） |
| `io/forcefield/moltemplate.py`、`io/data/lammps_bond_react.py`、`io/emit/openmm.py` | 无 | **molpy 原生扩展** |
| `core/fields.py` 的 FieldFormatter 层（各格式 formatter 子类） | numpy 面语义翻译，molrs 侧无对应职责 | **molpy 原生扩展**（下沉后保留的正是这一层） |

"molpy 原生扩展"不是委婉语：它们**不再出现在任何待清理清单里**，也不受下沉门约束——门只管"molrs 已覆盖的能力"。

**永久反重复门（取代临时防扩张门）。** `tests/test_sink_policy.py` 的判据从"行数不得超过基线"改为**能力谓词**：molpy 中任何模块，若其实现的能力 molrs 已在消费面提供，即红，提示"该能力由 molrs 提供，molpy 侧只保留 numpy 面；确需自实现请附维护者豁免"。这条门**永久有效**，不是发布窗口的临时冻结——5000 行 io 重复正是没有这道门的后果；把原则写进文档而不设门，等于什么都没做。

**compute 改造用词。** 不是"继承 `molrs.compute.Compute`"，而是**符合 molrs `Compute` 契约**——契约是 `@runtime_checkable` Protocol，只要求 `compute(...)`，**显式继承可选**。删除 `<molpy>/src/molpy/compute/base.py` 的自有框架基类；**24 个薄壳保留**（维护者已裁定，与 05 一致），提供 `compute(...)` 即满足；`__call__` 与 `dump()` **不在 molrs 契约里**（05 已明确），由 molpy 侧薄壳自行提供并存续。

**数值不变证明。** compute 改造是纯契约变更：24 个薄壳用**目录扫描**枚举，每个以同输入同参数跑改前/改后，`assert_array_equal` **逐位相同**——不用容差。io 与 Box 下沉是**跨实现替换**：逐格式在 `tests/tests-data/` 语料上把改前 molpy 输出与改后（molrs 委托）输出逐列对拍，要求逐位相同；任何逐位不同**指名上报**（格式、列、最大 ULP），由维护者裁定是修 molrs 内核还是作为数值变更写进 11 号的迁移说明——**不得**就地改成 `assert_allclose`。

**`molrs` 词的用户可见清除**在 11 执行；本规范只负责对象层。

### Reuse decision

- `reuse` 05 落地的 `molrs.compute.Compute` Protocol 作为唯一契约——molpy 不再定义框架类。
- `reuse` `molrs.md.__all__` 作为 identity 测试的枚举源（谓词而非名单）。
- `reuse` molrs 的 `NeighborList` 引擎类型作为三重同名的收敛目标（`.claude/notes/notes.md` binder-surface-symmetry：三面同名同形同默认）。
- `reuse` molrs 既有的 pdb / top / prmtop / inpcrd / lammps-data / lammps-molecule / lammps-log / forcefield-xml 读写器作为 io 下沉的**目标实现**——它们已在 `molrs.io` / `molrs.ff` 的公开面上，无需新绑定。
- `reuse` `molrs.Box` 的 `wrap` / `delta` / `images` / `unwrap` / `to_frac` / `to_cart` / `nearest_plane_distance` 作为几何 override 的目标实现；molpy `Box` 已继承 `molrs.Box`（`core/box.py:13`），下沉是删 numpy 分支、留签名，不是加桥接。
- `reuse` `tests/tests-data/` 语料作为逐格式对拍的 fixture 来源（molrs 侧同一份语料）。
- `generalize` molpy 24 个薄壳的构造契约到 `__init__(**config)`；薄壳只保留领域参数校验与 molpy 侧自有的 `__call__` / `dump()`。
- `generalize` `tests/test_sink_policy.py`：从"行数基线冻结"扩为"能力谓词的永久反重复门"。
- `new` — none（能力判据用的是既有 molrs 公开面的枚举，不新建注册表）。

## Files to create or modify

- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/md/__init__.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/compute/base.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/compute/__init__.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/compute/neighborlist.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/compute/spatial.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/writers.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/data/pdb.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/data/top.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/data/amber.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/data/lammps.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/data/lammps_molecule.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/log/lammps.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/io/forcefield/xml.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/core/box.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/__init__.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/CLAUDE.md`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_molrs_mirror.py` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_sink_policy.py` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_io_parity.py` (new)
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-10-molpy-mirror.py` (new)

## Tasks

- [ ] Write failing identity test enumerating `molrs.md.__all__` and asserting `molpy.md.X is molrs.md.X` for every name (`/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/tests/test_molrs_mirror.py`)
- [ ] Write failing bit-identity tests capturing current results of all 24 `molpy.compute` shells before the contract change
- [ ] Write failing sink-policy gates: per-format bit-identical parity over `tests/tests-data/` for every duplicated capability molrs covers (`tests/test_io_parity.py`), plus the permanent no-duplicate-capability gate (`tests/test_sink_policy.py`)
- [ ] Implement the verbatim `molpy.md` re-export in `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/src/molpy/md/__init__.py`
- [ ] Delete molpy's own compute framework class and make all 24 shells conform to the molrs `Compute` Protocol, discovered by directory scan (explicit subclassing optional)
- [ ] Resolve the three name collisions across `molpy/src/molpy/compute/neighborlist.py`, `compute/spatial.py` and `io/writers.py`
- [ ] Sink every duplicated implementation whose capability molrs already covers — the pdb / top / amber / lammps-data / lammps-molecule / lammps-log / forcefield-xml readers and writers plus the `Box` geometry overrides — onto the molrs implementation, keeping only the numpy-facing formatter layer
- [ ] Record the sink-to-molrs principle, the capability criterion and the capability ledger in `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/CLAUDE.md`, naming the remainder "molpy-native extensions"
- [ ] Add regression example `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molpy/regressions/release-0-14-10-molpy-mirror.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

molpy 单测位于 `<molpy>/tests/`，平铺命名；格式语料为 `<molpy>/tests/tests-data/`（与 molrs 同一份 committed 语料）。

- identity（happy path）：`molrs.md.__all__` 逐名 `is` 断言；反向断言 `molpy.md` 不引入 `molrs.md` 没有的公开名（防镜像层偷加东西）。
- 契约满足：每个薄壳 `isinstance(obj, molrs.compute.Compute)` 为真；断言**不要求**继承（至少一个薄壳保持不继承以证明结构足够）。
- compute 改造的数值不变：24 个薄壳**目录扫描**枚举，固定输入跑改前/改后 `assert_array_equal`（逐位，无容差）；改前基线以 09 分支输出为准并写死进测试。
- **io 逐格式对拍（本轮最重的一条）**：每个下沉格式在 `tests/tests-data/` 的对应语料上，改前 molpy 读出的 Frame 与改后（委托 molrs）读出的 Frame **逐块逐列**对拍——数值列 `assert_array_equal`（逐位）、字符串列相等、列名集合相等；写出路径同样以字节对拍。语料用**目录扫描**枚举，不写死文件名单。任何逐位不同：测试失败并打印格式、列名与最大 ULP，按铁律上报，**不得**改成 `assert_allclose`。
- Box 几何对拍：`wrap` / `unwrap` / `get_images` / `diff` / `dist` / `make_fractional` / `make_absolute` / `get_distance_between_faces` 在正交与三斜两种盒子、含跨界点的固定坐标集上，改前 numpy 实现与改后 molrs 委托逐位对拍；同一条上报规则。
- 命名冲突：`NeighborList` / `Region` / `GroFieldFormatter` 各解析到唯一定义，由一条指名胜出模块的测试钉死；`type(obj).__module__` 对 re-export 类指向 molrs（证明没被重新包装成同名新类）。
- 永久反重复门：谓词遍历 molpy 模块，任何"molrs 已在消费面提供该能力"的自实现即红。**bite-proof**：新写一个 20 行的 pdb 迷你解析器，门应变红，验证后还原——没红过的门等于没有门。
- 原生扩展面：断言 h5 / moltemplate / lammps_bond_react / ac / emit-openmm / FieldFormatter 层**不被**该门约束（它们在门的豁免集合里，且豁免理由写在断言体内）。
- 回归样例 `<molpy>/regressions/release-0-14-10-molpy-mirror.py`：只 `import molpy`，读一个 committed pdb、跑一个 RDF 与一个 5 步 MD，断言写死黄金值——同时证明"用户只需要 molpy"。
- 域验证：由逐位不变断言承担（内核物理在 molrs 侧已验证）。

## Open questions (maintainer ruling required)

- 无。（24 个薄壳保留，已裁定；io 与 Box 的去留改由能力判据决定，不再是排期问题。）

## Out of scope

- molrs 未提供的能力（HDF5、moltemplate、lammps-bond-react、AC、openmm emit、FieldFormatter 层）——它们是 molpy 原生扩展，不在下沉范围
- 把 molrs 的 Rust `read_ac` 绑定到 Python 面（molrs 侧新功能，另开）
- 文档与拼写清扫（11）
- 联合冒烟与 molpy tag（12）
- 任何 molrs 侧改动
