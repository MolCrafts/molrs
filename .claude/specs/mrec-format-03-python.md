---
title: Record/Trajectory 的读写迁到 molrs.io.mrec
slug: mrec-format-03-python
status: approved
created: 2026-08-30
depends_on:
  - mrec-format-02-io
---

# mrec-format-03-python — Record/Trajectory 的读写迁到 molrs.io.mrec

## Summary

Python 科学记录 I/O 从 `molrs.Record.read` / `write` 与 `molrs.Trajectory.read` / `write` 迁到 `molrs.io.mrec`。该子模块提供 `TrajectoryReader`（惰性 `FrameSequence`，与 wasm `js_name` 同名）以及与 Rust 路径门对应的 `read_record` / `write_record` / `write_trajectory`。这些名字只活在 `molrs.io.mrec`，不从 `molrs.io` 根再导出。不设 `FrameReader`（与 `io::reader::FrameReader` trait / 02 禁令冲突），不设 `make_*`，不设上帝 `Reader`，不设 `SystemReader`。

## Domain basis

不含物理。本规范搬公开门的属主，不改数组布局或单位。

## Design

`Record` / `Trajectory` 是内存载体，不再拥有落盘门。读写全部迁到 `molrs.io.mrec`。

| 符号 | 职责 |
|------|------|
| `molrs.io.mrec.read_record(path) -> Record` | 整记录路径门 |
| `molrs.io.mrec.write_record(path, record)` | 整记录路径门 |
| `molrs.io.mrec.TrajectoryReader(path)` | 惰性序列，包装 Rust `FrameSequence`；一次一帧 |
| `molrs.io.mrec.write_trajectory(path, trajectory)` | 与 Rust `write_trajectory_file` 对应 |

`TrajectoryReader` **只**在 `molrs.io.mrec` 出现。`molrs.io.TrajectoryReader` 仍是 LAMMPS/XYZ/DCD dump 拼接器。`Record.read` / `write` / `Trajectory.read` / `write` 删除。system 段走 `read_record` / `write_record`。

### Reuse decision

- `reuse` 今天 `PyRecord::read` / `write` 与 `PyTrajectory::read` / `write` 的函数体——搬家
- `reuse` Rust `FrameSequence` 作为 Python `TrajectoryReader` 的内部游标
- `generalize` Record/Trajectory 上的落盘方法为 `molrs.io.mrec` 路径门
- `new` — `python/molrs/io/mrec.py`；不铸 Python `FrameReader`

## Files to create or modify

- `molrs-python/src/io/mrec.rs` (new)
- `molrs-python/src/io/mod.rs`
- `molrs-python/src/lib.rs`
- `molrs-python/src/core/store/record.rs`
- `molrs-python/src/core/store/trajectory.rs`
- `molrs-python/python/molrs/io/mrec.py` (new)
- `molrs-python/python/molrs/io/__init__.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_mrec.py` (new)
- `molrs-python/tests/test_record.py`
- `regressions/mrec-format-03-python.py` (new)

## Tasks

- [ ] Write failing unit tests for molrs.io.mrec (molrs-python/tests/test_mrec.py → TestTrajectoryReader, test_read_record, test_write_record)
- [ ] Implement PyO3 doors in molrs-python/src/io/mrec.rs and register the mrec submodule
- [ ] Implement python/molrs/io/mrec.py exporting TrajectoryReader, read_record, write_record, write_trajectory
- [ ] Strip Record.read/write and Trajectory.read/write from src/core/store/record.rs, trajectory.rs, and python/molrs/_lib.pyi
- [ ] Update molrs-python/tests/test_record.py to call molrs.io.mrec and assert the dump concatenator molrs.io.TrajectoryReader is unchanged
- [ ] Add docstring per rustdoc/google on the new Python/Rust doors
- [ ] Add regression example regressions/mrec-format-03-python.py (public API only; hard-coded goldens)
- [ ] Verify TrajectoryReader is not re-exported at molrs.io root and FrameReader is absent
- [ ] Run full check + test suite

## Testing strategy

`molrs-python/tests/test_mrec.py`。`TestTrajectoryReader.test_frame`：惰性读第 0 帧。`test_read_record` / `test_write_record`：system + meta 往返。`Record.read` / `Trajectory.write` 为 `AttributeError`。`from molrs.io import TrajectoryReader` 仍是 dump 拼接器。回归：`write_record` 到 `tmp/*.mrec`，`read_record` 断言 `format_name == "mrec"`。

## Out of scope

- 不在 `molrs.io` 根导出 mrec `TrajectoryReader`。
- 不设 Python `FrameReader` / `make_*` / `SystemReader`。
- 不改 dump 拼接器 `molrs.io.TrajectoryReader`。
- 不改 wasm `js_name`（04）或 molpy 门面（05）。
