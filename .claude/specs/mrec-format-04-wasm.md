---
title: wasm js_name TrajectoryReader，从 io::mrec 引入
slug: mrec-format-04-wasm
status: approved
created: 2026-08-30
depends_on:
  - mrec-format-02-io
---

# mrec-format-04-wasm — js_name TrajectoryReader，从 io::mrec 引入

## Summary

wasm 侧今天名为 `RecordReader` 的类型包装的是 `FrameSequence`（轨迹游标），JS 名改为 `TrajectoryReader`，不是 `FrameReader`。`FrameSequence` 从 `molrs::io::mrec` 引入。适配层文件路径 `molrs-wasm/src/io/zarr/` 可以保留。

## Domain basis

不含物理。本规范只改 JS 导出名与 Rust 引入路径。

## Design

`molrs-wasm/src/io/zarr/mod.rs` 仍包装 `RefCell<FrameSequence>` + `MemoryStore`；`#[wasm_bindgen(js_name = TrajectoryReader)]`。`use molrs::io::mrec::FrameSequence`。不在 wasm 新增 `FrameReader`。

### Reuse decision

- `reuse` 现有函数体（MemoryStore 装载、FrameSequence::open、read_frame）
- `generalize` `js_name = RecordReader` 为 `js_name = TrajectoryReader`
- `new` — 无

## Files to create or modify

- `molrs-wasm/src/io/zarr/mod.rs`
- `molrs-wasm/src/io/mod.rs`
- `molrs-wasm/README.md`
- `molrs-wasm/pkg/molrs.d.ts`
- `regressions/mrec-format-04-wasm.md` (new)

## Tasks

- [ ] Write failing unit tests or export pin that the wasm js_name is TrajectoryReader
- [ ] Implement js_name = TrajectoryReader and import FrameSequence from molrs::io::mrec
- [ ] Update molrs-wasm/src/io/mod.rs module table and README.md
- [ ] Regenerate pkg/molrs.d.ts so it exports TrajectoryReader and not RecordReader
- [ ] Add regression example regressions/mrec-format-04-wasm.md
- [ ] Verify the source file still lives at molrs-wasm/src/io/zarr/mod.rs
- [ ] Run full check + test suite

## Testing strategy

源码门：`js_name = TrajectoryReader` 存在；无 `FrameReader` / `RecordReader` JS 名；`use molrs::io::mrec::FrameSequence`。pkg d.ts 导出 `TrajectoryReader`。

## Out of scope

- 不把 `molrs-wasm/src/io/zarr/` 改名为 `mrec/`。
- 不新增 wasm `FrameReader`。
- 不改 `loadZarrStore`（06）。
