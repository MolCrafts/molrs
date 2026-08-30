---
spec: mrec-format-04-wasm
created: 2026-08-30
criteria:
  - id: ac-001
    summary: wasm js_name is TrajectoryReader, not FrameReader
    type: code
    pass_when: |
      molrs-wasm/src/io/zarr/mod.rs has js_name = TrajectoryReader on the
      type that wraps FrameSequence; it does not use js_name FrameReader
      or js_name RecordReader; readFrame/countFrames/countAtoms remain.
    status: pending
  - id: ac-002
    summary: FrameSequence is imported from molrs::io::mrec
    type: code
    pass_when: |
      molrs-wasm/src/io/zarr/mod.rs contains use molrs::io::mrec::FrameSequence
      and does not import molrs::io::zarr::FrameSequence; the file path
      molrs-wasm/src/io/zarr/mod.rs still exists.
    status: pending
  - id: ac-003
    summary: pkg and docs list TrajectoryReader
    type: code
    pass_when: |
      molrs-wasm/pkg/molrs.d.ts exports class TrajectoryReader and does
      not export RecordReader; io/mod.rs and README.md list TrajectoryReader.
    status: pending
---

# Acceptance — mrec-format-04-wasm

JS 名与所包装的惰性轨迹游标一致；引入路径走 02 的公开模块。
