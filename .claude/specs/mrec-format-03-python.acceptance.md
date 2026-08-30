---
spec: mrec-format-03-python
created: 2026-08-30
criteria:
  - id: ac-001
    summary: Record and Trajectory no longer own read/write
    type: code
    pass_when: |
      PyRecord and PyTrajectory have no read/write pymethods;
      python/molrs/_lib.pyi Record and Trajectory have no read/write;
      molrs.Record.read and molrs.Trajectory.write raise AttributeError.
    status: pending
  - id: ac-002
    summary: molrs.io.mrec exposes TrajectoryReader and record doors
    type: runtime
    pass_when: |
      molrs.io.mrec.TrajectoryReader(path) lazily yields frames from
      FrameSequence; molrs.io.mrec.read_record/write_record round-trip
      a Record that carries system and meta; no FrameReader, no make_*,
      no molrs.io.mrec.Reader class exists.
    status: pending
  - id: ac-003
    summary: mrec names stay in the mrec submodule
    type: code
    pass_when: |
      python/molrs/io/__init__.py does not re-export TrajectoryReader
      from mrec; molrs.io.TrajectoryReader remains the dump concatenator;
      `from molrs.io.mrec import TrajectoryReader` works.
    status: pending
  - id: ac-004
    summary: regression uses molrs.io.mrec path doors
    type: runtime
    pass_when: |
      regressions/mrec-format-03-python.py writes via write_record to a
      *.mrec directory, reads via read_record, and asserts format_name
      "mrec"; it never calls Record.read or Trajectory.write.
    status: pending
---

# Acceptance — mrec-format-03-python

I/O 属主是 `molrs.io.mrec` 的原语，不是内存载体上的方法；不另立 `FrameReader`。
