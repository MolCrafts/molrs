---
slug: release-0-14-10-molpy-mirror
created: 2026-08-25
criteria:
  - id: ac-001
    summary: the sink direction and the capability criterion are written down
    type: docs
    pass_when: |
      molpy CLAUDE.md states that any surface shared with molrs sinks into
      molrs, that molpy is re-export plus numpy-facing extension only, and
      carries the capability criterion (molrs covers it on a molpy-consumable
      surface → sink in 0.14; molrs does not → molpy-native extension) together
      with the ledger listing pdb, top, amber, lammps-data, lammps-molecule,
      lammps-log, forcefield-xml and Box geometry as sunk, and HDF5, ac,
      moltemplate, lammps-bond-react, openmm emit and the FieldFormatter layer
      as molpy-native extensions.
    status: pending
  - id: ac-002
    summary: molpy.md re-exports molrs.md by identity
    type: runtime
    pass_when: |
      For every name in molrs.md.__all__, molpy.md.X is molrs.md.X; molpy.md
      exposes no public name absent from molrs.md.
    status: pending
  - id: ac-003
    summary: molpy owns no compute framework class and all shells conform
    type: code
    pass_when: |
      molpy/src/molpy/compute/base.py defines no framework base class, all 24
      shells satisfy isinstance(obj, molrs.compute.Compute), and at least one
      does so without inheriting from the Protocol.
    status: pending
  - id: ac-004
    summary: the contract change moved no numbers
    type: runtime
    pass_when: |
      For all 24 shells (enumerated by directory scan), fixed-input results
      after the change are bit-identical to the recorded pre-change baselines
      (assert_array_equal, no tolerance).
    status: pending
  - id: ac-005
    summary: every sunk format reproduces its pre-sink output bitwise
    type: scientific
    pass_when: |
      For each duplicated capability molrs covers (pdb, top, amber prmtop and
      inpcrd, lammps-data, lammps-molecule, lammps-log, forcefield-xml) over the
      tests/tests-data corpus enumerated by directory scan, post-sink Frames
      match pre-sink Frames column by column with assert_array_equal (no
      tolerance) and identical column-name sets; written output matches byte for
      byte. Any divergence is reported with format, column and max ULP, and was
      not softened into assert_allclose.
    status: pending
  - id: ac-006
    summary: Box geometry delegates to molrs without moving numbers
    type: scientific
    pass_when: |
      wrap, unwrap, get_images, diff, dist, make_fractional, make_absolute and
      get_distance_between_faces return bit-identical results before and after
      delegation on both an orthogonal and a triclinic box with boundary-
      crossing points; no numpy reimplementation of a molrs kernel remains in
      molpy/src/molpy/core/box.py.
    status: pending
  - id: ac-007
    summary: the anti-duplication gate is permanent and bites
    type: runtime
    pass_when: |
      tests/test_sink_policy.py fails when any molpy module implements a
      capability molrs already provides on a molpy-consumable surface; the gate
      was demonstrated red by adding a 20-line pdb parser and then reverted; the
      molpy-native extensions sit in an exemption set whose reason is stated in
      the test body.
    status: pending
  - id: ac-008
    summary: the three name collisions have one answer each
    type: runtime
    pass_when: |
      NeighborList, Region and GroFieldFormatter each resolve to exactly one
      definition, pinned by a test naming the winning module.
    status: pending
  - id: ac-009
    summary: molrs kernels are not re-wrapped under molpy names
    type: code
    pass_when: |
      For each re-exported class, type(obj).__module__ points into molrs, not
      into a molpy shadow class of the same name.
    status: pending
  - id: ac-010
    summary: mirror regression runs on molpy alone
    type: runtime
    pass_when: |
      `python regressions/release-0-14-10-molpy-mirror.py` in molpy exits 0
      importing only molpy, imports no third-party scientific package, and
      matches its embedded pdb-read, RDF and 5-step MD goldens.
    status: pending
  - id: ac-011
    summary: full gate green after the sink
    type: runtime
    pass_when: |
      ruff format --check, ruff check, ty check and `pytest tests/ -m "not
      external"` all pass in molpy.
    status: pending
out_of_scope:
  - capabilities molrs does not provide (HDF5, ac, moltemplate, lammps-bond-react, openmm emit, FieldFormatter layer) — molpy-native extensions
  - binding molrs's Rust read_ac to the Python surface
  - docs / spelling sweep
  - joint smoke and molpy tag
  - any molrs change
---

# Acceptance — release-0-14-10-molpy-mirror

下沉方向已成书面政策，去留由**能力判据**而非排期决定：molrs 覆盖的能力本轮全部下沉并逐格式逐位对拍，molrs 没有的正名为 molpy 原生扩展；反重复门是永久的、且被证明会咬。
