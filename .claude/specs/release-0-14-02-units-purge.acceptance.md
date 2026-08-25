---
slug: release-0-14-02-units-purge
created: 2026-08-25
criteria:
  - id: ac-001
    summary: no MD unit-conversion symbol or md::units module survives
    type: code
    pass_when: |
      ripgrep for MD_ENERGY|energy_to_md|preset_energy_to_md|kb_md|
      set_energy_scale|energy_scale across molrs/src, molrs-python/src,
      molrs-python/python and regressions/ returns zero hits outside
      .claude/notes/ history; molrs/src/md/units.rs does not exist and
      molrs/src/md/mod.rs declares no units module.
    status: pending
  - id: ac-002
    summary: the preset facility lives in core::units with LAMMPS-free type naming
    type: code
    pass_when: |
      molrs/src/core/units/preset.rs defines UnitPreset and UnitPresetRegistry
      with named constructors and a register extension point; no type or module
      name of the preset facility contains "Lammps"; the preset name strings
      "real"/"metal"/"lj" are accepted; no make_* / build_* / create_* factory
      is added; nothing unit-preset-shaped remains under molrs/src/ff or
      molrs/src/md.
    status: pending
  - id: ac-003
    summary: ff and zarr are consumers, not rival vocabularies
    type: code
    pass_when: |
      LammpsUnits and LammpsUnitSystem no longer exist as types;
      ff/forcefield/lammps_units.rs holds only a token→preset-name adapter with
      no unit data; io/store/zarr/mod.rs resolves its unit semantics through
      core::units; the zarr enum's full removal is recorded as a dated
      follow-up in .claude/notes/notes.md.
    status: pending
  - id: ac-004
    summary: preset constants are references, not copies
    type: scientific
    pass_when: |
      A #[cfg(test)] test asserts UnitPreset::real().boltzmann() is
      bit-identical to core::units::constants::BOLTZMANN_REAL, and no second
      numeric literal for Boltzmann's constant exists under molrs/src.
    status: pending
  - id: ac-005
    summary: collapsing the ff unit table changed no parameters
    type: scientific
    pass_when: |
      The LAMMPS force-field reader produces bit-identical parameters for a
      committed .ff fixture before and after the change, and existing Zarr
      metadata still reads back unchanged.
    status: pending
  - id: ac-006
    summary: MaxwellBoltzmann kbt reshape is physics-neutral
    type: scientific
    pass_when: |
      A #[cfg(test)] test in molrs/src/md/maxwell.rs asserts the velocity array
      from new(kbt = BOLTZMANN_REAL * 300.0, seed) equals the pre-change
      new(300.0, seed) array bitwise (assert_eq!, no tolerance).
    status: pending
  - id: ac-007
    summary: thermo temperature needs an explicit kb
    type: runtime
    pass_when: |
      MD().run(frame, 10, dt=1.0, thermo=5) without kb= raises ValueError whose
      message contains "kb="; with kb= supplied the temp column is finite.
    status: pending
  - id: ac-008
    summary: units regression reproduces the hard-coded golden
    type: runtime
    pass_when: |
      `python regressions/release-0-14-02-units-purge.py` exits 0, imports no
      third-party scientific package, and reproduces its embedded 10-step total
      energy golden to within 1e-12 relative.
    status: pending
  - id: ac-009
    summary: full gate green after the purge
    type: runtime
    pass_when: |
      cargo lib tests, cargo doc tests and the molrs-python tox py env all pass.
    status: pending
out_of_scope:
  - MD(dtype=) driver shape
  - Potential Protocol
  - full removal of the zarr UnitSystem enum
  - md::LJCut / PairLJCut kernel merge
---

# Acceptance — release-0-14-02-units-purge

单位制只有一个家（`core::units`）、一个词汇、一份常数，且类型名不带 LAMMPS；MD 内部零换算；收编与 kbt 改形均以逐位相等证明未动物理。
