---
slug: release-0-14-04-md-driver
created: 2026-08-25
criteria:
  - id: ac-001
    summary: MD takes a numpy dtype and rejects float32
    type: runtime
    pass_when: |
      MD(dtype=np.float64) constructs; MD(dtype=np.float32) raises and the
      message names the Rust integrators as the future home.
    status: pending
  - id: ac-002
    summary: precision vocabulary is gone
    type: code
    pass_when: |
      ripgrep for PRECISIONS|resolve_prec|\bprec\b under
      molrs-python/python/molrs/md/ returns zero hits.
    status: pending
  - id: ac-003
    summary: exactly one MDState, PyO3-owned with setters
    type: runtime
    pass_when: |
      molrs.md.MDState is molrs._lib.md.MDState, it is not a NamedTuple, at
      least one attribute is assignable, and no FrameVelocityVerlet symbol
      exists anywhere in molrs-python/python.
    status: pending
  - id: ac-004
    summary: .pyi matches the runtime and annotates force params as md.Potential
    type: runtime
    pass_when: |
      A test enumerating dir(molrs._lib.md) finds a matching declaration in
      molrs-python/python/molrs/_lib.pyi for every public name and no .pyi
      declaration without a runtime counterpart; every force-accepting
      parameter is annotated md.Potential and no line contains both "Union["
      and "Potential".
    status: pending
  - id: ac-005
    summary: the driver accepts structural potentials without isinstance guards
    type: runtime
    pass_when: |
      An object defining only calc_energy_forces and inheriting nothing is
      accepted by set_potential / set_forcefield, and driver.py contains no
      isinstance check against md.Potential.
    status: pending
  - id: ac-006
    summary: experimental warning is scoped to molrs.md
    type: runtime
    pass_when: |
      `import molrs` records zero FutureWarning; `import molrs.md` records
      exactly one whose message contains "experimental in 0.14".
    status: pending
  - id: ac-007
    summary: Rust NVE conservation stays the authority
    type: scientific
    pass_when: |
      The #[cfg(test)] conservation test in molrs/src/md/integrators.rs runs 64
      Ar-like atoms for 1000 steps at dt=1 fs and asserts relative energy drift
      < 5e-5 (measured 1.71e-5).
    status: pending
  - id: ac-008
    summary: driver-level NVE regression reproduces the drift bound
    type: runtime
    pass_when: |
      `python regressions/release-0-14-04-md-driver-nve.py` exits 0, uses only
      public molrs API, imports no third-party scientific package, asserts
      relative drift < 5e-5 over 1200 steps and rebuild_count > 0; the stale
      regressions/release-0-14-01-md-driver-nve.py no longer exists.
    status: pending
  - id: ac-009
    summary: no pre-existing pytest failures remain
    type: runtime
    pass_when: |
      A full `uv --directory molrs-python run --no-sync tox -e py` run reports
      0 failures, including the 6 Frame.meta MetaValue cases and the keys-tuple
      case that were red against the rebuilt extension.
    status: pending
out_of_scope:
  - float32 / mixed Rust integrators
  - per-type mixing and special_bonds exclusion in the pair path
  - wasm / capi md bindings
---

# Acceptance — release-0-14-04-md-driver

驱动只有一个类、一个状态类型、一个 numpy 风格的精度入口；力的入参是结构化的且注解为单一 `md.Potential`；长物理跑在 `regressions/`，`tests/` 只剩接缝；全量 pytest 零失败。
