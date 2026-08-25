---
slug: release-0-14-03-potential-protocol
created: 2026-08-25
criteria:
  - id: ac-001
    summary: structure alone satisfies Potential
    type: runtime
    pass_when: |
      A class defining only calc_energy_forces and inheriting nothing satisfies
      isinstance(obj, molrs.md.Potential) and is accepted by VelocityVerlet.
    status: pending
  - id: ac-002
    summary: LJCut and Potentials satisfy the same Protocol
    type: runtime
    pass_when: |
      isinstance(md.LJCut(1.0, 1.0, 2.5), md.Potential) and
      isinstance(md.Potentials(), md.Potential) are both True.
    status: pending
  - id: ac-003
    summary: exactly one Potential identity, and it is the Protocol
    type: runtime
    pass_when: |
      molrs.md.Potential is molrs.ff.Potential, it is a runtime_checkable
      Protocol, and no PyO3 class is exported under any name spelled Potential
      (the former PyPotential base is gone from the exported surface).
    status: pending
  - id: ac-004
    summary: concrete Rust arms precede the duck-typed fallback
    type: code
    pass_when: |
      In molrs-python/src/md.rs the PyLJCut and PyPotentials extract arms of
      take_potential appear before the duck-typed SubclassPotential fallback,
      and the dispatch test was demonstrated red under the reversed order.
    status: pending
  - id: ac-005
    summary: Potentials keeps native move semantics
    type: runtime
    pass_when: |
      A Potentials passed to VelocityVerlet is consumed (second use raises
      ValueError), proving it did not take the Python-dispatch fallback.
    status: pending
  - id: ac-006
    summary: no Union potential signature survives
    type: code
    pass_when: |
      ripgrep over molrs-python/python/molrs/_lib.pyi finds no line containing
      both "Union[" and "Potential"; the 3 former sites read "md.Potential".
    status: pending
  - id: ac-007
    summary: isinstance limits are documented, not assumed away
    type: docs
    pass_when: |
      protocol.py docstring states that runtime_checkable isinstance checks
      method presence only (PEP 544), that it is not for hot paths, that
      fast-path dispatch does not rely on it, and that the contract is
      coordinates-only; a test pins that a wrong-signature object still passes
      isinstance.
    status: pending
  - id: ac-008
    summary: subclass exceptions pass through unchanged
    type: runtime
    pass_when: |
      A custom force whose calc_energy_forces raises KeyError("boom") surfaces
      KeyError("boom") verbatim out of VelocityVerlet.advance_n.
    status: pending
  - id: ac-009
    summary: molrs.md exports Potentials
    type: runtime
    pass_when: "\"Potentials\" in molrs.md.__all__ and molrs.md.Potentials is molrs.ff.Potentials."
    status: pending
  - id: ac-010
    summary: the Python contract stays coordinates-only
    type: runtime
    pass_when: |
      A duck-typed potential records the arguments it receives; across a
      VelocityVerlet run with a VerletSkin attached it is called with
      coordinates only — no neighbour table, pair index, displacement or box
      argument ever crosses into Python.
    status: pending
  - id: ac-011
    summary: protocol regression reproduces its golden with no inheritance
    type: runtime
    pass_when: |
      `python regressions/release-0-14-03-potential-protocol.py` exits 0, its
      custom potential inherits from nothing, it imports no third-party
      scientific package, and it matches its embedded 5-step total-energy
      golden to 1e-12 relative.
    status: pending
out_of_scope:
  - Rust trait layer redesign
  - Rust-side per-step pair data seam mechanics (spec 14)
  - extends= base-class surgery
  - driver dtype= shape
  - wasm / capi symmetry
---

# Acceptance — release-0-14-03-potential-protocol

`Potential` 是唯一类型且是结构化的：不继承也算数，导出面只有 Protocol 一个身份，具体 Rust 类型仍走快路径（臂序有咬合证明），Python 契约只收坐标（14 号的每步 pair 数据不越过 FFI），`.pyi` 里再无 Union。
