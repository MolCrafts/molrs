---
slug: release-0-14-05-compute-protocol
created: 2026-08-25
criteria:
  - id: ac-001
    summary: Compute is a runtime_checkable Protocol satisfied structurally
    type: runtime
    pass_when: |
      `from molrs.compute import Compute` succeeds, it is a runtime_checkable
      Protocol, and a class defining only compute() satisfies isinstance.
    status: pending
  - id: ac-002
    summary: existing kernels satisfy it unmodified
    type: runtime
    pass_when: |
      A scan-driven test over molrs.compute kernel classes shows they satisfy
      isinstance(k, Compute) with zero source changes to those kernels.
    status: pending
  - id: ac-003
    summary: the contract is compute() and nothing else
    type: code
    pass_when: |
      The Protocol declares only compute; neither __call__ nor dump is a
      contract member, and protocol.py contains no implementation of either.
    status: pending
  - id: ac-004
    summary: isinstance limits are documented
    type: docs
    pass_when: |
      protocol.py docstring states runtime_checkable isinstance checks method
      presence only (PEP 544) and is unsuitable for hot paths; a test pins that
      a wrong-signature object passes isinstance.
    status: pending
  - id: ac-005
    summary: the layout exception is recorded next to the invariant
    type: docs
    pass_when: |
      molrs-python/python/molrs/compute/__init__.py docstring names protocol.py
      as an explicit exception to "one subpackage per molrs::compute module",
      alongside the existing rdf / shape exceptions.
    status: pending
  - id: ac-006
    summary: ComputeResult and DescriptorRow are untouched
    type: code
    pass_when: |
      git diff for this spec shows no change to molrs/src/compute/result.rs and
      no Python wrapper class around ComputeResult / DescriptorRow.
    status: pending
  - id: ac-007
    summary: compute-protocol regression reproduces its golden
    type: runtime
    pass_when: |
      `python regressions/release-0-14-05-compute-protocol.py` exits 0, its
      analysis class inherits from nothing, it imports no third-party
      scientific package, and it matches its embedded first-nonzero-bin golden
      exactly.
    status: pending
out_of_scope:
  - molpy shell rework
  - __call__ / dump implementations
  - Fit / Check pipeline stages
  - analysis kernel algorithms
---

# Acceptance — release-0-14-05-compute-protocol

`Compute` 是结构化契约且只有 `compute` 一条：45 个内核零改动即满足，非 object-safe 的 Rust trait 不再是障碍，`__call__` / `dump()` 的归属随 molpy 薄壳裁定而定，不由 molrs 预先承诺。
