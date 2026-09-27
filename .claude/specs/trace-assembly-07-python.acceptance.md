---
slug: trace-assembly-07-python
created: 2026-09-27
criteria:
  - id: ac-001
    summary: molrs.Trace is built by __init__ from (k,3) float64 points
    type: runtime
    pass_when: |
      `pytest molrs-python/tests/test_trace.py` passes: Trace(2x3) has len 2
      and float64 (2,3) points equal to the input; (0,3) -> len 0; (2,2) ->
      ValueError; frozen; `hasattr(molrs.Trace, "from_points")` is False.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-002
    summary: CoarseGrain.positions and bead_types cross the seam
    type: runtime
    pass_when: |
      `pytest molrs-python/tests/test_coarsegrain.py` passes the accessor
      cases: positions([h2,h0]) float64 (2,3) in order; bead_types list[str];
      stale handle -> ValueError.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-003
    summary: Coarsener(source).coarsen(groups, names) returns a CoarseGrain
    type: runtime
    pass_when: |
      `pytest molrs-python/tests/test_coarsener.py` passes: 2 beads, 1 bond,
      bead_types ["A","B"]; Atomistic source accepted; Frame source ->
      TypeError; overlap -> ValueError with an integer handle and no
      "NodeId(".
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-004
    summary: Perceive(graph).linear_paths() works and find_* is unchanged
    type: runtime
    pass_when: |
      `pytest molrs-python/tests/test_linear_paths.py` passes:
      Perceive(cg).linear_paths() == [[a,b,c],[d]]; star -> ValueError naming
      the centre; Perceive().linear_paths() -> TypeError;
      Perceive().find_rings(mol) returns an Atomistic.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-005
    summary: Assembler accepts Fragment and Atomistic values and stamps mol_id
    type: runtime
    pass_when: |
      `pytest molrs-python/tests/test_assembler.py` passes: type(world) is
      molrs.Fragment, n_atoms 7, n_ports 2, atoms mol_id values {1, 2} with
      the Li at 2; unknown name -> ValueError naming trace/unit;
      non-TracePlacer placer -> TypeError; TracePlacer() takes no arguments.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-006
    summary: ElementTypifier types by element and is single-homed
    type: runtime
    pass_when: |
      `pytest molrs-python/tests/test_element_typifier.py` passes: atoms type
      ["O","H","H"], bonds type ["H-O","H-O"]; molrs.ff.typifier.ElementTypifier
      exists and molrs.ff has no ElementTypifier attribute.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-007
    summary: Only the script's names are added, each with a stub
    type: code
    pass_when: |
      _lib.pyi declares Trace (__init__, points, __len__), CoarseGrain.positions,
      CoarseGrain.bead_types, Coarsener (__init__, coarsen), Perceive.__init__
      (graph=None), Perceive.linear_paths, TracePlacer (__init__), Assembler
      (__init__, assemble) and ElementTypifier, each with a docstring; no new
      #[pyfunction], #[classmethod] or Fragment.link_many is added; `pytest
      molrs-python/tests/test_stub_parity.py` passes.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-008
    summary: The operator seam test runs the script's shape
    type: runtime
    pass_when: |
      `pytest molrs-python/tests/test_backmap_seam.py` passes a test that sets
      coordinates with atoms["x","y","z"] = arr and then runs find ->
      Coarsener.coarsen -> Perceive.linear_paths -> Trace(positions) /
      bead_types -> Assembler(lib, TracePlacer()).assemble ->
      ElementTypifier().typify(world.to_atomistic()), asserting 3 groups, 3
      sites, sorted path lengths [1,2], n_atoms 7, n_ports 2, mol_id {1,2} and atom
      types within {"C","H","Li"}.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "maturin develop + 8 named test files: 24 passed; 10 neighbouring files 189 passed"
  - id: ac-009
    summary: Driver-class assembly takes under 60 s in release and scales linearly
    type: performance
    evaluator_hint: "one-off scratchpad script, maturin develop --release"
    pass_when: |
      With a release build, Assembler.assemble on 100x100 units of a 33-atom
      ported monomer + 10,000 Li + 450,000 13-atom PC singles returns in < 60
      s with n_atoms == 6,170,200; chain-only 5,000 vs 10,000 units give median
      t ratio <= 2.5 over 3 runs; numbers and host are recorded in notes.md
      2026-09-27.
    status: pending
  - id: ac-010
    summary: The molrs full gate passes once
    type: runtime
    pass_when: |
      On the final molrs tree `cargo fmt --check`, `cargo mrs-clippy -- -D
      warnings`, `cargo mrs-test`, `cargo mrs-doctest`, `uv --directory
      molrs-python run --no-sync tox -e py` and `prek run --all-files
      --hook-stage pre-push` exit 0, and no chain commit touches
      molrs/src/io/data/xyz.rs.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "fmt, mrs-clippy, mrs-test 3125, mrs-doctest 101, tox -e py and prek pre-push all green 2026-09-28; xyz.rs excluded"
  - id: ac-011
    summary: Testing notes follow the new names
    type: docs
    pass_when: |
      testing.md and the notes.md seam-exception entry name the stage list
      (tuple-key write -> find -> Coarsener.coarsen -> Perceive.linear_paths ->
      Trace -> Assembler.assemble -> ElementTypifier.typify) and still scope
      the exception to that one file; src/perceive.rs's module doc describes
      the held-graph Perceive and the Coarsener class.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
---

# Acceptance criteria

`ac-007` guards ruling (d). `ac-009` carries the operator's assembly bars.
Step-6 bars on the whole run are in trace-assembly-08.
