
---
spec: cgsmiles-02d-python-fragment
created: 2026-09-21
criteria:
  - id: ac-001
    summary: molrs.Fragment exists as a shadowed leaf with both kinds registered
    type: code
    pass_when: |
      `import molrs; f = molrs.Fragment()` succeeds, `type(f) is molrs.Fragment`,
      `isinstance(f, molrs.Graph) and isinstance(f, molrs.GraphViews)` are both
      True, `set(f.kinds()) >= {"bonds", "ports"}`, and
      `molrs._lib.Fragment.__module__ == "molrs._lib"`.
    status: pending
  - id: ac-002
    summary: three typed writers over the generic world, no hand-written relation API
    type: code
    pass_when: |
      In `molrs-python/src/core/system/molgraph.rs`, `PyFragment` is followed by
      `graph_world_impl!(PyFragment);`, its `#[pymethods]` block declares
      `add_atom`, `add_bond`, `add_port`, `n_ports`, `n_atoms`, `set_frag_id`,
      `frag_id`, `inherit_frag_ids`, `copy`, `to_frame`, `from_frame` and, besides `#[new]`, nothing else (no `port`, no `remove_port`, no `set_port_prop`), `n_atoms` delegates
      to `Fragment::n_atoms`, and `molrs.Fragment().remove_relation("ports", h)`
      removes a port.
    status: pending
  - id: ac-003
    summary: from_core is one plain generic function and all three leaves delegate
    type: code
    pass_when: |
      `rg 'py\.import\("molrs"\)' molrs-python/src/core/system/molgraph.rs`
      returns exactly one line, inside
      `fn from_core_shadowed<T>(py: Python<'_>, leaf: T) -> PyResult<Py<T>>`
      bounded by `T: PyClass<BaseType = PyGraph, Frozen = False>` and using
      `T::NAME`; no trait is introduced for it; and each of
      `PyAtomistic::from_core`, `PyCoarseGrain::from_core` and
      `PyFragment::from_core` has a body that is a single delegating call.
    status: pending
  - id: ac-004
    summary: every graph-out path yields the shadowed views class
    type: runtime
    pass_when: |
      `molrs-python/tests/test_views.py` and `tests/test_fragment.py` assert
      `type(result) is molrs.Fragment` for `frag.copy()`,
      `molrs.Fragment.from_frame(frag.to_frame())`, the values of
      `CGSmilesIR(...).to_fragment()` and the result of
      `Conformer(...).generate(frag)`, and all pass under
      `uv --directory molrs-python run --no-sync tox -e py`.
    status: pending
  - id: ac-005
    summary: ports, frag_id and inherit_frag_ids survive the Frame and pickle round trips
    type: runtime
    pass_when: |
      In `tests/test_fragment.py`, a fragment with one `"$"` port and a labelled
      atom returns `n_ports == 1`, `port["port_kind"] == "$"` and the same
      `frag_id` values after both `Fragment.from_frame(frag.to_frame())` and
      `copy()`; `inherit_frag_ids()` returns `1` and the degree-1 neighbour then
      reads the propagated id; in `tests/test_pickle.py`,
      `type(roundtrip(_lib.Fragment())) is _lib.Fragment` and a pickled
      `molrs.Fragment` returns with its port and `frag_id` intact.
    status: pending
  - id: ac-006
    summary: geometry systems move a Fragment's own atoms, not the empty base
    type: runtime
    pass_when: |
      In `tests/test_fragment.py`, after `molrs.translate(frag, (1.0, 0.0, 0.0))`
      on a fragment whose first atom was at x = 0.0, that atom reads x == 1.0;
      the test fails with a wrong value and no exception when the `PyFragment`
      arm is removed from `with_world_mut`.
    status: pending
  - id: ac-007
    summary: Conformer.generate returns the leaf type it was handed
    type: runtime
    pass_when: |
      `Conformer(speed="fast", seed=42).generate(frag)` returns
      `(molrs.Fragment, ConformerReport)` with `n_ports` unchanged and finite
      x/y/z (Å) on every atom; `generate(mol)` on an `Atomistic` still returns a
      `molrs.Atomistic`; `generate(object())` raises `TypeError` whose message
      names both `Atomistic` and `Fragment`.
    status: pending
  - id: ac-008
    summary: CGSmilesIR.to_fragment returns named public Fragments
    type: runtime
    pass_when: |
      `molrs.io.CGSmilesIR("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}").to_fragment()`
      returns a `dict` whose values satisfy `type(v) is molrs.Fragment`, with
      `result["PEO"].n_ports == 2` and `result["OH"].n_ports == 1`.
    status: pending
  - id: ac-009
    summary: the port vocabulary is the glyph, and def_port reaches core add_port
    type: runtime
    pass_when: |
      `frag.def_port(anchor, h, "$")` succeeds and `frag.ports[0]["port_kind"] == "$"`;
      both `frag.def_port(anchor, h, "Z")` and `frag.add_port(a, h, "Z")` raise
      `ValueError` (the first only holds if `def_port` reaches core `add_port`,
      since `_create_relation` would accept `"Z"`); `views.py`'s `def_port` body
      contains `self.add_port(`; `rg 'fn add_port' molrs-python/src` shows
      `kind: &str`; and `rg 'PortKind::from_code|port_kind.*u32' molrs-python/src`
      returns nothing.
    status: pending
  - id: ac-010
    summary: def_bond stamps both bond facts, as the native writer does
    type: runtime
    pass_when: |
      In `tests/test_fragment.py`, after `frag.def_bond(a, b)`,
      `frag.bonds[0]["bond_type"] == 1` and `frag.bonds[0]["bond_number"] == 1`;
      `views.py`'s `Fragment.def_bond` body contains `self.add_bond(`.
    status: pending
  - id: ac-011
    summary: views.Port endpoints do not collide with RelationRef.handle
    type: code
    pass_when: |
      `molrs-python/python/molrs/views.py` declares `Port` with endpoint
      properties `anchor` and `handle_atom` and no property named `handle`;
      `frag.ports[0].handle` returns the relation handle while
      `frag.ports[0].handle_atom` returns the capping-H `Atom`.
    status: pending
  - id: ac-012
    summary: stubs, docs page and docstrings match the new surface
    type: docs
    pass_when: |
      `molrs-python/python/molrs/_lib.pyi` contains `class Fragment(Graph)`
      listing every new pymethod and two `@overload`s of `Conformer.generate`
      (Atomistic→Atomistic, Fragment→Fragment); `molrs-python/docs/reference/python.md`
      contains `::: molrs.Fragment`; `tests/test_stub_parity.py` passes; the
      `generate` docstring names `Fragment` in both `Parameters` and `Returns`;
      `rg 'parse_smiles' molrs-python/src/conformer/mod.rs` and
      `rg 'graph_world_body' molrs-python/src` both return nothing.
    status: pending
  - id: ac-013
    summary: the full binder gate is green and the diff is confined to molrs-python
    type: code
    pass_when: |
      `cargo fmt --manifest-path molrs-python/Cargo.toml --check`,
      `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D warnings`,
      `cargo clippy --manifest-path molrs-wasm/Cargo.toml --target wasm32-unknown-unknown --all-targets -- -D warnings`
      and `uv --directory molrs-python run --no-sync tox -e py` all succeed, and
      `git diff --name-only <this link's base>..HEAD` lists only paths under
      `molrs-python/`.
    status: pending
out_of_scope:
  - The 0.15.0 version bump and the release checklist (cgsmiles-03-release).
  - Any change under molrs/src — Fragment, PortKind, to_fragment and the generic generate are 02a/02b/02c.
  - A validating port(id) reader, remove_port, core_mut, a PyPort pyclass.
  - merge / replicate / induced_subgraph / extract_subgraph for fragments and a PyExtractedSubgraph arm.
  - Force fields, MD, perception and typifiers accepting a Fragment; a Python-level Fragment to Atomistic promotion.
  - WASM / capi / cxxapi surfaces for Fragment.
  - Routing the pickle restore path (views.py:989-999) through add_port.
  - The three routed debts — views.Atomistic.def_bond writing classless bonds (/mol:fix), the n_nodes/n_atoms/n_beads triple spelling (/mol:refactor), and the dead parse_smiles / molrs.SmilesIR docstring names in molrs-python/src/io/mod.rs:2096,2167,2188 (/mol:docs).
  - Collapsing views.py's four monkey-patch pairs; caching the shadow-class lookup.
---

# Acceptance criteria

"Done" means a Python user can build a fragment, read and write its ports and
bonds through live views, move it, embed it and get a fragment back, and parse
one out of a CGsmiles string — with every one of those paths handing back the
same `molrs.Fragment` class, and with the `from_core` duplication removed rather
than tripled.

- **ac-001 / ac-002** pin the shape: one leaf, one data slot, the generic ECS
  surface from the macro, and exactly three typed writers beside it. ac-002 is
  the "no second door" check — a hand-written port relation API, or a
  `remove_port` with no core method behind it, fails it.
- **ac-003** is the iron-law item. The third leaf was allowed only because it
  removed the duplication it would otherwise have created, and it must do so
  with a plain function: if a trait, a registry or a second name for
  `#[pyclass(name = …)]` appears, the criterion fails.
- **ac-004 / ac-005** are the shadow-class and persistence contracts.
  `type(x) is molrs.Fragment` is the assertion that catches a bare-pyclass
  regression, checked on all four graph-out paths.
- **ac-006** is the silent-wrong-answer fix, written so that removing the
  `with_world_mut` arm fails on a *value* — which is how the bug would have
  reached a user.
- **ac-007 / ac-008** are the two new entry points, each asserting leaf-in /
  leaf-out and the error type for a bad argument.
- **ac-009 / ac-010** are the load-bearing routing decisions: both `def_port`
  and `def_bond` must reach the native writers, so that Python-built ports are
  validated and Python-built bonds carry their class. A `ValueError` from
  `def_port(..., "Z")` is only possible if the call reached core `add_port`.
- **ac-011** pins the endpoint name that avoids the `RelationRef.handle` slot
  collision.
- **ac-012 / ac-013** are the documentation and gate bars; ac-013 additionally
  asserts the blast radius — nothing outside `molrs-python/` moves.
```

Revision notes for the orchestrator (outside the documents):

- **🔴 1** — `ShadowedLeaf` is gone; `from_core_shadowed<T>` is a plain generic fn with the `Frozen = False` bound and `T::NAME`, and Task 3, the Reuse decision entry and ac-003 all say so. The "apologises" sentence now anchors on `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molrs/molrs-python/src/core/system/molgraph.rs:978-982` (verified: that is the only `from_core` commentary; `:1595-1603` is the Systems/leaf-first comment and `PyCoarseGrain::from_core` carries none).
- **🔴 2** — `def_port` routing is now criterion ac-009 with the `"Z"` → `ValueError` discriminator and the `self.add_port(` body check; the unverifiable "no integer code anywhere" clause is replaced by the two `rg` checks. Task 1 carries both rejection tests.
- **NEW (02a/02b reconciliation)** — `add_atom` / `add_bond` added to `PyFragment` with the `molgraph.rs:656,665,1439` templates and named as the production callers for 02a's three Rust methods; `views.Fragment.def_bond` routes through the native writer; `def_atom` stays on the generic node path (stated); the `bond_type == 1` / `bond_number == 1` assertion is Task 1 + ac-010. `views.Atomistic.def_bond` (`views.py:875-878`) writing classless bonds is named in a new *Debt found* section and routed to `/mol:fix`. Verified the prop keys are `"bond_type"` / `"bond_number"` (`molrs/src/core/store/schema/mod.rs:511`, written by `atomistic.rs:206-214`).
- **🟡 3, 4, 6, 7, 8, 9, 10** and **🟢 11–14** all applied. Anchors re-verified against the tree: `atomistic.rs:103-116`, `coarsegrain.rs:86-94`, `perceive.rs:95,126,142,161,179,200,226`, `ff/charge.rs:187,224,284,354`, `ff/scale_lj.rs:14,24`, `ff/mod.rs:242-247`, `_lib.pyi:841-877`, `conformer/mod.rs:224,229,239`, `molgraph.rs:16,749,1448`, `views.py:259,300,571-590,875-878,989-999`.
- One deviation worth flagging: `PyAtomistic` has no `n_bonds` pymethod, so `PyFragment` exposes `n_atoms` and `n_ports` only — a bond count is `len(frag.bonds)` / `n_relations("bonds")`. Criteria count went 12 → 13 to keep the new `def_bond` contract its own checkable line.
