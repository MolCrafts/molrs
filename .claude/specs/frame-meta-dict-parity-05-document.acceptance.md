---
spec: frame-meta-dict-parity-05-document
created: 2026-09-22
criteria:
  - id: ac-001
    summary: Every fixed-length vector door hands back a tuple
    type: code
    pass_when: |
      In molrs-python/src/core/store/frame.rs, meta_value_to_py's sequence arm
      and json_to_py's array arm build a PyTuple; infer_meta_value has a tuple
      branch beside its PyList branch and py_to_json has a tuple arm; and
      tests/test_frame.py asserts a tuple (not a list) for a vector dtype read
      through __getitem__, values, items, get, pop, popitem, setdefault and
      copy, with test_exact_dtype_roundtrip's :134 assertion reading
      (1.0, 2.0, 3.0, 4.0, 5.0, 6.0).
    status: pending
  - id: ac-002
    summary: MetaDocument exists with its full surface written out
    type: code
    pass_when: |
      molrs-python/src/core/store/frame.rs defines a frozen MetaDocument
      pyclass whose #[pymethods] block contains __getitem__, __len__,
      __iter__, __contains__, keys, values, items, get, __eq__, __ne__,
      __repr__ and copy, and defines no __setitem__, __delitem__, update, pop,
      popitem, clear, setdefault, __or__ or __ior__; frame.meta[k] for a
      json-dtype object returns it; and doc.copy() returns a dict that
      json.dumps accepts with no nested MetaDocument left in it.
    status: pending
  - id: ac-003
    summary: MetaDocument and FrameMeta are unhashable, like dict
    type: code
    pass_when: |
      Both PyMetaDocument and PyFrameMeta declare
      `#[classattr] const __hash__: Option<Py<PyAny>> = None;`, no sentence in
      the repo claims PyO3 sets __hash__ from __eq__, and a single test
      function in tests/test_frame.py asserts pytest.raises(TypeError) for
      hash(doc), hash(frame.meta) and hash({}).
    status: pending
  - id: ac-004
    summary: doc == dict is True while isinstance(doc, dict) is False
    type: code
    pass_when: |
      One test function in tests/test_frame.py asserts, for the same document,
      doc == {"tool": "molrec", "run": 3}, isinstance(doc, dict) is False,
      isinstance(doc, collections.abc.Mapping) is True,
      doc != {"tool": "other"} is True and (doc != equal_dict) is False; and
      tests/test_frame.py:140 passes unchanged.
    status: pending
  - id: ac-005
    summary: Pickling a Frame with a json meta key still round-trips
    type: runtime
    pass_when: |
      `uv --directory molrs-python run --no-sync tox -e py` passes with
      tests/test_pickle.py:52-57 and :297-302 unedited, MetaValue.value and
      MetaValue.__reduce__ decoding through JsonForm::Plain, and new
      assertions that MetaValue("json", {...}).value is a plain dict, that
      MetaValue("f64x6", [1,2,3,4,5,6]).value is (1.0, 2.0, 3.0, 4.0, 5.0,
      6.0) after a round-trip, and that a MetaDocument read off a restored
      frame equals the original dict.
    status: pending
  - id: ac-006
    summary: Identity and bulk doors still round-trip after the freeze
    type: code
    pass_when: |
      tests/test_frame.py asserts frame.meta[k] = frame.meta[k] preserves
      value and dtype for a scalar, an f64x6 and a json key, and that
      dict(f.meta), {**f.meta} and g.meta = dict(f.meta) preserve every key
      with g.meta.dtype(k) == f.meta.dtype(k) for those three dtypes.
    status: pending
  - id: ac-007
    summary: The swallowed nested write now raises, and .copy() is the way back
    type: code
    pass_when: |
      test_nested_document_is_a_snapshot in tests/test_frame.py is replaced by
      a test asserting pytest.raises(TypeError) for f.meta["run"]["step"] = 2
      and for the held-document form, then doc = f.meta["run"].copy();
      doc["step"] = 2; f.meta["run"] = doc; f.meta["run"] == {"step": 2}; and
      no test in the repo performs an in-place write through a value read from
      frame.meta.
    status: pending
  - id: ac-008
    summary: MetaDocument is exported, ABC-registered and stubbed
    type: code
    pass_when: |
      molrs.MetaDocument imports; molrs-python/src/lib.rs adds the class
      beside PyMetaValue; molrs-python/python/molrs/__init__.py lists it in
      the import block and __all__ and calls Mapping.register(MetaDocument)
      beside MutableMapping.register(FrameMeta) with a comment naming
      molvis/python/src/molvis/wire.py:389,530 and
      molrec/tests/molrs_adapter.py:110-113; _lib.pyi declares the class plus
      the corrected MetaValue.value and the ten FrameMeta return types; and
      tests/test_stub_parity.py passes unchanged.
    status: pending
  - id: ac-009
    summary: The rule, the json.dumps asymmetry and the order caveat are documented
    type: docs
    pass_when: |
      Every doc site states that every door of frame.meta hands back frozen
      values and that the top-level container copy() returns is the only
      mutable thing, names the json.dumps asymmetry with the
      json.dumps(frame.meta["run"].copy()) idiom, and says nested document
      order is unspecified: the PyFrameMeta rustdoc and the Frame.meta getter
      docstring in molrs-python/src/core/store/frame.rs, the FrameMeta
      docstring and MetaValue.value in _lib.pyi, Frame.to_dict's docstring in
      python/molrs/frame.py, and a `::: molrs.MetaDocument` entry in
      docs/reference/python.md. This link does not edit migration-0-14.md.
    status: pending
  - id: ac-010
    summary: The bulk-door warrant is restated and the cross-repo items routed
    type: docs
    pass_when: |
      The implementation summary states that molrs-python frame.py:639 and
      :662-663, molpy pdb.py:66, molpy amber.py:74 and molrec
      molrs_adapter.py:110,260 were each checked and round-trip, and names two
      molrec follow-ups — zarr.py:783-785 breaking on json.dumps of a document
      value (with the .copy() / .typed() idioms) and molrs_adapter.py:259-261
      being already stale independent of this link. Neither follow-up is
      gated on a molrs tag or a later minor. .claude/notes/notes.md carries
      those two plus the serde_json preserve_order dependence and the
      three-way json_to_py triplication routed to /mol:refactor.
    status: pending
  - id: ac-011
    summary: Full gate green on the changed tree
    type: runtime
    pass_when: |
      `cargo fmt --check`, `cargo mrs-clippy -- -D warnings`,
      `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets --
      -D warnings`, `cargo mrs-test && cargo mrs-doctest` and
      `uv --directory molrs-python run --no-sync tox -e py` all exit 0, with
      no test skipped, xfailed or weakened relative to the pre-change tree.
    status: pending
---

# Acceptance criteria

**ac-001 · ac-002 — the uniform rule.** One decoder changes, so every door changes with it. The criteria name the arms (`meta_value_to_py`, `json_to_py`, `infer_meta_value`, `py_to_json`) because a freeze applied per door would be the wrong shape even if it passed the same assertions, and name the eight doors because the rule is about all of them, not about `frame.meta[k]`.

**ac-003 — the `__hash__` slot.** pyo3 0.28 wires `Py_tp_hash` only when `__hash__` or `#[pyclass(hash)]` is present, and nothing clears it because `__eq__` exists. Omitting the `classattr` yields identity hashing, under which two equal documents hash differently and `doc in some_set` is silently `False`. `hash({})` is asserted in the same function so the bar reads as dict parity. `FrameMeta` carries the same defect today and is fixed in the same edit.

**ac-004 — registration supplies no methods.** `Mapping.register()` is virtual: it grants `isinstance` and inherits no mixin, so `__eq__`, `__ne__`, `__contains__` and `get` must be written out. Both halves of the claim are asserted in one function so neither can drift alone. `__ne__` is pinned in both directions because a richcompare slot answering only `Py_EQ` lets `!=` fall back to identity.

**ac-005 — the reduce path.** `_frame_ctor_args` pickles through `frame.meta.typed()`, and `MetaValue.__reduce__` packs `self.value`. Decoding that one call site with `JsonForm::Plain` is what keeps `test_pickle.py:52-57` and `:300-302` green **unedited**, which is why the criterion requires those lines to be unchanged: they are the regression guard, not the thing under repair.

**ac-006 · ac-007 — what did and did not break.** The identity `frame.meta[k] = frame.meta[k]` and the bulk doors are the contract this link promises to keep; the in-place nested write is the contract it promises to break, loudly. Both are pinned so the pair cannot be half-delivered.

**ac-009 — the doc sites.** The rustdoc currently states the opposite ("mutates a copy"). The live docs state the frozen-value rule. This link does not edit a migration guide.

**ac-010 — the warrant is not a grep.** A `.meta[` pattern cannot see `dict(self.meta)`, `{**frame.meta}` or `out.meta = dict(frame.meta)`; the five bulk sites are checked by hand across molrs, molpy and molrec. The two molrec items are named rather than edited here — one is caused by this link, the other was already stale — because they are another repo's. They are not gated on a tag. Silence about either would be the process failure the iron law names.

There is no `regressions/` criterion: molrs has no such tree (`CLAUDE.md` § Build & Test Commands; precedent `.claude/specs/cgsmiles-03-release.md:177`). The executable public-API bars are ac-005, ac-006 and ac-007, all run by `tox -e py` in the standing gate.
