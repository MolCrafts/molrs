---
spec: frame-meta-dict-parity-04-dict-views
created: 2026-09-22
criteria:
  - id: ac-001
    summary: Clone-per-read map() is gone; one whole-map clone remains
    type: code
    pass_when: |
      molrs-python/src/core/store/frame.rs has no `fn map` on PyFrameMeta.
      A search for `f.meta.clone()` under molrs-python/src returns exactly one
      line, inside mapping_to_meta_map. The file contains no `typed_for`.
    status: pending
  - id: ac-002
    summary: with and with_mut closures do not call Python or tag_of
    type: code
    pass_when: |
      Every `with` and `with_mut` closure in the PyFrameMeta impl and
      pymethods block touches only MetaMap (get, contains_key, keys, iter,
      len, clone, insert, remove, extend, clear). None of those closures
      contains get_item, call_method, extract, meta_value_to_py,
      infer_meta_value, tag_of, or typed_for. absorb contains exactly one
      `f.meta.extend`. __setitem__, absorb, and setdefault do not call tag_of.
    status: pending
  - id: ac-003
    summary: keys/values/items are live collections.abc views
    type: runtime
    pass_when: |
      In molrs-python/tests/test_frame.py::TestFrameMeta, after writing t,
      then a, then stress, isinstance checks pass for collections.abc.KeysView,
      ValuesView, and ItemsView, and each view's type __module__ is
      collections.abc with __name__ KeysView, ValuesView, or ItemsView.
      list(m.keys()) == ["t", "a", "stress"] and list(m.values()) is the f32
      300.0, then 1, then [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]. After
      k, v, i = m.keys(), m.values(), m.items() and m["z"] = "late", "z" in k
      is True and len(k), len(v), len(i) are all 4. An empty meta's keys view
      has length 0 and bool False; a view held across clear() reports length 0.
    status: pending
  - id: ac-004
    summary: popitem returns the last inserted pair
    type: runtime
    pass_when: |
      On the t, a, stress fixture, popitem() returns the stress pair.
      On a fresh fixture, assigning m["t"] = 9 and then calling popitem()
      still returns the stress pair. Calling popitem until the map is empty
      and then once more raises KeyError.
    status: pending
  - id: ac-005
    summary: Iteration does not recurse and keeps insertion order
    type: runtime
    pass_when: |
      list(m), list(m.keys()), dict(m), and list(m.items()) each return
      without RecursionError, and list(m) == ["t", "a", "stress"] for the
      t-then-a-then-stress fixture. ["a", "stress", "t"] fails.
    status: pending
  - id: ac-006
    summary: Non-str lookup is absent; non-str write is TypeError
    type: runtime
    pass_when: |
      For a non-empty meta m: (1 in m) is False, m.get(1) is None,
      m.get(1, "d") == "d", m.pop(1, "d") == "d", 1 not in m.keys(),
      (1, 2) not in m.items(), and m.get([]) is None. m[1] and del m[1]
      each raise KeyError whose args are (1,). m[1] = 2 and
      m.setdefault(1, 2) each raise TypeError.
    status: pending
  - id: ac-007
    summary: View set algebra and membership match the mapping
    type: runtime
    pass_when: |
      For the t, a, stress fixture with a == 1 and t == 300.0:
      m.keys() & {"t"} == {"t"},
      m.keys() | {"z"} == {"t", "a", "stress", "z"},
      ("a", 1) in m.items(), ("a", 2) not in m.items(), and
      300.0 in m.values().
    status: pending
  - id: ac-008
    summary: update(self) does not panic or re-pin dtypes
    type: runtime
    pass_when: |
      Starting from t = MetaValue("f32", 300.0) and a = 1, both m.update(m)
      and m |= m return without pyo3_runtime.PanicException, list(m) stays
      ["t", "a"], the values still read back, dtype("a") == "i64", and
      dtype("t") == "f64".
    status: pending
  - id: ac-009
    summary: Delete during values() iteration raises KeyError
    type: runtime
    pass_when: |
      A TestFrameMeta test asserts pytest.raises(KeyError) around a loop
      that deletes a not-yet-visited key while iterating m.values(). The
      assertion names KeyError, not RuntimeError.
    status: pending
  - id: ac-010
    summary: Stub types the three views and the key rule
    type: code
    pass_when: |
      molrs-python/python/molrs/_lib.pyi imports ItemsView, KeysView, and
      ValuesView from collections.abc and, inside the FrameMeta block,
      declares keys() -> KeysView[str], values() -> ValuesView[Any], and
      items() -> ItemsView[str, Any]. __getitem__, __delitem__, get, and pop
      take key: object; __setitem__ and setdefault keep key: str. The
      FrameMeta docstring states live views, insertion order, the non-str
      key rule, and KeyError on delete-during-iteration, and it still says a
      plain write re-infers. tests/test_stub_parity.py passes, and the stub
      declares no new view class.
    status: pending
  - id: ac-011
    summary: Full check and the Python and Rust suites pass
    type: runtime
    pass_when: |
      `cargo fmt --check && cargo mrs-clippy -- -D warnings && cargo clippy
      --manifest-path molrs-cxxapi/Cargo.toml --all-targets -- -D warnings`,
      `cargo mrs-test && cargo mrs-doctest`, and
      `uv --directory molrs-python run --no-sync tox -e py` all pass.
    status: pending
  - id: ac-012
    summary: hash(frame.meta) raises TypeError
    type: runtime
    pass_when: |
      molrs-python/src/core/store/frame.rs sets
      `#[classattr] const __hash__: Option<Py<PyAny>> = None` on PyFrameMeta,
      and a TestFrameMeta test asserts pytest.raises(TypeError) around
      hash(frame.meta) for both an empty meta and a meta with one key.
    status: pending
  - id: ac-013
    summary: One private abc_view builds all three stdlib views
    type: code
    pass_when: |
      molrs-python/src/core/store/frame.rs has one private fn abc_view on
      PyFrameMeta. keys, values, and items each call it and do not themselves
      import. py.import("collections.abc") appears once, inside abc_view, and
      is not stored in a static, OnceLock, or field. No new public view type
      is declared, and views.py is unchanged.
    status: pending
out_of_scope:
  - "__reversed__ and fromkeys"
  - "A new public view type or any edit to views.py"
  - "molrs/src, including meta.rs and sequence.rs (link 07); molrs-capi; molrs-wasm; molrs-cxxapi"
  - "Restoring typed_for or tag_of on the write path"
  - "MetaDocument's hash and freezing read values (link 05); frame.meta's hash is this link"
  - "dict's RuntimeError on mutation during iteration, and TypeError for unhashable lookups"
  - "A notes-page edit or a migration page"
  - "A regressions/ script"
---

# Acceptance — frame-meta-dict-parity-04-dict-views

Done means `frame.meta.keys` / `values` / `items` are live `collections.abc` views in insertion order, built by one private `abc_view`, lookups treat a non-`str` as absent while writes reject it, no store-borrow closure runs Python, `popitem` is still the last inserted pair, `m.update(m)` neither panics nor pins an `f32` tag, and `hash(frame.meta)` raises `TypeError`. The `KeyError` raised when a not-yet-visited key is deleted during `values()` iteration is asserted in `TestFrameMeta` and stated on the `FrameMeta` docstring.

## AC-001 — Clone-per-read map() is gone; one whole-map clone remains

`PyFrameMeta::map` is the clone-per-read link 02 routed here. Deleting it leaves exactly one `f.meta.clone()`, in `mapping_to_meta_map`, which is the tag-preserving `frame.meta = other.meta` copy. `typed_for` stays deleted.

## AC-002 — with and with_mut closures do not call Python or tag_of

This is the panic guard, greppable. `absorb` infers outside the borrow and calls `extend` once. The write path does not read `tag_of`, so link 03's re-inference survives the borrow rewrite. `popitem`'s mutable closure only removes the last iterated pair.

## AC-003 — keys/values/items are live collections.abc views

The golden order is the insertion order `t`, `a`, `stress`, not `a`, `stress`, `t`. The type name `KeysView` in `collections.abc` rejects both a fresh list and `dict_keys`. Liveness is the write after the view is taken. Empty and `clear()` cover the length-0 edge.

## AC-004 — popitem returns the last inserted pair

`stress` is last inserted. Overwriting `t` must not change which pair `popitem` returns, because an existing key keeps its position. Empty `popitem` is still `KeyError`.

## AC-005 — Iteration does not recurse and keeps insertion order

Routing `__iter__` through the view's `keys()` is infinite recursion. This criterion is the witness that `__iter__` walks `MetaMap::keys` forward instead.

## AC-006 — Non-str lookup is absent; non-str write is TypeError

Both halves of the key rule, plus the unhashable lookup (`m.get([]) is None`) and the original object inside `KeyError.args`.

## AC-007 — View set algebra and membership match the mapping

`KeysView` and `ItemsView` are sets. The union's expected members are the inserted keys plus `z`, compared as a set.

## AC-008 — update(self) does not panic or re-pin dtypes

`m.update(m)` and `m |= m` are the re-entrant case. They must return, keep insertion order, and re-infer `f32` to `f64`. Surviving as `f32` means `tag_of` was put back on the write path.

## AC-009 — Delete during values() iteration raises KeyError

The key list is materialized up front, so a later `__getitem__` of a deleted key raises `KeyError`. The test names `KeyError` so a `RuntimeError` assertion cannot be substituted quietly.

## AC-010 — Stub types the three views and the key rule

The stub is the user-visible contract for the return types, the lookup/write split, and the `KeyError` divergence. Link 03's re-inference sentence stays. `test_stub_parity.py` fails if a new view class is exported.

## AC-011 — Full check and the Python and Rust suites pass

The project check, the Rust unit and doctest gate, and `tox -e py`. No capi or wasm gate is owed, because those trees are not edited.

## AC-012 — hash(frame.meta) raises TypeError

`__eq__` is by value, so an unset `__hash__` is identity hashing. The classattr is `#[classattr] const __hash__: Option<Py<PyAny>> = None`, the slot link 05 spells `#[classattr] const __hash__ = None`. Link 05 still sets that classattr on `MetaDocument`; a later `hash(frame.meta)` check is a non-regression of this criterion.

## AC-013 — One private abc_view builds all three stdlib views

Three call sites is the second use, so the `collections.abc` import lives in one private method. The accessors call it. The import is not cached, and `views.py` is not edited.

There is no `regressions/` criterion. molrs has no such tree (`molrs/Cargo.toml:14-18`, `CLAUDE.md` § Build & Test Commands, `.claude/specs/cgsmiles-03-release.md:179`). The executing bars are AC-003 through AC-009 and AC-012, run by `tox -e py`.
