---
title: "FrameMeta dict parity — live mapping views, one key-acceptance rule, read/write borrow split"
slug: frame-meta-dict-parity-04-dict-views
status: in-progress
created: 2026-09-22
revised: 2026-09-22
chain: frame-meta-dict-parity (01-ordered → 02-binder-order → 03-untyped-write → 04-dict-views → 05-document → 07-sequence)
depends_on:
  - frame-meta-dict-parity-01-ordered
  - frame-meta-dict-parity-02-binder-order
  - frame-meta-dict-parity-03-untyped-write
---

# FrameMeta dict parity — live mapping views, one key-acceptance rule, read/write borrow split

## Summary

`frame.meta` is registered as a `MutableMapping`, but `keys()`, `values()` and `items()` each return a fresh list, so a handle taken before a write goes stale and `m.keys() & {"a"}` raises. This link makes those three accessors return live `collections.abc` views of the same `FrameMeta`, applies one key rule (a lookup of a non-`str` is absent; a write of a non-`str` is `TypeError`), and splits reads from writes so neither `with` nor `with_mut` runs Python — the rewrite that would otherwise make `m.update(m)` panic by re-entering the store borrow. `PyFrameMeta::map`'s clone-per-read is deleted. Order stays the insertion order links 01 and 02 already established: views and `__iter__` walk `MetaMap::keys` / `iter` forward with no sort, and `popitem` stays last-inserted, only collapsed into one `with_mut`. Deleting a not-yet-visited key while iterating `values()` or `items()` raises `KeyError`; that divergence is behavior, covered by tests and by the `FrameMeta` docstring. `__eq__` is by value and `__hash__` is unset, so `hash(frame.meta)` is identity today; this link sets the class unhashable, and `hash(frame.meta)` raises `TypeError`. This link lands in the unreleased 0.15 tree. It adds no version literal, no pin window, and no migration page.

## Domain basis

None. This link changes Python mapping mechanics at the binder seam — views, key types, and borrow scope. No equation, unit, or physical quantity is involved, so `$META.science.required` raises no domain-basis obligation here. Stated rather than silently omitted.

## Design

**Constitution.** There is no `.claude/notes/law.md` in this tree. The binding design rules are the `mol:bootstrap` block of `CLAUDE.md` (iron law — no silent debt; Prefer / Forbid; Shape check; Tests) together with `.claude/notes/architecture-rules.md` and `.claude/notes/testing.md`. This link adds no public type. `keys` / `values` / `items` stay methods on `PyFrameMeta`, the type that owns the mapping. The one shared constructor is a private method on that type, not a free function. No factory, no second view class, no new field.

**Chain position.** **01** makes `MetaMap` insertion-ordered (`iter` / `keys` forward; `remove` keeps the order of the survivors; re-inserting an existing key keeps its position). **02** deletes the binder-side sorts, so Python, wasm, C, and C++ enumerate that order, and `popitem` is last-inserted. **03** deletes `PyFrameMeta::typed_for`; `__setitem__`, `absorb`, and `setdefault` call `infer_meta_value` (`molrs-python/src/core/store/frame.rs:869`) and do not read the stored dtype to coerce a write. This link depends on all three. It must not put a sort back, must not add a method on `MetaMap`, and must not call `typed_for` or `tag_of` on the write path. `tag_of` (`frame.rs:128`) stays the implementation of `dtype()` only.

Line numbers below are the tree before 02 and 03 land. After those links, `keys` no longer sorts (`:238-242`) and `typed_for` (`:134-147`) is gone. The edits here are against that result.

### What is deleted: `PyFrameMeta::map`

`map` (`frame.rs:122-126`) clones the whole `MetaMap` on every read. It goes away. No replacement method is added.

- **One whole-map clone remains, inlined at its only caller.** `mapping_to_meta_map` (`:78-83`) copies another `PyFrameMeta` tags and all, because `frame.meta = other_frame.meta` must not re-infer. That arm becomes `view.inner.with(|f| f.meta.clone())` directly. A search for `f.meta.clone()` under `molrs-python/src` finds that one line. This is the copy path link 03 left intact; it is not a write-path dtype lookup.
- **Single-key reads clone one `MetaValue`.** `__getitem__`, `__contains__`, `get`, `dtype` / `tag_of`, and `setdefault`'s presence check take one `with` (`FrameRef::with` → `store.borrow()`, `molrs-ffi/src/shared.rs:86-88`), copy the one value or a `bool` out, and convert with `meta_value_to_py` (`frame.rs:783`) only after the closure returns. `setdefault`, on a miss, calls `infer_meta_value` outside any borrow, then one `with_mut` that only `insert`s. It does not call `tag_of` to coerce the default, and it does not call `__getitem__` while the mutable borrow is held.
- **Bulk snapshots clone pairs once.** `as_dict` (`:163-169`) and `typed` (`:345-359`) take one `with`, collect `Vec<(String, MetaValue)>` from `MetaMap::iter` in forward order, release, then build the `PyDict`. `copy`, `__eq__`, `__repr__`, `__or__`, and `__ror__` keep going through `as_dict`. Their order is therefore the same insertion order as iteration, which is what link 02 already required of `dict(m)` and `repr(m)`.

### Borrow split

`with_mut` is `store.borrow_mut()` (`molrs-ffi/src/shared.rs:95-97`) on the one `RefCell` that holds every frame. `RefCell` allows many shared borrows and one mutable borrow. A Python call inside either closure can re-enter `frame.meta` and panic. So:

- A `with` closure only touches `MetaMap` (`get`, `contains_key`, `keys`, `iter`, `len`, `clone`). `meta_value_to_py` runs after it returns.
- A `with_mut` closure only touches `MetaMap` (`insert`, `remove`, `extend`, `clear`). `infer_meta_value` runs before it is entered. No `Bound`, `get_item`, `call_method`, `extract`, `meta_value_to_py`, `infer_meta_value`, `tag_of`, or `typed_for` appears inside either closure.

`absorb` (`:172-189`) is the case that forces the rule. It walks an arbitrary mapping — including `self`, via `m.update(m)` and `m |= m` — and today interleaves `get_item` with a per-key `store`. Folding that loop into one `with_mut` panics: the iteration calls `__iter__` / `__getitem__`, which take `borrow()` under an active `borrow_mut()`. That panic is not what the current sequential code does; it is what a naive batching would do. `.claude/notes/ffi.md` Rule 1 forbids a panic on this seam. The rewrite therefore builds `Vec<(String, MetaValue)>` with **no borrow held**, calling `infer_meta_value` on each plain value, then enters `with_mut` once and calls `f.meta.extend(pairs)`. Both arms (mapping-with-`keys`, and iterable of pairs) share that shape. `update`'s positional argument and its `kwargs` each call `absorb` once, so each is one `extend`, not one `extend` per key. `with_frame_mut` snapshots every block key on every `with_mut` (`molrs-ffi/src/store.rs:126-149`); batching is what drops that from N passes to one per `absorb`.

Because `get_item` on a `FrameMeta` returns the plain decoded value, `m.update(m)` re-infers, exactly as link 03's `m[k] = m[k]` does. An `f32` value read back as a Python `float` is stored as `f64`. The borrow rewrite must not "fix" that by reading `tag_of` or by special-casing `PyFrameMeta` inside `absorb`. Tag-preserving copy of a whole meta stays the `mapping_to_meta_map` arm above, and `typed()`.

`popitem` (`:285-292`) after link 02 pops the last key of `keys()` and then `take`s it — two borrows. Once `keys()` returns a view, that body does not compile. It becomes one `with_mut`: `MetaMap::iter().next_back()` (link 01's `MetaIter` is a `DoubleEndedIterator`), clone the owned pair, `remove` that key, return the pair, and call `meta_value_to_py` outside. Empty meta raises `KeyError` with the existing message `popitem(): metadata is empty`. The closure returns `Option`; it does not `expect`. Last-of-iteration is last-inserted, including after an overwrite of an earlier key, because link 01 keeps an existing key's position. No sort, and no new `MetaMap` method.

`__setitem__` stays: `infer_meta_value` outside, then one insert. `__delitem__`, `pop`, and `clear` already mutate inside `with_mut` with no Python; they keep that shape.

### Views

`keys` / `values` / `items` return `collections.abc.KeysView` / `ValuesView` / `ItemsView` constructed on the `FrameMeta` itself. The three accessors are the second use of one construction, so they call one private method on `PyFrameMeta` — `abc_view(slf, py, class_name)` — and do not each repeat the import. The method is not a `#[pymethods]` item and not a Python type. Its body is the per-call import:

`py.import("collections.abc")?.getattr(class_name)?.call1((slf,))`

with `slf: &Bound<'_, Self>`. The import stays inside the method and is not stored on a static, `OnceLock`, or field, for the same reason `from_core_shadowed` (`molrs-python/src/core/system/molgraph.rs:635-638`) imports per call: a cached module handle would hide a later rebinding. `py.import("collections.abc")` appears once, in `abc_view`. Each accessor takes **zero** store borrows; `abc_view` takes none either. `collections.abc` is not imported anywhere in this repo today; the stub gains the import. No new Python class is added. `tests/test_stub_parity.py` compares class names only, so a new pyclass would fail it and a signature change will not.

`_DictView.keys` (`molrs-python/python/molrs/views.py:61`) returns the underlying mapping's live view (`self.data.keys()`). That is the idea to copy. `PyFrameMeta` is the mapping, so it returns the stdlib view of `slf`, not a subclass of `_DictView` and not `views.py` edited.

**`__iter__` must not call `keys()`.** Today it is `PyList::new(py, self.keys()?)` (`:221-223`). `KeysView.__iter__` is `yield from self._mapping`, so `__iter__` → `keys()` → `KeysView.__iter__` → `__iter__` does not terminate. `__iter__` takes one `with`, collects `MetaMap::keys()` into a `Vec<String>` in forward order, and returns an iterator over that list. It does not sort the vector. `as_dict` and `typed` use `MetaMap::iter`, not `keys()`, for the same reason.

The materialized key list is required by the borrow rule: an iterator that held `with` across Python yields would panic on the next write. It is also why mutation during iteration does not match `dict`.

| Call | Store borrows |
|---|---|
| `m.keys()` / `m.values()` / `m.items()` | 0 |
| `len(view)`, `k in m.keys()` | 1 |
| iterating `m.keys()` | 1 (`__iter__`) |
| iterating `m.values()` / `m.items()` | 1 (`__iter__`) + 1 per key (`__getitem__`) |
| `(k, v) in m.items()` | 1 (`__getitem__` inside `KeyError`) |
| `dict(m)` / `m.copy()` / `m.typed()` | 1 |

The extra borrows on `values()` / `items()` iteration are sequential and shared. They cannot panic. They are the cost of a live view; the bulk readers keep the single-borrow snapshot.

A view holds the `FrameMeta`, which holds a `FrameRef`, so it reads the live map after the name `frame` is dropped and after `m["z"] = "late"`. `type(m.keys()).__module__` is `collections.abc` and `__name__` is `KeysView` (the same for `ValuesView` and `ItemsView`) — not `dict_keys`, which would be a snapshot of some other dict, and not a molrs class.

### One key-acceptance rule

A lookup can answer "absent" for a key the map cannot hold. A write cannot store that key, so it refuses.

- Lookups — `__getitem__`, `__contains__`, `get`, `pop`, `__delitem__` — take `&Bound<'_, PyAny>`. `extract::<String>()` failing means absent: `1 in m` is `False`, `m.get(1)` is `None`, `m.get(1, "d")` and `m.pop(1, "d")` return `"d"`, `m[1]` and `del m[1]` raise `KeyError` whose argument is the original object `1`, not the string `"1"`. A missing `str` still raises `KeyError` from `__getitem__`, `__delitem__`, and `pop` without a default. `__contains__` is already declared `key: object` in the stub (`_lib.pyi:348`); the Rust signature is what still takes `&str` (`frame.rs:213`).
- Writes — `__setitem__`, `setdefault` — keep `&str` in Rust and `key: str` in the stub, so a non-`str` raises `TypeError` from extraction. Do not widen them and re-raise.

`KeysView.__contains__` is `key in self._mapping`. `ItemsView.__contains__` looks up `self._mapping[key]` inside `try/except KeyError`. Both match `dict` only if a non-`str` lookup is a `KeyError` / `False`, not a `TypeError`.

Hashability is not consulted, because no Python object is stored as a key. `m.get([])` is `None` and `[] in m` is `False`, where `dict` raises `TypeError`. That is the lookup divergence the rule buys.

### Unhashable, fixed in this link

`PyFrameMeta` defines `__eq__` (`frame.rs:225`) and no `__hash__`. pyo3 0.28 wires `Py_tp_hash` only when `__hash__` or `#[pyclass(hash)]` is present; with `__eq__` set and the slot left unset, `hash(frame.meta)` succeeds by identity while equality is by value, so two metas that compare equal hash apart and `m in {m}` is not the value-semantics question `dict` answers with `TypeError`. This link already edits the class. Iron law: fix it here, do not leave it for a later link.

The classattr is the one link 05 spells `#[classattr] const __hash__ = None`, written so the slot typechecks:

`#[classattr] const __hash__: Option<Py<PyAny>> = None;`

on `PyFrameMeta`'s `#[pymethods]` impl. `hash(frame.meta)` raises `TypeError`, the same as `hash({})`. Link 05 still owns `MetaDocument`'s hash and sets that same classattr on the new type. A `hash(frame.meta)` assertion in link 05 is a non-regression of this link, not a second policy.

### Insertion-order goldens

The discriminating fixture writes three keys in a non-alphabetical order: `t`, then `a`, then `stress`, with values `MetaValue("f32", 300.0)`, `1`, and `MetaValue("f64x6", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])`.

- `list(m)`, `list(m.keys())`, `list(m.items())` keys, and `list(dict(m))` are `["t", "a", "stress"]`. The alphabetical permutation `["a", "stress", "t"]` fails.
- `list(m.values())` is `[pytest.approx(300.0), 1, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]`.
- `popitem()` returns the `stress` pair. Overwriting `m["t"]` afterwards does not move `t`; `popitem()` still returns `stress`.

Link 02's `test_mapping_protocol` correction (non-alphabetical fixture, `list(f.meta)` rather than `sorted(f.meta)`) must stay green. Link 03's re-inference tests must stay green: a plain write does not keep an `f32` tag.

### Docstring rot in the file this link already edits

Link 03 rewrites the `PyFrameMeta` class rustdoc (`frame.rs:100-115`) and the `FrameMeta` stub docstring (`_lib.pyi:333-343`): a plain write re-infers, `MetaValue` pins that write only. This link appends to those two texts; it does not replace the dtype paragraph. The sentences added are: `keys` / `values` / `items` are live `collections.abc` views in insertion order; a non-`str` lookup is absent and a non-`str` write raises `TypeError`; deleting a not-yet-visited key while iterating `values()` or `items()` raises `KeyError`.

The `PyFrame.meta` getter rustdoc (`frame.rs:590-596`) still says an existing key's dtype is kept. That sentence is false once link 03 has landed, and this link has the file open. Correct that sentence to link 03's wording. Do not restore the typed-slot rule anywhere in the file.

### Shape check

1. The owning type is `PyFrameMeta`. View accessors are methods on it. `MetaMap` is not given a new method; it already iterates in order.
2. Each accessor does one thing: hand back the stdlib view. Batching `absorb` is the same method it already is, not a new pipeline.
3. The whole-map clone has one caller and stays inline. The view construction has three call sites, which is the second use, so it is one private method on `PyFrameMeta` (`abc_view`). That method is not a public symbol and not a type.
4. `PyFrameMeta` keeps the single field `inner: FrameRef`.

### Breaking change

`keys()` / `values()` / `items()` stop returning `list`. Indexing or in-place sorting the result breaks. Iteration, comprehensions, `len`, and `in` keep working. Experimental 0.15, and `architecture-rules.md` § *Naming*: the return type changes, it is not wrapped in a compatibility alias. `MutableMapping.register(FrameMeta)` (`molrs-python/python/molrs/__init__.py:100`) stays as it is; a virtual registration does not supply `fromkeys` or the view mixins, which is why the views are built explicitly.

### Reuse decision

- `reuse MetaMap::keys` / `MetaMap::iter` / `MetaMap::extend` (`molrs/src/core/store/meta.rs:343-354`, insertion-ordered after link 01). `__iter__` collects `keys()` forward. Snapshots and `popitem` use `iter` (`next_back` for the last pair). `absorb` calls `extend` once. No sort. No new method on `MetaMap`.
- `generalize PyFrameMeta::map` — delete the clone-per-read. Single-key reads borrow one value; bulk readers borrow one snapshot; the one whole-map clone is inlined in `mapping_to_meta_map`. Not a second clone helper.
- `generalize PyFrameMeta::absorb` — link 03 already removed `typed_for`. This link makes the conversion happen outside any borrow and replaces the per-key `store` loop with one `extend`. Do not put `typed_for` or `tag_of` back.
- `typed_for` — pattern only, do not imitate. Its `tag_of` arm was the typed-slot rule link 03 deleted. Copying it would coerce writes again.
- `_DictView` (`molrs-python/python/molrs/views.py:61`) — pattern: return the underlying mapping's live view. Do not subclass it, do not edit `views.py`, do not add a public view type. The return is the stdlib `collections.abc` view of `PyFrameMeta`.
- `reuse MutableMapping.register(FrameMeta)` (`molrs-python/python/molrs/__init__.py:100`). Unchanged. Do not edit `__init__.py`.
- `reuse infer_meta_value` and `meta_value_to_py`. The borrow split only moves when they run.
- `reuse tag_of` for `dtype()` only.
- `reuse ffi_error_to_pyerr` (`molrs-python/src/store.rs`) for store failures. No new error mapping.
- `molrs/src/io/zarr/sequence.rs` — not used. Link 07 owns sequence schema. This link does not edit it.

## Files to create or modify

- `molrs-python/src/core/store/frame.rs` — delete `map`; inline the one whole-map clone in `mapping_to_meta_map`; read paths clone one value or one pair snapshot and convert outside the borrow; `__iter__` walks `MetaMap::keys` forward and does not call `keys()`; `absorb` infers outside any borrow and calls `extend` once; `popitem` is one `with_mut` over `iter().next_back()`; lookup methods accept any key object; private `abc_view` returns the stdlib view and `keys` / `values` / `items` call it; `#[classattr] const __hash__: Option<Py<PyAny>> = None` on `PyFrameMeta`; append the view / key-rule / `KeyError` sentences to the `PyFrameMeta` rustdoc without replacing link 03's dtype paragraph; correct the `PyFrame.meta` getter sentence at `:590-596`.
- `molrs-python/python/molrs/_lib.pyi` — import `ItemsView`, `KeysView`, `ValuesView` from `collections.abc`; change `keys` / `values` / `items` to those returns; type lookup keys as `object` (`__getitem__`, `__delitem__`, `get`, `pop`); leave `__setitem__` and `setdefault` as `key: str`; append the same view, key-rule, and `KeyError` sentences to the `FrameMeta` docstring, keeping link 02's insertion-order / last-inserted `popitem` sentences and link 03's dtype paragraph.
- `molrs-python/tests/test_frame.py` — seam tests on `TestFrameMeta` (`:124`).

No new files. `molrs/src/core/store/meta.rs`, `molrs-capi`, `molrs-wasm`, and `molrs/src/io/zarr/sequence.rs` are not in this list.

## Tasks

- [x] Write failing seam tests in `molrs-python/tests/test_frame.py::TestFrameMeta` for live `collections.abc` views, the insertion-order golden (`t`, then `a`, then `stress`), and LIFO `popitem` returning the `stress` pair even after an earlier key is overwritten
- [x] Write failing seam tests in `molrs-python/tests/test_frame.py::TestFrameMeta` for the non-`str` key rule, for `m.update(m)` and `m |= m` returning without `PanicException` while an `f32` tag re-infers to `f64`, for `KeyError` (not `RuntimeError`) when a not-yet-visited key is deleted during `values()` iteration, and for `hash(frame.meta)` raising `TypeError`
- [x] Generalize `PyFrameMeta::map` in `molrs-python/src/core/store/frame.rs` by deleting the clone-per-read: single-key reads clone one `MetaValue` inside `with` and convert outside; `as_dict` and `typed` take one `MetaMap::iter` snapshot; `mapping_to_meta_map` inlines the single `f.meta.clone()`; `__iter__` collects `MetaMap::keys` forward inside one `with` and does not call `keys()` or sort; set `#[classattr] const __hash__: Option<Py<PyAny>> = None` on `PyFrameMeta`; correct the `PyFrame.meta` getter rustdoc at `:590-596` to link 03's re-inference wording
- [x] Generalize `PyFrameMeta::absorb` in `molrs-python/src/core/store/frame.rs` so both arms call `infer_meta_value` outside any borrow and then `MetaMap::extend` once inside one `with_mut`, and collapse `popitem` to one `with_mut` that removes the last pair from `MetaMap::iter` (`next_back`) and converts the value afterwards; no `with` or `with_mut` closure calls Python, `tag_of`, or `typed_for`
- [x] Add private `abc_view` on `PyFrameMeta` in `molrs-python/src/core/store/frame.rs` and return `collections.abc.KeysView` / `ValuesView` / `ItemsView` from `keys` / `values` / `items` by calling it (one per-call `collections.abc` import, no cached module, zero store borrows), and declare those returns plus the live-view, insertion-order, and delete-during-iteration sentences in `molrs-python/python/molrs/_lib.pyi` without replacing the link 03 dtype paragraph
- [x] Widen `__getitem__` / `__contains__` / `get` / `pop` / `__delitem__` in `molrs-python/src/core/store/frame.rs` and the matching signatures in `molrs-python/python/molrs/_lib.pyi` so a non-`str` key is absent (`KeyError` carrying the original object on subscript and delete), while `__setitem__` and `setdefault` stay `str` and raise `TypeError`
- [ ] Run full check + test suite

## Testing strategy

Per `.claude/notes/testing.md` and `CLAUDE.md` § Testing Rules: bindings prove the seam only. No numeric science is re-derived, there is no third-party oracle, and there is no source-text gate. Nothing is added under `molrs/src/`, so there is no new `#[cfg(test)]` module. Each test below is one behaviour of `FrameMeta` on `molrs-python/tests/test_frame.py::TestFrameMeta`. The inner loop is `uv --directory molrs-python run --no-sync tox -e py` (or pytest on that file once the wheel is current). The closing gate is the project's check and `cargo mrs-test && cargo mrs-doctest` plus that tox run.

Happy path and goldens, one test each:

- `keys()` / `values()` / `items()` are instances of `collections.abc.KeysView` / `ValuesView` / `ItemsView`, and `type(...).__module__` is `collections.abc` with `__name__` `KeysView` / `ValuesView` / `ItemsView`.
- Fixture written `t`, `a`, `stress` as in Design: `list(m.keys()) == ["t", "a", "stress"]` and `list(m.values())` is the `f32` float, then `1`, then the six-element list. `["a", "stress", "t"]` is a failure.
- Liveness: `k = m.keys(); v = m.values(); i = m.items(); m["z"] = "late"` → `"z" in k`, and `len(k)`, `len(v)`, `len(i)` are all 4.
- Set algebra and membership: `m.keys() & {"t"} == {"t"}`, `m.keys() | {"z"} == {"t", "a", "stress", "z"}`, `("a", 1) in m.items()`, `("a", 2) not in m.items()`, `300.0 in m.values()`.
- `popitem()` returns the `stress` pair. After `m["t"] = 9` on a fresh fixture, `popitem()` still returns `stress`. Repeated `popitem` empties the map; one more raises `KeyError`.

Edge cases, one test each:

- Empty meta: all three views have length 0, and `bool(m.keys())` is `False`. A view held across `m.clear()` reports length 0.
- `list(m)`, `list(m.keys())`, `dict(m)`, and `list(m.items())` return without `RecursionError`, and `list(m) == ["t", "a", "stress"]`.
- Lookups: `(1 in m)` is `False`, `m.get(1)` is `None`, `m.get(1, "d") == "d"`, `m.pop(1, "d") == "d"`, `1 not in m.keys()`, `(1, 2) not in m.items()`, `m.get([])` is `None`. `m[1]` and `del m[1]` raise `KeyError` with args `(1,)`. `m[1] = 2` and `m.setdefault(1, 2)` raise `TypeError`.
- `m.update(m)` and `m |= m`, starting from `{"t": MetaValue("f32", 300.0), "a": 1}`, return normally (no `pyo3_runtime.PanicException`), leave `list(m) == ["t", "a"]`, leave the values readable, leave `dtype("a") == "i64"`, and set `dtype("t") == "f64"`. An assertion that `dtype("t")` is still `"f32"` is a failure: that would mean the write path called `tag_of` again.
- Deleting a not-yet-visited key inside `for v in m.values()` raises `KeyError`, and the assertion names `KeyError`, not `RuntimeError`.
- `hash(frame.meta)` raises `TypeError` (`pytest.raises(TypeError)`), for an empty meta and for a meta with one key. Equality is unchanged: two frames with the same meta items still compare equal.

Non-regression, already owned by earlier links, must stay green: link 02's insertion-order test and the corrected `test_mapping_protocol`; link 03's plain-write re-inference tests (`test_a_plain_write_takes_the_values_own_dtype`, `test_a_plain_write_replaces_the_slots_dtype`, and the narrowed `test_reassigning_a_read_value_is_an_identity`); `test_copying_a_frame_keeps_exact_dtypes` (`:234-242`), because `molrs.Frame(f)` still copies through `PyFrameMeta`.

No domain validation applies.

**No `regressions/` example.** molrs has no `regressions/` tree: `CLAUDE.md` § Build & Test Commands says the benchmark and regression systems are being redesigned outside this repo, `molrs/Cargo.toml:14-18` sets `autotests` / `autobenches` / `autoexamples` to false, and `.claude/specs/cgsmiles-03-release.md:179` is the standing precedent. The executable public-API artifact is `TestFrameMeta`, run by `uv --directory molrs-python run --no-sync tox -e py`. No `type: runtime` criterion points at a `regressions/` script.

## Out of scope

- `__reversed__` on `FrameMeta`. `reversed(m.keys())` stays `TypeError`. No in-tree caller, and adding it next to views that would only half-support it fails Shape check #3.
- `fromkeys`. A virtual `MutableMapping` registration does not provide it, and a live view of one frame has no classmethod that returns a plain `dict`.
- A new public view type, a subclass of `_DictView`, or any edit to `molrs-python/python/molrs/views.py`.
- Any edit under `molrs/src/`, including `molrs/src/core/store/meta.rs` and `molrs/src/io/zarr/sequence.rs` (link 07). No new `MetaMap` method and no sort.
- `molrs-capi`, `molrs-wasm`, and `molrs-cxxapi`. Link 02 already removed their sorts. This link does not put an order policy back.
- Restoring `typed_for` or calling `tag_of` from `__setitem__`, `absorb`, or `setdefault`. Link 03 deleted that rule.
- `MetaDocument`'s `__hash__`, and freezing read values. Link 05 sets `#[classattr] const __hash__ = None` on `MetaDocument` and does not re-decide `PyFrameMeta`. A later `hash(frame.meta)` assertion is a non-regression of the classattr this link adds.
- Matching `dict`'s `RuntimeError` on mutation during iteration. The materialized key list raises `KeyError` instead; that is tested and written on the `FrameMeta` docstring. A version counter on the iterator is a different design.
- `TypeError` for unhashable lookup keys. Recorded above as the lookup divergence.
- A notes-page edit or a migration page. The `KeyError` divergence ships as tests plus the docstring, in this unreleased 0.15 tree.
- `__init__.py`. The existing `MutableMapping.register(FrameMeta)` is reused unchanged.
