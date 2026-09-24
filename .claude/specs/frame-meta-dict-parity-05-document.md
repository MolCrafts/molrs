---
title: "MetaDocument — every meta door hands back a frozen value"
slug: frame-meta-dict-parity-05-document
status: in-progress
created: 2026-09-22
chain: frame-meta-dict-parity (01-ordered → 02-binder-order → 03-untyped-write → 04-dict-views → 05-document → 07-sequence)
revised: 2026-09-22
depends_on: frame-meta-dict-parity-04-dict-views
---

# MetaDocument — every meta door hands back a frozen value

## Summary

`frame.meta["run"]["step"] = 3` looks like a write into the frame and is not: the decoded `dict` is a snapshot, the edit vanishes, and nothing says so at the time. This link closes that door by making the rule uniform instead of documenting the exception — **every door of `frame.meta` hands back frozen values**: a fixed-length vector comes back as a `tuple` instead of a `list`, and a `json` document comes back as a new frozen mapping, `molrs.MetaDocument`, which raises `TypeError` on the swallowed write instead of accepting it. The top-level container `frame.meta.copy()` returns is the only mutable thing the surface hands out, and `MetaDocument.copy()` is the same unfreeze one level down. Reading is otherwise unchanged (scalars unwrap as before), every bulk door (`dict(frame.meta)`, `{**frame.meta}`, `frame.meta = dict(other.meta)`) still round-trips, and `frame.meta[k] = frame.meta[k]` stays an identity for every dtype. The break rides the **unreleased** `0.15.0`, so no published contract breaks and molpy's pin moves once. The itemized breaking list this link produces is published by `frame-meta-dict-parity-06-release-notes`.

## Domain basis

None. This is binder-surface semantics: no equation, no physical unit, no literature reference is involved. `$META.science.required` is true for molrs; this link declares no physics, so the section is stated and empty by rule, not skipped.

## Design

**Constitution.** `.claude/notes/law.md` does not exist in this repo, and none is expected (`.claude/specs/cgsmiles-03-release.md:22` records that). The governing rules are `CLAUDE.md` § *Design preferences* — the iron law *no silent debt*, *OOP by default*, *Primitive single-responsibility public APIs*, the *Shape check* — plus `CLAUDE.md` § *Testing Rules*, `.claude/notes/architecture-rules.md` § *Binder rules* and `.claude/notes/notes.md` § *Binding-surface symmetry*.

**Scope.** `molrs-python` only. The Rust core is untouched: `MetaValue` and `MetaMap` keep storing `serde_json::Value`; nothing about storage, Zarr, or the FFI handle changes. `molrs-wasm` and `molrs-capi` expose meta as strings and scalars only (`getMeta` / `getMetaScalar`), so the binding-symmetry rule is satisfied vacuously — there is no list or document on those surfaces to keep symmetric.

### 1. The rule, and where it bites

`meta_value_to_py` (`molrs-python/src/core/store/frame.rs:783-816`) is the one decoder. It has **nine call sites**: `:62`, `:166`, `:196`, `:248`, `:257`, `:266`, `:279`, `:291`, `:308`. Eight are `frame.meta` doors — `__getitem__`, `values`, `items`, `get`, `pop`, `popitem`, `setdefault` and `as_dict` — and `as_dict` additionally serves `copy`, `__eq__`, `__repr__`, `__or__` and `__ror__`. Freezing inside `meta_value_to_py` therefore freezes **every** door at once; there is no door to forget, and no per-door helper. The ninth call site, `MetaValue.value` (`:62`), is the one deliberate exception (§ 4).

The rule is therefore *not* "reads of `frame.meta[k]`". It is: every value that leaves `frame.meta` is frozen, the dict `copy()` / `as_dict` builds is the only mutable container, and `MetaDocument.copy()` is that same unfreeze one level down.

### 2. `tuple` is the frozen sequence

Builtin rather than a parallel type, hashable, accepted by `np.asarray`, and encoded natively by `json.dumps` as an array. Three arms change:

- `meta_value_to_py`'s `list!` macro (`:789-793`) builds a `PyTuple` instead of a `PyList`. Both fixed-length vector dtypes and JSON arrays are covered, because the JSON array arm goes through `json_to_py` (`:928-934`), which changes with it.
- `infer_meta_value` (`:894-906`) gains a tuple branch beside its `PyList` branch. **Without it the round-trip silently downgrades**: a 6-tuple read off an `f64x6` key and written to a *new* key falls past the `PyList` test into the `json` arm and is stored as a JSON array. The branch is a widening of the existing test, not a second inference path.
- `py_to_json` (`:971-977`) gains a tuple arm beside its `PyList` arm, so a frozen array read off a `json`-dtype key can be written back.

Reading a `tuple` back into a fixed-length slot needs no change: `meta_value_from_dtype`'s `array()` helper (`:824-832`) extracts `Vec<T>`, and pyo3 0.28's `Vec<T>` extraction falls through to `extract_sequence` (`pyo3-0.28.3/src/conversions/std/vec.rs:83`), which accepts anything passing `PySequence_Check` — a tuple does. `test_frame.py:201-211` is the in-tree guard and stays green unchanged.

### 3. `MetaDocument` — the frozen mapping

A `#[pyclass(module = "molrs", name = "MetaDocument", frozen)]` over a `serde_json::Map<String, JsonValue>`, built by `json_to_py`'s object arm and placed beside `PyMetaValue` and `PyFrameMeta`. It is not a core type: `molrs` Rust callers hold `MetaValue::Json(serde_json::Value)` and need no wrapper; this exists only to give a Python reader a mapping it cannot silently mutate.

`Mapping.register(MetaDocument)` goes next to the existing `MutableMapping.register(FrameMeta)` (`molrs-python/python/molrs/__init__.py:100`) — but **a virtual ABC registration supplies `isinstance` and nothing else**: no mixin is inherited, so every method a caller expects must be written out. The surface is therefore, exhaustively:

| Member | Behaviour |
|---|---|
| `__getitem__(key)` | frozen value; `KeyError` when absent |
| `__len__`, `__iter__` | key count; iteration over keys |
| `__contains__(key)` | membership (explicit — no mixin) |
| `keys()`, `values()`, `items()` | the same `collections.abc` views link 04 settled for `FrameMeta` |
| `get(key, default=None)` | explicit — no mixin |
| `__eq__`, `__ne__` | see below (explicit — no mixin, and pyo3 slots nothing by itself) |
| `__repr__` | `MetaDocument({'step': 1})` — names its type |
| `copy()` | **deep** plain decode → `dict`/`list`/scalars; the unfreeze and `json.dumps` door |
| `__hash__` | `#[classattr] const __hash__: Option<Py<PyAny>> = None;` → unhashable, like `dict` |

Deliberately absent: `__setitem__`, `__delitem__`, `update`, `pop`, `popitem`, `clear`, `setdefault`, `__or__`, `__ior__`. A snapshot has no mutators; the write-back idiom is `doc = frame.meta["run"].copy(); doc["step"] = 2; frame.meta["run"] = doc`.

**`__eq__` compares frozen-side, one rule at both levels.** It builds its one-level frozen `dict` snapshot and delegates to `dict.__eq__`. So `doc == {"tool": "molrec", "run": 3}` is `True` while `isinstance(doc, dict)` is `False`; a nested dict compares recursively through the same method; an array member compares equal to a `tuple` and not to a `list` — exactly the rule `FrameMeta.__eq__` (`:225-227`) already applies at the top level. When `other` is itself a `MetaDocument`, `dict.__eq__` returns `NotImplemented` and Python's reflected call lands back in `MetaDocument.__eq__` with a plain `dict` on the right, which terminates in one bounce. `__ne__` is written out rather than assumed: a `tp_richcompare` that answers only `Py_EQ` lets `!=` fall back to identity, which would make `doc != equal_dict` true while `doc == equal_dict` is also true.

**`__hash__` is not free.** pyo3 0.28 wires `Py_tp_hash` only when `__hash__` or `#[pyclass(hash)]` is present (`pyo3-macros-backend-0.28.3/src/pymethod.rs:87,983`); **nothing** sets the slot to `None` because `__eq__` exists. Omitting it yields *identity* hashing, which for a value-semantics snapshot is actively wrong: two documents that compare equal hash differently, so `doc in some_set` is silently `False`. Hence the explicit `classattr`, pinned by `pytest.raises(TypeError): hash(doc)`.

**Same bug, pre-existing, in the surface this link is already editing.** `PyFrameMeta` defines `__eq__` (`:225`) and no `__hash__`, so `hash(frame.meta)` succeeds by identity where `hash({})` raises. It is one line in a file this link already changes; per the iron law it is fixed here, not left.

**Iteration order is unspecified.** Neither `molrs/Cargo.toml:86` nor `molrs-python/Cargo.toml:26` asks for `serde_json/preserve_order` (both read `serde_json = "1"`); the feature is on only because a transitive crate enables it. This link does **not** declare it — adding a feature to the published core crate's `serde_json` swaps every `Map` in molrs from `BTreeMap` to `IndexMap`, which is a core behaviour and allocation change owed a measurement, not a Python-surface convenience. So the stub and the docstring say "order is unspecified", no test asserts inner order, and the two levels are documented as differing. The accidental dependence is routed to `/mol:note`.

### 4. Pickling: `MetaValue.value` decodes `JsonForm::Plain`

`_frame_ctor_args` (`molrs-python/python/molrs/frame.py:693`) pickles meta via `frame.meta.typed()` → `dict[str, MetaValue]`, and `MetaValue.__reduce__` (`frame.rs:53-58`) packs `(dtype, self.value)` where `value` is `meta_value_to_py` (`:61-63`). If that call froze, a `json`-dtype key would reduce to a `MetaDocument` — a `#[pyclass]` with no `#[new]` and no `__reduce__`, which pickle refuses. That breaks `molrs-python/tests/test_pickle.py:52-57` and `:300-302`.

**Decision: `MetaValue.value` (and therefore `__reduce__`) decodes with `JsonForm::Plain`.** A private `enum JsonForm { Frozen, Plain }` is threaded through `meta_value_to_py` / `json_to_py`; eight call sites pass `Frozen`, the `MetaValue.value` call site (`:62`) passes `Plain`, and `MetaDocument::copy` passes `Plain` too. One decoder, one parameter, no second JSON bridge.

*Why this does not re-open the swallowed write.* `.value` is not a door of `frame.meta`; it is the payload of a detached value object, and specifically **the argument that rebuilds it** — `MetaValue(v.dtype, v.value)` reconstructs `v`, which is precisely what `__reduce__` needs, so the payload must be picklable builtins all the way down. `PyMetaValue` is already `frozen` and holds a **clone** lifted out of the map (`typed()` clones at `:345-359`); no spelling of `frame.meta.typed()[k]` writes through. A user who mutates the dict returned by `.value` has taken two explicitly named snapshot hops (`.typed()`, `.value`) and is mutating their own object, exactly as with `frame.meta.copy()`. The swallowed write the uniform rule closes is the *unnamed* one — the subscript chain `frame.meta["run"]["step"] = 3` that reads as live and is not.

The alternative — a `#[new]` plus `__reduce__` on `MetaDocument` — was rejected: it adds a public construction door for a type whose whole point is that the frame produces it, to make picklable a value that pickles perfectly well as its plain form.

Fixed-length vectors need no exception: `tuple` is a picklable builtin and extracts back through `array()`. `frame.py:693` is **not** edited — the pickle path is correct as written once `.value` is Plain.

### 5. The `json.dumps` asymmetry

`json.dumps(tuple)` is native and correct. `MetaDocument` is **not** a `dict` subclass and `json.JSONEncoder` does not consult `collections.abc.Mapping`, so `json.dumps(frame.meta["run"])` raises `TypeError` where today it returns. The idiom is `json.dumps(frame.meta["run"].copy())` — which is why `copy()` is a *deep* plain decode rather than a one-level unfreeze: a shallow copy would leave nested documents inside and raise again. This asymmetry is stated at every site that states the tuple claim.

### 6. Warrant: the bulk doors, restated

A `.meta[` grep cannot see bulk access, so the warrant is restated over the bulk doors by hand. Each still round-trips once the tuple and document arms of `infer_meta_value` / `py_to_json` land:

- `molrs-python/python/molrs/frame.py:639` — `dict(self.meta)` in `Frame.to_dict()`. Read-only; values are frozen, which is the intended change, and the **type** of `to_dict()["meta"][k]` changes accordingly, so its docstring is corrected.
- `molrs-python/python/molrs/frame.py:662-663` — `Frame.copy()` does `new.meta = self.meta`, which takes the whole-map fast path in `mapping_to_meta_map` and never decodes. Unaffected.
- `molpy/src/molpy/io/data/pdb.py:66` — `out.meta = dict(frame.meta)`. Round-trips.
- `molpy/src/molpy/io/forcefield/amber.py:74` — `{**frame.meta, **dict(structure.meta)}`. Round-trips.
- `molrec/tests/molrs_adapter.py:110,260` — `dict(frame.meta)`. Round-trips.

Two cross-repo items are **named and routed**, not fixed here:

1. `molrec/src/molrec/core/bindings/zarr.py:783-785` json-dumps meta values read through `frame.meta[key]` for `series.dtype == "json"`. It breaks on a document value under this rule. molrec's to fix: `json.dumps(value.copy())`, or `frame.meta.typed()[key].value`, which is plain by § 4.
2. `molrec/tests/molrs_adapter.py:259-261` reads `value.dtype` off `dict(frame.meta).items()`. molrs has not returned tagged values from `dict(meta)` since that door started handing out plain values — **the adapter is already stale, independent of this link**. molrec's to fix.

molpy and molrec are other repos. This link does not edit them and does not wait on a pin or a tag. The two molrec breaks below are named so they are fixed against this same 0.15 tree, outside this spec.

### 7. The behaviour this link changes

1. `frame.meta[k]` for every fixed-length vector dtype returns `tuple`, not `list`, at every door. In-tree change: `test_frame.py:134`.
2. `frame.meta == {"stress": [1.0, …]}` — a dict-literal comparison with a non-scalar value — flips from `True` to `False` (`tuple != list`).
3. `frame.meta[k]` for a `json`-dtype **object** returns `MetaDocument`: `isinstance(v, dict)` is `False`, `v == {...}` is `True`, `v["a"] = 1` raises `TypeError` instead of silently vanishing, `hash(v)` raises (as for `dict`), and `json.dumps(v)` raises — use `json.dumps(v.copy())`.
4. Arrays nested inside a json document are `tuple`s too.
5. `MetaValue("f64x6", …).value` returns a `tuple`; `MetaValue("json", …).value` stays a plain `dict` (§ 4, unchanged).
6. `hash(frame.meta)` now raises `TypeError` (was an identity hash) — dict parity, and a pre-existing defect fixed here.
7. Iteration order inside a `MetaDocument` is unspecified and always was. `FrameMeta` enumeration is insertion order once link 02 has landed; this link does not sort.

Not broken, and stated as such: `dict(frame.meta)`, `{**frame.meta}`, `frame.meta.copy()`, `frame.meta = dict(other.meta)`, `frame.meta[k] = frame.meta[k]`, pickling a `Frame`, `np.asarray` on a vector value.

### 8. Found, named, not fixed here

- `molrs`'s `serde_json` document ordering depends on a transitive crate enabling `preserve_order`. Documented as unspecified here; routed to `/mol:note`.
- `PyFrameMeta` declares `module = "molrs._lib"` while `PyMetaValue` declares `module = "molrs"`, though both are exported at `molrs.*`. `MetaDocument` follows `PyMetaValue`. Observed, unowned, not fixed.
- `_lib.pyi` parity is guarded at class-name level only, so the corrected return types are unguarded. Pre-existing; already on the deferred list.

### Reuse decision

- `reuse` — `json_to_py` / `py_to_json` (`frame.rs:910-981`): the one JSON↔Python bridge. It gains a `JsonForm` parameter and two arms; no second decoder and no parallel encoder is created.
- `reuse` — `meta_value_to_py` (`:783-816`): stays the single decode door all eight `frame.meta` readers call. The freeze happens inside it, so no door acquires a per-door helper.
- `pattern` — `PyMetaValue` (`:31-72`) dictates `MetaDocument`'s shape: `#[pyclass(module = "molrs", frozen)]`, getters only, values handed out by cloning, `__repr__` naming its own type.
- `pattern` — `MutableMapping.register(FrameMeta)` (`__init__.py:100`) is the registration site and spelling; `Mapping.register(MetaDocument)` goes beside it, with the explicit surface written out because registration supplies no methods.
- `pattern` — the read half of `_DictView` (`molrs-python/python/molrs/views.py:38-88`) **plus** `copy` / `__eq__` / `__repr__` from `PyFrameMeta` (`frame.rs:225-231, 337-339`); `_DictView` itself has none of those three.
- `new` — `MetaDocument`. No frozen mapping exists in the binder, and `MetaValue` is not one: it carries a tag plus a payload of any dtype, so giving it `keys()`/`__getitem__` would hand a `string`-dtype value a mapping surface.
- `generalize` — none.

## Files to create or modify

- `molrs-python/src/core/store/frame.rs` — `PyMetaDocument`; private `enum JsonForm`; `meta_value_to_py`; `json_to_py`; `infer_meta_value`; `py_to_json`; `PyMetaValue::value` and `__reduce__` on `JsonForm::Plain`; `PyFrameMeta` `#[classattr] const __hash__ = None`; the `PyFrameMeta` class rustdoc (`:100-115`) and the `Frame.meta` getter docstring (`:590-596`)
- `molrs-python/src/lib.rs` — `m.add_class::<PyMetaDocument>()` beside `PyMetaValue`
- `molrs-python/python/molrs/__init__.py` — import, `__all__`, `Mapping.register(MetaDocument)` beside `:100`
- `molrs-python/python/molrs/_lib.pyi` — new `MetaDocument` class; `MetaValue.value` (`:331`); the `FrameMeta` docstring; and the return types of the ten `FrameMeta` members that now hand back frozen values, including `copy() -> dict[str, Any]`
- `molrs-python/python/molrs/frame.py` — `Frame.to_dict` docstring (`:635-636`); no code change
- `molrs-python/tests/test_frame.py` — `TestFrameMeta` and a new `TestMetaDocument`
- `molrs-python/tests/test_pickle.py` — new assertions; `:52-57` and `:297-302` stay unchanged as the guard
- `molrs-python/docs/reference/python.md` — `::: molrs.MetaDocument` after `::: molrs.MetaValue`
- `.claude/notes/notes.md` — the routed items of § 8 and § 6

## Tasks

- [x] Write failing unit tests for the frozen `frame.meta` doors in `molrs-python/tests/test_frame.py`: every vector dtype comes back as `tuple` through all eight doors; `frame.meta["run"]` is a `MetaDocument` whose `__eq__`/`__ne__`/`__contains__`/`get`/`keys`/`values`/`items`/`len`/`iter`/`repr` behave, with `doc == {...}` true and `isinstance(doc, dict)` false asserted **in the same function**; `hash(doc)` and `hash(frame.meta)` both raise `TypeError`; the in-place nested write raises instead of vanishing; `doc.copy()` is a deep plain dict that `json.dumps` accepts; and `frame.meta[k] = frame.meta[k]` stays an identity for a scalar, an `f64x6` and a `json` key
- [x] Write failing unit tests for the value-object and pickle paths in `molrs-python/tests/test_pickle.py`: a `Frame` carrying a `json` meta key round-trips (`:52-57` unchanged), `MetaValue("json", {...}).value` is a plain `dict` and `MetaValue("f64x6", …).value` is a `tuple`, and a `MetaDocument` read off one frame assigns into another frame's new key without downgrading its dtype
- [x] Implement the frozen-sequence arms in `molrs-python/src/core/store/frame.rs`: `meta_value_to_py`'s `list!` macro builds a `PyTuple`, `json_to_py`'s array arm builds a `PyTuple`, `infer_meta_value` gains a tuple branch, and `py_to_json` gains a tuple arm
- [x] Implement `MetaDocument` and the private `JsonForm` in `molrs-python/src/core/store/frame.rs` — frozen pyclass with the surface enumerated in Design § 3, `#[classattr] const __hash__ = None` on `MetaDocument`, explicit `__eq__`/`__ne__`/`__contains__`/`get`, deep-plain `copy()`, the `py_to_json` `MetaDocument` arm, `MetaValue::value`/`__reduce__` on `JsonForm::Plain`. Leave `PyFrameMeta`'s `__hash__ = None` as link 04 set it; `hash(frame.meta)` in this link's tests is a non-regression
- [x] Register `MetaDocument` in `molrs-python/src/lib.rs` and export it from `molrs-python/python/molrs/__init__.py` (import, `__all__`, `Mapping.register(MetaDocument)` beside `:100` with a comment naming `molvis/python/src/molvis/wire.py:389,530` and `molrec/tests/molrs_adapter.py:110-113`)
- [x] Add `MetaDocument` to `molrs-python/python/molrs/_lib.pyi` and correct `MetaValue.value` (`:331`), the `FrameMeta` docstring, and the ten `FrameMeta` return types that now hand back frozen values
- [x] Document the uniform rule at its doc sites — the `PyFrameMeta` rustdoc, the `Frame.meta` getter docstring, `Frame.to_dict`'s docstring, and `::: molrs.MetaDocument` in `docs/reference/python.md` — each stating the rule, the `json.dumps` asymmetry (`json.dumps(frame.meta["run"].copy())`), and that nested document order is unspecified
- [x] Verify the bulk-door warrant by hand over the five sites in Design § 6 and record the two molrec items as same-version follow-ups, not as post-publish work
- [x] Record the routed items of § 8 and § 6 in `.claude/notes/notes.md`
- [ ] Run full check + test suite

## Testing strategy

Unit tests only, in the two existing binder test files that mirror the surface under change. Per `CLAUDE.md` § *Testing Rules*, a binding test smokes the FFI seam — construct, call, round-trip types, dtype at the boundary, error mapping — and re-derives no number the Rust suite already proves. There is no Rust-side change, hence no new `#[cfg(test)]` module.

Happy path, edge cases and the pickle guard are enumerated in the acceptance contract. Two points of method:

- **Equality and identity type are asserted in one function**, so `doc == dict` being true and `isinstance(doc, dict)` being false cannot drift apart.
- **No inner-order assertion.** Document iteration order is not pinned by any test, because the crate does not pin it (§ 3). Every ordering comparison goes through `sorted(...)` or `==` against a dict.

**No `regressions/` example.** molrs has no `regressions/` tree: `CLAUDE.md` § *Build & Test Commands* records that the benchmark and regression systems are being redesigned outside this repo, and `.claude/specs/cgsmiles-03-release.md:177` is the standing precedent for refusing one on that basis. The alternative was considered — a standalone public-API script — and rejected because it would be the only file in a tree this repo does not have and does not run, duplicating assertions the Python suite already owns at the same seam. The `type: runtime` criteria are written against `tox -e py` and the two named test files.

## Out of scope

- **The Rust core.** `molrs/src/core/store/meta.rs`, `MetaValue`, `MetaMap`, Zarr persistence and the FFI handle are untouched; storage is unchanged and no on-disk format moves.
- **`molrs-wasm` and `molrs-capi`.** Their meta doors are string- and scalar-valued; there is no list or document to freeze.
- **Declaring `serde_json/preserve_order`.** Named in § 8. Document iteration order stays unspecified here. Pinning it is a separate 0.15 change, not a compatibility shim and not this link.
- **A `MetaDocument` constructor or mutators.** Rejected in § 3 and § 4: a snapshot the frame produces needs no public `#[new]`, and a mutable frozen mapping is a contradiction.
- **`PyFrameMeta` unhashable.** Link 04 sets `#[classattr] const __hash__ = None` on `PyFrameMeta` and tests `hash(frame.meta)`. This link does not re-decide that. It still sets the same classattr on `MetaDocument`, and its `hash(frame.meta)` assertion is a non-regression.
- **`FrameMeta.__module__`.** Observed in § 8; changing it is a visible `repr` change with no consumer need.
- **Migration guides, release notes, version pins, and `cgsmiles-03-release`.** This link does not edit them. § 7 is the behaviour contract, not a changelog task.
- **molrec and molpy edits.** Named in § 6. They are other repos. This link does not edit them and does not gate them on a tag. They belong to this same 0.15, outside this spec.
- **Unifying the three `json_to_py` copies** (`frame.rs`, `molrs-python/src/core/store/record.rs:296-322`, `molrs-python/src/io/mrec.rs:702-728`). The `Record`/`mrec` payloads are owned values a caller re-submits wholesale, not live views, so nothing swallows a write to them and they stay plain. The code triplication is rot: routed to `/mol:refactor` and recorded in `.claude/notes/notes.md`, not only here.
