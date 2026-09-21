---
title: CGsmiles 02d — Fragment at the Python seam
slug: cgsmiles-02d-python-fragment
status: approved
created: 2026-09-21
chain: cgsmiles (01a → 01b → 01c → 01d → 01e → 02a → 02b → 02c → 02d → 03)
depends_on: cgsmiles-02a-fragment-core, cgsmiles-02b-to-fragment, cgsmiles-02c-conformer-fragment, cgsmiles-01e-python-ir
---

# CGsmiles 02d — Fragment at the Python seam

## Summary

02a gave `molrs` a third graph leaf — `Fragment`, a molecular graph with named,
unsatisfied valences — 02b builds them from a CGsmiles string, and 02c embeds
them in 3D. None of that is reachable from Python: `molrs.Fragment` does not
exist, `molrs.translate` on a fragment would silently move nothing, and
`Conformer.generate` accepts only an `Atomistic`. This link binds the leaf: a
`PyFragment` pyclass carrying the generic ECS world surface plus typed
`add_atom` / `add_bond` / `add_port` wrappers, the `views.py` `Fragment` /
`Port` pair that makes it the same kind of live-view object `Atomistic` and
`CoarseGrain` already are, a `PyFragment` arm in the generic geometry
dispatcher, a widened `Conformer.generate` that returns the leaf type it was
handed, and `CGSmilesIR.to_fragment()` returning `dict[str, Fragment]`. No Rust
core change; no wasm, capi or cxxapi change. Writing the third leaf also forced
the rot it would otherwise triplicate: `PyAtomistic::from_core`
(`molgraph.rs:963-986`) and `PyCoarseGrain::from_core` (`:1574-1592`) are
literal copies of one another, so this link extracts the single
`from_core_shadowed` helper the comment at `molgraph.rs:978-982` exists to
explain and rewires all three leaves onto it — the second real use that
CLAUDE.md § *Inline until the second real use* names as the trigger.

## Domain basis

No new physics. The vocabulary crossing this seam is 02a's, unchanged:
`PortKind` is the closed four-glyph set `$ < > !` (BigSMILES bonding
descriptors — Lin, T.-S. et al., *ACS Cent. Sci.* **5**, 1523–1531 (2019),
DOI 10.1021/acscentsci.9b00476 — plus CGsmiles' shared `!`), `port_label` is
free-form text where `""` means unnamed, and `port_order` is a `BondNumber`
multiplicity. Units: coordinates produced by `Conformer.generate` are Å (02c);
`frag_id` is a dimensionless per-atom instance ordinal. This spec asserts no
chemistry — per `.claude/notes/testing.md` § *Language bindings*, a binder test
proves only the seam, and every numeric claim about ports, embedding and
`frag_id` propagation is owned by the Rust unit tests in 02a / 02b / 02c.

## Design

### Contracts assumed from the unmerged predecessor links

These four links are staged and unmerged. Each contract is named so a drift is
reconciled rather than discovered at compile time.

- **02a** — `Fragment { graph, bond, port }` with `new` / `Default` / `Deref` /
  `DerefMut` to `MolGraph`, `try_from_molgraph`, `into_inner`, `as_molgraph`,
  `add_atom_xyz`, `add_atom_bare`, `add_bond`, `n_atoms`, `n_bonds`,
  `add_port(anchor, handle, kind: PortKind, label: &str, order: BondNumber) -> Result<PortId>`,
  `ports()`, `n_ports()`, `set_frag_id` / `frag_id`, `inherit_frag_ids()`,
  `to_frame` / `from_frame`, and `PortKind::{Symmetric, Left, Right, Shared}`
  with `as_str()` → `$ < > !` and `FromStr`. Four properties this link rests on
  explicitly:
  - `Fragment::new` registers `bonds` and `ports` at construction, as
    `Atomistic::new` (`atomistic.rs:103-116`) and `CoarseGrain::new`
    (`coarsegrain.rs:86-94`) register theirs. Without it a freshly constructed
    `molrs.Fragment` has no `ports` kind and every generic relation call raises.
  - `Fragment::add_bond` stamps `(BondType::Single, BondNumber::Single)` through
    `bond::write_bond_class` (02a Design; `molrs/src/core/system/bond.rs`), which `Atomistic::set_bond_class` (`atomistic.rs:206-214`) also delegates to (
    props `keys::BOND_TYPE` = `"bond_type"` and `keys::BOND_NUMBER` =
    `"bond_number"`), so a bond built through the leaf carries both bond facts.
  - `add_port` and `PortKind::from_str` return `MolRsError::validation`, which
    `molrs_error_to_pyerr` (`helpers.rs:81`) maps to Python `ValueError`. That
    mapping is what every domain-rejection test below asserts.
  - `Fragment` derives `Clone`, by peer parity with `atomistic.rs:72` and
    `coarsegrain.rs:57`. `PyFragment::copy` is
    `from_core(py, self.inner.clone())`, exactly as `PyCoarseGrain::copy`
    (`molgraph.rs:1511`) is; 02a places the *method* `copy` out of scope, which
    is a different thing from the derive.
- **02b** — `CGSmilesIR::to_fragment(&self) -> Result<BTreeMap<String, Fragment>, SmilesError>`.
- **02c** — `Conformer::generate<M: ElementGraph>(&self, mol: &M) -> Result<(M, ConformerReport), MolRsError>`,
  with `Fragment: ElementGraph`. Leaf in, same leaf out.
- **01e** — `#[pyclass(module = "molrs.io", name = "CGSmilesIR")] pub struct PyCGSmilesIR { inner: CGSmilesIR, input: String }`
  in `molrs-python/src/io/cgsmiles.rs`, and `molrs-python/tests/test_stub_parity.py`,
  the named consumer of this link's stub entries.

### `PyFragment` — one data slot, the generic world, three typed writers

`PyFragment` is declared next to `PyCoarseGrain` in
`molrs-python/src/core/system/molgraph.rs`, following `:1399-1426` exactly:

```rust
#[pyclass(module = "molrs._lib", name = "Fragment", extends = PyGraph, subclass)]
pub struct PyFragment { inner: Fragment }
```

`module = "molrs._lib"` marks a class shadowed by a `views.py` class of the same
name — the distinction `PyGraph` (`module = "molrs"`, `:585`) does not carry.
`#[new]` takes `(*_args, **_kwargs)` and returns `(Self, PyGraph)` so a Python
subclass needs no `__new__` shim. Inherent `mol()` → `self.inner.as_molgraph()`
and `mol_mut()` → `&mut *self.inner` (02a's `DerefMut`) feed
`graph_world_impl!(PyFragment);`. That macro is why the class must live in this
file: it is a plain `macro_rules!` (`:101`) with no `#[macro_export]`, so it is
in scope only here. A separate `core/system/fragment.rs` is rejected on that
ground alone — `with_world_mut` and `PyExtractedSubgraph` also live here, and
file length is a `/mol:refactor` question, not a placement one.

The macro supplies the whole relation surface — `register_kind`,
`add_relation`, `relation_ids`, `relation_nodes`, `set_relation_prop`,
`get_relation_prop`, `relation_keys`, `remove_relation`, `n_relations`
(`:194-334`) — so `ports` is reachable generically the moment `Fragment::new`
registers the kind. No hand-written relation API is added, and in particular
**no `remove_port`**: `frag.remove_relation("ports", id)` is the door,
`GraphViews._remove_relation` (`views.py:636-640`) already uses it for every
kind, and 02a ships no core `Fragment::remove_port` for a wrapper to call.

The typed `#[pymethods]`, each with a named caller:

| Method | Caller |
|---|---|
| `add_atom(symbol, x=None, y=None, z=None) -> int` | molpy's hand-built fragments; peer parity with `PyAtomistic::add_atom` (`:656`) / `PyCoarseGrain::add_bead` (`:1428`), neither of which has a `views` caller — `views.Fragment.def_atom` stays on the generic node path; the production caller 02a cites for `add_atom_xyz` / `add_atom_bare` |
| `add_bond(a, b) -> int` | `views.Fragment.def_bond`; 02a's `add_bond` |
| `add_port(anchor, handle, kind, label="", order=1) -> int` | `views.Fragment.def_port`; molpy's backmap builder |
| `n_ports` `#[getter]`, `n_atoms` `#[getter]` | this link's tests; molpy |
| `set_frag_id(atom, id)` / `frag_id(atom) -> int \| None` / `inherit_frag_ids() -> int` | 02c's relabel workflow, driven from Python |
| `copy()`, `to_frame()`, `from_frame()` `#[staticmethod]` | `views.Fragment.to_frame` (`views.py:514-525`); `to_fragment` consumers; this link's round-trip test |
| `core()` (inherent, not a pymethod) | `PyConformer::generate` |

`add_atom` and `add_bond` mirror `PyAtomistic::add_atom` (`:656`) /
`add_bond` (`:665`) and `PyCoarseGrain::add_bond` (`:1439`) line for line, and
they are the production callers that justify 02a shipping `add_atom_xyz`,
`add_atom_bare` and `add_bond` at all. `n_atoms` delegates to
`Fragment::n_atoms` exactly as `PyAtomistic::n_atoms` (`:749-751`) delegates to
`Atomistic::n_atoms`.

Deliberately absent: **`core_mut()`** (nothing in this link or a named successor
mutates a `Fragment` through the core handle — CLAUDE.md shape check 3), and a
**validating `port(id)` reader**. `views.Port` is the read path; a Rust-side
reader that re-parses the three props has no non-test consumer, and the link
that first needs one adds it with the caller that pins it.

**Why `PortKind` crosses as the glyph `str`, not an integer code.** The repo
precedent is numeric — `set_bond_type(handle, bond_type: u32)` (`:822`) with
`BondType::from_code`. The departure is deliberate and narrow. 02a made
`PortKind`'s only string form the notation glyph and its only stored form the
`port_kind` `Str` column in the Frame; a `u32` code table would be a **second
vocabulary, invented and owned by the binder**, that nothing else in the stack
speaks — the notation writes `[$]`, the Frame column holds `"$"`, molpy's
`def_port(..., "$")` passes `"$"`, and `port["port_kind"]` reads back `"$"`.
A code here would give one fact two spellings depending on which door you came
through, which the `Naming` rule forbids. `BondType` has no Frame-level string
form, which is why it is a code and this is not. Conversion is one
`PortKind::from_str` at ingress, raising `ValueError` for anything but the four
glyphs.

The class docstring carries one disambiguating line: `molrs.Fragment` is a
molecular graph with ports and is unrelated to the CL&Pol sense of "fragment"
in `FragmentScaling` / `FragmentAtoms` (`molrs/src/ff/scale_lj.rs:14,24`, bound
at `molrs-python/src/ff/mod.rs:242-247`), which is a polarizability-scaling
record. No rename; the two senses are disjoint and both are established.

### `from_core` — the found rot, extracted once

`PyAtomistic::from_core` (`:963-986`) and `PyCoarseGrain::from_core`
(`:1574-1592`) are literal copies: import `molrs`, `getattr` the public name,
compare against the native type, and either `Py::new` a fresh
`(leaf, PyGraph { inner: MolGraph::new() })` pair or construct the Python shadow
class and overwrite `inner`. A third copy triplicates it, so this link extracts
one plain generic function — no trait, no registry, no extension point that
nothing is pushing on:

```rust
use pyo3::pyclass::boolean_struct::False;

pub(crate) fn from_core_shadowed<T>(py: Python<'_>, leaf: T) -> PyResult<Py<T>>
where
    T: PyClass<BaseType = PyGraph, Frozen = False>,
{
    let public = py.import("molrs")?.getattr(T::NAME)?;
    if public.is(&py.get_type::<T>()) {
        return Py::new(py, (leaf, PyGraph { inner: MolGraph::new() }));
    }
    let object: Py<T> = public.call0()?.extract()?;
    *object.borrow_mut(py) = leaf;
    Ok(object)
}
```

`T::NAME` is the `PyTypeInfo` associated constant, i.e. the `name = "…"` already
written in each `#[pyclass]` attribute, so the public name is not declared a
second time. `Frozen = False` is required — `borrow_mut` does not exist without
it. The whole-struct assignment `*object.borrow_mut(py) = leaf` is equivalent to the old `.inner = inner` only because each of the three leaves has exactly one field; a leaf that grows a second field must assign `.inner` explicitly. Each leaf's `from_core` becomes one line,
`from_core_shadowed(py, PyAtomistic { inner })`, keeping the `pub(crate)` door
its ~14 existing call sites already name (`io/mod.rs:2172`, `ff/`, `perceive/`,
`PyExtractedSubgraph::from_atomistic`, …) so no call site changes and no second
public spelling appears.

Two properties the extraction preserves on purpose:

- **The shadow lookup stays per call.** `py.import("molrs")` is a `sys.modules`
  hit, not a re-import, and `getattr` is a type-dict lookup; caching either in a
  `GILOnceCell` would make a late re-binding of `molrs.Fragment` invisible for
  the life of the process, trading a correctness property for a dict lookup.
  What the extraction removes is the *duplication*, which is what the comment at
  `:978-982` is there to excuse.
- **The empty base `MolGraph` cannot be removed; it is structural.** PyO3
  constructs a subclass as base-then-subclass, and `PyGraph`'s only field is
  `inner: MolGraph` (`:586-588`), so an instance of a class declaring
  `extends = PyGraph` necessarily carries one. The alternatives are making
  `PyGraph.inner` optional — reintroducing "which graph is the real one?", the
  question leaf-first dispatch exists to answer — or dropping `extends = PyGraph`,
  which breaks the `isinstance(x, molrs.Graph)` molpy relies on.
  `MolGraph::new()` is empty collections (`molgraph.rs:448-451`), so the slot
  stays and the module doc gains one sentence naming the single helper.

### `with_world_mut` gains a `PyFragment` arm

`with_world_mut` (`:1608-1621`) dispatches leaf-first and falls through to
`mol.cast::<PyGraph>()`. A `PyFragment` **is** a `PyGraph`, so without an arm
`molrs.translate(frag, delta)` succeeds and translates the empty base graph:
wrong answer, no error, no exception. The arm goes after the `PyCoarseGrain`
arm and before the `PyGraph` fallthrough, the `PyTypeError` message gains
`Fragment`, and one arm serves all four geometry systems (`translate`,
`rotate`, `scale`, `align_direction`). The test that catches its absence is
written first and fails on a *value*, not an exception.

**`PyExtractedSubgraph` (`:455`) gets no arm.** 02a ships neither
`extract_subgraph` nor `induced_subgraph` on `Fragment`, so there is no producer
for a `from_fragment` constructor; its `graph` field is already `Py<PyAny>`, so
nothing type-specific is missing. The chemistry entry points get no arm either
and need none: they are typed `&PyAtomistic` at the signature
(`perceive.rs:95,126,142,161,179,200,226`; `ff/mod.rs:721,878`;
`ff/charge.rs:187,224,284,354`; `io/mod.rs:2254,2360,2410`), so PyO3's own
extraction rejects a `Fragment` with a `TypeError` before any body runs.

### `views.py` — `Port`, `Fragment`, and one writer per fact

`class Port(RelationRef[Atom])` goes next to `CGBond` (`:808-819`) with
`__slots__ = ()`, `_kind = "ports"`, `_arity = 2`. The three `port_*` props ride
the inherited `_DictView` with no extra code. The endpoints are named `anchor`
(`endpoints[0]`) and **`handle_atom`** (`endpoints[1]`), not `handle`:
`RelationRef.__slots__` contains `"handle"` and `RelationRef.__init__` assigns
`self.handle = handle`, the *relation's* own handle (`views.py:259,300`), so a
`handle` property would shadow the slot descriptor and make every `Port`
construction raise `AttributeError`. The collision is stated in the class
docstring so the name is not "corrected" back.

`class Fragment(GraphViews, _RsFragment)` goes next to `CoarseGrain`
(`:908-965`) with `_node_cls = Atom`,
`_relation_classes = {"bonds": Bond, "ports": Port}` — `Bond` reused unchanged,
because a fragment's nodes are atoms and its bonds are atom bonds — `atoms` /
`bonds` / `ports` view properties, and `__init__` delegating to
`GraphViews.__init__`.

The factories route by what each one has to stamp:

- `def_atom(mapping=None, /, **attrs)` keeps the generic node path
  (`self._create_node(..., cls=Atom)`), as `Atomistic.def_atom` does: a node
  carries no class to stamp, so there is nothing for a native call to add.
- `def_bond(a, b, /, **attrs)` calls native `self.add_bond(a.handle, b.handle)`
  and interns the result (`self._intern_relation("bonds", rid, cls=Bond)`), so a
  hand-built fragment's bonds carry both bond facts — `bond_type = 1` and
  `bond_number = 1` — exactly as a bond built in Rust does. It then applies `**attrs` through `ref.update(attrs)` and keeps `_create_relation`'s unwind (`views.py:586-589`: drop the interned ref, `remove_relation`, re-raise) on failure, so `frag.def_bond(a, b, order=2.0)` behaves as `Atomistic.def_bond` does. `def_port` takes no `**attrs`: its three props are the validated arguments themselves.
- `def_port(anchor, handle_atom, kind, label="", order=1)` calls native
  `self.add_port(...)` and interns likewise. `_create_relation`
  (`views.py:571-590`) would call the generic `add_relation` and write the three
  props raw, **bypassing 02a's validation entirely** — the anchor/handle roles,
  the `element == "H"` check, the anchor–handle bond check, the glyph parse and
  the `BondNumber::Unknown` rejection all live in core `add_port`. One writer,
  validated.

Both native-routing factories keep `_create_relation`'s endpoint-world guard
explicitly (`anchor.world is not self` → `ValueError`), because slotmap handles
from a foreign graph can alias rather than fail. `def_bead`
(`views.py:926-934`) is the existing precedent for a factory that routes through
a native typed call.

Validation is a **construction-time** contract. The pickle/`__reduce__` restore
path replays relations through `_load_graph` (`views.py:989-999`), which
re-writes the `port_*` props raw by design: it is restoring an object that was
already validated when it was built, and routing it through `add_port` would
re-run element and bond checks against a half-rebuilt graph. It stays as it is.

`_RsFragment` is imported beside `_RsCoarseGrain` (`:19-21`); the
`__init__` / `__reduce__` monkey-patch pair (`:1070-1075`) gains its fourth
entry; `__all__` (`:1078`) gains `Fragment` and `Port`. **`_dump_graph`
(`:1023-1050`) needs no `Fragment` branch** and must not grow one: ports are
relations and `frag_id` is a node property, so both ride the generic dump. The
`isinstance(graph, CoarseGrain)` branch at `:1045` exists only because bead
membership is a side map into a *foreign* world, which is precisely what 02a
decided `frag_id` would not be.

`__init__.py`: `Fragment` joins the `_lib` import block (`:71-76`, "Molecular
graph hierarchy"), then `views.Fragment` and `views.Port` join the views import
block (`:129-146`) — `views.Fragment` after `_lib.Fragment`, the deliberate
shadow, in the order `Atomistic` and `CoarseGrain` already use — and both names
join `__all__` (`:192`). `lib.rs:312` gains `m.add_class::<PyFragment>()?` after
`PyCoarseGrain` (base before subclasses).

### `PyConformer::generate` — widened, leaf in / leaf out

`generate` currently takes `mol: &PyAtomistic` and returns
`(Py<PyAtomistic>, PyConformerReport)` (`conformer/mod.rs:243-275`). It widens
to `mol: &Bound<'_, PyAny>` → `(Py<PyAny>, PyConformerReport)`, dispatching with
the crate's `.cast::<PyX>()` fallback chain (`molgraph.rs:1608`; `md.rs:474-477`;
`store/record.rs:271-274`; `helpers.rs:120`; `ff/mod.rs:396-397`) and returning
the leaf type it was handed, per 02c's `generate<M: ElementGraph> -> (M, _)`.
A `#[derive(FromPyObject)]` enum would read more nicely and is **not** used:
there is no such derive anywhere in molrs-python, so it would be a new idiom
introduced in passing. Anything else raises `PyTypeError` naming both accepted
types. The per-stage report mapping is now reached from two arms, so it is
extracted to a private `fn report_to_py(report: ConformerReport) -> PyConformerReport`
in the same file — second real use, same rule as `from_core`.

This is **molpy-visible but source-compatible**: `PyConformer` is `subclass`
(`:164-169`) and molpy subclasses it, but widening an accepted parameter breaks
no caller that passes an `Atomistic`, the return for that input is unchanged,
and an override calling `super().generate(mol)` is unaffected.

The docstring is rewritten with the signature: `Parameters` (`:224`) names
`Atomistic | Fragment`, `Returns` (`:229`) states the leaf-in/leaf-out contract
(`tuple[Atomistic, ConformerReport]` or `tuple[Fragment, ConformerReport]`,
coordinates in Å), and `Raises` gains `TypeError` for anything else. The
`Examples` block shows the fragment path as 02c specifies it — `generate`, then
the caller's own `frag.inherit_frag_ids()` relabel — and replaces the dead name
at `:239`: `parse_smiles` is not a Python symbol in any spelling, the only one
is `molrs.io.SmilesIR`.

### `CGSmilesIR.to_fragment()`

Bound on 01e's `PyCGSmilesIR` in `molrs-python/src/io/cgsmiles.rs`, returning a
`PyDict` of fragment name → `molrs.Fragment`, built with `PyDict::new` +
`set_item` per the `fragment_scaling_data` pattern (`ff/mod.rs:312-319`), errors
through `smiles_error_to_pyerr` per `PySmilesIR::to_atomistic`
(`io/mod.rs:2170-2173`). The values go through **`PyFragment::from_core`**, not
a bare `Py::new(py, PyFragment { .. })`: a bare pyclass instance is the native
leaf, so `type(f) is molrs.Fragment` is `False`, it has no `GraphViews`, and
`f.ports` / `f.def_atom` do not exist — the graph-out path would hand back a
degraded object, the failure mode `test_views.py:15-38` exists to catch.

### WASM, and the rest of the binders

`molrs-wasm` is untouched. `molrs-wasm/src/conformer.rs:84-85` calls
`Conformer::new(opts).generate(&atomistic)`, so `M` infers to `Atomistic` and
the call compiles unchanged under 02c's generic signature. The asymmetry with
this link's Python widening is structural rather than an omission: wasm exposes
conformer generation as the free function `generate3D(frame, speed, seed)` over
a `Frame` (`conformer.rs:72-89`), converting `Frame → Atomistic` internally, so
there is no wasm surface that could accept a `Fragment`.
`cargo clippy --manifest-path molrs-wasm/Cargo.toml --target wasm32-unknown-unknown --all-targets -- -D warnings`
(`docs/releasing.md:22`) is in the gate to prove the binder source stayed
compilable and untouched. `molrs-capi` and `molrs-cxxapi` have no graph-leaf
vocabulary at all.

### Reuse decision

- `graph_world_impl!` (`molgraph.rs:101`) — **reuse**. `PyFragment` gets the
  generic ECS surface from `graph_world_impl!(PyFragment);`; `ports` is reachable
  through `add_relation` / `relation_ids` / `set_relation_prop` with no new
  relation API.
- `PyCoarseGrain` declaration + `#[new]` (`:1399-1426`) — **reuse** as the
  template, including `module = "molrs._lib"` and the `(Self, PyGraph)` return.
- `PyAtomistic::add_atom` (`:656`) / `add_bond` (`:665`), `PyCoarseGrain::add_bond`
  (`:1439`) — **reuse** as the shape of `PyFragment`'s typed writers.
- `PyAtomistic::from_core` (`:963-986`) + `PyCoarseGrain::from_core`
  (`:1574-1592`) — **generalize** into the plain generic
  `from_core_shadowed<T>`; all three leaves rewired, no trait introduced, no
  third clone.
- `with_world_mut` (`:1608-1621`) — **pattern**. Gains a `PyFragment` arm in the
  existing `.cast` chain shape, message updated.
- `.cast` / `extract::<PyRef<T>>` chains (`md.rs:474-477`,
  `store/record.rs:271-274`, `helpers.rs:120`, `ff/mod.rs:396-397`) —
  **pattern** for `PyConformer::generate`'s dispatch; no `FromPyObject` derive.
- `CGBond(RelationRef[Bead])` (`views.py:808-819`) — **reuse** as the template
  for `Port`, with `handle_atom` in place of the colliding `handle`.
- `Bond(RelationRef[Atom])` (`views.py:725-736`) — **reuse unchanged** in
  `Fragment._relation_classes`.
- `class CoarseGrain(GraphViews, _RsCoarseGrain)` (`views.py:908-965`) —
  **reuse** as the Python-side template, including inherited `GraphViews.to_frame`
  (`:514-525`) upgrading to the rich `molrs.Frame`; `def_bead` (`:926-934`) is
  the precedent for a factory routing through a native typed call.
- `GraphViews` / `_relation_classes` (`views.py:473-549`) — **reuse**; the
  per-instance copy at `:499-503` lets `Fragment` register `ports` without
  touching the class dict.
- `_RsGraph.__init__ = _native_graph_init` pair (`views.py:1070-1075`) —
  **generalize** as far as the file allows: the pair gains its fourth entry and
  `_dump_graph:1045` gains no `Fragment` branch. Collapsing the four pairs into
  a loop is cosmetic with no behaviour attached; `/mol:refactor`.
- `PyConformer::generate` (`conformer/mod.rs:243-247`) — **generalize** to
  `&Bound<'_, PyAny>` with leaf-in / leaf-out dispatch.
- `PySmilesIR::to_atomistic` (`io/mod.rs:2168-2173`) + `fragment_scaling_data`
  (`ff/mod.rs:312-319`) — **pattern** for `to_fragment`'s error mapping and dict
  construction.
- **new** — `PyFragment`, `views.Port`, `views.Fragment`, `from_core_shadowed`,
  `to_fragment`: no existing symbol binds a fragment, and no existing helper
  performs the shadow-class construction.

### Debt found in the surface this link touches

Named per the iron law, routed rather than fixed here:

- **`views.Atomistic.def_bond` (`views.py:875-878`) writes classless bonds.** It
  goes through `_create_relation`, which calls the generic `add_relation`, so a
  bond built in Python carries neither `bond_type` nor `bond_number`, while
  `PyAtomistic.add_bond` (`:665`) → `Atomistic::add_bond` (`atomistic.rs:180`)
  stamps both. The Rust and Python construction doors disagree for `Atomistic`
  today. `Fragment.def_bond` is built the correct way from the start; fixing
  `Atomistic` is a behaviour change to an existing public path with its own
  regression test → **`/mol:fix`**.
- **Three spellings for one count.** `n_nodes` (the macro getter),
  `n_atoms` (`:749`) and `n_beads` (`:1448`) all answer "how many nodes", and
  `PyFragment` adds a fourth use of the pattern. Collapsing them is a public
  rename across three classes and two downstreams → **`/mol:refactor`**.
- **Dead docstring names in `molrs-python/src/io/mod.rs`** — `:2096` and `:2167`
  spell `parse_smiles`, which exists in no Python namespace, and `:2188` writes
  `molrs.SmilesIR` where the only spelling is `molrs.io.SmilesIR`. This link
  does not edit that file → **`/mol:docs`**.

## Files to create or modify

- `molrs-python/src/core/system/molgraph.rs` — `PyFragment` (+ `graph_world_impl!`,
  typed writers, `core`, `from_core`), `from_core_shadowed` with `PyAtomistic` /
  `PyCoarseGrain` rewired, the `with_world_mut` arm, module `//!` doc.
- `molrs-python/src/lib.rs` — `m.add_class::<PyFragment>()?` after
  `PyCoarseGrain` (`:312`).
- `molrs-python/src/conformer/mod.rs` — widened `generate`, extracted
  `report_to_py`, docstring `:217-242`.
- `molrs-python/src/io/cgsmiles.rs` — `to_fragment` on 01e's `PyCGSmilesIR`.
- `molrs-python/python/molrs/views.py` — `Port`, `Fragment`, `_RsFragment`
  import, patch pair, `__all__`.
- `molrs-python/python/molrs/__init__.py` — `_lib` import block, views import
  block, `__all__`.
- `molrs-python/python/molrs/_lib.pyi` — `class Fragment(Graph)`,
  `Conformer.generate` overloads.
- `molrs-python/docs/reference/python.md` — `::: molrs.Fragment` in
  *Topology and SMILES* (`:29-33`).
- `molrs-python/tests/test_fragment.py` (new) — the seam suite.
- `molrs-python/tests/test_views.py` — the `Fragment` shadow-type case.
- `molrs-python/tests/test_pickle.py` — the `Fragment` per-leaf case.

## Tasks

- [ ] Write failing seam tests in `molrs-python/tests/test_fragment.py`: `def_atom` ×3 + `def_bond` ×2 (assert `frag.bonds[0]["bond_type"] == 1` and `["bond_number"] == 1`) + one `def_port` on a real H handle; `n_ports == 1`, `n_atoms == 3`, `frag.ports[0] is frag.ports[0]`, `port.anchor` / `port.handle_atom`, `port["port_kind"] == "$"`, `port["port_label"] == ""`, `port["port_order"] == 1`; `set_frag_id` / `frag_id` plus `inherit_frag_ids()` returning the labelled count and propagating the id to a degree-1 neighbour; ports and `frag_id` surviving `copy()` and `Fragment.from_frame(to_frame())`; `def_port(..., "Z")` and `add_port(..., "Z")` each raising `ValueError`
- [ ] Implement `PyFragment` in `molrs-python/src/core/system/molgraph.rs` (pyclass, `#[new]`, `mol`/`mol_mut`/`core`, `graph_world_impl!(PyFragment);`, `add_atom`/`add_bond`/`add_port`/`n_ports`/`n_atoms`/`set_frag_id`/`frag_id`/`inherit_frag_ids`/`copy`/`to_frame`/`from_frame`), register it in `molrs-python/src/lib.rs` after `PyCoarseGrain`, and update the module `//!` doc to list `Fragment` in the hierarchy and to name the macro by its real name `graph_world_impl!` (`:16` says `graph_world_body!`, which does not exist)
- [ ] Write the failing `Fragment` shadow-type case in `molrs-python/tests/test_views.py` (`type(result) is molrs.Fragment` after `copy()` and `from_frame(to_frame())`, mirroring `:28-38`), then extract `pub(crate) fn from_core_shadowed<T>(py, leaf: T) -> PyResult<Py<T>> where T: PyClass<BaseType = PyGraph, Frozen = False>` in `molrs-python/src/core/system/molgraph.rs` and reduce `PyAtomistic::from_core`, `PyCoarseGrain::from_core` and `PyFragment::from_core` to one delegating line each
- [ ] Write the failing `Fragment` pickle case in `molrs-python/tests/test_pickle.py` (`type(roundtrip(_lib.Fragment())) is _lib.Fragment`; a `molrs.Fragment` with one port and a `frag_id` returns with both intact), then add `Port` and `Fragment` to `molrs-python/python/molrs/views.py` — `def_atom` on the generic node path, `def_bond` / `def_port` routing through the native writers plus `_intern_relation` — with the `_RsFragment` import, the `__init__`/`__reduce__` patch pair and `__all__`, and wire `molrs-python/python/molrs/__init__.py` (both import blocks, `__all__`)
- [ ] Write the failing `molrs.translate(frag, (1.0, 0.0, 0.0))` test in `molrs-python/tests/test_fragment.py` asserting the atom coordinates moved, then add the `PyFragment` arm to `with_world_mut` in `molrs-python/src/core/system/molgraph.rs` and name `Fragment` in its `PyTypeError`
- [ ] Write the failing `Conformer(speed="fast", seed=42).generate(frag)` test (returns a `molrs.Fragment` with coordinates and unchanged `n_ports`) plus a `TypeError` test for a non-graph argument, then widen `PyConformer::generate` in `molrs-python/src/conformer/mod.rs` to `&Bound<'_, PyAny>` with the `.cast` chain, extract `report_to_py`, and rewrite the docstring `Parameters` / `Returns` / `Raises` / `Examples` for both leaf types, the Å units and the caller's `inherit_frag_ids` step, replacing the dead `parse_smiles` name at `:239` with `molrs.io.SmilesIR`
- [ ] Write the failing `molrs.io.CGSmilesIR("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}").to_fragment()` test (a `dict` whose values are `molrs.Fragment`, `"PEO"` → `n_ports == 2`, `"OH"` → `n_ports == 1`), then implement `to_fragment` on `PyCGSmilesIR` in `molrs-python/src/io/cgsmiles.rs` through `PyFragment::from_core` and `smiles_error_to_pyerr`
- [ ] Add `class Fragment(Graph)` with every new pymethod to `molrs-python/python/molrs/_lib.pyi` (method-level, as `:841-877` does for `CoarseGrain`) plus the two `Conformer.generate` overloads, add `::: molrs.Fragment` to `molrs-python/docs/reference/python.md`, and write NumPy-style docstrings (`Parameters` / `Returns` / `Raises` / `Examples`, Å units, the `FragmentScaling` disambiguation, the `handle_atom` naming note) on every new symbol
- [ ] Run full check + test suite: `cargo fmt --manifest-path molrs-python/Cargo.toml --check`, `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D warnings` (`docs/releasing.md:20`), `cargo clippy --manifest-path molrs-wasm/Cargo.toml --target wasm32-unknown-unknown --all-targets -- -D warnings` (`docs/releasing.md:22`), `uv --directory molrs-python run --no-sync tox -e py`

## Testing strategy

Per `.claude/notes/testing.md` § *Language bindings*, these tests prove the
**seam only**: symbols import and construct, types and vocabularies cross
correctly, graph-out paths keep the public view type, errors map to the right
Python exception. Every chemical and geometric assertion about ports, embedding
and `frag_id` propagation belongs to the Rust unit tests in 02a / 02b / 02c and
is not repeated here. Tests are flat pytest functions importing only `molrs`
(plus `molrs._lib` where the native leaf is the subject).

**`molrs-python/tests/test_fragment.py`** (new) —
`test_fragment_ports_are_interned_live_views`; `test_def_bond_stamps_bond_class`
(`bond_type == 1`, `bond_number == 1`, the fact the native writer adds and the
generic relation path does not); `test_frag_id_round_trips` (per-atom write and
read, then `inherit_frag_ids()` returning `1` and the degree-1 neighbour
carrying the propagated id); `test_graph_out_paths_keep_the_public_fragment_type`
(`for result in (frag.copy(), molrs.Fragment.from_frame(frag.to_frame())):
type(result) is molrs.Fragment`, ports and `frag_id` intact);
`test_translate_moves_fragment_atoms`;
`test_conformer_returns_a_fragment` (one three-heavy-atom fragment,
`speed="fast"`, `seed=42`: result is `molrs.Fragment`, finite x/y/z in Å,
`n_ports` unchanged — runtime in seconds);
`test_conformer_rejects_a_non_graph`;
`test_def_port_rejects_an_unknown_kind` and
`test_add_port_rejects_an_unknown_kind` (both `ValueError`; the first is what
proves `def_port` reaches core `add_port`, since `_create_relation` would accept
`"Z"` silently); `test_to_fragment_returns_named_fragments`.

**`molrs-python/tests/test_views.py`** — one `Fragment` case in the shape of
`:28-38`, the assertion that catches a shadow-type regression.

**`molrs-python/tests/test_pickle.py`** — the per-leaf case beside the existing
`Atomistic` / `CoarseGrain` pair at `:208-227`.

**Per-leaf cases elsewhere, audited against the tree.** `test_replicate.py:15`
has a `CoarseGrain` case, but `replicate` wraps core `merge`, which 02a does not
ship for `Fragment` — no case, there is no method to test.
`test_graph_sink.py` is `Atomistic`-only (it has no `CoarseGrain` case) and
covers `merge` / `extract_subgraph` / `induced_subgraph`, none of which
`Fragment` ships; its one applicable subject, `copy`, is asserted in
`test_fragment.py` together with the shadow-type check — no case.
`test_subclass.py` contains no graph-leaf case at all (Box, Frame, Block only);
`views.Fragment(GraphViews, _RsFragment)` constructing *is* the subclass proof
and every test above exercises it — no case.

**Public-API example.** This repo has no `regressions/` tree (`CLAUDE.md`
§ Testing Rules). For a binder link the equivalent is `tests/test_fragment.py`
itself: public API only (`import molrs`), hard-coded expected values
(`n_ports == 2` for `"PEO"`, `n_ports == 1` for `"OH"`, `port_kind == "$"`,
`bond_type == 1`), no third-party software at test time and no values captured
from any. `PyFragment`'s NumPy `Examples` blocks carry the same script in prose,
and `molrs-python/tests/test_stub_parity.py` (01e) holds `_lib.pyi` to the new
class.

**Gate.** `cargo fmt --manifest-path molrs-python/Cargo.toml --check`;
`cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D warnings`
(`docs/releasing.md:20`);
`cargo clippy --manifest-path molrs-wasm/Cargo.toml --target wasm32-unknown-unknown --all-targets -- -D warnings`
(`docs/releasing.md:22`);
`uv --directory molrs-python run --no-sync tox -e py`.

## Out of scope

- **The 0.15.0 version bump.** The 16 literal `"0.14.0"` occurrences, the
  `Cargo.lock` regeneration and the `docs/releasing.md:9-64` checklist are the
  separate final link **`cgsmiles-03-release`**. This link changes no
  `Cargo.toml` and no `pyproject.toml`.
- **Any change under `molrs/src`.** `Fragment`, `PortKind`, `to_fragment` and
  the generic `generate` are 02a / 02b / 02c.
- **A validating `port(id)` reader on `PyFragment`** — `views.Port` is the read
  path; added by the link that first has a non-test consumer.
- **`remove_port` and `core_mut()`** — `remove_relation("ports", id)` is the
  removal door and 02a ships no core `remove_port`; nothing mutates a
  `Fragment` through the core handle.
- **A `PyPort` pyclass** — `views.Port` is the port view.
- **`merge` / `replicate` / `induced_subgraph` / `extract_subgraph` for
  fragments, and a `PyExtractedSubgraph` arm** — no core method exists to bind.
- **Force fields, MD, perception and typifiers accepting a `Fragment`** — those
  entry points are typed `&PyAtomistic` by contract; there is no Python-level
  `Fragment → Atomistic` promotion here (02c's `ElementGraph` is Rust-internal).
- **WASM / capi / cxxapi surfaces for `Fragment`** — no wasm surface can accept
  one; the C seams carry no graph-leaf vocabulary.
- **Routing the pickle restore path through `add_port`** — restore replays an
  already-validated object; `_load_graph` (`views.py:989-999`) stays raw.
- **The three routed debts named in Design** — `views.Atomistic.def_bond`
  writing classless bonds (`/mol:fix`), the `n_nodes` / `n_atoms` / `n_beads`
  triple spelling (`/mol:refactor`), and the dead `parse_smiles` /
  `molrs.SmilesIR` docstring names in `molrs-python/src/io/mod.rs:2096,2167,2188`
  (`/mol:docs`).
- **Collapsing `views.py`'s four monkey-patch pairs; caching the shadow-class
  lookup** — cosmetic and speculative respectively.
