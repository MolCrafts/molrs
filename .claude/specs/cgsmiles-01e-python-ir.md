---
title: CGsmiles IR Python bindings (chain link 01e)
slug: cgsmiles-01e-python-ir
status: approved
created: 2026-09-21
chain: cgsmiles (01a → 01b → 01c → 01d → 01e → 02a → 02b → 02c → 02d → 03)
depends_on: cgsmiles-01a-descriptors, cgsmiles-01b-graph, cgsmiles-01c-fragments, cgsmiles-01d-resolve
---

# CGsmiles IR Python bindings (chain link 01e)

## Summary

`molrs.io.CGSmilesIR("{[#PEO]|3}…")` becomes the Python door onto the CGsmiles reader that links 01a–01d build in Rust. Constructing it parses the string; reading it gives molpy every fact the notation carried — each resolution level's nodes with their names, partial charges, free-form annotations, bonding descriptors and `parent` back-pointers; each edge with its bond multiplicity and, when it was induced by a coarser level's pair, the `(level, pair)` that induced it; the fragment tables with their graph or atomistic bodies; and the resolved port pairs with both ends named — and `to_atomistic()` expands the lowest level into an `Atomistic`. Nothing is re-parsed on the Python side and no value crosses as an opaque string or an unnamed tuple position that a consumer must take apart again. The link also repairs the rot it sits on: the stub-freshness guard that `_lib.pyi` has always claimed but never had is added at class-name level and the 17 stale declarations it finds are removed, and the five docstrings advertising `molrs.parse_smiles` / `molrs.SmilesIR` — names the Python surface does not have — are corrected to the one spelling that exists.

## Domain basis

This link declares no physics: it is a binding over the notation rules that 01a–01d own (CGsmiles: cgsmiles.readthedocs.io; reference implementation github.com/gruenewald-lab/CGsmiles @ 910c9ee; Grünewald et al., *J. Chem. Inf. Model.* 2025, DOI 10.1021/acs.jcim.5c00064, cited for provenance only, as in 01b). No expected value below is re-derived here — the counts in the fixtures are the ones 01c and 01d already hand-derive and prove in Rust unit tests, reused only to prove that the seam hands the same numbers to Python.

Units at the boundary (`.claude/notes/science.md`): `CGNode.charge` is a **partial charge in elementary-charge units `e`**, crossing as a Python `float` (or `None` when the notation wrote no `q`); it is never a formal charge. `CGEdge.multiplicity` is a dimensionless count in `1..=4` — how many bonds the edge means between two instances, never a bond kind. `ResolvedPair.bond` and every index (`edge`, `index`, `port`, `parent`, `i`, `j`) is a dimensionless 0-based index.

## Design

### One front door, and why nothing here is called a Reader

The Python object is `molrs.io.CGSmilesIR(text)`: a `#[pyclass(module = "molrs.io", name = "CGSmilesIR")] pub struct PyCGSmilesIR { inner: molrs::io::smiles::CGSmilesIR, input: String }` with a `#[new] fn new(text: &str)` that calls `molrs::io::smiles::parse_cgsmiles`, mirroring `PySmilesIR` (`molrs-python/src/io/mod.rs:2102-2133`) field for field, including the retained `input` used only by `__repr__`. There is **no `CGSmilesReader`**: in this binding "Reader" already means a lazy, path-backed trajectory cursor (`XYZTrajReader`, `DCDTrajReader`, … — `module = "molrs.io.raw"`, `unsendable`, `read_frame(i)` / `n_frames`), and a text-in/IR-out parser is not that object. A reader-shaped `CGSmilesReader(text).read() -> CGSmilesIR` is molpy's to write, wrapping this class exactly as molpy's `SmilesReader` wraps `molrs.io.SmilesIR`. This paragraph is the traceable reason, and a shortened form of it goes into the `molrs.io` module docstring beside the existing "`SmilesIR` is here because SMILES is a *format*" paragraph (`python/molrs/io/__init__.py:20-22`).

`CGSmilesIR` exposes exactly two verbs' worth of surface: construction from text, and `to_atomistic()`. There is no `parse_cgsmiles` free function beside the constructor (that would be the dual public name `architecture-rules.md` § Naming forbids), no composite convenience that parses and expands in one call, and **no `n_levels` getter** — `len(ir.levels)` is the same fact, and two spellings of one fact is the rule's exact target. This is not an argument against `PySmilesIR::n_components`, which answers a different question: `components()` (`io/mod.rs:2190`) converts every component to an `Atomistic`, so `n_components` is an O(1) answer where the collection costs O(N·convert). `ir.levels` converts nothing, so no such getter is owed.

### Nested data: one pyclass per record, in the LammpsLog house style

Eight pyclasses, all `module = "molrs.io"`, all read-only:

| Python class | Rust wrapper | Getters |
|---|---|---|
| `CGSmilesIR` | `PyCGSmilesIR` | `levels`, `fragments`, `pairs`; methods `to_atomistic()`, `__repr__` |
| `CGGraph` | `PyCGGraph` | `nodes`, `edges` |
| `CGNode` | `PyCGNode` | `name`, `charge`, `annotations`, `descriptors`, `parent` |
| `CGEdge` | `PyCGEdge` | `i`, `j`, `multiplicity`, `derived_from` |
| `CGFragmentDef` | `PyCGFragmentDef` | `name`, `body` |
| `ResolvedPair` | `PyResolvedPair` | `edge`, `bond`, `src`, `dst`, `kind` |
| `PairEnd` | `PyPairEnd` | `end`, `index`, `port` |
| `BondingDescriptor` | `PyBondingDescriptor` | `kind`, `label`, `order` |

The seven nested classes follow `log.rs:18-160` exactly: `#[pyclass(module = "molrs.io", name = "…", frozen, skip_from_py_object)] #[derive(Clone)] struct Py… { inner: … }`, `#[getter]`s only, no setters, no `#[new]` (they are produced by `CGSmilesIR` and nowhere else), a `__repr__`, and nested values handed out as the matching class built by cloning the inner record (`log.rs:361`). `PyCGSmilesIR` itself is spelled like `PySmilesIR` (no `frozen`) because it is the same kind of object and the two must read alike.

`CGGraph` gets no `__len__`: on a graph the number is ambiguous between nodes and edges, and `LammpsThermo.__len__` is only defined because a thermo table has one obvious length. `annotations` crosses as `list[tuple[str, str]]` in parse order. `fragments` crosses as `list[dict[str, CGFragmentDef]]`, one dict per fragment block, filled in `BTreeMap` key order so iteration is deterministic. `pairs` crosses as `list[list[ResolvedPair]]`, parallel to `levels`, in resolution order.

### Two rules decide every remaining shape

**Rule 1 — an enum that *is* a count crosses as the count; an enum that is a name crosses as its lowercase name.** `CGBondOrder` is what 01b calls "a coarse-edge *multiplicity*", and `Single ↔ 1` is a bijection, so `CGEdge` exposes `multiplicity: int` and **no `order` getter**: two public getters for one bijective fact is the dual public name `architecture-rules.md` § Naming forbids, and the count is the form 01d's pair loop and molpy both consume. `ResolvedPair.kind` and `BondingDescriptor.order` are `BondKind`, which carries `aromatic` / `up` / `down` / `any` / `ring` beside the four counts and therefore is not a count at all; they cross as lowercase names. `DescriptorKind` (`"symmetric" | "left" | "right" | "shared"`) and `PairEnd.end` (`"sub" | "body"`) likewise.

**Rule 2 — when an existing public Python type distinguishes a sum's variants, there is no tag getter; when it does not (and inventing classes just to tell variants apart would be the alternative), the sum is a small frozen pyclass with a tag getter beside named payload getters.** Applied:

- `FragmentBody` — the payload types differ, so `CGFragmentDef` exposes `name` and `body` only. `body` returns a `CGGraph` or a `molrs.io.SmilesIR`; callers dispatch with `isinstance`. There is no `body_kind`: Python's type *is* the tag, and a second spelling of it is Rule 1's objection again.
- `EdgeOrigin` — one variant carries a payload and the other carries nothing, so `CGEdge` exposes `derived_from: tuple[int, int] | None` and **no `origin`**: `origin == "derived"` is exactly `derived_from is not None`.
- `PairEnd` — both variants carry `(usize, usize)`, so the payload type cannot distinguish them and Rule 2's second half applies: an eighth frozen pyclass with `end` (`"sub" | "body"`), `index` and `port`. An unnamed 3-tuple was rejected — a bare position is the thing this binding exists to prevent. Its rustdoc states which list `index` indexes: `levels[k+1].nodes` when `end == "sub"` (the child node carrying the port, whose own `parent` is the level-k instance), `levels[k].nodes` when `end == "body"` (the instance whose body holds the port, with `port` indexing that body's port list).

`BondingDescriptor` stays a pyclass under the same rule read from the other side: it is a product of three named, heterogeneous fields, and `PairEnd.port` is an **index into a node's `descriptors` list**, so molpy indexes it and then reads fields by name.

### Enums cross as lowercase strings, not as codes

Every enum that crosses by name crosses as the lowercase spelling of its Rust variant — `CGBondOrder` excepted, which crosses as its count per Rule 1. This follows the io-local convention: `build_smiles_emit_options` (`src/io/mod.rs:2288-2328`) and `helpers.rs:102` match lowercase names exhaustively and raise `PyValueError` on an unknown spelling.

The crate has a **conflicting** convention one module over. `set_bond_class(handle, bond_type, bond_number)` (`src/core/system/molgraph.rs:808`) crosses **two** `u32` codes, documented at `:806-807` as `0 unknown, 1 single, 2 double, 3 triple, 4 aromatic` and `0 unknown, 1 single, 2 double, 3 triple, 4 quadruple`; `bond_type` (`:829`) reads one back. As a *pair* those codes are expressive — 01d's `Quadruple → (BondType::Double, BondNumber::Quadruple)` loses nothing, since the quadruple lives in the second code. The reason strings win here is narrower and survives that: `BondKind` also carries `up`, `down`, `any` and `ring`, and **none of the four has a code in `BondType` or in `BondNumber`** (`core/system/bond.rs:28-54`) — they are notation-level facts with no storage encoding at all, and a value with no storage code crosses by its own name rather than by a code invented at the boundary. Encoding five of nine variants and stranding four would be a partial convention, which is worse than either whole one. The convention split itself is named in Out of scope and routed.

The four spellings that cannot occur today (`"up"`, `"down"`, `"any"`, `"ring"` — stereo and SMARTS-query kinds that never resolve an inter-fragment bond, per 01d R4.14) are still produced by the private mapping functions, which are **total**: `fn bond_kind_name(BondKind) -> &'static str` and `fn descriptor_kind_name(DescriptorKind) -> &'static str` match every variant with no `unreachable!()` and no `_ =>` arm, so no input can panic across the seam (`.claude/notes/ffi.md` Rule 1) and a future variant is a compile error rather than a runtime one. `"shared"` is likewise unreachable (01c raises `CgSquashUnsupported`) and likewise spelled.

### Reaching a `SmilesIR` body reuses `PySmilesIR`

`CGFragmentDef.body` on an atomistic body must return the existing `molrs.io.SmilesIR`, not a new CG-only wrapper — a second Python class around the same Rust `SmilesIR` would be the dual public name the naming rule forbids. `PySmilesIR`'s fields are private to `src/io/mod.rs`, so this link adds `pub(crate) fn PySmilesIR::from_core(inner: molrs::io::smiles::SmilesIR, input: String) -> Self` there, in a **plain `impl PySmilesIR` block, not `#[pymethods]`** — the `PyLammpsLog::new` precedent (`log.rs:631-635`) — so nothing new appears on the Python surface. The `input` handed to it is the fragment definition's own source text, recovered non-panickingly as `self.input.get(def.span.start..def.span.end).unwrap_or(&def.name).to_owned()`: a byte range that is not a char boundary yields the fragment name instead of a panic, since `input` only feeds `__repr__`. No raw `self.input[a..b]` slicing appears anywhere in the new file.

### Errors

`smiles_error_to_pyerr` (`src/helpers.rs:86`) is reused **as is** for every fallible call: `parse_cgsmiles` and `to_atomistic` both return the same `SmilesError` type that it already maps to `ValueError`. No second mapper, no `create_exception!` here. Its flattening of `kind` / `span` / `input` into a message string is pre-existing rot, named in Out of scope and routed with the fix that is already available in-tree (`src/error.rs:12,21`).

### Rot found in the touched surface, fixed here (iron law)

**1. The stub has no freshness guard, and 17 declarations have gone stale.** `python/molrs/_lib.pyi:1-6` names `tests/test_stub_parity.py` as the guard; no such file exists (`molrs-python/tests/` verified). This link writes it, at **class-name level**: it `ast.parse`s the `.pyi` and collects top-level `ClassDef` names, collects `{name for name in dir(molrs._lib) if isinstance(getattr(molrs._lib, name), type) and not name.startswith("_")}`, and asserts the two sets are equal in both directions. One structural exemption, and only one: a `_lib` **submodule** declared in the stub as a class (`class md:`, `_lib.pyi:2797`) is exempt, recognised by `inspect.ismodule(getattr(_lib, name))` — not by a name allowlist.

The test must read the **source-tree** stub, `Path(__file__).parents[1] / "python" / "molrs" / "_lib.pyi"`, not the installed package: `tox -e py` builds a wheel and force-installs it, then asserts the imported `molrs` resolves under `site-packages` (`pyproject.toml:96-107`), so `Path(molrs.__file__).parent / "_lib.pyi"` would check a copy of the stub rather than the file a contributor edits.

The drift is measured, not estimated: **17 stub-only top-level class names, 183 lines, at `_lib.pyi:1341-1528`, `:1880-1884` and `:2204-2207`**, and the repair direction for all of them is **deletion**, because each is a Python-side class whose type already comes from its own `.py` module and none is a `_lib` export:

- `:1341-1528` — 15 pure-Python view classes that live in `python/molrs/ff/forcefield.py`: `Parameters`, `Type`, `AtomType`, `BondType`, `AngleType`, `DihedralType`, `ImproperType`, `PairType`, `Style`, `AtomStyle`, `BondStyle`, `AngleStyle`, `DihedralStyle`, `ImproperStyle`, `PairStyle`.
- `:2204-2207` — `Compute`, a `Protocol` defined at `python/molrs/compute/protocol.py:17`.
- `:1880-1884` — `ChargeModel`, a `Protocol` the stub also uses as the declared base of three **real** `_lib` classes: `class BccModel(ChargeModel)` (`:1886`), `class MullikenModel(ChargeModel)` (`:1902`), `class GasteigerModel(ChargeModel)` (`:1909`). Deleting the entry therefore also means dropping the `(ChargeModel)` base from those three headers — the native classes have no Python base class, and the `Protocol` in the Python package still matches them structurally, so nothing a caller can observe is lost.

**Not to be touched:** `Block`, `Frame`, `Atomistic`, `CoarseGrain` and `ForceField` exist on **both** sides — a `_lib` class and a Python subclass or wrapper of the same name — and are correct stub entries. The parity test compares against what `_lib` exports, so it passes on them; an implementation that "cleans up duplicates" by removing them breaks the typed surface.

**2. The `SmilesIR` stub entry has drifted** (`_lib.pyi:1061-1066`): it declares only `n_components` and `to_atomistic`, missing `__init__`, `components`, `write_smiles`, `write_smarts` and `from_atomistic`. Repaired here, since the new test sits next to it and this link edits that file anyway.

**3. Dead names in docstrings.** `molrs.parse_smiles` and `molrs.SmilesIR` appear in `src/io/mod.rs:2096, 2124, 2167, 2188` and `_lib.pyi:3212`; neither exists — the only spelling is `molrs.io.SmilesIR` (`python/molrs/__init__.py:29` says so in prose). Both files are edited by this link, so all five lines are fixed here. The identical rot at `src/conformer/mod.rs:239` is **not** touched: 02d edits that file and fixes it there. Nothing executes docstring examples — `pyproject.toml:56-57` sets `testpaths` with no `--doctest-modules` — so these rot silently; adding a doctest runner is a deferred item for `/mol:note`, not work for this link.

### Placement

New file `molrs-python/src/io/cgsmiles.rs`, declared `pub mod cgsmiles;` from `src/io/mod.rs` beside `log` and `mrec` — the two precedents for a multi-pyclass family getting its own file, and `src/io/mod.rs` is already 2528 lines. Unlike those two the declaration is **ungated**: `log` / `mrec` are `#[cfg(feature = "fs")]` because they need the filesystem store, while `parse_cgsmiles` is pure text and the binder already enables `full` (hence `smiles`) unconditionally (`molrs-python/Cargo.toml:45`). Registration is `m.add_class::<io::cgsmiles::PyCGSmilesIR>()?` and its seven siblings in the flat `_lib` pymodule next to `src/lib.rs:257`; the `molrs.io` namespace is the shim, so each class is also re-exported by name in `python/molrs/io/__init__.py` (import block `:35-53`, `__all__` at `:946`) and declared in `_lib.pyi`. The `module = "molrs.io"` attribute is cosmetic, as it already is for the `Lammps*` family.

### Reuse decision

- `PySmilesIR` (`src/io/mod.rs:2102`) — **pattern** for `PyCGSmilesIR`'s shape (`{ inner, input }`, `#[new]` from `&str`, `__repr__` echoing the input), and **reuse** as the return type of `CGFragmentDef.body`, reached through the new `pub(crate) from_core` in a plain `impl` block. Not a reader class; no `CGSmilesReader`.
- `PySmilesIR::n_components` (`:2144`) — **pattern, not applicable**: it is an O(1) answer to a question whose collection form converts every component; `ir.levels` converts nothing, so no `n_levels` is owed. `PySmilesIR` exposes no nested data at all, so the nodes/edges/fragments/pairs surface is genuinely new.
- `LammpsThermo` / `LammpsLog*` family (`src/io/log.rs:18,48,91,163,205,251`) — **pattern** for all seven nested classes: one small `#[pyclass(frozen, skip_from_py_object)]` per record type, `#[getter]`s, `__repr__`, nested values built by cloning the inner record, re-exported one-by-one in the shim. `__getitem__` / `__contains__` / `__len__` are not copied — no CG record is a keyed table.
- `PyLammpsLog::new` (`log.rs:631-635`) — **pattern** for `PySmilesIR::from_core`: a `pub(crate)` constructor in a plain `impl`, invisible to Python.
- `build_smiles_emit_options` (`:2288`) and `helpers.rs:102` — **pattern**: lowercase strings at the boundary, matched exhaustively.
- `set_bond_class` / `bond_type` (`src/core/system/molgraph.rs:808,829`) — **pattern (rejected)**: a two-code storage protocol that has no encoding for four of `BondKind`'s nine variants; reason stated above, split routed.
- `smiles_error_to_pyerr` (`src/helpers.rs:86`) — **reuse** unchanged; `parse_cgsmiles` returns the same error type.
- `PySmilesIR::to_atomistic` (`:2170`) — **reuse** as the exact template: `.map_err(smiles_error_to_pyerr)?` then `PyAtomistic::from_core(py, mol)` (`src/core/system/molgraph.rs:963`).
- `create_exception!` (`src/error.rs:12,21`) — **available, not used**: a typed `CGSmilesError` is the fix for the error-flattening rot, and that fix is one decision (which errors get types) applied to the whole `io::smiles` surface, not a CG-only carve-out. Routed, not done here.
- Flat `_lib` pymodule + shim re-export (`src/lib.rs:139,257`; `python/molrs/io/__init__.py:49`) — **reuse**: no Rust-side submodule is introduced.
- `PyAtomistic::from_core` — **reuse**, unchanged.

### Predecessor contract this binding assumes — verify, do not add

The CG IR types (`CGSmilesIR`, `CGGraph`, `CGNode`, `CGEdge`, `CGFragmentDef`, `FragmentBody`, `ResolvedPair`, `PairEnd`, `EdgeOrigin`, `CGBondOrder`, `BondingDescriptor`) derive `Debug, Clone, PartialEq` — frozen into 01a's (`BondingDescriptor`, `DescriptorKind`: structurally forced, since `AtomNode` already derives all three at `molrs/src/io/smiles/chem/ast.rs:92-93` and carries `descriptors: Vec<BondingDescriptor>`), 01b's, 01c's and 01d's contracts. The nested wrappers hold clones, per the `LammpsLog` pattern. The implementation **verifies** those derives and does not add them: this link changes no file under `molrs/src/`, so a missing derive is a predecessor defect and a hard stop reported by `/mol:impl`, not a fix made here.

## Files to create or modify

- `molrs-python/src/io/cgsmiles.rs` (new) — the eight pyclasses and the two total enum-name functions.
- `molrs-python/src/io/mod.rs` — `pub mod cgsmiles;` beside `log` / `mrec`; `pub(crate) fn PySmilesIR::from_core` in a plain `impl` block; the four dead-name docstring fixes at `:2096, :2124, :2167, :2188`.
- `molrs-python/src/lib.rs` — register all eight classes next to `:257`.
- `molrs-python/python/molrs/io/__init__.py` — re-export the eight names (import block `:35-53`), add them to `__all__` (`:946`), extend the module docstring with the `CGSmilesIR` paragraph and the "no `CGSmilesReader`" rule.
- `molrs-python/python/molrs/_lib.pyi` — declare the eight classes; delete the 17 stale declarations (`:1341-1528`, `:1880-1884`, `:2204-2207`), the `(ChargeModel)` base from `:1886`, `:1902`, `:1909`, the orphaned header `:1333-1339` and the unused `Protocol` import (`:17`), and reword the comments at `:1874-1876` / `:2177-2182`; repair the `SmilesIR` entry (`:1061-1066`); fix the `molrs.SmilesIR` docstring name at `:3212`.
- `molrs-python/tests/test_cgsmiles.py` (new) — the FFI-seam smoke tests and the public-API example.
- `molrs-python/tests/test_stub_parity.py` (new) — class-name-level stub parity.

## Tasks

- [ ] Write `molrs-python/tests/test_stub_parity.py` (failing against today's stub): `ast`-parse the **source-tree** `Path(__file__).parents[1] / "python" / "molrs" / "_lib.pyi"` for top-level class names, compare both directions against the classes `molrs._lib` exports, exempting only `_lib` submodules via `inspect.ismodule`; then delete the 17 stale declarations from `molrs-python/python/molrs/_lib.pyi` (`:1341-1528` the 15 `ff/forcefield.py` view classes, `:1880-1884` `ChargeModel`, `:2204-2207` `Compute`) together with the `(ChargeModel)` base on `:1886`, `:1902`, `:1909`, the orphaned section header at `:1333-1339` (it describes the deleted view classes), the then-unused `Protocol` import at `:17`, and the comments at `:1874-1876` and `:2177-2182` that name the deleted types (rewritten so they describe what remains), leaving `Block` / `Frame` / `Atomistic` / `CoarseGrain` / `ForceField` untouched, and repair the drifted `SmilesIR` entry at `:1061-1066`
- [ ] Fix the four dead-name docstrings in `molrs-python/src/io/mod.rs` (`:2096`, `:2124`, `:2167`, `:2188`) and the one at `molrs-python/python/molrs/_lib.pyi:3212` to `molrs.io.SmilesIR`, and add `pub(crate) fn PySmilesIR::from_core(inner, input)` to a plain `impl PySmilesIR` block in `molrs-python/src/io/mod.rs`
- [ ] Write failing tests in `molrs-python/tests/test_cgsmiles.py` for the graph surface on F2 `{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}` — `len(ir.levels) == 1`, 5 nodes, 4 edges, node names, `charge is None`, `annotations == []`, `parent is None`, `edges[0].multiplicity == 1` and `edges[0].derived_from is None` — then implement `PyCGSmilesIR`, `PyCGGraph`, `PyCGNode`, `PyCGEdge` in `molrs-python/src/io/cgsmiles.rs` (new) with `pub mod cgsmiles;` in `molrs-python/src/io/mod.rs`, registration in `molrs-python/src/lib.rs`, shim re-exports in `molrs-python/python/molrs/io/__init__.py` and declarations in `molrs-python/python/molrs/_lib.pyi`
- [ ] Write failing tests in `molrs-python/tests/test_cgsmiles.py` for descriptors and fragment tables on F8 `{[#B1][#B2][#B1]}.{#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.{#PEO=[>]COC[<],#PE=[>]CC[<]}` — `len(ir.fragments) == 2`, `isinstance(fragments[0]["B1"].body, molrs.io.CGGraph)` with `body.nodes[0].descriptors[0]` reading `("right", "", None)` and `nodes[1].descriptors[0].kind == "left"`, `isinstance(fragments[1]["PEO"].body, molrs.io.SmilesIR)` — then implement `PyBondingDescriptor` and `PyCGFragmentDef` (body via `PySmilesIR::from_core`) in `molrs-python/src/io/cgsmiles.rs` plus their registration, shim re-export and stub declaration
- [ ] Write failing tests in `molrs-python/tests/test_cgsmiles.py` for pairs and derived-edge provenance — F2 `len(ir.pairs[0]) == 4` with `kind == "single"` and both ends `end == "body"`; F8 `len(ir.levels[1].nodes) == 6` with `parent` values `[0,0,1,1,2,2]`, `len(ir.levels[1].edges) == 5` whose first three have `derived_from is None` and whose last two have `derived_from == (0, 0)` and `(0, 1)`, and `ir.pairs[0][0].src.end == "sub"` with `int` `index` / `port` — then implement `PyResolvedPair` and `PyPairEnd` in `molrs-python/src/io/cgsmiles.rs` plus registration, shim re-export and stub declaration
- [ ] Write failing tests in `molrs-python/tests/test_cgsmiles.py` for expansion, error mapping and repr — F2 `ir.to_atomistic()` returns an `Atomistic` with `n_atoms == 11` and `n_relations("bonds") == 10`; `molrs.io.CGSmilesIR("{[#A]")` and `molrs.io.CGSmilesIR("")` raise `ValueError`; `repr(ir)` contains the input string — then implement `to_atomistic` (via `smiles_error_to_pyerr` then `PyAtomistic::from_core`) and `__repr__` in `molrs-python/src/io/cgsmiles.rs`
- [ ] Write a failing test asserting the namespace and single-spelling contract (`molrs.io.CGSmilesIR` imports; all eight names are in `molrs.io.__all__`; `molrs.io` has no `CGSmilesReader`; `molrs` has no `parse_cgsmiles`; a `CGEdge` has no `order` attribute; a `CGFragmentDef` has no `body_kind`; a `CGSmilesIR` has no `n_levels`), then extend the `molrs-python/python/molrs/io/__init__.py` module docstring with the `CGSmilesIR` paragraph and the recorded reason there is no `CGSmilesReader` (molpy wraps this class the way its `SmilesReader` wraps `molrs.io.SmilesIR`)
- [ ] Run the gate: `cargo fmt --manifest-path molrs-python/Cargo.toml --check`, `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D warnings`, and `uv --directory molrs-python sync --no-install-project --extra dev && uv --directory molrs-python run --no-sync tox -e py`; name the deferred items in the implementation summary for `/mol:note` (the `smiles_error_to_pyerr` flattening, method-level stub parity, the missing doctest runner at `pyproject.toml:57`, the names-vs-codes enum convention split, and `src/conformer/mod.rs:239` left for 02d)

## Testing strategy

Per `CLAUDE.md` § Testing Rules and `.claude/notes/testing.md`, binding tests prove the **seam only** — symbols import and construct, types and values cross correctly, error mapping works — and never re-derive numerics the Rust suite proves. Grammar and pairing depth stay in 01b–01d's inline Rust tests; every count asserted here is a value those tests already prove in Rust, reused to show Python sees the same number. Tests are flat under `molrs-python/tests/` (`pyproject.toml:57` `testpaths = ["tests"]`), self-contained per `tests/conftest.py:1-7` — no third-party scientific software, no fixture corpus, fixtures are inline string literals — and modelled on `tests/test_smiles_emit.py`. Each test targets one behaviour of one binding.

**`molrs-python/tests/test_cgsmiles.py`.**

- *Happy path, F2* `{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}`: `len(ir.levels) == 1`; `levels[0]` has 5 nodes and 4 edges; `[n.name for n in levels[0].nodes] == ["OH","PEO","PEO","PEO","OH"]`; every `charge is None`, `annotations == []`, `parent is None`; every edge `multiplicity == 1` and `derived_from is None`; `len(ir.fragments) == 1` with keys `{"OH","PEO"}`; `len(ir.pairs[0]) == 4`, each `kind == "single"`, each `src.end == dst.end == "body"`.
- *Happy path, F8* (the multi-level string above): `len(ir.levels) == 2`; `levels[0]` 3 nodes / 2 edges; `levels[1]` 6 nodes with `parent` values `[0,0,1,1,2,2]` and names `["PEO","PEO","PE","PE","PEO","PEO"]`; `levels[1].edges` has 5 entries, the first three with `derived_from is None` and the last two with `derived_from == (0, 0)` and `(0, 1)`; `fragments[0]["B1"].body` is a `molrs.io.CGGraph` whose `nodes[0].descriptors[0]` reads `kind == "right"`, `label == ""`, `order is None`, with `"left"` on node 1; `fragments[1]["PEO"].body` is a `molrs.io.SmilesIR`; `ir.pairs[0][0].src.end == "sub"`.
- *Enum spellings* — one test asserting every string that crosses is lowercase and drawn from its documented set (descriptor `kind`, `ResolvedPair.kind`, `BondingDescriptor.order`, `PairEnd.end`), so a future int-code regression fails loudly rather than reading as a truthy value.
- *Types at the boundary* — `charge` is a `float` (or `None`) and never a string, on a charged fixture `{[#A;q=-0.5]}`; `annotations` is a `list` of 2-tuples of `str`; `i`, `j`, `multiplicity`, `edge`, `bond`, `parent`, `index`, `port` are `int`; `derived_from` is a 2-tuple of `int` or `None`.
- *Single spelling* — a `CGEdge` has no `order`, a `CGFragmentDef` has no `body_kind`, a `CGSmilesIR` has no `n_levels`; `molrs.io` has no `CGSmilesReader` and `molrs` no `parse_cgsmiles`.
- *Error mapping* — `molrs.io.CGSmilesIR("{[#A]")` and `molrs.io.CGSmilesIR("")` each raise `ValueError` (the mapping `smiles_error_to_pyerr` performs) with a non-empty message. Which `SmilesErrorKind` was raised is 01b–01d's assertion, not this link's.
- *Read-only contract* — setting any getter (e.g. `ir.levels[0].nodes[0].name = "X"`) raises `AttributeError`, and `molrs.io.CGNode()` raises `TypeError`: the nested classes have no `#[new]`.
- *Public-API example (this repo's substitute for a `regressions/` script)* — `CLAUDE.md` § Testing Rules and `.claude/notes/testing.md` state there is no `regressions/` tree in this repo; the Rust links use a rustdoc doctest, but nothing runs doctests on the Python side (`pyproject.toml:57` has no `--doctest-modules`, named as a deferred item). The equivalent here is `test_cgsmiles_f2_public_api` in `molrs-python/tests/test_cgsmiles.py`: it goes through the public path only (`import molrs`; `molrs.io.CGSmilesIR(F2)`; `.to_atomistic()`) and asserts the hard-coded `n_atoms == 11`, `n_relations("bonds") == 10`, `len(ir.levels) == 1`, `len(ir.pairs[0]) == 4`. Values are hand-derived in 01c / 01d from the notation rules; no third-party tool produced them and none runs at test time.

**`molrs-python/tests/test_stub_parity.py`.** One test: the set of top-level class names in the source-tree `python/molrs/_lib.pyi` (read by `ast.parse`, never by regex over source text) equals the set of public classes `molrs._lib` exports, with the only exemption computed structurally by `inspect.ismodule`. Both directions are asserted; there is no name allowlist. The docstring records two facts: the gate is the native default-feature wheel `tox -e py` builds, and the stub is read from the source tree because that wheel is installed non-editable (`pyproject.toml:96-107`).

**Gate.** `cargo fmt --manifest-path molrs-python/Cargo.toml --check` and `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D warnings` (binders are addressed by manifest path, `CLAUDE.md` § Crate Structure; a root `cargo fmt` does not see this workspace); `uv --directory molrs-python sync --no-install-project --extra dev && uv --directory molrs-python run --no-sync tox -e py`.

## Out of scope

- **A `CGSmilesReader` class, and any reader-shaped `read()` surface** — "Reader" in this binding means a lazy path-backed trajectory cursor; the reader-shaped API belongs to molpy, wrapping `molrs.io.CGSmilesIR`.
- **`smiles_error_to_pyerr` flattening `kind` / `span` / `input` into a `ValueError` string** (`src/helpers.rs:86`) — pre-existing rot, found and named. The fix is a typed exception via `create_exception!` (`src/error.rs:12,21`, where `BlockDtypeError` / `UnitsError` already live), applied to the whole `io::smiles` error surface rather than to CGsmiles alone; routed through the implementation summary, not done here. This link deliberately does not add a second mapper.
- **Method- and parameter-level stub parity** across the 3441-line `_lib.pyi` — this link adds the class-name-level guard only, which is what makes a *new* class's absence from the stub impossible. The finer parity that `_lib.pyi:1-6` also claims is routed.
- **A doctest runner** — `pyproject.toml:56-57` sets `testpaths` with no `--doctest-modules`, so nothing executes the docstring examples this link repairs. Named for `/mol:note`; adding a runner would re-gate every existing docstring in one change and belongs in its own spec.
- **`src/conformer/mod.rs:239`** — the same `molrs.SmilesIR` dead name; 02d edits that file and fixes it there. Touching it here would put one fix in two links.
- **The names-vs-codes enum convention split** (lowercase names in `io`, the two-code `set_bond_class` / `bond_type` protocol in `core`, `src/core/system/molgraph.rs:808,829`) — named, reason for this link's choice recorded above, reconciliation routed. Changing the core codes is a breaking Python API change with its own spec.
- **Spans at the Python boundary** — `CGSmilesIR.span`, `CGNode.span`, `CGEdge.span` and `CGFragmentDef.span` are not exposed; no named caller needs byte offsets, and a `Span` encoding without one would be surface invented on spec. The Rust values are untouched and a later link can add them.
- **WASM and C bindings** for `parse_cgsmiles`, and any Zensical site page.
- **Writing CGsmiles from Python** (`Atomistic` or `CGSmilesIR` → string) — no link in this chain covers emission yet.
- **Any change to the Rust core crate** — the notation rules, error kinds, expected values and the `Debug, Clone, PartialEq` derives are 01a–01d's; a missing derive is a hard stop reported by `/mol:impl`, not repaired here.
- **The 0.15.0 version bump** — `cgsmiles-03-release`, the last link in the chain.
