---
title: amber-prmtop-complete-04-python — special-bonds weights on the Python ForceField surface
status: code-complete
created: 2026-09-04
depends_on: [amber-prmtop-complete-03-lammps]
---

# amber-prmtop-complete-04-python — special-bonds weights on the Python ForceField surface

## Summary

The AMBER 1-4 scale weights that phase 02 lands on the native `ForceField`
(from `SCEE_SCALE_FACTOR` / `SCNB_SCALE_FACTOR`) and phase 03 emits from the
LAMMPS and GROMACS writers are today invisible and unsettable from Python:
`molrs.ForceField` exposes no accessor for `SpecialBonds`, and the ergonomic
`ForceField._from_raw` replay drops them on the floor for **every** reader
result and for `subset`. This phase — binder only, `molrs-python/` — adds two
read-only numpy properties (`special_bonds_lj`, `special_bonds_coul`) and one
whole-struct writer (`set_special_bonds`) to `PyForceField`, forwards the
weights through `_from_raw`, types all three in `_lib.pyi`, and deletes the
dual-name Python alias `read_prmtop` so `read_amber_prmtop` is the single name
on the binder surface. No writer or reader code changes here: the regression
example's writer legs assert the `special_bonds` / `fudgeLJ` / `fudgeQQ`
contract **landed by phase 03**, and its prmtop leg asserts the divisor →
weight conversion **landed by phase 02**, so this spec cannot go green before
`amber-prmtop-complete-03-lammps` is merged.

## Domain basis

`SpecialBonds` carries two dimensionless triples, ordered `[1-2, 1-3, 1-4]`,
one for Lennard-Jones and one for Coulomb (`molrs/src/ff/forcefield/mod.rs:669-674`).
A weight of `0.0` fully excludes that neighbour class; `1.0` leaves it at full
strength. molrs's default is `[0.0, 0.0, 1.0]` for both
(`mod.rs:676-685`).

AMBER prmtop stores the 1-4 terms as **divisors**, not weights, and the reader
inverts them (`molrs/src/ff/forcefield/readers/prmtop.rs:200-211`):

```text
lj_14   = 1 / SCNB      (SCNB default 2.0)  → 0.5
coul_14 = 1 / SCEE      (SCEE default 1.2)  → 0.8333333333333334
```

`1/2.0 = 0.5` is binary-exact; `1/1.2` is not — its nearest `f64` is
`0.8333333333333334` (Python `repr`), and that literal is what a round-trip
must reproduce. All values cross the FFI seam as `f64` (`F = f64` is invariant,
CLAUDE.md § *Type Precision Principle*), so the Python side is
`numpy.float64`, never `float32`.

References:

- AMBER prmtop format, `SCEE_SCALE_FACTOR` / `SCNB_SCALE_FACTOR` sections —
  https://ambermd.org/FileFormats.php (per-dihedral divisors; absent sections
  default to 1.2 / 2.0).
- GAFF: J. Wang, R. M. Wolf, J. W. Caldwell, P. A. Kollman, D. A. Case,
  *Development and testing of a general Amber force field*,
  J. Comput. Chem. **25**, 1157 (2004). DOI 10.1002/jcc.20035 — the
  SCEE = 1.2 / SCNB = 2.0 convention.
- GLYCAM06: M. B. Tessier, M. B. Kirschner et al.,
  J. Comput. Chem. **29**, 622 (2008). DOI 10.1002/jcc.20820 — a force field
  that sets SCEE = SCNB = 1.0, i.e. why the divisors are read from the file
  rather than hard-coded.
- LAMMPS `special_bonds` command semantics (the `lj`/`coul` keyword form phase
  03's writer emits).

## Design

**Entities touched.** One existing pyclass, `PyForceField`
(`molrs-python/src/ff/mod.rs:212-215`), which owns a native
`molrs::ff::ForceField` in `inner`. The native type already has the full
accessor pair — `ForceField::special_bonds() -> &SpecialBonds`
(`molrs/src/ff/forcefield/mod.rs:717-719`) and
`ForceField::set_special_bonds(SpecialBonds)` (`:724-726`) — and `subset`
already copies the weights (`:961`). Nothing is added to the native layer.

**New public symbols (three, all members of `ForceField`).**

- `#[getter] special_bonds_lj(&self, py) -> Bound<'py, PyArray1<NpF>>` —
  a length-3 float64 **copy** of `inner.special_bonds().lj`.
- `#[getter] special_bonds_coul(&self, py) -> Bound<'py, PyArray1<NpF>>` —
  the same for `.coul`.
- `set_special_bonds(&mut self, lj: [f64; 3], coul: [f64; 3]) -> PyResult<()>`
  — a **whole-struct** write mirroring the native setter's signature; PyO3's
  fixed-size-array extraction raises `ValueError` for a wrong-length sequence,
  which is the error contract (no hand-rolled length check, no partial write).
  The docstring states the consequence: a caller who wants to change only the
  Coulomb triple reads `special_bonds_lj` first and passes it back — the write
  is read-modify-write by design, never a half-declared force field.

Shape check (CLAUDE.md § *Shape check*): the natural owning type is
`ForceField`, so these are members, not free functions; the getters are one
named read each and the setter is one named write, so no further split; there
is no context bag. Two getters rather than one `special_bonds` tuple/object
keeps each property a single numpy array the caller can do arithmetic on
directly, and it is the shape `PySimBox` already uses for its geometry triples.

**Naming and construction follow the closest in-tree pattern**, `PySimBox`'s
geometry properties (`molrs-python/src/core/spatial/simbox.rs:280-305`):
`#[getter]`, `Bound<'py, PyArray1<NpF>>`, `…to_owned().into_pyarray(py)` —
i.e. the returned array is a fresh copy, so mutating it cannot reach back into
the force field, and two reads return two distinct arrays. The new code reads
like `PySimBox::lengths`.

**Stored-but-not-applied semantics (architect 🔴 #2).** molrs realises 1-2 /
1-3 *exclusion* by **omitting** those pairs from the neighbour list, not by
multiplying a weight — the native doc says so
(`molrs/src/ff/forcefield/mod.rs:659-667`) and the Python MD driver refuses
pair-style MD over a bonded topology for exactly that reason
(`molrs-python/python/molrs/md/driver.py:337-347`). Only index `[2]` is
consumed by kernels. The doc comment on **all three** members must state that
entries `[0]` and `[1]` are stored and round-tripped for format fidelity but
are never applied by molrs kernels, and the seam suite pins that as behaviour
(a `set_special_bonds(lj=[1.0, 1.0, 0.5], …)` reads back `[1.0, 1.0, 0.5]`
unchanged). The signature stays a whole-struct write: exposing only `_14`
scalars would make the file-fidelity round-trip impossible.

**The production defect being fixed.** `ForceField._from_raw`
(`molrs-python/python/molrs/ff/forcefield.py:533-556`) rebuilds the ergonomic
subclass by replaying styles and types field-by-field. It is on the return path
of **13** call sites — every XML / OPLS / LAMMPS / prmtop / GROMACS reader
wrapper and `ForceField.subset` (`forcefield.py:559, 816, 820, 824, 828, 832,
836, 846, 853, 864, 869, 911`) — so the weights the native readers set are
discarded before any Python caller sees them. One forwarding line inside
`_from_raw`, placed after the style/type replay, repairs all 13 paths; no
per-reader patching.

**Dual-name deletion (routed here from phase 01's architect review).**
`.claude/notes/architecture-rules.md` § *Naming*: "No dual public names for the
same symbol (0.12: delete façades, not deprecate)." The binder carries **two**
such aliases in the same files, four lines apart, and both go: `read_prmtop`
(alias of `read_amber_prmtop`) at `molrs-python/src/io/mod.rs:1062-1066`, its
registration at `molrs-python/src/lib.rs:204`, the stub at
`molrs-python/python/molrs/_lib.pyi:888`, and the ergonomic re-exports at
`python/molrs/io/__init__.py:337-339` & `:911` and `python/molrs/io/raw.py:32`
& `:83`; and `read_inpcrd` (alias of `read_amber_inpcrd`) at
`molrs-python/src/io/mod.rs:1030-1034`, `molrs-python/src/lib.rs:202`,
`_lib.pyi:886`, `io/__init__.py:302-304` & `:896`, `raw.py:30` & `:81`. Routing
the second alias to "a separate spec" while deleting its twin in the same file
would be scope minimality outranking the Iron law (identical one-line edits,
same files, stage experimental) — so both are deleted here.
`read_amber_prmtop` / `read_amber_inpcrd` remain the single names. No consumer
outside molrs uses either alias — molpy calls `molrs.io.read_amber_prmtop`,
`molrs.ff.read_amber_prmtop_ff` and `molrs.io.read_amber_inpcrd`
(`molpy/src/molpy/io/forcefield/amber.py:53, 64`, `io/data/amber.py:56`), has zero
`read_inpcrd` references, and its own suite asserts `read_prmtop` is *not*
re-exported on the molpy facade (`molpy/tests/test_io/test_amber_prmtop.py:39-40`).
The **core** twin `read_inpcrd` (`molrs/src/io/data/inpcrd.rs:141`) is a different
layer and stays routed to a core surface-hygiene spec (phase 01 deletes only the
core `read_prmtop` it touches).

**Binder-surface symmetry: compared, not required.** The `.claude/notes/notes.md`
rule *[2026-08-10] 绑定面对称原则* is scoped to the **neighbour** API
(`NeighborList` / `Neighbors`), not to every type. Checked anyway:
`molrs-wasm/src/ff.rs` exposes no `ForceField` class at all (only the typifier
macro's `typify` / `toPotentials`, plus `Potentials` / `LBFGS` / `OptReport`),
and `molrs-capi/src/forcefield.rs` exposes builder + JSON entry points
(`molrs_ff_new` … `molrs_ff_from_json`) with no format reader and no
special-bonds accessor. Neither surface can observe the weights today, so there
is nothing to mirror; architect confirmed 🟢.

**Rejected alternative — a `PySpecialBonds` pyclass.** In-tree precedent exists:
`PyFragmentScaling` (`molrs-python/src/ff/mod.rs:218-232`) is a `frozen`,
`get_all` pyclass over an equally small value struct. It does not apply here.
`FragmentScaling` is a *named domain object* that callers construct and pass
around (`compute_k_ij`, `scale_lj`, `fragment_scaling_data`), so it needs an
identity on the Python surface. The special-bonds weights are two numeric
triples that callers *read into numpy arithmetic* and *write back*; wrapping
them would add a third spelling of the same six numbers (native struct +
pyclass + the `.pyi` type), and would hand out a non-numpy object where every
neighbouring property on this surface hands out `NDArray[np.float64]`. Two
ndarray getters plus one setter keep them numpy-native and typeable.

### Reuse decision

- `reuse` `molrs::ff::ForceField::special_bonds` /
  `set_special_bonds` / `SpecialBonds`
  (`molrs/src/ff/forcefield/mod.rs:669-726`) — the getters and setter are thin
  marshalling over these; no arithmetic, no defaults, no validation is
  re-implemented in the binder.
- `reuse` `ForceField::subset`'s weight copy (`mod.rs:961`) — already correct
  natively; this phase only stops the Python replay from throwing it away.
- `pattern` `PySimBox::{lengths, angles, origin}`
  (`molrs-python/src/core/spatial/simbox.rs:280-305`) — copied verbatim as the
  shape for a length-3 float64 property (`#[getter]`, `Bound<'py,
  PyArray1<NpF>>`, `into_pyarray`); no new spelling invented.
- `reuse` `ForceField._from_raw` (`forcefield.py:533-556`) — extended by one
  forwarding line rather than bypassed; `subset` and all reader wrappers keep
  their single funnel.
- `new — none.` No new type, module, helper or free function is introduced;
  every symbol added is a member of an existing class.

## Files to create or modify

- `molrs-python/src/ff/mod.rs`
- `molrs-python/src/io/mod.rs`
- `molrs-python/src/lib.rs`
- `molrs-python/python/molrs/ff/forcefield.py`
- `molrs-python/python/molrs/io/__init__.py`
- `molrs-python/python/molrs/io/raw.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/tests/test_forcefield_special_bonds.py` (new)
- `molrs-python/tests/test_io.py`
- `regressions/amber-prmtop-complete-04-python.py` (new)

## Tasks

- [ ] Write failing seam tests for the `ForceField` special-bonds accessors in `molrs-python/tests/test_forcefield_special_bonds.py` (shape/dtype, copy semantics, setter round-trip, `ValueError`, stored-only `[0]`/`[1]`, `_from_raw` forwarding)
- [ ] Write a failing surface test in `molrs-python/tests/test_io.py` asserting `read_prmtop` and `read_inpcrd` are absent from `molrs.io`, `molrs.io.__all__`, `molrs.io.raw` and `molrs._lib` while `read_amber_prmtop` / `read_amber_inpcrd` resolve
- [ ] Implement `special_bonds_lj` / `special_bonds_coul` getters and `set_special_bonds` on `PyForceField` in `molrs-python/src/ff/mod.rs`, with doc comments per `rustdoc` style giving the `[1-2, 1-3, 1-4]` order, the dimensionless units, and the stored-but-never-applied semantics of entries `[0]`/`[1]`
- [ ] Forward the special-bonds weights in `ForceField._from_raw` in `molrs-python/python/molrs/ff/forcefield.py`
- [ ] Delete the `read_prmtop` and `read_inpcrd` aliases from `molrs-python/src/io/mod.rs`, their registrations in `molrs-python/src/lib.rs`, and their re-exports in `molrs-python/python/molrs/io/__init__.py` and `molrs-python/python/molrs/io/raw.py`
- [ ] Add the two properties and `set_special_bonds` to `class ForceField` in `molrs-python/python/molrs/_lib.pyi`, delete the `read_prmtop` and `read_inpcrd` stub lines, and refresh the hand-written `class keys` subset (`_lib.pyi:829-845`) to exactly the names `register_keys` exposes at runtime (`molrs-python/src/schema.rs:170-184`: every `SCHEMA_COLUMNS` const_name — `EXCLUDE_14` included once phase 01 has landed — plus the five explicit ordered-group names `COORDS` / `VELOCITIES` / `QUAT` / `DIPOLE` / `ENDPOINTS`, typed `List[str]`), deleting the never-defined `ORDER` / `SYMBOL` entries — routed here from phase 01; a seam test asserts set-equality between the public names of `dir(molrs.keys)` and the stub's declared names. The stub is a hand copy of a generated list, so the refresh re-seeds the drift by construction: generating `class keys` from the table is routed to `/mol:refactor` (named debt, Out of scope)
- [ ] Add regression example `regressions/amber-prmtop-complete-04-python.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Bindings prove the **seam** only (`.claude/notes/testing.md` § *Language
bindings*): symbols import, dtypes at the boundary, non-mutating contracts,
error mapping. Numerical depth for the divisor → weight conversion stays in the
native unit tests owned by phase 02.

**Unit — `molrs-python/tests/test_forcefield_special_bonds.py`** (new; flat
per-concern file matching the existing `test_forcefield_subset.py` /
`test_forcefield_lammps_reader.py` convention). Each test targets a single
member of `PyForceField` or `ForceField._from_raw`:

- *Happy path.* A fresh `molrs.ff.ForceField()` reports `special_bonds_lj ==
  [0.0, 0.0, 1.0]` and `special_bonds_coul == [0.0, 0.0, 1.0]` (molrs default).
- *Dtype and shape.* Both properties return `numpy.ndarray`, `shape == (3,)`,
  `dtype == numpy.float64`.
- *Copy semantics.* Two successive reads return arrays that are not the same
  object; writing into a returned array leaves the next read unchanged (mirrors
  the non-mutating contract asserted for `PySimBox` geometry).
- *Setter round-trip, exact.* `set_special_bonds(lj=[0.0, 0.0, 0.5],
  coul=[0.0, 0.0, 1.0 / 1.2])` reads back `0.5` and the literal
  `0.8333333333333334` with `==` (no tolerance — this is an f64 copy, not a
  computation).
- *Edge — wrong length.* `set_special_bonds(lj=[0.0, 0.5], coul=[0.0, 0.0, 1.0])`
  raises `ValueError`, and the force field's weights are unchanged afterwards.
- *Edge — stored-only entries.* `set_special_bonds(lj=[1.0, 1.0, 0.5],
  coul=[1.0, 1.0, 0.5])` reads back `[1.0, 1.0, 0.5]` verbatim; the test's
  docstring states that molrs kernels never apply `[0]`/`[1]` (1-2 / 1-3 are
  excluded by omission from the neighbour list), so this asserts *storage
  fidelity*, not physics.
- *`_from_raw` forwarding.* `molrs.ff.read_lammps_forcefield_str(<AMBER-shaped
  inline fixture>)` yields a `ForceField` whose weights survive the ergonomic
  re-wrap; and `ff.subset(frame)` on that result preserves them too (the two
  `_from_raw` paths that regressed).

**Unit — `molrs-python/tests/test_io.py`** (existing): one added surface test
asserting, for each of `read_prmtop` and `read_inpcrd`, `not hasattr(molrs.io, name)`,
`name not in molrs.io.__all__`, `not hasattr(molrs.io.raw, name)`, and
`not hasattr(molrs._lib, name)`, while `molrs.io.read_amber_prmtop` and
`molrs.io.read_amber_inpcrd` still resolve.

Unit green for either file is that file alone:
`uv --directory molrs-python run --no-sync pytest tests/test_forcefield_special_bonds.py`
(the Python analogue of `$META.build.test_single`).

**Regression example — `regressions/amber-prmtop-complete-04-python.py`.**
One minimal public-API script under the repo-root `regressions/` tree, run by a
human, composing the chain the library deliberately does not compose for you:

1. `molrs.ff.read_amber_prmtop_ff_str(<inline minimal prmtop>)` — the fixture
   is a hand-written `%FLAG` text carrying `POINTERS`, `AMBER_ATOM_TYPE`,
   `ATOM_TYPE_INDEX`, `MASS`, the LJ `A`/`B` tables, and explicit
   `SCEE_SCALE_FACTOR` `1.2` / `SCNB_SCALE_FACTOR` `2.0`.
2. Assert the getters: `special_bonds_lj[2] == 0.5` and
   `special_bonds_coul[2] == 0.8333333333333334` (literals; phase 02's leg).
3. `molrs.ff.write_lammps_forcefield_str(ff)` — assert the emitted text
   contains the exact line
   `special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333`
   (phase 03's leg; `%.6f` via `fmt_num`,
   `molrs/src/ff/forcefield/writers/lammps.rs:1133-1135`).
4. `molrs.ff.write_gromacs_top_ff_str(ff)` — assert the `[ defaults ]` row
   carries `fudgeLJ` `0.500000` and `fudgeQQ` `0.833333` (phase 03's leg;
   precision 6, `writers/gromacs.rs:21-41`).

Provenance header: hand-written literals derived from the AMBER default
divisors — **no** external oracle, no AmberTools / LAMMPS / GROMACS at run
time (`molrs` + `numpy` only), per CLAUDE.md § *Testing Rules (MANDATORY)*.

## Out of scope

- **Any writer or reader change.** `writers/lammps.rs` and `writers/gromacs.rs`
  have zero `special_bonds` / `fudge` occurrences today; emitting them is
  `amber-prmtop-complete-03-lammps`'s deliverable, and the prmtop →
  `SpecialBonds` conversion is phase 02's. This spec only reads what they
  produce. `depends_on` is binding: the regression cannot pass before 03 merges.
- **Applying the `[0]` / `[1]` weights in any kernel.** molrs excludes 1-2 /
  1-3 by omitting pairs from the neighbour list; changing that is a native MD
  design decision, not a binder one.
- **wasm / capi mirrors.** Compared in Design and not needed: neither surface
  exposes a `ForceField` reader or the weights, and the symmetry rule in
  `.claude/notes/notes.md` is neighbour-API scoped.
- **molpy changes.** See *Follow-up (molpy)*; molrs ships and tags first
  (CLAUDE.md § *Release before molpy*).
- **Named debt, found and deliberately not fixed here (Iron law — reported, routed, not silently left):**
  - `molrs-python/python/molrs/_lib.pyi:1318-1322` — the `ForceField` stub
    declares only `name`, `style_names`, `to_potentials` while `PyForceField`
    exposes the whole `def_*style` / `def_*type` / `subset` / `types` /
    `set_type_param` builder surface. This spec adds three members and does not
    close the gap. Route `/mol:docs` for a full stub pass.
  - `_lib.pyi:829-845` `class keys` — a hand-copied mirror of the list
    `molrs-python/src/schema.rs:170-184` generates at runtime (and it types a
    submodule as a class). This phase refreshes it once (routed from phase 01)
    and adds a seam test so the next drift is caught, but the copy will drift
    again by construction. Route `/mol:refactor`: generate the stub block from
    `SCHEMA_COLUMNS`.
  - `molrs-python/python/molrs/ff/forcefield.py:533-556` — `_from_raw` replays
    "what a ForceField consists of" field-by-field, making it a **second owner**
    of that knowledge; the weights were dropped precisely because a new native
    field did not appear in the replay, and the next native field will regress
    the same way. A live second instance of the same loss: `subset`
    (`forcefield.py:558-559`) discards `self.units` through the same replay and
    the readers at `:847`/`:854` re-stamp it by hand. The single forwarding line
    for the weights is the sanctioned fix *today*; route `/mol:refactor` for an
    `_adopt`-style wrap-the-inner path on `PyForceField` that removes the replay
    (and the `units` re-stamping) entirely.
  - The **core** `read_inpcrd` alias (`molrs/src/io/data/inpcrd.rs:141`,
    `read_inpcrd` → `read_amber_inpcrd`) — a different layer, so out of scope
    for a binder-only phase (its binder mirror *is* deleted here). Route a core
    surface-hygiene spec alongside the phase-01 `read_prmtop` deletion.

## Follow-up (molpy)

Not part of this spec; queued for after molrs ships and tags the minor line.

- `molpy/src/molpy/io/forcefield/amber.py:53-66` (`AmberPrmtopReader`) returns
  `(structure, ff)` from `molrs.io.read_amber_prmtop` +
  `molrs.ff.read_amber_prmtop_ff`. Once this phase lands, that `ff` carries the
  1-4 weights, so molpy's AMBER → LAMMPS path can stop hard-coding
  `special_bonds amber` and read them off the force field instead.
- No molpy migration is needed for the `read_prmtop` deletion: molpy already
  calls only `read_amber_prmtop` / `read_amber_prmtop_ff`, and
  `molpy/tests/test_io/test_amber_prmtop.py:39-40` asserts the name is not on
  the molpy facade.
- Ordering is the iron law: molrs `master` + `vX.Y.Z` tag + publish **before**
  molpy bumps its minor pin.
