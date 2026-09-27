---
title: "trace-assembly-07: the script's molrs Python surface, seam test, performance and the gate"
slug: trace-assembly-07-python
status: code-complete
created: 2026-09-27
chain: trace-assembly (01-frame → 02-core → 03-perceive → 04-builder → 05-ff → 07-python → 08-molpy (06-io dropped 2026-09-27; routed /mol:fix))
depends_on: [trace-assembly-01-frame, trace-assembly-02-core, trace-assembly-03-perceive, trace-assembly-04-builder, trace-assembly-05-ff]
---

# trace-assembly-07: the script's molrs Python surface, seam test, performance and the gate

## Summary

This link binds exactly the remaining names the binding script (`/home/jicli594/work/backmap_pe_pma/backmap.py`) uses; the Block tuple-key write was link 01. The names:

- `molrs.perceive.Coarsener(cg).coarsen(groups, names)`
- `molrs.perceive.Perceive(sites).linear_paths()`
- `molrs.Trace(points)`
- `CoarseGrain.positions(handles)` / `CoarseGrain.bead_types(handles)`
- `molrs.builder.Assembler(lib, molrs.builder.TracePlacer()).assemble(traces, seqs)`, with `lib` values `Fragment` or `Atomistic`
- `molrs.ff.typifier.ElementTypifier().typify(atomistic)`

Everything is constructed through `__init__`, with no classmethods (ruling (b)) and no free functions (ruling (a)). The link adds stubs, rewrites the operator seam test to the script's shape, measures assembly once at the driver's scale in release, and runs the full gate once.

## Design

**Constitution.** molrs has no `law.md`. The rules are `CLAUDE.md` § Testing Rules (bindings smoke the seam), `.claude/notes/testing.md`, `architecture-rules.md` § Binder rules, and rulings (a)–(d).

**Bindings.**

- **`molrs.Trace(points)`** is a frozen pyclass in `molrs-python/src/core/spatial/trace.rs` (new).
  - `#[new]` takes a float64 `(k, 3)` array; `(0, 3)` is empty; any other second dimension raises `ValueError`.
  - `points` returns a `(k, 3)` float64 copy; `__len__` gives k.
  - There is no Python `from_points`.
  - It is registered in `src/core/spatial/mod.rs` and `src/lib.rs` and exported from `python/molrs/__init__.py`.
- **`CoarseGrain.positions(beads)` / `bead_types(beads)`** live in `src/core/system/molgraph.rs`. A stale handle raises `ValueError`.
- **`molrs.perceive.Coarsener(source)`** lives in `src/perceive.rs`. `source` is a `CoarseGrain` or `Atomistic`, else `TypeError`.
  - It holds the graph object and borrows its core at call time.
  - `coarsen(groups, names) -> CoarseGrain` releases the GIL. `CoarsenError` becomes `ValueError` with integer handles.
  - It lives beside `SubgraphMatcher`; molpy re-exports it at its root.
- **`molrs.perceive.Perceive(graph=None)`.** `PyPerceive::new()` gains an optional positional `graph: CoarseGrain`, which it holds.
  - `linear_paths() -> list[list[int]]` reads the held graph with the GIL released. `LinearPathError` becomes `ValueError`.
  - `Perceive().linear_paths()` raises `TypeError`. The `find_*` methods are unchanged.
  - The asymmetry, its removal condition and its owner are recorded in the link-04 notes entry.
- **`molrs.builder.TracePlacer()`** has no arguments. **`molrs.builder.Assembler(library, placer)`**, in `src/builder.rs`:
  - `library: Mapping[str, Fragment | Atomistic]`, copied at construction. An `Atomistic` becomes a portless Fragment through `Fragment::try_from_molgraph(a.clone().into_inner())`.
  - `placer` must be a `TracePlacer`, else `TypeError`.
  - `assemble(traces, names) -> Fragment` releases the GIL and returns the public `molrs.Fragment` through `PyFragment::from_core`.
  - `AssembleError` is rendered as `ValueError` naming the trace, unit and name. The inner link message comes from `link_error_to_pyerr`'s body, extracted to `pub(crate) fn link_error_message` (its second use).
- **`molrs.ff.typifier.ElementTypifier()`** lives in `src/ff/mod.rs`: `#[pyclass(module = "molrs.ff.typifier", extends = PyTypifier)]` with `PyTypifier::native(ElementTypifier::new())`, following `PyOPLSAATypifier` (`:1189-1212`). It is exported **only** from `python/molrs/ff/typifier.py`.
- **Stubs.** `python/molrs/_lib.pyi` gets every new name, with numpydoc docstrings.
- **Seam test.** `tests/test_backmap_seam.py` is rewritten to the script's steps 1 (tuple-key write only) and 3–6 on hand-built fixtures (operator exception, notes.md:1088). `testing.md` and the notes entry follow.
- **Performance.** Measured once with `maturin develop --release` and a scratchpad script that is not committed. `Assembler.assemble` on the driver class (100 × 100 units of a hand-built 33-atom `<`/`>` monomer, 10,000 1-atom Li and 450,000 13-atom portless PC singles on a 5 Å grid) must take < 60 s, with `n_atoms == 6,170,200`. Linearity: t(10k)/t(5k) ≤ 2.5 on chain-only inputs.
- **Gate.** Run once: `cargo fmt --check`, `cargo mrs-clippy -- -D warnings`, `cargo mrs-test`, `cargo mrs-doctest`, `tox -e py` and `prek run --all-files --hook-stage pre-push`. The unrelated uncommitted change in `molrs/src/io/data/xyz.rs` (+38/−1) stays out of this chain's commits (review 🟢).

### Reuse decision

- `generalize` the b6418561 `PyTrace`: only `__init__`, `points` and `__len__`.
- `generalize PyPerceive` (`src/perceive.rs:76-90`) with the optional held graph and `linear_paths`.
- `pattern PySubgraphMatcher` (`:310-352`) for `Coarsener`, and `PyOPLSAATypifier` for `ElementTypifier`.
- `generalize link_error_to_pyerr` (`molgraph.rs:148`) into a message function.
- `reuse PyFragment::from_core` (`molgraph.rs:2230`).
- `new` — only the pyclass wrappers.

## Files to create or modify

- `molrs-python/src/core/spatial/trace.rs` (new)
- `molrs-python/src/core/spatial/mod.rs`
- `molrs-python/src/core/system/molgraph.rs`
- `molrs-python/src/perceive.rs`
- `molrs-python/src/builder.rs`
- `molrs-python/src/ff/mod.rs`
- `molrs-python/src/lib.rs`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-python/python/molrs/__init__.py`
- `molrs-python/python/molrs/perceive.py`
- `molrs-python/python/molrs/builder.py`
- `molrs-python/python/molrs/ff/typifier.py`
- `molrs-python/tests/test_trace.py` (new)
- `molrs-python/tests/test_coarsegrain.py` (created in link 01; extended)
- `molrs-python/tests/test_coarsener.py` (new)
- `molrs-python/tests/test_linear_paths.py` (new)
- `molrs-python/tests/test_assembler.py` (new)
- `molrs-python/tests/test_element_typifier.py` (new)
- `molrs-python/tests/test_backmap_seam.py`
- `molrs-python/tests/test_stub_parity.py` (run only)
- `.claude/notes/testing.md`
- `.claude/notes/notes.md`

## Tasks

- [x] Write failing seam tests `molrs-python/tests/test_trace.py` (new) and the accessor cases in `molrs-python/tests/test_coarsegrain.py`
- [x] Bind `molrs.Trace(points)` in `molrs-python/src/core/spatial/trace.rs` (new), registered in `molrs-python/src/core/spatial/mod.rs` and `molrs-python/src/lib.rs` and exported from `molrs-python/python/molrs/__init__.py`, and `CoarseGrain.positions` / `bead_types` in `molrs-python/src/core/system/molgraph.rs`; loop `maturin develop` + those two pytest files
- [x] Write failing seam tests `molrs-python/tests/test_coarsener.py` (new) and `molrs-python/tests/test_linear_paths.py` (new)
- [x] Bind `Coarsener` and `Perceive(graph=None).linear_paths()` in `molrs-python/src/perceive.rs` (module doc amended), registered in `molrs-python/src/lib.rs` and exported from `molrs-python/python/molrs/perceive.py`; loop `maturin develop` + those two pytest files
- [x] Write failing seam tests `molrs-python/tests/test_assembler.py` (new) and `molrs-python/tests/test_element_typifier.py` (new)
- [x] Bind `TracePlacer` and `Assembler` (library `Fragment | Atomistic`) in `molrs-python/src/builder.rs`, extracting `link_error_message` in `molrs-python/src/core/system/molgraph.rs`, and `ElementTypifier` in `molrs-python/src/ff/mod.rs`; register them in `molrs-python/src/lib.rs` and export from `molrs-python/python/molrs/builder.py` and `molrs-python/python/molrs/ff/typifier.py`; loop `maturin develop` + those two pytest files
- [x] Add stubs with numpydoc docstrings for every new name to `molrs-python/python/molrs/_lib.pyi` and run `molrs-python/tests/test_stub_parity.py`
- [x] Rewrite `molrs-python/tests/test_backmap_seam.py` to the script's shape, and update `.claude/notes/testing.md:58-60` and the notes.md:1088 seam-exception entry
- [x] Measure performance once with a release build (`maturin develop --release`, scratchpad script, not committed): `assemble` < 60 s with `n_atoms == 6,170,200`, and t(10k)/t(5k) ≤ 2.5; record the results in the 2026-09-27 entry of `.claude/notes/notes.md`
- [x] Run the full gate once: `cargo fmt --check`, `cargo mrs-clippy -- -D warnings`, `cargo mrs-test`, `cargo mrs-doctest`, `uv --directory molrs-python run --no-sync tox -e py`, `prek run --all-files --hook-stage pre-push` (with `molrs/src/io/data/xyz.rs` kept out of the chain's commits)

## Testing strategy

Seam tests only. Loop: `maturin develop` followed by the named pytest files.

- **`test_trace.py`.**
  - `Trace(2×3 float64)` → len 2, and `points` equals the input with dtype float64.
  - `(0, 3)` gives len 0; `(2, 2)` raises `ValueError`.
  - The object is frozen, and there is no `from_points`.
- **`test_coarsegrain.py`** (additions). `positions([h2, h0])` is float64 `(2, 3)` in order; `bead_types` returns `list[str]`; a stale handle raises `ValueError`.
- **`test_coarsener.py`.**
  - A `molrs.CoarseGrain` with 2 beads, 1 bond and types `["A","B"]`.
  - An `Atomistic` source is accepted; a `Frame` source raises `TypeError`.
  - Overlap raises `ValueError` with an integer handle.
- **`test_linear_paths.py`.**
  - `Perceive(cg).linear_paths() == [[a,b,c],[d]]`.
  - A star raises `ValueError` naming the centre.
  - `Perceive().linear_paths()` raises `TypeError`; `Perceive().find_rings(mol)` still works.
- **`test_assembler.py`.**
  - `{"M": Fragment, "Li": Atomistic}` over `[[M,M],[Li]]` returns `type(...) is molrs.Fragment` with `n_atoms == 7`, `n_ports == 2`, and `to_frame()["atoms"]["mol_id"]` in {1, 2} (Li = 2).
  - An unknown name raises `ValueError`; a non-`TracePlacer` placer raises `TypeError`.
  - `TracePlacer()` takes no arguments.
- **`test_element_typifier.py`.**
  - `ElementTypifier().typify(water).to_frame()`: atoms `type` `["O","H","H"]`, bonds `type` `["H-O","H-O"]`.
  - It is reachable at `molrs.ff.typifier.ElementTypifier` and **not** at `molrs.ff.ElementTypifier`.
- **`test_backmap_seam.py`** (operator exception).
  - *Fixture.* The 8-bead `4 1 1 1 1 1 1 4` chain (mass 72) plus one type-`"2"` bead at x = 40 Å (mass 7), with no `mol_id`. Coordinates are first set through `atoms["x","y","z"] = arr` (the link-01 write).
  - *Body.* `SubgraphMatcher.find` per rule → `Coarsener(cg).coarsen` → `Perceive(sites).linear_paths()` → `Trace(sites.positions(p))` / `sites.bead_types(p)` → `Assembler(lib, TracePlacer()).assemble` → `ElementTypifier().typify(world.to_atomistic())`.
  - *Hand counts.* 3 groups; 3 sites; sorted path lengths `[1, 2]`; `n_atoms == 7`; `n_ports == 2`; `mol_id` values `{1, 2}`; atom types within `{"C","H","Li"}`.

## Out of scope

- **A Python `link_many`, a `Placer` base, free `coarsen` / `linear_paths`, or `Trace.from_points`** (rulings (a), (b), (d)).
- **Resolving the Perceive asymmetry.** Recorded with its removal condition and owner.
- **The molpy re-exports and the real-data run** (link 08).
- **Method-level stub parity** (notes.md:1120) and **the dual typifier exports** (routed in link 05) stay open.
