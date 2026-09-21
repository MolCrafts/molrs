
---
spec: cgsmiles-01e-python-ir
created: 2026-09-21
criteria:
  - id: ac-001
    summary: CGSmilesIR is the single text front door, mirroring PySmilesIR
    type: code
    pass_when: |
      molrs-python/src/io/cgsmiles.rs exists, is declared `pub mod cgsmiles;`
      from molrs-python/src/io/mod.rs, and defines
      `#[pyclass(module = "molrs.io", name = "CGSmilesIR")] PyCGSmilesIR`
      with fields `{ inner, input }`, one `#[new] fn new(text: &str)` calling
      `molrs::io::smiles::parse_cgsmiles`, one `to_atomistic`, and `__repr__`;
      `PySmilesIR::from_core` is added in a plain `impl PySmilesIR` block (not
      `#[pymethods]`) in molrs-python/src/io/mod.rs; no `CGSmilesReader` type
      and no `parse_cgsmiles` #[pyfunction] exist anywhere in molrs-python.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "src/io/mod.rs:20 `pub mod cgsmiles;`; cgsmiles.rs:554 PyCGSmilesIR { inner, input } with #[new] (:581, parse_cgsmiles + smiles_error_to_pyerr), to_atomistic (:681), __repr__; PySmilesIR::from_core in a plain impl at src/io/mod.rs:2111; no CGSmilesReader type and no parse_cgsmiles pyfunction anywhere in molrs-python"
  - id: ac-002
    summary: Seven nested read-only pyclasses, one spelling per fact
    type: code
    pass_when: |
      molrs-python/src/io/cgsmiles.rs defines PyCGGraph, PyCGNode, PyCGEdge,
      PyCGFragmentDef, PyResolvedPair, PyPairEnd and PyBondingDescriptor, each
      `#[pyclass(module = "molrs.io", name = ..., frozen, skip_from_py_object)]`
      with `#[getter]`s only — no `#[setter]`, no `#[new]` — and each with a
      `__repr__`; PyCGEdge's getters are exactly i, j, multiplicity and
      derived_from (no `order`, no `origin`), PyCGFragmentDef's are exactly
      name and body (no `body_kind`), and PyCGSmilesIR has no `n_levels`.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "seven nested pyclasses all `frozen, skip_from_py_object` (cgsmiles.rs:87/150/221/277/330/401/469), zero #[setter], the only #[new] is CGSmilesIR's, eight __repr__; CGEdge getters i/j/multiplicity/derived_from only; CGFragmentDef name/body only; no n_levels; the one `fn order` is BondingDescriptor's"
  - id: ac-003
    summary: All eight classes registered, re-exported and declared in the stub
    type: code
    pass_when: |
      molrs-python/src/lib.rs has `m.add_class::<io::cgsmiles::Py*>()?` for all
      eight classes near line 257; python/molrs/io/__init__.py imports all eight
      names from `.._lib` in the block at :35-53 and lists them in `__all__`;
      python/molrs/_lib.pyi declares all eight as top-level classes.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "src/lib.rs eight add_class::<io::cgsmiles::Py*>; io/__init__.py imports the eight names and lists them in __all__; ast over _lib.pyi shows all eight top-level classes"
  - id: ac-004
    summary: Enum mappings are total and the seam cannot panic
    type: code
    pass_when: |
      molrs-python/src/io/cgsmiles.rs maps DescriptorKind and BondKind to
      lowercase variant-name strings, and every match over a predecessor enum
      in that file (DescriptorKind, BondKind, CGBondOrder, EdgeOrigin, PairEnd)
      lists every variant explicitly — no `_ =>` arm, no `unreachable!()` — maps
      CGBondOrder to its integer multiplicity, and contains no `.unwrap()`, no
      `.expect(`, no `panic!`, and no raw `self.input[` slicing (the
      fragment-body text uses `.get(..).unwrap_or(..)`).
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "cgsmiles.rs: zero `_ =>`, unreachable!, .unwrap(), .expect(, panic!, `self.input[`; multiplicity() at :244; two total name fns (9 + 4 arms)"
  - id: ac-005
    summary: F2 graph, fragment and pair values cross correctly
    type: runtime
    pass_when: |
      In molrs-python/tests/test_cgsmiles.py, constructing
      molrs.io.CGSmilesIR("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}")
      gives len(levels) == 1, 5 nodes with names
      ["OH","PEO","PEO","PEO","OH"], 4 edges each with multiplicity == 1 and
      derived_from None, every node charge None / annotations [] / parent None,
      len(fragments) == 1 with keys {"OH","PEO"}, and len(pairs[0]) == 4 with
      every kind == "single" and every src.end and dst.end == "body".
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "src/io/mod.rs:20 `pub mod cgsmiles;`; cgsmiles.rs:554 PyCGSmilesIR { inner, input } with #[new] (:581, parse_cgsmiles + smiles_error_to_pyerr), to_atomistic (:681), __repr__; PySmilesIR::from_core in a plain impl at src/io/mod.rs:2111; no CGSmilesReader type and no parse_cgsmiles pyfunction anywhere in molrs-python"
  - id: ac-006
    summary: F8 multi-level structure, parents, descriptors and provenance
    type: runtime
    pass_when: |
      In molrs-python/tests/test_cgsmiles.py, constructing molrs.io.CGSmilesIR
      on "{[#B1][#B2][#B1]}.{#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.{#PEO=[>]COC[<],#PE=[>]CC[<]}"
      gives len(levels) == 2; levels[1] has 6 nodes with parent values
      [0,0,1,1,2,2] and 5 edges whose first three have derived_from None and
      whose last two have derived_from == (0, 0) and (0, 1); fragments[0]["B1"]
      .body is a molrs.io.CGGraph whose node 0 descriptor reads ("right", "",
      None) and node 1 descriptor kind "left"; fragments[1]["PEO"].body is a
      molrs.io.SmilesIR; pairs[0][0].src.end == "sub" with int index and port.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "seven nested pyclasses all `frozen, skip_from_py_object` (cgsmiles.rs:87/150/221/277/330/401/469), zero #[setter], the only #[new] is CGSmilesIR's, eight __repr__; CGEdge getters i/j/multiplicity/derived_from only; CGFragmentDef name/body only; no n_levels; the one `fn order` is BondingDescriptor's"
  - id: ac-007
    summary: Public-API example reproduces the hard-coded F2 expansion
    type: runtime
    pass_when: |
      test_cgsmiles_f2_public_api in molrs-python/tests/test_cgsmiles.py uses
      only the public path (import molrs; molrs.io.CGSmilesIR(text);
      .to_atomistic()) and passes with the hard-coded literals n_atoms == 11,
      n_relations("bonds") == 10, len(levels) == 1 and len(pairs[0]) == 4; the
      file imports no third-party scientific package and spawns no subprocess.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "src/lib.rs eight add_class::<io::cgsmiles::Py*>; io/__init__.py imports the eight names and lists them in __all__; ast over _lib.pyi shows all eight top-level classes"
  - id: ac-008
    summary: Malformed input raises ValueError through the one existing mapper
    type: runtime
    pass_when: |
      molrs.io.CGSmilesIR("{[#A]") and molrs.io.CGSmilesIR("") each raise
      ValueError with a non-empty message, and molrs-python/src/io/cgsmiles.rs
      contains no error-conversion function of its own and no
      `create_exception!` — every fallible call ends in
      `.map_err(smiles_error_to_pyerr)`.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "cgsmiles.rs: zero `_ =>`, unreachable!, .unwrap(), .expect(, panic!, `self.input[`; multiplicity() at :244; two total name fns (9 + 4 arms)"
  - id: ac-009
    summary: Nested records are read-only, not constructible, single-spelled
    type: runtime
    pass_when: |
      Assigning to any getter of a CGGraph / CGNode / CGEdge / CGFragmentDef /
      ResolvedPair / PairEnd / BondingDescriptor instance raises AttributeError;
      molrs.io.CGNode() raises TypeError; and hasattr is False for
      CGEdge.order, CGEdge.origin, CGFragmentDef.body_kind, CGSmilesIR.n_levels,
      molrs.io.CGSmilesReader and molrs.parse_cgsmiles.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "tests/test_cgsmiles.py F2 graph/fragment/pair tests green under tox -e py (release wheel)"
  - id: ac-010
    summary: Stub parity guard exists, reads the source tree, passes without an allowlist
    type: runtime
    pass_when: |
      molrs-python/tests/test_stub_parity.py exists, reads
      Path(__file__).parents[1] / "python" / "molrs" / "_lib.pyi" with
      ast.parse, asserts set equality of top-level stub class names and public
      `molrs._lib` classes in both directions with the only exemption computed
      by inspect.ismodule, contains no hard-coded list of exempt class names,
      and passes under `tox -e py`.
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "tests/test_cgsmiles.py F8 level/parent/provenance/descriptor/body/pair-end tests green under tox -e py"
  - id: ac-011
    summary: The 17 stale stub declarations are deleted, the shared five kept
    type: docs
    pass_when: |
      python/molrs/_lib.pyi no longer declares Parameters, Type, AtomType,
      BondType, AngleType, DihedralType, ImproperType, PairType, Style,
      AtomStyle, BondStyle, AngleStyle, DihedralStyle, ImproperStyle, PairStyle,
      ChargeModel or Compute; BccModel, MullikenModel and GasteigerModel are
      declared with no base class; the orphaned section header formerly at
      :1333-1339 and the `Protocol` import are gone and no comment names a
      deleted type; Block, Frame, Atomistic, CoarseGrain and ForceField are
      still declared; and `class SmilesIR` declares __init__,
      n_components, to_atomistic, components, write_smiles, write_smarts and
      from_atomistic.
    status: verified
    verified_by: agent-auto
    last_checked: 2026-09-21
    note: "ast over _lib.pyi: none of the 17 stale names declared; Block/Frame/Atomistic/CoarseGrain/ForceField kept; BccModel/MullikenModel/GasteigerModel with no base; Protocol import gone; SmilesIR declares __init__, components, from_atomistic, n_components, to_atomistic, write_smarts, write_smiles; no comment names a deleted type (implementer grep)"
  - id: ac-012
    summary: Dead SmilesIR/parse_smiles names removed from the files this link edits
    type: docs
    pass_when: |
      Neither `molrs.parse_smiles` nor `molrs.SmilesIR` appears anywhere in
      molrs-python/src/io/mod.rs or python/molrs/_lib.pyi (the five sites at
      io/mod.rs:2096,2124,2167,2188 and _lib.pyi:3212 now read
      `molrs.io.SmilesIR`), while molrs-python/src/conformer/mod.rs:239 is
      unchanged and left to 02d.
    status: verified
    verified_by: agent-auto
    last_checked: 2026-09-21
    note: "zero `molrs.parse_smiles`/`molrs.SmilesIR` in src/io/mod.rs and _lib.pyi; src/conformer/mod.rs:239 unchanged (no diff)"
  - id: ac-013
    summary: molrs.io docstring records CGSmilesIR and the no-Reader rule
    type: docs
    pass_when: |
      The module docstring of python/molrs/io/__init__.py names CGSmilesIR and
      states that there is no CGSmilesReader because "Reader" here means a lazy
      path-backed trajectory cursor, with the reader-shaped API left to molpy.
    status: verified
    verified_by: agent-auto
    last_checked: 2026-09-21
    note: "molrs.io module docstring names CGSmilesIR and states there is no CGSmilesReader because Reader means a lazy path-backed trajectory cursor, reader-shaped API left to molpy"
  - id: ac-014
    summary: Full gate green for the binder and the Python suite
    type: runtime
    pass_when: |
      `cargo fmt --manifest-path molrs-python/Cargo.toml --check`, `cargo
      clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D
      warnings` and `uv --directory molrs-python run --no-sync tox -e py` all
      exit 0, with no test skipped or xfailed in
      molrs-python/tests/test_cgsmiles.py or test_stub_parity.py.
    status: verified
    verified_by: agent-auto
    last_checked: 2026-09-21
    note: "ast over _lib.pyi: none of the 17 stale names declared; Block/Frame/Atomistic/CoarseGrain/ForceField kept; BccModel/MullikenModel/GasteigerModel with no base; Protocol import gone; SmilesIR declares __init__, components, from_atomistic, n_components, to_atomistic, write_smarts, write_smiles; no comment names a deleted type (implementer grep)"
out_of_scope:
  - A CGSmilesReader class or any reader-shaped read() API (molpy's job).
  - Typed CGsmiles exceptions via create_exception!; smiles_error_to_pyerr keeps flattening kind/span/input (rot named and routed).
  - Method- and parameter-level stub parity across _lib.pyi.
  - A doctest runner for Python docstrings (pyproject.toml:57).
  - The dead name at molrs-python/src/conformer/mod.rs:239 (02d).
  - Reconciling the names-vs-codes enum convention with core's two-code set_bond_class / bond_type.
  - Spans (CGSmilesIR/CGNode/CGEdge/CGFragmentDef .span) at the Python boundary.
  - WASM and C bindings; Zensical site pages; CGsmiles emission from Python.
  - Any change to the Rust core crate, including the Debug/Clone/PartialEq derives (verified here, never added).
  - The 0.15.0 version bump (cgsmiles-03-release).
---

# Acceptance criteria

"Done" for this link means molpy can read a CGsmiles string through
`molrs.io.CGSmilesIR` and reach every fact the notation carried — levels,
nodes, edges, fragment tables, resolved pairs and both ends of each pair —
plus `to_atomistic()`, without re-parsing a string, decoding an integer or
reading an unnamed tuple position.

`ac-001` through `ac-004` are the shape of the binding: one constructor from
text, one expansion verb, seven read-only nested records in the `LammpsLog`
house style, exactly one public spelling per fact, and mappings that are total
and panic-free. They are checked by reading `molrs-python/src/io/cgsmiles.rs`,
`src/lib.rs`, `python/molrs/io/__init__.py` and `python/molrs/_lib.pyi` — no
build required. `ac-002` and `ac-009` are the same rule from the two sides: a
getter that duplicates another fact (`order` beside `multiplicity`, `origin`
beside `derived_from`, `body_kind` beside a typed `body`, `n_levels` beside
`levels`) fails both.

`ac-005` through `ac-009` are the seam itself, proved by
`molrs-python/tests/test_cgsmiles.py` against the chain's F2 and F8 fixtures
with values hand-derived in 01c and 01d. `ac-007` is this repo's stand-in for a
`regressions/` script: the repo has no such tree (`CLAUDE.md` § Testing Rules),
and nothing executes Python docstrings, so the runnable public-API example is a
named test that uses only `import molrs` and hard-coded literals. No
third-party scientific software runs, and no number here is captured from one.

`ac-010` through `ac-013` are the iron-law repairs this link owes for the
surface it touches: the stub-freshness guard `_lib.pyi` has claimed since it
was written, the 17 stale declarations that guard finds, the `SmilesIR` entry
that had drifted next to where the new classes are declared, and the five
docstring sites advertising Python names that do not exist. `ac-010` and
`ac-011` are the same repair stated as behaviour and as content, so a green
`ac-010` bought with a name allowlist fails `ac-010`'s own wording; `ac-011`
additionally fails if the five names that legitimately exist on both sides
(`Block`, `Frame`, `Atomistic`, `CoarseGrain`, `ForceField`) are removed.

`ac-014` is the gate, invoked per manifest because the binder is its own
workspace. A skipped or xfailed test in either new file fails it, as does any
clippy warning.
```

Changes applied, with the two facts I re-verified in the tree:

- **`ChargeModel` is load-bearing in the stub** — `_lib.pyi:1880` declares it `class ChargeModel(Protocol)` and it is the declared base of `BccModel` (`:1886`), `MullikenModel` (`:1902`) and `GasteigerModel` (`:1909`). Deleting the entry without dropping those three bases would leave three dangling names, so the task and `ac-011` name both edits together.
- **`tox -e py` installs a non-editable wheel and asserts it** — `pyproject.toml:96-107` builds with maturin, force-installs the wheel, then runs `assert 'site-packages' in str(pathlib.Path(molrs.__file__).resolve())`. A parity test resolving the stub from `molrs.__file__` would therefore check the wheel's copy, not the file a contributor edits; the spec pins `Path(__file__).parents[1] / "python" / "molrs" / "_lib.pyi"`.

Also folded in: `multiplicity`-only on `CGEdge` (rule recorded as "an enum that *is* a count crosses as the count"), the corrected two-code reason (`up`/`down`/`any`/`ring` have no code in `BondType` **or** `BondNumber` — quadruple is codable as a pair), `PairEnd` promoted to the eighth frozen pyclass, `body_kind`/`origin` dropped under the one sum-encoding rule, `Clone` reduced to verify-don't-add, `cargo fmt --manifest-path molrs-python/Cargo.toml --check`, `bond_type` at `molgraph.rs:829`, the shim block `:35-53` in both documents, the corrected `n_components` rationale, `from_core` pinned to a plain `impl`, and `.unwrap()`/`.expect(`/raw slicing added to `ac-004`.
