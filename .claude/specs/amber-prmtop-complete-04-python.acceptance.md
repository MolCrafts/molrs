---
slug: amber-prmtop-complete-04-python
criteria:
  - id: ac-001
    summary: ForceField exposes length-3 float64 special-bonds properties
    type: code
    pass_when: |
      On a fresh molrs.ff.ForceField(), both special_bonds_lj and
      special_bonds_coul return a numpy.ndarray with shape (3,) and
      dtype numpy.float64 whose value equals [0.0, 0.0, 1.0]
      (molrs default), asserted in
      molrs-python/tests/test_forcefield_special_bonds.py.
    status: pending
  - id: ac-002
    summary: Getters hand out copies, never a view into the force field
    type: code
    pass_when: |
      In molrs-python/tests/test_forcefield_special_bonds.py, two
      successive reads of special_bonds_lj return distinct array objects,
      and writing a new value into a returned array leaves the next read
      of that property unchanged.
    status: pending
  - id: ac-003
    summary: set_special_bonds round-trips exactly and rejects wrong length
    type: code
    pass_when: |
      set_special_bonds(lj=[0.0, 0.0, 0.5], coul=[0.0, 0.0, 1.0 / 1.2])
      reads back exactly 0.5 and 0.8333333333333334 (== , no tolerance);
      set_special_bonds(lj=[0.0, 0.5], coul=[0.0, 0.0, 1.0]) raises
      ValueError and leaves the previously stored weights unchanged.
    status: pending
  - id: ac-004
    summary: 1-2 / 1-3 entries are stored and returned verbatim
    type: code
    pass_when: |
      set_special_bonds(lj=[1.0, 1.0, 0.5], coul=[1.0, 1.0, 0.5]) reads
      back [1.0, 1.0, 0.5] on both properties, in a test whose docstring
      states that molrs kernels never apply entries [0]/[1].
    status: pending
  - id: ac-005
    summary: _from_raw forwards weights on reader and subset paths
    type: code
    pass_when: |
      A ForceField from molrs.ff.read_lammps_forcefield_str(<AMBER-shaped
      inline fixture>) reports the fixture's 1-4 weights (not the
      [0,0,1] default), and ff.subset(frame) on that result reports the
      same weights — both asserted in
      molrs-python/tests/test_forcefield_special_bonds.py.
    status: pending
  - id: ac-006
    summary: Three members documented with units and stored-only semantics
    type: docs
    pass_when: |
      The doc comments of special_bonds_lj, special_bonds_coul and
      set_special_bonds in molrs-python/src/ff/mod.rs each state the
      [1-2, 1-3, 1-4] order, that the weights are dimensionless, and
      that entries [0]/[1] are stored and round-tripped but never
      applied by molrs kernels; class ForceField in
      molrs-python/python/molrs/_lib.pyi declares both properties as
      NDArray[np.float64] plus set_special_bonds.
    status: pending
  - id: ac-007
    summary: read_prmtop and read_inpcrd aliases are gone from the binder surface
    type: code
    pass_when: |
      `rg -n '\b(read_prmtop|read_inpcrd)\b' molrs-python/src molrs-python/python`
      returns no match (the negative assertions in molrs-python/tests/test_io.py
      are the only remaining spellings), and that surface test asserts each name
      is absent from molrs.io, molrs.io.__all__, molrs.io.raw and molrs._lib
      while molrs.io.read_amber_prmtop and molrs.io.read_amber_inpcrd still
      resolve; and class keys in molrs-python/python/molrs/_lib.pyi declares
      exactly the names register_keys (molrs-python/src/schema.rs:170-184)
      exposes at runtime — every SCHEMA_COLUMNS const_name at the commit this
      phase lands on (after phase 01 that includes EXCLUDE_14) plus the five
      explicit ordered-group names COORDS, VELOCITIES, QUAT, DIPOLE, ENDPOINTS
      (typed List[str]) — with the never-defined ORDER and SYMBOL entries
      removed. The single normative check is a seam test asserting that the
      set of public names in dir(molrs.keys) equals the set of names the stub
      declares.
    status: pending
  - id: ac-008
    summary: Regression example reproduces the AMBER 1-4 weights end to end
    type: runtime
    pass_when: |
      `uv --directory molrs-python run python
      ../regressions/amber-prmtop-complete-04-python.py` exits 0, using
      only molrs + numpy at run time. The script reads its inline prmtop
      fixture (SCEE 1.2 / SCNB 2.0) with read_amber_prmtop_ff_str,
      asserts special_bonds_lj[2] == 0.5 and special_bonds_coul[2] ==
      0.8333333333333334, asserts write_lammps_forcefield_str output
      contains the literal line phase 03 emits —
      `special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333`
      — and asserts write_gromacs_top_ff_str output carries fudgeLJ
      0.500000 and fudgeQQ 0.833333 on its [ defaults ] row.
    status: pending
  - id: ac-009
    summary: Full check and test suite green after the change
    type: runtime
    pass_when: |
      `prek run --all-files --hook-stage pre-push` and
      `uv --directory molrs-python sync --no-install-project --extra dev
      && uv --directory molrs-python run --no-sync tox -e py` both pass
      with no new warnings.
    status: pending
---

# Acceptance criteria

**ac-001 – ac-005** are the seam contract for the three new `ForceField`
members and the `_from_raw` repair. They are `code` rather than `scientific`
because nothing here computes physics: the binder copies six `f64`s across the
FFI boundary. The divisor → weight arithmetic is owned and tested natively by
phase 02.

**ac-004** deserves its own criterion because it is the one place the API's
honesty is checkable: the whole-struct setter accepts values molrs will never
apply, and the only defence against a future reader silently relying on them is
a test plus a doc comment that say so.

**ac-006** is `docs`, not `code`, because its failure mode is a caller who
multiplies by `special_bonds_lj[0]` believing molrs already did not.

**ac-007** is `code`: the deletion is observable both by grep and by the import
surface, and both halves must hold — a stub line left behind in `_lib.pyi` is
still a second public name.

**ac-008** is `runtime`, not `scientific`: the example lives in this repo's
`regressions/` tree with hard-coded literals, and its writer legs are assertions
about the *format contract phase 03 landed*, not about a physical observable
measured against an external benchmark. If phase 03 has not merged, this
criterion fails — that is the intended reading of `depends_on`.
