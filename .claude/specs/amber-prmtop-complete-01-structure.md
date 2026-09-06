---
slug: amber-prmtop-complete-01-structure
title: amber-prmtop-complete-01-structure — the prmtop structure reader reads every structure section, or refuses out loud
status: code-complete
created: 2026-09-04
---

# amber-prmtop-complete-01-structure — the prmtop structure reader reads every structure section, or refuses out loud

## Summary

`molrs/src/io/data/prmtop.rs` today reads eight of the ~thirty sections an AMBER
prmtop actually carries, and silently drops the rest: residue labels, molecule
membership, the exclusion list, the GB radii/screen pair, the tree chain, the
periodic box, and the sign bit that says a torsion's 1-4 term is suppressed all
reach the reader and are thrown away. Worse, three of those omissions are not
merely missing data — they are *wrong* data downstream: a PME run built from a
molrs prmtop Frame has no `exclusions` block, so the real-space correction it
already knows how to apply (`ff/potential/kspace/pme.rs:901`) never fires. This
phase makes the structure reader total over the structure half of the format:
every FileFormats section is either decoded into the Frame, length-checked and
deliberately discarded, or refused with a named error. Nothing is inferred, and
nothing is guessed — a column that the file does not carry is absent, not
fabricated. Phase 01 stays inside `molrs/src/io/` plus one declarative row in the
schema vocabulary; phases 02–04 carry the force-field tables, the writer, and the
Python surface.

## Domain basis

Format reference: Amber **PARM/prmtop** specification,
<https://ambermd.org/FileFormats.php> § *"PARM" (topology) files* (accessed
2026-09-04). Amber internal units: <https://ambermd.org/Questions/units.html>
(Length Å, Energy kcal/mol, Mass amu, Charge as described below).

- **Charge.** `CHARGE` holds `q · 18.2223`, where `18.2223 ≈ √332.0522` folds
  Amber's Coulomb constant into the charge so that `E = qᵢqⱼ/r` needs no prefactor.
  The reader divides by the literal `18.2223` (`CHARGE_CONVERSION_FACTOR`,
  `prmtop.rs:48`) to reach electron units. **Do not re-derive this factor** from
  molrs' own `C ≈ 332.0637133` (`.claude/notes/science.md`): `18.2223² = 332.05222`,
  a **relative offset of 3.46e-5**. Amber's literal is the format's contract; the
  discrepancy is Amber's, and reproducing it bit-for-bit is what makes a round trip
  through AmberTools exact.
- **Residues.** `RESIDUE_LABEL` is `NRES` Fortran `20a4` fields; `RESIDUE_POINTER`
  is `NRES` **1-based** first-atom indices. Atom `a` belongs to residue `r` iff
  `RESIDUE_POINTER[r] ≤ a+1 < RESIDUE_POINTER[r+1]` (sentinel `NATOM+1`). `res_id`
  is already derived this way (`prmtop.rs:389-419`); `res_name` is
  `RESIDUE_LABEL[res_id]`.
- **Molecules.** `ATOMS_PER_MOLECULE` is `NSPM` counts summing to `NATOM`, written
  only when `IFBOX > 0`. `mol_id` is 1-based. When the section is absent the column
  is **absent** — molecule membership is never re-derived from bond connectivity,
  because a solvated system's molecule partition is a fact of the file, not of the
  graph.
- **Exclusions.** `NUMBER_EXCLUDED_ATOMS` is `NATOM` counts; `EXCLUDED_ATOMS_LIST`
  is a flat run of `NNB` **1-based** partner numbers, all greater than the owning
  atom, with a single `0` standing for "this atom excludes nothing". The list is the
  Ewald real-space correction set (Essmann et al., *J. Chem. Phys.* **103**, 8577
  (1995), DOI [10.1063/1.470117](https://doi.org/10.1063/1.470117)), consumed in
  molrs by `pme_ctor` at `ff/potential/kspace/pme.rs:901`.
- **Dihedral pointers.** `DIHEDRALS_{INC,WITHOUT}_HYDROGEN` are 5-tuples of
  coordinate-array indices: atom number = `|N|/3 + 1`. Pointer `0` therefore denotes
  atom 1, which is why the sign — not the value — carries the flags. A **negative
  3rd pointer** means the end-group (1-4) non-bonded term of this torsion is *not*
  computed; Amber sets it on all but one term of a multi-term torsion and on
  ring-closing torsions, so counting 1-4 pairs without it double-counts. A negative
  4th pointer marks an improper (already read, `prmtop.rs:353`). The 3rd-pointer sign
  is currently destroyed by `unsigned_abs()` at `prmtop.rs:356`.
- **Box.** `BOX_DIMENSIONS` is `(OLDBETA, bx, by, bz)`. `IFBOX = 1` is rectangular.
  `IFBOX = 2` is the truncated octahedron, whose unit cell is triclinic with all
  three angles `arccos(−1/3) = 109.4712206344907°`; the cell is built from that exact
  value, **not** from `OLDBETA` (which Amber writes truncated to `1.09471219E+02` and
  its own readers ignore). `IFBOX = 3` is refused. Lengths are Å.
- **GB metadata.** `RADII` / `SCREEN` are the per-atom Born radius and screening
  parameter of the generalized-Born models (Onufriev, Bashford & Case, *Proteins*
  **55**, 383 (2004), DOI [10.1002/prot.20033](https://doi.org/10.1002/prot.20033));
  `RADIUS_SET` names the set as free text (e.g. `modified Bondi radii (mbondi2)`).
  `RADII` is Å; `SCREEN` is dimensionless.
- Force fields whose prmtops this reader must survive: GAFF (Wang et al., *J. Comput.
  Chem.* **25**, 1157 (2004), DOI
  [10.1002/jcc.20035](https://doi.org/10.1002/jcc.20035)) and ff14SB (Maier et al.,
  *J. Chem. Theory Comput.* **11**, 3696 (2015), DOI
  [10.1021/acs.jctc.5b00255](https://doi.org/10.1021/acs.jctc.5b00255)).

## Design

**One entry point, one file.** `read_amber_prmtop_from_reader` →
`parse_flag_sections` → `build_frame` stays the whole story; every addition below
is a decode step inside `build_frame` or a private row-decoder beside
`decode_bonds` / `decode_angles` / `decode_dihedrals`. No new public type, no new
public function, no options struct. The reader is a primitive that does one named
thing (`CLAUDE.md` § *Prefer* / § *Shape check*), and composition — prmtop + inpcrd
+ force field — remains the caller's job.

**New per-atom columns** on the `"atoms"` block:

| column | dtype | source | absent when |
|---|---|---|---|
| `res_name` (canonical, `keys::RES_NAME`) | String | `RESIDUE_LABEL[res_id]` | `RESIDUE_LABEL` missing |
| `mol_id` (canonical, `keys::MOL_ID`, 1-based) | UInt | `ATOMS_PER_MOLECULE` | section missing — **never** inferred from bonds |
| `tree` (format-local) | String | `TREE_CHAIN_CLASSIFICATION` | section missing |
| `gb_radius` (format-local) | Float, Å | `RADII` | section missing |
| `gb_screen` (format-local) | Float, — | `SCREEN` | section missing |

`tree`, `gb_radius` and `gb_screen` are deliberately **unregistered** columns. That
is legal and named as such by the schema's own doc (`schema/mod.rs:26-29`: "A column
key with no spec is unconstrained — the extension point for … format-local
columns"), it has live precedent (`cif.rs:450,456,462` — `chain_id`, `occupancy`,
`b_iso`), and it is the right call because these three have no cross-format
consumer: no other reader in `io/data/` produces a Born radius or an Amber tree
chain, and no potential reads them. The reader's rustdoc must say this explicitly —
in the words molpy's canonical-fields gate uses, "a debt, not a licence" — so the
omission reads as a decision, not an oversight. Downstream, that gate
(`molpy/tests/test_io/test_canonical_fields.py:48-52`, empty prmtop allow-set) will
go red on these three names; that is a **named follow-up** (see *Follow-up (molpy)*),
not a surprise. `res_name` and `mol_id` are
the opposite case — already canonical, already produced by other readers — and are
written through the `molrs::store::keys` constants.

**New `"exclusions"` block**, built against the *existing* `BlockSpec` at
`schema/mod.rs:384-395` (`atomi`/`atomj`, UInt, 0-based, `open: true`). Rows are
emitted in atom order with `atomi < atomj`; the `0` placeholders are dropped rather
than encoded as a self-pair; the flattened list length is cross-checked against
`NNB` and a mismatch is an `InvalidData` refusal. The consumer already exists and
already reads exactly these two column names (`pme.rs:901-902`), so this block is
`reuse`, not a new contract.

**`exclude_14` — one builder, one column set.** `DihedralRow` gains
`exclude_14: bool`, set from `chunk[2] < 0` **before** index canonicalisation,
exactly beside the existing `is_improper = chunk[3] < 0` (`prmtop.rs:353`); `k`
keeps using `chunk[2].unsigned_abs()`. The column is produced by the single
`build_dihedral_block` (`prmtop.rs:474`) and therefore appears in **both** the
`"dihedrals"` block and the mirrored `"impropers"` block with **identical values for
matching rows** — there is one builder and one column set, so the flag cannot
diverge between the two homes. A unit test asserts that parity directly.

That mirroring is itself pre-existing rot and is named here rather than ignored
(`CLAUDE.md` § *Iron law*): `prmtop.rs:662-673` writes improper rows into two blocks,
so one row set has two homes. It is not fixed in this phase because collapsing it
changes the Frame contract that `meta["n_dihedrals"] = NPHIH + MPHIA` and the
LAMMPS-style `"impropers"` consumers both depend on — a behaviour change needing its
own regression, not a rider on a reader-completeness spec.
**Debt route: `/mol:fix prmtop-improper-two-homes` — one home for improper rows.**
Timing: this phase widens the mirror by one column and phase 02 does not touch the
structure blocks, so the fix is filed **when this phase closes**, before the chain
moves on — not deferred past 04.

**`exclude_14` is `DType::Bool`, and it is registered.** The sibling flag `is_14` is
`Bool` (`schema/mod.rs:175-182`); a second flag encoded as UInt 0/1 would put two
dtype conventions on the same concept. Registering it makes the dtype key-globally
enforced by `Block::insert` rather than by this reader's discipline. The row goes
into `SCHEMA_COLUMNS` **between `"element"` and `"id"`** (sorted order, asserted by
`columns_sorted_unique_and_documented`, `schema/mod.rs:529`), spelled `DType::Bool`
like `is_14` rather than via the `use DType::{…}` list at `schema/mod.rs:87`:

```
col!("exclude_14", "EXCLUDE_14", DType::Bool, Scalar, "",
     "Whether this torsion's 1-4 non-bonded term is suppressed (AMBER negative 3rd pointer)")
```

plus `consts::EXCLUDE_14`, its entry in the `consts_agree_with_the_table` list
(`schema/mod.rs:610-650`), and `"exclude_14"` appended to the `optional` lists of the
`dihedrals` and `impropers` `BlockSpec`s (both `open: true`, so this is the
documentation-and-completeness gate `BlockSpec::optional` is defined to be,
`schema/block.rs:50`). **This one-row registration is judged in-scope for this
io-phase**: single crate (`arch.style: crate-graph` — `io` already depends on
`core::store::schema` legally, no new module edge), a declaration table rather than
core logic (the enforcement code `Block::insert` is untouched; `schema/mod.rs:26-29`
names format-local columns as the extension point registration promotes), and no
binder *code* consequence (`molrs-python/src/schema.rs:170-183` projects the table by
loop). One caveat, stated rather than hidden: `molrs-python/python/molrs/_lib.pyi:829-845`
carries a **hand-written, already-stale** `class keys` subset (it lacks `RES_NAME`,
`IS_14`, `VX`, …) — a third partial home for the vocabulary. It is not a projection,
so this row cannot break it, but the drift is real and is **routed to phase 04**
(`amber-prmtop-complete-04-python`, the binder phase) as a named stub-refresh task.

**Box.** `IFBOX = 1` → `SimBox::ortho(lengths, origin=0, [true;3])`, mirroring
`io/data/inpcrd.rs:257-261`. `IFBOX = 2` → `SimBox::new(SimBox::matrix_from_lengths_angles(lengths, [109.4712206; 3])?, origin=0, [true;3])`
(the constructor takes **degrees**, `simbox.rs:122-124`). `IFBOX = 3` → refusal.
`IFBOX = 0` / no `BOX_DIMENSIONS` → `frame.simbox` stays `None`. `OLDBETA` is
recorded verbatim as `meta["oldbeta"]` for provenance and is never used to build the
cell. The rustdoc states the precedence rule for callers who read both files: **an
inpcrd box wins over a prmtop box**, because the prmtop box is the topology-time
default and the inpcrd box is the current state. The two readers stay independent —
precedence is documented, not coded, since neither reader may call the other.

**Flat meta scalars** (house pattern — `frame.meta` is already a flat map of
POINTERS fields, `prmtop.rs:654-656`; `MetaValue::F64`/`I64`/`String` all exist):
`radius_set` (String), `solvent_iptres` / `solvent_nspm` / `solvent_nspsol` (i64,
from `SOLVENT_POINTERS`), `oldbeta` (f64). None of these has an in-tree consumer
today, exactly like the POINTERS fields already in `meta`: they are **file provenance**
the reader is asked to carry so that a prmtop round trip (a writer is not part of this
chain) and a human inspecting the Frame see what the file said. They are kept in
`meta`, not columns, because they describe the file, not an atom; the alternative —
dropping `SOLVENT_POINTERS`/`OLDBETA` on the floor — is the silent-drop this spec
exists to end.

**Length-checked and discarded:** `JOIN_ARRAY` and `IROTAT` must each carry `NATOM`
entries (Amber writes them, Amber ignores them). A wrong length is a refusal; a
right length produces no column. Checking without keeping is the honest reading of a
field the format declares dead.

**Loud refusals**, all `ErrorKind::InvalidData` via the existing `invalid_data`
helper, all phrased `"… are not supported"`:

| trigger | message subject |
|---|---|
| `CMAP_COUNT > 0` | CMAP terms |
| `IFPERT > 0` | perturbed (IFPERT>0) prmtop files |
| `IFCAP > 0` | solvent-cap (IFCAP>0) prmtop files |
| `IPOL = 1` | polarizable (IPOL=1) prmtop files |
| `CTITLE` or `FORCE_FIELD_TYPE` present | CHAMBER prmtop files |
| any `NONBONDED_PARM_INDEX` entry `< 0` | 10-12 hydrogen-bond prmtop files |
| `LENNARD_JONES_CCOEF` present | 12-6-4 Lennard-Jones prmtop files |
| `IFBOX = 3` | IFBOX=3 prmtop files |
| any read section length ≠ its POINTERS count | (existing per-section message form) |

A refusal is the point: today these files parse into a Frame that is quietly missing
half its physics.

**Public-surface deletions.** `read_prmtop` (`prmtop.rs:687-690`) is a pure alias of
`read_amber_prmtop` with **zero in-tree Rust callers** (verified: the only other hits
are the Python binder and its stubs). `architecture-rules.md` § *Naming* — "No dual
public names for the same symbol (0.12: delete façades, not deprecate)" — makes this
an Iron-law delete, not a deprecation. It is deleted in this spec.
The Python mirror is **binder layer and is routed, not fixed here**:
`molrs-python/src/io/mod.rs:1062-1066`, registered `molrs-python/src/lib.rs:204`,
stub `molrs-python/python/molrs/_lib.pyi:888`, re-exported
`python/molrs/io/__init__.py:337` + `:911` and `python/molrs/io/raw.py:32` + `:83`
→ **named task in `amber-prmtop-complete-04-python`**. Checked as instructed:
`molpy/tests/test_io/test_canonical_fields.py:25` defines `_read_prmtop` but it calls
`mp.io.read_amber(...)`, **not** `molrs.io.read_prmtop` — so no molpy test pins the
alias and phase 04 can delete it cleanly.

**Documentation truth.** `ff/potential/kspace/pme.rs` claims the exclusions columns
are named `"i"` / `"j"` at **both** `:10` and `:857`, while the code at `:901-902`
reads `atomi` / `atomj`. Phase 01 owns **both** lines — `amber-prmtop-complete-02`
must drop `:857` from its scope. This spec is the first thing to make that block
actually appear in a real Frame, so shipping it while the consumer's doc lies about
its schema is exactly the silent debt the Iron law forbids.

**Named debt, routed, not fixed here:**

- The `%FLAG` lexer is duplicated between `io/data/prmtop.rs:128-172` and
  `ff/forcefield/readers/prmtop.rs:~88-110`. The architect **CONFIRMED there is no
  legal shared home**: `architecture-rules.md` § *Module dependency rules* lets both
  `io` and `ff` depend on `core` + `perceive` but forbids `ff → io`, and a text lexer
  for one file format does not belong in `core`. Kept as a **Debt: note** entry, not
  a task.
- `gro.rs:433` and `:524` emit/read `"resname"` where the schema says `res_name`
  (`schema/mod.rs:264-271`); `xyz.rs:1469` carries the same spelling in a test
  fixture. Introducing `res_name` from prmtop makes the divergence load-bearing.
  **Route: `/mol:fix resname-schema-key` — `gro.rs:433,524` + `xyz.rs:1469`.**

### Reuse decision

- **`reuse` `io/data/prmtop_tables::parse_a4_names`** (`prmtop_tables.rs:56-67`) —
  byte-identical to the local `read_a4_names` (`prmtop.rs:106-117`). This phase adds
  two more `20a4` sections (`RESIDUE_LABEL`, `TREE_CHAIN_CLASSIFICATION`), which is
  the second real use that ends the "inline until the second use" rule
  (`CLAUDE.md` § *Prefer*). **The local copy is deleted** and all four call sites
  (`ATOM_NAME`, `AMBER_ATOM_TYPE`, and the two new ones) go through
  `prmtop_tables::parse_a4_names`. The existing `a4_names_chunking` unit test retargets.
- **`reuse` the `exclusions` `BlockSpec`** (`schema/mod.rs:384-395`) and its consumer
  `pme_ctor` (`pme.rs:901-902`) — the new block is written to that exact contract.
  No new block name, no new column names.
- **`reuse` `SimBox::ortho` / `SimBox::matrix_from_lengths_angles` + `SimBox::new`**
  (`simbox.rs:106`, `:122`, `:60`) — no new box construction path; the ortho call
  mirrors `inpcrd.rs:257-261` verbatim.
- **`reuse` `is_14`'s dtype convention** (`schema/mod.rs:175-182`) — `exclude_14` is
  the same kind of thing and gets the same `DType::Bool`, same `Scalar`, same empty unit.
- **`reuse` `prmtop.rs`'s own `invalid_data` / `parse_tokens` / `section_ints` /
  `insert_*_col` family** (`prmtop.rs:54-100`, `:503-508`) — every new decode step is
  written in these terms, so the new code reads like the existing code.
- **`new` — a private `insert_bool_col` in `prmtop.rs`.** `poscar.rs:167` has a
  module-private twin, but the `insert_*_col` family is deliberately per-module across
  `io/data/` (`prmtop.rs`, `inpcrd.rs`, `poscar.rs` each own one); adding the fourth
  member of `prmtop.rs`'s own family is the local pattern, whereas hoisting a shared
  helper would be a new abstraction in `io/data/` that nothing else asked for. The
  family-wide duplication is a pattern-level observation, not rot, and is left alone.
- **`new` — the per-section decoders** (residue labels, molecule ids, exclusions, tree,
  GB pair, box, refusals). Nothing in the tree reads any of these sections: grepping
  `RESIDUE_LABEL|EXCLUDED_ATOMS_LIST|BOX_DIMENSIONS|ATOMS_PER_MOLECULE|SOLVENT_POINTERS|RADIUS_SET|TREE_CHAIN|JOIN_ARRAY|IROTAT|RADII|SCREEN|CAP_INFO|IPOL|CMAP|LENNARD_JONES_CCOEF|FORCE_FIELD_TYPE|CTITLE`
  across `**/*.rs` returns only `NONBONDED_PARM_INDEX` in `ff/forcefield/readers/prmtop.rs:395`
  (a force-field table, phase 02's business). There is nothing to generalize.

## Files to create or modify

- `molrs/src/io/data/prmtop.rs`
- `molrs/src/core/store/schema/mod.rs`
- `molrs/src/ff/potential/kspace/pme.rs`
- `regressions/amber-prmtop-complete-01-structure.py` (new)

## Tasks

- [x] Write failing unit tests in the `#[cfg(test)] mod tests` of `molrs/src/io/data/prmtop.rs` covering `res_name`/`mol_id`/`tree`/`gb_radius`/`gb_screen`, the `exclusions` block, `exclude_14` in both `dihedrals` and `impropers`, `IFBOX` 1/2/3, the meta scalars, and every refusal message
- [x] Register `exclude_14` in `molrs/src/core/store/schema/mod.rs` — `col!` row (`DType::Bool`, `Scalar`, unit `""`) sorted between `"element"` and `"id"`, `consts::EXCLUDE_14`, its entry in `consts_agree_with_the_table`, and `"exclude_14"` in the `dihedrals` + `impropers` `BlockSpec::optional` lists
- [x] Implement the per-atom columns `res_name`, `mol_id`, `tree`, `gb_radius`, `gb_screen` in `molrs/src/io/data/prmtop.rs`, reading `20a4` sections through `prmtop_tables::parse_a4_names` and deleting the local `read_a4_names`
- [x] Implement the `exclusions` block in `molrs/src/io/data/prmtop.rs` (`atomi < atomj`, 0-based uint, `0` placeholders dropped, flattened length cross-checked against `NNB`)
- [x] Implement `exclude_14` on `DihedralRow`, `decode_dihedrals` and `build_dihedral_block` in `molrs/src/io/data/prmtop.rs` so the single builder emits the column into both `"dihedrals"` and `"impropers"`
- [x] Implement `BOX_DIMENSIONS` → `frame.simbox` (IFBOX 1 ortho, IFBOX 2 at `arccos(−1/3)`) and the `radius_set` / `oldbeta` / `solvent_iptres` / `solvent_nspm` / `solvent_nspsol` meta keys in `molrs/src/io/data/prmtop.rs`
- [x] Implement the refusals and POINTERS length cross-checks in `molrs/src/io/data/prmtop.rs` (CMAP, IFPERT, IFCAP, IPOL, CHAMBER, negative `NONBONDED_PARM_INDEX`, `LENNARD_JONES_CCOEF`, `IFBOX=3`, `JOIN_ARRAY`/`IROTAT` length)
- [x] Delete the `read_prmtop` alias and refresh the module-header rustdoc (Output Frame list, format-local unregistered columns, inpcrd box precedence) in `molrs/src/io/data/prmtop.rs`, and correct the stale `"i"`/`"j"` exclusions rustdoc at `molrs/src/ff/potential/kspace/pme.rs` lines 10 and 857
- [x] Add regression example `regressions/amber-prmtop-complete-01-structure.py` (public API only; hard-coded goldens, no third-party runtime)
- [x] Run full check + test suite

## Testing strategy

Unit tests live in `#[cfg(test)] mod tests` inside `molrs/src/io/data/prmtop.rs`,
next to the code, per `CLAUDE.md` § *Testing Rules* — there is **no**
`molrs/tests/` tree in this repo. Each test targets one decode concern of this one
module. Single-test gate:
`cargo test -p molcrafts-molrs --lib --features full,filesystem io::data::prmtop`.
Schema-table tests run under `... store::schema`. Fixtures are inline
`&str` constants; no `tests-data/` dependency, no AmberTools at test time.

The existing `LITFSI_HEAD` constant (`prmtop.rs:716-793`) is extended in place — it
already contains three dihedral rows whose **3rd pointer is negative**
(`12 21 -24 27 4`, `12 21 -24 30 4`, `12 21 -24 33 4`), which is the `exclude_14`
case, with no negative 4th pointer anywhere. A second, deliberately tiny inline
fixture supplies the improper case (one row with a negative 4th pointer *and* a
negative 3rd pointer) so the dihedral/improper flag-parity assertion has something
to compare.

Happy path:

1. `res_name` — `RESIDUE_LABEL` `TF  LI  ` + `RESIDUE_POINTER 1 16` gives
   `res_name[0..15] == "TF"`, `res_name[15] == "LI"`, matching the already-asserted
   `res_id` pattern (`prmtop.rs:850-853`).
2. `mol_id` — `ATOMS_PER_MOLECULE 15 1` gives `[1]*15 + [2]`; **column absent** when
   the section is removed, asserted as `atoms.get_uint("mol_id").is_none()`.
3. `tree` / `gb_radius` / `gb_screen` — present with `NATOM` rows and the exact
   parsed values; each independently absent when its section is removed.
4. `exclusions` — for a hand-written `NUMBER_EXCLUDED_ATOMS` / `EXCLUDED_ATOMS_LIST`
   pair, the block has exactly the expected row count (list length minus the `0`
   placeholders), every row satisfies `atomi < atomj`, and the values are the
   1-based file numbers minus one.
5. `exclude_14` — on `LITFSI_HEAD`, `dihedrals` has 27 rows and exactly **3** with
   `exclude_14 == true`, at **row indices 12, 14 and 16** (section order:
   `DIHEDRALS_INC_HYDROGEN` is empty, so `WITHOUT_HYDROGEN` order is preserved).
6. **Flag parity across the mirror** — on the improper fixture, for every row of the
   `impropers` block, `exclude_14` equals the value on the matching `dihedrals` row
   (matched by the `(atomi, atomj, atomk, atoml, type_id)` tuple). This is the test
   the two-homes mirror requires.
7. `meta` — `radius_set` as the verbatim `RADIUS_SET` line, `oldbeta == 90.0`,
   `solvent_iptres` / `solvent_nspm` / `solvent_nspsol` as the three
   `SOLVENT_POINTERS` integers.
8. `read_amber_prmtop_from_reader` on the unmodified `LITFSI_HEAD` still produces the
   Frame the four existing tests assert — no regression in the eight sections
   already read.

Edge cases:

- `EXCLUDED_ATOMS_LIST` consisting entirely of `0` placeholders → the `exclusions`
  block exists with **zero rows** (schema-typed empty), not absent.
- Every optional section absent → the Frame equals today's output plus nothing;
  no column is fabricated and `frame.simbox.is_none()`.
- `JOIN_ARRAY` / `IROTAT` of the right length → no column appears; of the wrong
  length → `InvalidData`.
- Each refusal trigger gets its own test asserting `ErrorKind::InvalidData` **and**
  that the message contains the subject string plus `"are not supported"`.
- `EXCLUDED_ATOMS_LIST` length ≠ `NNB` → `InvalidData` naming both counts.

Domain validation (hard-coded expected values, `$META.science.required`):

- `IFBOX = 1`, `BOX_DIMENSIONS = 90.0 30.0 30.0 30.0` → `frame.simbox` is orthogonal
  with lengths `(30.0, 30.0, 30.0)` and volume `27000.0` (tolerance `1e-9`).
- `IFBOX = 2`, `BOX_DIMENSIONS = 109.4712190 30.0 30.0 30.0` → the cell matrix from
  `matrix_from_lengths_angles([30;3], [109.4712206;3])`, whose first row is
  `[30.0, −10.0, −10.0]` because `30·cos(arccos(−1/3)) = −10.0` exactly, and whose
  volume is `30³·√(1 − 3·(1/9) + 2·(−1/3)³) = 27000·√(16/27) = 20784.6096908265…`
  (tolerance `1e-6`). `meta["oldbeta"] == 109.4712190`, i.e. the file's truncated
  value is preserved and *not* the value used to build the cell.
- `IFBOX = 3` → refused.
- Charge: `charge[15] == 1.0` within `1e-12` (`1.82223000E+01 / 18.2223`), and the
  16-atom sum is `0.0` within `0.01` — the existing assertions, re-run to prove the
  `18.2223` literal was not "corrected" to `√332.0637133`.

**Regression example:** `regressions/amber-prmtop-complete-01-structure.py` — a
minimal public-API script writing the extended LiTFSI fixture text to a temp file,
calling `molrs.io.read_amber_prmtop` (already exported,
`molrs-python/python/molrs/io/__init__.py:307`) and asserting, as **literals in the
script**: 16 atoms; `res_name` = 15×`"TF"` + `"LI"`; `mol_id` = 15×`1` + `2`;
`dihedrals` 27 rows with `exclude_14` true at exactly `{12, 14, 16}`; `impropers`
flag parity; `exclusions` row count and first row; box lengths `(30, 30, 30)` and
`meta["oldbeta"] == 90.0`; `charge[15] == 1.0`. The goldens are hand-derived from the
format spec and the fixture text — **no AmberTools, RDKit or any third-party
scientific package is imported or subprocessed**, at test time or run time, per
`CLAUDE.md` § *Testing Rules (MANDATORY)*. The script header records provenance and
the runner, following `regressions/mrec-format-02-io.py`:
`uv --directory molrs-python run python ../regressions/amber-prmtop-complete-01-structure.py`.

Doctests: `cargo test --doc -p molcrafts-molrs --features full,filesystem` must stay
green after the `read_prmtop` deletion and the rustdoc edits (`--lib` does not cover
doctests, and the deleted alias may be referenced by a rustdoc link).

## Out of scope

- **Force-field parameter tables** — `BOND_FORCE_CONSTANT`, `DIHEDRAL_PERIODICITY`,
  `LENNARD_JONES_ACOEF`/`BCOEF`, `NONBONDED_PARM_INDEX`, `HBOND_ACOEF`,
  `SCEE`/`SCNB_SCALE_FACTOR`. Phase `amber-prmtop-complete-02-forcefield`.
- **1-4 weights in the LAMMPS / GROMACS / XML writers and readers.** Phase
  `amber-prmtop-complete-03-lammps`.
- **Every binder surface** — `molrs-python`, `molrs-wasm`, `molrs-capi`, the `.pyi`
  stub, and the deletion of the Python `read_prmtop` alias. Phase
  `amber-prmtop-complete-04-python`.
- **Collapsing the improper mirror** (`prmtop.rs:662-673`). Routed:
  `/mol:fix prmtop-improper-two-homes`.
- **The `resname` / `res_name` drift** in `gro.rs:433,524` and `xyz.rs:1469`. Routed:
  `/mol:fix resname-schema-key`.
- **De-duplicating the `%FLAG` lexer** between `io/data/prmtop.rs` and
  `ff/forcefield/readers/prmtop.rs`. Architect-confirmed: no legal shared home under
  `architecture-rules.md` § *Module dependency rules* (`ff → io` is forbidden, and a
  format lexer does not belong in `core`). Kept as a **Debt: note** entry; no route,
  because there is nowhere legal to route it to today.
- **Coupling the prmtop and inpcrd readers** so box precedence is enforced in code.
  Precedence is documented in the rustdoc only; the two readers stay independent.
- **Registering `tree` / `gb_radius` / `gb_screen`** in `SCHEMA_COLUMNS` — no
  cross-format consumer exists, so they stay format-local by design.
- Considered and rejected: inferring `mol_id` from connected components when
  `ATOMS_PER_MOLECULE` is absent (a solvated system's partition is a fact of the
  file, and a silently-invented column is worse than a missing one), and
  auto-generating the exclusion list from bond topology when the sections are absent
  (it would make a truncated file look complete to PME).

## Follow-up (molpy)

- `molpy/tests/test_io/test_canonical_fields.py:52` lists
  `("amber-prmtop", _read_prmtop, frozenset())` — an empty allow-set, i.e. **every**
  column of a prmtop Frame must be canonical. This spec adds three deliberately
  format-local columns (`tree`, `gb_radius`, `gb_screen`), so once phase 04 lands and
  molpy re-pins, that entry must either grow those three names (with the same "this
  is a debt, not a licence" comment the file already carries at `:48-50`) or the
  molpy side must project them away. Flagged now so molpy does not discover it as a
  red gate. `_read_prmtop` itself calls `mp.io.read_amber(...)`, not
  `molrs.io.read_prmtop`, so the alias deletion in phase 04 does not touch it.
- molpy pins molrs by **major.minor**; per `CLAUDE.md` § *Release before molpy*,
  none of this reaches molpy until phases 01–04 are on `master` under a `vX.Y.Z`
  tag and published. No molpy change is part of this spec.
