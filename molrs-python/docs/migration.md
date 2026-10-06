# Migration guide

This guide lists the breaking changes in each minor release, grouped by
surface. Every downstream project pins one molrs minor line (see
[Consuming molrs from another project](https://github.com/MolCrafts/molrs/blob/master/docs/interop.md)),
so moving to a new minor means working through its section. Changes that only
add features are not listed here, apart from a short "Also new" list at the
end of each section; [What's new in 0.16](release-notes.md) walks through the
new features.

## 0.15 → 0.16

0.16 holds every force field in one force-field IR, which adopts the LAMMPS
standard ([Force-field IR](guides/forcefield-ir.md) is the reference). Most
of what breaks is force-field code: the meaning of some stored numbers,
records, the engine readers and writers, and the Rust error types. The
sections run from the IR's definitions and records, through compiling, the
new styles and the engines, to the typifiers, the IR protocol and the Python
API; each states what 0.15 did. molrs 0.15.1 was versioned on `master` but
never published, so coming from 0.15.0 read
[From 0.15.0: the 0.15.1 changes](#from-0150-the-0151-changes) as well.

### ABI line, pickles and JSON

- **ABI line.** The capsule names are now `molrs.FrameRef/0.16`,
  `molrs.ForceFieldRef/0.16` and `molrs.RegionRef/0.16`, so handles cannot be
  exchanged with a 0.15 build. Rebuild consumer extensions and re-pin them to
  `>=0.16.0,<0.17`. The frame vocabulary version stays 2 (`atomm` and
  `cmaps` are additions).
- **Records are `molrec_version` 2**, which 0.15 cannot read; 0.16 reads
  0.15's version-1 records (see [Records](#records-molrec_version-2)).
- **A `ForceField` pickled by 0.15 does not unpickle in 0.16** (a pickle
  now carries each style's arity).
- **Force-field JSON is not converted.** `molrs_ff_from_json` reads a
  0.15 `molrs_ff_to_json` document as written, ½k harmonic `k`, radians,
  `D`, `a_thole` and `fourier` included: convert it as
  [the force-field IR](#the-force-field-ir-adopts-the-lammps-standard)
  says, or move it through a `*.mrec` record, which is converted on read.
- **Every binding sees the force-field changes.** WASM, C and C++ return the
  same typifier output (out-of-plane impropers centre first, MMFF `theta0`
  in degrees), write box-less LAMMPS data the same way and read version-1
  records the same way as Rust and Python; the C++ trajectory writer refuses
  to append to a version-1 store. Their exported symbols and `MolrsStatus`
  codes are unchanged.

### The force-field IR adopts the LAMMPS standard

molrs holds force fields in one force-field IR, which adopts LAMMPS's
definitions as its standard: every style's energy expression, factors and
parameter units are those of the LAMMPS style it corresponds to, and every
angle-valued parameter is in degrees. The readers, writers and typifiers
moved with the kernels, so **the energy of a physical system read from a
file or typed by a typifier does not change** (except where 0.15 was wrong:
the bug fixes are marked as such below). What changes is the meaning of
stored numbers, so a force field **built by hand** (`def_type`) or stored by
an older molrs needs its values converted:

- **`bond harmonic` `k` is LAMMPS's `K`**: E = k(r − r0)², no ½.
  `k_new = k_old / 2`.
- **`angle harmonic` `k` is LAMMPS's `K`**: E = k(θ − theta0)².
  `k_new = k_old / 2`.
- **Angle-valued parameters are degrees** (they were radians):
  `angle harmonic` / `class2` `theta0`, `improper harmonic` `chi0`,
  `dihedral periodic` `phase` / `phase<m>`, `dihedral charmm` `phase`,
  `improper periodic` `phase`, `dihedral class2` `phi1..phi3`, and the
  per-instance MMFF `theta0` column of `angles`. `deg = math.degrees(rad)`.
  Force constants stay per radian².
- **`bond morse` `D` is `d0`** (LAMMPS's `D0`); `pair morse` reads `d0` in
  both its compiled and neighbour-driven forms (the compiled one read `D0`).
- **`pair thole` `a_thole` is `damp`** (LAMMPS's `pair_coeff` name).
- **The `dihedral fourier` style is gone**; it was an alias of
  `dihedral periodic` (same kernel, same params). The LAMMPS reader reads
  `dihedral_style fourier` as `periodic`, the prmtop reader produces
  `periodic`, and the LAMMPS writer writes `periodic` as `fourier`.
- **Out-of-plane impropers list the centre first.** `uff_inversion` and
  `mmff_oop` rows (UFF and MMFF typifier output) are `(centre, a, b, c)`;
  they were `(a, centre, b, c)`. Their type names follow (`UFF {tj}-{ta}-…`;
  MMFF keeps its key as the name, its endpoints centre first).
- **The LAMMPS reader converts nothing.** A coefficient is stored as written
  (it used to store `k = 2K` and radians), and the force field declares the
  file's `units`: a `metal` include is a `metal` force field (it used to be
  converted to `real`), with LAMMPS's `metal` Coulomb constant on `coul/cut`.
  `lammps_coeff_params` returns the coefficients as written.
- **The LAMMPS writer writes coefficients as stored** (`K = k`, no
  `to_degrees`), converting only when the target `units` differs from the
  force field's. `lammps_coeff_values` takes params in the `units` it
  writes. `lammps_units`' `to_store_*` / `from_store_*` / `*_k_lammps`
  helpers and the `½k` form maps are gone; `LammpsFfUnits::scale(from, to)`
  returns the `UnitScale` that converts a parameter by its dimension
  (`UnitScale::apply`).
- **Readers of other engines convert to the force-field IR.** GROMACS:
  `k = k_b/2`, `k = k_θ/2`, degrees kept. OpenMM XML: `k/2` for bonds and
  angles, radians → degrees. AMBER prmtop: `k = RK`, `k = TK`, radians →
  degrees. The GAFF and OPLS-AA tables (`GaffTypifier`, `OPLSAATypifier`)
  and the GAFF estimator's empirical constants follow; the frcmod writer
  writes `RK = k`, the GROMACS and OpenMM writers convert back.
- **`dihedral charmm` `w` is priced** (bug fix: it was read and ignored,
  pricing a CHARMM field's 1-4 pairs at zero); see
  [1-4 interactions](#1-4-interactions).
- **`dihedral harmonic` has a kernel** (`k[1 + sign·cos(nφ)]`, LAMMPS's); the
  LAMMPS reader read it but nothing priced it.
- **`forcefield` sections state `"angle": "degree"`** beside their preset.
  A record molrs 0.15 wrote is converted on read (see
  [Records](#records-molrec_version-2)). A section built in memory with
  `"angle": "radian"` beside a preset is refused (its preset and its angle
  unit disagree): convert its parameters as above and restate the unit.
- **Generic XML (`<BondStyle>` …) no longer renames `k0` to `k`.** The two
  meant the same `½k` number; in the force-field IR they do not, so a `k0`
  attribute is kept as `k0` and a kernel that needs `k` refuses it.

### Parameter defaults and refusals

- **One place applies a default.** `StyleSpec::gather` fills every default a
  style's spec declares (style params, per-type rows, each term of an
  indexed family) and checks each stated value's kind before any kernel — a
  built-in constructor, a generic kernel, an expression — sees a parameter;
  the 1-4 exceptions and `materialize_one_four` read through it too. No
  built-in kernel states a default of its own, so an absent parameter
  prices the same in every kernel. Where this changes what 0.15 did:
  - `pair coul/cut` and `coul/charmm` take `dielectric` 1 when the style
    omits it (LAMMPS's default; 0.15 refused it as "force-field data").
  - `pair lj/cut`, `lj/class2`, `buck`, `morse` and `coul/cut` declare
    `cutoff` with default ∞ (untruncated), which their compiled kernels
    already did; see [Pair cutoffs](#pair-cutoffs-and-the-pme-box) for the
    neighbour-driven forms.
  - `pair lj/class2` mixes by its `mixing` (default `sixthpower`, LAMMPS's)
    when a pair has no cross row; 0.15 refused any pair without one.
  - The zeros 0.15's kernels read for an absent parameter are the specs'
    declared defaults, so energies do not change: `dihedral opls` `k1..k4`,
    `multi/harmonic` `a1..a5`, `class2` `k1..k3` / `phi1..phi3`, `charmm`
    and `periodic` `phase`, `charmm` `w`, `improper harmonic` `chi0`,
    `improper periodic` `phase`, `pair mmff_vdw` `da` (neither donor nor
    acceptor); `angle mmff_stbn` declares the `linear` column it reads.
  - `mixing` accepts the canonical names only (`arithmetic`, `geometric`,
    `sixthpower`); foyer's `combining_rule="lorentz"` is translated by the
    OPLS-AA (foyer XML) reader, and the `lorentz` / `lorentz-berthelot`
    aliases are gone.
- **Every missing or ill-typed parameter is a typed `IrError`.** A built-in
  constructor that lacks a parameter raises `MissingParam` (`style`, `type`,
  `param`; Python `molrs.ff.ir.MissingParam`) where 0.15 raised a plain
  `ValueError` with a message; a per-instance column a typifier did not bake
  (`kb` of `mmff_bond`, …) is `MissingParam` too. `BadValue` (`style`,
  `type`, `param`, `reason`; Python `molrs.ff.ir.BadValue`) refuses a value
  of the wrong kind (text for a number, an array of another rank), text
  outside its declared choices (`mixing = "lorentz"`) or a value outside its
  domain (a non-integer `n` of `lj/cut`, `inner >= cutoff` of a CHARMM
  switch). An unlike pair of `buck` / `morse` with no cross row is
  `NoMixing`. Rust: `VdwStyleParams::from_style` returns
  `Result<Self, IrError>`.
- **Python raises the typed refusal everywhere.** Every refusal of the
  force-field IR — from `def_style` / `def_type`, `PotentialCompiler.compile`
  and `defer`, the optimizers and every writer — is a subclass of
  `molrs.ff.ir.IrError`, itself a `ValueError`, carrying its fields as
  attributes (`err.style`, `err.param`, …). `except ValueError` still catches
  them; code that matched the exact type or the message text does not.
  Structural failures (a missing block or column, an unknown type label)
  stay plain `ValueError`.
- **`dihedral periodic`** refuses a gap in its terms (`k1`, `k3` without
  `k2`: `MissingParam` `k2`; 0.15 silently dropped `k3`) and a row spelling
  both `k` and `k1` (`BadValue`).

### Records: molrec_version 2

Every record molrs 0.16 writes is `molrec_version` 2
(`molrs.io.mrec.schema.MOLREC_VERSION`, Rust `molrs::MOLREC_VERSION`). In
version 2 the `forcefield` section is the force-field IR, so some stored
numbers mean something else than in the version-1 records molrs 0.15 wrote.
0.16 never reads a version-1 record as version 2: it converts every changed
number exactly on read, or refuses the record by name.

- **Readers accept versions 1 and 2 and refuse a newer one.** A store
  without `molrec_version` predates version 1 and is read by version 1's
  rules. `read_mrec_meta` (and `MolRec.meta` in Rust) hands `meta` back as
  stored, so a converted record still says `molrec_version: 1`.
- **Writers always stamp 2.** A producer's `molrec_version` in `meta` is
  overwritten (0.15 kept it, and refused an unsupported one). Writing a
  converted record back writes a version-2 record.
- **What a version-1 `forcefield` section converts** (its angle unit is
  `units.angle`, else the radian of every version-1 preset, the `lj` preset
  included, which stated none):

  | Version 1 | Version 2 |
  |---|---|
  | `units` | `"angle": "degree"` |
  | `bond harmonic`, `drude harmonic` `k` (½k form) | `k / 2` |
  | `angle harmonic` `k` (½k form) | `k / 2` |
  | `theta0` (`angle harmonic`, `class2`, `mmff_angle`, `uff_angle`), `chi0` (`improper harmonic`), `phase` / `phase<m>` (`dihedral periodic`, `charmm`, `improper periodic`, `trefoil`), `phi1..phi3` (`dihedral class2`) | degrees |
  | `bond morse` `D`, `pair morse` `D0` | `d0` |
  | `pair thole` `a_thole` | `damp` |
  | `dihedral fourier` | `dihedral periodic` |
  | `improper mmff_oop`, `uff_inversion` rows, centre second | centre first (`itom` and `jtom` swap) |

  A force constant per radianⁿ stays as it is; one stated per degree (a
  section with `"angle": "degree"` and no preset) is re-expressed per radian.
- **Frames convert too** (`system`, `frame`, every trajectory frame): a
  relation row's parameter columns convert as the parameter of the same name
  of the row's style — MMFF's per-instance `angles.theta0` becomes degrees —
  and an `mmff_oop` / `uff_inversion` row swaps `atomi` and `atomj`. A
  record without its force field converts by its own columns (`theta0`,
  `chi0`, `phase*`, `phi*` are angles; an `impropers` row carrying `koop` or
  `K` is an out-of-plane row).
- **Refused, by name:** a `pair14` style (no 0.15 reader produced one), a
  multi-term `improper periodic`, an `expression` on a converted style, an
  unknown `angle` / `dihedral` / `improper` style that carries parameters, an
  angle unit other than the radian and the degree, and a version-1 trajectory
  opened for appending (`TrajectoryWriter` on an existing 0.15 store: read it
  and write a new record).
- **Unchanged:** `dihedral charmm` `w` (the same 1-4 weight in both
  versions; 0.16 prices it), every other style's numbers, `special_bonds`,
  and the atom order of every other improper. A 0.15 record of an
  OpenMM-read field keeps the improper rows 0.15 priced; re-read the XML with
  0.16 for OpenMM's own order.
- **The `forcefield` section holds more.** Any type parameter may be an
  array (molrec's `f64[T, S…]`: one shape per column, every trailing axis at
  least 1, finite values in a non-null row); a `cmap` table's `grid` is
  further `f64[T, N, N]` (`N ≥ 2`). A non-float or empty-axis array column,
  a non-square grid, an array under an annotation or canonical key, and an
  array style param are refused by `validate` / `to_section`.
  `ForceField.from_section` reads a `cmap` style and every category beyond
  the seven (0.15 refused both); see
  [The force-field IR as a protocol](#the-force-field-ir-as-a-protocol).
- **`pair14` is no category.** molrec retired it; `category_arity("pair14")`
  is `None`, and a `pair14` table is kept as
  unknown content (no arity or restatement check). No reader produced it.
- **`ForceFieldSection.validate` refuses a `pair lj/charmm` `one_four`**
  other than `"regular"` / `"epsilon14"`, so `read_mrec_forcefield` refuses
  such a record before anything turns it into a `ForceField`.

Every 0.15 test record (`molrs/src/io/zarr/testdata/v1`, written by the
published molrs 0.15.0 wheel: MMFF, harmonic / periodic, morse / class2 /
charmm, `fourier` under `lj` units, class2 under `metal`) prices in 0.16 to
the energy and forces 0.15.0 computed for it.

### Compiling

- **Rust: compiling returns `CompileError`.** `PotentialCompiler::compile`
  and `compile_typed` return `Result<_, CompileError>` (0.15: `String`):
  `CompileError::Ir(IrError)` for a refusal of the force-field IR,
  `NoBox { category, style, reason }`, `Invalid(String)` for anything else.
  It implements `Display` and `std::error::Error`, `err.ir()` is the IR
  refusal when it is one, and it converts from `String` / `&str` /
  `IrError`; a caller that propagated the `String` maps it
  (`.map_err(|e| e.to_string())`). `KernelConstructor` and every built-in
  `*_ctor` return `Result<Member, CompileError>` (0.15: `String`); a custom
  constructor that returns `Err(message.into())` or uses `?` on a `String`
  error still compiles.
- **A force field with several styles of one bonded category compiles.**
  `PotentialCompiler` hands each table-driven style only the rows of its own
  types (0.15 handed every style every row, so `angle harmonic` beside
  `angle charmm` failed with "unknown angle type"). A row whose type no style
  of the category defines is still an error, naming the type.
- **The compiled `pair lj/cut` prices its Mie exponents** (bug fix). The
  compiled door priced 12-6 whatever `n` / `m` the style declared; it now
  prices `C ε[(σ/r)ⁿ − (σ/r)ᵐ]` as the typed door does. A 12-6 style is
  unchanged.
- **The compiled `pair buck` and `morse` find a pair's row from its two
  atoms' types** (the self row, else the cross row; neither is `NoMixing`),
  as their neighbour-driven forms and LAMMPS's `pair_coeff i j` do. 0.15
  read a per-pair `type` column off the `pairs` block instead, which
  `intramolecular_pairs` never writes, so the compiled door failed on any
  list molrs built; that column is no longer read.

### Pair cutoffs and the PME box

- **`PotentialCompiler.compile` (the `pairs`-list door) truncates every
  pair style at its `cutoff`**, exactly as `compile_typed` and LAMMPS do: a
  `pairs` row prices only at `r < cutoff`, a shifted `lj/cut` (`shift`,
  `pair_modify shift yes`) is shifted to zero there, and the CHARMM styles
  switch between `inner` and `cutoff`. 0.15 priced every listed pair
  whatever the style's `cutoff`. This holds for every pair style: the
  built-ins, a run-time `ScalarForm`, a Python callable and an expression.
  A `special_bonds` 1-4 pair is truncated with the rest (LAMMPS applies the
  special weights inside its cutoff test); the 1-4 exceptions kernel — a
  `dihedral charmm` `w` pair, a per-pair override — is not, as LAMMPS's
  `dihedral_style charmm` prices its 1-4 pair at any distance. A pair
  exactly at the cutoff prices nothing (`rsq < cutsq`; 0.15's typed
  `lj/cut` priced it).
- **A style that states no `cutoff` is untruncated**: the declared default
  ∞ of `lj/cut`, `lj/class2`, `buck`, `morse`, `coul/cut`, and every style
  whose spec has none. Fields read from OpenMM XML (`NoCutoff`), a prmtop or
  a GROMACS topology without a stated cutoff price as before. Only a field
  that states a `cutoff` shorter than some listed pair prices differently,
  and then as LAMMPS does.
- **The neighbour-driven `lj/class2`, `buck`, `morse` and `coul/cut` stop
  at their `cutoff`** (`r < cutoff`, as LAMMPS and `lj/cut` do; 0.15 priced
  every pair the neighbour table held, beyond the cutoff too), and **refuse
  an absent (∞) `cutoff`** (`BadValue`), as `lj/cut` and the generic pair
  kernel already did: a neighbour sum is not finite without one.
- **`pair coul/long/pme` takes its cell from the frame** (`frame.box`), as
  LAMMPS's kspace takes its simulation box; the undeclared style params
  `box_xx` … `box_zz` it read are gone (a field stating them now states
  parameters nothing reads). A frame without a box, with a box not periodic
  in x, y and z, or with a cell outside LAMMPS's restricted triclinic form
  is refused by name: Rust `CompileError::NoBox`, Python `ValueError`.

### Urey–Bradley (angle charmm)

A new style, LAMMPS's `angle_style charmm`: `angle charmm`,
E = k(θ − theta0)² + k_ub(r₁₃ − r_ub)², type params `k` (energy/rad²),
`theta0` (degrees), `k_ub` (energy/length²), `r_ub` (length) — the
`angle_coeff t K theta0 K_ub r_ub` numbers as written. It is an angle style,
not a category: `def_style("angle", "charmm")` (Python: an `AngleStyle`), the
`forcefield` section's `angle.charmm` table and `*.mrec` carry it like any
angle style, and both compile doors price it. The 1-3 spring adds no
exclusion; which 1-3 pairs a pair style sees is `special_bonds`'s answer, as
before. See [Force-field IR](guides/forcefield-ir.md#ureybradley). The
LAMMPS ([hybrid styles](#lammps)), OpenMM (`AmoebaUreyBradleyForce`),
GROMACS (`[ angletypes ]` funct 5) and chamber prmtop readers and writers
carry it.

### CMAP

The `cmap` category (five endpoints) and its kernel `cmap charmm` — LAMMPS
`fix cmap`, ported step for step
([Force-field IR](guides/forcefield-ir.md#cmap)) — are new, and LAMMPS's
`fix cmap` files are read and written.

- **Rust: `StyleDefs` is `#[non_exhaustive]` and has a `Cmap` variant.**
  `StyleDefs::Cmap(Vec<CmapType>)` holds five-endpoint types
  (`itom` … `mtom`). A `match` on `StyleDefs` outside molrs needs a wildcard
  arm. `ForceField::def_style("cmap", …)` defines a style (0.15 returned
  `DefError::UnknownCategory`), and its `def_type` takes exactly five
  endpoints.
- **Rust: `ENDPOINT_COLUMNS` has five entries.**
  `store::forcefield_section::ENDPOINT_COLUMNS` is `[&str; 5]`
  (`itom` … `ltom`, `mtom`), and `category_arity("cmap")` is `Some(5)`
  (0.15: `None`). A `cmap` style table must carry all five endpoint columns.
- **Frame vocabulary.** The canonical key `atomm` (`u64`, fifth relation
  endpoint) and the block `cmaps` (relation of arity 5, optional `type`,
  `type_id`, `style`) are new; `subset`, `replicate` and the validator
  renumber and range-check `atomi` … `atomm`. `keys::ENDPOINTS` (Rust
  `[&str; 5]`, Python `molrs.keys.ENDPOINTS`) gains `atomm`, so a block
  without a spec now reads an `atomm` column as an endpoint into `atoms`, an
  `atomm` column at another dtype than `u64` is refused, and the fifth
  endpoint column of a five-node `MolGraph` relation is `atomm` (0.15:
  `atom4`).
- **A `cmaps` block compiles** at both doors (0.15 refused the category).
  Each row needs a 2-D square `grid` (any N ≥ 2; LAMMPS files are 24×24).
- **The LAMMPS data reader reads `fix cmap`'s sections.** The header line
  `N crossterms` (or `N cmap crossterms`) and the `CMAP` section become a
  `cmaps` block (`atomi` … `atomm`, numeric `type_id`, the map's index in
  the `fix cmap` file). 0.15 refused the header line and the `CMAP` section
  (unless skipped with `with_skipped_section("CMAP")`, which still skips
  it); a `CMAP` section without the header count is refused. The
  `lammps_counts` meta gains `crossterms=N`.
- **The LAMMPS data writer writes a `cmaps` block** as `N crossterms` and a
  `CMAP` section, types by label id; `mol_id` is required for it as for every
  bonded block. Read such a file in LAMMPS with
  `read_data <file> fix cmap crossterm CMAP`.
- **`TypeLabels` covers `cmaps`**, with the inventory meta key
  `cmap_type_labels` (`keys::CMAP_TYPE_LABELS`). A frame whose `cmaps`
  block has rows but neither `type` nor `type_id` is refused by every
  writer that resolves type labels.
- **Rust: `LammpsWriteOptions` has a `cmap_file` field** (`Option<String>`,
  default `None`; with `skip_special_bonds`, see [LAMMPS](#lammps), two new
  fields): a struct literal needs them or `..Default::default()`. An
  include written for a system with `cmaps` labels needs it — the writer
  refuses without — and starts with `fix cmap all cmap <cmap_file>` and
  `fix_modify cmap energy yes` beside `units` (the fix must precede
  `read_data`). Python: `write_lammps_forcefield(…, cmap_file=…)`, same for
  `write_lammps_forcefield_str`.
- **The LAMMPS include reader reads `fix <id> <group> cmap <file>`** (the
  file relative to the include when read from a path) and `fix_modify` of
  that fix; any other `fix` line is refused by name (0.15: "unknown LAMMPS
  keyword `fix`").
- **Python.** `molrs.ff.CmapStyle` / `CmapType` (with `itom` … `mtom`);
  `def_style("cmap", …)` returns a `CmapStyle`.
- **C API.** `"cmap"` is a category; `molrs_schema_column_dtype("atomm")`
  is `"uint"`.
- **New.** `assign_cmaps(frame, ff)` builds the `cmaps` block from a frame's
  dihedrals and the field's cmap rows (forward matching only);
  `read_lammps_cmap_str` / `LammpsCmapFile` / `LammpsFfReader::read_cmap_str`
  read a `fix cmap` file into rows named `"1"` … `"K"`;
  `LammpsFfWriter::write_cmap_str` and `lammps_cmap_str` write one (CHARMM's
  own file comes back line for line); `CmapGrid` / `CmapCharmm` are the
  kernel. Python: `molrs.ff.assign_cmaps`, `read_lammps_cmap`,
  `write_lammps_cmap`.

### Array parameters

- **Rust: `Params` holds `f64` arrays** beside numbers and strings
  (`set_array`, `get_array`, `iter_arrays`); `==` and `same_parameters`
  compare them exactly, so two definitions differing only in an array are a
  conflict. Code that copies a `Params` key by key through `iter()` and
  `iter_strings()` loses the arrays.
- **Python.** A param value may be an array: `def_type(**params)`,
  `def_style(params=…)` and `Type.__setitem__` take a numpy array or a
  nested list / tuple of numbers and store float64, so a list value that
  raised `TypeError` in 0.15 is now stored. `params` and `t[key]` return
  arrays as float64 numpy arrays; pickles carry them.
- **C API.** `molrs_ff_to_json` writes an optional `array_params` object (on
  a style or a type, only when it holds an array param: nested lists, one
  level per axis), and `molrs_ff_from_json` reads it and refuses a ragged or
  non-numeric one.
- **Records.** An array parameter is a `f64[T, S…]` column (see
  [Records](#records-molrec_version-2)); it round-trips through
  `to_section` / `from_section`, a `*.mrec` store and `molrs.io.mrec`.

### 1-4 interactions

LAMMPS's three 1-4 mechanisms are all priced, LAMMPS's way; see
[Force-field IR](guides/forcefield-ir.md#1-4-interactions) for the formulas.

- **New pair styles `lj/charmm` and `coul/charmm`** — the halves of LAMMPS
  `pair_style lj/charmm/coul/charmm`: per type `epsilon`, `sigma`,
  `epsilon14`, `sigma14`; style `inner`, `cutoff` (both required), `mixing`
  (default `arithmetic`); CHARMM's energy switch at both compile doors, the
  Coulomb force LAMMPS's switched force. The LAMMPS reader reads
  `pair_style lj/charmm/coul/charmm inner outer [inner2 outer2]` and
  `pair_coeff` with two or four numbers (two store `epsilon14 = epsilon`,
  `sigma14 = sigma`); the writer writes them back with four. A data file's
  `Pair Coeffs # lj/charmm/coul/charmm` is refused (no switching cutoffs);
  `lj/charmm/coul/long` is refused.
- **`lj/charmm` style param `one_four`** — `"regular"` (default; a
  `special_bonds` 1-4 pair at the regular ε/σ, LAMMPS) or `"epsilon14"` (at
  ε₁₄/σ₁₄, OpenMM / GROMACS pairtypes); any other value is refused.
  `ForceField.materialize_one_four(frame)` (Rust
  `ForceField::materialize_one_four`) writes a frame's 1-4 pairs as override
  rows from the field's declarations; compiling an `"epsilon14"` field for a
  frame without them is refused, and the LAMMPS writer refuses the param.
- **`dihedral charmm` `w` prices its end atoms' 1-4 pair**,
  `w·[LJ(ε₁₄, σ₁₄) + C qᵢqⱼ/r]` with no cutoff, a pair at the ends of
  several dihedrals taking the sum. Refused, as in LAMMPS: `w > 0` beside
  `special_bonds` 1-4 weights other than 0, or without `lj/charmm` (and a
  Coulomb style); `w` outside [0, 1]. The OpenMM and GROMACS writers refuse
  a non-zero `w`.
- **Per-pair overrides on `pairs`.** Float columns `epsilon`, `sigma`,
  `charge_product`, `lj_scale`, `coul_scale` (null cell: the force field's
  value; `molrs::store::schema::PAIR_OVERRIDE_COLUMNS`). Explicit values are
  final and the scales replace the `special_bonds` weight (or `w`).
  Precedence: override > `w` > `special_bonds`. The LAMMPS data-file writer
  and the Python LAMMPS force-field writers refuse a frame carrying them.
- **One exceptions kernel at both doors.** Every override pair and every
  `w > 0` pair is priced by one more member (`PairExceptions`, an indexed
  member). `compile` drops the override rows from the `pairs` list the pair
  styles see; `compile_typed` reads those rows and weights them 0 in every
  pair member.
- **Rust: `TypedMember` is `(Member, Option<PairWeights>)`** (was
  `Option<BondDistanceWeights>`). Build the MD weights with
  `SpecialWeights::new(&w.special_weights(&topo))` (0.15:
  `topo.special_weights(&w)`, which still exists for a
  `BondDistanceWeights` but misses the pairs an override weights 0);
  `PairWeights::by_distance()` is the old table.

### Torsions

- **`dihedral nharmonic`** (LAMMPS's `Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ`, params `a1..aN`,
  contiguous, N ≥ 1) is a new style: kernel, LAMMPS reader
  (`dihedral_coeff t N A1 … AN`) and writer. A gap (`a1`, `a3` without
  `a2`) or a missing `a1` is refused at compile time and by the writer.
- **Rust: `molrs::ff::forcefield::torsion` is new** — the exact maps
  between every torsion form and a Fourier series (`FourierSeries`,
  `TorsionRefusal`, one type per form with `from_params` / `to_params`,
  `torsion_series(category, style, style_params, row)` for any registered
  torsion style); see
  [Torsion forms and their exact conversions](guides/forcefield-ir.md#torsion-forms-and-their-exact-conversions).
  `<Form>::from_series` is exact, the constant included, or refuses
  (`TorsionRefusal::ConstantOffset`); `<Form>::nearest(series)` is the
  constant-blind answer. A `dihedral charmm` row with `w ≠ 0` has no series
  (`TorsionRefusal::OneFourWeight`).

### LAMMPS

- **Bonded `hybrid` styles and `angle_style charmm` are read.**
  `angle_style hybrid harmonic charmm` defines both styles, and each
  `angle_coeff t <sub-style> …` line is a type of the sub-style it names; the
  same for `bond`, `dihedral` and `improper`. A data file's
  `Angle Coeffs # hybrid` section reads the same way.
- **The reader refuses a bonded coefficient line with extra numbers.**
  LAMMPS refuses them too; 0.15 dropped them, so an
  `angle_coeff t K theta0 K_ub r_ub` line under `angle_style harmonic` lost
  its Urey–Bradley term silently.
- **The writer writes one `*_style hybrid` line** when a category's used
  types span several LAMMPS styles, each coefficient line naming its
  sub-style (0.15 wrote one `*_style` line per style, and LAMMPS kept only the
  last). Every data-file `* Coeffs` section it writes names its style in the
  header comment (`Angle Coeffs # charmm`, `# hybrid` with the sub-style on
  each row), as LAMMPS's `write_data` does, so `read_data_coeffs` reads a
  non-`harmonic` section back under the right style.
- **Newly read and written**: `bond morse`, `improper cvff`, `bond class2`,
  `angle class2` and `dihedral class2` (cross-term lines at zero; a non-zero
  cross term is refused), `pair buck`, `pair morse`, `pair lj/class2`
  (always `sixthpower`, as LAMMPS mixes it), a `hybrid` of such pair styles
  and a `hybrid/overlay` of one with `coul/cut` / `coul/long`. A `Pair
  Coeffs # morse` data-file section reads (it was refused).
- **`pair_style lj/cut` alone has no Coulomb style.** LAMMPS prices no
  charge under it; 0.15 added a `coul/cut` beside it, which priced the
  charges. `lj/cut/coul/cut` reads as before.
- **`lj/cut/coul/long` reads as `lj/cut` + `coul/long/pme`** (its cutoff and
  LAMMPS's constant), not as a plain `coul/cut` cut off at the same
  distance: the Ewald sum is not a cut-off sum. The include's
  `kspace_style` states an accuracy, not an `alpha`, so the Ewald
  parameters are not read, and compiling the style refuses until they are
  stated. The writer writes such a style back as `lj/cut/coul/long`, and
  refuses one that states its Ewald parameters (molrs's smooth PME is not
  LAMMPS's PPPM). Other `lj/cut/coul/*` Coulombs (`debye`, `dsf`, `wolf`, …)
  are refused, not read as plain.
- **`pair_modify shift yes`** reads as `lj/cut`'s `shift` and is written
  (0.15 dropped it both ways); a Mie `lj/cut` (`n`, `m` ≠ 12, 6) is
  refused by the writer (0.15 wrote a 12-6 line).
- **The writer converts per dimension.** A field written in another unit
  style has every parameter scaled by its `Dim` (`E*L^6`, `1/L`, …), not per
  style arm; numbers agree with 0.15's to the last digit or two.
  `pair_coeff` lines name the lower type id first (LAMMPS sets nothing for
  `pair_coeff I J` with `I > J`). `lj/charmm/coul/charmm` is followed by
  `pair_modify mix arithmetic` (LAMMPS's default for it, now stated).
- **A pair `hybrid` states each sub-style's mixing rule**: one
  `pair_modify pair <sub-style> mix <rule>` per sub-style whose spec declares
  `mixing` (its rule, else the spec's default `arithmetic`); 0.15 wrote none,
  so LAMMPS mixed an `lj/cut` sub-style geometrically.
- **Writer refusals instead of silent output**: a buffered (`delta ≠ 0`) or
  `dielectric ≠ 1` `coul/cut`; a pair style with no LAMMPS form (`coul/tt`,
  …; it was written into a `pair_style hybrid` line LAMMPS cannot read); a
  `pair_coeff` with a per-pair cutoff (molrs has none; it was dropped); a
  style with types in a category LAMMPS has no `*_style` for (a run-time
  category, `drude`; it was dropped); a non-integer `dihedral charmm` phase
  (LAMMPS reads it as integer degrees; it is now written as an integer,
  where 0.15 wrote `180.000000`, which LAMMPS refused).
- **`skip_pair_style` skips the `pair_style` line only.**
  `write_lammps_forcefield(_str)(…, skip_pair_style=True)` (Rust
  `LammpsWriteOptions::skip_pair_style`) keeps `special_bonds` and
  `pair_modify mix` / `shift`: they are the force field's, and LAMMPS's
  defaults (`0 0 0`, `geometric` for `lj/cut`) are not the field's. 0.15
  dropped both, so a relaxation that set its own `pair_style` ran with
  LAMMPS's 1-4 weights and mixing rule. A caller that states its own 1-4
  weights passes the new `skip_special_bonds=True`. `pair_modify` needs a
  pair style: read a `skip_pair_style` include after the input's
  `pair_style`.
- **A box-less frame is written inside the bounds of its atoms.**
  `write_lammps_data` gives a frame without a box the axis-aligned bounds of
  its coordinates widened by 1 length unit on every side (0.15 wrote a
  `0 1` placeholder box, which LAMMPS wraps the atoms into under `p` and
  loses them from under `f` / `s`). Read such a file with
  `boundary s s s`; a non-finite coordinate is refused. A frame meant to be
  periodic carries its box, written as before.
- **A style's LAMMPS form is part of its spec** (see
  [Engine codecs](#engine-codecs-and-refusals)): the reader, the writer,
  `lammps_coeff_params` and `lammps_coeff_values` go through it.

### OpenMM XML

The OpenMM reader (`OplsXmlReader`, Python `read_opls_xml`;
`read_forcefield_xml` dispatches to it) and writer (`XmlForceFieldWriter`,
`write_forcefield_xml`) cover every force OpenMM's `app.ForceField` builds
from a Class-I file, and refuse the rest by name; see
[Force-field IR](guides/forcefield-ir.md#openmm-xml). Checked against
OpenMM's own energies (Reference platform) and LAMMPS on CHARMM36, AMBER
ff14SB and OPLS-AA molecules.

- **OpenMM impropers price as OpenMM does (bug fix; energies change).** An
  `<Improper class1 class2 class3 class4>` lists the centre first and OpenMM
  prices the dihedral `(c2, c3, c1, c4)`. molrs stored the file order and
  priced `(c1, c2, c3, c4)` — a different dihedral, 0.40× OpenMM's energy on
  the regression molecule. The reader now stores `(c2, c3, c1, c4)` (AMBER's
  order, centre third, the order `improper periodic` is priced in everywhere
  — GROMACS, AMBER, LAMMPS `cvff`); `ordering="charmm"` rows without
  wildcards stay as written, `ordering="smirnoff"` is refused. The writer
  writes the inverse, and refuses `improper cvff` as a periodic improper
  (OpenMM cannot price a dihedral that starts at the centre; 0.15 wrote rows
  OpenMM priced over a different dihedral). A frame built for an OpenMM-read
  field lists each improper's atoms in its type's endpoint order, now
  `(c2, c3, c1, c4)`.
- **New sections read and written.** `<LennardJonesForce>` (`sigma14`,
  `epsilon14`, `<NBFixPair>`) is `pair lj/charmm` + `coul/charmm`;
  `<AmoebaUreyBradleyForce>` is `angle charmm`; `<CMAPTorsionForce>` is
  `cmap charmm` (OpenMM's map shifted by N/2 per index and transposed into
  φ-major); `<CustomTorsionForce energy="k*(theta-theta0)^2">` is
  `improper harmonic` (`theta0 = 0`; the writer uses
  `k*(abs(theta)-theta0)^2` for `chi0 ≠ 0`).
- **`<RBTorsionForce>` reads as `dihedral multi/harmonic`** (`nharmonic` when
  C₅ ≠ 0), exactly and with its constant; it was `dihedral opls`, which
  refused C₅ ≠ 0 and ΣCₙ ≠ 0. The OPLS-AA typifier built from XML takes its
  dihedral candidates from every dihedral style.
- **Coulomb uses OpenMM's constant.** An OpenMM-read `coul/cut` /
  `coul/charmm` has `coulomb = 332.06371329919216` (OpenMM's `ONE_4PI_EPS0`),
  not LAMMPS `real`'s 332.06371: Coulomb energies of an OpenMM-read field are
  9.9·10⁻⁹ larger than in 0.15.
- **`NonbondedForce` states its mixing.** Without the foyer
  `combining_rule`, `lj/cut` gets `mixing = "arithmetic"` (OpenMM's rule)
  instead of none.
- **1-4 parameters of their own.** A `<LennardJonesForce>` whose types carry
  `sigma14` / `epsilon14` gives its `lj/charmm` the style param
  `one_four = "epsilon14"` (see [1-4 interactions](#1-4-interactions)); call
  `ForceField.materialize_one_four(frame)` before compiling such a frame.
- **Refused, by name, instead of skipped or misread**: every
  `Custom*Force` other than the harmonic improper (0.15 skipped them
  silently), `<Script>`, an RB `<Improper>`, a bonded row naming neither
  `class{n}` nor `type{n}` (0.15 read it as a `*` wildcard; OpenMM ignores
  it), `ordering="smirnoff"`, odd CMAP sizes, a `<NonbondedForce>` with
  `epsilon ≠ 0` beside a `<LennardJonesForce>`, and `<PeriodicImproperForce>`
  (the section molrs 0.15.0's writer made up; 0.15.1 still read it):
  rewrite such a file with `<PeriodicTorsionForce>`. `<Info>` and
  `<Patches>` are skipped like `<Residues>`.
- **Writer.** Writes everything above, plus `dihedral charmm` (`w = 0`),
  `harmonic` and `class2` as periodic terms, `multi/harmonic`, `nharmonic`
  and `opls` as RB, explicit LJ cross rows as `<NBFixPair>` (0.15 refused
  them), charges from the atom types (0.15 read them from the pair rows,
  where no reader puts them, and wrote 0), class / type endpoints as the
  field names them, `combining_rule` when not arithmetic. `bond morse`,
  `bond class2`, `angle class2`, `improper cvff` and any registered or
  instance expression style are written as `CustomBondForce`,
  `CustomAngleForce`, `CustomTorsionForce`, `CustomNonbondedForce`, or a
  `<Script>`-built `CustomCompoundBondForce` (a run-time compound category),
  the parameters in IR units and the expression rewritten exactly
  (`4.184*(E[r → 10*r])`); 0.15 refused them, and silently skipped
  run-time categories. It refuses every style it has no form for (0.15
  skipped bond, angle and pair styles it did not know). Placeholder class
  types are no longer written as atom types.
- **Writer: every `<Type>` has a class** (OpenMM refuses a file without):
  a type without one — a prmtop's, a LAMMPS file's — is written as its own
  class. **Two types on the same labels** (as OpenMM's generator matches
  them) are refused when their parameters differ (OpenMM would price every
  such term with the first) and written once when they agree; a proper's
  periodic and RB rows on one quartet are refused (OpenMM adds both). A
  shifted or Mie `lj/cut` and a force field in units other than `real` are
  refused.
- **`write_forcefield_xml(path, ff, precision)`: `precision` is optional**
  (`Option<usize>` in Rust, also for `write_forcefield_xml_str` and
  `XmlForceFieldWriter::with_precision`); the default writes each number in the shortest
  form that reads back to the same float (0.15: six decimals).

### GROMACS

The GROMACS reader holds everything the force-field IR can, and the writer
is its inverse; see
[Force-field IR](guides/forcefield-ir.md#gromacs-topologies). Checked
against GROMACS 2025.3 and LAMMPS term by term on a CHARMM, an AMBER and an
OPLS-AA dipeptide (`scripts/gromacs_engine_check.sh`).

- **`[ dihedraltypes ]` funct 3 (Ryckaert–Bellemans) reads as
  `dihedral multi/harmonic`** (`aₙ₊₁ = (−1)ⁿ Cₙ`, constant included), or
  `dihedral nharmonic` when C₅ ≠ 0 — no longer as `dihedral opls`, and no
  longer refused for C₅ ≠ 0 or ΣCₙ ≠ 0. Code that looked a GROMACS-read RB
  type up under `dihedral opls` finds it under `multi/harmonic`; the energy
  is the same (with ΣCₙ = 0, as 0.15 required). The writer writes
  `multi/harmonic` and `nharmonic` (N ≤ 6) as funct 3 and **`dihedral opls`
  as funct 5** (GROMACS's Fourier dihedral, which is `opls` term for term;
  0.15 wrote funct 3), and the reader reads funct 5 back as `opls`.
- **New directives read and written:** `[ pairtypes ]` (an `lj/charmm` +
  `coul/charmm` field declared `one_four = "epsilon14"`, its 1-4 parameters
  as `epsilon14` / `sigma14`, divided by fudgeLJ), `[ angletypes ]` funct 5
  (`angle charmm`), `[ dihedraltypes ]` funct 9 (consecutive rows on equal
  labels are one multi-term `dihedral periodic`; the writer writes a
  multi-term type as funct-9 rows instead of refusing it), funct 5, funct 2
  at ξ₀ = 180°, the 2-name form, `[ cmaptypes ]` (`cmap charmm`, grid
  unchanged). A field whose `[ pairtypes ]` give 1-4 parameters other than
  the generated ones reads with `lj/charmm`, not `lj/cut`; it switches
  between `inner` and `cutoff`, which the caller sets (GROMACS keeps them in
  the `.mdp`). Price a system read with it after
  `ForceField.materialize_one_four(frame)`, which writes its 1-4 pairs out.
- **`[ defaults ]` gen-pairs `no`** is read (1-4 LJ weight 1, every pair
  priced by its own parameters) instead of refused.
- **A row's own type is named deterministically.** A `[ dihedrals ]` (…)
  row with parameters of its own becomes one type per distinct parameter
  set, `<labels>@gmx_<n>`; equal sets were told apart by the text of a
  hash map, so the same file could read as a different number of types,
  numbered differently, from run to run. `Params`' `Debug` now lists its
  keys in order, and equal rows are one type.
- **`#define` macros are expanded** in rows (0.15 recorded them and never
  expanded them), a line ending in `\` continues, and text before the first
  section is read past as GROMACS reads it past (charmm27's `forcefield.itp`
  banner was refused). `with_include_dir(dir)` resolves `#include` against
  GROMACS's share directory, as `-I` does.
- **Unknown atom types are refused** in `[ nonbond_params ]` and
  `[ pairtypes ]` (0.15 stored a cross row on an undefined type).
- **A bonded type restated in reverse with other parameters is refused**
  (0.15 stored it as a second type; the same orientation was already a
  `TypeConflict`).
- **Coulomb uses GROMACS's constant.** A GROMACS-read `coul/cut` /
  `coul/charmm` has `coulomb = GROMACS_COULOMB` = 332.06371329919205
  (GROMACS's `ONE_4PI_EPS0`, CODATA 2018), not LAMMPS `real`'s: Coulomb
  energies of a GROMACS-read field are 9.9·10⁻⁹ larger than in 0.15, and
  equal GROMACS's.
- **Whole topologies read: `GromacsTopFfReader::read_system` / Python
  `molrs.ff.read_gromacs_system`** read the molecule sections too, into the
  force field and a typed frame (0-based indices): GROMACS's own type lookup,
  rows with their own parameters as types `<labels>@gmx_<n>`, `[ pairs ]`
  rows with parameters as per-pair overrides, the nrexcl pair list (every
  pair of two molecules too, up to `MAX_ATOMS_FOR_A_FULL_PAIR_LIST` atoms,
  so `compile` prices them as GROMACS does), exclusions, constraints and
  settles, `[ molecules ]` repeated. The force-field reader still refuses
  molecule sections, now naming `read_system` (0.15 pointed at
  `molrs.io.read_top`, which reads structure only, 1-based).
- **Whole topologies written: `GromacsTopFfWriter::write_system_str(ff,
  frame)`**, the inverse of `read_system`: one `[ moleculetype ]` per
  molecule, each row with its type's parameters, `[ pairs ]` with the
  override cells, `[ exclusions ]` for the pairs the frame does not price.
- **Writer refusals:** `dihedral charmm` with `w > 0`, `epsilon14` /
  `sigma14` on an `lj/charmm` not declared `one_four = "epsilon14"`, two
  types GROMACS would read as one (same labels in one function-code table),
  a force field in units other than `real`. A Coulomb constant other than
  LAMMPS `real`'s is no longer refused (GROMACS prices at its own, as the
  LAMMPS and OpenMM writers already let their engines do), and
  `dihedral class2` is written as funct-9 rows at phase φₙ + 180°.

### AMBER prmtop

The prmtop readers (`read_amber_prmtop`, `AmberPrmtopFfReader` /
`read_amber_prmtop_ff`) read what they refused, and the frame and the force
field they return change where they did. AMBER stays read-only. See
[Force-field IR](guides/forcefield-ir.md#amber-prmtop) for the full map.

- **Chamber (CHARMM) prmtops read.** 0.15 refused `%FLAG CTITLE` /
  `FORCE_FIELD_TYPE`. Now: every angle is `angle charmm` with its
  Urey–Bradley term, `CHARMM_IMPROPERS` are `improper harmonic` (centre
  first; a ψ₀ off 0°/180° is refused), CMAP is `cmap charmm` and a `cmaps`
  block, Lennard-Jones is `lj/charmm` (with `epsilon14` / `sigma14` and
  `one_four = "epsilon14"` when the file's 1-4 table differs) and Coulomb
  `coul/charmm` at 332.0716; charges are de-scaled by √332.0716 (not
  18.2223). The force field's `name` is `"CHARMM"`. Its pair styles carry no
  `inner` / `cutoff`: declare them before compiling. With `one_four =
  "epsilon14"`, build the pair list (`intramolecular_pairs`) and call
  `ForceField.materialize_one_four(frame)` before compiling.
- **CMAP reads** (ff19SB's `CMAP_*` too); 0.15 refused `CMAP_COUNT > 0`.
- **Non-uniform `SCEE` / `SCNB` read.** 0.15 refused two divisors among the
  1-4 rows. `special_bonds` is now the divisor most 1-4 rows carry (it was
  the one value), and the frame gains a `pairs` block — only when some pair
  is weighted otherwise — listing those 1-4 pairs with `coul_scale` /
  `lj_scale` cells. It is not a pair list: `intramolecular_pairs` builds the
  full list and keeps the cells (it used to drop a frame's `pairs`). An
  `intramolecular_pairs` call on a frame whose `pairs` row with an override
  names a 1-2 / 1-3 pair raises.
- **Multi-term impropers read.** 0.15 refused a negative-`PN` chain on an
  improper and silently kept only the first term of several improper rows on
  one quartet. Such an improper is now one `improper periodic` type
  `<quartet>@<n>` and one `impropers` row per term (a single-term improper
  is unchanged).
- **A phase within 0.004 rad of ±π is ±180° exactly**, as sander takes it
  (tleap writes π as `3.14159400`); 0.15 stored 180.0000153°.
- **Dihedral types.** The terms of a `dihedral periodic` type are sorted by
  periodicity (they were in the file's type-id order). Two torsions of one
  type quartet with different terms (tleap reuses a quartet's first match:
  GAFF2's `hc-c3-ca-ca` alone beside `hc-c3-ca-ca` + `X -c3-ca-X`) are two
  types, the second named `<quartet>@<n>` in both readers; 0.15 merged their
  terms into one type, silently.
- **Atom types.** A type name that stands for atoms of two LJ classes or
  masses is split into `<name>~<class>` types (0.15 raised a
  `TypeConflict`); every bonded type name and frame label uses the split
  names.
- **Refused, by name:** a 1-4 row on a negative-`PN` chain, or on a bonded /
  angle-end pair; `IPOL > 0` (0.15: only `IPOL = 1`). The force-field reader
  now refuses `IPOL > 0` as the frame reader does.

### Engine codecs and refusals

Engine I/O follows the force-field IR's protocol; see
[Force-field IR](guides/forcefield-ir.md#engine-codecs).

- **`StyleSpec.lammps`** (`LammpsForm::{None, Positional, Custom}`). A style
  registered with `LammpsForm::positional()` (Python
  `register_style(..., lammps="positional")`, `"positional:<name>"` for
  another LAMMPS name, `ir.StyleSpec.lammps`, or afterwards
  `ir.register_engine_form("lammps", category, name, form)`) is read and
  written by the LAMMPS reader and writer with nothing else written:
  `params` in order, each converted by its `Dim`. `StyleInfo.lammps` names a
  style's form.
- **Every engine refusal is `NoEngineForm`.** The GROMACS and frcmod
  writers refuse a style that is not built in, and the LAMMPS and OpenMM
  writers a style with no form, as "`<engine>` has no form for `<category>`
  `` `<style>` ``: …", typed: `ForceFieldWriter::write_str` / `write` (and
  `write_amber_frcmod` / `write_amber_frcmod_str`, `write_forcefield_xml` /
  `write_forcefield_xml_str`, `GromacsTopFfWriter::write_system_str`,
  `lammps_coeff_values`, the LAMMPS writer's `write_data_coeffs_str` /
  `write_cmap_str`) return **`WriteError`**
  instead of `String`: it dereferences to its message (so
  `err.contains(…)` still reads it, and `String::from(err)` converts) and
  `err.ir()` is the `IrError::NoEngineForm` when an engine refused a style.
  Python raises `molrs.ff.ir.NoEngineForm` (a `ValueError`) from every
  writer, as from `register_engine_form`.
- **Readers and writers take a registry**: `LammpsFfReader::with_registry`,
  `LammpsFfWriter::with_registry`, `XmlForceFieldWriter::with_registry`
  (the process-wide one by default).

### GAFF and GAFF2

- **`GaffTypifier` in Python**: `molrs.ff.typifier.GaffTypifier(
  parameter_set="gaff" | "gaff2")` (also `molrs.ff.GaffTypifier`), the Rust
  `GaffTypifier` behind the usual `Typifier` interface; compose it after
  `AtdTypifier(parameter_set=…)`, which types the atoms. See
  [Force-field IR](guides/forcefield-ir.md#gaff-and-gaff2).
- **`gaff2.dat` is 2.2.30** (AmberTools 26.1; it was an older 2.2): new
  atom type `hb`, the impropers `X -X -cc-X`, `X -X -cd-X`, `X -X -nc-X`,
  `X -X -nd-X` (10.5 kcal/mol), and revised torsion rows (34 rows of the
  old table replaced by 31). GAFF2-typed energies change accordingly.
  `gaff.dat` is unchanged.
- **Impropers are built as AmberTools builds them.** `GaffTypifier` puts an
  improper wherever tleap does, with tleap's atom order and parmchk2's
  estimate (0.15 put one at each `PARMCHK.DAT`-planar centre, peripherals
  sorted by type, with its own estimate: 1.1 where parmchk2 gives 10.5 and
  back, e.g. every GAFF2 `c2` / `ce` / `cc` centre). Improper energies of
  GAFF-typed molecules change; ethylene under GAFF2 goes from 1.1 to
  10.5 kcal/mol per improper.
- **Every term the table lacks is estimated as parmchk2 estimates it**:
  torsions (0.15's analogy ranking picked other rows, e.g. indole's
  `ca-ca-cd-cc`, and refused guanidinium's `nh-cz-nh-hn`), and bonds and
  angles, whose `estimate_penalty` is now parmchk2's (caffeine's `c-cc-na`:
  2.6, was 2.15). A bond parmchk2 finds no analog for is now a missing term
  (parmchk2 writes it with a zero length, `ATTN`); 0.15 made one up from
  Badger's rule (`hc-br`). `gaff_estimator` is gone: GAFF no longer goes
  through `Parmchk2Estimator`, which stays as the generic estimator other
  force fields (OPLS-AA's `with_default_estimator`) borrow, with its own
  scoring, unchanged.
- **Estimated type names** carry their analog and penalty,
  `<types>@<analog>_<penalty>` (`c3-o-c-os@c3.o.c.oh_8.5`,
  `c-cc-na@c2.cc.na_2.6`), so one output force field can hold two
  estimates of a name; 0.15 named them by their types alone.
- **`AtdTypifier` types the bond orders antechamber perceives.** By default
  (`bond_orders="perceive"`, Rust `AtdBondOrders::Perceive`) the bond orders
  are judged from the connectivity as `bondtype -j full` judges them, and
  the graph's own are ignored, so a molecule types as `antechamber` types the
  mol2 file with the same atom and bond order; 0.15 typed the orders the
  graph stated (aromatic bonds kekulized by molrs). The atom types of a
  molecule with two Kekulé structures can change (cyclooctatetraene drawn
  `C1=CC=CC=CC=C1` is now `cc cc cd cd …`, antechamber's), as can every
  type that depends on ring classes or the colouring, which now follow
  antechamber's `ring.c`, `atadjust` and `cpadjust` (anthracene's middle
  ring is `ca`, was `cc` / `cd`; o-terphenyl's second bridge carbon `cq`,
  was `cp`), under every table — and with them the AM1-BCC and Gasteiger
  charges, which type through the same path. `bond_orders="input"` keeps the
  graph's orders. Every hydrogen must be drawn.
- **Rust: `AtdRule::alternate` is an `Option<Alternate>`** (the partner name
  and the `AlternatePass`, `Conjugated` or `Bridge`), was
  `Option<&'static str>`; `cp` carries `cq` in the `Bridge` pass.
- **Rust: `ParmchkPenalty` names the columns as parmchk2 reads them**
  (`bl blf cba cbaf ba baf ctor tor ps`): `AngleCentre` / `AngleCentreForce`
  are columns 2 / 3 and `Angle` / `AngleForce` 4 / 5 (0.15 had them the
  other way round), `TorsionCentre` is 6 and `Torsion` 7 (likewise
  swapped). Code that read a column by variant reads the other one now.

### Typifier matches carry any relation kind

- **Rust: a typifier `Match` carries any relation kind.** The fixed fields
  `bonds`, `angles`, `dihedrals` and `impropers` are gone; `Match::links`
  (`IndexMap<String, Vec<Annotations>>`) maps a graph relation kind (the
  Frame block of its category: `"bonds"`, a custom `"urey_bradleys"`) to
  rows positional against that kind's own rows. Replace `m.bonds = rows` with
  `*m.link_mut("bonds") = rows` (or `m.links.insert(..)`), `m.bonds.push(a)`
  with `m.link_mut("bonds").push(a)`. A type under a kind defines a type of
  the category whose block the kind is (`typifier::link_category`); a
  non-empty vector for a kind the graph lacks is an error naming it.
  `write_onto` defines the kinds in the graph's registration order, as
  before for the four built-ins.
- **Rust: `Match::assign_terms(graph, kind, library, key)`** (new) types
  every row of a relation kind against the library's type rows of its
  category by the atoms' types: slot by slot with wildcards, in the orders
  the category's `EndpointOrder` allows (reversible, ordered or unordered),
  fewest wildcards first, then table order (molrec's rule). It returns the
  positions nothing matched.
- **Python: `Match(nodes, links=...)` takes a kind name as a key** as well
  as a relation class (`{Bond: rows, "urey_bradleys": rows}`); a relation
  class other than `Bond` / `Angle` / `Dihedral` / `Improper` / `Port`, or
  a key that is neither class nor `str`, still raises `TypeError`, and
  naming one kind twice (`{Bond: …, "bonds": …}`) `ValueError`.
  `repr(Match)` lists the kinds: `Match(nodes=2, links={bonds=1}, styles=0,
  pairs=0)`.

### The force-field IR as a protocol

The force-field IR is a protocol (`molrs::ff::ir`, Python `molrs.ff.ir`,
new in 0.16): categories and styles are registrations of one form, the
built-ins sealed among them; see
[Extending the force-field IR](guides/extending-forcefield-ir.md). What
changes for code written against 0.15:

- **Rust: `register_kernel` / `register_kernel_with` return
  `Result<(), IrError>`** and register into the IR registry (0.15: `()`,
  overriding whatever was there). A built-in is sealed (`IrError::Sealed`);
  registering the same constructor again is a no-op, another one under a
  taken name is `IrError::Conflict`. To change a built-in's behaviour,
  register a style of another name.
- **Categories beyond the seven.** A category the IR registry declares
  (molrec's `constraint`, `drude`, `virtual_site`, or a custom one
  registered with `molrs::ff::ir::register_category`), or one read from a
  record that nothing declares, is a style category like `bond`:
  - **Rust: `Style::category()` and `StyleDefs::category()` return `&str`**
    (0.15: `&'static str`); the string borrows the style.
  - **Rust: `StyleDefs::Relation { category, arity, types }`** holds every
    such category, its types `RelationType { name, endpoints, params }`
    (`endpoints: SmallVec<[String; 5]>`).
  - **Rust: `DefError` carries owned categories.** `DefError::Arity` has
    `category: String, expected: String` and `DefError::TypeConflict` has
    `category: String` (0.15: `&'static str`). `DefError::Unsupported` is
    gone (nothing produced it); `DefError::CategoryArity { category,
    expected, got }` is new: a style of a category with another number of
    endpoints than its registration, or than the styles of it the force
    field holds. `DefError` is not `#[non_exhaustive]`: a `match` on it needs
    an arm for `CategoryArity` (and, coming from 0.15.0, `PairConflict`).
  - **Rust: `ForceField::def_style` accepts every declared category.** The
    arity of a category beyond the seven comes from the process-wide IR
    registry (`ForceField::def_style_in` takes a registry), else from the
    styles of it the force field already holds; anything else is still
    `DefError::UnknownCategory`. `ForceField::def_style_with_arity` defines
    a style of a category nothing declares, with the arity given (what
    `from_section` does with a record's endpoint columns). New:
    `Style::arity`, `StyleDefs::arity`, `ForceField::get_relationtypes`.
  - **`ForceField::from_section` keeps a category beyond the seven** (it
    refused `virtual_site` and every unknown category): its arity is the
    registry's, or the count of its table's endpoint columns.
  - **Compiling.** A style of a category nothing declares is priced as a
    compound custom category: from its block `<name>s` by its style's
    `expression`, or refused by name (``no kernel for <category> `<style>` ``)
    when the block has rows and nothing prices it; a block that is absent
    prices nothing.
  - **Python: `ForceField.def_style` returns a `RelationStyle`** for such a
    category; its `def_type(name, *endpoints, **params)` takes as many
    `AtomType` endpoints as the category's `arity` (`ValueError` otherwise)
    and returns a `RelationType`. `ForceField.styles`, `get_styles` and
    `get_types` include these styles (0.15 dropped them silently);
    `get_styles(RelationStyle)` / `get_types(RelationType)` select them all.
  - **C API.** `molrs_ff_def_style` and `molrs_ff_def_type` accept the same
    categories as `ForceField::def_style`.
- **Custom styles persist.** A custom style or category is stored in a
  `*.mrec` record as its molrec style entry, and a process that registered
  nothing reads it back:
  - `to_section` writes a registered custom style's `expression` when the
    style has none of its own (a built-in is written with none; a style's
    own `expression` is written, and read back, byte for byte). Rust:
    `ForceField::to_section_in(&Registry)`.
  - A style nothing registers, with no expression, is read whole (its rows,
    its array params, its category); compiling it is refused by name:
    ``no kernel for <category> `<style>`: register it
    (molrs.ff.ir.register_style) or give it an expression``. A style with
    an expression is priced by it, bit for bit as in the process that
    registered it.
  - A registered custom style whose instance carries an `expression` that
    differs from the registry's is priced by its registered kernel; at
    first compile the instance expression is checked against it, energy and
    derivative to 1e-10, and a disagreeing one is refused (`IrError::Disagree`,
    naming the style). A registered style with neither kernel nor
    expression is priced by the instance's.
- **Form conversions** (new): `ForceField.canonical()`,
  `ForceField.to_form(category, style)` and `ForceField.fit_form(category,
  style, q, w=None, *, kt=None, offset=False)` (Rust: the same on
  `ForceField`, `fit_form` taking a `molrs::ff::ir::Metric`), over the form
  families `torsion` (canonical `dihedral periodic`), `bond`, `angle`
  (canonical `harmonic`) and `lj` (canonical `pair lj/cut`); see
  [Converting between forms](guides/forcefield-ir.md#converting-between-forms).
  An out-of-image conversion is refused (`IrError::OutOfImage`, Python
  `molrs.ff.ir.OutOfImage`, a `ValueError`).

### Python: kernels live in `molrs.ff.potential`

`LJCut` moved from `molrs.md` to `molrs.ff.potential` (molpy: `molpy.md.LJCut`
→ `molpy.potential.LJCut`), beside the `Potential` protocol, which `molrs.md`
no longer re-exports either; nor does it re-export `Potentials`
(`molrs.ff.Potentials` / `molpy.Potentials`). The integrators still accept all
of them.

New in the same module: `kernel(category, style, atoms, *, charges=None,
**params)`, the kernel of **any** style the force-field IR prices — a
built-in, a style registered through `molrs.ff.ir` (expression or Python
kernel), a style of a custom category, an unregistered style given its
`expression=` — over explicit instances: `atoms` `(n, arity)`, each per-term
parameter a number or one value per term **as stored** (angle values in
degrees, indexed families as `k1`, `k2`, …), style parameters (`cutoff`,
`coulomb`, …) a number or a string, per-atom charges as `charges=`. It is
built by the code `PotentialCompiler.compile` runs (Rust:
`molrs::ff::potential::Instances`), returns a `Potentials`, and
`Potentials.push` moves it into a larger collection. There is no class per
built-in style: one builder covers every registered style, custom ones
included.

```python
from molrs.ff import Potentials
from molrs.ff.potential import kernel

pots = Potentials()
pots.push(kernel("bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.4))
pots.push(kernel("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5))
pots.push(kernel("dihedral", "periodic", [[0, 1, 2, 3]],
                 k1=1.3, periodicity1=1, phase1=0.0, k2=0.4, periodicity2=2, phase2=180.0))
pots.push(kernel("pair", "coul/cut", [[0, 3]], charges=q, coulomb=332.06371, dielectric=1.0))
energy, forces = pots.calc_energy_forces(pos)
```

| 0.15 | 0.16 |
|---|---|
| `from molrs.md import LJCut, Potential` | `from molrs.ff.potential import LJCut, Potential` |
| `from molpy.md import LJCut` | `from molpy.potential import LJCut` |
| `molrs.md.Potentials` | `molrs.ff.Potentials` |

### Module ownership

Every module has one job, and every public symbol one path. Paths that moved
or went away:

**Core, io, perceive, compute, conformer, md (single-responsibility pass A2)**

- Minimum image: `molrs::compute::util` is gone. `MicHelper` is
  `molrs::spatial::simbox::Mic` (`SimBox::mic()`, or `Mic::ortho(lengths)`
  for bare edge lengths); `get_positions_ref` is crate-private.
- `molrs::op::random::standard_normal` is the one Gaussian draw (md had two
  private copies).
- `molrs::conformer::etkdg` is private (`generate_3d_impl` was a second door
  to `Conformer::generate`); its distance-geometry objectives are internal,
  and the stages minimize with `molrs::optimize::minimize_lbfgs_rms` instead
  of a steepest-descent of their own, so embedded geometries differ.
- Removed from `io` (force-field formats belong to `ff::forcefield`):
  - `molrs::io::data::top::*` (`read_top`, `read_top_frame`, `write_top`,
    `TopReader`, `TopFrameWriter`) → `ff::forcefield::readers::gromacs::
    read_system` / `ff::forcefield::writers::gromacs::write_system_str`
    (0-based indices). Python: `molrs.io.read_top` →
    `molrs.ff.read_gromacs_system`; `molrs.io.write_top` is removed (the
    Rust system writer has no Python binding yet).
  - `molrs::io::data::frcmod::*` (`read_frcmod`, `parse_frcmod`,
    `format_frcmod`, `write_frcmod`, `FrcmodFile`) → `ff::forcefield::
    writers::frcmod::write_amber_frcmod`. Python: `molrs.io.read_frcmod`,
    `parse_frcmod`, `write_frcmod` removed (`molrs.ff.write_amber_frcmod`
    writes one).
  - `molrs::io::data::prmtop_tables::decode_{bond,angle,dihedral,nonbond}_params`
    and their row aliases, and `io::data::prmtop::read_amber_prmtop_sections`;
    `parse_pointers` / `parse_a4_names` are crate-private. Python:
    `molrs.io.prmtop_parse_pointers`, `prmtop_parse_a4_names`,
    `prmtop_decode_*`, `read_amber_prmtop_sections` removed
    (`molrs.ff.read_amber_prmtop_ff` reads the parameters).
- One SMARTS parser: `molrs::io::smiles::parse_smarts`, which
  `perceive::smarts::SmartsPattern` now compiles from. Its IR gains
  `AtomPrimitive::{AtomicNumber, RingSizeRange, RingBondCount, ContextLabel}`;
  `[#6]` is `AtomicNumber(6)`, no longer `Element { "C" }`. `perceive::smarts`
  needs the `smiles` feature (`ff` enables it).
- One way to perceive onto a graph, the `molrs::perceive::Perceive` builder.
  Crate-private now: `perceive::hydrogens::add_hydrogens` (→
  `Perceive::find_hydrogens`), `aromaticity::perceive_aromaticity` (→
  `find_aromaticity`), `bond_order::find_bond_orders`,
  `bond_type::{find_bond_types, find_kekule_orders}` (→ the same-named
  builder methods), `bond_type::{find_bond_types_from_connectivity,
  assign_kekule_numbers}`. The side-table queries stay public.
- `molrs::perceive::{Coarsener, CoarsenError}` → `molrs::builder::{Coarsener,
  CoarsenError}` (Python path unchanged: `molrs.perceive.Coarsener`).
- One name per handle and payload type: `AtomId`, `BeadId` → `NodeId`;
  `BondId`, `AngleId`, `DihedralId`, `ImproperId`, `PortId` → `RelationId`;
  `Bead` → `Atom`; `Bond`, `Angle`, `Dihedral`, `Improper` → `Relation`
  (all in `molrs::system::molgraph` and at the crate root).

### From 0.15.0: the 0.15.1 changes

molrs 0.15.1, a patch on the 0.15 ABI line, was versioned on `master` but
never tagged or published; its changes ship in 0.16. Coming from the
published 0.15.0, these behaviours change as well (where 0.16 goes further,
the bullet says so):

- **LJ cross rows are applied.** A `pair/lj/cut` row whose two endpoints
  differ overrides the style's `mixing` rule for that type pair. 0.15.0
  ignored it at compile time, so energies and forces of any force field
  holding one (hand-built, `scale_lj` output, or read as below) change.
- **LAMMPS force-field reader.**
  - A cross `pair_coeff i j` is kept as a pair type; it used to be dropped.
  - A repeated `pair_coeff` for the same pair (in either order) replaces the
    earlier one, as LAMMPS does; 0.15.0 kept the first.
  - A cross `pair_coeff` with a wildcard (`pair_coeff c3 * …`) is an error;
    it used to be dropped.
  - `read_data_coeffs` reads a data file's `PairIJ Coeffs` section; it used
    to be skipped.
- **GROMACS force-field reader and writer.** `[ nonbond_params ]` (funct 1)
  is read as cross rows and written from them; 0.15.0 refused both. Other
  funct codes are refused.
- **One row per pair.** A pair style prices each unordered pair of atom
  types once. `def_type` on a pair style that restates a stored pair (either
  order, any name) with equal parameters is a no-op, so the restating row is
  not stored; with different parameters it raises `ValueError`
  (`DefError::PairConflict` in Rust). 0.15.0 stored both rows and refused the
  reversed conflict only when the kernel was compiled, and let the last of
  two same-order rows win. Python's `PairStyle.def_type` on an equal restated
  pair returns the stored row's handle and name. A `forcefield` section with
  such a conflict in a `pair` table is now refused on read and by
  `ForceFieldSection.validate()`.
- **AMBER prmtop reader.** A non-Lorentz–Berthelot off-diagonal LJ entry
  (NBFIX) becomes a cross row; 0.15.0 refused the file. The `lj/cut` style
  carries `mixing = "arithmetic"` (0.15.0 left it unset, which meant the
  same rule), so it compares unequal to a hand-built style without it in
  `ForceField.merge`.
- **OpenMM XML reader.** A `<PeriodicTorsionForce>` `<Proper>` in OpenMM's
  `k{m}/periodicity{m}/phase{m}` spelling is `dihedral/periodic` (0.15.0 read
  it as an all-zero `dihedral/opls`), and an `<Improper>` there is
  `improper/periodic` (0.15.0 skipped it). An `<Improper>` under
  `<PeriodicImproperForce>` (0.15.0 skipped the section) is refused in 0.16
  (see [OpenMM XML](#openmm-xml)). A row with neither
  spelling, with both, an incomplete term, a multi-term `<Improper>`, or
  another child tag is an error.
- **OpenMM XML writer.** Impropers are `<Improper>` rows under
  `<PeriodicTorsionForce>`, not `<PeriodicImproperForce>`. 0.15.0 wrote a
  harmonic improper, a `multi/harmonic` dihedral and an LJ cross row as rows
  that read back wrong or not at all; 0.16 writes them as a
  `CustomTorsionForce`, RB terms and `<NBFixPair>` rows (see
  [OpenMM XML](#openmm-xml)).
- **LAMMPS force-field writer.** `dihedral/periodic` is written as
  `dihedral_style fourier`; a force field without pair types writes no
  `pair_coeff` lines instead of failing.
- **Python native typifiers** can be subclassed, but a subclass that
  defines `match` or `library` raises `TypeError` at class creation.

### Also new in 0.16

- `ForceField.materialize_params(frame, *, prefix)` (Rust
  `ForceField::materialize_params(&mut frame, prefix)`) writes the
  parameters a force field gives each relation row and atom of a typed frame
  as columns `<prefix><parameter>` (null where a row's type lacks one) and
  returns block → columns written; see
  [Force-field IR](guides/forcefield-ir.md#parameters-as-frame-columns).
- The cross-engine equivalence check and the completeness matrix
  ([Force-field IR](guides/forcefield-ir.md#completeness)), and the
  `molrs-ext-example` crate, a third party extending the IR through the
  public API alone (`scripts/check.sh ext`).

## 0.14 → 0.15

### All surfaces

- **ABI line.** The capsule names are now `molrs.FrameRef/0.15`,
  `molrs.ForceFieldRef/0.15` and `molrs.RegionRef/0.15`, so handles cannot be
  exchanged with a 0.14 build. Rebuild consumer extensions and re-pin them to
  `>=0.15.0,<0.16`.
- **Frame vocabulary version is 2** (it was 1). This is
  `schema::FRAME_VOCAB_VERSION`, `molrs.schema.VOCAB_VERSION`,
  `schemaVocabVersion()` and `molrs_schema_vocab_version()`.
- **Floats are `f64` only.** f16/f32 columns and f32 meta values are gone.
  Zarr f16/f32 arrays and `"f32"` meta tags are refused on read, not widened.
  Python widens a float16/float32 array to float64 when it is inserted.
- **Image flags are `i32`** everywhere (SimBox, MD, binders). They were `i64`.
- **LAMMPS harmonic impropers evaluate at half the 0.14 energy.** molrs's
  `improper/harmonic` kernel is `k·(χ − χ₀)²`, LAMMPS's own form, but the
  LAMMPS force-field reader stored `k = 2K` (the bond/angle `½k` map) and the
  writer emitted `K = k/2`. Every improper read from a LAMMPS include or data
  file was evaluated at twice the energy LAMMPS gives it. The reader now
  stores `k = K` and the writer emits `K = k`, so energies and forces of
  LAMMPS-read impropers halve, and a force field whose impropers were built
  with the kernel's `k` writes a `K` twice the 0.14 value. The GROMACS
  reader and writer were already right (`k = k_ξ/2`).
- **Force-field string params have one name per fact.** The atom-type string
  params the readers set are renamed to the names molrec's `forcefield`
  section uses: OpenMM/OPLS `class_` → `class` and `def_` → `smarts` (the
  XML `def` SMARTS), and the GROMACS `[ atomtypes ]` `bond_type` → `class`.
  The XML and GROMACS writers read the new names. Code that looked the old
  keys up (`params.get_str("class_")`, `Type.params["def_"]`, …) must use the
  new ones; `type_` is unchanged.
- **The MMFF `pair/mmff_vdw` rows lose their numeric `type` param.** It
  repeated the row's name (the MMFF atom type) and no kernel read it.
- **Frame meta keeps insertion order.** It used to be sorted by key, so every
  meta iterator and index-based accessor now returns keys in insertion order.
- **CoarseGrain frame layout.** A CoarseGrain frame now uses
  `atoms` + `bonds(atomi, atomj)`, plus an optional `members(ibead, atom)`
  block. 0.14 wrote `beads` + `cgbonds(ibead, jbead)`, and old frames in that
  shape no longer load. `from_frame` requires a `bead_type`, `type` or
  `type_id` column and refuses more than one `mol_id`.
- **Force-field model.** A force field is built only through
  `def_style` / `def_type`. Re-defining a style or type with different params
  is an error; an identical re-definition does nothing. Compiling is done by
  `PotentialCompiler`, not by the force field, and typing is done by
  `Typing<T>` (Rust), whose `forcefield()` holds only the types it assigned.
- **Energies change.**
  - The OPLS-AA tables were regenerated from GROMACS `oplsaa.ff` v2026.3: classes
    are now GROMACS `bond_type`, and mixing is geometric (it was arithmetic).
  - Python force fields from `read_opls_xml` / `read_lammps_forcefield` now
    honour the declared `mixing`.
- **GROMACS force-field I/O.**
  - Function codes 2 and 3 were swapped; now 3 = Ryckaert–Bellemans.
  - The reader refuses molecule sections and sections it does not model,
    unless you skip them.
  - `[ atomtypes ]` without `[ defaults ]` is an error.
  - Atom types carry only `[ atomtypes ]` params. Per-atom `[ atoms ]` data is
    read with `read_top`.
- **Stricter readers.**
  - prmtop bond/angle types no longer carry `id`.
  - A second parameter set under one type name is an error.
  - LAMMPS data refuses unknown sections and unparseable header lines unless you
    skip them. Header `units` are read into meta.
  - The LAMMPS ff reader declares `"lj"` for a `units lj` include.
  - Malformed type-label inventory meta (`"1C"`, `"x:C"`, repeated id) is an
    error.
- **LAMMPS coefficient output is driven by the frame's type labels.** Types
  no label uses are not written. A label with no matching type is refused, and
  so is an explicit cross pair in data-file `Pair Coeffs` (write it in the
  `*.ff` include instead).
- **Topology vocabulary (molrec conventions).** New canonical keys: `fx`
  `fy` `fz` (`f64`), `formal_charge` (`i64`), `atom_map` (`u64`), `chain`
  `icode` `altloc` `style` (`string`), `occupancy` `b_factor` (`f64`), `ibead`
  (`u64`); key group `FORCES`. New blocks `constraints`, `drudes`,
  `virtual_sites` (null trailing endpoints allowed) and `members`
  (`ibead` → `atoms`, `atom` → declared). Readers moved to them:
  - PDB: `chain_id` → `chain`; the reader now also stores `altloc`, `icode`,
    `occupancy` and `b_factor` (a blank field is `""`), and the writer
    writes all five back.
  - mmCIF: `chain_id` → `chain`; `res_seq` (`i32`) → `res_id` (`u64`,
    nullable: `.`/`?` are null, a negative number is refused); `b_iso` →
    `b_factor`; also `icode` and `altloc`.
  - GRO: `resname` → `res_name`, `atom_name` → `name` (reader and writer).
  - extxyz: the `resname` property reads as `res_name` and is written back
    as `resname`.
  - LAMMPS molecule JSON: frame meta `units` is the units object
    `{"preset": "real"}`; a bare string is still accepted on write.
  Since `formal_charge` is canonical `i64`, a graph property of that name
  must be an integer (an integral float is accepted) and `to_frame` emits it
  as `i64`.
- **Row references (`targets`).** A `u64` column may declare the block its
  values index (`Block::set_target`, `"<block>"` or `"/<section>/<block>"`;
  `/trajectory/…` is refused). It is persisted as the block group's
  `targets` attribute (frame/system) or in `sequence_schema` (trajectory),
  and writers and readers refuse a reference that does not resolve (missing
  block, value past its rows; null rows reference nothing). Absolute targets
  are checked against the record's `frame` / `system` when present.
- **`schema::relation_endpoints(name, has_column, declared)`** returns
  `Vec<RowReference { column, target }>` (empty when none) instead of
  `Option<(target, Vec<column>)>`, honouring a block's declared targets;
  `EndpointSpec` is `{ columns: &[(column, EndpointTarget)] }` and
  `BlockSpec::endpoint_columns()` returns a `Vec`. Python
  `molrs.schema.relation_endpoints(name, columns, targets=None)` returns
  `[(column, target), …]` (empty, not `None`). `BlockDoc` / Python
  `BlockSpec` gain `declared_endpoints`. `Validator` range-checks declared
  targets, reports `ViolationKind::MissingTarget`, and skips null rows.
- **Aligned trajectory blocks.** `SequenceSchema::declare_aligned(block,
  target)` (Python `declare_aligned`) pins a block's rows to another's at
  every resolved frame (`aligned_with` in `sequence_schema`): the writer
  refuses a frame where the aligned block is present and its target absent
  or of another row count (restate it when the target's count changes), the
  reader refuses such a store, and both refuse an aligned block named like a
  `system` block. Declared only: a schema derived with `from_frames` (and so
  the record door `write_record_file` / `write_mrec_trajectory`) declares no
  alignment.
- **`Frame::subset` / `Frame::replicate` accept `members`** (it was refused):
  `members.ibead` and every declared same-frame reference are renumbered or
  offset; absolute and undeclared references are copied unchanged.
- **`to_frame` can fail.** A node property whose dtype contradicts the
  schema, for example `set_atom(i, "x", "left")`, used to abort the process.
  It is now an error on every surface.
- **`*.mrec` records follow the molrec contract.**
  - `meta.molrec_version` is checked only when it is present. 0.14 refused a
    non-empty `meta` without it; 0.15 opens such a store. A present value must
    still be an integer in `1..=MOLREC_VERSION`, and `null`, `0`, a string, a
    float or a newer version is refused. Writers still stamp
    `molrec_version: 1` on every record.
  - An undefined cell (`box` with `cell_defined: false`) ignores its
    `vectors`. Readers accept any matrix there, zeros included, and do not
    invert it; writers write the identity. 0.14 refused a singular matrix
    even for an undefined cell.
  - A canonical identifier column (`id`, `atomic_number`, `mol_id`, `res_id`,
    `type_id`, `atomi`…`atoml`, `bond_type`, `bond_number`) stored at a width
    other than `u64` is refused on read, naming the column and its width. 0.14
    widened it silently. Inserting a narrow unsigned array in memory still
    widens it to `u64`. `SequenceSchema::declare_column` (Python
    `SequenceSchema.declare_column`) refuses such a key at a narrower width.
  - An observable whose `kind` is neither `scalar` nor `vector` no longer
    fails the record read. It is carried as `ObservableKind::Other(String)`
    and written back unchanged. Observable data with no
    `observables/meta/<name>` entry is still refused.
  - A root section the reader does not know is ignored. 0.14 read every
    unknown section as a frame group: arrays directly under it were dropped,
    and a sequence-shaped one could fail the read.
  - The frame, system and trajectory doors (`read_frame_file`,
    `read_system_file`, `read_trajectory_file`; Python `read_mrec`,
    `read_mrec_system`, `read_mrec_trajectory`) decode only `meta` and their
    own section. A broken trajectory or observables section no longer fails
    `read_mrec`. The lazy trajectory reader (`FrameSequence::open`, Python
    `TrajectoryReader`) and the WASM readers now validate
    `meta.molrec_version` too.
  - **Stricter reads of malformed stores** (molrec `storage.md`,
    `ragged.md`): a frame/system block group without an integer `count`
    attribute is refused; a canonical column (`x`, `ix`, `element`, …, not
    only the `u64` identifiers) stored at any dtype but its declared one is
    refused, while writers convert an in-memory column of another width of
    the family (`ix` as `i64`) to the declared one, refusing a value that
    does not fit; `SequenceSchema::declare_column` refuses a canonical key at
    another dtype. A trajectory with one elision marker (`uniform_rows` /
    `dense_updates`) without the other, a non-positive `uniform_rows`,
    markers on a block with no columns, a `step_progression` /
    `time_progression` without `nstep`, or a non-integer `nstep` is
    refused. A declared block with neither index nor markers still reads as
    absent.
  - **An undefined cell is periodic on no axis.** With
    `cell_defined: false` an omitted `boundary` reads all-`false` (0.14 read
    it all-periodic), a stored periodic flag is refused, and writers refuse
    an undefined `SimBox` with a periodic flag.
  - **Metrics series names** that Zarr forbids as nodes are escaped on their
    first byte (`.` → `%2E`, `..` → `%2E.`, `__x` → `%5F_x`); the empty name
    is refused at write.
  - **Every reader decodes `zstd` and `numcodecs.shuffle`**, the wasm32
    build included (molrec's must-decode set). A build without
    `zarr-codecs` decodes `zstd` through a pure-Rust decoder (`ruzstd`), so
    the `zarr` feature now pulls `ruzstd` and `inventory`. It still cannot
    *encode* `zstd`. A store holding such an array — every `f64` column with
    a declared precision — cannot be read by molrs 0.14 or by any reader
    without the two codecs.
  - **Frame meta is typed on disk.** A `frame` / `system` group writes each
    meta value in its typed JSON form and adds the attribute `_meta_types`
    (`{key: tag}`), so a value reads back at its tag: an `i32` stays `i32`,
    an `f64x3` stays a vector, `1.0` stays `f64`. NaN and ±∞ are stored as
    `"NaN"` / `"Infinity"` / `"-Infinity"` (0.14 wrote JSON `null`, losing
    them), and a `u64`/`i64` beyond ±2⁵³ as a decimal string. A store without
    `_meta_types` is inferred as before. A meta key named `_meta_types` is
    refused at write, and a typed value in any other form is refused at
    read. The same forms are used for `sequence_schema` fills (a NaN fill is
    now storable; a `null` fill is refused) and for the `{dtype, value}`
    envelope of the `serde` / `stream` wire form.
  - **Declared precision.** An `f64` column may declare a precision `p`
    (`Block::set_precision`, Python `Block.set_precision`). Writers then
    store `round_half_even(x / q) · q` with `q` the largest power of two
    `≤ p` (error `≤ p/2`), pipe the column through `numcodecs.shuffle` +
    `zstd` level 3 (`gzip` level 1 without `zarr-codecs`), and record `p`:
    as the array attribute `precision` on `frame` / `system`, and in the
    column's `sequence_schema` entry on a trajectory. Readers hand back the
    stored values and the declaration (`Block::precision`). A trajectory
    compares the *rounded* values with the previous update, so a change
    below `q/2` writes no update; a frame stating a precision other than the
    pinned one is refused. Coordinates go from 24 to about 7.6 B/atom/frame
    at `p = 1e-3` Å. The default is unchanged: no precision, raw floats.

### Rust crate (`molcrafts-molrs`)

#### Cargo features

- **`default` is now `["rayon"]`**, so you get core only. It used to be
  `["full", "filesystem", "rayon"]`.
  ```toml
  molrs = { package = "molcrafts-molrs", version = "0.15", features = ["full", "filesystem"] }
  ```
- **Feature `ff` now enables `io`.**

#### Crate-root re-exports removed

| 0.14 | 0.15 |
|---|---|
| `molrs::{perceive_aromaticity, add_hydrogens, implicit_h_count, remove_hydrogens}` | `molrs::perceive::{aromaticity, hydrogens}::…` |
| `molrs::{RingInfo, find_rings, max_ring_system_size}` | `molrs::perceive::rings::…` |
| `molrs::{MatchOptions, Reaction, RingPrimitive, SmartsMatch, SmartsPattern}` | `molrs::perceive::smarts::…` |
| `molrs::{BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d, chiral_volume, find_chiral_centers}` | `molrs::perceive::stereo::…` |
| `molrs::compute_gasteiger_charges` | `molrs::ff::charge::compute_gasteiger_charges` |
| `molrs::smiles` | `molrs::io::smiles` |
| `molrs::Record` | `molrs::MolRec` |
| `molrs::SchemaValue` | `serde_json::Value` |

#### Store: column access

- **The typed getters are removed.** `Block::get_{float,int,uint,bool,u8,string}`
  and their `_mut` forms are gone. Use `Block::get(key) -> Option<&Column>` /
  `get_mut`, then a `Column::as_*` projection: `as_float`, `as_int` (i32),
  `as_uint` (u64), `as_bool`, `as_u8`, `as_string`, plus the new
  `as_i8/i16/i64/u16/u32/c64/c128`, each with a `_mut` variant.
  ```rust
  // 0.14
  let x = block.get_float("x").unwrap();
  let ids = block.get_uint_mut("id").unwrap();
  // 0.15
  let x = block.get("x").and_then(|c| c.as_float()).unwrap();
  let ids = block.get_mut("id").and_then(|c| c.as_uint_mut()).unwrap();
  ```
- **`BlockView::get_{float,…,string}` are removed.** Use
  `view.get(key) -> Option<&ColumnView>`, then `.as_float()` and so on, which
  return `ArrayViewD`.
- **Trait `ColumnAccess` is removed** (`as_*_view`, `nrows`, `dtype`,
  `shape`). Call the inherent `Column` / `ColumnView` methods instead.
- **`BlockAccess::get_*_view(key)` → `BlockAccess::column(key) -> Option<ColumnView>`.**
- **`FrameAccess::get_*(block, key)` → `FrameAccess::column(block, key)`.**
  ```rust
  frame.get_uint("bonds", "atomi")                          // 0.14
  frame.column("bonds", "atomi").and_then(|c| c.as_uint())  // 0.15
  ```
- **`Block::has_int` / `has_uint` cover every signed / unsigned width.**
  `as_int` / `as_uint` still return only i32 / u64 columns, so `has_int(k)`
  no longer guarantees that `as_int()` is `Some`. Match on `Column` or
  `dtype()` instead.
- **`Block::has_f32` is removed.**
- **`BlockError` has new variants** (`ValidityLength`, `MissingColumn`,
  `StackDtype`, `StackShape`), which breaks exhaustive `match`es.

#### Store: dtypes, meta, frame

- **f16/f32 removed.** `DType::{Float16, Float32}`, `Column::{Float16, Float32}`,
  `Column::{from_f16, from_f32, from_f16_holder, from_f32_holder, as_f16(_mut), as_f32(_mut)}`
  are gone. `BlockDtype` is no longer implemented for `f16` / `f32`, so call
  `mapv(f64::from)` before inserting.
- **f32 meta removed.** `MetaValue::{F32, F32x3, F32x6, F32x9}` and `as_f32`
  are gone; use `F64*` and `as_f64`. XTC/TRR `time` / `precision` meta are now
  `f64`.
- **`MetaMap` is an `IndexMap`.**
  - `iter()` returns `MetaIter`.
  - `impl IntoIterator for MetaMap` (by value) is removed; iterate `&meta`.
- **Ordered maps elsewhere.**
  - `Frame::into_inner` returns `IndexMap<String, Block>`.
  - `Frame::from_map` takes `impl IntoIterator<Item = (String, Block)>`.
  - `FrameView::from_parts` takes an `IndexMap`.
  - `Relation.props` is an `IndexMap`.
- **`store::frame::FRAME_SCHEMA_VERSION` is removed**, with no replacement.
- **`MolRec::extra_sections` is removed.** Unknown root sections are ignored
  on read, so nothing fills it. To read a producer's own frame-shaped section,
  call `io::mrec::read_frame_section_store(store, name)`.
- **`ObservableKind` gains `Other(String)`** and is no longer `Copy`.
  `ObservableKind::parse(s) -> Option<Self>` is replaced by
  `ObservableKind::from(s)`, which never fails, and `as_str` now returns a
  `&str` borrowed from the kind. Exhaustive `match`es must handle `Other`.

#### Schema and units

- **`keys::canonical_dtype(key)` → `schema::column(key).map(|s| s.dtype)`.**
- **`ColumnSpec.unit: &str` → `ColumnSpec.dimension: ColumnDim`.**
  `ColumnDoc` also gains a `dimension` field, which breaks struct literals.
- **`schema::Validator` is a unit struct.** `with_annotations`, `dtype_of` and
  `ViolationKind::AnnotationConflict` are removed.
- **`units::PRESET_DIMENSIONS` → `units::PresetDim::ALL`** (call `.name()` for
  the string).
- **`units::constants::PICOSECOND_S` is removed.**
- **The LJ preset temperature unit `lj_epsilon` is renamed `lj_epsilon_over_kB`.**

#### Math: `core::math` → `op`

- **`core::math::{norm3, cross3, mat3_mul_vec, det3, inv3, matmul}` are removed.**
  Use `molrs::op::vec3::{norm, cross, dot, …}` and
  `molrs::op::linalg::{det3, inv3}`. They take plain arrays
  (`Vec3 = [F; 3]`, `Mat3 = [[F; 3]; 3]`); convert from ndarray with
  `op::types::{to_vec3, to_mat3}`. For `matmul`, use ndarray `.dot()`.
- **`core::math::diagonalize` is removed** (`eigvals_sym_3x3`, `eigh_sym_3x3`,
  `eigh_largest_sym_4x4`). Use `op::linalg::{eigh_sym_3x3, eigh_sym_4x4}`; the
  largest 4×4 pair is the first one `eigh_sym_4x4` returns.
- **`ff::potential::geometry::{cross3, dot3, mag3}` → `op::vec3::{cross, dot, norm}`.**
- **`compute::density::kabsch` is removed** (`kabsch`, `det3`). Use
  `op::superpose::superpose(reference, target, weights, gap_tol)`.
- **`compute::environment::angular_separation::Quat` → `op::types::Quat`.**

#### Spatial

- **`geometry::rotate` returns `Result<(), MolRsError>`.** It refuses a zero
  or non-finite axis and a non-finite angle; it used to silently do nothing.
- **`geometry::align_direction` is removed.** Compose the replacement:
  ```rust
  if let Some((axis, angle)) = molrs::op::rigid::alignment(from_dir, to_dir) {
      geometry::rotate(mol, axis, angle, Some(from))?;
  }
  geometry::translate(mol, [to[0] - from[0], to[1] - from[1], to[2] - from[2]]);
  ```
- **`SimBox::try_new` is removed.** Use `SimBox::new` (same arguments) or
  `SimBox::from_matrix(h, origin, pbc)`.
- **`SimBox::new_cell(h, …, cell_defined: false)` ignores `h`.** The box
  carries the identity matrix, so `h_view()` / `matrix()` return the identity
  and a singular `h` is no longer an error. A defined cell is unchanged.
- **Image flags are `I` (i32), not `i64`.** This affects
  `SimBox::{wrap_shifts, images, unwrap}`, `periodic::ghosts::…::forward_comm`,
  `MDState.images`, and the `wrap_shifts` argument of `ForceProvider::compute` /
  `compute_into` and the `md::pairs` `advance` / `refresh` methods. Custom
  `ForceProvider` impls must update.

#### Molecular graphs (`core::system`)

- **`system::mapping` is removed**: `CGMapping`, `WeightScheme`,
  `coarsen_with_templates`, `backmap` and their root re-exports. The nearest
  replacements are `perceive::Coarsener` (coarsening),
  `CoarseGrain::from_atom_frame` and `builder::Assembler` (backmapping). They
  are not 1:1.
- **Graph methods now return `Result`.**
  - `MolGraph::add_node_with` returns `Result<NodeId, MolRsError>`.
  - `MolGraph::merge`, `Atomistic::merge` and `CoarseGrain::merge` return
    `Result<…, MolRsError>`. A property conflict is now an error.
  - `Atomistic::to_frame` and `CoarseGrain::to_frame` return
    `Result<Frame, MolRsError>`.
- **`from_frame` / `read_frame` refuse what they used to drop.** That covers a
  relation block missing an endpoint column, a relation prop whose dtype
  conflicts with the kind, and a registered relation block the receiver lacks.
  Null cells in nullable columns stay null; they used to become `0`/`""`/`false`.
- **`MolGraph::node_table_mut`, `molgraph::Group` and `GroupId` are removed.**
- **`Topology::from_frame` returns `TopologyError`.** It returned `MolRsError`;
  `TopologyError` converts with `From`, so `?` still works.
- **`Topology::add_angles` is removed.** Loop over `add_angle`.
- **`Element::is_early_atom` is removed.**

#### I/O

- **`io::reader::{open_txt, open_gz}` are removed.**
- **`io::zarr::schema::validate_trajectory(t)` is removed.** Call
  `t.validate()?` and `frame.validate()?` on each frame.
- **`io::data::lammps_data::lammps_type_ids_from_frame` →
  `core::store::type_labels::TypeLabels::from_frame(&frame)?`.**
- **PDB.** `io::data::pdb::{ModelRecord, parse_model_record, is_ter, HetAtmRecord}`
  are removed. `parse_hetatm_record` returns `Option<AtomRecord>`.
- **SMILES errors.** `SmilesError::new(kind, span, input)` →
  `SmilesError::new(kind, span, input, notation)`, and `SmilesError` gains a pub
  `notation` field. `SmilesErrorKind` gains descriptor and `Cg*` variants,
  which breaks exhaustive `match`es.
- **`smiles::AtomSpec` gains a pub `descriptors` field**, which breaks struct
  literals.
- **`LAMMPSDataReader` refuses unknown sections.** Opt out per section with
  `.with_skipped_section("Ellipsoids")`.
- **A LAMMPS dump `type` field holding type labels reads as `type`.** A
  numeric `type` field is still the `type_id` column; a non-numeric one
  (`dump_modify … types labels`) is now the string `type` column, where 0.14
  stored the strings under `type_id`. `write_lammps_dump` writes the `type`
  field from `type_id` when the frame has it and from the `type` labels
  otherwise, never both, and puts it after `id`. It formats each value from
  its column's stored dtype and refuses a column a dump field cannot hold
  (complex, more than one value per row, a string that is empty or contains
  whitespace) instead of writing a blank or placeholder.
- **`stream::Publisher::send_async` → `send`.**
- **`MetaValue::to_attr_value` is removed.** Use
  `MetaValue::to_typed_json` (typed JSON form) and
  `MetaValue::from_typed_json(tag, value)`; `from_attr_value` stays, as the
  inference of an untyped value. `MetaValue::from_json_value` now reads the
  typed JSON payload forms (`"NaN"`, decimal strings beyond 2⁵³).
- **`io::mrec::write_trajectory_file(path, trajectory)` →
  `write_trajectory_file(path, trajectory, meta)`.** `meta` is an
  `Option<&JsonMap>`, like `write_frame_file` and `write_system_file`. Pass
  `None` to keep the old behaviour.

#### Force field: construction

- **Removed from `ForceField`:**
  `def_{atom,bond,angle,dihedral,improper,pair}style`,
  `with_{atom,bond,angle,pair}style`, `def_type(category, style, name, params)`,
  `styles_mut`, `remove_style`, `subset(&Frame)` (no replacement) and
  `to_potentials` / `to_typed_potentials` (see the next subsection).
- **Removed from `Style`:** `def_{atom,bond,angle,dihedral,improper,pair}type`,
  `try_def_type`, the panicking `def_type(name, &[(&str, f64)])` and
  `to_potential` / `to_typed_potential`.
- **The two remaining entry points** are
  `ForceField::def_style(category, name, Params) -> Result<&mut Style, DefError>`
  and `Style::def_type(name, endpoints, Params) -> Result<&mut Style, DefError>`.
  - Endpoints are explicit; the name is never split on `-`.
  - The number of endpoints follows the category: atom 0, pair 1 or 2, bond 2,
    angle 3, dihedral and improper 4.
  ```rust
  // 0.14
  ff.def_bondstyle("harmonic").def_bondtype("a", "b", &[("k", 100.0), ("r0", 1.2)]);
  ff.def_pairstyle("lj/cut", &[]).def_pairtype("a", None, &[("epsilon", 0.1), ("sigma", 3.0)]);
  // 0.15
  ff.def_style("bond", "harmonic", Params::new())?
      .def_type("a-b", &["a", "b"], Params::from_pairs(&[("k", 100.0), ("r0", 1.2)]))?;
  ff.def_style("pair", "lj/cut", Params::new())?
      .def_type("a", &["a"], Params::from_pairs(&[("epsilon", 0.1), ("sigma", 3.0)]))?;
  ```
- **`Style.name` / `.params` / `.defs` are private.** Read them with `name()` /
  `params()` / `defs()`; write params with `set_param` / `set_str_param`.
- **`Style::rename_type` returns `Result<bool, DefError>`** (it returned `usize`).
- **`DefTypeError` is renamed `DefError`.**
  - New variants: `UnknownStyle`, `TypeConflict`, `StyleConflict`,
    `UnitsConflict`, `SpecialBondsConflict`, `Name`.
  - It derives `PartialEq` but no longer `Eq`.
- **Units and special_bonds are declared state.** `units()` defaults to
  `"real"`; `declared_units()` / `declared_special_bonds()` return `None`
  until something declares them.

#### Force field: compiling

- **`ff.to_potentials(&frame)` → `PotentialCompiler::new(&ff).compile(&frame)`**,
  and `to_typed_potentials` → `compile_typed`
  (`molrs::ff::potential::PotentialCompiler`).
- **`ff::potential::extract_coords(&frame)` → `frame.coords()?`**, which
  returns N×3; flatten it if you need the old layout.
  **`write_coords` → `frame.set_coords(view)`.**
- **`Potentials::members_mut` is removed.**

#### Force field: typing

- **The `Typifier` trait now requires exactly two methods:**
  `r#match(&self, &mut Atomistic) -> Result<Match, String>` and
  `library(&self) -> &ForceField`. `type Mol` and `fn typify` are gone.
  `Typing<T>` runs a typifier and owns the output force field.
- **The inherent `typify()` / `ff()` methods are removed** from
  `MMFF94Typifier`, `MMFF94STypifier`, `OPLSAATypifier` and `UFFTypifier`.
  Wrap the typifier in `Typing` instead. `AtdTypifier` and
  `BCCAtomChargeTypifier` are wrapped the same way.
  ```rust
  // 0.14
  let t = MMFF94Typifier::new();
  let typed = t.typify(&mol)?;
  let pots = t.ff().to_potentials(&typed.to_frame())?;
  // 0.15
  let mut typing = Typing::new(MMFF94Typifier::new());
  let typed = typing.typify(&mol)?;
  let frame = typed.to_frame()?;
  let pots = PotentialCompiler::new(typing.forcefield()).compile(&frame)?;
  ```
  `typing.forcefield()` holds only the types assigned so far; use
  `typing.library()` to get the whole library.
- **GAFF is a typifier.** `ff::gaff_forcefield`, `ff::forcefield::gaff`,
  `GaffError` and `MissingTerm` are removed. `GaffParameterSet` still resolves
  at `ff::`.
  ```rust
  let (typed, ff) = gaff_forcefield(set, &atd_typed)?;                    // 0.14
  let mut g = Typing::new(GaffTypifier::new(set));                        // 0.15
  let typed = g.typify(&atd_typed)?;
  let ff = g.forcefield();
  ```
- **OPLS.**
  - `OPLSAATypifier::oplsaa()` returns `Self`; it returned `Result<Self, String>`.
  - `LayeredTypingEngine::typify` → `assign`.
  - Removed: the `opls::typing` module (`typify_atoms`),
    `opls::{typify_bonded, typify_bonded_with}`, `LAYER_PRIORITY_STRIDE` and
    `OplsTypingMeta::priorities()`. Ranking is now pairwise override dominance.
  - `ff::params::OplsAtomRow` loses `def`, `overrides`, `priority` and `layer`.
    The typing rules are now `OplsRuleRow` in `ff::params::OPLSAA_TYPING`.
    `OplsAtomRow.class` is now the GROMACS `bond_type`.
  - Strict typing errors on any untyped atom; it used to return `Ok` with the
    atom left untyped.
- **Label formats.**
  - MMFF `stbn_type` is `{sbt}_{i}_{j}_{k}`.
  - MMFF torsions found on restart are `{tt}@{sec}_…`.
  - UFF bonded labels use `TypeName` with `@` bond orders.
- **`ff::mmff::MmffMolProperties::is_setup_complete` is removed.**

#### Force field: readers and writers

- **GROMACS.**
  - `ff::read_gromacs_top_ff(path)` → `GromacsTopFfReader::new().read(path)`.
  - `ff::write_gromacs_top_ff(path, ff, p)` →
    `GromacsTopFfWriter::new().with_precision(p).write(ff, path)`;
    `write_gromacs_top_ff_str` → `.write_str(ff)`.
  - `GromacsTopFfReader` loses its public fields and `include_dirs`. Use
    `with_include`, and `with_skipped_directive(name)` to read past a section.
- **LAMMPS.**
  - `LammpsFfReader::with_default_units` is removed; units come from the file
    and default to `"real"`.
  - `LammpsFfWriter::new()` → `LammpsFfWriter::new(&labels)`, and
    `with_options(opts)` → `with_options(&labels, opts)`, where
    `labels = TypeLabels::from_frame(&frame)?`. The writer is now
    `LammpsFfWriter<'a>`.
  - `LammpsWriteOptions` loses `atom_types` … `improper_types` and `type_ids`.

#### Perception, compute, optimize, builder

- **Perception returns `Result`.** `perceive::hydrogens::{add_hydrogens, remove_hydrogens}`
  and `Perceive::find_hydrogens` return `Result<Atomistic, MolRsError>`.
- **Rotatable bonds take a policy.** `detect_rotatable_bonds(g)`,
  `detect_rotatable_bonds_with_downstream(g)` and `Perceive::find_rotatable(mol)`
  take a second argument, `UnknownBondPolicy`.
  `UnknownBondPolicy::NotRotatable` is the default.
- **`compute::util` helpers removed.** `{get_f_slice, get_positions, mic_disp, PositionSlices}`
  are gone; use `get_positions_ref` and `MicHelper::from_simbox(..).disp(..)`.
- **`VanHoveResult::self_second_moment` is removed.**
- **`optimize::SoftSpec::with_{sigma, repulsion, rcut, bond_k, angle_k}` are removed.**
  The defaults are now fixed.
- **`builder::WalkOutput.paths: Vec<Vec<F3>>` → `traces: Vec<Trace>`.** Read
  the points with `.points()`.
- **`conformer::distgeom::experimental_torsions_with_provenance` is removed.**
- **`ScaleLjError` gains `InvalidMass`**, which breaks exhaustive `match`es.

### Python (`molrs`)

#### Modules and top-level names

- **Removed modules:** `molrs.frame`, `molrs.views`, `molrs.ff.forcefield` and
  `molrs.ff.potential.soft`.
  - `Block`, `Frame`, `Atomistic`, `CoarseGrain`, the view classes and
    `NodeRef` / `RelationRef` / `Refs` / `RelationBuckets` are native classes
    at `molrs.<Name>`.
  - `ForceField`, `Style`, `Type` and the `*Style` / `*Type` classes are at
    `molrs.ff.<Name>`.
  - Pickles that name `molrs.frame.*` or `molrs.views.*` will not unpickle.
  - `SoftPotential` has no replacement.
- **Removed top-level names:** `molrs.FRAME_SCHEMA_VERSION` and `molrs.GraphViews`.
- **Rigid-body functions are now chainable methods** on `Atomistic` and
  `CoarseGrain`:
  - `molrs.translate(mol, d)` → `mol.translate(d)`
  - `molrs.rotate(mol, axis, angle)` → `mol.rotate(axis, angle, about=None)`
  - `molrs.scale(mol, s)` → `mol.scale([s, s, s], about=None)`
- **`molrs.align_direction` is removed.** Compose `rotate` and `translate`, or
  use `molrs.op.superpose`.
- **`molrs.schema.VOCAB_VERSION` is still exported**, but its value is now 2.
- **`ColumnSpec` changed.** It gains a `dimension` field before `unit`, which
  shifts positional construction. Schema `uint` keys are now `uint64` (they
  were `uint32`).

#### Block

- **`Block` is no longer a `MutableMapping`.**
  - Removed: `get`, `items`, `values`, `pop`, `update`, `setdefault`, `clear`.
  - Kept: `keys()`, iteration, `in` and `b[key]`.
- **Removed typed accessors.** `get_f32` / `get_f64` / `has_f32` are gone. Use
  `b[key]`, and `has_f64` / `has_int` / `has_uint` / `has_string` /
  `dtype(key)` to check the type.
- **Other removals:**

  | 0.14 | 0.15 |
  |---|---|
  | `b.view(key)` | `b[key]` |
  | `b.to_dict()` | `{k: b[k] for k in b}` |
  | `Block.from_dict(d)` | `Block(d)` |
  | `b.sort_(key)` (in place) | `b = b.sort(key)` |
  | `b.iterrows()`, `b.itertuples()` | removed |
  | `Block(vars_, nrows, shape)` | `Block(data=None)`, then `.resize(n)` / `.set_shape(s)` |
- **Indexing changed.** `b[i]` with an int now raises `TypeError` (it returned
  a row dict). `b[callable]` is removed.
- **`b.shape` changed.** It was `(nrows, ncols)`; it is now `[nrows]`, or the
  N-D grid shape.
- **Tuple-key write spreads columns.** `b["x", "y", "z"] = arr` now spreads an
  `(N, k)` array over k columns. It used to store one column named
  `"('x', 'y', 'z')"`.

#### Frame and meta

- **Removed typed accessors:** `has_f32` / `has_f64` / `get_f32` /
  `get_f64(block, key)`. Use `frame[block][key]`.
- **`to_dict()` and the `blocks` property are removed.** Use
  `[frame[k] for k in frame.keys()]`.
- **`Frame(blocks, meta, box)` takes `meta` and `box` as keyword-only
  arguments.** Passing an existing `Frame` to `Frame(...)` is removed; use
  `.copy()`.
- **JSON meta comes back frozen.** A JSON object in `frame.meta` is now a
  frozen `MetaDocument`, not a `dict`, and arrays come back as tuples.
  ```python
  frame.meta["run"]["step"] = 3            # 0.14; raises in 0.15
  doc = frame.meta["run"].copy(); doc["step"] = 3; frame.meta["run"] = doc
  ```
- **`Atomistic.to_frame` / `CoarseGrain.to_frame` raise `ValueError`** on a
  dtype conflict.

#### Graph views

- **`Atom.is_virtual` → `isinstance(atom, molrs.VirtualSite)`.**
- **`Angle` / `Dihedral` / `Improper` `itom` … `ltom` and `CGBond.ibead` /
  `jbead` → `.endpoints`.** `Bond.itom` / `jtom` remain.
- **`CoarseGrain.del_bead(b)` → `cg.despawn(b.handle)`.**
- **The `nodes` property is removed** (it came from `GraphViews`).
- **`RelationBuckets` keeps only `exact_bucket(cls)`.** Custom relation view
  classes can no longer be registered.
- **`Refs` is no longer a `list` subclass.** Use `list(refs)`.
- **`NodeRef` loses `setdefault` / `pop`.**
- **`NodeRef`, `RelationRef` and `Refs` cannot be constructed directly.**
- **Constructors take fewer arguments.** `Atomistic(...)` /
  `CoarseGrain(...)` take only `**props`, and `Graph()` takes no arguments.
- **`replicate` grows `self` in place:**
  ```python
  w = mol.replicate(n)                                     # 0.14
  w = molrs.Atomistic()                                    # 0.15
  w.replicate(mol, np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)),
              np.arange(n, dtype=np.int32))
  ```
- **Integer props outside the int32 range raise `OverflowError`.** They used
  to wrap silently.

#### Box, MD, units

- **`Box.matrix` → `Box.h`.**
- **Image flags are int32.** This applies to `Box.images`,
  `Box.unwrap(images=)` and `MD` `images`. int64 input is refused; use
  `.astype(np.int32)`.
- **`UnitRegistry.Unit(expr)` / `.Quantity(v, expr)` → `.parse(expr)` /
  `.quantity(v, expr)`.**
- **`molrs.md.MD.set_forcefield` requires a `ForceField`.**

#### `molrs.io`

- **Trajectory functions renamed:**

  | 0.14 | 0.15 |
  |---|---|
  | `read_trr`, `read_xtc` (eager list) | `read_trr_trajectory`, `read_xtc_trajectory` (lazy; `list(...)` for a list) |
  | `write_dcd`, `write_trr`, `write_xtc`, `write_lammps_traj` | `write_{dcd,trr,xtc,lammps}_trajectory` |
  | `read_gro` → `list[Frame]` | `read_gro` → first `Frame`; `read_gro_trajectory` for all |
  | `lammps_type_ids_from_frame` | removed; the force-field writers take the frame |
- **Removed parameters.** The reserved `frame=` parameter is gone from
  `read_xyz`, `read_pdb`, `read_mol2`, `read_lammps_data`,
  `read_lammps_molecule`, `read_lammps_trajectory`, `read_amber_inpcrd`,
  `read_amber_prmtop`, `read_top` and `read_xsf`. `atom_style=` is gone from
  `read_lammps_data` / `write_lammps_data`.
- **`molrs.io.raw` renames:**
  - `read_lammps` / `write_lammps` → `read_lammps_data` / `write_lammps_data`
  - `read_lammps_traj` / `write_lammps_traj` → `*_lammps_trajectory`
  - `read_dcd` / `read_trr` / `read_xtc`, and the `write_*` forms → `*_trajectory`
  - `read_chgcar_file` / `read_cube_file` / `write_cube_file` →
    `read_chgcar` / `read_cube` / `write_cube`
- **`molrs.io.mrec` renames:**
  - `read_frame` / `write_frame` → `molrs.io.read_mrec` / `write_mrec`
  - `read_system` / `write_system` → `read_mrec_system` / `write_mrec_system`
  - `read_trajectory` / `write_trajectory` → `read_mrec_trajectory` /
    `write_mrec_trajectory`
  - `read_meta` → `read_mrec_meta`
  - `sections` → `mrec_sections`
- **`GroFieldFormatter` is removed.** The native GRO reader and writer use
  the canonical `res_name` / `name`, so `molrs.io.read_gro` /
  `write_gro` pass frames through unchanged and `molrs.io.raw.read_gro*`
  return canonical names too.
- **`meta=` takes what `frame.meta` hands out.** `write_mrec`,
  `write_mrec_system`, `write_mrec_trajectory` (which gains `meta=`) and
  `TrajectoryWriter(meta=)` accept a `dict`, a `MetaDocument`, or any mapping
  (`frame.meta` included), with nested tuples and documents. 0.14 took only a
  `dict` of lists and dicts, and raised `TypeError` for anything else.
- **A non-finite float inside a JSON meta document raises `ValueError`.**
  `frame.meta["doc"] = {"t": float("nan")}` used to store `null`. A
  top-level `float("nan")` is an `f64` value and is kept.
  `SequenceSchema.declare_meta_with_fill` infers an untagged fill the way
  `frame.meta` does, so a NaN fill is an `f64` fill.
- **A labelled LAMMPS dump `type` field reads as `atoms["type"]`**, not
  `atoms["type_id"]`; see the Rust crate's I/O entry, which also covers
  `write_lammps_trajectory`.
- **Trajectory `time` has no unit.** The `TrajectoryReader.time` and
  `TrajectoryWriter.append(time=)` docs no longer say fs. A record does not
  store a unit for time; the producer's convention applies.

#### `molrs.ff`

- **Styles are defined through one method.**
  `def_{atom,bond,angle,dihedral,improper,pair}style(name, …)` →
  `def_style(category, name, params=None)`. Also removed:
  `def_style(StyleInstance)`, the unbound `Style()`, `BondHarmonicStyle`,
  `AngleHarmonicStyle`, `DihedralOPLSStyle`, `PairCoulLongStyle` and
  `Parameters`.
- **`Style.def_type` takes the type name first,** and its endpoints must be
  `AtomType` handles; strings raise `TypeError`.
  ```python
  # 0.14
  ff.def_atomstyle("full").def_type("c"); ...
  ff.def_bondstyle("harmonic").def_type("c", "h", k=340.0, r0=1.09)
  ff.def_pairstyle("lj/cut", cutoff=10.0).def_type("c", epsilon=0.1, sigma=3.4)
  # 0.15
  ats = ff.def_style("atom", "full")
  c, h = ats.def_type("c"), ats.def_type("h")
  ff.def_style("bond", "harmonic").def_type("c-h", c, h, k=340.0, r0=1.09)
  ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type("c", c, epsilon=0.1, sigma=3.4)
  ```
- **Removed `ForceField` methods:** `rename_type`, `remove_type`,
  `remove_style`, `subset(frame)` and `map_type` (no replacements);
  `def_*type`, `def_type(category, …)`, `style_params`, `types`,
  `type_endpoints`, `set_type_param` and `set_type_str_param`.
- **Renamed or reshaped `ForceField` members:**
  - `style_names()` → `[(s.category, s.name) for s in ff.styles]`
  - `special_bonds_lj` / `special_bonds_coul` are removed; declare the
    triples with `set_special_bonds(lj, coul)`
- **`ForceField(name, units="real")` → `units=None`.** It declares no units
  unless given; the `units` getter still reports `"real"`.
- **`merge` carries more and fails loudly.** It now carries `special_bonds`,
  style params and units, and raises `ValueError` on a conflict instead of
  keeping the first definition.
- **`Type` changes.**
  - `Type.params` is a plain `dict` (no `.kwargs` / `.args`).
  - `Type[k]` returns `None` for a missing key.
  - `*Type.matches()` is removed.
- **Compiling moved to `PotentialCompiler`:**
  - `ff.to_potentials(frame)` → `molrs.ff.PotentialCompiler(ff).compile(frame)`
  - `ff.to_potentials(None)` → `PotentialCompiler(ff).defer()`
  - `ff.to_typed_potentials(frame)` → `PotentialCompiler(ff).compile_typed(frame)`
- **`extract_coords(frame)` → `frame.coords.ravel()`**, or pass the frame to
  `Potentials` directly.
- **The `*_str` readers and writers are removed**: `read_forcefield_xml_str`,
  `read_opls_xml_str`, `read_lammps_forcefield_str`, `read_amber_prmtop_ff_str`,
  `read_gromacs_top_ff_str`, `write_gromacs_top_ff_str` and
  `write_forcefield_xml_str`. Use the path-based functions, which accept
  `PathLike`. `write_lammps_forcefield_str` remains.
- **LAMMPS writers take the frame.** The new signatures are
  `write_lammps_forcefield(path, ff, frame, *, precision, skip_pair_style, skip_units, units)`,
  `write_lammps_forcefield_str(ff, frame, …)` and
  `write_lammps_data_coeffs(ff, frame, *, precision, units)`. `frame` is
  required, the options are keyword-only, and `atom_types` … `improper_types` /
  `type_ids` are removed.
- **Typifiers implement `match`.** A `molrs.ff.typifier.Typifier` subclass
  implements `match(graph) -> Match`. Defining `typify` in a subclass raises
  `TypeError`.
  - `typify` returns a typed copy.
  - `forcefield()` returns the output (the types assigned so far, as a copy);
    use `library()` for the full library.
  - The native typifiers cannot be subclassed.
  - `molrs.ff.typifier.TGraph` is removed.
  ```python
  t = molrs.ff.MMFF94Typifier()
  typed = t.typify(mol)
  pots = molrs.ff.PotentialCompiler(t.forcefield()).compile(typed.to_frame())
  ```

### WASM / JS (`@molcrafts/molrs`)

- **Block typed accessors are removed:**

  | 0.14 | 0.15 |
  |---|---|
  | `hasF32`, `hasF64`, `hasI32`, `hasU32`, `hasStr` | `has(key)` + `dtype(key)` |
  | `getF32`, `getF64`, `getI32`, `getU32`, `getStr` | `get(key, fallback?)` |
  | `copyColF`, `copyColI32`, `copyColU32`, `copyColStr` | `copy(key)` |
  | `viewColF`, `viewColI32`, `viewColU32` | `view(key)` |
  | `setColF`, `setColI32`, `setColU32`, `setColStr` | `set(key, data, shape?)` |
  | `createColF`, `createColI32`, `createColU32` | `set(key, new Float64Array(n), shape)` |
  - `set` infers the dtype from the typed array. It refuses `Float32Array`,
    plain `number[]` and mixed arrays.
  - `view` / `copy` return the column's own typed array; an i64 column comes
    back as a `BigInt64Array`.
  - `dtype(key)` throws for a missing key (it returned `undefined`).
- **`Block.nrows()` → `Block.nrows`** (a getter).
- **`Block.shape` changed.** The old no-argument `shape()` is now the getter
  `structuralShape` (`number[] | undefined`). `shape(key)` returns one
  column's shape, and `setShape` takes `number[]`.
- **Frame methods renamed:**

  | 0.14 | 0.15 |
  |---|---|
  | `getBlock(k)` → `undefined` if absent | `get(k)`, throws if absent; check with `has(k)` |
  | `insertBlock(k, b)` (moves `b`) | `set(k, b)` (deep copy; `b` stays usable) |
  | `removeBlock(k)` | `remove(k)` |
  | `blockNames()` | `keys()` (insertion order) |
  | `frame.renameColumn(block, old, new)` | `frame.get(block).renameColumn(old, new)` |
  | Frame-level `has*` / `get*` typed accessors | `frame.get(block).get(key)` |
- **`Mesh.verticesF32` / `faceNormalsF32` → `vertices` / `faceNormals`.**
- **`Wasm*Stream` classes changed.** `parseRangeInInput` returns a `Frame`
  that the caller must free.
  - Removed: `releaseFrame`, `blockCount`, `blockName`, `columnCount`,
    `columnName`, `columnDtype`, `columnLen`, `columnPtrF64`, `columnPtrU32`,
    `columnPtrI32`, `columnStrings`, `boxH`, `boxOrigin` and `boxPbc`.
  - Read the returned Frame with `keys()`, `get()`, `view()`, `copy()` and
    `.box` instead.
- **`schemaColumnDtype(key)` returns `"f64"` / `"i32"` / `"u64"`** (it returned
  `"float"` / `"int"` / `"uint"`). `schemaDocument()` keeps the old names.
- **Typifiers keep state.** `UFFTypifier`, `MMFF94Typifier` and
  `MMFF94STypifier` wrap a `Typing`.
  - `toPotentials` compiles only what `typify` assigned, so call `typify` first.
  - A conflicting re-typing throws.
- **Error prefixes changed:** `to_potentials:` → `toPotentials:` (typifiers) and
  `compile:` (`LBFGS.run`).
- **New throws.** SMILES → Frame, the conformer, perception, `typify`,
  `findHydrogens` and `removeHydrogens` can throw `"toFrame: …"` when a
  property contradicts the schema.

### C API (`molrs.h`)

- **`molrs_schema_column_dtype` names every dtype.** It answered `"string"`
  for any width outside `float`/`int`/`uint`/`bool`/`u8`; it now returns
  the dtype's own name (`"i64"` for `formal_charge`, `"u16"`, `"c64"`, …).

- **`MolrsDType` reports the stored variant.** Values 0–4 keep their names and
  numbers, but each now means exactly one type; 5–12 are new.

  | Value | Constant | 0.15 meaning | 0.14 meaning |
  |---|---|---|---|
  | 0 | `MOLRS_D_TYPE_FLOAT` | f64 | any float width |
  | 1 | `MOLRS_D_TYPE_INT` | i32 | i8/i16/i32/i64 |
  | 2 | `MOLRS_D_TYPE_BOOL` | bool (1 byte) | bool |
  | 3 | `MOLRS_D_TYPE_U_INT` | u64 | u8/u16/u32/u64 |
  | 4 | `MOLRS_D_TYPE_STRING` | string | string |
  | 5 | `MOLRS_D_TYPE_INT8` | i8 | — |
  | 6 | `MOLRS_D_TYPE_INT16` | i16 | — |
  | 7 | `MOLRS_D_TYPE_INT64` | i64 | — |
  | 8 | `MOLRS_D_TYPE_U8` | u8 | — |
  | 9 | `MOLRS_D_TYPE_U_INT16` | u16 | — |
  | 10 | `MOLRS_D_TYPE_U_INT32` | u32 | — |
  | 11 | `MOLRS_D_TYPE_COMPLEX64` | 2×f32 | — |
  | 12 | `MOLRS_D_TYPE_COMPLEX128` | 2×f64 | — |

  An i64 column no longer reports `INT`. A `switch` on
  `molrs_block_col_dtype` must handle 5–12.
- **Three dtype-agnostic accessors replace the nine typed ones.**
  `molrs_block_get_{F,I,U}`, `molrs_block_get_{F,I,U}_mut` and
  `molrs_block_copy_{F,I,U}` are replaced by `molrs_block_get`,
  `molrs_block_get_mut` and `molrs_block_copy`. The dtype is an
  out-parameter.
  ```c
  /* 0.14 */
  const double* x; size_t n;
  molrs_block_get_F(blk, key, &x, &n);
  /* 0.15 */
  const uint8_t* raw; size_t n; MolrsDType dt;
  molrs_block_get(blk, key, &raw, &n, &dt);
  if (dt == MOLRS_D_TYPE_FLOAT) { const double* x = (const double*)raw; /* … */ }
  ```
  - `out_len` is still an element count, but `molrs_block_copy`'s buffer size
    is now in **bytes**.
  - A string column is `MOLRS_STATUS_TYPE_MISMATCH`; in 0.14 it was
    `KEY_NOT_FOUND`. A missing key is still `KEY_NOT_FOUND`.
  - `molrs_block_set_{F,I,U}` and `molrs_block_col_commit` are unchanged.
- **`molrs_ff_def_{atom,bond,angle,pair}style` →
  `molrs_ff_def_style(ff, category, name, keys, vals, n)`.** It covers every
  category, including dihedral and improper. Conflicting params return
  `INVALID_ARGUMENT`.
- **`molrs_ff_def_type` takes explicit endpoints** (9 arguments; it took 7).
  ```c
  molrs_ff_def_type(ff, "bond", "harmonic", "CT-OH", pk, pv, 2);              /* 0.14 */
  const char* e[] = {"CT", "OH"};
  molrs_ff_def_type(ff, "bond", "harmonic", "CT-OH", e, 2, pk, pv, 2);        /* 0.15 */
  ```
  - The name is never split.
  - A missing style is `INVALID_ARGUMENT`; it used to be created.
  - A conflicting re-definition is `INVALID_ARGUMENT`; an identical one does
    nothing.
- **The force-field JSON format changed.**
  - `molrs_ff_from_json` requires `str_params` on every style and `endpoints`
    / `str_params` on every type. Unknown or missing keys are refused, so
    **JSON written by 0.14 is refused**.
  - `molrs_ff_to_json` writes the new shape, with `units` / `special_bonds`
    only when declared.
- **`molrs_ff_new` declares no units and no special_bonds.**
- **Removed:** `molrs_c_api_version()`, `MOLRS_C_API_VERSION` and
  `molrs_frame_schema_version()`, with no replacement.
- **f32 metadata removed.** `MolrsMetaType` loses `F32` (5), `F32x3` (13),
  `F32x6` (15) and `F32x9` (17); the remaining values are unchanged.
  `MolrsMetaValue` loses `f32_value` and `f32x9`, which changes its layout, so
  recompile.
- **`molrs_frame_meta_key(index)` uses insertion order**, not sorted order.
- **`molrs_frame_from_smiles` can return `MOLRS_STATUS_INTERNAL_ERROR`.**
- **`molrs_schema_vocab_version()` returns 2.**

### C++ (`molrs-cxxapi`)

- **Removed:** `cxx_api_version()`, `frame_schema_version()` and the
  `CXX_API_VERSION` file. Gate on `cxx_api_capabilities()` instead; its bits
  are unchanged.
- **f32 metadata removed.** `MetaType` loses `F32`, `F32x3`, `F32x6` and
  `F32x9`. Later values shift down (for example `F64` 6 → 5, `String` 7 → 6,
  `F64x9` 18 → 14), so stored integer values do not carry over. `MetaEntry`
  loses `f32_value` and `f32_values`.
- **`frame_meta_entries` returns insertion order**, not sorted order.

### `molrs-ffi`

- **No public items were removed.** The capsule names move to the `0.15` line;
  see [All surfaces](#all-surfaces).

### Also new in 0.15

- **CGsmiles:** `io::smiles::parse_cgsmiles` → `CGSmilesIR`, the fragment SMILES
  dialect, and Python `molrs.io.CGSmilesIR` with its record classes.
  `SmilesError` (Python `ValueError`) carries `kind` / `span` / `input` /
  `notation`.
- **Assembly:** the site-graph `builder::Assembler`, `SitePlacer`,
  `GrowthPlacer`, `AxisOrienter` and `perceive::Coarsener`; ports on every
  graph type; `SubgraphMatcher`; `Frame::subset`; and the `molrs.op` numeric
  base. The `Fragment` / `TracePlacer` types seen on the 0.15 development
  branch never shipped.
- **Force field:**
  - `ForceField::merge`, `empty_like`, `units` / `set_units`;
    `Typing<T>` / `Match`; `ElementTypifier`.
  - `core::store::type_labels::{TypeName, TypeLabels}`.
  - `lammps_coeff_params` / `lammps_coeff_values`; the frcmod writer.
  - `ForceField::to_section` / `from_section`: a force field as molrec's
    `forcefield` record section (units declared, never converted).
- **Force-field section:** `ForceFieldSection` (`store::forcefield_section`,
  with `style_block_name`, `parse_style_block_name`, `unit_preset`),
  `MolRec::forcefield`, and the doors `io::mrec::write_forcefield_file` /
  `read_forcefield_file`; a `*.mrec` carries a `forcefield/` group. Python:
  `molrs.io.mrec.ForceFieldSection`, `ForceField.to_section` /
  `ForceField.from_section`, `forcefield=` on `molrs.io.write_mrec` /
  `write_mrec_system`, and `molrs.io.write_mrec_forcefield` /
  `read_mrec_forcefield` (which returns the section, or `None`).
- **Store:** nullable columns (`insert_nullable`, `validity`; persisted in
  zarr); Python `MetaDocument`.
- **Aligned blocks:** `SequenceSchema::{declare_aligned, aligned_with}`
  (Python too).
- **Row references:** `Block::{set_target, target, clear_target, targets}`,
  `SequenceSchema::{declare_target, target}`, `schema::{RowReference,
  check_target, EndpointTarget}`; Python `Block.set_target` / `target` /
  `targets` (pickled), `SequenceSchema.declare_target` / `target`; WASM
  `Block.target`. `keys::units_preset`.
- **Declared precision:** `store::precision::{quantum, quantize,
  quantize_in_place, check_precision, PRECISION_MIN, PRECISION_MAX}`;
  `Block::{set_precision, precision, clear_precision, precisions}`;
  `SequenceSchema::{declare_precision, precision}`. Python
  `Block.set_precision` / `Block.precision` (pickled with the block) and
  `SequenceSchema.declare_precision` / `precision`; WASM `Block.precision`;
  C++ `frame_set_precision`.
