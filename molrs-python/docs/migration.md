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
- **A pickle names a class by its public Python path**, and most of those
  moved (see [Python paths](#python-paths)): a 0.15 pickle holding, say, a
  `molrs.Box` or a `molrs.Element` does not unpickle in 0.16. Nor does a
  pickle naming `molrs._lib.*` (a `Frame`, `Block`, `Atomistic` or
  `CoarseGrain` pickled by 0.15 or an earlier 0.16 build): the native module
  is `molrs._native` in 0.16.
- **Force-field JSON is not converted.** `molrs_forcefield_from_json` reads a
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
  helpers and the `½k` form maps are gone; `LammpsUnitConverter::scale(from, to)`
  returns the `UnitScale` that converts a parameter by its dimension
  (`UnitScale::apply`).
- **Readers of other engines convert to the force-field IR.** GROMACS:
  `k = k_b/2`, `k = k_θ/2`, degrees kept. OpenMM XML: `k/2` for bonds and
  angles, radians → degrees. AMBER prmtop: `k = RK`, `k = TK`, radians →
  degrees. The GAFF and OPLS-AA tables (`GaffTypifier`, `OplsAaTypifier`)
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
  `param`; Python `molrs.ff.ir.MissingParamError`) where 0.15 raised a plain
  `ValueError` with a message; a per-instance column a typifier did not bake
  (`kb` of `mmff_bond`, …) is `MissingParam` too. `BadValue` (`style`,
  `type`, `param`, `reason`; Python `molrs.ff.ir.BadValueError`) refuses a value
  of the wrong kind (text for a number, an array of another rank), text
  outside its declared choices (`mixing = "lorentz"`) or a value outside its
  domain (a non-integer `n` of `lj/cut`, `inner >= cutoff` of a CHARMM
  switch). An unlike pair of `buck` / `morse` with no cross row is
  `NoMixing`. Rust: `PairMmffVdwStyleParams::from_style` returns
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
(`molrs.io.mrec.MOLREC_VERSION`, Rust `molrs::io::mrec::MOLREC_VERSION`). In
version 2 the `forcefield` section is the force-field IR, so some stored
numbers mean something else than in the version-1 records molrs 0.15 wrote.
0.16 never reads a version-1 record as version 2: it converts every changed
number exactly on read, or refuses the record by name.

- **Readers accept versions 1 and 2 and refuse a newer one.** A store
  without `molrec_version` predates version 1 and is read by version 1's
  rules. `molrs.io.read_mrec_meta` (and `MolRec.meta` in Rust) hands `meta` back as
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
  opened for appending (`MrecWriter.open` on an existing 0.15 store: read it
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
  array style param are refused by `validate` / `ForceFieldSection::from_forcefield`.
  `ForceFieldSection.to_forcefield` reads a `cmap` style and every category beyond
  the seven (0.15 refused both); see
  [The force-field IR as a protocol](#the-force-field-ir-as-a-protocol).
- **`pair14` is no category.** molrec retired it; `category_arity("pair14")`
  is `None`, and a `pair14` table is kept as
  unknown content (no arity or restatement check). No reader produced it.
- **`ForceFieldSection.validate` refuses a `pair lj/charmm` `one_four`**
  other than `"regular"` / `"epsilon14"`, so `molrs.io.read_mrec_forcefield` refuses
  such a record before anything turns it into a `ForceField`.

Every 0.15 test record (`molrs/src/io/mrec/zarr_storage/testdata/v1`, written by the
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
  `*_constructor` (0.15: `*_ctor`) return `Result<ForceTerm, CompileError>`
  (0.15: `Result<Member, String>`); a custom
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
  `ff::ir::ENDPOINT_COLUMNS` is `[&str; 5]`
  (`itom` … `ltom`, `mtom`), and `category_arity("cmap")` is `Some(5)`
  (0.15: `None`). A `cmap` style table must carry all five endpoint columns.
- **Frame vocabulary.** The canonical key `atomm` (`u64`, fifth relation
  endpoint) and the block `cmaps` (relation of arity 5, optional `type`,
  `type_id`, `style`) are new; `subset`, `replicate` and the validator
  renumber and range-check `atomi` … `atomm`. `keys::ENDPOINTS` (Rust
  `[&str; 5]`, Python `molrs.core.keys.ENDPOINTS`) gains `atomm`, so a block
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
- **Rust: `LammpsForcefieldWriteOptions` has a `cmap_file` field** (`Option<String>`,
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
- **Python.** `molrs.ff.forcefield.CmapStyle` / `CmapType` (with `itom` … `mtom`);
  `def_style("cmap", …)` returns a `CmapStyle`.
- **C API.** `"cmap"` is a category; `molrs_schema_column_dtype("atomm")`
  is `"uint"`.
- **New.** `assign_cmaps(frame, ff)` builds the `cmaps` block from a frame's
  dihedrals and the field's cmap rows (forward matching only);
  `read_lammps_cmap_str` parses a `fix cmap` file into a `LammpsCmapFile`
  (its `UNITS:` tag and raw maps) and `LammpsForcefieldReader::read_cmap_str`
  reads it into a force field, rows named `"1"` … `"K"`;
  `LammpsForcefieldWriter::write_cmap_str` and `write_lammps_cmap_str` write one (CHARMM's
  own file comes back line for line); `CmapGrid` / `CmapCharmm` are the
  kernel. Python: `molrs.ff.typifier.assign_cmaps`,
  `molrs.io.read_lammps_cmap_forcefield` / `write_lammps_cmap_forcefield`.

### Array parameters

- **Rust: `Params` holds `f64` arrays** beside numbers and strings
  (`set_array`, `get_array`, `iter_arrays`); `==` and `same_parameters`
  compare them exactly, so two definitions differing only in an array are a
  conflict. Code that copies a `Params` key by key through `iter()` and
  `iter_strings()` loses the arrays.
- **Python.** A param value may be an array: `def_type(**params)`,
  `def_style(params=…)` and `ForceFieldType.__setitem__` take a numpy array or a
  nested list / tuple of numbers and store float64, so a list value that
  raised `TypeError` in 0.15 is now stored. `params` and `t[key]` return
  arrays as float64 numpy arrays; pickles carry them.
- **C API.** `molrs_forcefield_to_json` writes an optional `array_params` object (on
  a style or a type, only when it holds an array param: nested lists, one
  level per axis), and `molrs_forcefield_from_json` reads it and refuses a ragged or
  non-numeric one.
- **Records.** An array parameter is a `f64[T, S…]` column (see
  [Records](#records-molrec_version-2)); it round-trips through
  `ForceFieldSection.from_forcefield` / `to_forcefield`, a `*.mrec` store and
  `molrs.io.mrec`.

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
  value; `molrs::core::schema::PAIR_OVERRIDE_COLUMNS`). Explicit values are
  final and the scales replace the `special_bonds` weight (or `w`).
  Precedence: override > `w` > `special_bonds`. The LAMMPS data-file writer
  and the Python LAMMPS force-field writers refuse a frame carrying them.
- **One exceptions kernel at both doors.** Every override pair and every
  `w > 0` pair is priced by one more member (`PairExceptions`, an indexed
  member). `compile` drops the override rows from the `pairs` list the pair
  styles see; `compile_typed` reads those rows and weights them 0 in every
  pair member.
- **Rust: `WeightedTerm` (0.15 `TypedMember`) is
  `(ForceTerm, Option<PairWeights>)`** (0.15: `(Member,
  Option<BondDistanceWeights>)`). Build the MD weights with
  `w.special_weights(&topo)`, which returns `ff::potential::SpecialWeights` (0.15:
  `topo.special_weights(&w)`, which still exists for a
  `BondDistanceWeights` but misses the pairs an override weights 0);
  `PairWeights::by_distance()` is the old table.

### Torsions

- **`dihedral nharmonic`** (LAMMPS's `Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ`, params `a1..aN`,
  contiguous, N ≥ 1) is a new style: kernel, LAMMPS reader
  (`dihedral_coeff t N A1 … AN`) and writer. A gap (`a1`, `a3` without
  `a2`) or a missing `a1` is refused at compile time and by the writer.
- **Rust: `molrs::ff::ir::torsion` is new** — the exact maps
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
  each row), as LAMMPS's `write_data` does, so `read_lammps_data_coeffs`
  reads a non-`harmonic` section back under the right style.
- **`read_lammps_data_coeffs` takes the frame.** The signature is
  `read_lammps_data_coeffs(frame, *, units=None)` with the frame
  `molrs.io.read_lammps_data` returned: it reads the frame's
  `meta["lammps_coeffs_text"]` and names each row's type id by the label the
  file's `* Type Labels` section gave it (the reader's
  `meta["<kind>_type_labels"]`, ids as written). `units` defaults to the
  data file's own (its `write_data` title line), else `"real"`; a `units`
  that disagrees with the file's raises `ValueError`, as does a frame with no
  `* Coeffs` sections. The 0.15 form — the coefficient text plus
  `units` and `atom_labels` … `improper_labels` maps the caller parsed out of
  the meta — is removed:
  ```python
  # 0.15
  ff = read_lammps_data_coeffs(frame.meta["lammps_coeffs_text"], units="real",
                               atom_labels={1: "c3", 2: "hc"}, ...)
  # 0.16
  ff = molrs.io.read_lammps_data_coeffs(frame)
  ```
  In Rust, `LammpsForcefieldReader::read_data_coeffs(&frame, units: Option<&str>)`
  replaces `read_data_coeffs(text, &LammpsTypeLabelMaps, units)`;
  `LammpsTypeLabelMaps` is no longer public. The meta keys are
  `core::keys::{LAMMPS_COEFFS_TEXT, LAMMPS_UNITS}`, and
  `TypeLabels::declared_ids(frame, block)` gives a block's inventory with
  its ids as written.
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
  style has every parameter scaled by its `ParamDimension` (`E*L^6`, `1/L`, …), not per
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
  `LammpsForcefieldWriteOptions::skip_pair_style`) keeps `special_bonds` and
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

The OpenMM reader (`OpenmmXmlReader`, `read_openmm_xml_forcefield`; 0.15
`OplsXmlReader` / `read_opls_xml`, and `read_forcefield_xml`, which
dispatched to it) and writer (`OpenmmXmlWriter`,
`write_openmm_xml_forcefield`; 0.15 `XmlForceFieldWriter` /
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
  `coul/charmm` has `coulomb = 332.06371329919216` (OpenMM's `ONE_4PI_EPS0`,
  `OPENMM_ONE_4PI_EPS0` in kJ·nm·mol⁻¹·e⁻², in `real` units),
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
- **`write_openmm_xml_forcefield(path, ff, precision)`: `precision` is optional**
  (`Option<usize>` in Rust, also for `write_openmm_xml_forcefield_str` and
  `OpenmmXmlWriter::with_precision`); the default writes each number in the shortest
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
  `coul/charmm` has `coulomb` = 332.06371329919205 kcal·Å·mol⁻¹·e⁻²,
  `GROMACS_ONE_4PI_EPS0` (GROMACS's `ONE_4PI_EPS0`, 138.93545764438196
  kJ·nm·mol⁻¹·e⁻², CODATA 2018) in `real` units, not LAMMPS `real`'s: Coulomb
  energies of a GROMACS-read field are 9.9·10⁻⁹ larger than in 0.15, and
  equal GROMACS's.
- **Whole topologies read: `GromacsTopForcefieldReader::read_system` / Python
  `molrs.io.read_gromacs_top_system`** read the molecule sections too, into the
  force field and a typed frame (0-based indices): GROMACS's own type lookup,
  rows with their own parameters as types `<labels>@gmx_<n>`, `[ pairs ]`
  rows with parameters as per-pair overrides, the nrexcl pair list (every
  pair of two molecules too, up to `MAX_ATOMS_FOR_A_FULL_PAIR_LIST` atoms,
  so `compile` prices them as GROMACS does), exclusions, constraints and
  settles, `[ molecules ]` repeated. The force-field reader still refuses
  molecule sections, now naming `read_system` (0.15 pointed at
  `molrs.io.read_top`, which reads structure only, 1-based).
- **Whole topologies written: `GromacsTopForcefieldWriter::write_system_str(ff,
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

The prmtop readers (`read_amber_prmtop`, `AmberPrmtopForcefieldReader` /
`read_amber_prmtop_forcefield`) read what they refused, and the frame and the force
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
  the one value), and the frame `AmberPrmtopForcefieldReader::read_system` / Python
  `molrs.io.read_amber_prmtop_system` returns (with the force
  field, as `read_gromacs_top_system` does) gains a `pairs` block — only when
  some pair is weighted otherwise —
  listing those 1-4 pairs with `coul_scale` / `lj_scale` cells (the
  structure reader alone, `read_amber_prmtop`, has none). It is not a pair list: `intramolecular_pairs` builds the
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
  another LAMMPS name, `ir.StyleDeclaration.lammps`, or afterwards
  `ir.register_engine_form("lammps", category, name, form)`) is read and
  written by the LAMMPS reader and writer with nothing else written:
  `params` in order, each converted by its `ParamDimension`. Python
  `StyleSpec.lammps` names a
  style's form.
- **Every engine refusal is `NoEngineForm`.** The GROMACS and frcmod
  writers refuse a style that is not built in, and the LAMMPS and OpenMM
  writers a style with no form, as "`<engine>` has no form for `<category>`
  `` `<style>` ``: …", typed: `ForceFieldWriter::write_str` / `write` (and
  `write_amber_frcmod` / `write_amber_frcmod_str`, `write_openmm_xml_forcefield` /
  `write_openmm_xml_forcefield_str`, `GromacsTopForcefieldWriter::write_system_str`,
  `lammps_coeff_values`, the LAMMPS writer's `write_data_coeffs_str` /
  `write_cmap_str`) return **`ForceFieldWriteError`**
  instead of `String`: it dereferences to its message (so
  `err.contains(…)` still reads it, and `String::from(err)` converts) and
  `err.ir()` is the `IrError::NoEngineForm` when an engine refused a style.
  Python raises `molrs.ff.ir.NoEngineFormError` (a `ValueError`) from every
  writer, as from `register_engine_form`.
- **Readers and writers take a registry**: `LammpsForcefieldReader::with_registry`,
  `LammpsForcefieldWriter::with_registry`, `OpenmmXmlWriter::with_registry`
  (the process-wide one by default).

### GAFF and GAFF2

- **`GaffTypifier` in Python**: `molrs.ff.typifier.GaffTypifier(
  parameter_set="gaff" | "gaff2")`, the Rust
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

- **Rust: a typifier's `TypeAssignment` (0.15 `Match`) carries any relation
  kind.** The fixed fields `bonds`, `angles`, `dihedrals` and `impropers` are
  gone; `TypeAssignment::links`
  (`IndexMap<String, Vec<Annotations>>`) maps a graph relation kind (the
  Frame block of its category: `"bonds"`, a custom `"urey_bradleys"`) to
  rows positional against that kind's own rows. Replace `m.bonds = rows` with
  `*m.link_mut("bonds") = rows` (or `m.links.insert(..)`), `m.bonds.push(a)`
  with `m.link_mut("bonds").push(a)`. A type under a kind defines a type of
  the category whose block the kind is (`typifier::link_category`); a
  non-empty vector for a kind the graph lacks is an error naming it.
  `write_onto` defines the kinds in the graph's registration order, as
  before for the four built-ins.
- **Rust: `TypeAssignment::assign_terms(graph, kind, library, key)`** (new) types
  every row of a relation kind against the library's type rows of its
  category by the atoms' types: slot by slot with wildcards, in the orders
  the category's `EndpointOrder` allows (reversible, ordered or unordered),
  fewest wildcards first, then table order (molrec's rule). It returns the
  positions nothing matched.
- **Python: `TypeAssignment(nodes, links=...)` (0.15 `Match`) takes a kind
  name as a key** as well
  as a relation class (`{Bond: rows, "urey_bradleys": rows}`); a relation
  class other than `Bond` / `Angle` / `Dihedral` / `Improper` / `Port`, or
  a key that is neither class nor `str`, still raises `TypeError`, and
  naming one kind twice (`{Bond: …, "bonds": …}`) `ValueError`.
  `repr(TypeAssignment)` lists the kinds: `TypeAssignment(nodes=2,
  links={bonds=1}, styles=0, pairs=0)`.

### The force-field IR as a protocol

The force-field IR is a protocol (`molrs::ff::ir`, Python `molrs.ff.ir`,
new in 0.16): categories and styles are registrations of one form, the
built-ins sealed among them; see
[Extending the force-field IR](guides/extending-forcefield-ir.md). What
changes for code written against 0.15:

- **Rust: `register_kernel` / `register_kernel_with` are removed**; register
  through the IR registry, `molrs::ff::ir::register_style(StyleSpec::new(category,
  name).source(source), Some(Kernel::constructor(ctor)))` (0.15: `()`, overriding
  whatever was there). A built-in is sealed (`IrError::Sealed`); registering
  the same constructor again is a no-op, another one under a taken name is
  `IrError::Conflict`. To change a built-in's behaviour, register a style of
  another name.
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
    `ForceFieldSection::to_forcefield` does with a record's endpoint
    columns). New:
    `Style::arity`, `StyleDefs::arity`, `ForceField::get_relationtypes`.
  - **`ForceFieldSection::to_forcefield` keeps a category beyond the seven** (it
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
  - **C API.** `molrs_forcefield_def_style` and `molrs_forcefield_def_type` accept the same
    categories as `ForceField::def_style`.
- **Custom styles persist.** A custom style or category is stored in a
  `*.mrec` record as its molrec style entry, and a process that registered
  nothing reads it back:
  - `ForceFieldSection::from_forcefield` writes a registered custom style's
    `expression` when the
    style has none of its own (a built-in is written with none; a style's
    own `expression` is written, and read back, byte for byte). Rust:
    `ForceFieldSection::from_forcefield_in(&ff, &Registry)`.
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
  `ForceField`, `fit_form` taking a `molrs::ff::ir::FitMetric`), over the form
  families `torsion` (canonical `dihedral periodic`), `bond`, `angle`
  (canonical `harmonic`) and `lj` (canonical `pair lj/cut`); see
  [Converting between forms](guides/forcefield-ir.md#converting-between-forms).
  An out-of-image conversion is refused (`IrError::OutOfImage`, Python
  `molrs.ff.ir.OutOfImageError`, a `ValueError`).

### Module ownership

Every module has one job, and every public symbol has exactly one path. Rust
paths that moved or went away, by subsystem:

#### One path per symbol

The rule, applied crate-wide:

- **The crate root holds subsystems only.** The core data model is one
  module, `molrs::core`, with every public name flat on it
  (`molrs::core::Frame`, `molrs::core::SimBox`, `molrs::core::Atomistic`,
  `molrs::core::Quantity`, `molrs::core::MolRsError`); there is no
  `molrs::store`, `system`, `spatial`, `math`, `units` or `error` any more
  (see [One core](#one-core-molrscore-and-molrscore)). Nothing is flattened
  to the root: not core's types (`molrs::Frame`, `molrs::Element`, …), not
  the builders (`molrs::GrapheneBuilder`, …), and not `molrs::VERSION`
  (use `env!("CARGO_PKG_VERSION")` of the crate you build, or Python
  `molrs.__version__`).
- **A facade re-exports its implementation files, which are private.** Where
  a module re-exports a child, the child is private and the facade carries
  its whole public API; a child module stays public only as a namespace whose
  items nothing re-exports (`ff::potential::pair`, `core::keys`,
  `core::schema`, `core::constants`,
  `ff::params::atomtype_amber`, `io::pdb`, …).
- **One owner per symbol.** Re-exports of another module's items are gone
  (`spatial::region::FNx3`, `io::mrec::schema::MOLREC_VERSION`,
  `ff::typifier::opls::BondedTerm`, the `ff::params::ATOMTYPE_*` copies).

| 0.15 | 0.16 |
|---|---|
| `molrs::Frame`, `Block`, `FrameAccess`, `FrameView`, `ForceFieldSection`, `MetaMap`, `MetaValue`, `MolRec`, `Trajectory`, … (crate root, `molrs::core::…`, `molrs::store::frame::Frame`, `store::block::Block`, …) | `molrs::core::{Frame, Block, …}` |
| `molrs::Atomistic`, `Element`, `MolGraph`, `NodeId`, `RelationId`, `Topology`, `CoarseGrain`, … (crate root, `molrs::system::{atomistic, molgraph, topology, coarsegrain, bond, bond_weights, extract, graph_hash, link, port}::…`) | `molrs::core::{Atomistic, Element, …}` (also `FromMolGraph`, `TopologyError`) |
| `molrs::SimBox`, `BoxKind`, `Mic`, `CenterError` (crate root), `molrs::spatial::{simbox, geometry, mesh, periodic, trace}::…` | `molrs::core::{SimBox, Mic, BoxKind, BoxError, TriMesh, DEGENERATE_AREA2, GhostSet, ImageRange, Trace, CenterError}`; the `MolGraph::{translate, rotate, scale, center}` methods |
| `molrs::spatial::neighbors::{aabb, bruteforce, filter, grid}::…`, `spatial::region::{region, cylinder, ellipsoid, half_space, polyhedron, sphere_union}::…` | `molrs::core::…`, `molrs::core::…` |
| `molrs::units::{dimension, error, preset, quantity, registry, unit}::…` (and the crate-root `Unit`, `UnitRegistry`, …) | `molrs::core::…` |
| `molrs::math::virial::Virial` | `molrs::core::Virial` |
| `molrs::types::{F, F3, FNx3, …, I, Idx, Pbc3}` (also `molrs::core::types`) | `molrs::op::{F, F3, Fnx3, …}` (flat), the one owner of the scalar and array aliases |
| `molrs::store::schema::consts::…` | `molrs::core::keys::…` |
| `molrs::store::schema::{block, column, document, validator, violation}::…` | `molrs::core::schema::…` |
| `molrs::GrapheneBuilder`, `CarbonTubeBuilder`, `Assembler`, … | `molrs::builder::…` |
| `molrs::compute::<family>::X`, `compute::<family>::<file>::X` (`compute::order::Nematic`, `compute::distribution::AtomGroups`, `compute::dynamics::persist::pair_survival_tcf`, `compute::dielectric::compute_dipole_moment`, …) | `molrs::compute::X` — now also the `*Args` aliases, `EinsteinDiffusionResult`, `InternalCoordinate` (was `AnyObservable`), `Observable` and the distribution observables, `VORONOI_BOUNDARY`, `steinhardt_qlm`; see [Wave S4](#wave-s4-analysis-perception-geometry-dynamics) for the renames |
| `molrs::ff::potential::<family>::<file>::X` (`pair::lj_cut::LJCut`, `bond::harmonic::BondHarmonic`, `kspace::pme::PmePotential`, …) | `molrs::ff::potential::<family>::X`, renamed in [Wave S3](#wave-s3-force-field) (`pair::PairLjCut`, …; also `pair::{PairMmffVdwAtomParams, PairMmffVdwStyleParams, lj_ab_to_sigma_epsilon}`, `angle::AngleCharmmParams`, `kspace::PairCoulLongPmeParams`) |
| `molrs::ff::potential::{compile, error, instances}::…` | `molrs::ff::potential::{PotentialCompiler, CompileError, ExplicitTerms}` |
| `molrs::ff::ir::{category, dim, engine, error, expression, form, registry, spec}::…`, `ff::ir::engine::positional` | `molrs::ff::ir::…`, `molrs::ff::ir::positional` |
| `molrs::ff::params::{gaff, gaff2, gaff_equiv, gaff_empirical, bccparm, bccparm_abcg2, clpol, gasparm, oplsaa, oplsaa_typing}::…` | `molrs::ff::params::…` (`GAFF`, `OPLSAA_ATOMS`, …; a table's row arrays, `GAFF_BONDS`, …, are reached through its table) |
| `molrs::ff::params::ATOMTYPE_AMBER`, … | `molrs::ff::params::atomtype_amber::ATOMTYPE_AMBER`, … (each beside its own `RULES` / `WILDATOMS`) |
| `molrs::ff::typifier::{am1bcc, atd, element, estimate, gaff, opls, uff}::…` (`typifier::gaff::GaffParameterSet`, `typifier::opls::OplsTypingMeta`, `typifier::estimate::Provenance`, …) | `molrs::ff::typifier::…` (`GaffParameterSet`, `OplsTypingMetadata`, `Provenance`, …; `cmap` and `mmff` stay namespaces) |
| `molrs::ff::typifier::opls::Estimator` (a trait alias) | `ParameterInterpolator<Term = BondedTerm>` |
| `molrs::io::format::{read_frame, write_frame, FrameFormat}` | removed: every door names its format (see [Wave S2](#wave-s2-io-per-format)) |
| `molrs::io::smiles::{smiles, chem::ast, error}::…` | `molrs::io::smiles::…` (see [Wave S2](#wave-s2-io-per-format)) |
| `molrs::io::log::lammps::…`, `molrs::io::mesh::stl::…` | `molrs::io::lammps::…` (the records), `molrs::io::read_lammps_log`, `molrs::io::read_stl` |
| `molrs::io::mrec::schema::{MOLREC_VERSION, RESERVED_META_KEYS}` | `molrs::io::mrec::{MOLREC_VERSION, RESERVED_META_KEYS}` |
| `molrs::io::mrec::{FrameSequence, FrameSequenceWriter}` | `molrs::io::mrec::{MrecReader, MrecWriter}` (Python `molrs.io.mrec.MrecReader` / `MrecWriter`, WASM `MrecReader`) |
| `molrs::io::log::parse_lammps_log_text` | `molrs::io::read_lammps_log_str` |
| `molrs::md::{error, forces, integrators, maxwell, pairs, types}::…` | `molrs::md::…`; `com_velocity` and `kinetic_energy` are `molrs::compute::{center_of_mass_velocity, kinetic_energy}` |
| `molrs::optimize::lbfgs::…`, `perceive::{builder, subgraph}::…`, `signal::{acf, grid, window}::…`, `stream::message::…` | `molrs::optimize::…`, `molrs::perceive::…` (the builder is gone: [Wave S4](#wave-s4-analysis-perception-geometry-dynamics)), `molrs::signal::…` (also `SignalError`), `molrs::stream::…` |

No longer public (each had no user outside the crate; reached before only
through a file module): the pair styles' `*_typed_ctor`,
`improper::cvff::signed_cosine_ctor`, `cmap::charmm::GRID`,
`ff::ir::{LAMMPS_STYLE_CATEGORIES, builtin_forms}`,
`store::forcefield_section::{ANNOTATION_COLUMNS, CMAP_GRID, ENDPOINT_COLUMNS,
MIXING_RULES, ONE_FOUR_VALUES, UNIT_QUANTITIES, category_arity,
check_pair_restatements, is_parameter_column, unit_preset}`,
`system::port::PORTS`, the transport helpers `apply_unbiased_norm`,
`component_means`, `gradient_axis0_order2`, `unbiased_cartesian_acf_scaled` (gone: `compute::autocorrelation` is the one ACF),
`compute::environment::angular_separation::angular_distance`,
`conformer::distgeom` (private as a whole), and the typifier internals
`estimate::{Candidate, CandidateSet, DEFAULT_IMPROPER, is_wildcard,
substitution_table, empirical::{angle_k, angle_theta0, bond_k}}`,
`opls::{CandidateTables, NoMatch, deps::OplsDependencyAnalyzer,
layered::{LayeredTypingEngine, MAX_CIRCULAR_ITERATIONS}}`. Removed outright:
`store::forcefield_section::parse_style_block_name` (unused) and
`store::meta::META_TYPES_ATTR` (private to the `*.mrec` adapter, its one user).

#### One core: `molrs::core` and `molrs.core`

The data model is one module in every language. Rust's `molrs::core` and
Python's `molrs.core` hold every public name flat; the implementation files
are private. Three vocabularies stay submodules: `core::keys`,
`core::schema` and `core::constants` (Python `molrs.core.keys`,
`molrs.core.schema`, `molrs.core.constants`). The simulation cell is
`SimBox` in Rust only (Rust reserves `Box`); Python, JS, C and C++ say
`Box`. Builds of the 0.16 line before this change spelled the paths in the
left column.

| Earlier 0.16 builds | 0.16 |
|---|---|
| `molrs::store::X`, `molrs::system::X`, `molrs::spatial::X`, `molrs::units::X`, `molrs::math::X`, `molrs::error::MolRsError` | `molrs::core::X`, `molrs::core::MolRsError` |
| `molrs::spatial::neighbors::X`, `molrs::spatial::region::X` | `molrs::core::X` |
| `molrs::math::{complex::Complex, spherical_harmonics::ylm_all, wigner3j::wigner_3j, wigner_d::wigner_d_matrix, …}` | `molrs::core::{Complex, ylm_all, wigner_3j, wigner_d_matrix, …}` |
| `molrs::store::{type_labels, precision}::X` | `molrs::core::X` |
| `molrs::store::typed_json` | crate-private (`typed_json::{encode_complex, decode_complex}` removed: unused) |
| `molrs::store::keys`, `molrs::store::schema`, `molrs::units::constants` | `molrs::core::keys`, `molrs::core::schema`, `molrs::core::constants` |
| `molrs::units::{lookup_preset, preset_names, register_preset, replace_preset}` | `molrs::core::{lookup_unit_preset, unit_preset_names, register_unit_preset, replace_unit_preset}` |
| `molrs::spatial::{translate, rotate, scale, center, CenterError}`; `Atomistic::{translate, rotate, scale, center}`, `CoarseGrain::{translate, rotate, scale, center}` (Rust) | `MolGraph::{translate, rotate, scale, center}` over `as_molgraph()` / `as_molgraph_mut()`, `molrs::core::CenterError` (Python keeps the methods) |
| `molrs::system::BondType` | `molrs::core::BondOrder` (the chemical bond class; the `bond_type` key is unchanged) |
| `molrs::compute::{BondOrder, BondOrderResult}`, Python `molrs.compute.BondOrder`, JS `BondOrder` | `molrs::compute::{BondOrientationalOrder, BondOrientationalOrderResult}`, `molrs.compute.BondOrientationalOrder`, JS `BondOrientationalOrder` |
| `molrs::store::ColumnHolder`, `Column::from_<dtype>_holder` | `molrs::core::ColumnArray`, `Column::from_<dtype>_array` |
| `molrs::store::ObservableData` | `molrs::core::ObservableValues` |
| `molrs::system::entity_table::{Column, Cell}` | `molrs::core::{EntityColumn, EntityCell}` |
| `molrs::system::{Topology::find_rings, TopologyRingInfo}` | `molrs::perceive::{perceive_rings, RingInfo}` (the one ring perception; RingInfo keeps RDKit's name) |
| `molrs::compute::NodeId` and the unused graph variants of `ComputeError` | removed |
| `molrs::store::{ForceFieldSection, StyleEntry, EndpointKey, style_block_name, MolRec, Observables, MOLREC_VERSION, RESERVED_META_KEYS}` | `molrs::io::mrec::…` (feature `zarr`, which now enables `ff`) |
| `ForceField::to_section()`, `to_section_in(&registry)`, `ForceField::from_section(&section)`; Python `ForceField.to_section()` / `ForceField.from_section(s)` | `ForceFieldSection::from_forcefield(&ff)`, `from_forcefield_in(&ff, &registry)`, `section.to_forcefield()`; Python `molrs.io.mrec.ForceFieldSection.from_forcefield(ff)` / `section.to_forcefield()` |
| `store::forcefield_section::{ENDPOINT_COLUMNS, ANNOTATION_COLUMNS, CMAP_GRID, is_parameter_column, category_arity, MIXING_RULES, ONE_FOUR_VALUES}` | `molrs::ff::ir::{ENDPOINT_COLUMNS, ANNOTATION_COLUMNS, CMAP_GRID, is_parameter_column, category_arity}`, `molrs::ff::forcefield::combining_rule::COMBINING_RULES`, `molrs::ff::forcefield::one_four::ONE_FOUR_VALUES` |
| a section's `units.preset` of the LAMMPS styles only | the LAMMPS styles and `openmm` (`nm`, `kJ/mol`, `ps`) |
| `molrs::system::port::PORTS`, `perceive::equivalence::EQUIV_CLASS`, `perceive::bond_type::BCC_BOND_TYPE`, `io::data::lammps_bond_react::REACT_ID`, `io::data::lammps_data::{COEFFS_TEXT_META, UNITS_META}` | `molrs::core::keys::{PORTS, EQUIV_CLASS, BCC_BOND_TYPE, REACT_ID, LAMMPS_COEFFS_TEXT, LAMMPS_UNITS}`; also `FRAG_ID`, `VSITE`, `BEAD_ATOMS` (Python `molrs.core.keys.*`) |
| `molrs::ff::params::amber::{AMBER_SCEE, AMBER_SCNB}`, Python `molrs.ff.params.AMBER_SCEE` / `AMBER_SCNB` | `molrs::core::constants::{AMBER_SCEE, AMBER_SCNB}`, `molrs.core.constants.*` |
| `ff::constants::{MDYNE_A_TO_KCAL, VACUUM_DIELECTRIC, DEG2RAD}` (crate-private) | `molrs::core::constants::{KCAL_MOL_PER_MDYNE_ANGSTROM, VACUUM_DIELECTRIC}`; `f64::to_radians` |
| `molrs::ff::params::uff::G` (332.06) | `molrs::core::constants::UFF_COULOMB` |
| the spectroscopy literals (`c`, fs → s, m → cm, `1.438777`) | `molrs::core::constants::{SPEED_OF_LIGHT, SECOND_RADIATION_CONSTANT}` (`c₂ = 1.438776877` cm·K, CODATA 2018); fs → s and m → cm through the unit registry (`molrs::core::UnitFactor::new("fs", "s")`) |
| `molrs::VERSION` | `env!("CARGO_PKG_VERSION")`; Python `molrs.__version__` |
| Python `molrs.store`, `molrs.spatial`, `molrs.system`, `molrs.units` | `molrs.core` |
| Python `molrs.store.keys`, `molrs.store.schema` | `molrs.core.keys`, `molrs.core.schema` |
| Python `molrs.system.Graph` | `molrs.core.MolGraph` |
| Python `molrs.units.AMBER_COULOMB` | `molrs.core.constants.AMBER_COULOMB` (with every other constant) |
| Python `molrs.io.mrec.schema.MOLREC_VERSION` / `RESERVED_META_KEYS` | `molrs.io.mrec.MOLREC_VERSION` / `RESERVED_META_KEYS` |

A pickle names a class by its public path, so a `Box`, `Element`,
`MolGraph`, … pickled under `molrs.spatial` / `molrs.system` /
`molrs.store` does not unpickle, and neither does a `Frame`, `Block`,
`Atomistic` or `CoarseGrain` pickled as `molrs._lib.*` (0.15 and earlier
0.16 builds): the native module is `molrs._native` in 0.16.

#### Wave S2: io per format

`io` is one module per file format, not one per content kind:
`molrs::io::{data, trajectory, log, mesh, forcefield, streaming, zarr}` are
gone. A format's classes (readers, writers, indexers, records) live in
`io::<fmt>` / `molrs.io.<fmt>`; every door is a function at the top of
`io`, named after its format and the same in Rust and Python:
`read_<fmt>[_<what>]` / `write_<fmt>[_<what>]` for a path, `_str` for text in
memory, `_bytes` for bytes, `_trajectory` for every frame of a multi-frame
file. Family formats carry the family's name (`read_lammps_data`,
`read_amber_prmtop`, `read_vasp_poscar`, `read_gromacs_top_forcefield`).
**No door picks a format for the caller**: the extension-dispatched
`read_frame` / `write_frame` / `FrameFormat` and the `format=`-taking
frame-bytes doors are gone — call the format's own door. SMARTS is wholly
`perceive::smarts` / `molrs.perceive`, and SMILES and SMARTS share one
crate-private grammar, so `io` and `perceive` no longer depend on each other.
Builds of the 0.16 line before this change spelled the paths in the left
column.

**Modules and classes (Rust)**

| Earlier 0.16 builds | 0.16 |
|---|---|
| `io::data::pdb::{PDBReader, PDBWriter, PdbIndexBuilder}` | `io::pdb::{PdbReader, PdbWriter, PdbIndexBuilder}` (`PdbReader` is also a `TrajectoryReader`, `PdbReader::open(path)`) |
| `io::data::xyz::{XYZReader, XYZFrameWriter, XyzIndexBuilder}` | `io::xyz::{XyzReader, XyzWriter, XyzIndexBuilder}` (`XyzReader::open(path)`) |
| `io::data::gro::{GroReader, GroFrameWriter}` | `io::gro::{GroReader, GroWriter}` (`GroReader` is also a `TrajectoryReader`, `GroReader::open(path)`) |
| `io::data::sdf::{SDFReader, SdfIndexBuilder}` | `io::sdf::{SdfReader, SdfIndexBuilder}` |
| `io::data::mol2::{Mol2Reader, Mol2FrameWriter}`, `io::data::cif::{CifReader, CifFrameWriter}` | `io::mol2::{Mol2Reader, Mol2Writer}`, `io::cif::{CifReader, CifWriter}` |
| `io::data::poscar::{PoscarReader, PoscarFrameWriter}` | `io::vasp::{VaspPoscarReader, VaspPoscarWriter}` |
| `io::data::lammps_data::{LAMMPSDataReader, LAMMPSDataWriter, LammpsDataIndexBuilder}`, `io::trajectory::lammps_dump::{LAMMPSTrajReader, LAMMPSDumpWriter, LammpsDumpIndexBuilder}` | `io::lammps::{LammpsDataReader, LammpsDataWriter, LammpsDataIndexBuilder, LammpsDumpReader, LammpsDumpWriter, LammpsDumpIndexBuilder}` |
| `io::data::lammps_bond_react::{BondReactTemplate, BondReactSystem, DroppedRows}`, `io::log::{LammpsLog, …}` | `io::lammps::…` (the same names) |
| `io::forcefield::readers::lammps::{LammpsFfReader, LammpsCmapFile, LAMMPS_CMAP_DIM, LAMMPS_CMAP_MAX}`, `io::forcefield::writers::lammps::{LammpsFfWriter, LammpsWriteOptions, refuse_pair_overrides}` | `io::lammps::{LammpsForcefieldReader, LammpsCmapFile, LAMMPS_CMAP_DIM, LAMMPS_CMAP_MAX, LammpsForcefieldWriter, LammpsForcefieldWriteOptions, refuse_pair_overrides}` |
| `io::forcefield::lammps_units::{parse_style, LammpsFfUnits, LammpsLjReference}` | `io::lammps::{parse_lammps_units_style, LammpsUnitConverter, LammpsLjReference}` |
| `io::forcefield::readers::prmtop::AmberPrmtopFfReader`, `io::forcefield::writers::frcmod::AmberFrcmodFfWriter`, `io::data::prep::{PrepAtom, PrepResidue}` | `io::amber::{AmberPrmtopForcefieldReader, AmberFrcmodWriter, PrepAtom, PrepResidue}` |
| `io::forcefield::readers::gromacs::GromacsTopFfReader`, `io::forcefield::writers::gromacs::GromacsTopFfWriter` | `io::gromacs::{GromacsTopForcefieldReader, GromacsTopForcefieldWriter}` |
| `io::forcefield::readers::opls::OplsXmlReader`, `io::forcefield::writers::xml::XmlForceFieldWriter` | `io::openmm_xml::{OpenmmXmlReader, OpenmmXmlWriter}` |
| `io::forcefield::readers::clpol::AlphaFfRow` | `io::clpol::ClpolAlphaRow` |
| `io::forcefield::readers::ForceFieldReader`, `io::forcefield::writers::{ForceFieldWriter, WriteError}` | `io::reader::ForceFieldReader`, `io::writer::{ForceFieldWriter, ForceFieldWriteError}` |
| `io::trajectory::{dcd, trr, xtc}::{DcdReader, DcdWriter, DcdIndexBuilder, …}` | `io::{dcd, trr, xtc}::…` (the same names) |
| `io::trajectory::xdr`, `io::data::pdb::{AtomRecord, ConectRecord, Cryst1Record, parse_*_record, is_end, is_endmdl}`, `io::data::xyz::{Primitive, ExtValue, PropType, PropertySpec, XYZComment, parse_comment_line}`, `io::trajectory::dcd::{DcdHeader, ByteOrder, MarkerSize}`, `io::trajectory::trr::TrrHeader`, `io::data::prmtop::parse_flag_sections` | crate-private |
| `io::streaming::{FrameIndexEntry, FrameIndexBuilder}` | `io::frame_index::{FrameOffset, FrameIndexBuilder}` |
| `io::reader::validated`, `io::writer::check_before_write` | `io::reader::check_read_frame`, `io::writer::check_write_frame` |
| `io::writer::ToFrame`, `FrameWriter::write_from` | removed (`FrameWriter::write`; an `Atomistic` is `to_frame()` first) |
| `io::smiles::{SmilesIR, AtomNode, …, SmilesError, Notation}`; `io::smiles::{CGSmilesIR, CGGraph, CGNode, CGEdge, CGBondOrder, EdgeOrigin, CGFragmentDef, FragmentBody, PairEnd, ResolvedPair}` | `io::smiles::{SmilesIr, AtomNode, …, SmilesError, Notation}` (the IR, its AST nodes, `SmilesError`, `SmilesReader`, the emit options); `io::cgsmiles::{CgSmilesIr, CgGraph, CgNode, CgEdge, CgBondOrder, EdgeOrigin, CgFragmentDef, FragmentBody, PairEnd, ResolvedPair}` |
| `io::smiles::frame_reader::{SmilesReader, parse_atomistic}` | `io::smiles::SmilesReader` (`parse_atomistic` private) |
| `io::mrec::{schema, column_dtype}` | `io::mrec::{validation, dtype_from_schema_tag}` |
| `io::zarr` (crate-private) | `io::mrec::zarr_storage` (crate-private) |

**Doors (Rust and Python, one name)**

| Earlier 0.16 builds (Rust / Python) | 0.16 (`molrs::io::` / `molrs.io.`) |
|---|---|
| `io::read_frame`, `io::write_frame`, `io::FrameFormat` / `molrs.io.read_frame`, `write_frame` | removed: the format's own door (`read_pdb`, `read_vasp_poscar`, …) |
| `stream::{frame_to_bytes, bytes_to_frame}(…, MessageFormat)` / `molrs.io.read_frame_bytes(data, format=)`, `write_frame_bytes(frame, format=)` | `io::{write_msgpack_frame_bytes, read_msgpack_frame_bytes, write_json_frame_str, read_json_frame_str}` / `molrs.io.` the same names |
| `data::pdb::{read_pdb_frame, read_pdb_traj, parse_frame_bytes}`, `write_pdb_frame(W)`, `write_pdb_traj(W)` | `read_pdb`, `read_pdb_trajectory`, `read_pdb_bytes`, `write_pdb(path)`, `write_pdb_trajectory(path)` (`PdbWriter` over a stream) |
| `data::xyz::{read_xyz_frame, read_xyz_traj, parse_frame_bytes, write_xyz_frame(W), write_xyz_traj(W), read_xyz_frame_from_reader, parse_xyz_frame_str}` | `read_xyz`, `read_xyz_trajectory`, `read_xyz_bytes`, `write_xyz(path)`, `write_xyz_trajectory(path)`; the stream ones are `XyzReader` / `XyzWriter` |
| `data::gro::read_gro` (every frame), `read_gro_frame(R)`, `write_gro_traj`, `write_gro_frame(W)` / Python `read_gro` (first frame) | `read_gro` (the first frame), `read_gro_trajectory` (every frame), `write_gro_trajectory`; `GroReader` / `GroWriter` |
| `data::sdf::parse_frame_bytes`; `data::mol2::{read_mol2_all, write_mol2_frame(W)}`; `data::cif::{read_cif_all, write_cif_frame(W)}` | `read_sdf_bytes` (+ new `read_sdf`, `read_sdf_trajectory`); `read_mol2_trajectory`; `read_cif_trajectory`; `Mol2Writer` / `CifWriter` |
| `data::xsf::{read_xsf_from_reader, write_xsf_frame(W)}`, `data::cube::{read_cube_from_reader, write_cube_to_writer}`, `trajectory::cube_traj::{read_cube_trajectory, read_cube_trajectory_files}` | `read_xsf_str`, `write_xsf_str`, `read_cube_str`, `write_cube_str`, `read_cube_trajectory` (`read_cube_trajectory_files` removed: unused) |
| `data::poscar::{read_poscar, write_poscar, read_poscar_from_reader, write_poscar_to_writer}`, `data::chgcar::{read_chgcar, read_chgcar_from_reader}` / Python `read_chgcar` | `read_vasp_poscar`, `write_vasp_poscar`, `read_vasp_poscar_str`, `write_vasp_poscar_str`, `read_vasp_chgcar`, `read_vasp_chgcar_str` / Python `read_vasp_chgcar` (+ new `read_vasp_poscar`, `write_vasp_poscar`, `read_sdf`, `read_cif`, `write_cif`) |
| `data::prmtop::read_amber_prmtop_from_reader`, `data::inpcrd::{read_amber_inpcrd_from_reader, read_amber_inpcrd_into}` | `read_amber_prmtop_str`, `read_amber_inpcrd_str`, `amber::merge_inpcrd(&mut frame, read_amber_inpcrd(path)?)` (Python `read_amber_inpcrd(path, frame=)` unchanged) |
| `data::ac::{read_ac, parse_ac}`, `data::prep::{read_prep, parse_prep, format_prep, write_prep}` / Python `read_ac`, `read_prep`, `write_prep` | `read_amber_ac`, `read_amber_ac_str`, `read_amber_prep`, `read_amber_prep_str`, `write_amber_prep_str`, `write_amber_prep` / the same names |
| `forcefield::readers::prmtop::read_amber_prmtop_ff` / `read_amber_prmtop_ff`; Python `read_gromacs_top_ff`, `write_gromacs_top_ff` | `read_amber_prmtop_forcefield`; `read_gromacs_top_forcefield`, `write_gromacs_top_forcefield` |
| `forcefield::xml::read_forcefield_xml[_str]` (sniffed three layouts) / Python `read_forcefield_xml`, `read_opls_xml` | one door per layout: `read_openmm_xml_forcefield[_str]` (OpenMM's schema), `read_molrs_xml_forcefield[_str]` (molrs's own; new inverse `write_molrs_xml_forcefield[_str]`), `read_mmff_xml_forcefield[_str]` (an MMFF parameter set) |
| `forcefield::writers::xml::write_forcefield_xml[_str]` / Python `write_forcefield_xml` | `write_openmm_xml_forcefield[_str]` |
| `forcefield::xml::{read_opls_typing_xml_str, read_mmff_params_xml_str}` | `read_openmm_xml_opls_typing_str`, `read_mmff_xml_params_str` |
| `forcefield::readers::lammps::read_lammps_cmap_str`, `forcefield::writers::lammps::lammps_cmap_str` | `read_lammps_cmap_str`, `write_lammps_cmap_str` |
| Python `read_lammps_cmap(path)`, `write_lammps_cmap(path, forcefield, frame)` (a `ForceField`, unlike Rust's `read_lammps_cmap_str`, which returns the file's raw `LammpsCmapFile`) | `read_lammps_cmap_forcefield`, `write_lammps_cmap_forcefield` (`LammpsForcefieldReader::read_cmap_str` / `LammpsForcefieldWriter::write_cmap_str`); `read_lammps_cmap_str` / `write_lammps_cmap_str` stay the raw `fix cmap` grids |
| `forcefield::readers::clpol::{read_alpha_ff, parse_alpha_ff}` | `read_clpol_alpha`, `read_clpol_alpha_str` |
| `data::lammps_data::parse_frame_bytes`, `trajectory::lammps_dump::{read_lammps_dump, write_lammps_dump, open_lammps_dump, parse_frame_bytes}` | `read_lammps_data_bytes`, `read_lammps_dump_trajectory`, `write_lammps_dump_trajectory`, `LammpsDumpReader::open`, `read_lammps_dump_bytes` |
| `data::lammps_molecule::{read_lammps_molecule (by extension), write_lammps_molecule(…, format)}` / Python `write_lammps_molecule(path, frame, format=)` | `read_lammps_molecule` / `write_lammps_molecule` (native text), `read_lammps_molecule_json` / `write_lammps_molecule_json` |
| `data::lammps_bond_react::write_bond_react_map` / `write_bond_react_map` | `write_lammps_bond_react_map` |
| `log::{read_lammps_log(path), read_lammps_log_with_style(path, style)}` | `read_lammps_log(path, style)` |
| `trajectory::{dcd, trr, xtc}::{read_X, write_X, open_X, parse_frame_bytes}`, `dcd::{parse_frame_with_header, parse_frame_with_decoder_context}` | `read_X_trajectory`, `write_X_trajectory`, `XReader::open(path)`, `read_X_bytes`; `read_dcd_bytes(bytes, context)` |
| `mesh::{read_stl, parse_stl}` | `read_stl`, `read_stl_bytes` |
| `csv::{block_from_csv, block_to_csv}` / Python `read_block_csv(source)`, `write_block_csv(block, path=None)` | `read_csv_block(path)`, `read_csv_block_str(text)`, `write_csv_block(path, block)`, `write_csv_block_str(block)` / the same names |
| `mrec::{read_frame_file, write_frame_file, read_system_file, write_system_file, read_trajectory_file, write_trajectory_file, read_forcefield_file, write_forcefield_file, read_meta_file, read_record_file, write_record_file}` / Python `read_mrec`, `write_mrec` (Structure) | `read_mrec_frame`, `write_mrec_frame`, `read_mrec_system`, `write_mrec_system`, `read_mrec_trajectory`, `write_mrec_trajectory`, `read_mrec_forcefield`, `write_mrec_forcefield`, `read_mrec_meta`, `read_mrec`, `write_mrec` (a whole `MolRec`) / Python `read_mrec_frame`, `write_mrec_frame` |
| `mrec::{read_record_store, write_record_store, read_frame_section_store, section_names_store}` | `read_mrec_storage`, `write_mrec_storage`, `read_mrec_frame_storage`, `mrec::section_names_storage` |
| `mrec::{pack, open_packed, open_trajectory_sequence}`, `MrecReader::open(store)` / Python `molrs.io.mrec.pack`, `molrs.io.mrec.schema` | `mrec::{pack_mrec_zip, open_mrec_zip}`, `MrecReader::open(path)`, `MrecReader::from_storage(store)` / `molrs.io.mrec.pack_mrec_zip`, `molrs.io.mrec.validation` |
| `MrecWriter::{create(store, …), create_at(path, …), open(store), open_at(path)}` | `MrecWriter::{create_in_storage(store, …), create(path, …), from_storage(store), open(path)}` |
| `smiles::{parse_smiles, parse_fragment_smiles, to_atomistic, fragment_to_atomistic, from_atomistic, validate_smiles, read_smiles}` / Python `read_smiles` | `SmilesIr::{parse, from_fragment}`, `ir.to_atomistic()`, `ir.to_atomistic_with_descriptors()`, `SmilesIr::from_atomistic`, `ir.validate(text)`, `read_smiles_str` / Python `read_smiles_str` |
| `smiles::{write_atomistic_smiles, write_smiles, write_fragment_smiles}` / Python `SmilesIR.write_smiles()` | `write_smiles_str(mol, &options)` / `molrs.io.write_smiles_str(mol, **flags)` (an IR's own text writers are crate-private) |
| `smiles::parse_cgsmiles` | `CgSmilesIr::parse`; new `read_cgsmiles_str` (the molecule, its lowest level expanded) |
| `smiles::{parse_smarts, write_smarts, write_local_smarts, local_smarts_ir, LocalSmartsOptions, NeighborStyle}` / Python `molrs.io.write_smarts(mol, center, …)`, `SmilesIR.write_smarts()` | `perceive::smarts::{SmartsPattern::from_environment(mol, center, &EnvironmentOptions), NeighborStyle}` and `SmartsPattern`'s `Display` / `molrs.perceive.SmartsPattern.from_environment(mol, center, …)` and `str(pattern)` |

**Python modules**

| Earlier 0.16 builds | 0.16 |
|---|---|
| `molrs.io.trajectory.TrajectoryReader` (a generic concatenator over native `LAMMPSTrajReader`, `DCDTrajReader`, `XYZTrajReader`, `TRRTrajReader`, `XTCTrajReader`) | the format's own lazy reader, what `read_<fmt>_trajectory` returns: `molrs.io.pdb.PdbReader`, `molrs.io.xyz.XyzReader`, `molrs.io.gro.GroReader`, `molrs.io.lammps.LammpsDumpReader`, `molrs.io.dcd.DcdReader`, `molrs.io.trr.TrrReader`, `molrs.io.xtc.XtcReader` (one path or a list of paths; the same surface). `read_pdb_trajectory` and `read_gro_trajectory` return their reader too, not a `list` (`.read_all()`). |
| `molrs.io.log.LammpsLog`, … | `molrs.io.lammps.LammpsLog`, … |
| `molrs.io.lammps_bond_react.BondReactTemplate` | `molrs.io.lammps.BondReactTemplate` |
| `molrs.io.smiles.{SmilesIR, CGSmilesIR, CGGraph, CGNode, CGEdge, CGFragmentDef, ResolvedPair, PairEnd}`, `molrs.core.CGBond` | `molrs.io.smiles.SmilesIr`, `molrs.io.cgsmiles.{CgSmilesIr, CgGraph, CgNode, CgEdge, CgFragmentDef, ResolvedPair, PairEnd}`, `molrs.core.CgBond` (`BondingDescriptor` stays `molrs.io.smiles`'s) |
| `SmartsPattern` `repr` `SmartsPattern(num_query_atoms=N)` | `SmartsPattern('<smarts>')`; `str(pattern)` is the SMARTS text |

**WASM, C and C++**

| Earlier 0.16 builds | 0.16 |
|---|---|
| JS `parseSMILES(s)` | `SmilesIr.parse(s)` (the IR, `toFrame()`), `readSmilesStr(s)` (one molecule's `Frame`) |
| JS `readSTL` | `readStlBytes` |
| JS `CIFReader`, `GROReader`, `MOL2Reader`, `POSCARReader`, `XSFReader`, `CHGCARReader`, `AcReader` | `CifReader`, `GroReader`, `Mol2Reader`, `VaspPoscarReader`, `XsfReader`, `VaspChgcarReader`, `AmberAcReader` (`CubeReader`, `AmberInpcrdReader` unchanged) |
| JS `XYZStream`, `PDBStream`, `SDFStream`, `LAMMPSStream`, `LAMMPSTrajStream`, `DCDStream`, `XTCStream`, `TRRStream`, `FrameIndexEntry` | `XyzStream`, `PdbStream`, `SdfStream`, `LammpsDataStream`, `LammpsDumpStream`, `DcdStream`, `XtcStream`, `TrrStream`, `FrameOffset` |
| JS `writeFrame(frame, fmt)` | one writer per format: `writePdbStr`, `writeXyzStr`, `writeGroStr`, `writeMol2Str`, `writeCifStr`, `writeXsfStr`, `writeCubeStr`, `writeVaspPoscarStr`, `writeLammpsDataStr`, `writeLammpsDumpStr` (CIF, GRO and MOL2 now check the Frame schema first, as every writer class does) |
| JS `writeFrameBytes(frame, fmt)`, `readFrameBytes(data, fmt)` | `writeDcdBytes`, `writeTrrBytes`, `writeXtcBytes`, `writeMsgpackFrameBytes` / `readMsgpackFrameBytes`, `writeJsonFrameStr` / `readJsonFrameStr` |
| JS `MrecReader.fromStore` | `MrecReader.fromStorage` |
| C `molrs_frame_from_smiles`, C++ `xyz_read_first_frame`, `read_first_frame`, `write_frame_xyz_typed` | over `molrs::io::read_smiles_str`, `read_xyz`, `read_mrec_frame` and `XyzWriter` (the C++ XYZ writer now checks the Frame schema first); renamed in [Wave S5](#wave-s5-bindings) |

**Acronyms are cased as words** here too, as in `ff` (`PdbReader`, `Mmff`):
the SMILES / CGsmiles IR and record types are `SmilesIr`, `CgSmilesIr`,
`CgGraph`, `CgNode`, `CgEdge`, `CgFragmentDef`, `CgBondOrder` (Rust), the
CoarseGrain bond view is `molrs.core.CgBond`, and the JS class is `SmilesIr`;
MMFF's van der Waals rows are `ff::params::mmff::{MmffVdw, MmffVdwStyle}`.
`DType` and the WASM `NDArray` keep numpy's spelling (`numpy.dtypes.*DType`,
`numpy.typing.NDArray`). The analysis names (`RDF`, `MSD`, `PMFTXY`, …) are
not covered by this change; [Wave S4](#wave-s4-analysis-perception-geometry-dynamics)
cases them (`Rdf`, `Msd`, `PmftXy`, …).

#### Engine constants are unit facts (`core::constants`)

Each engine's Coulomb constant and charge factor has one owner,
`molrs::core::constants`, used by `io` and `ff` alike:

| 0.15 | 0.16 |
|---|---|
| `molrs::ff::params::amber::AMBER_COULOMB` | `molrs::core::constants::AMBER_COULOMB` |
| `molrs::io::data::prmtop::CHARGE_CONVERSION_FACTOR` | `molrs::core::constants::AMBER_CHARGE_FACTOR` |
| `molrs::io::data::prmtop_tables::CHAMBER_COULOMB` | `molrs::core::constants::CHARMM_COULOMB` |
| `molrs::ff::forcefield::readers::opls::OPENMM_COULOMB` | `molrs::core::constants::OPENMM_ONE_4PI_EPS0` (OpenMM's own value, kJ·nm·mol⁻¹·e⁻²) |
| `molrs::ff::forcefield::readers::gromacs::GROMACS_COULOMB` | `molrs::core::constants::GROMACS_ONE_4PI_EPS0` (GROMACS's own value, kJ·nm·mol⁻¹·e⁻²) |
| `molrs::compute::voronoi::BOHR_TO_ANG` (and the cube reader's copy) | the unit registry: `molrs::core::UnitFactor::new("bohr", "angstrom")` |
| `molrs::compute::distribution::KB_KCAL_PER_MOL_K` (1.987204e-3) | `molrs::core::constants::BOLTZMANN_REAL` (1.98720425864083e-3): `CombinedDistributionResult::free_energy` moves by 1.3·10⁻⁷ relative |

The private kcal ↔ kJ and nm ↔ Å copies in the GROMACS, OpenMM XML, `.gro`,
`.trr` and `.xtc` code are gone: unit conversions go through the unit
registry (`molrs::core::UnitFactor::new("kcal", "kJ")`,
`UnitRegistry::factor`; Python `molrs.core.UnitRegistry().factor("kcal",
"kJ")`), not through constants. MMFF's 332.0716 is `MMFF_COULOMB`, and
OPLS-AA's 1-4 weights are `OPLS_LJ_14` / `OPLS_COULOMB_14`.
LAMMPS `real`'s `qqr2e` stays `COULOMB_REAL`. Python's `AMBER_COULOMB` is
`molrs.core.constants.AMBER_COULOMB` (it was `molrs.ff.AMBER_COULOMB`).

The AMBER 1-4 divisors `SCEE` = 1.2 / `SCNB` = 2.0 are
`core::constants::{AMBER_SCEE, AMBER_SCNB}`, and only the force-field
reader assumes them. The prmtop structure reader holds no 1-4
weight at all: the per-pair `"pairs"` block (`coul_scale` / `lj_scale` from
`SCEE` / `SCNB`) is force-field meaning and comes from
`io::amber::AmberPrmtopForcefieldReader::{read_system, read_system_str}`
(Python `molrs.io.read_amber_prmtop_system`), which return
`(ForceField, Frame)` like `GromacsTopForcefieldReader::read_system`
(Python `read_gromacs_top_system`); `io::read_amber_prmtop` now
never has a `"pairs"` block, and the refusal of a 1-4 row on a bonded /
angle-end pair moved with it. A file without `SCEE_SCALE_FACTOR` /
`SCNB_SCALE_FACTOR` (pre-Amber-11) still gets no `"pairs"` block;
`read_amber_prmtop_forcefield` prices it as before. The prmtop tables are
crate-private: its ff-only helpers (`one_four_weights` / `OneFourWeights`,
`chamber_urey_bradleys` / `UreyBradley`) moved into the force-field reader,
and the naming helpers both readers share are internal.

#### Force fields (`ff`)

`ir` is the force-field IR (adopts the LAMMPS standard) and its registry,
`forcefield` the `ForceField` data model, `potential` the kernels,
`typifier` typing, `charge` the charge models, `params` the shipped tables.
No file format is `ff`'s: every file reader and writer is `io`'s, one module
per format (0.15 `molrs::ff::forcefield::{readers, writers, xml,
lammps_units}` and the `molrs::ff` re-exports of their items; the names are
in [Wave S2](#wave-s2-io-per-format)). `ff` names
`io` only in its test-only engine checks.

- **No re-exports at `molrs::ff`.** Each name is at its owner:
  - `ForceField`, `SpecialBonds` → `ff::forcefield::`;
    `read_forcefield_xml[_str]` → `io::read_molrs_xml_forcefield[_str]`
    (molrs's own layout) or `io::read_openmm_xml_forcefield[_str]`.
  - `ForceFieldReader` → `io::reader::ForceFieldReader`; `LammpsFfReader`,
    `GromacsTopFfReader`, `OplsXmlReader`, `AmberPrmtopFfReader` →
    `io::lammps::LammpsForcefieldReader`,
    `io::gromacs::GromacsTopForcefieldReader`,
    `io::openmm_xml::OpenmmXmlReader`,
    `io::amber::AmberPrmtopForcefieldReader`; `read_amber_prmtop_ff` →
    `io::read_amber_prmtop_forcefield`.
  - `ForceFieldWriter`, `WriteError` → `io::writer::{ForceFieldWriter,
    ForceFieldWriteError}`; `LammpsFfWriter` / `LammpsWriteOptions`,
    `GromacsTopFfWriter`, `AmberFrcmodFfWriter`, `XmlForceFieldWriter` →
    `io::lammps::{LammpsForcefieldWriter, LammpsForcefieldWriteOptions}`,
    `io::gromacs::GromacsTopForcefieldWriter`, `io::amber::AmberFrcmodWriter`,
    `io::openmm_xml::OpenmmXmlWriter`; `write_amber_frcmod[_str]` →
    `io::write_amber_frcmod[_str]`; `write_forcefield_xml[_str]` →
    `io::write_openmm_xml_forcefield[_str]`.
  - `BccModel`, `BccParameterSet`, `ChargeError`, `ChargeModel`,
    `MullikenModel` → `ff::charge::`.
  - `FragmentAtoms`, `FragmentScaling`, `ScaleLjError`, `compute_k_ij`,
    `scale_lj` → `ff::clpol_scaling::`.
  - `assign_cmaps` → `ff::typifier::cmap::assign_cmaps` (also no longer at
    `ff::typifier::`); `GaffParameterSet` → `ff::typifier::GaffParameterSet`.
- **IR vocabulary is the IR's.** `ParamSource`, `RowSource`, `SpecialClass`
  and `KernelConstructor` are at `ff::ir::` only (were also
  `ff::potential::` and `ff::potential::registry::`; that module is private,
  and `ff::ir::Registry` is the one registry). `ScalarForm`,
  `CompoundForm` and `ParamColumns` are at `ff::potential::form_kernel::`
  only (were also `ff::ir::`); `form_kernel`'s `bonded` / `compound` /
  `form` / `pair` modules are private. `ff::potential::{lookup_kernel, lookup_typed_kernel,
  lookup_param_source, lookup_row_source}` are removed: ask the registry,
  `ff::ir::with_global_registry(|r| r.style(category, name))` /
  `r.param_source(..)` / `r.row_source(..)`.
- **`molrs::md::SpecialWeights` → `molrs::ff::potential::SpecialWeights`**,
  and `PairWeights::special_weights(&topo)` returns it (was the per-atom
  lists to feed `SpecialWeights::new`).
- **The soft packing potential is a potential; `Lbfgs` is the one
  optimizer.** `molrs::optimize::{SoftSpec, SoftLbfgs}` and
  `optimize::soft` are removed. `SoftSpec` is
  `molrs::ff::potential::soft::SoftSpec`; minimize with
  `Lbfgs::new(Arc::new(spec.potential(frame.simbox.as_ref())),
  LbfgsSettings { … })`. `SoftPotential` resolves its own pairs: its
  springs at the first configuration it sees (as `SoftLbfgs` did), its
  non-bonded pairs rebuilt whenever an atom has moved half a 1 Å skin
  (`SoftLbfgs` rebuilt them once per run), in any box (was cubic only).
  `SoftSpec::{build_bonded, build_nb, build_potential, params, sigma, a_rep,
  b_attract, rcut, k_bond, k_ang}`, `SoftPotential::{new, n_pairs}`,
  `HarmTerm` and `NbTerm` are removed.
- **`molrs::ff::mmff` is removed**: MMFF typing (aromaticity, atom types,
  charges, the parameter resolver) is private to `ff::typifier::mmff`, and
  `MmffVariant` / `MmffMolProperties` with it — pick a variant by picking
  `Mmff94Typifier` or `Mmff94sTypifier`. `ff::mmff::da::{DA_NEITHER,
  DA_DONOR, DA_ACCEPTOR}` → `ff::params::mmff::`.
  `ff::typifier::mmff::params::{MMFFAtomProp, MMFFParams}` →
  `ff::params::mmff::MmffProp` (the one row type) and
  `ff::typifier::mmff::MmffAtomProperties`.
- **One hybridization: `molrs::perceive::Hybridization`**, with
  `perceive::{perceive_hybridizations, perceive_conjugated_atoms}` — RDKit's
  `setHybridization` / `setConjugation`, checked against RDKit 2026.03 on 27
  molecules. It replaces `ff::mmff::hybrid::Hyb`, the UFF typifier's private
  heuristic and the conformer's `distgeom::mol_features::Hybridization`.
  MMFF is unchanged. UFF atom labels now follow RDKit: an amide N, a
  conjugated carbonyl C / O, an ester O are `N_R` / `C_R` / `O_R` (0.15:
  `N_3` / `C_2` / `O_2` / `O_3`), and angle and torsion terms follow the new
  hybridizations, so UFF energies of such molecules change — to RDKit's
  (acetanilide's bonded UFF energy now equals RDKit's to 1e-14). ETKDG's
  bounds move with them, and its 1-2 bounds are the UFF typifier's rest
  lengths (`conformer::distgeom`'s private UFF table is gone), so a
  phosphorus bond now gets its UFF length (`P_3+3`) where 0.15 fell back to
  the van der Waals guess.
- **No amide bond order.** UFF and the ETKDG 1-2 bounds price an amide C–N
  at its graph order 1, as RDKit 2026.03 does (`ff::params::uff::AMIDE_BOND_ORDER`,
  1.41, is removed; a UFF amide bond's type is `C_R-N_R@1`).
- **A typifier reads no file.** The typing-metadata readers are
  `io::{read_mmff_xml_params_str, read_openmm_xml_opls_typing_str}`
  (the latter refuses a dangling or cyclic `overrides`), and a caller's own
  XML builds a typifier from what they read:
  `Mmff94Typifier::from_parts(read_mmff_xml_params_str(xml)?,
  read_mmff_xml_forcefield_str(xml)?)` (`Mmff94sTypifier` alike) and
  `OplsAaTypifier::new(read_openmm_xml_opls_typing_str(xml)?,
  read_openmm_xml_forcefield_str(xml)?)`. There is no
  `from_xml_str` constructor. Python's `OplsAaTypifier(xml)` takes the XML
  as 0.15's `OPLSAATypifier(xml)` did.
- **BCC tables are the charge model's.** `ff::typifier::{BccParameterSet,
  BCCCorrectionTable, BCCCorrector}` are removed: `BccParameterSet` is at
  `ff::charge::BccParameterSet` only, and the corrections are applied by
  `BccModel::correct` (the second door, `BCCCorrector::apply`, is gone with
  `BCCCorrectionTable`). `ff::typifier::BccAtomChargeTypifier` (0.15
  `BCCAtomChargeTypifier`) stays.
  `BccModel::new` returns `BccModel` (was `Result`; it could not fail).
- **`molrs::perceive::equivalence::average_charges` is removed**: the
  class-mean is a charge-model step, applied by `ChargeModel::assign` for a
  model that declares it (`needs_equivalencing`).
- **`molrs::math::pair_form::lj_ab_to_sigma_epsilon` →
  `molrs::ff::potential::pair::lj_ab_to_sigma_epsilon`.**
- **`molrs::store::record_v1` is removed** (the version-1 conversion is
  crate-private in the `*.mrec` reader): reading a `molrec_version` 1
  record needs the `zarr` feature, which enables `ff`.
- **CMAP is ordered.** The built-in `cmap` category's endpoint order is
  `Ordered` (was `Reversible`): a cmap type row matches its five atoms as
  written only, since reversing them would swap φ and ψ against the grid.
  Python: `molrs.ff.ir.categories()` reports `"ordered"` for it.

#### Core, io, perceive, compute, conformer, md

- Minimum image: `molrs::compute::util` is gone. `MicHelper` is
  `molrs::core::Mic` (`SimBox::mic()`, or `Mic::ortho(lengths)` for bare
  edge lengths); `get_positions_ref` is crate-private.
- `molrs::op::random::standard_normal` is the one Gaussian draw (md had two
  private copies).
- `molrs::conformer::etkdg` is private (`generate_3d_impl` was a second door
  to `Conformer::generate`); its distance-geometry objectives are internal,
  and the stages minimize with the crate's L-BFGS engine instead of a
  steepest-descent of their own, so embedded geometries differ.
- Force-field formats are `io`'s, one module per format (see
  [Wave S2](#wave-s2-io-per-format)):
  - `molrs::io::data::top::*` (`read_top`, `read_top_frame`, `write_top`,
    `TopReader`, `TopFrameWriter`) →
    `io::gromacs::GromacsTopForcefieldReader::read_system` /
    `io::gromacs::GromacsTopForcefieldWriter::write_system_str`
    (0-based indices). Python: `molrs.io.read_top` →
    `molrs.io.read_gromacs_top_system`; `molrs.io.write_top` →
    `molrs.io.write_gromacs_top_system`.
  - `molrs::io::data::frcmod::*` (`read_frcmod`, `parse_frcmod`,
    `format_frcmod`, `write_frcmod`, `FrcmodFile`) →
    `io::write_amber_frcmod`. Python: `molrs.io.read_frcmod`,
    `parse_frcmod`, `write_frcmod` removed
    (`molrs.io.write_amber_frcmod` writes one).
  - `molrs::io::data::prmtop_tables::decode_{bond,angle,dihedral,nonbond}_params`
    and their row aliases, and `io::data::prmtop::read_amber_prmtop_sections`;
    `parse_pointers` / `parse_a4_names` are crate-private. Python:
    `molrs.io.prmtop_parse_pointers`, `prmtop_parse_a4_names`,
    `prmtop_decode_*`, `read_amber_prmtop_sections` removed
    (`molrs.io.read_amber_prmtop_forcefield` reads the parameters).
- One SMARTS parser, the crate's line-notation grammar (shared with SMILES),
  which `perceive::smarts::SmartsPattern` compiles from. Its IR gains
  `AtomPrimitive::{AtomicNumber, RingSizeRange, RingBondCount, ContextLabel}`;
  `[#6]` is `AtomicNumber(6)`, no longer `Element { "C" }`. `perceive::smarts`
  needs the `smiles` feature (`ff` enables it).
- Perception is free functions in two shapes, `perceive_<fact>` (a side
  table) and `assign_<fact>` (writes a clone): see
  [Wave S4](#wave-s4-analysis-perception-geometry-dynamics).
  `kekule::assign_kekule_numbers` is crate-private.
- `molrs::perceive::{Coarsener, CoarsenError}` → `molrs::builder::{Coarsener,
  CoarsenError}` (Python: `molrs.perceive.Coarsener` →
  `molrs.builder.Coarsener`).
- One door, one owner (wave 2):
  - `molrs::op::types::{F3x3, FN}` are gone: one alias per type, `Fnx3`
    for any `Array2<F>` (a 3×3 box matrix included) and `F3` for any
    `Array1<F>` (molpack used neither).
  - `molrs::io::reader::open_file` (a "compatibility wrapper") is gone:
    `open_seekable`.
  - `molrs::io::data::vasp_common` is the crate-private VASP header reader
    (`fractional_to_cartesian`, `parse_floats`, … no longer public).
    POSCAR, CONTCAR and CHGCAR place fractional rows through
    `SimBox::to_cart`, and the POSCAR writer's `Direct` rows come from
    `SimBox::to_frac` (from the cell origin; it ignored the origin). The CIF
    reader builds its cell with `SimBox::matrix_from_lengths_angles` — its
    `SimBox` was the transpose of the cell for a triclinic CIF — and places
    `fract_*` through `to_cart`; DCD uses the same constructor and
    `SimBox::{lengths, angles}`.
  - `SimBox` holds its `Mic` and routes `shortest_vector[_impl]` through
    `Mic::apply`: one minimum-image kernel.
  - One io error helper (`InvalidData`), crate-private; the LAMMPS readers'
    `io::lammps::common` is split into `fields` (tokens, numbers, type
    references) and `columns` (Frame columns, dump attribute names).
  - `molrs::perceive::aromaticity` and `molrs::ff::forcefield::lammps_codecs`
    are crate-private (they had no public item).
  - `molrs::optimize::minimize_lbfgs_rms` is crate-private (the ETKDG
    stages are its one user); `optimize::Lbfgs` is the optimizer and
    `optimize::OptimizationReport` its one report.
  - `molrs::ff::charge::compute_gasteiger_charges` is gone:
    `GasteigerModel` is the door.
  - The record section's units table derives each section preset from
    `core::UnitPreset::builtin` (new), the one table of preset units,
    spelled as the record spells them; the presets a section may state are
    the LAMMPS `units` styles and `openmm` (`nm`, `kJ/mol`, `ps`), as
    molrec lists them.
  - Hybridization perception takes an element's first valence from
    `Element::default_valences` (its private table is gone): Ga, In, Sn, Sb,
    Te, Rb, Cs, Sr and Ba now have one, and Ge's is 2.
  - Version-1 records are converted by the `*.mrec` reader itself
    (`ff::forcefield::record_v1` → the reader's private `record_v1`): the
    conversion needs only core data and the force-field IR's table
    vocabulary (`ff::ir`), which the record codec's `zarr` feature enables.
- One name per handle and payload type: `AtomId`, `BeadId` → `NodeId`;
  `BondId`, `AngleId`, `DihedralId`, `ImproperId`, `PortId` → `RelationId`;
  `Bead` → `Atom`; `Bond`, `Angle`, `Dihedral`, `Improper` → `Relation`
  (all in `molrs::core`).

#### Python paths

The Python package follows the same rule. `molrs` holds the subsystems and
nothing else — `core`, `op`, `perceive`, `io`, `ff`, `optimize`, `md`,
`conformer`, `builder`, `compute`, `signal`, `stream` — and every symbol has
one public path: the Python module named after its Rust owner
(`molrs.core.Frame` is `molrs::core::Frame`). A
class's and a function's `__module__` and `__name__` are that path (0.15
functions said `molrs._lib`, and the `molrs.op`, `molrs.ff.ir` and
`molrs.core.schema` ones said `op`, `ir`, `schema`), so `repr`, pickle and
the docs name every symbol the way it is imported. A public module exports
exactly its `__all__`: no `typing` or stdlib name imported for an annotation
(`molrs.io.Path`, `molrs.ff.ir.Callable`, `annotations`, …) is reachable on
it any more.

- A Rust namespace below a subsystem's facade is not a Python module of its
  own: `ff::potential::pair::PairLjCut` is `molrs.ff.potential.PairLjCut`,
  `perceive::smarts::Reaction` is `molrs.perceive.Reaction`,
  the core's neighbour search and regions are `molrs.core`, and the
  `io::{data, trajectory, mesh, csv}` formats' functions are `molrs.io`.
  Kept as modules: the vocabularies `molrs.core.keys`,
  `molrs.core.schema` and `molrs.core.constants`, `molrs.ff.ir`, and one submodule of `molrs.io` per
  format that owns classes (below).
- **Every file-format factory has one shape** — a function at the top of
  `molrs.io`, `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]`, or a class
  `molrs.io.<fmt>.<Fmt>Reader` / `<Fmt>Writer`. So every file reader and
  writer is `molrs.io`'s: structure and trajectory files, force-field files
  (`molrs.ff.read_lammps_forcefield` and the rest, 0.15, are
  `molrs.io.read_lammps_forcefield`, …; they are not `molrs.ff.forcefield`'s,
  which is the `ForceField` data model only), the `*.mrec` doors
  (0.15 `molrs.io.read_mrec` / `write_mrec` are `read_mrec_frame` /
  `write_mrec_frame`, beside their `_system` / `_trajectory` / `_forcefield`
  partners and `read_mrec_meta`), the wire-encoded frames (0.15
  `molrs.io.read_frame_bytes` / `write_frame_bytes` are
  `read_msgpack_frame_bytes` / `write_msgpack_frame_bytes` and
  `read_json_frame_str` / `write_json_frame_str`; not `molrs.stream`'s, which
  is the transport) and the new `molrs.io.read_smiles_str` /
  `write_smiles_str`. A reader of in-memory text carries `_str`:
  `molrs.io.parse_lammps_log_text` is `molrs.io.read_lammps_log_str`. The
  full list is [Wave S2](#wave-s2-io-per-format).
- **A class that belongs to one format is that format's submodule's**: the
  lazy trajectory readers `molrs.io.pdb.PdbReader`, `molrs.io.xyz.XyzReader`,
  `molrs.io.gro.GroReader`, `molrs.io.dcd.DcdReader`,
  `molrs.io.trr.TrrReader`, `molrs.io.xtc.XtcReader` and
  `molrs.io.lammps.LammpsDumpReader` (what each `read_<fmt>_trajectory`
  returns), `molrs.io.mrec` (`MrecReader` / `MrecWriter` — the lazy store
  cursor and its writer, named as Rust's `molrs::io::mrec::{MrecReader,
  MrecWriter}` — `SequenceSchema`, `ForceFieldSection`, `section_names`,
  `pack_mrec_zip`, `validation`), `molrs.io.smiles` (`SmilesIr`,
  `SmilesError`, `BondingDescriptor`), `molrs.io.cgsmiles` (the CGsmiles
  records) and `molrs.io.lammps` (the `Lammps*` log records,
  `BondReactTemplate`). The top of `molrs.io` holds functions only.
- **One door per fact on a class.** `SmartsPattern.find_matches(mol,
  mapped=True)` is gone (each `SmartsMatch.mapping` is the
  `{map_number: atom}` dict it returned), and so are `SmartsMatch.as_list()`
  / `as_dict()` (`atoms` / `mapping`), `Trajectory.from_frames` (the
  constructor) and `Trajectory.count_frames()` (`len(traj)`).
- `molrs.ff` holds only its submodules, one per `molrs::ff` submodule:
  `forcefield` (the `ForceField` data model and its handles; no file
  format), `potential`, `typifier`, `charge`, `ir`, `params`, `clpol_scaling`.
- `molrs.compute` is flat, as `molrs::compute` is; the domain subpackages
  (`molrs.compute.density`, …) are gone.
- Removed, with no second spelling left behind: `molrs.io.raw` and
  `molrs.fields` (every reader emits the canonical column names — PDB, LAMMPS
  dump / molecule, MOL2 and XYZ map their own spellings in Rust), the
  `molrs.io.read_gro` / `write_gro` / `read_gro_trajectory` /
  `write_gro_trajectory` pass-through wrappers (the compiled functions are
  those names now), `molrs.io.write_smiles` (an alias; SMILES text is
  written by `molrs.io.write_smiles_str(mol, **flags)`), the eager
  `list[Frame]` readers behind `molrs.io.raw.read_{dcd,trr,xtc,xyz,lammps}_trajectory`
  (`molrs.io.read_*_trajectory(path).read_all()`), the native `*TrajReader`
  classes as public names (`molrs.io.read_*_trajectory` returns the format's
  own reader, `molrs.io.<fmt>.<Fmt>Reader`), `Atomistic.max_ring_system_size()`
  (`molrs.perceive.perceive_rings(mol).max_ring_system_size()`), and the
  `molrs.md` lazy loader.
- The protocol and driver modules are private: `molrs.compute.Compute`,
  `molrs.ff.potential.Potential` and `molrs.md.MdDriver` are the only spellings.

**Core: the top level is subsystems only**

| 0.15 | 0.16 |
|---|---|
| `molrs.Angle` | `molrs.core.Angle` |
| `molrs.Atom` | `molrs.core.Atom` |
| `molrs.Atomistic` | `molrs.core.Atomistic` |
| `molrs.Bead` | `molrs.core.Bead` |
| `molrs.Block` | `molrs.core.Block` |
| `molrs.BlockDtypeError` | `molrs.core.BlockDtypeError` |
| `molrs.Bond` | `molrs.core.Bond` |
| `molrs.Box` | `molrs.core.Box` |
| `molrs.CGBond` | `molrs.core.CgBond` |
| `molrs.CoarseGrain` | `molrs.core.CoarseGrain` |
| `molrs.Cuboid` | `molrs.core.Cuboid` |
| `molrs.Cylinder` | `molrs.core.Cylinder` |
| `molrs.Dihedral` | `molrs.core.Dihedral` |
| `molrs.DrudeParticle` | `molrs.core.DrudeParticle` |
| `molrs.Element` | `molrs.core.Element` |
| `molrs.Ellipsoid` | `molrs.core.Ellipsoid` |
| `molrs.ExtractedSubgraph` | `molrs.core.ExtractedSubgraph` |
| `molrs.Frame` | `molrs.core.Frame` |
| `molrs.FrameMeta` | `molrs.core.FrameMeta` |
| `molrs.Graph` | `molrs.core.MolGraph` |
| `molrs.HalfSpace` | `molrs.core.HalfSpace` |
| `molrs.Improper` | `molrs.core.Improper` |
| `molrs.MasslessSite` | `molrs.core.MasslessSite` |
| `molrs.MetaDocument` | `molrs.core.MetaDocument` |
| `molrs.MetaValue` | `molrs.core.MetaValue` |
| `molrs.NeighborList` | `molrs.core.NeighborList` |
| `molrs.NeighborQuery` | `molrs.core.NeighborQuery` |
| `molrs.Neighbors` | `molrs.core.Neighbors` |
| `molrs.NodeRef` | `molrs.core.NodeRef` |
| `molrs.Parallelepiped` | `molrs.core.Parallelepiped` |
| `molrs.Polyhedron` | `molrs.core.Polyhedron` |
| `molrs.Port` | `molrs.core.Port` |
| `molrs.Quantity` | `molrs.core.Quantity` |
| `molrs.Reaction` | `molrs.perceive.Reaction` |
| `molrs.Refs` | `molrs.core.Refs` |
| `molrs.Region` | `molrs.core.Region` |
| `molrs.RelationBuckets` | `molrs.core.RelationBuckets` |
| `molrs.RelationRef` | `molrs.core.RelationRef` |
| `molrs.ScalarObservable` | `molrs.core.ScalarObservable` |
| `molrs.Sphere` | `molrs.core.Sphere` |
| `molrs.SphereUnion` | `molrs.core.SphereUnion` |
| `molrs.Topology` | `molrs.core.Topology` |
| `molrs.Trace` | `molrs.core.Trace` |
| `molrs.Trajectory` | `molrs.core.Trajectory` |
| `molrs.TriMesh` | `molrs.core.TriMesh` |
| `molrs.Unit` | `molrs.core.Unit` |
| `molrs.UnitPreset` | `molrs.core.UnitPreset` |
| `molrs.UnitRegistry` | `molrs.core.UnitRegistry` |
| `molrs.UnitsError` | `molrs.core.UnitsError` |
| `molrs.VectorObservable` | `molrs.core.VectorObservable` |
| `molrs.VerletSkin` | `molrs.core.VerletSkin` |
| `molrs.VirtualSite` | `molrs.core.VirtualSite` |
| `molrs.keys` | `molrs.core.keys` (and `molrs.keys.<NAME>` → `molrs.core.keys.<NAME>`) |
| `molrs.schema` | `molrs.core.schema` (and its block-name constants) |
| `molrs.schema.BlockSpec` | `molrs.core.schema.BlockSpec` |
| `molrs.schema.ColumnSpec` | `molrs.core.schema.ColumnSpec` |
| `molrs.schema.block` | `molrs.core.schema.block` |
| `molrs.schema.blocks` | `molrs.core.schema.blocks` |
| `molrs.schema.column` | `molrs.core.schema.column` |
| `molrs.schema.columns` | `molrs.core.schema.columns` |
| `molrs.schema.relation_endpoints` | `molrs.core.schema.relation_endpoints` |
| `molrs.schema.to_json` | `molrs.core.schema.to_json` |
| `molrs.schema.to_markdown` | `molrs.core.schema.to_markdown` |
| `molrs.Atomistic.max_ring_system_size()` | `molrs.perceive.perceive_rings(mol).max_ring_system_size()` |

**Force fields: `molrs.ff` holds only its submodules**

| 0.15 | 0.16 |
|---|---|
| `molrs.ff.AMBER_COULOMB` | `molrs.core.constants.AMBER_COULOMB` |
| `molrs.ff.AMBER_SCEE` | `molrs.core.constants.AMBER_SCEE` |
| `molrs.ff.AMBER_SCNB` | `molrs.core.constants.AMBER_SCNB` |
| `molrs.ff.AngleStyle` | `molrs.ff.forcefield.AngleStyle` |
| `molrs.ff.AngleType` | `molrs.ff.forcefield.AngleType` |
| `molrs.ff.AtdTypifier` | `molrs.ff.typifier.AtdTypifier` |
| `molrs.ff.AtomStyle` | `molrs.ff.forcefield.AtomStyle` |
| `molrs.ff.AtomType` | `molrs.ff.forcefield.AtomType` |
| `molrs.ff.BccModel` | `molrs.ff.charge.BccModel` |
| `molrs.ff.BondStyle` | `molrs.ff.forcefield.BondStyle` |
| `molrs.ff.BondType` | `molrs.ff.forcefield.BondType` |
| `molrs.ff.CmapStyle` | `molrs.ff.forcefield.CmapStyle` |
| `molrs.ff.CmapType` | `molrs.ff.forcefield.CmapType` |
| `molrs.ff.DihedralStyle` | `molrs.ff.forcefield.DihedralStyle` |
| `molrs.ff.DihedralType` | `molrs.ff.forcefield.DihedralType` |
| `molrs.ff.ForceField` | `molrs.ff.forcefield.ForceField` |
| `molrs.ff.FragmentScaling` | `molrs.ff.clpol_scaling.FragmentScaling` |
| `molrs.ff.GaffTypifier` | `molrs.ff.typifier.GaffTypifier` |
| `molrs.ff.GasteigerModel` | `molrs.ff.charge.GasteigerModel` |
| `molrs.ff.ImproperStyle` | `molrs.ff.forcefield.ImproperStyle` |
| `molrs.ff.ImproperType` | `molrs.ff.forcefield.ImproperType` |
| `molrs.ff.MMFF94STypifier` | `molrs.ff.typifier.Mmff94sTypifier` |
| `molrs.ff.MMFF94Typifier` | `molrs.ff.typifier.Mmff94Typifier` |
| `molrs.ff.Match` | `molrs.ff.typifier.TypeAssignment` |
| `molrs.ff.MullikenModel` | `molrs.ff.charge.MullikenModel` |
| `molrs.ff.OPLSAATypifier` | `molrs.ff.typifier.OplsAaTypifier` |
| `molrs.ff.PairStyle` | `molrs.ff.forcefield.PairStyle` |
| `molrs.ff.PairType` | `molrs.ff.forcefield.PairType` |
| `molrs.ff.Potential` | `molrs.ff.potential.Potential` |
| `molrs.ff.PotentialCompiler` | `molrs.ff.potential.PotentialCompiler` |
| `molrs.ff.Potentials` | `molrs.ff.potential.Potentials` |
| `molrs.ff.RelationStyle` | `molrs.ff.forcefield.RelationStyle` |
| `molrs.ff.RelationType` | `molrs.ff.forcefield.RelationType` |
| `molrs.ff.Style` | `molrs.ff.forcefield.Style` |
| `molrs.ff.Type` | `molrs.ff.forcefield.ForceFieldType` |
| `molrs.ff.Typifier` | `molrs.ff.typifier.Typifier` |
| `molrs.ff.assign_cmaps` | `molrs.ff.typifier.assign_cmaps` |
| `molrs.ff.clpol_polarizability` | `molrs.ff.params.clpol_polarizability` |
| `molrs.ff.compute_k_ij` | `molrs.ff.clpol_scaling.compute_k_ij` |
| `molrs.ff.fragment_scaling_data` | `molrs.ff.params.clpol_fragment_scaling()` |
| `molrs.ff.intramolecular_pairs` | `molrs.ff.potential.intramolecular_pairs` |
| `molrs.ff.potential.protocol.Potential` | `molrs.ff.potential.Potential` |
| `molrs.ff.read_amber_prmtop_ff` | `molrs.io.read_amber_prmtop_forcefield` |
| `molrs.ff.read_forcefield_xml` | `molrs.io.read_openmm_xml_forcefield` (an OpenMM file) or `molrs.io.read_molrs_xml_forcefield` (molrs's own layout) |
| `molrs.ff.read_gromacs_system` | `molrs.io.read_gromacs_top_system` |
| `molrs.ff.read_gromacs_top_ff` | `molrs.io.read_gromacs_top_forcefield` |
| `molrs.ff.read_lammps_cmap` | `molrs.io.read_lammps_cmap_forcefield` |
| `molrs.ff.read_lammps_data_coeffs` | `molrs.io.read_lammps_data_coeffs` |
| `molrs.ff.read_lammps_forcefield` | `molrs.io.read_lammps_forcefield` |
| `molrs.ff.read_opls_xml` | `molrs.io.read_openmm_xml_forcefield` |
| `molrs.ff.write_amber_frcmod` | `molrs.io.write_amber_frcmod` |
| `molrs.ff.write_forcefield_xml` | `molrs.io.write_openmm_xml_forcefield` |
| `molrs.ff.write_gromacs_system` | `molrs.io.write_gromacs_top_system` |
| `molrs.ff.write_gromacs_top_ff` | `molrs.io.write_gromacs_top_forcefield` |
| `molrs.ff.write_lammps_cmap` | `molrs.io.write_lammps_cmap_forcefield` |
| `molrs.ff.write_lammps_data_coeffs` | `molrs.io.write_lammps_data_coeffs` |
| `molrs.ff.write_lammps_forcefield` | `molrs.io.write_lammps_forcefield` |
| `molrs.ff.write_lammps_forcefield_str` | `molrs.io.write_lammps_forcefield_str` |
| `molrs.ff.potential.protocol` | private; `Potential` is `molrs.ff.potential.Potential` |
| `molrs._lib.TypedPotentials (only path)` | `molrs.ff.potential.WeightedTerms` |

**I/O**

| 0.15 | 0.16 |
|---|---|
| `molrs.fields.FieldFormatter` | removed (readers emit canonical names) |
| `molrs.fields.LammpsFieldFormatter` | removed (readers emit canonical names) |
| `molrs.fields.PdbFieldFormatter` | removed (readers emit canonical names) |
| `molrs.io.raw` | removed — every reader in `molrs.io` emits canonical names |
| `molrs.io.raw.DCDTrajReader` | removed: `molrs.io.read_dcd_trajectory(path)` (a lazy `molrs.io.<fmt>.<Fmt>Reader`) |
| `molrs.io.raw.LAMMPSTrajReader` | removed: `molrs.io.read_lammps_dump_trajectory(path)` (a lazy `molrs.io.<fmt>.<Fmt>Reader`) |
| `molrs.io.raw.TRRTrajReader` | removed: `molrs.io.read_trr_trajectory(path)` (a lazy `molrs.io.<fmt>.<Fmt>Reader`) |
| `molrs.io.raw.XTCTrajReader` | removed: `molrs.io.read_xtc_trajectory(path)` (a lazy `molrs.io.<fmt>.<Fmt>Reader`) |
| `molrs.io.raw.XYZTrajReader` | removed: `molrs.io.read_xyz_trajectory(path)` (a lazy `molrs.io.<fmt>.<Fmt>Reader`) |
| `molrs.io.raw.parse_lammps_log_text`, `molrs.io.parse_lammps_log_text` | `molrs.io.read_lammps_log_str` |
| `molrs.io.raw.read_amber_inpcrd` | `molrs.io.read_amber_inpcrd` |
| `molrs.io.raw.read_amber_prmtop` | `molrs.io.read_amber_prmtop` |
| `molrs.io.raw.read_chgcar` | `molrs.io.read_vasp_chgcar` |
| `molrs.io.raw.read_cube` | `molrs.io.read_cube` |
| `molrs.io.raw.read_dcd_trajectory` | `molrs.io.read_dcd_trajectory(path).read_all()` |
| `molrs.io.raw.read_gro` | `molrs.io.read_gro` |
| `molrs.io.raw.read_gro_trajectory` | `molrs.io.read_gro_trajectory(path).read_all()` |
| `molrs.io.raw.read_lammps_data` | `molrs.io.read_lammps_data` |
| `molrs.io.raw.read_lammps_log` | `molrs.io.read_lammps_log` |
| `molrs.io.raw.read_lammps_molecule` | `molrs.io.read_lammps_molecule` |
| `molrs.io.raw.read_lammps_trajectory` | `molrs.io.read_lammps_dump_trajectory(path).read_all()` |
| `molrs.io.raw.read_mol2` | `molrs.io.read_mol2` |
| `molrs.io.raw.read_pdb` | `molrs.io.read_pdb` |
| `molrs.io.raw.read_pdb_trajectory` | `molrs.io.read_pdb_trajectory(path).read_all()` |
| `molrs.io.raw.read_trr_trajectory` | `molrs.io.read_trr_trajectory(path).read_all()` |
| `molrs.io.raw.read_xsf` | `molrs.io.read_xsf` |
| `molrs.io.raw.read_xtc_trajectory` | `molrs.io.read_xtc_trajectory(path).read_all()` |
| `molrs.io.raw.read_xyz` | `molrs.io.read_xyz` |
| `molrs.io.raw.read_xyz_trajectory` | `molrs.io.read_xyz_trajectory(path).read_all()` |
| `molrs.io.raw.write_cube` | `molrs.io.write_cube` |
| `molrs.io.raw.write_dcd_trajectory` | `molrs.io.write_dcd_trajectory` |
| `molrs.io.raw.write_gro` | `molrs.io.write_gro` |
| `molrs.io.raw.write_gro_trajectory` | `molrs.io.write_gro_trajectory` |
| `molrs.io.raw.write_lammps_data` | `molrs.io.write_lammps_data` |
| `molrs.io.raw.write_lammps_dump_local` | `molrs.io.write_lammps_dump_local` |
| `molrs.io.raw.write_lammps_molecule` | `molrs.io.write_lammps_molecule` |
| `molrs.io.raw.write_lammps_trajectory` | `molrs.io.write_lammps_dump_trajectory` |
| `molrs.io.raw.write_mol2` | `molrs.io.write_mol2` |
| `molrs.io.raw.write_pdb` | `molrs.io.write_pdb` |
| `molrs.io.raw.write_pdb_trajectory` | `molrs.io.write_pdb_trajectory` |
| `molrs.io.raw.write_trr_trajectory` | `molrs.io.write_trr_trajectory` |
| `molrs.io.raw.write_xsf` | `molrs.io.write_xsf` |
| `molrs.io.raw.write_xtc_trajectory` | `molrs.io.write_xtc_trajectory` |
| `molrs.io.raw.write_xyz` | `molrs.io.write_xyz` |
| `molrs.io.raw.write_xyz_trajectory` | `molrs.io.write_xyz_trajectory` |
| `molrs.io.write_smiles` | `molrs.io.write_smiles_str(mol, **flags)` |
| `molrs.fields` | removed — the Rust readers emit canonical column names |
| `molrs.io.mrec_sections` | `molrs.io.mrec.section_names` |
| `molrs.io.mrec.TrajectoryReader` | `molrs.io.mrec.MrecReader` |
| `molrs.io.mrec.TrajectoryWriter` | `molrs.io.mrec.MrecWriter` |
| `molrs.io.TrajectoryReader` | the format's own reader, `molrs.io.<fmt>.<Fmt>Reader` (`molrs.io.dcd.DcdReader`, …) |
| `molrs.io.SmilesIR`, `molrs.SmilesIR` | `molrs.io.smiles.SmilesIr` |
| `molrs.io.SmilesError` | `molrs.io.smiles.SmilesError` |
| `molrs.io.CGSmilesIR`, `CGGraph`, `CGNode`, `CGEdge`, `CGFragmentDef`, `ResolvedPair`, `PairEnd` | `molrs.io.cgsmiles.{CgSmilesIr, CgGraph, CgNode, CgEdge, CgFragmentDef, ResolvedPair, PairEnd}` |
| `molrs.io.BondingDescriptor` | `molrs.io.smiles.BondingDescriptor` |
| `molrs.io.LammpsLog`, `LammpsLogHeader`, `LammpsRun`, `LammpsThermo`, `LammpsWarning`, `LammpsPerformance`, `LammpsTimingBreakdown`, `LammpsTimingRow`, `LammpsCpuUse`, `LammpsLoadBalance`, `LammpsLoopTime`, `LammpsMemoryUsage`, `LammpsNeighborStatistics` | `molrs.io.lammps.…` (the same names) |
| — | `molrs.io.read_smiles_str(smiles)`: one molecule, connectivity only (a `'.'`-separated set is refused, naming `SmilesIr(s).components()`) |

**Analysis: `molrs.compute` is flat, as the Rust facade is**

| 0.15 | 0.16 |
|---|---|
| `molrs.compute.cluster` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.cluster.CenterOfMass` | `molrs.compute.CenterOfMass` |
| `molrs.compute.cluster.CenterOfMassResult` | `molrs.compute.CenterOfMassResult` |
| `molrs.compute.cluster.Cluster` | `molrs.compute.Cluster` |
| `molrs.compute.cluster.ClusterCenters` | `molrs.compute.ClusterCenters` |
| `molrs.compute.cluster.ClusterCentersResult` | `molrs.compute.ClusterCentersResult` |
| `molrs.compute.cluster.ClusterProperties` | `molrs.compute.ClusterProperties` |
| `molrs.compute.cluster.ClusterResult` | `molrs.compute.ClusterResult` |
| `molrs.compute.cluster.GyrationTensor` | `molrs.compute.GyrationTensor` |
| `molrs.compute.cluster.InertiaTensor` | `molrs.compute.InertiaTensor` |
| `molrs.compute.cluster.RadiusOfGyration` | `molrs.compute.RadiusOfGyration` |
| `molrs.compute.density` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.density.GaussianDensity` | `molrs.compute.GaussianDensity` |
| `molrs.compute.density.LocalDensity` | `molrs.compute.LocalDensity` |
| `molrs.compute.density.RDF` | `molrs.compute.Rdf` |
| `molrs.compute.density.RDFResult` | `molrs.compute.RdfResult` |
| `molrs.compute.density.SpatialDistribution` | `molrs.compute.SpatialDistribution` |
| `molrs.compute.density.SpatialDistributionResult` | `molrs.compute.SpatialDistributionResult` |
| `molrs.compute.dielectric` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.dielectric.Dielectric` | the functions `molrs.compute.{dipole_moment, current_density, static_dielectric_constant, decompose_current}` |
| `molrs.compute.diffraction` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.diffraction.StaticStructureFactorDebye` | `molrs.compute.StaticStructureFactorDebye` |
| `molrs.compute.distribution` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.distribution.AngleDistribution` | `molrs.compute.DistributionFunction` (`"angle"`) |
| `molrs.compute.distribution.CombinedDistribution` | `molrs.compute.CombinedDistribution` |
| `molrs.compute.distribution.CombinedDistributionResult` | `molrs.compute.CombinedDistributionResult` |
| `molrs.compute.distribution.DihedralDistribution` | `molrs.compute.DistributionFunction` (`"dihedral"`) |
| `molrs.compute.distribution.DistanceDistribution` | `molrs.compute.DistributionFunction` (`"distance"`) |
| `molrs.compute.distribution.DistributionResult` | `molrs.compute.DistributionResult` |
| `molrs.compute.dynamics` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.dynamics.Acf` | `molrs.compute.Acf` |
| `molrs.compute.dynamics.AcfResult` | `molrs.compute.AcfResult` |
| `molrs.compute.dynamics.VanHove` | `molrs.compute.VanHove` |
| `molrs.compute.dynamics.VanHoveResult` | `molrs.compute.VanHoveResult` |
| `molrs.compute.environment` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.environment.BondOrder` | `molrs.compute.BondOrientationalOrder` |
| `molrs.compute.fitting` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.fitting.CumulativeTrapezoid` | `molrs.compute.CumulativeTrapezoid` |
| `molrs.compute.fitting.LinearFit` | `molrs.compute.LinearFit` |
| `molrs.compute.fitting.Plateau` | `molrs.compute.Plateau` |
| `molrs.compute.hbond` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.hbond.HBondCriterion` | `molrs.compute.HBondCriterion` |
| `molrs.compute.hbond.HBonds` | `molrs.compute.HBonds` |
| `molrs.compute.hbond.HBondsResult` | `molrs.compute.HBondsResult` |
| `molrs.compute.ml` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.ml.DescriptorRow` | `molrs.compute.DescriptorRow` |
| `molrs.compute.ml.KMeans` | `molrs.compute.Kmeans` |
| `molrs.compute.ml.KMeansResult` | `molrs.compute.KmeansResult` |
| `molrs.compute.ml.Pca2` | `molrs.compute.Pca` |
| `molrs.compute.ml.PcaResult` | `molrs.compute.PcaResult` |
| `molrs.compute.msd` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.msd.MSD` | `molrs.compute.Msd` |
| `molrs.compute.msd.MSDResult` | `molrs.compute.MsdResult` |
| `molrs.compute.msd.MSDTimeSeries` | `molrs.compute.MsdTimeSeries` |
| `molrs.compute.order` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.order.Hexatic` | `molrs.compute.Hexatic` |
| `molrs.compute.order.LegendreReorientation` | `molrs.compute.LegendreReorientation` |
| `molrs.compute.order.LegendreReorientationResult` | `molrs.compute.LegendreReorientationResult` |
| `molrs.compute.order.Nematic` | `molrs.compute.Nematic` |
| `molrs.compute.order.SolidLiquid` | `molrs.compute.SolidLiquid` |
| `molrs.compute.order.Steinhardt` | `molrs.compute.Steinhardt` |
| `molrs.compute.pmft` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.pmft.PMFTXY` | `molrs.compute.PmftXy` |
| `molrs.compute.protocol.Compute` | `molrs.compute.Compute` |
| `molrs.compute.spectroscopy` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.spectroscopy.DipoleAutocorrelationSpectrum` | `molrs.compute.DipoleAutocorrelationSpectrum` |
| `molrs.compute.spectroscopy.DipoleRateCrossSpectrum` | `molrs.compute.DipoleRateCrossSpectrum` |
| `molrs.compute.spectroscopy.EinsteinHelfandSpectrum` | `molrs.compute.EinsteinHelfandSpectrum` |
| `molrs.compute.spectroscopy.GreenKuboSpectrum` | `molrs.compute.GreenKuboSpectrum` |
| `molrs.compute.spectroscopy.IRSpectrum` | `molrs.compute.IrSpectrum` |
| `molrs.compute.spectroscopy.PowerSpectrum` | `molrs.compute.PowerSpectrum` |
| `molrs.compute.spectroscopy.RamanSpectrum` | `molrs.compute.RamanSpectrum` |
| `molrs.compute.spectroscopy.ResonanceRamanSpectrum` | `molrs.compute.ResonanceRamanSpectrum` |
| `molrs.compute.spectroscopy.RoaSpectrum` | `molrs.compute.RoaSpectrum` |
| `molrs.compute.spectroscopy.VcdSpectrum` | `molrs.compute.VcdSpectrum` |
| `molrs.compute.spectroscopy.conductivity_sum_rule` | `molrs.compute.ConductivitySumRule` (`.check(...)`) |
| `molrs.compute.spectroscopy.kramers_kronig` | `molrs.compute.KramersKronig` (`.check(...)`) |
| `molrs.compute.spectroscopy.polarizability_finite_field` | `molrs.compute.polarizability_finite_field` |
| `molrs.compute.spectroscopy.route_agreement` | `molrs.compute.RouteAgreement` (`.check(...)`) |
| `molrs.compute.transport` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.transport.DebyeFit` | `molrs.compute.DebyeFit` |
| `molrs.compute.transport.DebyeRelaxation` | `molrs.compute.DebyeRelaxation` |
| `molrs.compute.transport.DipoleRateCross` | `molrs.compute.DipoleRateCross` |
| `molrs.compute.transport.EinsteinConductivity` | `molrs.compute.EinsteinConductivity` |
| `molrs.compute.transport.EinsteinDiffusion` | `molrs.compute.EinsteinDiffusion` |
| `molrs.compute.transport.GreenKuboConductivity` | `molrs.compute.GreenKuboConductivity` |
| `molrs.compute.transport.GreenKuboDiffusion` | `molrs.compute.GreenKuboDiffusion` |
| `molrs.compute.transport.Onsager` | `molrs.compute.OnsagerCorrelation` (a compute: `.compute(...)`) |
| `molrs.compute.transport.Persist` | the function `molrs.compute.pair_survival_tcf` |
| `molrs.compute.transport.VACF` | `molrs.compute.Vacf` |
| `molrs.compute.voronoi` | `molrs.compute` (flat; the domain subpackages are gone) |
| `molrs.compute.voronoi.DensityGrid` | `molrs.compute.DensityGrid` |
| `molrs.compute.voronoi.MolecularMoments` | `molrs.compute.MolecularMoments` |
| `molrs.compute.voronoi.RadicalVoronoi` | `molrs.compute.RadicalVoronoi` |
| `molrs.compute.voronoi.VoronoiCells` | `molrs.compute.VoronoiCells` |
| `molrs.compute.voronoi.VoronoiIntegration` | `molrs.compute.VoronoiIntegration` |
| `molrs.compute.voronoi.voronoi_domains` | `molrs.compute.VoronoiDomainAnalysis` (`.analyze(...)`) |
| `molrs.compute.voronoi.voronoi_voids` | `molrs.compute.VoronoiVoidAnalysis` (`.analyze(...)`) |
| `molrs.compute.protocol` | private; `Compute` is `molrs.compute.Compute` |

**Perception, builders, MD**

| 0.15 | 0.16 |
|---|---|
| `molrs.md.driver.MD` | `molrs.md.MdDriver` |
| `molrs.perceive.Coarsener` | `molrs.builder.Coarsener` |
| `molrs.md.driver` | private (`molrs.md._driver`); the driver is `molrs.md.MdDriver` |
| `SmartsPattern.find_matches(mol, mapped=True)` | `[m.mapping for m in pattern.find_matches(mol)]` |
| `SmartsMatch.as_list()` | `SmartsMatch.atoms` |
| `SmartsMatch.as_dict()` | `SmartsMatch.mapping` |

**Store**

| 0.15 | 0.16 |
|---|---|
| `molrs.store.Trajectory.from_frames(frames, step, time)` | `molrs.core.Trajectory(frames, step, time)` |
| `molrs.store.Trajectory.count_frames()` | `len(traj)` |

#### WASM and C++ bindings

The binders follow the same rule: one module per molrs owner, one name per
symbol. `molrs-wasm` is laid out as `core/{block, frame, schema, topology,
types, spatial/{simbox, region, mesh, neighbors}}`, `io/` (now with
`smiles`), `perceive`, `compute/<family>` (one file per `molrs::compute`
family plus `catalog`), `conformer`, `ff`, `optimize` and `builder`;
`molrs-cxxapi` is split into `frame`, `io`, `compute`, `charge` and `region`.
The JS namespace stays flat. What changes for callers:

| 0.15 (JS) | 0.16 (JS) |
|---|---|
| `new LinkedCell(cutoff, storeDistSq?, storeDiff?).build(frame)` | `const nl = new NeighborList(cutoff); nl.build(frame); nl.neighbors({ distSq, disp })` |
| `new BruteForce(cutoff, …).build(frame)` | `NeighborList.bruteForce(cutoff)`, then `build` / `neighbors` (no 8 000-atom refusal) |
| `new LinkedCell(cutoff).query(refFrame, otherFrame)` | `new NeighborQuery(refFrame, cutoff).query(otherFrame)` (both columns kept) |
| `topology.findRings()` → `TopologyRingInfo` (`numRings`, `ringSizes`, `rings`, `isAtomInRing`, `numAtomRings`, `atomRingMask`) | `assignRings(frame)` → a new `Frame` whose atoms and bonds carry `is_in_ring` and `n_rings` |
| `Topology.fromFrame(frame)` read `bonds.i` / `bonds.j`, so a canonical frame came back with no bonds | reads `bonds.atomi` / `atomj` (`molrs::core::Topology::from_frame`); a missing endpoint column or an out-of-range atom throws |
| `new XYZReader(text)` / `PDBReader` / `SDFReader` / `LAMMPSReader` / `LAMMPSTrajReader` (whole-content readers) | removed: `XyzStream` / `PdbStream` / `SdfStream` / `LammpsDataStream` / `LammpsDumpStream` (`allocInputBuffer` → `feedIndexChunk` + `finishIndex` → `parseRangeInInput` per frame) are the one reader of those formats |
| `new DCDReader(bytes)` / `TRRReader` / `XTCReader` | removed: `DcdStream` / `TrrStream` / `XtcStream` |
| `new TrajectoryReader(files)` / `TrajectoryReader.fromZip` / `.fromStore` (a `*.mrec` store) | `new MrecReader(files)` / `MrecReader.fromZip` / `.fromStorage`: named as Rust's `molrs::io::mrec::MrecReader` |
| `new LBFGS(pots)` (an internal O(N²) topology pair list, N ≤ 2000) | `new Lbfgs(pots, nl.neighbors())`: the table is required and comes from a `NeighborList` (or `NeighborList.bruteForce`); the force field's `special_bonds` decide whether 1-2 / 1-3 pairs are kept, as `intramolecular_pairs` does |

**The analysis exports drop the `Wasm` prefix** and take their molrs
owner's name, cased as words ([Wave S4](#wave-s4-analysis-perception-geometry-dynamics),
[Wave S5](#wave-s5-bindings)): `WasmVACF` → `Vacf`, `WasmPca2` → `Pca`,
`WasmPMFTXY` → `PmftXy`, and 0.15's unprefixed `RDF` and `MSD` are `Rdf` and
`Msd`. The analyses that are functions in Rust and Python are functions in
JS (`hbondLifetimes`, `hbondComponents`, `pairSurvivalTcf`,
`staticDielectricConstant`, `staticDielectricConstantComponents`, for the
classes `HBondLifetime`, `HBondNetwork`, `PairPersistence` and
`StaticDielectric`), the three `*Distribution` classes are one
`DistributionFunction`, and `AngularSeparation` is
`AngularSeparationGlobal` / `AngularSeparationNeighbor`. The 0.16 classes:
`AngularSeparationGlobal`, `AngularSeparationNeighbor`, `BondOrientationalOrder`, `CombinedDistribution`, `CorrelationFunction`, `Cubatic`, `CumulativeTrapezoid`, `DebyeFit`, `DebyeRelaxation`, `DiffractionPattern`, `DistributionFunction`, `EinsteinConductivity`, `EinsteinDiffusion`, `EinsteinHelfandSpectrum`, `EnvironmentMatch`, `GaussianDensity`, `GreenKuboConductivity`, `GreenKuboDiffusion`, `GreenKuboSpectrum`, `HBonds`, `Hexatic`, `IrFlux`, `IrSpectrum`, `Kmeans`, `LinearFit`, `LocalDensity`, `LocalDescriptors`, `Nematic`, `OnsagerCorrelation`, `Pca`, `PcaResult`, `Plateau`, `PmftR12`, `PmftXy`, `PmftXyt`, `PmftXyz`, `PowerSpectrum`, `RadicalVoronoi`, `RamanSpectrum`, `RamanTensor`, `RoaCrossTensor`, `RoaSpectrum`, `RotationalAutocorrelation`, `SolidLiquid`, `SpatialDistribution`, `SphereVoxelization`, `StaticStructureFactorDebye`, `Steinhardt`, `Vacf`, `VanHove`, `VcdCrossFlux`, `VcdSpectrum`, `VoronoiDomainAnalysis`, `VoronoiVoidAnalysis`. The compute catalog follows: each entry's `wasmExport`
is the new name, and `molrsComputeCatalog().version` is 6.

No other export keeps the prefix either. The chunk-fed trajectory streams take
the names of their reader family:

| 0.15 | 0.16 |
|---|---|
| `WasmXyzStream` | `XyzStream` |
| `WasmPdbStream` | `PdbStream` |
| `WasmSdfStream` | `SdfStream` |
| `WasmLammpsDataStream` | `LammpsDataStream` |
| `WasmLammpsDumpStream` | `LammpsDumpStream` |
| `WasmDcdStream` | `DcdStream` |
| `WasmXtcStream` | `XtcStream` |
| `WasmTrrStream` | `TrrStream` |
| `WasmArray` | `NDArray` (`Array` is a JS global) |

molvis pins `@molcrafts/molrs` 0.15.0 and adopts all these names when it
moves to 0.16.

`readMsgpackFrameBytes` / `writeMsgpackFrameBytes` and `readJsonFrameStr` /
`writeJsonFrameStr` (0.15 `readFrameBytes` / `writeFrameBytes`) need the
`stream` feature (on by default); before, a custom build with `io`
but without `stream` did not compile. `CarbonTubeBuilder` is compiled only
with the `builder` feature (on by default).

`covalentRadius` is bound in the wasm `core` module (`core/element.rs`, over
`molrs::core::Element`), not the crate root; its JS name is unchanged.
`molrs-wasm`'s `io::reader` holds only the formats with no stream (CIF, Cube,
CHGCAR, GRO, MOL2, POSCAR, XSF, inpcrd, AC).

C API (`molrs.h`): `molrs_forcefield_to_json` / `molrs_forcefield_from_json` (0.15
`molrs_ff_to_json` / `molrs_ff_from_json`) read and write
the core `forcefield` record section as JSON, `{"document": {…}, "tables":
{<block>: Block}}` — the serde form (`serde` feature, `molrs/src/serialize.rs`)
of `molrs::io::mrec::ForceFieldSection`, i.e.
`ForceFieldSection::from_forcefield` / `to_forcefield`, the section an `*.mrec` record stores. The C-API-only
document (`name` / `units` / `special_bonds` / `styles[]` with `params` /
`str_params` / `array_params` / `types`) is gone and refused on read; a force
field `ForceFieldSection::from_forcefield` refuses is `InvalidArgument` from
`molrs_forcefield_to_json`.
capi's `F` is `molrs::op::F` (the header keeps `typedef double F;`).
molrs-capi and molrs-cxxapi link molrs with `full,filesystem,rayon,serde`.

C++ (`molrs-cxxapi`):

- **`write_frame_xyz` is removed**: it was `write_frame_xyz_typed` (now
  `write_xyz_frame`) with no metadata. Pass an empty
  `rust::Vec<KeyedMetaValue>`.
- **The `zarr` cargo feature is removed.** It was on by default and the crate
  did not build without it; the `*.mrec` writers and readers are always
  present.
- **`src/bridge.rs` is committed** (no longer git-ignored). `build.rs` still
  generates it and rewrites it only when the text changes; Atomiverse's
  CMake reads it from the source tree (`corrosion_add_cxxbridge`, and
  `MolrsContract.cmake`'s feature probes) before any cargo build.

#### Wave S3: force field

Every `ff` name now states what it is, with one pattern per kind of thing
and acronyms cased as words (`Mmff`, `Uff`, `OplsAa`, `Bcc`, `LjCut`).
Where a name in this table appears elsewhere in this 0.15 → 0.16 guide, the
right-hand column here is the 0.16 name. No old name is kept as an alias.

**Kernels** (`molrs::ff::potential`) are `<Category><Style>`; every
constructor is `<category>_<style>_constructor` (`bond_harmonic_ctor` →
`bond_harmonic_constructor`, …). The registered style strings (`mmff_stbn`,
`uff_lj`, …) are data and do not change.

| 0.16 before S3 | 0.16 |
|---|---|
| `bond::MMFFBondStretch`, `bond::UffBond` | `bond::BondMmff`, `bond::BondUff` |
| `angle::MMFFAngleBend`, `angle::MMFFStretchBend`, `angle::UffAngle` | `angle::AngleMmff`, `angle::AngleMmffStretchBend`, `angle::AngleUff` |
| `angle::CharmmAngleParams` | `angle::AngleCharmmParams` |
| `dihedral::MMFFTorsion`, `dihedral::UffTorsion`, `dihedral::DihedralOPLS` | `dihedral::DihedralMmff`, `dihedral::DihedralUff`, `dihedral::DihedralOpls` |
| `improper::MMFFOutOfPlane`, `improper::UffInversion` | `improper::ImproperMmff`, `improper::ImproperUff` |
| `pair::LJCut`, `pair::PairLJCharmm`, `pair::PairLJClass2` | `pair::PairLjCut`, `pair::PairLjCharmm`, `pair::PairLjClass2` |
| `pair::MMFFVdW`, `pair::VdwAtomParams`, `pair::VdwStyleParams`, `pair::UffVdW` | `pair::PairMmffVdw`, `pair::PairMmffVdwAtomParams`, `pair::PairMmffVdwStyleParams`, `pair::PairUffVdw` |
| `kspace::PmePotential`, `kspace::PmeParams`, `pme_ctor` | `kspace::PairCoulLongPme`, `kspace::PairCoulLongPmeParams`, `pair_coul_long_pme_constructor` |
| `mmff_stbn_ctor`, `mmff_oop_ctor`, `uff_inversion_ctor`, `uff_lj_ctor` | `angle_mmff_stretch_bend_constructor`, `improper_mmff_constructor`, `improper_uff_constructor`, `pair_uff_vdw_constructor` |
| `PairPotential::{pair_eval, eval_pairs, eval_table}`, `LJCut::eval` | `pair_energy_force`, `energy_forces_pairs`, `energy_forces_table`, `PairLjCut::energy_forces_skin` |
| `Member`, `TypedKernel`, `TypedMember` | `ForceTerm`, `ScaledTerm`, `WeightedTerm` |
| `Instances` | `ExplicitTerms` |
| `generic::{ScalarForm, CompoundForm, ParamCols, …}` | `form_kernel::{ScalarForm, CompoundForm, ParamColumns, …}` |
| `geometry::*` (flat-array adapters over `op::vec3`) | crate-private; use `molrs::op::vec3` |
| `KernelRegistry` | crate-private; `ff::ir::Registry` is the one registry |
| `ir::Kernel::Ctor`, `Kernel::ctor` | `Kernel::Constructor`, `Kernel::constructor` |

**The force-field IR** (`molrs::ff::ir`):

| 0.16 before S3 | 0.16 |
|---|---|
| `Dim`, `IrError::Dim` | `ParamDimension`, `IrError::Dimension` |
| `Mix`; `forcefield::mixing::{Mixing, MIXING_RULES}` | `ParamCombination`; `forcefield::combining_rule::{CombiningRule, COMBINING_RULES}` |
| `Value`, `Sample`, `Refusal`, `Metric`, `Residual` | `ParamValue`, `ConformanceSample`, `FormRefusal`, `FitMetric`, `FitResidual` |
| `with_global` | `with_global_registry` |
| `ir::expr::*`, `ExprError` | `ir::expression::*`, `ExpressionError` |
| `positional::no_extra` | `positional::refuse_undeclared_params` |
| `ff::forcefield::torsion` | `ff::ir::torsion` |

`core::schema::ColumnDim` is `ColumnDimension`. The three dimension types
stay three, each named for what it is: `core::Dimension` is the SI
base-dimension exponent vector of the unit algebra; `ff::ir::ParamDimension`
is a force-field parameter's exponents over energy, length, angle, charge
and mass, where angle is a base dimension (an angle *value* is stored in
degrees, a per-radian constant never converts), which SI cannot state;
`ColumnDimension` is what a Frame column measures in a unit preset,
including "not a quantity".

**Typifiers** (`molrs::ff::typifier`):

| 0.16 before S3 | 0.16 |
|---|---|
| `Typifier::r#match(graph) -> Match` | `Typifier::assign(graph) -> TypeAssignment` |
| `Typifier::library()` | `Typifier::source_forcefield()` (`forcefield()` stays the typed output) |
| `Typing::library()` | `typing.typifier().source_forcefield()` |
| `BCCAtomChargeTypifier`, `OPLSAATypifier`, `UFFTypifier` | `BccAtomChargeTypifier`, `OplsAaTypifier`, `UffTypifier` |
| `mmff::{MMFF94Typifier, MMFF94STypifier}` | `mmff::{Mmff94Typifier, Mmff94sTypifier}` |
| `mmff::{MMFFAtomProp, MMFFParams}` | `ff::params::mmff::MmffProp` (the one row type), `mmff::MmffAtomProperties` (`get_prop` → `get`) |
| `OplsTypingMeta` | `OplsTypingMetadata` |
| `TypifierParameterContext` | `EstimationInputs` |

MMFF aromaticity perception now lives with the other perception, in
`molrs::perceive` (crate-private, beside the MMFF topology snapshot it runs
on).

**Charges and tables**:

| 0.16 before S3 | 0.16 |
|---|---|
| (the C++ and Python bindings' own name tables) | `BccParameterSet::{ALL, name, from_name}` |
| `ff::params` `gaff_equiv` | `ff::params` `parmchk` (same items) |
| `ff::scale_lj::{scale_lj, compute_k_ij, FragmentScaling, …}` | `ff::clpol_scaling::{…}` |
| `ff::scale_lj::builtin_fragment_scaling()` | `ff::params::clpol_fragment_scaling()` |

**Python**:

| 0.16 before S3 | 0.16 |
|---|---|
| `molrs.ff.potential.LJCut` (`eval`, `eval_table`, `eval_pairs`, `pair_eval`) | `PairLjCut` (`energy_forces_skin`, `energy_forces_table`, `energy_forces_pairs`, `pair_energy_force`) |
| `molrs.ff.potential.kernel(category, style, atoms, **params)` | `molrs.ff.potential.compile_explicit_terms(…)` |
| `molrs.ff.potential.TypedPotentials` | `WeightedTerms` |
| `molrs.ff.ir.Param`, `CategoryInfo`, `StyleInfo` | `ParamSpec`, `CategorySpec`, `StyleSpec` (as in Rust) |
| `molrs.ff.ir.StyleSpec` (the declarative subclass helper) | `StyleDeclaration` |
| `molrs.ff.ir.unregister` | `unregister_style` |
| `molrs.ff.ir.<Variant>` exceptions (`Arity`, `Dim`, `Sealed`, …) | `<Variant>Error` (`ArityError`, `DimensionError`, `SealedError`, …), all still `IrError` subclasses |
| `molrs.ff.forcefield.Type` | `ForceFieldType` |
| `Style.types` (property) | `Style.get_types()` |
| `molrs.ff.typifier.Match`; a subclass's `match(graph)` and `library()` | `TypeAssignment`; `assign(graph)` and `source_forcefield()` |
| `OPLSAATypifier`, `MMFF94Typifier`, `MMFF94STypifier` | `OplsAaTypifier`, `Mmff94Typifier`, `Mmff94sTypifier` |
| `molrs.ff.scale_lj` (`scale_lj(..., frag_data=)`, `fragment_scaling_data()`) | `molrs.ff.clpol_scaling` (`scale_lj(..., fragment_table=)`); the table is `molrs.ff.params.clpol_fragment_scaling()` |

**WASM**: the JS classes `UFFTypifier`, `MMFF94Typifier` and `MMFF94STypifier`
are `UffTypifier`, `Mmff94Typifier` and `Mmff94sTypifier`. **C++**:
`am1_bcc_assign_frame_from_base` names its correction family exactly as
`BccParameterSet::from_name` does (`"bcc"`, `"abcg2"`; no longer trimmed or
case-folded).

#### Wave S4: analysis, perception, geometry, dynamics

Names in `compute`, `perceive`, `op`, `md`, `optimize`, `conformer`,
`signal`, `builder` and `stream` state their job, acronyms are cased as
words (`Msd`, `Rdf`, `Lbfgs`), counts are `n_*` (numpy / freud), and every
Python name is the Rust name. No old name is kept as an alias.

**`compute`** (Rust and Python, WASM where bound):

| 0.16 pre-release | Now |
|---|---|
| `MSD`, `MSDAccumulator`, `MSDResult`, `MSDTimeSeries` | `Msd`, `MsdAccumulator`, `MsdResult`, `MsdTimeSeries` |
| `RDF`, `RDFAccumulator`, `RDFResult` | `Rdf`, `RdfAccumulator`, `RdfResult` |
| `VACF`, `VACFAccumulator` | `Vacf`, `VacfAccumulator` |
| `PMFTXY`, `PMFTXYT`, `PMFTXYZ`, `PMFTR12` (+ `*Args`, `*Result`) | `PmftXy`, `PmftXyt`, `PmftXyz`, `PmftR12` |
| `IRFlux`, `IRSpectrum` | `IrFlux`, `IrSpectrum` |
| `COMResult`, `RgResult` | `CenterOfMassResult`, `RadiusOfGyrationResult` |
| `Pca2` | `Pca` |
| `MatchEnv`, `MatchEnvResult` | `EnvironmentMatch`, `EnvironmentMatchResult` |
| `PersistResult` | `PairSurvivalResult` |
| `OnsagerResult` | `OnsagerCorrelationResult` |
| `AnyObservable` | `InternalCoordinate` |
| `DistKind`, `NetworkResult`, `LifetimeResult` | `HBondDistanceKind`, `HBondNetworkResult`, `HBondLifetimeResult` |
| `DomainAnalysis`, `DomainResult`, `VoidAnalysis`, `VoidResult`, `Face`, `BOUNDARY` | `VoronoiDomainAnalysis`, `VoronoiDomainResult`, `VoronoiVoidAnalysis`, `VoronoiVoidResult`, `VoronoiFace`, `VORONOI_BOUNDARY` |
| `compute_qlm`, `compute_current_density`, `compute_dipole_moment` | `steinhardt_qlm`, `current_density`, `dipole_moment` |
| `autocorrelation(&series, max_lag)`; `transport::unbiased_cartesian_acf(_scaled)` | `autocorrelation(series.view(), max_lag, mean_subtract)` — the one multiple-time-origin ACF; a `(T, D)` series is `series.view().insert_axis(Axis(1))` |
| `fitting::forward_fft_onesided` (crate-private) | `molrs::signal::forward_fft_onesided` |
| `fit.slope / (2.0 * dims)` by hand | `EinsteinDiffusionResult::diffusion_coefficient(n_dims, window)` |
| `md::kinetic_energy`, `md::com_velocity` | `compute::kinetic_energy`, `compute::center_of_mass_velocity`; new `compute::kinetic_temperature` |
| — | `compute::planar_orientation_angles`, `compute::orientation_quaternions`: the per-particle orientations a PMFT reads (quaternion columns or an `orientations` head–tail block) |

The implementation modules follow (`ml` → `clustering` + `decomposition`,
`density/spatial` → `spatial_distribution`, `dynamics/persist` →
`pair_survival`, `environment/match_env` → `environment_match`, `traits` +
`result` → `analysis_contract`); they are private.

Python `molrs.compute` drops its namespace classes for the Rust shapes:

| Before | Now |
|---|---|
| `Dielectric.compute_dipole_moment(…)` (and the other static methods) | `dipole_moment(…)`, `current_density(…)`, `static_dielectric_constant(…)`, `decompose_current(…)` |
| `Persist.pair_survival_tcf(…)` | `pair_survival_tcf(…)` |
| `Onsager.correlation(p_i, p_j, dt, n)` | `OnsagerCorrelation().compute(p_i, p_j, dt, n)` |
| `AngleDistribution(n)`, `DihedralDistribution(n)`, `DistanceDistribution(n, lo, hi)` | `DistributionFunction("angle", n)`, `DistributionFunction("dihedral", n)`, `DistributionFunction("distance", n, lo, hi)` |
| `kramers_kronig(f, re, im, eps_inf)`, `conductivity_sum_rule(f, s, j2, v, t)`, `route_agreement(d)` | `KramersKronig(eps_inf).check(f, re, im)`, `ConductivitySumRule(j2, v, t).check(f, s)`, `RouteAgreement().check(d)` |
| `voronoi_domains(cells, labels)`, `voronoi_voids(cells, mask, v)` | `VoronoiDomainAnalysis().analyze(cells, labels)`, `VoronoiVoidAnalysis().analyze(cells, mask, v)` |
| `PmftXy.compute` reads an `orientations` block only | reads that block or the `quatw`…`quatk` columns (not both) |
| — | `kinetic_energy`, `kinetic_temperature`, `center_of_mass_velocity` |

**`perceive`**: the `Perceive` builder is deleted (Rust, Python, WASM). Every
perception is a free function, `perceive_<fact>` for a side table and
`assign_<fact>` for writing the fact onto a clone, as RDKit splits a query
from an `Assign*`:

| Before | Now |
|---|---|
| `rings::find_rings` | `perceive_rings` (Python `perceive_rings(mol)`; `RingInfo` has no constructor) |
| `Perceive::find_rings` | `assign_rings` |
| `Perceive::find_aromaticity` | `assign_aromaticity` |
| `Perceive::find_hydrogens` | `add_hydrogens` (an edit, beside `remove_hydrogens`) |
| `Perceive::find_stereo` | `assign_stereo` |
| `Perceive::find_rotatable` | `assign_rotatable_bonds` |
| `Perceive::find_bond_orders` | `assign_bond_orders` |
| `Perceive::find_kekule_orders` | `assign_kekule_bond_orders` |
| `Perceive::find_bond_types[_from_connectivity]` | `assign_bcc_bond_types[_from_connectivity]` |
| `Perceive::find_equivalence_classes[_with]` | `assign_equivalence_classes(mol, opts)` |
| `bond_order::judge_bond_orders` | `perceive_bond_orders` |
| `rotatable::detect_rotatable_bonds[_with_downstream]` | `perceive_rotatable_bonds[_with_downstream]` |
| `stereo::{find_chiral_centers, assign_stereo_from_3d, assign_bond_stereo_from_3d}` | `perceive_chiral_centers`, `perceive_tetrahedral_stereo`, `perceive_bond_stereo` |
| `equivalence::find_equivalence_classes` | `perceive_equivalence_classes` |
| `hybridizations`, `conjugated_atoms` | `perceive_hybridizations`, `perceive_conjugated_atoms` |
| `ring_class::{ring_classes, RingSlot, RingFacts}` | `perceive_ring_classes`, `AntechamberRingMembership`, `AntechamberRingSummary` |
| `rings::max_ring_system_size(mol)` | `perceive_rings(mol).max_ring_system_size()` |
| `rotatable::atom_id_to_index` | removed (a test helper) |
| `RingInfo::{num_rings, num_atom_rings, num_bond_rings}` | `n_rings`, `n_atom_rings`, `n_bond_rings` |

Every `perceive` leaf module is private (`perceive::rings::RingInfo` →
`perceive::RingInfo`); `bond_type` is split into `kekule` and
`bcc_bond_class`. WASM: `new Perceive().findRings(f)` → `assignRings(f)`,
likewise `assignAromaticity`, `addHydrogens`, `removeHydrogens`,
`assignKekuleBondOrders`.

**`op`**: the leaf modules are private and re-exported flat, except
`op::vec3`, which stays a namespace (`add`, `sub`, `dot`, `dihedral`, …).

| Before | Now |
|---|---|
| `op::types::{F, Vec3, …}` | `op::{F, Vec3, …}` |
| `op::{geometry, linalg, rigid, so3, random, superpose}::X` | `op::X` |
| `superpose::Fit`, `SuperposeError` (Python `molrs.op.Fit`) | `Superposition`, `SuperpositionError` (Python `molrs.op.Superposition`) |
| `rigid::{apply, apply_all, compose}` | `transform_point`, `transform_points`, `compose_rigid` |
| `rigid::{about, alignment, frame, nerf}` | `rotation_about`, `alignment_axis_angle`, `orthonormal_frame`, `place_from_internal_coords` |

**`md`** holds integrators and force providers only:

| Before | Now |
|---|---|
| `md::Comm` | `core::GhostHalo` (errors are `GhostError`); `GhostPairs::comm()` → `halo()` |
| `md::Direct` | `md::SelfPairedForces` |
| `md::scalar_mass` | `md::uniform_masses` |
| `md::{kinetic_energy, com_velocity}` | `compute::{kinetic_energy, center_of_mass_velocity}` |
| Python `MD.num_edges` | `MdDriver.n_edges`; `MdDriver.run(thermo=…)` prices KE and T in Rust |

**`optimize`**:

| Before | Now |
|---|---|
| `LBFGS::new(pot, fmax, max_steps, max_step, memory)` | `Lbfgs::new(pot, LbfgsSettings { … })`; `LbfgsSettings::DEFAULT` is the one set of defaults (fmax 0.05, max_steps 500, max_step 0.2, memory 8) every binding reads |
| `Optimizer::run`, `LBFGS::run_coords` | `Optimizer::minimize`, `Lbfgs::minimize_coords` |
| `LBFGS::minimize(pot, coords, …)`, `LBFGS::minimize_batch(…)` | `minimize_lbfgs(pot, coords, &settings)`, `minimize_lbfgs_batch(…)` |
| `OptReport`, crate `MinResult` | `OptimizationReport` (adds `final_grad_rms`) |
| Python `LBFGS(...).run(x)` → `(x, OptReport)` | `Lbfgs(...).minimize(x)` → `(x, OptimizationReport)` |
| WASM `new LBFGS(p, n).run(f, 200)` → `OptReport {steps, energy, maxForce}` | `new Lbfgs(p, n).minimize(f)` (500 steps by default) → `OptimizationReport {converged, nSteps, finalEnergy, finalFmax, finalGradRms}` |

**`ff::potential`** gains `intramolecular_pairs_from_neighbors` (the
exclusion and 1-4 rules of `intramolecular_pairs` over a neighbour table; the
WASM `Lbfgs` uses it) and `improper::ImproperDistance` (LAMMPS
`improper_style distance`, the ETKDG planarity term).

**`conformer`**: `conformer::distgeom` is private. The ETKDG second stage
prices its M6 torsions with the `dihedral periodic` kernel and its planarity
with `ImproperDistance`, and now applies the flat-ring basic-knowledge
torsions it used to compute and drop, so embedded geometries of molecules
with sp2 rings change slightly. Modules: `graph` → `topological_distance`,
`distgeom/{knowledge, matrix, smooth}` → `basic_knowledge_torsions`,
`bounds_matrix`, `triangle_smoothing`.

**`signal`, `builder`, `stream`, core**:

| Before | Now |
|---|---|
| `signal/grid.rs` | `signal/frequency_grid.rs` (`signal::frequency_grid` unchanged) |
| `builder/{strategy, walk}.rs` | `builder/{growth_strategy, self_avoiding_walk}.rs`; paths `builder::GrowthStrategy` etc. unchanged |
| `stream::MessageFormat` | `stream::FrameEncoding` |
| `NeighborQuery::{free, free_columns}` (Python `NeighborQuery.free`) | `unbounded`, `unbounded_columns` (Python `NeighborQuery.unbounded`) |
| `SimBox::isin` (Python `Box.isin`) | `contains`, the `Region` verb |
| `QueryMode::SelfQuery { num_points }`, `Neighbors.num_points`, `num_query_points`, `num_pairs`, `num_clusters`, `num_neighbors`, `num_components` (JS `numPoints`, …) | `n_points`, `n_query_points`, `n_pairs`, `n_clusters`, `n_neighbors`, `n_components` (JS `nPoints`, …) |

**WASM catalog** (version 5): exports are cased as words (`Rdf`, `Msd`,
`Vacf`, `PmftXy`, `IrFlux`, `EnvironmentMatch`, `PairSurvival`, `Pca`), and
the `rdf.*` / `voronoi.*` id prefixes are gone: `density.radial_distribution`,
`locality.radical_voronoi`, `locality.voronoi_domain_analysis`,
`locality.voronoi_void_analysis`; `dynamics.pair_persistence` is
`dynamics.pair_survival`.

**Acronyms and counts, every subpackage** (after S2 and S3 met S4): the
casing rule now holds in every Python subpackage, and `test_public_paths`
checks all of them (numpy's `DType` and `HBond`, where H is the element,
are kept).

| Before | Now |
|---|---|
| `md::MDState` (Python `molrs.md.MDState`) | `md::MdState` (`molrs.md.MdState`) |
| Python `molrs.md.MD` (the driver) | `molrs.md.MdDriver` |
| `compute::{KMeans, KMeansResult}` (Python and WASM `KMeans`) | `Kmeans`, `KmeansResult` (the module is `kmeans`) |
| `op::{FNx3, FNx3View}` | `op::{Fnx3, Fnx3View}` |
| `SmartsPattern::num_query_atoms` (Python `SmartsPattern.num_query_atoms`) | `n_query_atoms` |
| Python `RingInfo.num_rings()` | `RingInfo.n_rings()` |

#### Wave S5: bindings

The WASM, C, C++ and FFI bindings follow the same rules as the core: a
binding's name is its molrs owner's name, cased for the language
(camelCase / PascalCase in JS, acronyms as words, `snake_case` in C and
C++); the cell is `Box` (Rust alone says `SimBox`) and its matrix is `h`;
counts are `n_*`; dtype strings are core `DType::name()` (`float`, `int`,
`uint`, `i8`, …, `c128`); frame metadata is `get_meta` / `set_meta` /
`meta_keys` on every surface. No old name is kept as an alias. Builds of
the 0.16 line before this change spelled the names in the left column.

**molrs-ffi** (Rust; molrs-python, molrs-wasm, molrs-capi and molrs-cxxapi
build on it):

| Earlier 0.16 builds | 0.16 |
|---|---|
| `molrs_ffi::Store` (`store.rs`) | `molrs_ffi::FrameArena` |
| `molrs_ffi::SharedStore`, `new_shared()` | `molrs_ffi::FrameArenaCell` (`Rc<RefCell<FrameArena>>`), `FrameArenaCell::default()` |
| `FrameRef.store`, `BlockRef.store` | `FrameRef.arena`, `BlockRef.arena` |
| `Store::{copy,view,borrow}_col_{F,I,U}` | removed: `BlockRef::{copy,borrow}_{f,i,u}` are the one column accessor set |
| `OwnedColumn` (unreachable) | `molrs_ffi::OwnedColumn` |

**WASM (JS)**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `Block.dtype`, `schemaColumnDtype`, `NDArray.dtype` → `"f64"`, `"i32"`, `"u64"` | `"float"`, `"int"`, `"uint"` (core `DType::name()`) |
| `Frame.metaNames` | `Frame.metaKeys` (beside `getMeta` / `setMeta`) |
| `Box.hMatrix()`, `Box.getCorners()`, `Box.to_frac`, `Box.to_cart` | `Box.h()`, `Box.corners()` (Rust `SimBox::corners`, was `get_corners`), `Box.toFrac`, `Box.toCart` |
| `NDArray.is_empty`, `NDArray.write_from` | `NDArray.isEmpty`, `NDArray.writeFrom` |
| `Mesh` | `TriMesh` |
| `schemaJson` | removed: `schemaDocument` |
| `mrecSections(files)` | `sectionNames(source)` |
| `readMrecFrame(files)`, `readMrecFrameFromZip(bytes)` | `readMrecFrame(source)`: `source` is a file map or a packed zip's bytes |
| `MrecReader.countFrames`, `countAtomsAtFirstFrame` | `MrecReader.nFrames`, `nAtomsAtFirstFrame` |
| `readLammpsLogThermo(text)` → `ThermoTable` | `readLammpsLogStr(text, style?)` → the core `LammpsLog` record |
| `isLammpsLog` (a WASM-only check) | `isLammpsLog`, over core `molrs::io::lammps::is_lammps_log` (Python `molrs.io.lammps.is_lammps_log`) |
| `DistanceDistribution`, `AngleDistribution`, `DihedralDistribution` | `DistributionFunction(observable, nBins, min?, max?)` |
| `StaticDielectric` | `staticDielectricConstant`, `staticDielectricConstantComponents` |
| `HBondLifetime`, `HBondNetwork`, `PairSurvival` (classes) | `hbondLifetimes`, `hbondComponents`, `pairSurvivalTcf` (functions, as in Rust and Python) |
| `AngularSeparation` (`computeGlobal` / `computeNeighbor`) | `AngularSeparationGlobal`, `AngularSeparationNeighbor` (each `.compute`) |
| `GreenKuboDielectricSpectrum`, `EinsteinHelfandDielectricSpectrum` | `GreenKuboSpectrum`, `EinsteinHelfandSpectrum` |
| `generate3D(frame, speed, seed)` | `new Conformer(speed?, addHydrogens?, seed?).generate(frame)` |
| `Potentials.energyForces` | `Potentials.calcEnergyForces` |

A molrs analysis that is a type stays a JS class under the Rust name; one
that is a free function in Rust and Python is a free function in JS. The
compute catalog (`molrsComputeCatalog().version` 6) marks those entries
`inputKind: "function"`. Regions hold `molrs_ffi::RegionRef`, coordinates
cross through `Frame::coords` / `set_coords`, and `SphereUnion.nSpheres`
now counts spheres (it returned 3). The crate's modules mirror molrs:
`core/nd_array.rs` (was `core/types.rs`), `io/lammps_log.rs` (was
`io/log.rs`), `io/mrec.rs`.

**C (`molrs.h`)**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `molrs_frame_from_smiles` | `molrs_read_smiles_str` (core `io::read_smiles_str`; atoms `element`, `mass`, …; bonds `atomi`, `atomj`, `bond_type`, `bond_number`; no coordinates) |
| `molrs_ff_new`, `_drop`, `_def_style`, `_def_type`, `_to_json`, `_from_json` | `molrs_forcefield_new`, `_drop`, `_def_style`, `_def_type`, `_to_json`, `_from_json` |
| `molrs_ff_style_count`, `molrs_ff_get_style_name` | `molrs_forcefield_n_styles`, `molrs_forcefield_style_name` |
| `molrs_frame_put_meta`, `molrs_frame_read_meta`, `molrs_frame_meta_count` | `molrs_frame_set_meta`, `molrs_frame_get_meta`, `molrs_frame_n_meta` (with `molrs_frame_meta_key(i)`) |
| `molrs_schema_column_count`, `molrs_schema_block_count` | `molrs_schema_n_columns`, `molrs_schema_n_blocks` |
| `molrs_block_set_F`, `_I`, `_U` | `molrs_block_set_f64`, `_i32`, `_u64` |
| `molrs_sizeof_F`, `_I`, `_U` | removed (the widths are fixed: 8, 4, 8) |
| `molrs_block_col_commit` | removed (a no-op) |
| `MOLRS_D_TYPE_U_INT`, `INT8`, `INT16`, `INT64`, `U_INT16`, `U_INT32`, `COMPLEX64`, `COMPLEX128` | `MOLRS_D_TYPE_UINT`, `I8`, `I16`, `I64`, `U16`, `U32`, `C64`, `C128`: `MOLRS_D_TYPE_` + the upper-cased `DType::name()` (values unchanged) |
| a box handle argument named `h` | `box_handle`; `h` is only the cell matrix (`molrs_box_h`) |

`molrs_shutdown` now drops regions too: a region handle from before it is
`MOLRS_STATUS_INVALID_REGION_HANDLE` (it stayed alive). In Rust,
`molrs-capi`'s modules are private and every C item is at the crate root.

**C++ (`molrs-cxxapi`)**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `MetaEntry` | `KeyedMetaValue` |
| `frame_meta_entries`, `frame_set_meta_entry` | `frame_meta_keys` + `frame_get_meta(fref, key)` (a missing key throws), `frame_set_meta` |
| `frame_box`, `frame_set_box` | `frame_box_h`, `frame_set_box_h` |
| `frame_column_u32`, `frame_set_column_u32` (they moved u64) | `frame_column_u64`, `frame_set_column_u64` |
| `xyz_read_first_frame` | `read_xyz_frame` |
| `write_frame_xyz_typed` | `write_xyz_frame` |
| `write_frame`, `read_first_frame` (a one-frame `*.mrec`) | `write_mrec_frame`, `read_mrec_frame` (core `io::write_mrec_frame` / `read_mrec_frame`: the record's `frame` section) |
| — | `read_mrec_trajectory_frame(path, index)`: a frame of what an `MrecWriterRef` wrote |
| `TrajectoryWriterRef`, `trajectory_writer_*`, `CXX_CAP_TRAJECTORY_WRITER` | `MrecWriterRef`, `mrec_writer_*`, `CXX_CAP_MREC_WRITER` (same bit) |
| `MsdCompute` / `msd_compute_new`, `RdfCompute` / `rdf_compute_new`, `VacfCompute` / `vacf_compute_new`, `DiffusionCompute` / `diffusion_compute_new` | `Msd` / `msd_new`, `Rdf` / `rdf_new`, `Vacf` / `vacf_new`, `EinsteinDiffusion` / `einstein_diffusion_new` |

Malformed input is an error, not a guess: a region centre, lengths or
axis that is not 3 values, a ragged point list, unequal coordinate
columns, a field block of the wrong size and a malformed or singular `h`
all throw (the region constructors fell back to the origin or zero, and
the XYZ / element frame builders dropped a box `SimBox::new` refused).
Atomiverse (`compat/molrs-016`) moves with these names, and its
`MolrsContract.cmake` probes read the committed `src/bridge.rs`
(`ATV_MOLRS_HAS_MREC_WRITER`).

**Scripts**: the engine checks share `scripts/engine_check_tables.py`
(`read_energy_tsv`, `element_of_mass` over `molrs.core.Element`,
`molecule_ids`, `lammps_thermo` over `molrs.io.read_lammps_log`), and take
unit conversions (kcal ↔ kJ, nm ↔ Å) from molrs's unit registry
(`UnitRegistry.factor`) and `COULOMB_REAL` (332.06371) from
`molrs.core.constants`.

#### Wave S6: residual names

The last pass over the 0.16 names: what the earlier waves left. The rules
are theirs — a door names its format, `read_` / `write_` is a file door at
the top of `io`, counts are `n_*`, acronyms are cased as words, the cell is
`Box` (Rust `SimBox`) — and one more: **every unit conversion goes through
the unit registry.** No old name is kept as an alias. Builds of the 0.16
line before this change spelled the names in the left column.

**Units.** `core::constants` holds physical constants (CODATA 2018 /
SI 2019) and the constants engines define as data; the unit-conversion
factors are gone. A conversion names its two units and is resolved by the
registry: Rust `static KCAL_TO_KJ: UnitFactor = UnitFactor::new("kcal",
"kJ")` (resolved once, `KCAL_TO_KJ.get()`), `UnitRegistry::factor(from,
to)` or `Quantity::to`; Python `molrs.core.UnitRegistry().factor("kcal",
"kJ")`. A power of ten is the correctly rounded factor (`factor("angstrom",
"nm") == 0.1`). The registry's units are built from the constants, so each
has one source: `bohr` is `BOHR_RADIUS`, `eV` and `e` are
`ELEMENTARY_CHARGE`, `hartree` is `HARTREE_ENERGY`, `dalton` is
`ATOMIC_MASS_CONSTANT`, `statC` and `debye` derive from `SPEED_OF_LIGHT`;
the calorie (4.184 J) is the registry's own definition.
`molrs::module_boundaries` (and `test_public_paths.py` for `scripts/`)
fails on a conversion-factor constant or a hand-written factor.

| Earlier 0.16 builds | 0.16 |
|---|---|
| `constants::KJ_PER_KCAL` (4.184) | `UnitFactor::new("kcal", "kJ")` / Python `UnitRegistry().factor("kcal", "kJ")` |
| `constants::ANGSTROM_PER_NM` (10) | `UnitFactor::new("nm", "angstrom")` |
| `constants::ANGSTROM_PER_BOHR` (0.52917721067, CODATA 2014) | `UnitFactor::new("bohr", "angstrom")` = 0.529177210903 (CODATA 2018, `constants::BOHR_RADIUS`) |
| `constants::ANGSTROM3_PER_CM3` | `UnitFactor::new("cm^3", "angstrom^3")` |
| `constants::ANGSTROM_M`, `FEMTOSECOND_S`, `CENTIMETER_PER_METER` | `UnitFactor::new("angstrom", "m")`, `("fs", "s")`, `("m", "cm")` |
| `constants::OPENMM_COULOMB`, `GROMACS_COULOMB` (kcal·Å·mol⁻¹·e⁻²) | `constants::OPENMM_ONE_4PI_EPS0`, `GROMACS_ONE_4PI_EPS0` (kJ·nm·mol⁻¹·e⁻², as the engines state them) × `UnitFactor::new("kJ*nm", "kcal*angstrom")` |
| `ff::params::{OPLSAA_LJ_14, OPLSAA_COULOMB_14}` | `constants::{OPLS_LJ_14, OPLS_COULOMB_14}` (beside `AMBER_SCEE` / `AMBER_SCNB`) |
| MMFF's `332.0716` literal (`MMFF_ELE_STYLE.coulomb`) | `constants::MMFF_COULOMB` |
| — | `constants::{PLANCK, BOHR_RADIUS, HARTREE_ENERGY, ATOMIC_MASS_CONSTANT, COULOMB_CONSTANT}`; `constants::ALL`, the `(name, value)` table Python's `molrs.core.constants` is built from |

Numbers that change: bohr ↔ Å moves from CODATA 2014 to 2018 (relative
4.4·10⁻¹⁰; Gaussian cube files and Voronoi charge integration); the
`debye` unit gains digits (1e-21/c instead of a 12-digit literal, relative
5·10⁻¹³). `UnitPreset("micro")` and `UnitPreset("nano")` stated
`boltzmann()` in J/K and `coulomb()` as 1; they now state them in their own
units, as LAMMPS does (micro 1.380649·10⁻⁸, 8.9875518·10⁶; nano
0.01380649, 230.70776).

**Rust**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `stream::{read_msgpack_frame_bytes, write_msgpack_frame_bytes, read_json_frame_str, write_json_frame_str}` → `Result<_, StreamError>` | `io::` the same names → `std::io::Result` (`InvalidData` on a bad payload); feature `stream` |
| — (Python only) | `io::{read_gromacs_top_forcefield, read_gromacs_top_system}(path, &GromacsTopReadOptions)`, `io::{write_gromacs_top_forcefield, write_gromacs_top_system}(path, …, precision)`, `io::read_amber_prmtop_system`, `io::{read_lammps_forcefield, read_lammps_forcefield_str}`, `io::{write_lammps_forcefield, write_lammps_forcefield_str, write_lammps_data_coeffs, write_lammps_cmap_forcefield}(…, frame, LammpsForcefieldWriteOptions)`, `io::{read_lammps_data_coeffs, read_lammps_cmap_forcefield}` |
| `io::{read_lammps_trajectory, write_lammps_trajectory}` | `io::{read_lammps_dump_trajectory, write_lammps_dump_trajectory}` |
| `io::mrec::validation::read_version` | `io::mrec::validation::molrec_version_of` |
| `io::mrec::StyleEntry` | `io::mrec::SectionStyle` |
| `io::lammps::read_lammps_log_str(text, path, style)` | `read_lammps_log_str(text, source_name, style)` |
| `op::{translate, rotate, scale, center, CenterError}` | `MolGraph::{translate, rotate, scale, center}`, `core::CenterError`; `op` names no other molrs module |
| `ff::ir::LammpsCodec::{read_extra, read_style_args}`, `positional::{read_named, read_style_args}` | `parse_extra`, `parse_style_args`, `positional::{parse_named, parse_style_args}` |
| `Provenance::{write_onto, read_from}` | `Provenance::{apply_to, from_params}` |
| `TypeAssignment::write_onto` | `TypeAssignment::apply_to` |
| `MdState::write_to` | `MdState::apply_to` |
| `ff::potential::cmap::charmm::GRID` | `ff::ir::CMAP_GRID` (the one key) |
| `AtdParameterSet` (no names) | `AtdParameterSet::{ALL, name, from_name}` (antechamber `-at` flags); `AtdBondOrders::{ALL, name, from_name}`, `GaffParameterSet::{ALL, from_name}` |
| `DType::{Int8, Int16, Int64, UInt, UInt16, UInt32, Complex64, Complex128}` (and the same `Column` / `ColumnView` variants) | `I8, I16, I64, Uint, U16, U32, C64, C128` — `DType::name()` (`float`, `int`, `uint`, `i8`, …, `c128`) in Rust's casing, as C's `MOLRS_D_TYPE_<NAME>` |
| `Block::nrows`, `BlockView::nrows`, `Column::nrows`, `ColumnView::nrows`, `BlockAccess::nrows` | `n_rows` |
| `conformer::ForceFieldKind::MMFF94` | `ForceFieldKind::Mmff94` |
| `io::smiles::Notation::CGsmiles` | `Notation::CgSmiles` |
| `perceive::TetrahedralStereo::{CW, CCW}` | `TetrahedralStereo::{Clockwise, CounterClockwise}` (the atom's `stereo` string stays `"CW"` / `"CCW"`) |
| `ObservableRecord.data` | `ObservableRecord.values` |
| `RoaCrossArgs`, `RoaCrossResult` | `RoaCrossTensorArgs`, `RoaCrossTensorResult` |
| `VcdCrossArgs`, `VcdCrossResult` | `VcdCrossFluxArgs`, `VcdCrossFluxResult` |
| `ResonanceRamanArgs` | `ResonanceRamanTensorArgs` |
| `CorrelationArgs` | `CorrelationFunctionArgs` |
| `core::BlockTypes` | `core::BlockTypeLabels` |
| `VerletSkin::rebuild_count`, `FrameAccess::block_count`, `stream::Publisher::client_count` | `n_rebuilds`, `n_blocks`, `n_clients` |

The compute module `clustering` (k-means only) is `kmeans`, beside
`cluster` (freud's cluster analysis); the types keep their paths
(`compute::Kmeans`, `KmeansResult`).

**Python**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `molrs._lib` (the native module; `_lib.pyi`) | `molrs._native` (`_native.pyi`): a pickle naming `molrs._lib.*` does not unpickle |
| `molrs.io.read_gromacs_system`, `write_gromacs_system` | `molrs.io.read_gromacs_top_system`, `write_gromacs_top_system` |
| `molrs.io.read_lammps_trajectory`, `write_lammps_trajectory` | `molrs.io.read_lammps_dump_trajectory`, `write_lammps_dump_trajectory` |
| `molrs.io.read_lammps_log_str(text, path=…)` | `read_lammps_log_str(text, source_name=…)` |
| `Block.nrows`, `Block.resize(nrows)` | `Block.n_rows`, `Block.resize(n_rows)` |
| `ScalarObservable(name, data, …)`, `.data`; `VectorObservable` alike | `ScalarObservable(name, values, …)`, `.values` |
| `NeighborList(cutoff, points, simbox=…)`, `md` integrators' `simbox=` | `box=` |
| `NeighborList.rebuild_count`, `md` driver / integrators' `rebuild_count` | `n_rebuilds` |
| `molrs.stream.Publisher.client_count` | `n_clients` |
| `molrs.core.constants.{KJ_PER_KCAL, ANGSTROM_PER_NM, …}` | `molrs.core.UnitRegistry().factor(from, to)` (see **Units** above) |

`molrs.core.constants` is generated from Rust's `constants::ALL`, so a
constant added in Rust appears with no binding edit. The Python force-field
doors now call the Rust doors above. molpack's ABI handshake calls
`molrs._ffi_abi_token()` on the top-level package and is unaffected.

**WASM (JS)**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `Block.nrows` | `Block.nRows` |
| `Block.get(key, fallback)` beside `Block.copy(key)` | `Block.copy(key, fallback?)` |

The crate's io sources mirror `molrs::io`, one module per format
(`io/xyz.rs`, `io/pdb.rs`, …, `io/lammps.rs` with `io/lammps/log.rs`,
`io/frame_encoding.rs`, `io/frame_index.rs`, `io/stl.rs`); `compute/ml.rs`
is `compute/decomposition.rs` and `compute/kmeans.rs`. JS export names are
unchanged.

**C (`molrs.h`)**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `molrs_schema_json` | `molrs_schema_document` (as JS `schemaDocument`) |
| `molrs_block_nrows`, `molrs_block_ncols` | `molrs_block_n_rows`, `molrs_block_n_columns` |

**C++ (`molrs-cxxapi`)**:

| Earlier 0.16 builds | 0.16 |
|---|---|
| `am1_bcc_assign_frame_from_base` | `assign_am1_bcc_charges` |
| `frame_block_nrows` | `frame_block_n_rows` |

**molrs-ffi**: `BlockRef::nrows` → `BlockRef::n_rows`. Atomiverse
(`compat/molrs-016`) moves with the C++ names.

**Scripts**: `split_system`, the OpenMM residue-topology builder
(`openmm_residue_topology`) and the LAMMPS dump force reader
(`lammps_dump_forces`, over `molrs.io.read_lammps_dump_trajectory`) live
once, in `scripts/engine_check_tables.py`; `molecule_ids` is
`molrs.core.Topology.connected_components`; `gen_param_tables.py` takes
element symbols and numbers from `molrs.core.Element`; every unit
conversion is `engine_check_tables.unit_factor(from, to)` over the unit
registry.

### Python: kernels live in `molrs.ff.potential`

`LJCut` moved from `molrs.md` to `molrs.ff.potential`, as `PairLjCut` (see
[Wave S3](#wave-s3-force-field); molpy: `molpy.md.LJCut` →
`molpy.potential.LJCut`), beside the `Potential` protocol, which `molrs.md`
no longer re-exports either; nor does it re-export `Potentials`
(`molrs.ff.potential.Potentials` / `molpy.Potentials`). The integrators still
accept all of them.

New in the same module: `compile_explicit_terms(category, style, atoms, *,
charges=None, **params)`, the kernel of **any** style the force-field IR prices — a
built-in, a style registered through `molrs.ff.ir` (expression or Python
kernel), a style of a custom category, an unregistered style given its
`expression=` — over explicit instances: `atoms` `(n, arity)`, each per-term
parameter a number or one value per term **as stored** (angle values in
degrees, indexed families as `k1`, `k2`, …), style parameters (`cutoff`,
`coulomb`, …) a number or a string, per-atom charges as `charges=`. It is
built by the code `PotentialCompiler.compile` runs (Rust:
`molrs::ff::potential::ExplicitTerms`), returns a `Potentials`, and
`Potentials.push` moves it into a larger collection. There is no class per
built-in style: one builder covers every registered style, custom ones
included.

```python
from molrs.ff.potential import Potentials, compile_explicit_terms

pots = Potentials()
pots.push(compile_explicit_terms("bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.4))
pots.push(compile_explicit_terms("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5))
pots.push(compile_explicit_terms("dihedral", "periodic", [[0, 1, 2, 3]],
                                 k1=1.3, periodicity1=1, phase1=0.0, k2=0.4, periodicity2=2, phase2=180.0))
pots.push(compile_explicit_terms("pair", "coul/cut", [[0, 3]], charges=q, coulomb=332.06371, dielectric=1.0))
energy, forces = pots.calc_energy_forces(pos)
```

| 0.15 | 0.16 |
|---|---|
| `from molrs.md import LJCut, Potential` | `from molrs.ff.potential import PairLjCut, Potential` |
| `from molpy.md import LJCut` | `from molpy.potential import LJCut` |
| `molrs.md.Potentials` | `molrs.ff.potential.Potentials` |

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
  - `read_lammps_data_coeffs` reads a data file's `PairIJ Coeffs` section;
    it used to be skipped.
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
  defines the typing hooks raises `TypeError` at class creation (0.15.1:
  `match` or `library`; 0.16: `assign` or `source_forcefield`).

### Structure files and geometry

- **MOL2 columns are canonical in Rust.** The reader writes `type` (the
  SYBYL atom type), `res_id` / `res_name` (the substructure) and the bonds'
  SYBYL token as `type`; 0.15 wrote `atom_type`, `subst_id`, `subst_name`
  and `sybyl_bond_type`, and the writer reads the new names. `molrs.io.read_mol2`
  / `write_mol2` returned these names already and are now the compiled
  functions themselves; `molrs.fields.Mol2FieldFormatter` is removed. The
  writer also keeps `res_id` (0.15 read it as signed and wrote `1`).
- **Extended XYZ: `species` reads as `element`, and a wide property is one
  column.** `name:R:3` becomes one `(N, 3)` column `name` (0.15:
  `name_1` … `name_3`), the shape the writer writes back as `name:R:3`.
  `molrs.io.read_xyz` is the compiled function; `molrs.fields.XyzFieldFormatter`
  is removed. (Rust callers: `read_xyz` no longer emits `species`.)
- **LAMMPS data files: every typed block carries a string `type`** — the
  file's `* Type Labels` label, or the numeric id spelled as a label when
  the file has none — beside `type_id`. A labelled file written back is
  numbered by its sorted labels (string labels win over `type_id`, as for
  any frame), which can renumber types; the labels, and so the system, are
  the same. A system with Drude particles (a `drudes` block, or atoms whose
  `vsite` is `"drude"`) gets a `# fix drude flags (atom-type order): …`
  header comment.
- **`write_lammps_data` writes a stated `mass`.** An atom's `mass` column
  is its type's mass; 0.15 replaced it with the element's periodic-table
  mass whenever the element was known, so a Drude core (lighter than its
  element by the shell) or a united atom was written with the wrong mass.
  Rows without a `mass` column still take the element's.
- **`io::data::inpcrd::read_inpcrd` is removed**; it was an alias of
  `read_amber_inpcrd`.
- **`ff::potential::geometry::compute_angle` / `compute_dihedral`** are now
  `op::vec3::angle` / `dihedral` over the flat array: the dihedral is in
  `(−π, π]` (an exact `−π` folds to `π`), and an angle with a zero-length arm
  is `π/2` instead of NaN. Energies agree with 0.15 to rounding.

### Also new in 0.16

- `ForceField.materialize_params(frame, *, prefix)` (Rust
  `ForceField::materialize_params(&mut frame, prefix)`) writes the
  parameters a force field gives each relation row and atom of a typed frame
  as columns `<prefix><parameter>` (null where a row's type lacks one) and
  returns block → columns written; see
  [Force-field IR](guides/forcefield-ir.md#parameters-as-frame-columns).
- New structure doors: `molrs.io.read_sdf`, `read_cif` / `write_cif`,
  `read_vasp_poscar` / `write_vasp_poscar` (Rust `molrs::io::` the same
  names, plus `read_sdf_trajectory`, `read_cif_trajectory`,
  `read_mol2_trajectory`), and lazy PDB and GRO trajectory readers.
- `Frame::concat` / `Frame.concat(frames)`: frames joined block by block,
  relation endpoints offset past the earlier parts (`replicate` for parts
  that differ).
- `op::vec3::{angle, dihedral}` and `op::place_from_internal_coords` (NeRF
  placement from internal coordinates).
- LAMMPS data: `LammpsDataReader::with_atom_style` /
  `read_lammps_data(path, atom_style=None)` fixes the `Atoms` layout as
  LAMMPS's `atom_style` does; `TypeLabels::declare` /
  `write_lammps_data(path, frame, type_labels={"atoms": [...]})` declares
  type labels no row uses.
- `molrs::io::amber::merge_inpcrd(&mut frame, read_amber_inpcrd(path)?)` /
  `read_amber_inpcrd(path, frame=None)` lays an inpcrd's coordinates onto an
  existing frame (a prmtop's structure).
- LAMMPS `fix bond/react`: `io::lammps::BondReactTemplate`,
  `io::{write_lammps_bond_react_map, write_lammps_bond_react_system}` /
  `molrs.io.lammps.BondReactTemplate`, `molrs.io.write_lammps_bond_react_map`,
  `write_lammps_bond_react_system` (data, `.ff`, `_pre.mol`, `_post.mol` and
  `.map` with one type numbering; a type only a template uses is declared
  in the data file with the template's mass, and the `.ff` has no `units`
  line, so the input reads it after `read_data`).
- Regions: `mask(block)`, `region(block)` (the rows inside), `Cuboid.cube`,
  `Sphere.center` / `radius`, `Cuboid.origin` / `lengths`; `&` / `|` with a
  non-region return `NotImplemented`, so a selector's `__rand__` composes.
- Units: the `openmm` preset (nm, kJ/mol, ps), `UnitPreset::new` /
  `UnitPreset.register(name, units, boltzmann=, coulomb=, overwrite=False)`,
  `unit_preset_names` / `UnitPreset.names()`, `replace_unit_preset`, and the
  `boltzmann_constant` (`k_B`) unit.
- `molrs.io.write_gromacs_top_system(path, forcefield, frame, *, precision=6)`,
  the Python door of `GromacsTopForcefieldWriter::write_system_str` and the inverse
  of `read_gromacs_top_system`.
- CL&Pol: `ff::params::CLPOL_POLARIZABILITY` (`alpha.ff`, 78 types),
  `io::read_clpol_alpha` (`io::clpol::ClpolAlphaRow`), and
  `molrs.ff.params.clpol_polarizability(path=None)`.
- `molrs.core.Box(h=None, origin=None, pbc=None, cell_defined=None)`
  takes a `(3, 3)` matrix, a `(3,)` diagonal or any array-like of either;
  `None` or an all-zero matrix is no cell (a free box, `cell_defined`
  false), and `pbc` defaults to whether there is a cell. Explicit
  `cell_defined=False` keeps its meaning but now defaults `pbc` to
  non-periodic. (molpy's `Box` subclass did this; `molpy.Box` can now be
  `molrs.core.Box`.)
- `molrs.core.Trajectory`: `traj[-1]`, `traj[a:b:c]` (a sub-trajectory with
  its `step` / `time` labels sliced alike; Rust `Trajectory::select`),
  `traj.map(func)` and a `repr` (molpy's `Trajectory` subclass did these).
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
| `molrs::compute_gasteiger_charges` | `molrs::ff::charge::GasteigerModel` (`ChargeModel::assign`; the `(NodeId, charge)` wrapper is gone) |
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
