# What's new in 0.16

molrs 0.16 holds every force field in one **force-field IR**, which adopts the
LAMMPS standard: each style's energy expression, factors and parameter units
are those of the LAMMPS style it corresponds to, and every angle-valued
parameter is in degrees. LAMMPS, GROMACS, OpenMM XML and AMBER prmtop files
are read into it and written from it exactly, or refused by name, and the
conversions are checked term by term against the engines themselves. The IR
is also a protocol: a style or a category of the right shape registers into
it from Rust, Python or molpy, with nothing rebuilt, and is typed, priced,
stored and written like a built-in.

0.16 is a breaking release for force-field code. The energy of a system read
from a file or typed by a typifier does not change (the few exceptions are
bug fixes, listed under [Compatibility](#compatibility)), but some stored
numbers mean something else: a harmonic `k` is LAMMPS's `K` (no ½) and an
angle is in degrees. Work through the [migration guide](migration.md) when
upgrading from 0.15; this page lists the highlights.

```bash
cargo add molcrafts-molrs --features full,filesystem   # Rust
pip install "molcrafts-molrs>=0.16,<0.17"              # Python
npm install @molcrafts/molrs@0.16                      # JavaScript / TypeScript
```

C and C++ consumers download `molrs-capi-0.16.0-<platform>.tar.gz` from the
[GitHub release](https://github.com/MolCrafts/molrs/releases/tag/v0.16.0).

## Highlights

### The force-field IR adopts the LAMMPS standard

- One set of styles, each with LAMMPS's energy expression, factors and
  parameter names: `bond harmonic` and `angle harmonic` are k(x − x₀)²,
  `bond morse` names its well depth `d0`, `pair thole` its damping `damp`,
  and every equilibrium angle and phase is in degrees (force constants stay
  per radianⁿ). [Force-field IR](guides/forcefield-ir.md) is the reference
  for every style.
- Reading a LAMMPS force field and writing it back is the identity on
  coefficients; every other engine and every typifier table (GAFF,
  OPLS-AA, MMFF, UFF) converts at its reader, writer or typifier, never in a
  kernel.
- Spec defaults are applied in one place, so an absent parameter prices the
  same in every kernel tier, and every missing or ill-typed parameter is a
  typed `IrError` (`MissingParam`, `BadValue`, `NoMixing`, …; Python
  `molrs.ff.ir.*`, each a `ValueError`). Compile refusals are typed too
  (`CompileError`).
- `PotentialCompiler.compile` truncates every pair style at its `cutoff`, as
  `compile_typed` and LAMMPS do; `pair coul/long/pme` takes its cell from the
  frame's box.

### New styles and interactions

- **Urey–Bradley**: `angle charmm`, LAMMPS's `angle_style charmm`.
- **CMAP**: the `cmap` category (five endpoints, an `atomm` column and a
  `cmaps` block) with its kernel `cmap charmm`, a step-for-step port of
  LAMMPS `fix cmap`; LAMMPS `fix cmap` files and the data-file `CMAP`
  section are read and written, and `assign_cmaps` builds a frame's `cmaps`.
- **CHARMM 1-4 interactions**: `pair lj/charmm` and `coul/charmm` (LAMMPS
  `lj/charmm/coul/charmm`, switched) with `epsilon14` / `sigma14` and the
  style parameter `one_four`; `dihedral charmm` `w` prices its end atoms'
  1-4 pair; per-pair overrides (`epsilon`, `sigma`, `charge_product`,
  `lj_scale`, `coul_scale`) on a frame's `pairs` rows; one exceptions kernel
  prices them all, LAMMPS's way. `ForceField.materialize_one_four(frame)`
  writes a field's 1-4 pairs as override rows.
- **Torsions**: `dihedral nharmonic`, a `dihedral harmonic` kernel, and the
  exact algebra between every torsion form and a Fourier series
  (`molrs::ff::ir::torsion`).
- **Array parameters**: any type parameter may be an `f64` array, stored in
  a record as a `f64[T, S…]` column.

### Engines, read and written whole

- **LAMMPS**: bonded `hybrid` styles, `angle charmm`, `class2` bond / angle
  / dihedral, `pair buck`, `morse`, `lj/class2`, `lj/charmm/coul/charmm`,
  `lj/cut/coul/long` (as `coul/long/pme`), `pair_modify shift`, `fix cmap`.
  The include writer keeps `special_bonds` and the mixing rule under
  `skip_pair_style`, and a box-less frame is written inside the bounds of
  its atoms.
- **OpenMM XML**: every force `app.ForceField` builds from a Class-I file —
  `<LennardJonesForce>` with `<NBFixPair>`, `<AmoebaUreyBradleyForce>`,
  `<CMAPTorsionForce>`, Ryckaert–Bellemans as `dihedral multi/harmonic`, the
  harmonic `CustomTorsionForce` improper — read and written; expression
  styles are written as `Custom*Force`s; impropers are priced in OpenMM's
  atom order.
- **GROMACS**: `[ pairtypes ]`, `[ cmaptypes ]`, `[ angletypes ]` funct 5,
  `[ dihedraltypes ]` funct 2, 3, 5 and 9, `#define` macros, `gen-pairs no`,
  and whole topologies both ways: `read_gromacs_system` returns the force
  field and a typed frame, `GromacsTopFfWriter::write_system_str` writes one
  back. GROMACS's own Coulomb constant is kept.
- **AMBER prmtop**: chamber (CHARMM) prmtops with Urey–Bradley, CHARMM
  impropers, CMAP and their 1-4 table; ff19SB's CMAP; per-pair `SCEE` /
  `SCNB`; multi-term impropers.
- **Checked against the engines.** One molecule per family (ff14SB, GAFF2,
  CHARMM36 from a chamber prmtop and from OpenMM XML, OPLS-AA from GROMACS)
  is read from its native format, written to every format that can hold it
  and priced by LAMMPS, OpenMM, GROMACS and sander term by term; a generated
  completeness matrix states, for every style and setting, which formats
  hold it exactly and which refuse it
  ([Cross-engine equivalence](guides/forcefield-ir.md#cross-engine-equivalence),
  [Completeness](guides/forcefield-ir.md#completeness)).

### GAFF and GAFF2 as AmberTools builds them

- `GaffTypifier` in Python (`molrs.ff.typifier.GaffTypifier(parameter_set=
  "gaff" | "gaff2")`), composed after `AtdTypifier`.
- Atom types follow antechamber's bond-order perception (`bondtype -j
  full`), its ring classes and its colouring; impropers are placed as tleap
  places them, and every missing bond, angle, torsion and improper is
  estimated as parmchk2 estimates it, the estimate's analog and penalty in
  the type name. `gaff2.dat` is 2.2.30 (AmberTools 26.1).
- `ForceField.materialize_params(frame, prefix=…)` writes the parameters a
  force field gives each row of a typed frame as columns.

### The force-field IR as a protocol

- **Registry** (`molrs::ff::ir`, Python `molrs.ff.ir`): categories and styles
  are data — a category's arity, block and coordinate; a style's ordered
  parameters, each with a dimension, and its energy. Built-ins are sealed
  registrations of the same form. `register_style`, `register_category`,
  `styles`, `categories`, `evaluate`, `unregister_style`.
- **Three kernel tiers**: an energy expression (a Lepton-style grammar with
  `distance`, `angle` and `dihedral` over points), a native scalar or
  compound form, or a Python callable;
  `molrs.ff.potential.compile_explicit_terms(category, style, atoms, **params)`
  (Rust `ff::potential::ExplicitTerms`) builds the kernel of any registered
  style over explicit terms.
- **Custom categories** beyond the seven built-in ones are relation styles
  over a `<category>s` block; a typifier's `TypeAssignment.links` types any relation
  kind (`{Bond: rows, "urey_bradleys": rows}`).
- **Persistence**: a custom style is stored in a `*.mrec` record with its
  expression, so a process that registered nothing reads and prices it.
- **Form conversions**: `ForceField.canonical()`, `to_form(category, style)`
  (exact or refused) and `fit_form(…)` (least squares with its residual).
- **Engine codecs**: a style carries its LAMMPS form (`positional` or
  custom), OpenMM writes expression styles as `Custom*Force`s, and an
  engine that cannot hold a style refuses it as `NoEngineForm`.
- **The proof**: `molrs-ext-example`, a crate depending on molrs only
  through its public API, adds a pair style, a category and an expression
  style and prices them against LAMMPS; see
  [Extending the force-field IR](guides/extending-forcefield-ir.md).

### What molpy and molpack used to do on top

The structure-file and geometry helpers molpy and molpack kept as their own
copies are molrs's now, so both re-export them by identity:
`molrs.io.read_frame` / `write_frame` pick a format from the file name;
`Frame.concat` joins frames with their topology offset; `op::vec3::{angle,
dihedral}` and `op::rigid::nerf` are the internal-coordinate kernels; MOL2
and extended XYZ read straight into canonical columns; a LAMMPS data file
reads with its type labels as `type` and an optional `atom_style`, writes
extra type labels and the `fix drude` flags; `read_amber_inpcrd` fills an
existing frame; the LAMMPS `fix bond/react` file set (templates, map, data
and force field with one type numbering) is written natively; regions mask
blocks; the `openmm` unit preset and `k_B` are native; and the CL&Pol
`alpha.ff` table ships in `ff::params`.

### Records: `molrec_version` 2

- Every record molrs 0.16 writes is `molrec_version` 2, in which the
  `forcefield` section is the force-field IR (degrees, un-halved `k`, the
  `cmap` category, array parameters, `pair lj/charmm` `one_four`, per-pair
  overrides, custom styles with their expressions).
- A version-1 record (molrs 0.15) is converted exactly on read, or refused
  by name; it is never read as version 2. Every 0.15.0 test record prices in
  0.16 at the energy 0.15.0 computed for it.
- A `forcefield` section may state the `openmm` units preset (`nm`,
  `kJ/mol`, `ps`) beside the LAMMPS styles, as molrec lists it; such a
  section reads as a force field in the `openmm` preset.

### One path per symbol

Every module has one job and every public symbol one path, in Rust and in
Python alike. The Rust crate root holds subsystems only (`molrs::core`,
`molrs::ff`, `molrs::io`, …), and so does `import molrs`:
`molrs.core.Frame`, `molrs.core.Box`, `molrs.core.Atomistic`,
`molrs.ff.forcefield.ForceField`, `molrs.ff.potential.PotentialCompiler`,
`molrs.compute.Rdf`. The data model is **one core**: every core name is flat
on `molrs::core` / `molrs.core`, with three vocabularies as submodules —
`core::keys` (every column, frame-meta and graph key, in one place),
`core::schema` and `core::constants` (every physical and engine constant:
CODATA values, each engine's Coulomb constant, AMBER's 1-4 divisors, UFF's
Coulomb constant, unit factors). The cell is `SimBox` in Rust and `Box`
everywhere else; the graph is `MolGraph` everywhere; the chemical bond class
is `BondOrder`, and freud's bond-orientational histogram is
`BondOrientationalOrder`. Ring perception has one owner,
`molrs::perceive::perceive_rings`; whole-graph moves are `molrs::op`'s
(`translate`, `rotate`, `scale`, `center`); the
record's `ForceFieldSection` and `MOLREC_VERSION` are `molrs::io::mrec`'s.
`molrs.__version__` is the package version. `molrs.io.raw`,
`molrs.fields` and the alias functions are gone — every reader emits the
canonical column names. A function's `__module__` names its public path as
a class's does. The [migration guide](migration.md#python-paths) lists every
old → new path.

Every file-format factory has one shape: a function at the top of
`molrs.io` or a class of the format's own submodule — see
[io, one module per format](#io-one-module-per-format).

### io, one module per format

- **One module per format.** `molrs::io` is organized by file format, not by
  content kind: `io::{pdb, xyz, gro, sdf, mol2, cif, vasp, dcd, trr, xtc,
  lammps, amber, gromacs, openmm_xml, clpol, smiles, cgsmiles, mrec}` hold
  each format's classes (`PdbReader`, `LammpsDumpReader`, `OpenmmXmlWriter`,
  …, acronyms cased as words), and Python mirrors them (`molrs.io.pdb`,
  `molrs.io.lammps`, …).
- **Every door names its format**, in Rust and Python alike:
  `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]` at the top of `io`, with
  `_str` / `_bytes` for memory and `_trajectory` for every frame. No door
  picks the format for the caller: `read_frame` / `write_frame` /
  `FrameFormat` and the `format=`-taking frame-bytes doors are gone; the wire
  encodings are `read_msgpack_frame_bytes` / `write_msgpack_frame_bytes` and
  `read_json_frame_str` / `write_json_frame_str`.
- **Lazy readers per format.** Each `read_<fmt>_trajectory` returns that
  format's reader (`molrs.io.dcd.DcdReader`, …; PDB and GRO now too), over
  one path or a list of paths; Rust readers open with `<Fmt>Reader::open`.
- **Honest XML doors.** `read_openmm_xml_forcefield` /
  `write_openmm_xml_forcefield` are inverses, the molrs-native layout has its
  own pair (`read_molrs_xml_forcefield` / `write_molrs_xml_forcefield`, the
  writer new), and an MMFF parameter set is `read_mmff_xml_forcefield`; the
  old reader's layout sniffing is gone.
- **New doors**: `read_sdf`, `read_cif` / `write_cif`, `read_vasp_poscar` /
  `write_vasp_poscar`, `read_smiles_str` / `write_smiles_str`,
  `read_cgsmiles_str`, `read_lammps_molecule_json` /
  `write_lammps_molecule_json`, `read_csv_block[_str]` /
  `write_csv_block[_str]`.
- **SMARTS is perception's, wholly.** `SmartsPattern.from_environment(mol,
  center, …)` (Rust `SmartsPattern::from_environment`) and `str(pattern)`
  replace `write_smarts` / `write_local_smarts`; SMILES and SMARTS share one
  crate-private grammar, so `io` and `perceive` depend on neither.
- **Line-notation IRs cased as words.** `SmilesIr`, `CgSmilesIr`, `CgGraph`,
  `CgNode`, `CgEdge`, `CgFragmentDef`, `CgBondOrder` and `molrs.core.CgBond`
  (were `SmilesIR`, `CGSmilesIR`, `CG*`).
- **`fix cmap` doors say what they return.** Python
  `read_lammps_cmap_forcefield` / `write_lammps_cmap_forcefield` (were
  `read_lammps_cmap` / `write_lammps_cmap`) read and write a `ForceField`;
  Rust `read_lammps_cmap_str` / `write_lammps_cmap_str` stay the raw grids
  (`LammpsCmapFile`).
- The [migration guide](migration.md#wave-s2-io-per-format) lists every old →
  new name.

### Force-field names

Every `ff` name states what it is, in Rust and Python alike, with acronyms
cased as words; the [migration guide](migration.md#wave-s3-force-field)
lists each one.

- **Kernels are `<Category><Style>`**: `BondHarmonic`, `PairLjCut`,
  `BondMmff`, `AngleMmffStretchBend`, `PairUffVdw`, …; constructors are
  `<category>_<style>_constructor`. Python `molrs.ff.potential.PairLjCut`
  evaluates with `energy_forces_skin` / `_table` / `_pairs`, and
  `compile_explicit_terms` (Rust `ExplicitTerms`) builds any style's kernel
  over explicit terms.
- **The force-field IR** names its items for what they are: `ParamSpec`,
  `CategorySpec`, `StyleSpec` (Python too), `ParamDimension`,
  `ParamCombination`, `CombiningRule`, `ParamValue`, `ConformanceSample`,
  `FitMetric` / `FitResidual`, `FormRefusal`; one registry
  (`ff::ir::Registry`), one expression module (`ff::ir::expression`), the
  torsion algebra at `ff::ir::torsion`. Python refusals are
  `<Variant>Error` (`SealedError`, `DimensionError`, …) under `IrError`, and
  a style declared as a class subclasses `molrs.ff.ir.StyleDeclaration`.
- **Typifiers**: `assign(graph) -> TypeAssignment` is the one hook and
  `source_forcefield()` the force field typed against (`forcefield()` stays
  the typed output); `OplsAaTypifier`, `Mmff94Typifier`, `Mmff94sTypifier`,
  `UffTypifier`, `BccAtomChargeTypifier`. MMFF aromaticity is perception's.
- **Tables**: CL&Pol scaling is `molrs.ff.clpol_scaling`
  (`scale_lj(..., fragment_table=)`), its shipped table
  `molrs.ff.params.clpol_fragment_scaling()`; parmchk2's table is
  `ff::params`' `parmchk`; the BCC correction families are named by
  `BccParameterSet::from_name` everywhere. The force-field model's type
  handle is `ForceFieldType`, and `Style.get_types()` its one accessor.

### Analysis, perception, geometry and dynamics

- **Names that state their job, Python equal to Rust.** Acronyms are cased
  as words (`Msd`, `Rdf`, `Vacf`, `PmftXy`, `IrSpectrum`, `Lbfgs`), counts
  are `n_*`, and every `molrs.compute` name is the Rust one: the
  `Dielectric` / `Persist` namespaces are the functions `dipole_moment`,
  `current_density`, `static_dielectric_constant`, `decompose_current` and
  `pair_survival_tcf`; `Onsager` is `OnsagerCorrelation`; the three
  `*Distribution` classes are `DistributionFunction(observable, …)`; the
  spectral checks are `KramersKronig`, `ConductivitySumRule`,
  `RouteAgreement`; the Voronoi analyses are `VoronoiDomainAnalysis` /
  `VoronoiVoidAnalysis`.
- **Perception is two verbs.** The `Perceive` builder is gone:
  `perceive_<fact>(mol)` returns a side table (`perceive_rings` →
  `RingInfo`, `perceive_bond_orders`, `perceive_rotatable_bonds`, …) and
  `assign_<fact>(mol)` writes the fact onto a clone (`assign_rings`,
  `assign_aromaticity`, `assign_stereo`, `assign_bcc_bond_types`, …), in
  Rust, Python and WASM alike; `add_hydrogens` is public beside
  `remove_hydrogens`.
- **One implementation each.** One multiple-time-origin ACF
  (`compute::autocorrelation`) behind `Acf`, `Vacf`, the Debye dipole ACF and
  the Green–Kubo current ACF; one Einstein `D = slope / (2 n_dims)`
  (`EinsteinDiffusionResult::diffusion_coefficient`, which the C++ binding
  calls); one kinetic energy and temperature (`compute::kinetic_energy`,
  `kinetic_temperature`, which `md.MD`'s thermo calls); one PMFT orientation
  reader (`compute::planar_orientation_angles`: quaternion columns or
  head–tail axes, for Python and WASM alike); one exclusion rule for a
  neighbour-table pair list (`ff::potential::intramolecular_pairs_from_neighbors`,
  used by the WASM optimizer).
- **`md` is integrators and force providers.** The periodic ghost halo is
  `core::GhostHalo`; `Direct` is `SelfPairedForces`.
- **One optimization report.** `optimize::Lbfgs` takes `LbfgsSettings`, whose
  `DEFAULT` every binding reads (the WASM optimizer ran 200 steps by default,
  now 500 like the others); `Optimizer::minimize` and Python
  `Lbfgs.minimize` return the one `OptimizationReport`.
- **ETKDG uses the force-field kernels.** Its torsions are priced by the
  `dihedral periodic` kernel and its sp2 planarity by the new
  `improper_style distance` kernel (`ImproperDistance`), with analytic
  forces, and the flat-ring basic-knowledge torsions it computed are now
  applied.
- **Flat `op`.** `molrs::op::X` matches `molrs.op.X` (`op::vec3` stays a
  namespace); the superposition result is `Superposition`, and the rigid
  helpers say what they do (`transform_point`, `rotation_about`,
  `orthonormal_frame`, `place_from_internal_coords`). `Box.contains` and
  `NeighborQuery.unbounded` replace `isin` and `free`.

### Packaging

- FFI capsules move to the `0.16` ABI line (`molrs.FrameRef/0.16`, …):
  extensions built against 0.15 must be rebuilt and re-pinned to
  `>=0.16.0,<0.17`.

## Also in 0.16: the unpublished 0.15.1

molrs 0.15.1 was versioned on `master` but never tagged or published, so
its changes ship in 0.16. It completes the class I force-field model:
explicit Lennard-Jones cross rows (NBFIX) are read from every format that
has them, priced by the kernel, and stored in records; OpenMM torsions are read in OpenMM's own spelling; and
the core Python classes can be subclassed. A few readers and writers now
refuse what they used to drop or mistranslate; the
[migration guide](migration.md) lists each behaviour change.

### Explicit LJ cross rows

- **Energy change:** a `pair/lj/cut` row between two different atom types
  (`def_type("A-B", a, b, epsilon=…, sigma=…)`) now prices that type pair in
  place of the style's `mixing` rule, in `PotentialCompiler.compile` and
  `compile_typed` alike. 0.15.0 compiled such rows but ignored them, so the
  pair got the mixed value; this includes the cross rows `scale_lj` writes.
- Readers that carry cross rows now keep them:
  - LAMMPS `pair_coeff i j ε σ` with `i ≠ j` in a `*.ff` include, and the
    data-file `PairIJ Coeffs` section (both used to be dropped);
  - GROMACS `[ nonbond_params ]` (funct 1), which used to be refused;
  - AMBER prmtop off-diagonal `LENNARD_JONES_ACOEF/BCOEF` entries that are
    not Lorentz–Berthelot, which used to be refused.
- Writers: the GROMACS writer emits `[ nonbond_params ]` and the LAMMPS
  include writer `pair_coeff i j`. The OpenMM XML writer writes a cross row
  as an `<NBFixPair>` of a `<LennardJonesForce>` (0.15.0 wrote it as an
  `<Atom>` row).
- A cross row is a `pair` table row with `itom != jtom` in a record's
  `forcefield` section, so it round-trips through `ForceField.to_section` /
  `from_section` and `*.mrec`.
- A pair is found by its two atom types in either order, so a pair style
  holds one row per pair. `def_type` restating a pair already defined (`B-A`
  after `A-B`, or a second name on `A-B`) is a no-op when the parameters are
  equal and a `ValueError` when they differ; a stored `forcefield` section
  whose `pair` table restates a pair with other parameters is
  refused by `ForceFieldSection.validate()`, `ForceField.from_section` and
  every `*.mrec` reader (molrec forcefield, linking rule 3). `name` and the
  annotation columns (`desc`, `doi`, `smarts`, …) are not compared.
- The AMBER prmtop reader states `mixing = arithmetic` on its `lj/cut` style
  instead of leaving the rule to the kernel default.

### OpenMM force-field XML

- `<PeriodicTorsionForce>` rows in OpenMM's `k1/periodicity1/phase1 …`
  spelling read as `dihedral/periodic` with every term. 0.15.0 read them as
  CL&P `c0..c3` and stored zeros, so such torsions contributed nothing. The
  CL&P spelling still reads as `dihedral/opls`.
- `<Improper>` rows under `<PeriodicTorsionForce>` read as
  `improper/periodic` (they used to be skipped), and the writer emits
  periodic impropers there; `<PeriodicImproperForce>`, which OpenMM does not
  have, is no longer written, and 0.16 refuses to read it: rewrite a file
  0.15.0 wrote with it under `<PeriodicTorsionForce>`.

### LAMMPS force-field writer

- `dihedral/periodic` is written as `dihedral_style fourier` (it had no
  form), including the one-term `k` / `periodicity` / `phase` spelling.
- A bonded-only force field writes without `pair_coeff` lines instead of
  demanding a self pair for every atom label.

### Python

- The core data classes are subclassable: `Block`, `Frame`, `Atomistic`,
  `CoarseGrain`, `ForceField` and its style and type handles,
  `ForceFieldSection`, the typifiers, units, neighbour lists, meshes,
  trajectory observables, views and more. A subclass instance pickles as
  its own class and keeps its instance attributes.
- The native typifiers (`OplsAaTypifier`, `Mmff94Typifier`, …) can be
  subclassed too. Their `assign` and `source_forcefield` run in Rust, so a subclass that
  defines either raises `TypeError`; subclass `Typifier` to supply your own.
- `ForceField.special_bonds` reads the `(lj, coul)` weights back.

## Compatibility

The energy of a physical system read from a file or typed by a typifier is
the same as in 0.15, except where 0.15 was wrong:

- OpenMM impropers are priced over the dihedral OpenMM prices (0.15 priced a
  different one).
- A CHARMM field's 1-4 pairs are priced by `dihedral charmm` `w` (0.15 read
  `w` and ignored it).
- Coulomb energies of OpenMM- and GROMACS-read fields use the engine's own
  constant (9.9·10⁻⁹ relative).
- GAFF / GAFF2 impropers and estimated terms are AmberTools's, and the atom
  types of a molecule follow antechamber's bond orders.
- A `pairs` list prices only the pairs inside each style's `cutoff`.
- `pair_style lj/cut` alone read from LAMMPS prices no charges, as in LAMMPS.
- An explicit LJ cross row (NBFIX) prices its pair in place of the mixing
  rule (0.15.0 compiled it and ignored it).

ETKDG conformers of molecules with sp2 rings differ slightly from 0.15: the
flat-ring torsions RDKit applies are now applied, and the second stage's
torsion and planarity forces are analytic instead of finite differences.

Records molrs 0.16 writes are version 2 and cannot be read by 0.15. A
`ForceField` pickled by 0.15 does not unpickle in 0.16.

See the [migration guide](migration.md) for every breaking change.

## 0.15.0

molrs 0.15 settles two foundations: the column store has one accessor and
reports the exact dtype of every column, and `*.mrec` record files follow the
[molrec](https://docs.molcrafts.org/molrec/) contract end to end, from typed
metadata to compressed trajectories and stored force fields. The force-field
model is rebuilt around explicit definitions, a compiler and a typing step.

0.15 is a breaking release on every surface. Work through the
[migration guide](migration.md) when upgrading from 0.14; this page lists the
highlights.

### One column accessor, one dtype per column

- Rust reads a column through `Block::get(key)` (or
  `FrameAccess::column(block, key)`) and a `Column::as_*` projection —
  `as_float`, `as_int`, `as_uint`, `as_bool`, `as_string`, and the new
  `as_i8` … `as_c128`. The per-dtype getters (`get_float`, `get_uint`, …) are
  gone.
- Each surface reports the variant a column is stored at: Python
  `Block.dtype(key)`, WASM `Block.dtype(key)` with typed arrays chosen by
  dtype (`get` / `view` / `copy` / `set`), and the C API's `MolrsDType`, which
  now names all thirteen stored types and reads any column through
  `molrs_block_get` / `molrs_block_get_mut` / `molrs_block_copy`.
- Floats are `f64` only. f16/f32 columns and f32 metadata are removed; Python
  widens a float32 array on insert.
- Frame metadata keeps insertion order. Python hands JSON metadata back as a
  frozen `MetaDocument`.

### Record files (`*.mrec`) follow molrec

- **Python doors** in `molrs.io`: `write_mrec` / `read_mrec`,
  `write_mrec_system` / `read_mrec_system`, `write_mrec_trajectory` /
  `read_mrec_trajectory`, `write_mrec_forcefield` / `read_mrec_forcefield`,
  `mrec_sections` and `read_mrec_meta`. Streaming lives in `molrs.io.mrec`:
  `SequenceSchema`, `TrajectoryWriter`, the lazy `TrajectoryReader`, and
  `pack` to collapse a closed store into one `*.mrec.zip`.
- **Typed metadata.** A frame or system group stores each meta value with its
  dtype (`_meta_types`), so an `i32` stays `i32` and NaN survives.
- **Declared precision.** `Block.set_precision` / `SequenceSchema.declare_precision`
  round an `f64` column to a binary grid within `p/2` and compress it with
  shuffle + zstd. Coordinates drop from 24 to about 7.6 B/atom/frame at
  `p = 1e-3` Å and 5.8 at `p = 1e-2` Å. Every molrs reader, the WASM build
  included, decodes zstd.
- **Topology conventions.** Canonical `chain`, `res_id`, `res_name`, `icode`,
  `altloc`, `occupancy`, `b_factor`, `formal_charge` and force columns, and
  the blocks `constraints`, `virtual_sites`, `drudes` and `members`. The PDB,
  mmCIF, GRO and extxyz readers produce them.
- **Row references and aligned blocks.** A `uint64` column can declare the
  block it indexes (`Block.set_target`, `SequenceSchema.declare_target`), and
  a trajectory block can be pinned row-for-row to another
  (`SequenceSchema.declare_aligned`). Writers refuse a reference that does not
  resolve.
- **Force-field section.** `ForceField.to_section` / `from_section` map a
  force field onto the molrec `forcefield` section, and
  `write_mrec(..., forcefield=ff)` stores it next to the structure it
  parameterizes.
- Readers ignore root sections they do not know, read only their own section,
  and accept stores without `molrec_version`.

The [Record files guide](guides/records.md) walks through all of it.

### Force fields

- A force field is built only through `def_style(category, name, params)`
  and `def_type(name, endpoints, params)`, with explicit endpoints.
  Re-defining a type with different parameters is an error.
- `PotentialCompiler(ff).compile(frame)` is the one compile path;
  `Typing<T>` (Rust) and the Python `Typifier` base run a typifier whose only
  hook is `match`, and `forcefield()` holds exactly the types it assigned.
  GAFF is a typifier.
- OPLS-AA follows GROMACS `oplsaa.ff` (v2026.3) with geometric mixing.
- **Energy change:** harmonic impropers read from LAMMPS input now evaluate at
  the energy LAMMPS gives them; 0.14 doubled them. The LAMMPS force-field
  writer emits the matching coefficient.

### Building and assembling structures

- A site-graph `Assembler` with `SitePlacer`, `GrowthPlacer` and
  `AxisOrienter`, ports on every graph type, `SubgraphMatcher`, and
  `perceive::Coarsener` for coarse-graining.
- CGsmiles: `parse_cgsmiles` (Python `molrs.io.CGSmilesIR`) parses the
  coarse-grained notation and expands it to atoms.
- `molrs.op`: the vector, linear-algebra and superposition kernels as a
  public module.

### Packaging

- The Rust crate's default features are core only (plus `rayon`); name the
  subsystems you use, or `full`.
- FFI capsules move to the `0.15` ABI line (`molrs.FrameRef/0.15`, …):
  extensions built against 0.14 must be rebuilt and re-pinned to
  `>=0.15.0,<0.16`. The frame vocabulary version is 2.

### Compatibility

- 0.15 reads records more strictly than 0.14: a store that breaks the molrec
  contract (a canonical column at the wrong dtype, a block group without a
  row count, a dangling row reference, …) is refused instead of repaired, and
  a CoarseGrain frame in the 0.14 `beads` + `cgbonds` layout no longer loads.
  The migration guide's `*.mrec` section lists each case.
- Records written by 0.15 with a declared precision use zstd and cannot be
  read by 0.14. Without a declared precision, nothing is rounded.
- Force-field JSON written by the 0.14 C API is refused by 0.15.

See the [migration guide](migration.md) for the full list of breaking
changes, surface by surface.
