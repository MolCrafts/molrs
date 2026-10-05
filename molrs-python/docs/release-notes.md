# What's new in 0.15

molrs 0.15 settles two foundations: the column store has one accessor and
reports the exact dtype of every column, and `*.mrec` record files follow the
[molrec](https://docs.molcrafts.org/molrec/) contract end to end, from typed
metadata to compressed trajectories and stored force fields. The force-field
model is rebuilt around explicit definitions, a compiler and a typing step.

0.15 is a breaking release on every surface. Work through the
[migration guide](migration.md) when upgrading from 0.14; this page lists the
highlights.

```bash
cargo add molcrafts-molrs --features full,filesystem   # Rust
pip install "molcrafts-molrs>=0.15,<0.16"              # Python
npm install @molcrafts/molrs@0.15                      # JavaScript / TypeScript
```

C and C++ consumers download `molrs-capi-0.15.0-<platform>.tar.gz` from the
[GitHub release](https://github.com/MolCrafts/molrs/releases/tag/v0.15.0).

## Highlights

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

## Compatibility

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
