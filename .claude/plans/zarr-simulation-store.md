# Spec: Zarr Simulation Store

## Summary

A Zarr V3-based storage system for molecular simulations. One store holds one
simulation (system + force field + trajectory). An optional `Archive` layer
groups multiple simulations for browsing and comparison. Replaces the existing
`ZarrFrame`, `ZarrStoreWriter`, `ZarrStoreReader`, and `StoreData` entirely.

## Motivation

The existing Zarr code is two disconnected pieces — `ZarrFrame` for snapshots
and `ZarrStoreWriter/Reader` for trajectories — with no way to bundle topology,
force field, and trajectory together. A simulation produces all three. They
belong in one store.

## API Conventions

1. **Verb-noun**: all public methods follow verb-noun structure
   (e.g., `read_system`, `write_forcefield`, `append_frame`, `list_simulations`).
2. **Abbreviations are ALL CAPS**: `RDF`, `MSD`, `PE`, `KE`, `FF`, `PBC`.
   Never `rdf`, `Rdf`, `msd`, `Msd`.
3. **No backward compatibility**: delete all existing zarr code, rewrite from
   scratch. All call sites are updated.
4. **Simple API surface**: few types, obvious usage, no builder-pattern chains
   unless clearly needed.

## Scope

- **Crate**: `molrs-core` (rewrite `src/io/zarr/`)
- **Affected call sites**:
  - `molrs-core/src/lib.rs` — re-exports
  - `molrs-wasm/src/io/zarr/mod.rs` — WASM bindings
  - `molrs-core/tests/reproduce_wasm.rs` — integration test
- **Deleted types**: `ZarrFrame`, `ZarrStoreWriter`, `StoreBuilder`, `StoreData`,
  `ZarrStoreReader`
- **New types**: `SimulationStore`, `Archive`, `TrajectoryWriter`,
  `TrajectoryReader`, `TrajectoryFrame`, `TrajectoryConfig`, `Provenance`,
  `UnitSystem`
- **Feature flags**: existing `filesystem` + `zarr`, no new flags

## On-Disk Layout

### Single Simulation

```
simulation.zarr/
+-- zarr.json                        # { molrs_format: "simulation", version: 1 }
|
+-- system/                          # static Frame (topology + initial coords)
|   +-- meta/                        #   group attrs = Frame.meta
|   +-- simbox/                      #   h [3,3], origin [3], PBC [3]
|   +-- atoms/                       #   element, mass, charge, type, x, y, z, ...
|   +-- bonds/                       #   i, j, type
|   +-- angles/                      #   optional
|   +-- dihedrals/                   #   optional
|   +-- impropers/                   #   optional
|
+-- forcefield/                      # ForceField (optional)
|   +-- zarr.json                    #   attrs: { name }
|   +-- styles/
|       +-- bond__harmonic/          #   "{category}__{sanitized_name}"
|       |   +-- zarr.json            #   attrs: { category, style_name, style_params }
|       |   +-- name/                #   [T] string
|       |   +-- itom/                #   [T] string
|       |   +-- jtom/                #   [T] string
|       |   +-- k0/                  #   [T] f64  (one array per param key)
|       |   +-- r0/                  #   [T] f64
|       +-- pair__lj_cut/
|       +-- atom__full/              #   no itom/jtom, just name + params
|
+-- trajectory/                      # time-dependent data (optional)
|   +-- step/                        #   [T]      i64
|   +-- time/                        #   [T]      f64
|   +-- x/                           #   [T, N]   f32
|   +-- y/                           #   [T, N]   f32
|   +-- z/                           #   [T, N]   f32
|   +-- vx/, vy/, vz/               #   optional velocities
|   +-- fx/, fy/, fz/               #   optional forces
|   +-- box_h/                       #   [T,3,3]  f32 (time-varying cell)
|   +-- <scalar>/                    #   [T] f64  (PE, KE, temperature, ...)
|
+-- provenance/
    +-- zarr.json                    #   attrs: { program, version, method, seed }
```

### Archive (Multi-Simulation)

```
archive.zarr/
+-- zarr.json                        # { molrs_format: "archive", version: 1 }
|
+-- nvt_300k/                        # each child = one simulation
|   +-- system/
|   +-- forcefield/
|   +-- trajectory/
|   +-- provenance/
|
+-- nvt_400k/
|   +-- system/
|   +-- forcefield/
|   +-- trajectory/
|   +-- provenance/
|
+-- npt_1atm/
    +-- system/
    +-- forcefield/
    +-- trajectory/
    +-- provenance/
```

`SimulationStore` operates on a store + path prefix. Standalone: prefix = `/`.
Inside archive: prefix = `/{experiment_name}`. Same code, zero duplication.

## Rust API

### Core Types

```rust
/// Unit system tag.
pub enum UnitSystem {
    Real,     // kcal/mol, Angstrom, fs, g/mol, e
    Metal,    // eV, Angstrom, fs, amu, e
    Gromacs,  // kJ/mol, nm, ps, g/mol, e
    Atomic,   // Hartree, Bohr, amu_e, e
}

/// Provenance metadata.
pub struct Provenance {
    pub program: Option<String>,
    pub version: Option<String>,
    pub method:  Option<String>,
    pub seed:    Option<u64>,
}

/// Configuration for trajectory arrays.
pub struct TrajectoryConfig {
    pub positions:  bool,
    pub velocities: bool,
    pub forces:     bool,
    pub box_h:      bool,
    pub scalars:    Vec<String>,    // e.g. ["PE", "KE", "temperature"]
    pub chunk_size: u64,
}

/// Per-frame data for trajectory append.
pub struct TrajectoryFrame<'a> {
    pub step:       i64,
    pub time:       f64,
    pub positions:  Option<&'a [f32]>,   // flat [3N]
    pub velocities: Option<&'a [f32]>,
    pub forces:     Option<&'a [f32]>,
    pub box_h:      Option<&'a [f32]>,   // [9] row-major
    pub scalars:    HashMap<String, f64>,
}
```

### SimulationStore

```rust
pub struct SimulationStore { /* store + prefix */ }

impl SimulationStore {
    // -- Create / Open --

    pub fn create_file(
        path: impl AsRef<Path>,
        system: &Frame,
        forcefield: Option<&ForceField>,
        units: UnitSystem,
        provenance: Provenance,
    ) -> Result<Self, MolRsError>;

    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, MolRsError>;

    pub fn open_store(store: ReadableWritableListableStorage) -> Result<Self, MolRsError>;

    // -- System --

    pub fn read_system(&self) -> Result<Frame, MolRsError>;
    pub fn count_atoms(&self) -> u64;

    // -- ForceField --

    pub fn read_forcefield(&self) -> Result<Option<ForceField>, MolRsError>;

    // -- Trajectory --

    pub fn create_trajectory(
        &mut self,
        config: TrajectoryConfig,
    ) -> Result<TrajectoryWriter, MolRsError>;

    pub fn open_trajectory(&self) -> Result<Option<TrajectoryReader>, MolRsError>;

    // -- Provenance --

    pub fn read_provenance(&self) -> Result<Provenance, MolRsError>;
    pub fn read_units(&self) -> Result<UnitSystem, MolRsError>;
}
```

### TrajectoryWriter / TrajectoryReader

```rust
pub struct TrajectoryWriter { /* ... */ }

impl TrajectoryWriter {
    pub fn append_frame(&mut self, frame: &TrajectoryFrame) -> Result<(), MolRsError>;
    pub fn count_frames(&self) -> u64;
    pub fn close(self) -> Result<(), MolRsError>;
}

pub struct TrajectoryReader { /* ... */ }

impl TrajectoryReader {
    pub fn count_frames(&self) -> u64;

    /// Read a full Frame at index t (system + overlay).
    pub fn read_frame(&self, t: u64) -> Result<Frame, MolRsError>;

    /// Read positions at frame t as flat [3N].
    pub fn read_positions(&self, t: u64) -> Result<Vec<f32>, MolRsError>;
    pub fn read_velocities(&self, t: u64) -> Result<Vec<f32>, MolRsError>;

    /// Read a scalar at one frame.
    pub fn read_scalar(&self, name: &str, t: u64) -> Result<f64, MolRsError>;

    /// Read all step/time values.
    pub fn read_steps(&self) -> Result<Vec<i64>, MolRsError>;
    pub fn read_times(&self) -> Result<Vec<f64>, MolRsError>;

    /// Read a scalar for all frames.
    pub fn read_scalar_series(&self, name: &str) -> Result<Vec<f64>, MolRsError>;

    /// Read positions for a range of frames.
    pub fn read_positions_range(
        &self,
        range: Range<u64>,
    ) -> Result<Vec<Vec<f32>>, MolRsError>;
}
```

### Archive

```rust
pub struct Archive { /* store */ }

impl Archive {
    pub fn create_file(path: impl AsRef<Path>) -> Result<Self, MolRsError>;
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, MolRsError>;

    /// List all simulation names.
    pub fn list_simulations(&self) -> Result<Vec<String>, MolRsError>;

    /// Open an existing simulation by name.
    pub fn open_simulation(&self, name: &str) -> Result<SimulationStore, MolRsError>;

    /// Create a new simulation entry.
    pub fn create_simulation(
        &mut self,
        name: &str,
        system: &Frame,
        forcefield: Option<&ForceField>,
        units: UnitSystem,
        provenance: Provenance,
    ) -> Result<SimulationStore, MolRsError>;

    /// Remove a simulation.
    pub fn remove_simulation(&mut self, name: &str) -> Result<(), MolRsError>;
}
```

## ForceField Serialization

Each `Style` maps to a Zarr group under `/forcefield/styles/`.

Group name: `{category}__{sanitized_name}` (e.g., `pair__lj_cut` for
`("pair", "lj/cut")`). Original name in `style_name` attr.

| ForceField field | Zarr location |
|---|---|
| `Style.name` | group attr `style_name` |
| `Style.category()` | group attr `category` |
| `Style.params` | group attr `style_params` (JSON map) |
| Type `name` | string array `name/` |
| Type `itom/jtom/ktom/ltom` | string arrays (per category) |
| Type param values | one f64 array per param key |

**Atom styles**: `name` + param columns only.
**Bond**: `name`, `itom`, `jtom` + params.
**Angle**: + `ktom`. **Dihedral/Improper**: + `ltom`.
**Pair**: `name`, `itom`, `jtom` + params.
**KSpace**: style-level params only, no type arrays.

## File Organization

```
molrs-core/src/io/zarr/
+-- mod.rs              # pub types, re-exports
+-- simulation.rs       # SimulationStore
+-- archive.rs          # Archive
+-- trajectory.rs       # TrajectoryWriter, TrajectoryReader, TrajectoryFrame
+-- forcefield.rs       # write_forcefield(), read_forcefield()
+-- frame_io.rs         # Block/Column/SimBox <-> Zarr helpers
+-- error.rs            # From<zarrs::*Error> impls
```

## Data Flow

```
Create:
  Frame + FF --> SimulationStore::create_file()
                   +-- write_system(frame)       -> /system/
                   +-- write_forcefield(ff)      -> /forcefield/
                   +-- write_provenance()        -> /provenance/

MD run:
  sim.create_trajectory(config)  -> TrajectoryWriter
    loop { writer.append_frame() }  -> /trajectory/
    writer.close()

Read:
  SimulationStore::open_file(path)
    .read_system()      -> Frame
    .read_forcefield()  -> Option<ForceField>
    .open_trajectory()  -> TrajectoryReader
      .read_frame(t)
      .read_positions(t)
      .read_scalar("PE", t)

Archive:
  Archive::open_file("study.zarr")
    .list_simulations()          -> ["nvt_300k", "nvt_400k"]
    .open_simulation("nvt_300k") -> SimulationStore (prefix="/nvt_300k")
      .read_system() ...
```

## Constraints & Invariants

1. **One trajectory per simulation**. No multi-trajectory.
2. **Immutable system**: after create, system Frame is read-only.
3. **Append-only trajectory**: frames append, never insert or overwrite.
4. **Float types**: system uses `F` alias. Trajectory: f32 for positions/
   velocities/forces, f64 for scalars/time.
5. **Coordinate layout**: system stores `x`, `y`, `z` as separate `[N]` columns.
   Trajectory stores separate `[T, N]` arrays. Flat `[3N]` conversion on I/O.
6. **ForceField params**: all values f64 (matches `Params` internals).
7. **Style naming**: `/` -> `_` in group names. Original in `style_name` attr.
8. **Archive prefix**: `SimulationStore` is parameterized by store + prefix.
   Standalone: `"/"`. Inside archive: `"/{name}"`.

## Deletion List

Remove entirely before implementing:

| File | What |
|---|---|
| `molrs-core/src/io/zarr/frame.rs` | `ZarrFrame` struct + read/write |
| `molrs-core/src/io/zarr/store.rs` | `ZarrStoreWriter`, `StoreBuilder`, `StoreData`, `ZarrStoreReader` |
| `molrs-core/src/io/zarr/mod.rs` | old re-exports |
| `molrs-core/src/lib.rs:97` | old `pub use io::zarr::{...}` line |
| `molrs-wasm/src/io/zarr/mod.rs` | `ZarrReader` / `ZarrWriter` (rewrite for new API) |
| `molrs-core/tests/reproduce_wasm.rs` | old `ZarrStoreReader` test (rewrite) |

## Test Criteria

### Unit Tests

1. **System round-trip**: create store -> `read_system()` -> compare blocks, meta,
   simbox.
2. **ForceField round-trip**: write FF with atom/bond/angle/pair/dihedral/improper/
   kspace styles -> read -> compare all names, categories, params.
3. **Trajectory round-trip**: write 100 frames -> `read_positions(t)`,
   `read_scalar("PE", t)` at specific indices -> verify.
4. **Partial read**: write 1000 frames -> `read_positions(500)` -> only that frame
   loaded.
5. **Scalar series**: `read_scalar_series("PE")` -> all values.
6. **No trajectory**: create with system only -> `open_trajectory()` returns `None`.
7. **MemoryStore**: write to filesystem -> load into MemoryStore -> `open_store()`
   -> verify system, FF, trajectory.

### Archive Tests

8. **List/open**: create archive with 3 simulations -> `list_simulations()` -> open
   each -> verify independent data.
9. **Remove**: `remove_simulation()` -> no longer in list.
10. **Isolation**: modifying trajectory of one simulation does not affect another.

### Numerical

11. **Precision**: f32/f64 arrays round-trip bit-exact.
12. **Chunk boundary**: 25 frames, chunk_size=10, read frames 9/10/11.

## Performance

| Operation | Target |
|-----------|--------|
| `create_file()` | < 10ms for 10K atoms |
| `read_system()` | < 10ms for 10K atoms |
| `read_forcefield()` | < 5ms for ~100 types |
| `append_frame()` | < 1ms per frame (10K atoms) |
| `read_positions(t)` | < 2ms for 10K atoms |
| `read_frame(t)` | < 10ms for 10K atoms |
| `list_simulations()` | < 1ms |
