# I/O Formats

File I/O maps external molecular formats into `Frame`. The conversion is not a
lossless universal schema; each format carries a different mix of topology,
coordinates, unit-cell data, and metadata. molrs keeps the mapping explicit by
placing parsed columns into named blocks and attaching a simulation box when
the source format provides one.

PDB and XYZ are common coordinate-oriented inputs. LAMMPS data and dump files
carry simulation-oriented structure. CHGCAR and Gaussian Cube introduce grid
data in addition to atoms. Zarr-based frame-sequence storage keeps trajectories
in a layout designed for repeated analysis.

## Reader Expectations

Readers should be treated as boundary code. They validate enough structure to
produce a frame, but domain assumptions still belong to the caller. A file can
be syntactically valid while missing the columns required for a later force
field or neighbor query.

For development, happy-path I/O tests use real files from the shared
`tests-data` repository. Hand-written strings are reserved for narrow parser
edge cases. This keeps the docs and tests aligned with files that users are
likely to encounter.

## Worked Example: Read, Inspect, Write

```python
from pathlib import Path
import molpy

frame = molpy.io.read_xyz("water.xyz")
atoms = frame["atoms"]

print("blocks:", frame.keys())
print("rows:", atoms.nrows)
print("columns:", atoms.keys())
print("first atom:", atoms.view("element")[0])

out = Path("water-copy.xyz")
molpy.io.write_xyz(str(out), frame)
print("wrote:", out)
```

The reader produces a frame; the writer consumes a frame. The format controls
which columns survive. XYZ preserves element symbols and coordinates, while
LAMMPS formats can carry more simulation topology and box information.

### XYZ connectivity extension

MolCrafts hand-written XYZ can preserve simple topology with a `Connct` field
on the comment line:

```text
3
name=water Connct="[0,1,0,2]"
O  0.0000  0.0000  0.0000
H  0.9572  0.0000  0.0000
H -0.2390  0.9266  0.0000
```

`Connct` is a flat list of zero-based atom-index pairs, so this example defines
bonds 0–1 and 0–2. Each pair becomes one row in the frame's `bonds` block using
the canonical `atomi` and `atomj` columns. Bond order is implicitly one. The XYZ
writer emits the same field when a frame already has canonical bond columns.

## Format Expectations

| Format | Typical content | Notes |
| --- | --- | --- |
| PDB | atoms, residues, coordinates, optional box | Good for structural biology interop, not a complete force-field container |
| XYZ | element symbols, coordinates, optional `Connct` pairs | Simple coordinate snapshots with a MolCrafts single-bond topology extension |
| LAMMPS data | atoms, bonds, box, simulation topology | Used for simulation setup and engine interop |
| LAMMPS `*.ff` include | pair / bond / angle / dihedral / improper coeffs | AMBER/GAFF flavour; see [Force Fields](force-field.md#lammps-ff-includes) for unit conversion |
| LAMMPS dump | trajectory frames | Often read lazily when files are large |
| CHGCAR / Cube | atoms plus grid fields | Useful for volumetric scalar data |
| Trajectory / Zarr | frames, trajectories | Designed for repeated analysis and richer metadata |

## Lazy LAMMPS Trajectory Access

Large trajectory files should not always be read into memory at once.
`LAMMPSTrajReader` builds an index and reads frames on demand:

```python
reader = molpy.io.raw.LAMMPSTrajReader("dump.lammpstrj")
print("frames:", len(reader))

first = reader[0]
last = reader.read_step(len(reader) - 1)

print(first["atoms"].nrows)
print(last["atoms"].nrows if last is not None else "missing")
```

Use `read_lammps_traj` when the file is small and a list of frames is more
convenient. Use `LAMMPSTrajReader` when you want random access or streaming
behavior.

## Why Tests Use Real Files

Format readers fail in the details: optional records, whitespace, indexing
conventions, element naming, triclinic boxes, and incomplete metadata. Synthetic
strings rarely cover those details. molrs therefore tests happy paths against
real files from the shared test-data repository and keeps hand-written snippets
for narrow malformed-input cases.
