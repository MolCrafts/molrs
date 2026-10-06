# Record files (`*.mrec`)

A molrs **record** is one self-describing package on disk: a `meta` document
plus one or more sections — a snapshot (`frame`), a topology (`system`), a
frame sequence (`trajectory`), a force field (`forcefield`), observables, or a
run `status`. It is a directory whose name ends in `.mrec`, with every array
stored as Zarr V3, and a closed directory can be packed into one
`*.mrec.zip` file.

molrs writes and reads records by the [molrec](https://docs.molcrafts.org/molrec/)
contract. The [layout chapter](https://docs.molcrafts.org/molrec/layout/)
names every group, array and attribute; this page shows how molrs maps onto
it. The examples are Python, but the Rust functions in `molrs::io::mrec` are
the same doors with the same rules, and the WASM package reads the same files.

| What you have | Write | Read |
| --- | --- | --- |
| A `Frame` (snapshot) | `molrs.io.write_mrec` | `molrs.io.read_mrec` |
| A `Frame` (topology) | `molrs.io.write_mrec_system` | `molrs.io.read_mrec_system` |
| A `Trajectory` in memory | `molrs.io.write_mrec_trajectory` | `molrs.io.read_mrec_trajectory` |
| A run too large for memory | `molrs.io.mrec.TrajectoryWriter` | `molrs.io.mrec.TrajectoryReader` |
| A `ForceField` | `molrs.io.write_mrec_forcefield`, or `forcefield=` on `write_mrec` | `molrs.io.read_mrec_forcefield` |
| Any record | — | `molrs.io.mrec_sections`, `molrs.io.read_mrec_meta` |

## Write and read a frame

Build a small frame. Canonical columns (`x`, `element`, `res_id`, `chain`, …)
adopt their schema dtype on insert, so `res_id` below is stored as `uint64`
however the array was spelled.

```python
import numpy as np
import molrs

atoms = molrs.Block({
    "element": ["O", "H", "H"],
    "x": np.array([0.000, 0.757, -0.757]),
    "y": np.array([0.000, 0.586, 0.586]),
    "z": np.zeros(3),
    "res_id": np.array([1, 1, 1]),
    "res_name": ["HOH", "HOH", "HOH"],
    "chain": ["A", "A", "A"],
})
bonds = molrs.Block({
    "atomi": np.array([0, 0], dtype=np.uint64),
    "atomj": np.array([1, 2], dtype=np.uint64),
})
frame = molrs.Frame(
    {"atoms": atoms, "bonds": bonds},
    box=molrs.Box.cube(20.0),
)
frame.meta["title"] = "water"
frame.meta["temperature"] = 300.0

print(frame["atoms"].dtype("res_id"))
```

`write_mrec` writes the `frame` section and a `meta` document. Each writer
stamps the current `molrec_version` (2) into `meta`, over any value you
supplied. A record molrs ≤ 0.15 wrote (`molrec_version` 1) still reads: its
force-field numbers are converted to the force-field IR on the way in,
exactly, or the record is refused (see the
[migration guide](../migration.md#records-molrec_version-2)).

```python
molrs.io.write_mrec("water.mrec", frame, meta={"producer": "quickstart"})

print(sorted(molrs.io.mrec_sections("water.mrec")))
print(molrs.io.read_mrec_meta("water.mrec"))

back = molrs.io.read_mrec("water.mrec")
print(back["atoms"]["chain"], back["atoms"]["res_id"])
print(back.box.lengths)
```

Frame meta is **typed on disk**: the writer adds a `_meta_types` attribute,
so every value reads back at the dtype it was stored with, `float("nan")`
included. A JSON object comes back as a frozen `MetaDocument`.

```python
frame.meta["n_steps"] = molrs.MetaValue("i32", 5000)
frame.meta["run"] = {"ensemble": "NVT", "thermostat": "langevin"}
molrs.io.write_mrec("water.mrec", frame)

back = molrs.io.read_mrec("water.mrec")
print(back.meta.dtype("n_steps"), back.meta["n_steps"])
print(back.meta["run"]["ensemble"])
```

A topology goes in the `system` section, next to or instead of a snapshot:
`write_mrec(path, frame, system=topology)` writes both, and
`write_mrec_system` / `read_mrec_system` handle a topology on its own. Each
read decodes `meta` and its own section only, so a damaged trajectory in the
same store cannot fail `read_mrec`.

## Topology conventions

The molrec conventions give fixed names to the residue and chain facts the
format readers produce. `chain`, `res_id`, `res_name`, `icode`, `altloc`,
`occupancy` and `b_factor` are canonical `atoms` columns; the PDB, mmCIF, GRO
and extxyz readers fill them under those names. Extra blocks with fixed
meanings are `constraints`, `virtual_sites`, `drudes` and `members` (bead →
atom membership in a coarse-grained frame). `molrs.schema` prints the whole
vocabulary.

A `uint64` column can declare which block its values index. Writers store
that as the block's `targets` attribute and refuse a reference that does not
resolve, so a dangling index fails at write time instead of at analysis time:

```python
contacts = molrs.Block({
    "site": np.array([0, 2], dtype=np.uint64),
    "distance": np.array([2.8, 3.1]),
})
contacts.set_target("site", "atoms")
frame["contacts"] = contacts
molrs.io.write_mrec("water.mrec", frame)

print(molrs.io.read_mrec("water.mrec")["contacts"].targets())
```

## Declared precision

Coordinates rarely carry 16 significant digits of information. An `f64`
column can declare an absolute precision `p` in its own units. The writer then
stores each value rounded to a binary grid — the largest power of two `q ≤ p`,
so the error is at most `p / 2` — and compresses the column with byte shuffle
plus zstd. The declaration is stored with the column and reads back.

```python
for key in ("x", "y", "z"):
    frame["atoms"].set_precision(key, 1e-3)
molrs.io.write_mrec("water.mrec", frame)

back = molrs.io.read_mrec("water.mrec")
print(back["atoms"].precision("x"))
print(np.abs(back["atoms"]["x"] - frame["atoms"]["x"]).max() <= 0.5e-3)
```

On a 3000-atom trajectory moving 0.05 Å per frame, lossless coordinates
take 24 B/atom/frame; `p = 1e-3` Å takes about 7.6 and `p = 1e-2` Å about
5.8. Without a declaration nothing is rounded. Memory is never touched: only
the stored copy is rounded. A store with a declared precision needs a reader
that decodes zstd and shuffle, as every molrs build since 0.15 does (the WASM
package included); molrs 0.14 cannot read it.

## Trajectories

A `Trajectory` that fits in memory goes through one call each way:

```python
frames = []
for i in range(4):
    f = frame.copy()
    f["atoms"]["x"] = frame["atoms"]["x"] + 0.01 * i
    frames.append(f)

traj = molrs.Trajectory(
    frames,
    step=np.arange(4, dtype=np.int64) * 100,
    time=np.arange(4, dtype=np.float64) * 0.2,
)
molrs.io.write_mrec_trajectory("run.mrec", traj, meta={"producer": "quickstart"})

loaded = molrs.io.read_mrec_trajectory("run.mrec")
print(len(loaded), loaded.step)
```

Only what changes between frames is written: a block identical to the previous
frame's is not stored again.

### Streaming with a pinned schema

A long run is written frame by frame. Pin a `SequenceSchema` first — the
blocks, columns and dtypes every frame will carry — then append. Derive the
schema from a representative frame and add declarations:

- `declare_precision(block, column, p)` — round that column on every frame,
  as above. A change smaller than half a grid step is not an update, so a
  rattling-but-static block costs nothing.
- `declare_target(block, column, target)` — a `uint64` column indexes rows of
  `target`.
- `declare_aligned(block, target)` — `block` has one row per row of `target`
  at every frame (per-atom forces next to `atoms`). The writer refuses a frame
  that breaks it.

```python
from molrs.io.mrec import SequenceSchema, TrajectoryReader, TrajectoryWriter

schema = SequenceSchema.from_frame(frames[0])
for key in ("x", "y", "z"):
    schema.declare_precision("atoms", key, 1e-3)

with TrajectoryWriter("stream.mrec", schema, meta={"producer": "quickstart"}) as writer:
    for i, f in enumerate(frames):
        writer.append(f, step=100 * i, time=0.2 * i)

with TrajectoryReader("stream.mrec") as reader:
    print(len(reader), reader.step)
    last = reader[len(reader) - 1]
    print(last["atoms"]["x"])
```

`TrajectoryReader` decodes one frame per access, so a store larger than
memory can be walked with a plain `for frame in reader:` loop. A writer
flushes every `flush_every` frames; if the process dies, the frames already
committed are readable and `TrajectoryWriter.open(path)` resumes appending.

### Packing

A closed directory store packs into a single `*.mrec.zip` — one file to copy,
upload or serve. `pack` replaces the directory with the archive and returns
its path; `TrajectoryReader` and the WASM readers open the archive directly.

```python
from molrs.io.mrec import pack

zipped = pack("stream.mrec")
print(zipped)

with TrajectoryReader(zipped) as reader:
    print(len(reader))
```

## Force fields

A `forcefield` section stores a force field as data: a document (styles,
units, mixing rule, special-bond weights) and one table per style. Units are
recorded as declared and read back as recorded; only a `molrec_version` 1
section's numbers are converted, as above. `ForceField.to_section` and
`ForceField.from_section` map between the two, and every record writer takes
a force field next to the structure it parameterizes:

```python
ff = molrs.ff.ForceField("water", units="real")
atom_style = ff.def_style("atom", "full")
o = atom_style.def_type("OW", mass=15.9994, charge=-0.8476)
h = atom_style.def_type("HW", mass=1.008, charge=0.4238)
ff.def_style("bond", "harmonic").def_type("OW-HW", o, h, k=450.0, r0=1.0)
ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type(
    "OW", o, epsilon=0.1553, sigma=3.166
)

molrs.io.write_mrec("water.mrec", frame, forcefield=ff)
print(sorted(molrs.io.mrec_sections("water.mrec")))

section = molrs.io.read_mrec_forcefield("water.mrec")
print(section.name, sorted(section.tables))
restored = molrs.ff.ForceField.from_section(section)
print([(s.category, s.name) for s in restored.styles])
```

A pair style's table holds each atom type's own row (`itom == jtom`) and any
explicit cross row (`itom != jtom`, e.g. a CHARMM NBFIX), which prices that
type pair in place of the style's mixing rule. Both kinds round-trip. A
pair is found by its two types in either order, so a table holds one row per
pair: a section restating a pair with other parameters is refused on read.

`read_mrec_forcefield` returns `None` for a record without a force field.
`write_mrec_forcefield(path, ff)` writes a force-field package with no
structure at all.

## From Rust and the browser

The Rust doors live in `molrs::io::mrec` (features `zarr` and
`filesystem`): `write_frame_file` / `read_frame_file`, `write_system_file` /
`read_system_file`, `write_trajectory_file` / `read_trajectory_file`,
`write_forcefield_file` / `read_forcefield_file`, `section_names`, and for
streaming `SequenceSchema`, `FrameSequenceWriter`, `FrameSequence` and
`pack`. Their rustdoc on [docs.rs](https://docs.rs/molcrafts-molrs) carries
compiled examples.

```rust
use molrs::io::mrec::{read_frame_file, section_names, write_frame_file};

fn main() -> Result<(), molrs::MolRsError> {
    write_frame_file("water.mrec", &molrs::Frame::new(), None, None)?;
    let frame = read_frame_file("water.mrec")?;
    println!("{:?} {}", section_names("water.mrec")?, frame.len());
    Ok(())
}
```

In the browser, `@molcrafts/molrs` reads records from bytes:
`readMrecFrame` / `readMrecFrameFromZip` for a snapshot, `mrecSections` to
list sections, and `TrajectoryReader` (from a file map, a `*.mrec.zip`, or a
lazy range-request store) for trajectories. See the
[package README](https://github.com/MolCrafts/molrs/tree/master/molrs-wasm#trajectory-stores-mrec).
