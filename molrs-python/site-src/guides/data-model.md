# Data Model

molrs uses a table-oriented molecular model. A `Frame` is the unit that moves
through I/O, analysis, and visualization. Inside a frame, each named `Block`
holds columns of equal row count. The conventional `atoms` block stores one row
per atom, while optional blocks such as `bonds`, `angles`, and `dihedrals`
describe topology.

This structure is intentionally less rigid than a fixed molecule class. A PDB
reader can attach atom names and residue identifiers without forcing every
simulation pipeline to carry them. A trajectory analysis can add derived
columns without rewriting the topology. The contract is local: columns in the
same block agree on row count, and code that needs specific columns checks for
them at the subsystem boundary.

## A frame holds named blocks of columns

`Frame` is a dictionary-like container. `Block` is a column store backed by
typed arrays. In Python those columns are NumPy arrays; in WASM they appear as
typed arrays or memory-backed views; in Rust they are ndarray-backed columns.
The same column names carry across bindings. Coordinates use `x`, `y`, and `z`
columns on the `atoms` block.

The scientific precision contract is simple: floating-point data is `f64`.
Older feature flags that suggested switching the workspace to `f32` are now
deprecated no-ops. This matters for interop because external packages often
default to single precision for visualization, while molrs algorithms assume
double precision for geometry and force-field work.

## A box adds periodicity to a frame

A simulation box (`SimBox`, exposed as `molrs.Box` in Python) is attached to a
frame when periodic boundary conditions matter. The box stores the lattice
matrix, origin, and per-axis periodic flags. Neighbor search, wrapping,
distance calculations, and RDF normalization read the same box metadata, so
changing it changes the physical interpretation of the coordinates. Minimum
image convention is a cheap `Mic` value derived from the box — not a separate
global setting.

**Regions are not boxes.** Spatial query volumes (`Sphere`, `Cuboid`,
`Parallelepiped`, boolean combinations) live under `spatial::region` and answer
"is this point inside?" questions for packing and selection. A `Parallelepiped`
describes a general triclinic *region*; a `SimBox` describes the periodic
lattice of a simulation cell. Do not use one where the other is required.

Frames without a simulation box are still valid. Several analysis paths can
fall back to a free-boundary box inferred from coordinates and padding. That is
useful for isolated molecules and point clouds, but periodic systems should
carry an explicit box to avoid accidental finite-volume assumptions.

## Worked Example: Build a Frame by Hand

This example creates a water molecule as a coordinate table. It does not create
a full molecular graph; it creates the data shape that readers, writers, and
analysis routines expect.

```python
import numpy as np
import molpy

atoms = molpy.Block()
atoms.insert("element", ["O", "H", "H"])
atoms.insert("x", np.array([0.0000, 0.9572, -0.2399], dtype=np.float64))
atoms.insert("y", np.array([0.0000, 0.0000,  0.9266], dtype=np.float64))
atoms.insert("z", np.array([0.0000, 0.0000,  0.0000], dtype=np.float64))
atoms.insert("mass", np.array([15.999, 1.008, 1.008], dtype=np.float64))

frame = molpy.Frame()
frame["atoms"] = atoms
frame.box = molpy.Box.cube(30.0)
frame.meta = {"name": "water", "source": "docs"}

frame.validate()
print(frame.keys())
print(frame["atoms"].nrows)
print(frame.box.lengths())
```

Expected output shape:

```text
['atoms']
3
[30. 30. 30.]
```

The coordinate columns are separate because many file formats and simulation
engines store them that way. If another library wants an `N x 3` matrix,
construct it at the boundary:

```python
xyz = np.column_stack(
    [atoms.view("x"), atoms.view("y"), atoms.view("z")]
).astype(np.float64, copy=False)
print(xyz.shape)  # (3, 3)
```

## Add a Topology Block

Topology in a frame is just another block. Bond indices are zero-based because
they refer to row positions in the `atoms` block.

```python
bonds = molpy.Block()
bonds.insert("i", np.array([0, 0], dtype=np.uint32))
bonds.insert("j", np.array([1, 2], dtype=np.uint32))
bonds.insert("order", np.array([1.0, 1.0], dtype=np.float64))
frame["bonds"] = bonds

print(frame.keys())
print(frame["bonds"].view("i"), frame["bonds"].view("j"))
```

This representation is deliberately plain. A force-field typifier can inspect
the graph-level topology, while writers can serialize frame-level topology
without needing to understand every chemistry operation that created it.

## Common Mistakes

The most common shape error is mixing column lengths in one block. `Block`
allows a two-dimensional column such as a `(N, 3)` position matrix, but its
leading dimension must still be `N`, the row count shared with the other
columns.

Another common mistake is using `float32` because a visualization library
returns it. molrs accepts some numeric inputs flexibly, but scientific
algorithms are written around double precision. Convert to `np.float64` before
storing coordinates used for geometry, force fields, or analysis.
