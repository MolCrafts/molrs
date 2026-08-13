# Python Quickstart

This quickstart follows a complete Python workflow: parse a molecule, generate
three-dimensional coordinates, convert to a `Frame`, attach a simulation box,
build a neighbor list, run RDF, evaluate MMFF94 energy, and write an XYZ file.

The goal is not to memorize every class. The goal is to see the boundary
between the graph representation (`Atomistic`) and the table representation
(`Frame`), because most molrs workflows cross that boundary deliberately.

## 1. Parse a Molecule

`molrs.io.SmilesIR` returns an intermediate representation. Convert it to
`Atomistic` when you want a graph with atoms and bonds.

```python
import molrs

ir = molrs.io.SmilesIR("CCO")  # ethanol
mol = ir.to_atomistic()

print("components:", ir.n_components)
print("heavy atoms:", mol.n_atoms)
print("bonds:", mol.n_bonds)
```

Expected output shape:

```text
components: 1
heavy atoms: 3
bonds: 2
```

At this stage there are no generated coordinates and implicit hydrogens are
not yet explicit graph nodes. That is why the next step runs the embedding
pipeline instead of trying to read `x`, `y`, and `z` columns from the graph.

## 2. Generate 3D Coordinates

Embedding converts topology into coordinates. Use a seed in examples so that
the result is reproducible across runs.

```python
mol3d, report = molrs.conformer.Conformer(speed="fast", seed=42).generate(mol)

print("atoms after embedding:", mol3d.n_atoms)
print("final energy:", report.final_energy)
print("stages:", [stage.stage for stage in report.stages])
```

`Conformer.generate` returns a plain `(mol, report)` tuple: construct the
generator once with your parameters, then call `generate` for each molecule.

## 3. Convert to a Frame

Writers and analyses operate on frames. A frame is a dictionary-like container
of named blocks. The `atoms` block holds coordinate columns and element data.

```python
frame = mol3d.to_frame()
atoms = frame["atoms"]

print("frame blocks:", frame.keys())
print("atom columns:", atoms.keys())
print("rows:", atoms.nrows)
print("first x values:", atoms.view("x")[:3])
```

You should see an `atoms` block and usually a `bonds` block. Coordinate columns
are one-dimensional double-precision NumPy arrays. If you build a frame by
hand, use `np.float64` for scientific floating-point columns.

## 4. Attach a Periodic Box

Several analyses need to know whether coordinates are periodic. Attach a
`Box` before building neighbor lists if the system should be interpreted as a
periodic simulation cell.

```python
import numpy as np

frame.box = molrs.Box.cube(
    20.0,
    pbc=np.array([True, True, True], dtype=np.bool_),
)

print("box lengths:", frame.box.lengths())
print("box volume:", frame.box.volume())
```

The box is in the same length unit as your coordinates. molrs does not silently
convert between nanometer and angstrom conventions. Pick a unit convention for
the workflow and keep it consistent.

## 5. Build a Neighbor List and RDF

Neighbor search turns coordinates into pair lists. RDF then consumes the frame
and the neighbor list rather than doing its own distance search.

```python
points = np.column_stack(
    [atoms.view("x"), atoms.view("y"), atoms.view("z")]
).astype(np.float64, copy=False)

nl = molrs.NeighborList(6.0)
nl.build(points, frame.box)
neigh = nl.neighbors()

print("pairs:", nlist.n_pairs)
print("first pairs:", nlist.query_point_indices()[:5], nlist.point_indices()[:5])

from molrs.compute.density import RDF
rdf = RDF(64, 6.0)
rdf_result = rdf.compute(frame, nlist)
print("rdf bins:", len(rdf_result.bin_centers))
print("first g(r):", rdf_result.rdf[:5])
```

For a single ethanol molecule in a large box, the RDF is just a small example
of the API shape. For real RDF work, pass frames from a trajectory and build
neighbor lists with the same cutoff and boundary assumptions for each frame.

## 6. Evaluate MMFF94 Energy

Force-field evaluation starts from the molecular graph, not from arbitrary
coordinate tables. The typifier creates a compiled potential set compatible
with the graph. Coordinates are then extracted from the frame as a flat `3N`
array.

```python
typifier = molrs.ff.MMFF94Typifier()
typed = typifier.typify(mol3d)
typed_frame = typed.to_frame()
print("typed blocks:", typed_frame.keys())

try:
    # Non-bonded terms need an explicit pairs block (no optimizeGeometry sugar).
    typed_frame["pairs"] = molrs.ff.intramolecular_pairs(typed_frame)
    potentials = typifier.forcefield().to_potentials(typed_frame)
    coords = molrs.ff.extract_coords(typed_frame)

    energy, forces = potentials.calc_energy_forces(coords)
    print("energy:", energy)
    print("coords shape:", coords.shape)
    print("forces shape:", forces.shape)
except ValueError as exc:
    print("potential build skipped:", exc)
```

MMFF94 typing and potential compilation are separate steps on purpose. Typing
gives a labeled graph; `forcefield().to_potentials(frame)` is the shared
compile path every force field uses (including UFF on Rust/WASM). Potential
compilation is stricter because every term must resolve to a supported
parameter — some molecules can typify successfully while compilation still
reports incomplete coverage.

When potential compilation succeeds, the coordinate shape is `(3 * n_atoms,)`,
not `(n_atoms, 3)`. Reshape only for display:

```python
forces_xyz = forces.reshape(mol3d.n_atoms, 3)
print("force balance:", np.abs(forces_xyz.sum(axis=0)).max())
```

## 7. Write an XYZ File

The I/O layer writes frames. This is the final boundary where the graph-based
work has become a portable coordinate table.

```python
molrs.io.write_xyz("ethanol.xyz", frame)
roundtrip = molrs.io.read_xyz("ethanol.xyz")
print("roundtrip atoms:", roundtrip["atoms"].nrows)
```

The XYZ format stores coordinates and element symbols, but it does not preserve
the full force-field state or every topology detail. Use richer formats or
frame-sequence Zarr when a workflow needs trajectory data.

## Summary

This quickstart crossed the main molrs boundaries:

- SMILES text became a graph-like `Atomistic`.
- `Conformer.generate` produced coordinates and diagnostics.
- `to_frame` produced the columnar representation used by I/O and analysis.
- `Box` supplied the boundary model for neighbor search.
- `RDF` consumed an explicit neighbor list.
- `MMFF94Typifier` compiled potentials for energy and force evaluation.
