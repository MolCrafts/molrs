# molcrafts-molrs

[![PyPI](https://img.shields.io/pypi/v/molcrafts-molrs.svg)](https://pypi.org/project/molcrafts-molrs/)

Python bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

This package is the first public preview of the Python API. Install with
`pip install molcrafts-molrs` and import it as `molrs`.

## Install

```bash
pip install molcrafts-molrs
```

## Quick start

```python
import numpy as np
import molrs

# Parse SMILES and generate 3D coordinates
ir = molrs.parse_smiles("CCO")
frame = ir.to_frame()
result = molrs.generate_3d(frame)

# Build a system from scratch
frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", np.array([0.0, 0.96, -0.24], dtype=np.float32))
atoms.insert("y", np.array([0.0, 0.0, 0.93], dtype=np.float32))
atoms.insert("z", np.zeros(3, dtype=np.float32))
atoms.insert("element", ["O", "H", "H"])
frame["atoms"] = atoms
```

## Support

- Python 3.9+
- `pip install molcrafts-molrs`
- `import molrs`
- Packing users should prefer explicit `seed=` for reproducible placements

## API

### Data model

- **`Frame`** — dict-like container of named `Block`s + optional `Box`
- **`Block`** — column store backed by numpy arrays
- **`Box`** — simulation box with periodic boundaries

### I/O

- `molrs.read_pdb(path)` / `molrs.read_xyz(path)` → `Frame`
- `molrs.parse_smiles(smiles)` → `SmilesIR` → `.to_frame()`

### Neighbor search (freud-style)

```python
nq = molrs.AABBQuery(box, positions, cutoff=5.0)
nlist = nq.query_self()              # self-query (unique pairs)
nlist = nq.query(query_positions)    # cross-query

nlist.query_point_indices   # np.array, uint32
nlist.point_indices         # np.array, uint32
nlist.distances             # np.array, float
```

### Analysis

```python
rdf = molrs.RDF(bins=100, r_max=5.0)
result = rdf.compute(nlist, box)     # auto self/cross normalization

msd = molrs.MSD.from_reference(ref_frame)
result = msd.compute(frame)

cluster = molrs.Cluster(min_size=5)
result = cluster.compute(frame, nlist)
```

### Molecular packing

```python
target = (
    molrs.Target(frame, count=100)
    .with_name("water")
    .with_constraint(molrs.InsideBox([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]))
    .with_constraint_for_atoms([1], molrs.AbovePlane([0.0, 0.0, 1.0], 5.0))
    .constrain_rotation_z(0.0, 20.0)
)

packer = (
    molrs.Packer(tolerance=2.0, precision=0.01)
    .with_nloop0(40)
    .with_pbc_box([40.0, 40.0, 40.0])
    .with_progress(False)
)
result = packer.pack([target], max_loops=200, seed=42)
print(result.natoms, result.converged)
```

### Force field

```python
typifier = molrs.MMFFTypifier()
potentials = typifier.build(atomistic)
```

## Development

```bash
maturin build
pip install target/wheels/*.whl
pytest -q
```

## License

BSD-3-Clause
