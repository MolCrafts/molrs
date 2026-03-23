# molcrafts-molrs

[![PyPI](https://img.shields.io/pypi/v/molcrafts-molrs.svg)](https://pypi.org/project/molcrafts-molrs/)

Python bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

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
atoms.insert("symbol", ["O", "H", "H"])
frame["atoms"] = atoms
```

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
target = molrs.Target(frame, count=100).with_name("water")
result = molrs.Packer(tolerance=2.0, precision=0.01).pack([target])
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
