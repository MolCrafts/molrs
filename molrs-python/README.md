# molcrafts-molrs

[![PyPI](https://img.shields.io/pypi/v/molcrafts-molrs.svg)](https://pypi.org/project/molcrafts-molrs/)

Python bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

This package is the first public preview of the Python API. Install with
`pip install molcrafts-molrs` and import it as `molrs`.

Full documentation lives at <https://docs.molcrafts.org/molrs/>. The Python
API reference is rendered at <https://docs.molcrafts.org/molrs/reference/python/>.

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
mol = ir.to_atomistic()
result = molrs.generate_3d(mol, molrs.EmbedOptions(speed="fast", seed=42))
frame = result.mol.to_frame()

# Build a system from scratch
frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", np.array([0.0, 0.96, -0.24], dtype=np.float64))
atoms.insert("y", np.array([0.0, 0.0, 0.93], dtype=np.float64))
atoms.insert("z", np.zeros(3, dtype=np.float64))
atoms.insert("element", ["O", "H", "H"])
frame["atoms"] = atoms
```

## Support

- Python 3.9+
- `pip install molcrafts-molrs`
- `import molrs`
- Offline API help is available with `help(molrs.Frame)` and related symbols

## API

### Data model

- **`Frame`** — dict-like container of named `Block`s + optional `Box`
- **`Block`** — column store backed by numpy arrays
- **`Box`** — simulation box with periodic boundaries

### I/O

- `molrs.read_pdb(path)` / `molrs.read_xyz(path)` → `Frame`
- `molrs.parse_smiles(smiles)` → `SmilesIR` → `.to_frame()`

### Neighbor search and analysis

```python
nq = molrs.NeighborQuery(box, positions, cutoff=5.0)
nlist = nq.query_self()

from molrs.compute.density import RDF
rdf = RDF(100, 5.0)
result = rdf.compute(frame, nlist)
```

### Force field

```python
typifier = molrs.MMFF94Typifier()
potentials = typifier.build(atomistic)
```

## Development

```bash
maturin build
pip install target/wheels/*.whl
pytest -q
```

| Component | Version |
| --- | --- |
| Python | 3.9+ |
| PyO3 | Managed by `molrs-python/Cargo.toml` |
| maturin | 1.x |

## License

BSD-3-Clause
