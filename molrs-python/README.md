# molcrafts-molrs

[![PyPI](https://img.shields.io/pypi/v/molcrafts-molrs.svg)](https://pypi.org/project/molcrafts-molrs/)

Python bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

Install with `pip install molcrafts-molrs` and `import molrs`. Full docs:
<https://docs.molcrafts.org/molrs/>.

## Install

```bash
pip install molcrafts-molrs
```

Requires Python 3.12+.

## Quick start (0.13 surface)

```python
import numpy as np
import molrs

# SMILES → atomistic graph (class API under molrs.io)
mol = molrs.io.SmilesIR("CCO").to_atomistic()

# 3D coordinates
from molrs.conformer import Conformer

mol = Conformer().generate(mol)

# Force field: typify → pairs → potentials (no typifier.build())
from molrs.ff import MMFF94Typifier, extract_coords, intramolecular_pairs

typed = MMFF94Typifier().typify(mol)
frame = typed.to_frame()
frame["pairs"] = intramolecular_pairs(frame)
pots = MMFF94Typifier().forcefield().to_potentials(frame)
coords = extract_coords(frame)
energy, forces = pots.calc_energy_forces(coords)
assert forces.shape == (frame["atoms"].nrows, 3)
```

## Package layout

| Import | Owns |
|--------|------|
| `molrs` (top level) | Core: `Frame`, `Block`, `Atomistic`, `Box`, neighbors, … |
| `molrs.io` | Readers/writers, `SmilesIR` |
| `molrs.ff` | Force fields, typifiers, potentials |
| `molrs.compute` | RDF, MSD, transport, dielectric, … |
| `molrs.conformer` | 3D generation |
| `molrs.perceive` | Rings / aromaticity builder |

Users of analysis APIs: analysis time is **fs** (LAMMPS `real`). MSD needs
**unwrapped** coordinates. VACF is the unbiased \(C(\tau)\) used for Green–Kubo D
and VDOS.

## Development

```bash
maturin develop --release
pytest -q
```
