# In-process MD

Take constants from `molpy.UnitPreset`, build a neighbour skin, attach `LJCut`,
and step `VelocityVerlet`. The engine is unit-agnostic.

```python
import numpy as np
import molpy
from molpy import Box, md

kb = molpy.UnitPreset("real").boltzmann()
pos = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
rc, skin = 7.5, 1.0
nl = md.VerletSkin(
    md.NeighborList(rc + skin), rc, pos, Box.cube(20.0), skin=skin
)
eps = 0.238  # caller units; MD does not convert
vv = md.VelocityVerlet(
    1.0, potential=md.LJCut(eps, 3.405, rc), neighbors=nl, mass=np.full(2, 39.948)
)
state = vv.initial(pos, np.zeros_like(pos))
state = vv.advance_n(state, 100)
```

Precision is `md.MD(dtype=np.float64)` only. `thermo=` needs an explicit `kb=`:

```python
import molpy
from molpy import md

driver = md.MD(dtype=np.float64)
driver.set_forcefield(ff).set_neighbors(cutoff=7.5, skin=2.0)
state = driver.run(
    frame, 1000, dt=1.0, kb=molpy.UnitPreset("real").boltzmann(), thermo=100
)
```

## Custom forces

`molpy.md.Potential` is a `runtime_checkable` Protocol. Subclassing is optional
table-stakes; a duck-typed object is enough:

```python
class Spring:
    def calc_energy_forces(self, pos):
        return 0.05 * float((pos * pos).sum()), -0.1 * pos

vv = md.VelocityVerlet(1.0, potential=Spring(), mass=np.ones(2))
```
