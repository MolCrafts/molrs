# Migrating to 0.14

User-facing spelling is `molpy`. The engine is the Rust crate; application
code imports `molpy`.

## Units

The engine no longer converts energy or Boltzmann's constant for you.

```python
import molpy

kb = molpy.UnitPreset("real").boltzmann()
eps = (
    molpy.UnitRegistry()
    .quantity(0.238, "kilocalorie_per_mole")
    .to("amu * angstrom ** 2 / femtosecond ** 2")
    .value
)
```

## MD driver

`MaxwellBoltzmann` takes `kbt` (not temperature). `MD` takes `dtype=`. Thermo
sampling needs an explicit `kb=`.

```python
import numpy as np
from molpy import md

kb = molpy.UnitPreset("real").boltzmann()
md.MaxwellBoltzmann(kb * 300.0, seed=0)
driver = md.MD(dtype=np.float64)
driver.run(frame, n, dt=dt, kb=kb, thermo=100)
```

`Potential` is a `runtime_checkable` Protocol: any object with
`calc_energy_forces(pos)` is accepted. Do not subclass a compiled base class.

## Record / Frame store

Public names describe the object, not the storage backend.

```python
rec = molpy.Record()
rec.write(path)
loaded = molpy.Record.read(path)
frame.meta["note"] = "x"
```

On-disk `format_name` stays `"molrec"`. That is a file-format value, not an
API name. `Trajectory.read` / `Trajectory.write` follow the same rule.

## Neighbors

The loop owns rebuilds. Pair MIC is computed once, in `VerletSkin.pairs_at`,
and shared by every pair potential.

```python
from molpy import Box, md

nl = md.VerletSkin(
    md.NeighborList(rc + skin), rc, pos, Box.cube(20.0), skin=skin
)
vv = md.VelocityVerlet(1.0, potential=md.LJCut(eps, 3.405, rc), neighbors=nl, mass=mass)
```

There is one `LJCut`, re-exported at `molpy.md.LJCut`.

## ForceField categories

PME is a pair style, not a ForceField category:

```python
ff.def_pairstyle("coul/long/pme", {"alpha": 0.3})
```

## Compute

The contract is one method:

```python
class MyRdf:
    def compute(self, *args, **kwargs):
        ...
```

`molpy.compute.Compute` is a Protocol. Call aliases and dump helpers are not
part of it.

## Typifier

```python
typifier = molpy.ff.MMFF94Typifier()
typed = typifier.typify(mol)
pots = typifier.forcefield().to_potentials(typed.to_frame())
```
