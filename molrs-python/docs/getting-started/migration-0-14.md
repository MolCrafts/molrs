# Migrating to 0.14

User-facing spelling is `molrs`: the PyPI package is `molcrafts-molrs` and the
import name is `molrs`. The spellings below are the binding surface; the
`molpy` layer re-exports most of them under its own flat namespace.

## Units

The engine no longer converts energy or Boltzmann's constant for you.

```python
import molrs

kb = molrs.UnitPreset("real").boltzmann()
eps = (
    molrs.UnitRegistry()
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
import molrs
from molrs import md

kb = molrs.UnitPreset("real").boltzmann()
md.MaxwellBoltzmann(kb * 300.0, seed=0)
driver = md.MD(dtype=np.float64)
driver.run(frame, n, dt=dt, kb=kb, thermo=100)
```

`Potential` is a `runtime_checkable` Protocol: any object with
`calc_energy_forces(pos)` is accepted. Do not subclass a compiled base class.

## Record / Frame store

There is no `molrs.Record`. Write the object you have through `molrs.io.mrec`.
Schema checks live in `molrs::io::mrec::schema` and are bound at
`molrs.io.mrec.schema`.

```python
from molrs.io import mrec

mrec.write_frame(path, frame)
loaded = mrec.read_frame(path)
mrec.schema.validate_path(path)
```

On-disk identity is the `*.mrec/` path suffix plus a Zarr root. While the
record contract is in development `meta["molrec_version"]` is optional: an
absent key means no version validation, a present one must be an integer in
`1..=MOLREC_VERSION`. Writers stamp nothing; there is no brand key.

### Frame metadata

`frame.meta` is a mapping, and it writes through.

```python
frame.meta["title"] = "water"        # lands in the frame
frame.meta["title"]                  # 'water' — a plain str, not a box
dict(frame.meta)                     # plain values, ready for json.dumps
```

The dtype belongs to the key, the way `mrec` already declares it
(`SequenceSchema.declare_meta(key, dtype)`). Writing a plain value to a key that
already exists keeps that key's dtype and refuses a value it cannot hold, so
`frame.meta[k] = frame.meta[k]` is an identity:

```python
frame.meta["temperature"] = molrs.MetaValue("f32", 300.0)
frame.meta.dtype("temperature")      # 'f32'
frame.meta["temperature"] = 310.0    # still f32
frame.meta["temperature"] = "warm"   # TypeError: key 'temperature' is f32
```

A JSON document comes back decoded, which makes it a snapshot — nested edits are
read-modify-write:

```python
frame.meta["run"] = {"step": 1}
document = frame.meta["run"]
document["step"] = 2
frame.meta["run"] = document
```

There is no `Frame.from_dict`; `Frame(blocks=..., meta=...)` is the constructor,
and `dict(frame.meta)` is what you feed back into it.

### Streaming trajectories

A run too large to hold in memory is written frame by frame. Declare the
schema (or derive it from a representative frame), append, and let the
writer land whole chunks on its own cadence; `flush()` / `close()` commit
whatever is buffered, durably by default.

```python
from molrs.io import mrec

schema = (
    mrec.SequenceSchema()
    .declare_block("atoms", rows=n_atoms)
    .declare_column("atoms", "x", "f64")
    .declare_column("atoms", "y", "f64")
    .declare_column("atoms", "z", "f64")
    .declare_column("bonds", "atomi", "u64")
    .declare_column("bonds", "atomj", "u64")
    .declare_meta("temp", "f64")
)
with mrec.TrajectoryWriter(path, schema, meta={"creator": {"name": "molrs"}}) as w:
    w.append(first_frame, step=0, time=0.0)          # atoms + bonds
    for step, frame in run:                           # atoms only: bonds carry forward
        w.append(frame, step=step, time=step * dt)

reader = mrec.TrajectoryReader(path)                  # or the packed path from mrec.pack(path)
sub = reader.read_columns(i, [("atoms", "x"), ("atoms", "y"), ("atoms", "z")])  # a Frame
same_bonds = reader.block_update_at("bonds", i) == reader.block_update_at("bonds", i - 1)
```

Three states per block and frame: a frame that **omits** a declared block
carries it forward; a block presented with **zero rows** is present and empty;
a block with no update yet is absent. `flush_every=` overrides the landing
cadence, `compression=` chooses how floating-point columns are compressed
(`None` by default, `"gzip[:level]"`, `"zstd[:level]"`), `durable=False`
skips the fsync. `TrajectoryWriter.open(path)` reattaches after a crash and
rolls back anything past the last committed frame. Float columns are stored
raw, integer/bool/string columns and every index array gzip level 1, every
chunk ends in `crc32c`.

The layout is built so the common run costs one array per column and
nothing else: a regular block (fixed row count, updated every frame or
never) writes no index arrays, a fixed cell is a few `box/` attributes, and
`step` / `time` are `{start, stride}` attributes while they are arithmetic.
An NVT run with `x`, `y`, `z` and a fixed box is 11 files on disk however
long it is (until a column outgrows one 256 MiB shard).

## Neighbors

The loop owns rebuilds. Pair MIC is computed once, in `VerletSkin.pairs_at`,
and shared by every pair potential.

```python
from molrs import Box, NeighborList, VerletSkin, md

nl = VerletSkin(
    NeighborList(rc + skin), rc, pos, Box.cube(20.0), skin=skin
)
vv = md.VelocityVerlet(1.0, potential=md.LJCut(eps, 3.405, rc), neighbors=nl, mass=mass)
```

`NeighborList` and `VerletSkin` are core types and stay at the top level
(`molrs.NeighborList`, `molrs.VerletSkin`); only the integrators and potentials
live under `molrs.md`. There is one `LJCut`, re-exported at `molrs.md.LJCut`.

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

`molrs.compute.Compute` is a Protocol. Call aliases and dump helpers are not
part of it.

## Typifier

```python
typifier = molrs.ff.MMFF94Typifier()
typed = typifier.typify(mol)
pots = typifier.forcefield().to_potentials(typed.to_frame())
```
