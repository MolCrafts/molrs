# Force Fields

molrs force-field code separates typing from evaluation. Typing reads a
molecular graph and returns a typed graph with atom, bond, angle, dihedral, and
force-field-specific topology labels. Evaluation starts after that graph is
materialized as a `Frame` and consumes flat coordinate arrays in the form
`[x0, y0, z0, x1, y1, z1, ...]`.

That flat coordinate contract is important. It is not an `N x 3` matrix, even
when user-facing code displays coordinates as three columns. Potential kernels
operate on a contiguous `3N` vector so energy and force evaluation can stay
close to the numerical representation used by optimizers.

## The MMFF94 workflow is typify, then evaluate

MMFF94 typing starts from an `Atomistic` graph. The typifier assigns atom
types, charges, bond labels, angle labels, dihedral labels, and MMFF
out-of-plane terms, then returns a new typed `Atomistic`. `build(mol)` is a
separate convenience that compiles a `Potentials` object from the same typed
topology. The compiled object can evaluate energy or energy plus forces for
compatible coordinates.

Units should be treated as part of the interface. MMFF94 energies are reported
in kcal/mol, and coordinates are interpreted in angstrom. When molrs data is
passed to another codebase, convert units at the boundary instead of hiding the
conversion inside analysis code.

## OPLS-AA uses the same typifier contract

`OPLSAATypifier` also accepts and returns `Atomistic`:

```python
typifier = molrs.OPLSAATypifier(strict=True)
typed = typifier.typify(mol3d)
frame = typed.to_frame()
```

The constructor loads the embedded OPLS-AA XML by default. Pass a path or XML
string only when you need a different parameter source:

```python
typifier = molrs.OPLSAATypifier("oplsaa.xml", strict=False)
```

`typify()` writes atom labels and every bonded topology class supported by the
loaded OPLS-AA data. The bundled OPLS-AA data defines bonds, angles, and
dihedrals; it does not define an improper table.

## Worked Example: Energy and Forces

```python
import numpy as np
import molrs

mol = molrs.parse_smiles("CCO").to_atomistic()
mol3d, _report = molrs.Conformer(speed="fast", seed=42).generate(mol)
frame = mol3d.to_frame()

typifier = molrs.MMFF94Typifier()
typed = typifier.typify(mol3d)
typed_frame = typed.to_frame()
print("typed blocks:", typed_frame.keys())

try:
    typed_frame["pairs"] = molrs.intramolecular_pairs(typed_frame)
    potentials = typifier.forcefield().to_potentials(typed_frame)
    coords = molrs.extract_coords(typed_frame)

    energy, forces = potentials.eval(coords)

    print("terms:", len(potentials))
    print("energy:", energy)
    print("coords:", coords.shape)
    print("forces:", forces.shape)
    print(
        "max net force component:",
        np.abs(forces.reshape(mol3d.n_atoms, 3).sum(axis=0)).max(),
    )
except ValueError as exc:
    print("potential build skipped:", exc)
```

The coordinate and force arrays are flat. Reshaping them to `(N, 3)` is only a
display operation; pass the flat arrays back to potential evaluators.

The `try` block is intentional for the current preview surface. MMFF94 typing
can succeed even when potential compilation reports missing parameter coverage.
That distinction is useful: it tells you whether the failure is in chemistry
typing or in the stricter energy-evaluation path.

## Typing and evaluation are separate steps

Typing and evaluation answer different questions:

- `MMFF94Typifier.typify(mol)` returns a new typed `Atomistic`. That is the
  typifier's whole contract: labels and charges.
- `MMFF94Typifier.forcefield().to_potentials(frame)` compiles a `Potentials`
  object — the same call every other force field in molrs goes through. MMFF is
  not a special case, and there is no one-step `build`.
- `MMFF94STypifier` is the same surface over the MMFF94s ("static") parameter set:
  it flattens delocalised trivalent nitrogen (MMFF types 10 `NC=O` / 40 `NC=C`) by
  re-parameterising 11 out-of-plane rows and 42 torsion rows. Everything else —
  all 95 atom types, every bond / angle / stretch-bend / vdW / charge parameter —
  is shared, so molecules without such a nitrogen get identical answers.
- `Potentials.energy(coords)` returns only energy.
- `Potentials.eval(coords)` returns energy and forces.

Keep the `Potentials` object if you plan to evaluate many coordinate sets for
the same topology. Rebuilding potentials for every frame wastes work and can
hide topology drift.

## Coordinate Contract

`extract_coords(frame)` reads `x`, `y`, and `z` columns from `frame["atoms"]`
and returns:

```text
[x0, y0, z0, x1, y1, z1, ...]
```

If the frame is missing `atoms`, `x`, `y`, or `z`, extraction fails early. That
is better than silently evaluating an energy against malformed coordinates.

## LAMMPS `*.ff` includes

AMBER/GAFF-style LAMMPS force-field includes (the `*.ff` next to a data file)
round-trip through molrs:

| Direction | Rust | Python |
| --- | --- | --- |
| read | `LammpsFfReader` | `molrs.read_lammps_forcefield` / `_str` |
| write | `LammpsFfWriter` | `molrs.write_lammps_forcefield` / `_str` |

molrs stores harmonic stiffness in the `½k(x−x₀)²` form and angles in
**radians**. The writer inverts both for LAMMPS `real` units (`K = k/2`,
degrees). Pair styles that the reader split into `lj/cut` + `coul/cut` are
recombined into a single `lj/cut/coul/cut` line so geometric mixing still works.

```python
import molrs

ff = molrs.read_lammps_forcefield("system.ff")
# … edit styles / types …
molrs.write_lammps_forcefield(
    "system-out.ff",
    ff,
    precision=6,
    atom_types={"c3", "h1"},  # optional whitelist
)
```

Optional filters (`atom_types`, `bond_types`, …) drop coeffs for types that are
not present in a frame's labelmap — the same trap LAMMPS hits when a merged
force field still carries cap artifacts.

## Geometry optimization

`LBFGS` (under the `ff` feature) minimizes a molecule-bound `Potential` in
place. Prefer `LBFGS::new` / `with_defaults` when you own the potential as an
`Arc`, or the one-shot `LBFGS::minimize` / `minimize_batch` helpers for borrowed
potentials and flat coordinate buffers.

## Common Mistakes

Do not typify a frame that has lost graph semantics if the workflow needs
bond-order or valence information. Start from `Atomistic`, then convert to a
frame for coordinate extraction or writing.

Do not mix units. MMFF94 examples in molrs assume angstrom coordinates and
kcal/mol energies. If a source file was in nanometers, convert coordinates
before evaluating MMFF94.

Do not write a LAMMPS `*.ff` with molrs unit conventions. Always go through
`write_lammps_forcefield` (or `LammpsFfWriter`) so harmonic `K` and angles are
converted; pasting molrs numbers into an input script doubles stiffness and
interprets radians as degrees.
