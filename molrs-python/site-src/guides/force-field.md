# Force Fields

molrs force-field code separates typing from evaluation. Typing reads a
molecular graph and produces the frame-level parameters needed by potential
kernels. Evaluation then consumes flat coordinate arrays in the form
`[x0, y0, z0, x1, y1, z1, ...]`.

That flat coordinate contract is important. It is not an `N x 3` matrix, even
when user-facing code displays coordinates as three columns. Potential kernels
operate on a contiguous `3N` vector so energy and force evaluation can stay
close to the numerical representation used by optimizers.

## The MMFF94 workflow is typify, then evaluate

MMFF94 typing starts from an `Atomistic` graph. The typifier assigns atom
types, builds the force-field terms, and returns a compiled `Potentials`
object. The same compiled object can evaluate energy or energy plus forces for
compatible coordinates.

Units should be treated as part of the interface. MMFF94 energies are reported
in kcal/mol, and coordinates are interpreted in angstrom. When molrs data is
passed to another codebase, convert units at the boundary instead of hiding the
conversion inside analysis code.

## Worked Example: Energy and Forces

```python
import numpy as np
import molrs

mol = molrs.parse_smiles("CCO").to_atomistic()
mol3d, _report = molrs.Conformer(speed="fast", seed=42).generate(mol)
frame = mol3d.to_frame()

typifier = molrs.MMFFTypifier()
typed = typifier.typify(mol3d)
print("typed blocks:", typed.keys())

try:
    potentials = typifier.build(mol3d)
    coords = molrs.extract_coords(frame)

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

- `MMFFTypifier.typify(mol)` creates a typed `Frame` representation.
- `MMFFTypifier.build(mol)` compiles a `Potentials` object.
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

## Common Mistakes

Do not typify a frame that has lost graph semantics if the workflow needs
bond-order or valence information. Start from `Atomistic`, then convert to a
frame for coordinate extraction or writing.

Do not mix units. MMFF94 examples in molrs assume angstrom coordinates and
kcal/mol energies. If a source file was in nanometers, convert coordinates
before evaluating MMFF94.
