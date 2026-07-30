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

## Composition (no all-in-one façades)

Every force field walks the same primitives. Callers compose; molrs does not
hide the steps:

1. **Typify** — `typifier.typify(mol)` → new labeled graph.
2. **Frame** — `typed.to_frame()` for coordinate / pair columns.
3. **Pairs** — non-bonded terms need a `pairs` block (`atomi` / `atomj` /
   `is_14`). Python: `frame["pairs"] = molrs.intramolecular_pairs(frame)`.
   WASM / Rust optimizers may install topology pairs (or a caller-supplied
   neighbor list) before minimizing — still not a free-floating
   “optimize geometry” sugar.
4. **Compile** — `typifier.forcefield().to_potentials(frame)` (Python) or
   `typifier.ff().to_potentials(&frame)` (Rust). WASM collapses the private FF
   handle to `typifier.toPotentials(frame)` — still not a one-shot `build()`.
5. **Evaluate / minimize** — `potentials.eval(coords)` or `LBFGS(pots).run(...)`.

There is no `optimizeGeometry`, no typifier-level `build()`, and no GFN-FF in
this tree.

## Built-in typifiers

| Typifier | Surface | Notes |
| --- | --- | --- |
| `MMFF94Typifier` | Python, Rust, WASM | Halgren 1996; energies kcal/mol, Å |
| `MMFF94STypifier` | Python, Rust, WASM | “static” set (planar delocalised N) |
| `OPLSAATypifier` | Python, Rust | embedded OPLS-AA XML by default |
| `UFFTypifier` | Rust, WASM | RDKit-aligned UFF (Rappé 1992); full periodic-table param table; no electrostatics |

## The MMFF94 workflow is typify, then evaluate

MMFF94 typing starts from an `Atomistic` graph. The typifier assigns atom
types, charges, bond labels, angle labels, dihedral labels, and MMFF
out-of-plane terms, then returns a new typed `Atomistic`. Compiling potentials
is a separate step on the force field handle.

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

## UFF (Universal Force Field)

UFF is a first-class typifier in the Rust core (`molrs::ff::typifier::uff`) and
on the WASM face (`UFFTypifier`). It assigns RDKit-style UFF labels, bakes
per-instance force constants onto the graph, and compiles `uff_bond` /
`uff_angle` / `uff_torsion` / `uff_inversion` / `uff_lj` kernels. There is no
electrostatics path (RDKit UFF convention).

```rust,ignore
use molrs::ff::potential::intramolecular_pairs;
use molrs::ff::typifier::uff::UFFTypifier;

let t = UFFTypifier::new();
let mut frame = t.typify(&mol)?.to_frame();
frame.insert("pairs", intramolecular_pairs(&frame));
let pots = t.ff().to_potentials(&frame)?;
```

WASM composes the same steps without exposing a public `.ff()`:

```js
const typifier = new UFFTypifier();
const typed = typifier.typify(frame);
const pots = typifier.toPotentials(typed);  // bonded-only until pairs exist
const report = new LBFGS(pots /*, neighborList */).run(typed, 200);
// omit neighborList → optimizer installs a topology (bruteforce) pair list
```

Python bindings for `UFFTypifier` follow the same contract when exposed;
until then use the Rust or WASM path above.

## Worked Example: Energy and Forces (Python / MMFF94)

```python
import numpy as np
import molrs

mol = molrs.parse_smiles("CCO").to_atomistic()
mol3d, _report = molrs.Conformer(speed="fast", seed=42).generate(mol)

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

The `try` block is intentional. MMFF94 typing can succeed even when potential
compilation reports missing parameter coverage. That distinction tells you
whether the failure is in chemistry typing or in the stricter energy path.

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

`LBFGS` minimizes molecule-bound potentials. Python:

```python
opt = molrs.LBFGS(potentials, fmax=0.05, max_steps=500)
min_frame, report = opt.run(typed_frame)
```

On the WASM face, `new LBFGS(pots, neighborList?)` optionally takes a neighbor
list; without one it installs a topology (bruteforce) pair list, recompiles,
then runs. Prefer that composition over any deleted one-shot façade.

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

Do not expect a hand-written `CHANGELOG.md`. Release history is git tags and
GitHub Releases.
