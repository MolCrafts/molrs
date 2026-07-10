# SMILES and Topology

SMILES parsing begins as text and ends as molecular topology. The intermediate
representation records atoms, bonds, branches, ring closures, and disconnected
components. Converting that representation to `Atomistic` produces the graph
view used by 3D embedding and force-field typification.

The graph view is separate from coordinates. A SMILES string such as `CCO`
describes connectivity, not a conformation. Embedding is the step that turns
that connectivity into a three-dimensional molecule. Keeping those phases
separate makes failures easier to diagnose: parser errors are syntax or
valence problems, while embedding errors are geometry or optimization problems.

## Atomistic is the editable graph view

`Atomistic` stores atoms and bonds directly. It is the right abstraction when
building small molecules, adjusting bond orders, or passing topology into
MMFF94 typing. A frame can be produced from the graph when downstream code
needs columnar coordinates or file output.

Implicit hydrogens are not the same thing as explicit graph nodes. The Python
SMILES conversion leaves implicit hydrogens implicit; the embedding pipeline
can add hydrogens when `Conformer` is asked to. That keeps parsing faithful
to the input while letting coordinate generation construct the atomistic model
it needs for force-field refinement.

## Worked Example: Parse and Inspect Ethanol

```python
import molrs

ir = molrs.parse_smiles("CCO")
mol = ir.to_atomistic()

print("components:", ir.n_components)
print("atoms:", mol.n_atoms)
print("bonds:", mol.n_bonds)
```

Expected output:

```text
components: 1
atoms: 3
bonds: 2
```

Those three atoms are the heavy atoms from the SMILES string. Hydrogens can be
made explicit by the embedding pipeline:

```python
mol3d, _report = molrs.Conformer(speed="fast", add_hydrogens=True, seed=7).generate(mol)
print("explicit atoms:", mol3d.n_atoms)
```

For ethanol, the embedded molecule usually has nine atoms: three heavy atoms
plus six hydrogens. If you disable hydrogen addition, downstream force-field
typing may fail or produce a different chemistry model.

## Worked Example: Build a Graph Directly

Sometimes a workflow starts from known connectivity rather than text. Build an
`Atomistic` directly when you already know the atoms and bonds.

```python
import molrs

mol = molrs.Atomistic()
c = mol.add_atom("C")
o = mol.add_atom("O")
mol.add_bond(c, o)
mol.set_bond_order(c, o, 2.0)

print("atoms:", mol.n_atoms)
print("bonds:", mol.n_bonds)
```

This graph has no coordinates unless you passed `x`, `y`, and `z` to
`add_atom`. That is valid input for `Conformer.generate`, which is designed to assign
coordinates from topology.

## SMARTS Matching Returns Bindings

SMARTS matching is a graph query API. It returns match bindings, not a copied
subgraph and not a frame:

```python
pattern = molrs.SmartsPattern("[C:1][O:2]")
matches = pattern.find_matches(mol)

first = matches[0]
print(first.atoms)    # atom ids in query order
print(first.mapping)  # SMARTS map labels, when present
```

Use these bindings inside higher-level code such as a typifier or a reaction
builder. If you need a typed molecule, call a typifier; if you need columnar
arrays, convert the graph with `.to_frame()` after the graph operation is done.

## When to Use Frame Instead

Use `Atomistic` when topology is the main object: parsing SMILES, editing
bonds, assigning bond orders, embedding, or force-field typing. Use `Frame`
when tabular data is the main object: file I/O, trajectory analysis, neighbor
search, grid data, or interop with NumPy and TypeScript arrays.

Crossing between the two should be visible in code:

```python
mol = molrs.parse_smiles("c1ccccc1").to_atomistic()
mol3d, _report = molrs.Conformer().generate(mol)
frame = mol3d.to_frame()
```

That line is the handoff from graph semantics to table semantics.
