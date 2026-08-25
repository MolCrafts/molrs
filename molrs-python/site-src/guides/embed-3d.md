# 3D Embedding

3D embedding converts molecular topology into coordinates. molrs uses a staged
pipeline: distance geometry creates an initial arrangement, fragment and rotor
steps improve chemically familiar structures, and MMFF94-based minimization
refines the final geometry. The result is a new molecule with coordinates and a
report describing the stages that ran.

The input should be a chemically meaningful graph. Missing bond orders,
impossible valences, and disconnected fragments are topology problems that
embedding cannot reliably repair. The best workflow is to validate the graph
early, then treat the embedding report as geometry diagnostics rather than as a
parser error log.

## Speed and Reproducibility

The speed preset controls how much refinement the pipeline performs. Fast mode
is useful for visualization and interactive workflows. Medium is the default
tradeoff. Better mode spends more time on conformer quality and rotor search.

Embedding uses randomized steps, so reproducible examples should set a seed.
That is especially important for documentation, tests, and notebooks where a
changed conformation can make downstream energies or plots look surprising even
when the code is correct.

## Worked Example: Benzene from SMILES

```python
import molpy

mol = molpy.io.SmilesIR("c1ccccc1").to_atomistic()
mol3d, report = molpy.conformer.Conformer(speed="medium", seed=123).generate(mol)

print("atoms:", mol3d.n_atoms)
print("final energy:", report.final_energy)
for stage in report.stages:
    print(stage.stage, stage.steps, stage.converged)
```

The output molecule includes generated coordinates and, by default, explicit
hydrogens. The report is useful when a molecule embeds but the geometry looks
wrong: a non-converged final optimization or warning can point to the stage
that needs attention.

## Reading Coordinates from the Result

`Conformer.generate` returns an `Atomistic`, not a frame. Convert it when you need
column data:

```python
frame = mol3d.to_frame()
atoms = frame["atoms"]

for i in range(min(5, atoms.nrows)):
    print(
        i,
        atoms.view("element")[i],
        atoms.view("x")[i],
        atoms.view("y")[i],
        atoms.view("z")[i],
    )
```

This is also the point where writers take over. If you only need an XYZ file,
write the frame:

```python
molpy.io.write_xyz("benzene.xyz", frame)
```

## Choosing Speed

Use `speed="fast"` for interactive previews and smoke tests. Use
`speed="medium"` for documentation examples and routine work. Use
`speed="better"` when conformer quality matters more than latency.

The speed choice changes how much refinement work is attempted, not the
meaning of the input graph. A topology problem such as impossible valence or
missing element data should be fixed before trying a slower preset.

## Failure Checklist

If embedding fails, check these points in order:

- The input is an `Atomistic`, not a `Frame`.
- Every atom has an element symbol.
- Bond orders are sensible for the chemistry.
- Disconnected fragments are intentional.
- A seed is set if you are comparing behavior across runs.
