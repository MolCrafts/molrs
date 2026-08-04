# molrs

molrs is a molecular modeling toolkit with a Rust core and Python and
WebAssembly bindings. The project is organized around a shared data model:
`Frame` holds named `Block`s of columnar molecular data, optional simulation
box metadata, and enough topology to move between file I/O, geometry
generation, force-field evaluation, and trajectory analysis.

This site is the narrative layer for that system. Rust API reference stays on
docs.rs, while Python reference is injected from the installed binding module.
The WebAssembly package emits TypeScript declarations during the docs build;
the hosted site reserves `/reference/wasm/` for that generated reference.

## The same workflow runs in Python, Rust, and TypeScript

=== "Python"

    Parse ethanol from SMILES, generate a three-dimensional structure, convert
    it to a frame, and inspect coordinate columns.

    ```python
    import molrs

    ir = molrs.io.SmilesIR("CCO")
    mol = ir.to_atomistic()

    mol3d, _report = molrs.conformer.Conformer(speed="fast", seed=42).generate(mol)
    frame = mol3d.to_frame()

    atoms = frame["atoms"]
    print("atoms:", atoms.nrows)
    print("columns:", atoms.keys())
    print("x:", atoms.view("x")[:3])
    ```

    Expected shape of the result: the input graph has three heavy atoms, while
    the embedded molecule usually includes explicit hydrogens because
    `Conformer(add_hydrogens=True)` is the default.

=== "Rust"

    Use the facade crate with the `full` feature while learning, then narrow
    features when an application has a stable dependency boundary.

    ```rust
    use molrs::conformer::{Conformer, ConformerOptions};
    use molrs::smiles::{parse_smiles, to_atomistic};

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let ir = parse_smiles("c1ccccc1")?;
        let mol = to_atomistic(&ir)?;
        let (mol3d, report) = Conformer::new(ConformerOptions::default()).generate(&mol)?;

        println!("atoms: {}", mol3d.n_atoms());
        println!("final energy: {:?}", report.final_energy);
        Ok(())
    }
    ```

=== "TypeScript"

    Initialize the WebAssembly module once, then use the generated classes and
    functions as regular TypeScript exports.

    ```ts
    import init, { generate3D, parseSMILES, writeFrame } from "@molcrafts/molrs";

    await init();

    const ir = parseSMILES("CCO");
    const frame2d = ir.toFrame();
    const frame3d = generate3D(frame2d, "fast", 42);

    console.log(writeFrame(frame3d, "xyz"));
    ```

## One data model powers every subsystem

- [Data model](guides/data-model.md): `Atomistic` is the graph view,
  `Frame` is the columnar data view, and `Block` is the typed column store.
- [SMILES and topology](guides/smiles-and-topology.md): parse chemical strings
  into topology before deciding whether to embed coordinates or write tables.
- [Neighbor search](guides/neighbor-search.md): build pair lists once and reuse
  them for RDF, cluster analysis, and contact queries.
- [3D embedding](guides/embed-3d.md): use distance geometry plus MMFF94
  refinement to create coordinates from connectivity.
- [Force fields](guides/force-field.md): typify an `Atomistic`, compile
  potentials, then evaluate energy and forces on flat `3N` coordinate arrays.
- [I/O](guides/io-formats.md): read and write PDB, XYZ, LAMMPS, CHGCAR, Cube,
  and frame-sequence Zarr data through frames.
- [Trajectory analysis](guides/trajectory-analysis.md): run RDF, MSD, cluster,
  tensor, PCA, and k-means workflows on one frame or a sequence of frames.

## Find your starting point

Start with [Installation](getting-started/installation.md), then choose the
quickstart for your host language:

- [Python Quickstart](getting-started/quickstart-python.md) is the most complete
  end-to-end tutorial and mirrors the style of a notebook.
- [Rust Quickstart](getting-started/quickstart-rust.md) explains crate features
  and the facade layout.
- [WASM Quickstart](getting-started/quickstart-wasm.md) explains initialization,
  typed arrays, and browser bundling.

Use [Python Reference](reference/python.md), [Rust Reference](reference/rust.md),
and [WASM Reference](reference/wasm.md) when you need exact API details.

The conceptual guides are shared across languages. They explain how frames,
topology, simulation boxes, neighbor lists, force fields, and trajectories fit
together, so the same mental model carries from Rust to Python to WASM.
