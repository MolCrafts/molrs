# molrs

molrs is a molecular modeling toolkit with a Rust core and Python and
WebAssembly bindings. The project is organized around a shared data model:
`Frame` holds named `Block`s of columnar molecular data, optional simulation
box metadata, and enough topology to move between file I/O, geometry
generation, force-field evaluation, and trajectory analysis.

This site is the narrative layer for that system. The Rust API reference is on
docs.rs, the Python reference is rendered from the installed binding module,
and the npm package ships TypeScript declarations generated from the same Rust
doc comments.

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
    print("x:", atoms["x"][:3])
    ```

    Expected shape of the result: the input graph has three heavy atoms, while
    the embedded molecule usually includes explicit hydrogens because
    `Conformer(add_hydrogens=True)` is the default.

=== "Rust"

    Use the crate with the `full` feature while learning, then narrow
    features when an application has a stable dependency boundary.

    ```rust
    use molrs::conformer::{Conformer, ConformerOptions};
    use molrs::io::smiles::{parse_smiles, to_atomistic};

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

    The npm package is a bundler build: importing it loads the WebAssembly
    module, and the generated classes and functions are regular exports.

    ```ts
    import { generate3D, parseSMILES, writeFrame } from "@molcrafts/molrs";

    const ir = parseSMILES("CCO");
    const frame2d = ir.toFrame();
    const frame3d = generate3D(frame2d, "fast", 42);

    console.log(writeFrame(frame3d, "xyz"));
    ```

## What's new in 0.15

molrs 0.15 settles the column store and the on-disk record:

- **One column accessor.** `Block::get` / `FrameAccess::column` plus
  `Column::as_*` replace the per-dtype getters on every surface, and each
  column reports the dtype it is stored at (C: `MolrsDType`). Floats are
  `f64` only.
- **Record files follow the molrec contract.** Typed frame metadata,
  topology conventions (`chain`, `res_id`, `b_factor`, …), row references,
  aligned trajectory blocks, declared precision (coordinates in about 7.6
  instead of 24 bytes per atom per frame), and a `forcefield` section. See
  [Record files](guides/records.md).
- **Force fields** are built through `def_style` / `def_type`, compiled by
  `PotentialCompiler`, and typed by `Typing`; LAMMPS harmonic impropers now
  evaluate at the LAMMPS energy.

[What's new in 0.15](release-notes.md) lists the highlights, and the
[migration guide](migration.md) lists every breaking change from 0.14.

## What lives here

These docs cover the molrs **binding surface**: the per-language quickstarts,
the [record-file guide](guides/records.md), the API reference, and the
migration guide. Task-oriented Python guides (the data model, in-process MD,
SMILES and topology, neighbor search, 3D embedding, force fields, I/O, and
trajectory analysis) live in the
[molpy documentation](https://docs.molcrafts.org/molpy/), the Python library
built on molrs.

## Find your starting point

Start with [Installation](getting-started/installation.md), then choose the
quickstart for your host language:

- [Python Quickstart](getting-started/quickstart-python.md) is the most complete
  end-to-end tutorial and mirrors the style of a notebook.
- [Rust Quickstart](getting-started/quickstart-rust.md) explains crate features
  and the module layout.
- [WASM Quickstart](getting-started/quickstart-wasm.md) explains loading the
  module, typed-array columns, and browser bundling.

[Record files](guides/records.md) shows how to save frames, trajectories and
force fields as `*.mrec` records that every surface reads.

Use [Python Reference](reference/python.md), [Rust Reference](reference/rust.md),
and [WASM Reference](reference/wasm.md) when you need exact API details.

The data model is the same in all three: frames, topology, simulation boxes,
neighbor lists, force fields, and trajectories mean one thing across Rust,
Python, and WASM, so the mental model carries between them. The narrative
explanation of that model lives in the
[molpy documentation](https://docs.molcrafts.org/molpy/).
