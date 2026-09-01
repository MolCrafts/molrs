# Rust Quickstart

The Rust facade crate is intentionally small. It re-exports the core model and
exposes optional subsystems behind Cargo features, so applications can choose a
minimal dependency set or enable `full` while exploring.

## 1. Create a Project

```toml
[dependencies]
molrs = { package = "molcrafts-molrs", version = "0.12", features = ["full"] }
```

The `full` feature enables I/O, SMILES, compute, force-field, and embedding
subsystems. Once you know which layers your application uses, replace `full`
with a narrower feature list.

## 2. Parse Topology and Generate Coordinates

```rust
use molrs::conformer::{Conformer, ConformerOptions};
use molrs::smiles::{parse_smiles, to_atomistic};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ir = parse_smiles("c1ccccc1")?;
    let mol = to_atomistic(&ir)?;

    let (mol3d, report) = Conformer::new(ConformerOptions::default()).generate(&mol)?;
    let frame = mol3d.to_frame();

    println!("atoms: {}", frame.get("atoms").map_or(0, |block| block.nrows()));
    println!("final energy: {:?}", report.final_energy);
    Ok(())
}
```

The two-step parse is intentional. `parse_smiles` validates the text and
produces an intermediate representation. `to_atomistic` turns that intermediate
form into the molecular graph consumed by embedding and force-field code.

## 3. Understand the Facade Layout

The facade mirrors the workspace layout:

| Module | Feature | Purpose |
| --- | --- | --- |
| `molrs::*` | always | Core `Frame`, `Block`, topology, boxes, and regions |
| `molrs::io` | `io` | File readers and writers |
| `molrs::smiles` | `smiles` | SMILES parser and graph conversion |
| `molrs::conformer` | `embed` | 3D coordinate generation |
| `molrs::compute` | `compute` | RDF, MSD, clusters, descriptors |
| `molrs::ff` | `ff` | Force-field typing and potentials |

The lower-level crates are still documented individually on docs.rs. Use the
facade for application code; open the crate-specific references when you need
module internals or lower-level extension points.

## 4. Common Compile Errors

If `molrs::smiles` or `molrs::conformer` cannot be found, the Cargo feature is not
enabled. If code compiles but embedding fails at runtime, inspect the topology:
embedding expects chemically meaningful atoms and bonds, not just a coordinate
table.
