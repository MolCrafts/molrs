# molcrafts-molrs-pack

[![Crates.io](https://img.shields.io/crates/v/molcrafts-molrs-pack.svg)](https://crates.io/crates/molcrafts-molrs-pack)

Packmol-grade molecular packing in pure Rust. Part of the [molrs](https://github.com/MolCrafts/molrs) toolkit.

This crate is the first public preview of the packing API. The main user-facing
surface is `Molpack`, `Target`, `PackResult`, and the geometric constraint types.

## Features

- Faithful port of the Packmol algorithm (GENCAN optimizer)
- Geometric constraints: box, sphere, cylinder, ellipsoid, plane
- Three-phase packing: per-type init → constraint fitting → main loop with movebad heuristic
- `f64` feature for double-precision (recommended for packing)
- MSRV: Rust 1.85

## Usage

```rust
use molrs_pack::{InsideBoxConstraint, Molpack, Target};

let water_positions = [
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0],
];
let water_radii = [1.52, 1.20, 1.20];
let box_constraint = InsideBoxConstraint::new([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]);
let target = Target::from_coords(&water_positions, &water_radii, 100)
    .with_name("water")
    .with_constraint(box_constraint);

let mut packer = Molpack::new().tolerance(2.0).precision(0.01);
let result = packer
    .pack(&[target], 200, Some(42))
    .unwrap();

println!("converged: {}", result.converged);
```

Pass an explicit seed when you need reproducible placements. Enable the `f64`
feature for tighter numerical behavior on larger or more constrained systems.

## Examples

```bash
cargo run -p molcrafts-molrs-pack --release --example pack_mixture
cargo run -p molcrafts-molrs-pack --release --example pack_bilayer
cargo run -p molcrafts-molrs-pack --release --example pack_spherical
cargo run -p molcrafts-molrs-pack --release --example pack_interface
cargo run -p molcrafts-molrs-pack --release --example pack_solvprotein
```

## License

BSD-3-Clause
