# molcrafts-molrs-pack

[![Crates.io](https://img.shields.io/crates/v/molcrafts-molrs-pack.svg)](https://crates.io/crates/molcrafts-molrs-pack)

Packmol-grade molecular packing in pure Rust. Part of the [molrs](https://github.com/MolCrafts/molrs) toolkit.

## Features

- Faithful port of the Packmol algorithm (GENCAN optimizer)
- Geometric constraints: box, sphere, cylinder, ellipsoid, plane
- Three-phase packing: per-type init → constraint fitting → main loop with movebad heuristic
- `f64` feature for double-precision (recommended for packing)

## Usage

```rust
use molrs_pack::{Molpack, Target};
use molrs::Frame;

let water = Frame::new(); // ... populate with atoms
let target = Target::new(water, 100);

let result = Molpack::new(None)
    .with_tolerance(2.0)
    .with_precision(0.01)
    .pack(&[target])
    .unwrap();

println!("converged: {}", result.converged);
```

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
