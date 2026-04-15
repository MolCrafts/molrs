# molrs

[![Crates.io](https://img.shields.io/crates/v/molcrafts-molrs.svg)](https://crates.io/crates/molcrafts-molrs)
[![Documentation](https://docs.rs/molcrafts-molrs/badge.svg)](https://docs.rs/molcrafts-molrs)
[![PyPI](https://img.shields.io/pypi/v/molcrafts-molrs.svg)](https://pypi.org/project/molcrafts-molrs/)
[![npm](https://img.shields.io/npm/v/@molcrafts/molrs.svg)](https://www.npmjs.com/package/@molcrafts/molrs)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

A molecular modeling toolkit with Rust and Python interfaces.

Molecular packing (Packmol port) was split out into its own repository
[`MolCrafts/molpack`](https://github.com/MolCrafts/molpack) (crates.io:
`molcrafts-molpack`, PyPI: `molcrafts-molpack`). Add it as a separate
dependency when needed.

## Install

| Platform | Package | Install |
|----------|---------|---------|
| Rust | `molcrafts-molrs` | `cargo add molcrafts-molrs` |
| Rust packing | `molcrafts-molpack` | `cargo add molcrafts-molpack` |
| Python | `molcrafts-molrs` | `pip install molcrafts-molrs` |
| Python packing | `molcrafts-molpack` | `pip install molcrafts-molpack` |
| npm | `@molcrafts/molrs` | `npm install @molcrafts/molrs` |

## Features

- **Frame / Block** — hierarchical, column-oriented molecular data model
- **MolGraph** — petgraph-based molecular topology with SMILES parser
- **Embed** — 3D coordinate generation (distance geometry + MMFF94 minimization)
- **Neighbor search** — O(N) LinkCell with freud-style `AABBQuery` API
- **Compute** — RDF, MSD, cluster analysis (self-query & cross-query)
- **Force field** — MMFF94 bond/angle/torsion/vdw/electrostatic potentials
- **I/O** — PDB, XYZ, LAMMPS data, Zarr V3 trajectories
- **Packing** — see [`molpack`](https://github.com/MolCrafts/molpack) (separate crate)
- **Bindings** — Python (PyO3/maturin), WASM (wasm-bindgen), C/C++ (cbindgen)

## Crates

| Crate | Description |
|-------|-------------|
| [`molcrafts-molrs`](https://crates.io/crates/molcrafts-molrs) | Core library |
| [`molcrafts-molpack`](https://crates.io/crates/molcrafts-molpack) | Molecular packing (separate repo) |
| `molcrafts-molrs-ffi` | Handle-based FFI layer |
| `molcrafts-molrs-capi` | C/C++ API |

## Quick start

### Rust

```rust
use molrs::{parse_smiles, to_atomistic, generate_3d, EmbedOptions};

let ir = parse_smiles("c1ccccc1").unwrap();         // benzene
let mol = to_atomistic(&ir).unwrap();
let (mol3d, _report) = generate_3d(&mol, EmbedOptions::default()).unwrap();
```

### Python

```python
import molrs

ir = molrs.parse_smiles("CCO")
frame = ir.to_frame()
result = molrs.generate_3d(frame)
```

### JavaScript / TypeScript

```js
import init, { parseSMILES, generate3D, writeFrame } from "@molcrafts/molrs";

await init();
const frame = parseSMILES("CCO").toFrame();
const mol3d = generate3D(frame, "fast");
console.log(writeFrame(mol3d, "xyz"));
```

## Build & Test

```bash
# Rust
cargo build --workspace
cargo test --workspace --lib --tests --examples

# Python
cd molrs-python && maturin build && pip install target/wheels/*.whl && pytest -q

# WASM
cd molrs-wasm && wasm-pack build --target bundler --scope molcrafts --out-name molrs

# C API
cargo build --manifest-path molrs-capi/Cargo.toml
cd molrs-capi && cmake -S tests/cpp -B build-test && cmake --build build-test && ctest --test-dir build-test
```

## Support Matrix

- Rust MSRV: 1.85
- Python: 3.9+
- Python package name: `molcrafts-molrs`
- Python import name: `molrs`

## License

BSD-3-Clause
