<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;molrs
</h1>

<p><strong>Rust core for molecular modeling — data structures, I/O, and compute kernels, native and in the browser.</strong></p>

<p>
  <a href="https://img.shields.io/github/actions/workflow/status/MolCrafts/molrs/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI"><img src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molrs/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="https://crates.io/crates/molcrafts-molrs"><img src="https://img.shields.io/crates/v/molcrafts-molrs?style=flat-square&logo=rust&logoColor=white" alt="crates.io"></a>
  <a href="https://docs.rs/molcrafts-molrs"><img src="https://img.shields.io/docsrs/molcrafts-molrs?style=flat-square&logo=docsdotrs&logoColor=white" alt="docs.rs"></a>
  <a href="https://pypi.org/project/molcrafts-molrs/"><img src="https://img.shields.io/pypi/v/molcrafts-molrs?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/@molcrafts/molrs"><img src="https://img.shields.io/npm/v/@molcrafts/molrs?style=flat-square&logo=npm&logoColor=white" alt="npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-18432B?style=flat-square" alt="License"></a>
</p>

<p>
  <a href="https://docs.molcrafts.org/molrs/"><b>Documentation</b></a> &nbsp;&middot;&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;&middot;&nbsp;
  <a href="#molcrafts-ecosystem"><b>Ecosystem</b></a>
</p>

</div>

molrs is a Rust library for molecular modeling: a column-oriented data model, format readers and writers, trajectory analysis, force fields, and 3D structure generation. The same code runs natively, from Python (PyO3), and in the browser (WASM).

> **Under active development.** Public APIs may change between minor releases.

## Vision

Molecular modeling tools have long forced a choice between fast and reusable. The performance-critical kernels live in aging C and Fortran, while the science is written in Python wrappers that cannot run anywhere a Python interpreter does not. molrs exists to dissolve that split: one correct, well-tested implementation of the molecular data model and its compute kernels, written once in Rust.

That single implementation is meant to be portable everywhere molecules are studied — a native library, a Python package, and a WebAssembly module that runs in a browser tab with no install. The aspiration is that a researcher, a pipeline, and an interactive web tool can all reach for the exact same neighbor search, the same RDF, the same force-field evaluation, and get identical numbers.

By becoming the dependable core the rest of the MolCrafts ecosystem builds on, molrs aims to make high-performance, reference-grade molecular computation something you take for granted rather than something you reimplement.

## Capabilities

One crate, `molcrafts-molrs`, whose sub-systems are feature-gated modules
(`core` is always on; `full` enables everything):

| Module (feature) | Capability |
|------------------|------------|
| `core` *(always on)* | Frame / Block column store, MolGraph topology, elements, `SimBox` + MIC, spatial regions, neighbor search, SARW / generate |
| `io` | Readers / writers for PDB, XYZ, mol2, SDF, CIF, GRO, POSCAR, CHGCAR, Cube, LAMMPS data/dump, DCD, Zarr V3 trajectories (SMILES/SMARTS parser under the `smiles` feature) |
| `compute` | Trajectory analysis: RDF, MSD, clustering, gyration / inertia tensors, PCA, k-means, density, diffraction, PMFT, order parameters, dielectric, environment matching |
| `ff` | Force fields and potentials — MMFF94 / OPLS-AA / GAFF typing, LJ, PME; LAMMPS `*.ff` include read/write (`LammpsFfReader` / `LammpsFfWriter`); L-BFGS geometry optimization over a `Potential` |
| `conformer` | 3D conformer generation: ETKDGv3 distance geometry, experimental-torsion refinement, MMFF94 cleanup, stereo guards |
| `signal` | Signal processing — FFT-based autocorrelation, window functions, frequency grids |

A separate `molcrafts-molrs-cxxapi` crate (built from source, not published)
provides a CXX bridge for zero-copy integration with Atomiverse C++.

## Install

```bash
cargo add molcrafts-molrs
```

Opt into sub-systems via feature flags; `full` enables everything:

```toml
molcrafts-molrs = { version = "0.13", features = ["io", "smiles", "conformer"] }
```

| Environment | Install | Import / use |
|-------------|---------|----------------|
| **Rust** | `cargo add molcrafts-molrs` | `use molrs::…` |
| **Python (desktop)** | `pip install molcrafts-molrs` | `import molrs` |
| **Python (browser / Pyodide)** | `await micropip.install("molcrafts-molrs")` | `import molrs` |
| **JS (browser)** | `npm install @molcrafts/molrs` | wasm-bindgen API (not `import molrs`) |

PyPI ships **desktop** wheels (manylinux / macOS / Windows) and a **Pyodide
(Emscripten)** wheel for micropip. The npm package is a separate **wasm-bindgen**
build used by MolVis; it is not the Python extension.

```python
# In Pyodide (browser or Node)
import micropip
await micropip.install("molcrafts-molrs")
import molrs
```

> **Python nightly.** Bleeding-edge wheels go to `molcrafts-molrs-nightly`
> (`X.Y.Z.devN`) on the `nightly` branch. `pip install --pre molcrafts-molrs-nightly`
> imports as `molrs` and cannot sit beside stable `molcrafts-molrs`.

## Build from source

Building from source needs the Rust toolchain. The pinned channel, the
`rustfmt` / `clippy` components, and the `wasm32-unknown-unknown` target are all
declared in `rust-toolchain.toml`, so [`rustup`](https://rustup.rs/) selects
them automatically on the first build.

```bash
git clone https://github.com/MolCrafts/molrs.git
cd molrs
cargo build --workspace            # compile the library + the C bridge crate
bash scripts/fetch-test-data.sh    # fetch test fixtures (first run only)
cargo test --all-features          # run the test suite
```

**Python bindings** are built from the `molrs-python` crate with
[maturin](https://www.maturin.rs/). `maturin develop` compiles the PyO3
extension and installs it editable into the active virtualenv under the import
name `molrs`:

```bash
pip install maturin
maturin develop -m molrs-python/Cargo.toml --release
python -c "import molrs; print(molrs.io.SmilesIR('O').n_components)"
```

**WASM / npm** is built with [wasm-pack](https://rustwasm.github.io/wasm-pack/),
using the same flags as release publishing:

```bash
cd molrs-wasm
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

See the [installation guide](https://docs.molcrafts.org/molrs/getting-started/installation/)
for environment-verification snippets and the
[contributing guide](https://docs.molcrafts.org/molrs/contributing/) for the
documentation loop.

## Quick start

```rust
use molrs::conformer::{Conformer, ConformerOptions};
use molrs::smiles::{parse_smiles, to_atomistic};

let ir = parse_smiles("c1ccccc1").unwrap();          // benzene
let mol = to_atomistic(&ir).unwrap();
let (mol3d, _report) = Conformer::new(ConformerOptions::default()).generate(&mol).unwrap();
```

Python and JavaScript/TypeScript quickstarts live in the documentation.

## Documentation

- [Documentation site](https://docs.molcrafts.org/molrs/) — guides and references
- [Getting started](https://docs.molcrafts.org/molrs/getting-started/installation/) — Rust, Python, and WASM quickstarts
- [Guides](https://docs.molcrafts.org/molrs/guides/data-model/) — data model, SMILES, neighbor search, 3D embedding, force fields, I/O, trajectory analysis
- [Rust API reference](https://docs.rs/molcrafts-molrs) — full rustdoc on docs.rs

## MolCrafts ecosystem

| Project | Role |
|---------|------|
| [molpy](https://github.com/MolCrafts/molpy)     | Python toolkit — the shared molecular data model & workflow layer |
| **molrs** | Rust core — molecular data structures & compute kernels (native + WASM) — this repo |
| [molpack](https://github.com/MolCrafts/molpack) | Packmol-grade molecular packing (Rust + Python) |
| [molvis](https://github.com/MolCrafts/molvis)   | WebGL molecular visualization & editing |
| [molexp](https://github.com/MolCrafts/molexp)   | Workflow & experiment-management platform |
| [molnex](https://github.com/MolCrafts/molnex)   | Molecular machine-learning framework |
| [molq](https://github.com/MolCrafts/molq)       | Unified job queue — local / SLURM / PBS / LSF |
| [molcfg](https://github.com/MolCrafts/molcfg)   | Layered configuration library |
| [mollog](https://github.com/MolCrafts/mollog)   | Structured logging, stdlib-compatible |
| [molhub](https://github.com/MolCrafts/molhub)   | Molecular dataset hub |
| [molmcp](https://github.com/MolCrafts/molmcp)   | MCP server for the ecosystem |
| [molrec](https://github.com/MolCrafts/molrec)   | Atomistic record specification |

## Contributing

See [CONTRIBUTING](https://docs.molcrafts.org/molrs/contributing/) for development setup and guidelines.

## License

BSD-3-Clause — see [LICENSE](LICENSE).

<hr>

<div align="center">
<sub>Crafted with 💚 by <a href="https://github.com/MolCrafts">MolCrafts</a></sub>
</div>
