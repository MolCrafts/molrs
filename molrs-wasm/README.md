# @molcrafts/molrs

[![npm](https://img.shields.io/npm/v/@molcrafts/molrs.svg)](https://www.npmjs.com/package/@molcrafts/molrs)

WebAssembly bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

## Install

```bash
npm install @molcrafts/molrs
```

## Quick start

```js
import init, { parseSMILES, generate3D, writeFrame } from "@molcrafts/molrs";

await init();

// Parse SMILES → 3D coordinates → XYZ string
const ir = parseSMILES("CCO");
const frame = ir.toFrame();
const mol3d = generate3D(frame, "fast");
console.log(writeFrame(mol3d, "xyz"));
```

## API

### Data model

- **`Frame`** — container mapping string keys (`"atoms"`, `"bonds"`) to `Block`s
- **`Block`** — column store with typed arrays. Float columns use `Float32Array` by default and `Float64Array` when built with the `f64` feature.
- **`Box`** — simulation box with periodic boundary conditions

### I/O

- `parseSMILES(smiles)` → `SmilesIR` → `.toFrame()`
- `XYZReader`, `PDBReader`, `LAMMPSReader` — file format parsers
- `writeFrame(frame, "xyz" | "pdb" | "lammps-data" | "lammps-dump")` — serialize to string
- `MolRecReader` — MolRec Zarr V3 reader

### 3D generation

- `generate3D(frame, speed?, seed?)` — MMFF94 coordinate generation (`"fast"` | `"medium"` | `"better"`)

### Analysis

```js
import { LinkedCell, RDF } from "@molcrafts/molrs";

const lc = new LinkedCell(5.0);           // cutoff = 5.0 A
const nlist = lc.build(frame);            // self-query (unique pairs i < j)
const cross = lc.query(refFrame, other);  // cross-query

const rdf = new RDF(100, 5.0);
const result = rdf.compute(frame, nlist);
console.log(result.binCenters(), result.rdf());
```

- **`LinkedCell`** — cell-list neighbor search (`build()` for self-query, `query()` for cross-query)
- **`RDF`** — radial distribution function (periodic and free-boundary)
- **`MSD`** — mean squared displacement
- **`Cluster`** — distance-based cluster analysis

Frames without a simulation box are supported — a non-periodic bounding box is auto-generated.

### Block column conventions

| Block | Column | Type | Description |
|-------|--------|------|-------------|
| `atoms` | `symbol` | `string` | Element symbol |
| `atoms` | `x`, `y`, `z` | `F` | Cartesian coordinates |
| `atoms` | `mass` | `F` | Atomic mass |
| `atoms` | `charge` | `F` | Partial charge |
| `bonds` | `i`, `j` | `u32` | Atom indices |
| `bonds` | `order` | `F` | Bond order (1.0, 1.5, 2.0, 3.0) |

`F` is the workspace float type from `molrs-core`. In the default WASM build it is `f32`; with `--features f64` it becomes `f64`, and the JS API switches to `Float64Array` at the boundary.

## Build from source and use it for development

```bash
wasm-pack build --target bundler --scope molcrafts --out-name molrs
npm link
cd ../my-app
npm link @molcrafts/molrs
```

## License

BSD-3-Clause
