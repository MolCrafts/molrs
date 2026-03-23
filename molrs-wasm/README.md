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
- **`Block`** — column store with typed arrays (`Float32Array`, `Int32Array`, `Uint32Array`, `string[]`)
- **`Box`** — simulation box with periodic boundary conditions

### I/O

- `parseSMILES(smiles)` → `SmilesIR` → `.toFrame()`
- `XYZReader`, `PDBReader`, `LAMMPSReader` — file format parsers
- `writeFrame(frame, "xyz" | "pdb")` — serialize to string
- `SimulationReader` — Zarr V3 trajectory reader

### 3D generation

- `generate3D(frame, speed?)` — MMFF94 coordinate generation (`"fast"` | `"normal"` | `"thorough"`)

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
| `atoms` | `x`, `y`, `z` | `f32` | Cartesian coordinates |
| `atoms` | `mass` | `f32` | Atomic mass |
| `atoms` | `charge` | `f32` | Partial charge |
| `bonds` | `i`, `j` | `u32` | Atom indices |
| `bonds` | `order` | `f32` | Bond order (1.0, 1.5, 2.0, 3.0) |

## Build from source

```bash
wasm-pack build --target bundler --scope molcrafts --out-name molrs
```

## License

BSD-3-Clause
