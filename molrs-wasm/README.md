# @molcrafts/molrs

[![npm](https://img.shields.io/npm/v/@molcrafts/molrs.svg)](https://www.npmjs.com/package/@molcrafts/molrs)

WebAssembly bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

Full documentation lives at <https://docs.molcrafts.org/molrs/>. The package
ships its TypeScript declarations (`molrs.d.ts`); the
[WASM reference page](https://docs.molcrafts.org/molrs/reference/wasm/) maps
the main exports. Upgrading from 0.15? See the
[migration guide](https://docs.molcrafts.org/molrs/migration/).

## Install

```bash
npm install @molcrafts/molrs
```

## Quick start

```js
import { parseSMILES, generate3D, writeFrame } from "@molcrafts/molrs";

// Parse SMILES → 3D coordinates → XYZ string
const ir = parseSMILES("CCO");
const frame = ir.toFrame();
const mol3d = generate3D(frame, "fast");
console.log(writeFrame(mol3d, "xyz"));
```

The published package uses wasm-pack's `bundler` target: configure your bundler
to load WebAssembly modules. A custom `--target web` build instead exports an
async `init()` function that must be awaited before calling the API.

## API

### Data model

- **`Frame`** — container mapping string keys (`"atoms"`, `"bonds"`) to `Block`s
- **`Block`** — column store with typed arrays. Float columns are `Float64Array` (F = f64).
- **`Box`** — simulation box with periodic boundary conditions
- **`NDArray`** — owned `Float64Array` plus a shape (what `Box.origin()`, `Box.hMatrix()` and `Box.lengths()` return)
- **`Topology`** — the bond graph (`Topology.fromFrame(frame)` reads
  `bonds.atomi` / `atomj`): angles, dihedrals, impropers, connected components

Columns read like numpy: the column's dtype picks the array type, and no
method names a dtype.

```js
const atoms = frame.get("atoms");                 // Block; throws if absent (frame.has)
atoms.set("x", new Float64Array([0, 1, 2]));      // dtype from the constructor: f64
atoms.set("element", ["C", "C", "O"]);            // string[] → string
atoms.set("pos", new Float64Array(9), [3, 3]);    // optional row-major shape
const v = atoms.view("x");                        // zero-copy Float64Array; invalid after WASM memory grows
const x = atoms.copy("x");                        // owned Float64Array to keep
const q = atoms.get("charge", new Float64Array(atoms.nrows)); // copy, or this default if absent
atoms.dtype("x"); atoms.shape("pos"); atoms.has("x"); atoms.keys();
```

| dtype | `view` | `copy` / `get` | `set` accepts |
|-------|--------|----------------|---------------|
| `f64` | `Float64Array` | `Float64Array` | `Float64Array` (`Float32Array` is refused) |
| `i8` `i16` `i32` `i64` | same typed array | same typed array | same |
| `u8` `u16` `u32` `u64` | same typed array | same typed array | same |
| `bool` | throws; use `copy` | `boolean[]` | `boolean[]` |
| `string` | throws; use `copy` | `string[]` | `string[]` (and `[]`) |
| `c64` / `c128` | throws; use `copy` | `{ real, imag, shape, dtype }` | never |

### Perception

```js
const rings = assignRings(frame);     // atoms/bonds gain is_in_ring, n_rings
const arom  = assignAromaticity(frame);
const withH = addHydrogens(frame);
```

### I/O

- `parseSMILES(smiles)` → `SmilesIR` → `.toFrame()`
- `XYZStream`, `PDBStream`, `SDFStream`, `LAMMPSStream`, `LAMMPSTrajStream`,
  `DCDStream`, `XTCStream`, `TRRStream` — chunk-fed readers, the one reader of
  their format (`allocInputBuffer` → `feedIndexChunk` / `finishIndex` →
  `parseRangeInInput` per frame)
- `CIFReader`, `GROReader`, `MOL2Reader`, `POSCARReader`, `XSFReader`,
  `CubeReader`, `CHGCARReader`, `AmberInpcrdReader`, `AcReader` — whole-content
  readers of the formats with no stream
- `covalentRadius(symbol)` — the element table's covalent radius (Å)
- `writeFrame(frame, "xyz" | "pdb" | "lammps-data" | "lammps-dump")` — serialize to string

### 3D generation

- `generate3D(frame, speed?, seed?)` — MMFF94 coordinate generation (`"fast"` | `"medium"` | `"better"`)

### Force fields + geometry optimization

```js
const typifier = new UFFTypifier();                 // or MMFF94Typifier / MMFF94STypifier
const typed    = typifier.typify(frame);
const pots     = typifier.toPotentials(typed);      // compiles the typed output; no forcefield() handle
const nl       = new NeighborList(12.5);            // or NeighborList.bruteForce(12.5)
nl.build(typed);
const report   = new Lbfgs(pots, nl.neighbors()).minimize(typed);  // pairs come from the NeighborList
```

- **UFF** — full RDKit default table (entire periodic table + oxidation states)
- **MMFF94 / MMFF94s** — Merck force fields
- **no GFN-FF**
- **no** free-function `intramolecularPairs` / `insertIntramolecularPairs`

### Analysis

```js
import { NeighborList, Rdf } from "@molcrafts/molrs";

const nl = new NeighborList(5.0);         // cutoff = 5.0 A, O(N) cell list
nl.build(frame);                          // index only — no pair table
const nlist = nl.neighbors();             // materialize: distSq + disp

const rdf = new Rdf(100, 5.0);
const result = rdf.compute(frame);        // streams its own neighbor search
console.log(result.binCenters(), result.rdf());
```

A self search is **half-shell**: each unordered pair appears once, with
`i < j`. `neighbors()` keeps both physical columns by default — that names the
*columns*, not the pair direction. Pass `{ distSq: false }` or `{ disp: false }`
to drop one; a column that was not stored reads back as `undefined`, never as a
fabricated zero array. `disp` is the unnormalized minimum-image displacement
`r_j - r_i` (Å), flattened three values per pair.

- **`NeighborList`** — neighbor-search engine (`build` / `update` index,
  `neighbors` materializes); `NeighborList.bruteForce(cutoff)` selects the
  O(N²) reference backend
- **`Neighbors`** — the materialized pair table (`nPairs`, `queryPointIndices()`,
  `pointIndices()`, `distSq()`, `disp()`)
- **`NeighborQuery`** — the cross search: `new NeighborQuery(refFrame, cutoff)`
  indexes a reference frame, `query(otherFrame)` returns the directed pairs
- **`Rdf`** — radial distribution function (periodic and free-boundary)
- **`Msd`** — mean squared displacement
- **`Cluster`** — distance-based cluster analysis
- **`Vacf`**, **`Steinhardt`**, **`HBonds`**, **`PmftXy`**, **`RadicalVoronoi`**, …
  — one class per analysis, named after its molrs owner;
  `molrsComputeCatalog()` lists every one with its parameters

Neighbor searches support frames without a simulation box. RDF additionally
needs a normalization volume: for a frame without a box, pass it as the fourth
constructor argument, e.g. `new Rdf(100, 5.0, undefined, 1000.0)`.

### Block column conventions

Names and dtypes are the Frame schema. `schemaDocument()` is that vocabulary
(`schemaJson()` is the same document as text). `keysDocument()` is the constant
names (`X`, `BOND_TYPE`, `ATOMS`, `UNITS`, …) projected from the same tables.
The Rust and Python bindings print the vocabulary with `schema.to_markdown()`.

`F` is the molrs core float type — always `f64`.

## Trajectory stores (`*.mrec`)

Records written by molrs in Python or Rust (see
[Record files](https://docs.molcrafts.org/molrs/guides/records/)) read here
from bytes. `readMrecFrame(files)` / `readMrecFrameFromZip(bytes)` return the
`frame` section (or `undefined`), and `mrecSections(files)` lists the
sections. Every reader decodes `zstd` and `shuffle`, so columns written with
a declared precision read as they do natively.

`MrecReader` opens a MolRec trajectory (Zarr V3) and decodes one frame
per call; consecutive frames of the same chunk are slices, not decodes.

```js
import { MrecReader } from "@molcrafts/molrs";

// (a) every file in memory
const reader = new MrecReader(files);            // Map<path, Uint8Array>
// (b) a packed store
const zipped = MrecReader.fromZip(bytes);        // Uint8Array of *.mrec.zip
// (c) served on demand — only touched chunks cross into wasm
const lazy = MrecReader.fromStore({
  get: (key) => ...,                                   // Uint8Array | null
  getRange: (key, offset, length) => ...,              // length -1 = to end
  size: (key) => ...,                                  // number | null
  list: (prefix) => [...],                             // keys under prefix
});

reader.countFrames();
const frame = reader.readFrame(t);                     // Frame | undefined
const xyz = reader.readColumns(t, ["atoms/x", "atoms/y", "atoms/z"]);
reader.blockUpdateAt("bonds", t);                      // same value ⇒ same rows
reader.boxAt(t);
```

The host callbacks of `fromStore` are synchronous: in a Worker that is
`FileReaderSync` over `File` handles or a synchronous range request; on the
main thread hand the reader a `Map` instead.

## Build from source

```bash
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

This writes `pkg/` — the npm package. `pkg/package.json` is auto-generated by
wasm-pack with name `@molcrafts/molrs`. Consumers link it directly:

```jsonc
// consumer's package.json
"dependencies": {
  "@molcrafts/molrs": "link:../path/to/molrs-wasm/pkg"
}
```

Then `npm install` creates a symlink — rebuilding `pkg/` (via `wasm-pack build`)
is picked up immediately by the consumer's dev server. No `npm link` dance
needed.

### Variants (optional)

Default features are `smiles`, `io`, `compute`, `conformer`, `voronoi`, `stream`, and `builder`. To
build a smaller wasm containing only a subset, use Cargo features:

```bash
wasm-pack build --release --target bundler --scope molcrafts \
  --out-name molrs \
  --no-default-features --features io,smiles
```

Note: **variants are mutually exclusive at runtime** — you can't mix a
`compute`-only build with an `io`-only build in the same app, because each
produces a separate wasm module with its own `Frame` class identity.

## License

BSD-3-Clause
