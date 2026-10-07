# WASM Quickstart

The npm package `@molcrafts/molrs` exports the classes and functions generated
by `wasm-bindgen`. It is built for bundlers (Vite, webpack, Rollup with a
WebAssembly plugin): importing the package loads and starts the WebAssembly
module, so there is no separate initialization call.

## 1. Install and Import

```bash
npm install @molcrafts/molrs
```

```ts
import { SmilesIr, Conformer, writeXyzStr } from "@molcrafts/molrs";
```

Configure your bundler to load `.wasm` modules. A custom build with
`wasm-pack build --target web` instead exports an async default `init()`
that must be awaited before any other call.

## 2. Parse, Embed, and Export

```ts
const ir = SmilesIr.parse("CCO");
const frame2d = ir.toFrame();
const frame3d = new Conformer("fast", true, 42).generate(frame2d);

console.log(writeXyzStr(frame3d));
```

The API shape mirrors Rust and Python, with JavaScript naming conventions:
`SmilesIr::parse` becomes `SmilesIr.parse`, `to_frame` becomes `toFrame`,
`Conformer(...).generate` stays `Conformer.generate`, and the Rust door
`molrs::io::write_xyz_str` (Python `molrs.io.write_xyz_str`) becomes
`writeXyzStr` — every class and function is named after its molrs owner, and
every door names its format. A file arrives in the browser as text or bytes,
so JavaScript has the in-memory doors only: `read<Fmt>Str` / `write<Fmt>Str`
for text formats and `read<Fmt>Bytes` / `write<Fmt>Bytes` for binary ones,
the same set as Rust's and Python's `read_<fmt>_str` / `_bytes` and
`write_<fmt>_str` / `_bytes`; a format molrs reads with functions only (XSF,
cube, CHGCAR, AMBER inpcrd / `.ac`) has no reader class here either. The
TypeScript declarations in the package (`molrs.d.ts`) are the source of truth
for exported names.

```ts
import { readXyzStr, writePdbStr } from "@molcrafts/molrs";

const frame = readXyzStr(await file.text());
const pdb = writePdbStr(frame);
```

## 3. Inspect Columns

Frames contain blocks, and blocks contain typed columns. A column reads back
as the typed array of its stored dtype — `Float64Array` for floats — and no
method names a dtype. `dtype(key)` reports molrs core's dtype name (`"float"`,
`"int"`, `"uint"`, `"string"`, …), the one Python and the Frame schema use:

```ts
const atoms = frame3d.get("atoms"); // throws if absent; check with frame3d.has("atoms")

const x = atoms.copy("x");          // owned Float64Array
const y = atoms.copy("y");
const z = atoms.copy("z");

console.log(atoms.nRows, atoms.dtype("x"), x[0], y[0], z[0]);
```

`copy` returns an owned typed array that is safe to keep. `view(key)` returns
a zero-copy view into WebAssembly memory instead; it is faster for large
columns but becomes invalid when the module's memory grows, so read it
immediately or prefer `copy` until profiling says otherwise. `set(key, array)`
writes a column and takes its dtype from the typed array's constructor
(`Float32Array` and plain `number[]` are refused). The
[package README](https://github.com/MolCrafts/molrs/tree/master/molrs-wasm#core-data-model-molrscore)
has the full dtype table.

## 4. Read Record Files

The browser reads the same `*.mrec` [record files](../guides/records.md) that
Python and Rust write. A packed `*.mrec.zip` arrives as bytes:

```ts
import { MrecReader, readMrecFrame } from "@molcrafts/molrs";

const bytes = new Uint8Array(await (await fetch("run.mrec.zip")).arrayBuffer());
const reader = MrecReader.fromZip(bytes);
const first = reader.readFrame(0);
console.log(reader.nFrames(), first?.get("atoms").nRows);

const snapshot = readMrecFrame(
  new Uint8Array(await (await fetch("water.mrec.zip")).arrayBuffer()),
);
```

`readMrecFrame` (and `sectionNames`) take the packed bytes or a
`Map<path, Uint8Array>` of the record's files. `MrecReader.fromStorage` reads
chunks on demand through callbacks, so a
large trajectory never has to be downloaded whole.

## 5. Build from Source

The npm publish pipeline uses this bundler target:

```bash
cd molrs-wasm
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

This writes `molrs-wasm/pkg/`. That directory is generated and ignored by git.
Applications can link it locally during development:

```json
{
  "dependencies": {
    "@molcrafts/molrs": "link:../molrs-wasm/pkg"
  }
}
```

## 6. Variant Builds

The default npm build includes SMILES, I/O, compute, conformer, Voronoi,
builder, and stream support. Smaller local builds can disable default Cargo
features, but a page should load only one variant of the package at a time
because each WebAssembly module owns a distinct JavaScript class identity.
