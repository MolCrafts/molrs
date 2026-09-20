# WASM Reference

The WebAssembly package generates TypeScript declarations from Rust
`wasm-bindgen` comments during the docs workflow:

```bash
cd molrs-wasm
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

The generated declaration file is not committed. In CI, TypeDoc consumes
`molrs-wasm/pkg/molrs.d.ts` and publishes the generated reference at this same
URL, replacing this fallback page in the deployed artifact.

Until the generated page is available, use the package declarations in
`molrs-wasm/pkg/` after a local `wasm-pack build`, or inspect the npm package
for `@molcrafts/molrs`.
