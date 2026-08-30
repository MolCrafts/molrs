# mrec-format-04-wasm — wasm JS name TrajectoryReader wrapping FrameSequence

Public contract (hard-coded golden names; no live wasm-pack / browser / npm):

- JS export is `TrajectoryReader`, wrapping Rust `FrameSequence`.
- Adapter path stays `molrs-wasm/src/io/zarr/mod.rs` (not renamed to `mrec/`).
- No `FrameReader` JS name. No `RecordReader` JS name.
- Rust import is `use molrs::io::mrec::FrameSequence`.

Runnable gate (repo root). Provenance: hand-written name literals from spec
`mrec-format-04-wasm`, 2026-08-30; `rg` 14.x.

```bash
src=molrs-wasm/src/io/zarr/mod.rs
test -f "$src"
rg -n --fixed-strings 'js_name = TrajectoryReader' "$src"
! rg -n --fixed-strings 'js_name = RecordReader' "$src"
! rg -n --fixed-strings 'js_name = FrameReader' "$src"
! rg -n --fixed-strings 'js_class = FrameReader' "$src"
rg -n --fixed-strings 'use molrs::io::mrec::FrameSequence' "$src"
```

Unit pin: `cargo test --manifest-path molrs-wasm/Cargo.toml --lib export_pin`
(native `#[test]`, not a browser e2e).
