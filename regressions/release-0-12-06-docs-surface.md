# Docs surface 0.12 grep gates

Forbidden in published docs / examples:
- `molrs.parse_smiles` (Python free fn) — use `molrs.io.SmilesIR`
- `generate_3d` / `EmbedOptions`
- `potentials.eval`
- `typifier.build` for MMFF/OPLS
- multi-crate docs.rs: molcrafts-molrs-{core,io,ff,compute,conformer}
- version pins 0.0.15 / 0.10 / 0.11 in install examples

OK:
- Rust `molrs::smiles::parse_smiles` under feature `smiles`
- WASM `parseSMILES` naming note
