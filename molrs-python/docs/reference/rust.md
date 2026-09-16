# Rust Reference

Rust API reference is built and hosted by docs.rs. The Zensical site links to
those pages instead of copying signatures or rustdoc text.

| Crate | Reference |
| --- | --- |
| Library (single crate) | [`molcrafts-molrs`](https://docs.rs/molcrafts-molrs) |
| CXX bridge (source / Atomiverse) | `molcrafts-molrs-cxxapi` (not on crates.io as a separate science package) |

Always compiled inside `molcrafts-molrs`: `core`, `perceive`, `builder`.
Feature-gated modules: `io`, `smiles` (inside `io`), `ff`, `conformer`,
`compute`, `voronoi`, `signal`, `md`, `stream`. `full` bundles every one of
those except `stream`; the crate defaults are `full`, `stream`, `filesystem`,
`rayon`.

The Packmol-aligned packing workflow lives in the separate
[`molpack`](https://github.com/MolCrafts/molpack) repository.
