//! Gaussian Cube for the WASM API — the face of `molrs::io::cube`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readCubeStr` | `read_cube_str` |
//! | `writeCubeStr` | `write_cube_str` (needs a `"grid"` block) |
//!
//! molrs reads cube files with functions only, so JS has no reader class.

use wasm_bindgen::prelude::*;

read_door!(
    /// Read Gaussian cube text into a frame with:
    /// - `"atoms"` block: `element` (string), `atomic_number` (i32),
    ///   `charge` (F), `x`/`y`/`z` (F, **always Å** — Bohr files are converted
    ///   on read).
    /// - `"grid"` block: structural shape `[nx, ny, nz]` and one f64 column
    ///   per scalar field — `density` for single-density files,
    ///   `mo_<idx>` for negative-natoms multi-orbital files.
    /// - `box`: voxel cell × dims in Å.
    ///
    /// ```js
    /// const frame   = readCubeStr(await file.text());
    /// const density = frame.get("grid").copy("density"); // owned Float64Array
    /// ```
    readCubeStr => read_cube_str(text: &str), "Cube"
);
write_door!(
    /// Write `frame` as Gaussian cube text (needs a `"grid"` block).
    writeCubeStr => write_cube_str -> String, "Cube"
);
