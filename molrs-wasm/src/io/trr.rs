//! GROMACS TRR trajectories for the WASM API — the face of `molrs::io::trr`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `TrrStream` | `TrrIndexBuilder` + `read_trr_bytes` (the chunk-fed TRR reader) |
//! | `readTrrBytes`, `writeTrrBytes` | `read_trr_bytes`, `write_trr_bytes` (nm ↔ Å) |

use molrs::io::trr::TrrIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = TrrStream,
    indexer = TrrIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_trr_bytes(bytes),
}

read_door!(
    /// Read one GROMACS TRR frame from bytes (nm → Å).
    readTrrBytes => read_trr_bytes(bytes: &[u8]), "TRR"
);
write_door!(
    /// Write `frame` as a one-frame GROMACS TRR trajectory (Å → nm).
    writeTrrBytes => write_trr_bytes -> Vec<u8>, "TRR"
);
