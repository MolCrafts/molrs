//! GROMACS XTC trajectories for the WASM API — the face of `molrs::io::xtc`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `XtcStream` | `XtcIndexBuilder` + `read_xtc_bytes` (the chunk-fed XTC reader) |
//! | `readXtcBytes`, `writeXtcBytes` | `read_xtc_bytes`, `write_xtc_bytes` (nm ↔ Å) |

use molrs::io::xtc::XtcIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = XtcStream,
    indexer = XtcIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_xtc_bytes(bytes),
}

read_door!(
    /// Read one GROMACS XTC frame from bytes (nm → Å).
    readXtcBytes => read_xtc_bytes(bytes: &[u8]), "XTC"
);
write_door!(
    /// Write `frame` as a one-frame GROMACS XTC trajectory (Å → nm).
    writeXtcBytes => write_xtc_bytes -> Vec<u8>, "XTC"
);
