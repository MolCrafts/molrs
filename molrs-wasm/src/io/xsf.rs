//! XCrySDen XSF for the WASM API — the face of `molrs::io::xsf`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readXsfStr` | `read_xsf_str` |
//! | `writeXsfStr` | `write_xsf_str` |
//!
//! molrs reads XSF with functions only, so JS has no reader class.

use wasm_bindgen::prelude::*;

read_door!(
    /// Read XCrySDen XSF text: a periodic box from `CRYSTAL` + `PRIMVEC`, a
    /// free one from `MOLECULE`; atoms with `atomic_number`, `element` and
    /// `x`/`y`/`z` (Å).
    readXsfStr => read_xsf_str(text: &str), "XSF"
);
write_door!(
    /// Write `frame` as XCrySDen XSF text.
    writeXsfStr => write_xsf_str -> String, "XSF"
);
