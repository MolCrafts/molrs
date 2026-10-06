//! Per-element data — the WASM face of `molrs::system::Element`.

use wasm_bindgen::prelude::*;

/// Covalent radius (in angstrom) for an element symbol.
///
/// Case-insensitive lookup against the built-in periodic table. Returns
/// `null` (`undefined` in JS) for an unrecognised symbol. This is the
/// single source of truth for covalent radii — downstream code (e.g. bond
/// perception) should call this rather than carrying its own table.
///
/// # Example (JavaScript)
///
/// ```js
/// covalentRadius("C");  // 0.76
/// covalentRadius("h");  // 0.31 (case-insensitive)
/// covalentRadius("Xx"); // undefined
/// ```
#[wasm_bindgen(js_name = covalentRadius)]
pub fn covalent_radius(symbol: &str) -> Option<f64> {
    molrs::system::Element::by_symbol(symbol).map(|el| f64::from(el.covalent_radius()))
}
