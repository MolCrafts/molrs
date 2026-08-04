//! The SMILES system: parsing, validation, and atomistic-graph conversion.
//!
//! SMILES is a *serialization format* for concrete molecular structures. This
//! module owns everything that is specific to producing or consuming SMILES
//! strings — parsing entry point, element-symbol validation, and the IR →
//! [`Atomistic`](molrs::system::atomistic::Atomistic) conversion.
//!
//! The SMARTS query engine lives in [`crate::perceive::smarts`]. Shared AST
//! vocabulary and scanner live in [`chem`](crate::io::smiles::chem).

pub mod to_atomistic;
pub mod validate;

pub use crate::io::smiles::parser::parse_smiles;
pub use to_atomistic::to_atomistic;
pub use validate::validate_smiles;

/// The element symbol a SMILES atom symbol denotes.
///
/// SMILES writes aromatic atoms in lowercase (`c`, `n`, `se`); that is
/// notation, not an element symbol. Every consumer that keys off `element` —
/// mass tables, typifiers, force-field parameter lookup — expects the
/// canonical capitalisation, so both validation and graph construction
/// normalise through here.
pub(crate) fn canonical_element_symbol(symbol: &str) -> String {
    let mut chars = symbol.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
    }
}
