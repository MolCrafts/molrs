//! The SMILES system: parsing, validation, and atomistic-graph conversion.
//!
//! SMILES is a *serialization format* for concrete molecular structures. This
//! module owns everything that is specific to producing or consuming SMILES
//! strings — parsing entry point, element-symbol validation, and the IR →
//! [`Atomistic`](molrs::system::Atomistic) conversion.
//!
//! The SMARTS query engine lives in [`crate::perceive::smarts`]. Shared AST
//! vocabulary and scanner live in [`chem`](crate::io::smiles::chem).

mod from_atomistic;
mod local_smarts;
mod options;
mod to_atomistic;
mod validate;
mod write;

pub use from_atomistic::{from_atomistic, write_atomistic_smiles};
pub use local_smarts::{local_smarts_ir, write_local_smarts};
pub use options::{
    AromaticEmit, HydrogenEmit, LocalSmartsOptions, MultiComponentEmit, NeighborStyle,
    SmilesEmitOptions,
};
pub use to_atomistic::{fragment_to_atomistic, read_smiles, to_atomistic};
pub use validate::validate_smiles;
pub use write::{write_fragment_smiles, write_smarts, write_smiles};

use molrs::system::Element;

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

/// Whether `symbol`, as written in a SMILES atom, names a real element.
///
/// The lookup is on the canonical capitalisation, so the aromatic lowercase
/// `se` is the element `Se` and `[Xx]` is nothing at all. Both stages that
/// decide this question ask here — [`parse_smiles`] when it reads a bracket
/// atom, and [`validate_smiles`] when it re-checks an IR it did not build —
/// so the parser and the validator cannot drift into two answers about the
/// same symbol.
pub(crate) fn is_element_symbol(symbol: &str) -> bool {
    Element::by_symbol(&canonical_element_symbol(symbol)).is_some()
}
