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
