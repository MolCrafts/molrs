//! Integration tests for the `typifier` module tree. Mirrors `src/typifier/`.

#[path = "am1bcc.rs"]
mod am1bcc;
#[path = "am1bcc_antechamber.rs"]
mod am1bcc_antechamber;

#[path = "bcc_bond_type.rs"]
mod bcc_bond_type;

#[path = "estimate.rs"]
mod estimate;

#[path = "estimate_parity.rs"]
mod estimate_parity;

#[path = "mmff.rs"]
mod mmff;

#[path = "opls.rs"]
mod opls;
