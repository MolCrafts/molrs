//! Integration tests for the `typifier` module tree. Mirrors `src/typifier/`.

/// The AmberTools25 reference fixture, declared **once** for the whole test target:
/// both `am1bcc_antechamber` (the BCC half) and `equivalence_antechamber` (the
/// equivalencing half) read it, and loading the same file as two modules would
/// compile the 37-molecule table into the binary twice (`clippy::duplicate_mod`).
#[path = "antechamber_oracle.rs"]
pub mod antechamber_oracle;

#[path = "am1bcc.rs"]
mod am1bcc;
#[path = "am1bcc_antechamber.rs"]
mod am1bcc_antechamber;

#[path = "bcc_bond_type.rs"]
mod bcc_bond_type;

#[path = "equivalence_antechamber.rs"]
mod equivalence_antechamber;

#[path = "estimate.rs"]
mod estimate;

#[path = "estimate_parity.rs"]
mod estimate_parity;

#[path = "mmff.rs"]
mod mmff;

#[path = "opls.rs"]
mod opls;
