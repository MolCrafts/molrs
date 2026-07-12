//! Integration tests for the `typifier` module tree. Mirrors `src/typifier/`.

/// The AmberTools25 reference fixture, declared **once** for the whole test target:
/// both `am1bcc_antechamber` (the BCC half) and `equivalence_antechamber` (the
/// equivalencing half) read it, and loading the same file as two modules would
/// compile the 37-molecule table into the binary twice (`clippy::duplicate_mod`).
#[path = "antechamber_oracle.rs"]
pub mod antechamber_oracle;

/// Shared oracle -> `Atomistic` builder, so a new oracle test cannot quietly
/// build its molecule from antechamber's *answers* instead of a user's input.
#[path = "oracle_mol.rs"]
pub mod oracle_mol;

/// Shared source scanner for the two structural gates (`atd_no_runtime_parse`,
/// `parameter_set_required`).
#[path = "source_gate.rs"]
pub mod source_gate;

#[path = "am1bcc.rs"]
mod am1bcc;
#[path = "am1bcc_antechamber.rs"]
mod am1bcc_antechamber;

#[path = "atd_antechamber.rs"]
mod atd_antechamber;
#[path = "atd_no_runtime_parse.rs"]
mod atd_no_runtime_parse;

#[path = "bcc_bond_type.rs"]
mod bcc_bond_type;

#[path = "parameter_set_required.rs"]
mod parameter_set_required;

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
