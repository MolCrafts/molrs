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
#[path = "atd_conjugated.rs"]
mod atd_conjugated;
#[path = "atd_no_runtime_parse.rs"]
mod atd_no_runtime_parse;
#[path = "atd_tables_only.rs"]
mod atd_tables_only;

#[path = "bcc_bond_type.rs"]
mod bcc_bond_type;

#[path = "parameter_set_required.rs"]
mod parameter_set_required;

#[path = "equivalence_antechamber.rs"]
mod equivalence_antechamber;

#[path = "estimate.rs"]
mod estimate;

/// The last runtime text parse on the FF path (chem-perceive-10 ac-002) — RED until 11.
#[path = "estimate_tables_generated.rs"]
mod estimate_tables_generated;

/// The parmchk2 missing-parameter oracle (chem-perceive-10 ac-003) — RED until 11.
#[path = "parmchk2_oracle.rs"]
mod parmchk2_oracle;

#[path = "mmff.rs"]
mod mmff;

#[path = "opls.rs"]
mod opls;
