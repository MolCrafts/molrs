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
///
/// Named for the property it pins, like its sibling `atd_no_runtime_parse`. It used
/// to be `estimate_tables_generated.rs`, which named how the tables ARRIVED rather
/// than what they ARE — the exact mistake `chem-perceive-14` corrects everywhere else.
#[path = "estimate_no_runtime_parse.rs"]
mod estimate_no_runtime_parse;

/// The parmchk2 missing-parameter oracle (chem-perceive-10 ac-003) — RED until 11.
#[path = "parmchk2_oracle.rs"]
mod parmchk2_oracle;

#[path = "mmff.rs"]
mod mmff;

/// The two MMFF front doors (`mmff-typifier-split`) — RED until `MMFF94Typifier` /
/// `MMFF94STypifier` exist and `frame_builder` stops hardcoding the variant.
#[path = "mmff_variant.rs"]
mod mmff_variant;

/// mmff-orthogonal-02 ac-007 (runtime half) — every bond / angle / dihedral label
/// derives from the ONE RDKit-faithful resolver, so aromatic bonds are type 0 and
/// a 3-ring angle is type 3. RED until `classify.rs` is deleted: its classifiers
/// answer 1 and 0 respectively, and its angle signature cannot express the ring
/// rule at all.
#[path = "mmff_labels.rs"]
mod mmff_labels;

#[path = "opls.rs"]
mod opls;
