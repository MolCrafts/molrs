//! Integration tests for the `mmff` module tree (typing + charges),
//! validated per-atom against RDKit fixtures.

#[path = "typing.rs"]
mod typing;

#[path = "energy.rs"]
mod energy;

// `bespoke_gate.rs` + `BESPOKE.sha256` lived here: a SHA-256 pin on every file
// under `src/ff/mmff/`, so that mmff-orthogonal-01 could not touch the reference
// implementation it was measuring the generic path against. mmff-orthogonal-02 is
// the spec that deletes that tree, so the gate went with it — its own docs said so.
// Its successor is `deletion_gate.rs`, which asserts the mirror image.

/// mmff-orthogonal-02 ac-005 / ac-006 / ac-007 — the 4,065 dead XML rows, the
/// bespoke energy layer and the wrong classifiers are gone; the RDKit-faithful
/// resolver SURVIVES, out of `energy/`. RED until the deletion lands.
#[path = "deletion_gate.rs"]
mod deletion_gate;

/// mmff-orthogonal-02 ac-001 / ac-008 — the RDKit oracle and the parity
/// tolerances are the baseline the deletion is measured against, so neither may
/// be edited to make the suite green. GREEN today, and must stay green.
#[path = "oracle_gate.rs"]
mod oracle_gate;
