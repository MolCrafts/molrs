//! Integration tests for the `mmff` module tree (typing + charges),
//! validated per-atom against RDKit fixtures.

#[path = "typing.rs"]
mod typing;

#[path = "energy.rs"]
mod energy;

/// ac-011 — the bespoke path (`src/ff/mmff/`) is the reference frame the generic
/// fix is measured against, so mmff-orthogonal-01 must not touch a byte of it.
#[path = "bespoke_gate.rs"]
mod bespoke_gate;
