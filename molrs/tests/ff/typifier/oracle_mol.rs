//! Shared `AntechamberCase` -> `Atomistic` builder for the oracle-driven tests.
//!
//! Every typifier test that consumes `antechamber_oracle::CASES` needs the same
//! molecule, built from the INPUT half of the case ONLY — element, coordinates,
//! bond order, aromatic flag, formal charge. Never from antechamber's answers.
//! Keeping one builder here is a guard, not just DRY: a test that quietly fed the
//! typifier antechamber's own bond types would be testing a molecule no molrs
//! user has, and would go green on a perception bug.

use molrs::store::keys;
use molrs::{AtomId, Atomistic};

use super::antechamber_oracle::AntechamberCase;

/// Build the molecule a real user has after reading a mol2/SDF.
///
/// Returns the graph and the atom ids in oracle atom order, so an oracle column
/// can be zipped straight onto it.
pub fn build_case(case: &AntechamberCase) -> (Atomistic, Vec<AtomId>) {
    build_molecule(case.elements, case.xyz, case.formal_charges, case.bonds)
}

/// The builder itself, over the four INPUT columns every antechamber-derived
/// fixture carries: element, coordinates, formal charge, and `(i, j, order,
/// is_aromatic)` bonds.
///
/// Split out of [`build_case`] so a fixture that is not an `AntechamberCase` —
/// the out-of-oracle probes in `atd_conjugated.rs` — reaches the typifier through
/// the SAME builder rather than a copy of it. Two builders would be two chances to
/// hand the typifier a molecule no molrs user has (antechamber's perceived bond
/// types instead of the SDF's), and the copy would be the one nobody re-reads.
pub fn build_molecule(
    elements: &[&str],
    xyz: &[[f64; 3]],
    formal_charges: &[i32],
    bonds: &[(usize, usize, f64, bool)],
) -> (Atomistic, Vec<AtomId>) {
    let mut mol = Atomistic::new();
    let ids: Vec<AtomId> = elements
        .iter()
        .zip(xyz)
        .map(|(el, [x, y, z])| mol.add_atom_xyz(el, *x, *y, *z))
        .collect();
    for (aid, fc) in ids.iter().zip(formal_charges) {
        mol.set_atom(*aid, "formal_charge", *fc)
            .expect("set formal_charge");
    }
    for (i, j, order, aromatic) in bonds {
        let bid = mol.add_bond(ids[*i], ids[*j]).expect("add bond");
        mol.set_bond_prop(bid, keys::ORDER, *order)
            .expect("set bond order");
        if *aromatic {
            mol.set_bond_prop(bid, "is_aromatic", true)
                .expect("set aromatic");
        }
    }
    (mol, ids)
}

/// The atom `type` labels of `typed`, in oracle atom order.
pub fn atom_types(typed: &Atomistic, ids: &[AtomId]) -> Vec<Option<String>> {
    ids.iter()
        .map(|aid| {
            typed
                .get_atom(*aid)
                .expect("typed graph keeps the input atom ids")
                .get_str(keys::TYPE)
                .map(str::to_owned)
        })
        .collect()
}

/// Panic with a per-molecule report when any molecule disagreed with antechamber.
///
/// The oracle is 37 molecules; a bare `assert_eq!` per atom would report the
/// first disagreement and hide the shape of the failure (one bad element vs one
/// bad ring perception vs everything).
pub fn report(label: &str, failures: &[String], total: usize) {
    if failures.is_empty() {
        return;
    }
    panic!(
        "{label}: {}/{total} molecules disagree with antechamber\n{}",
        failures.len(),
        failures.join("\n")
    );
}
