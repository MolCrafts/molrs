//! MMFF94 bond/angle/torsion type classification.

use super::params::MMFFParams;

/// Classify MMFF bond type: 0=normal, 1=delocalized/aromatic.
pub(crate) fn classify_bond_type(t1: u32, t2: u32, bond_order: f64, params: &MMFFParams) -> u32 {
    // MMFF bond type 1 = single bond between atoms where at least one
    // has sbmb=1 and the bond is delocalized (aromatic or resonance)
    if (bond_order - 1.5).abs() < 0.1 {
        return 1; // aromatic bond
    }
    if bond_order < 1.25 {
        // single bond: check if both atoms have sbmb property
        let p1_sbmb = params.get_prop(t1).map(|p| p.sbmb).unwrap_or(0);
        let p2_sbmb = params.get_prop(t2).map(|p| p.sbmb).unwrap_or(0);
        if p1_sbmb == 1 && p2_sbmb == 1 {
            return 1;
        }
    }
    0
}

/// Classify MMFF angle type from bond types of the two bonds forming the angle.
///
/// Returns 0..8 based on MMFF angle type rules.
pub(crate) fn classify_angle_type(bt_ij: u32, bt_jk: u32) -> u32 {
    match (bt_ij, bt_jk) {
        (0, 0) => 0,
        (1, 0) | (0, 1) => 1,
        (1, 1) => 2,
        _ => 0,
    }
}

/// Classify MMFF torsion type from the three bond types in the dihedral.
///
/// Returns 0..5 based on MMFF torsion type rules.
pub(crate) fn classify_torsion_type(bt_ij: u32, bt_jk: u32, bt_kl: u32) -> u32 {
    // Only the central bond matters primarily
    if bt_jk == 1 {
        return 1;
    }
    if bt_ij == 1 && bt_kl == 1 {
        return 2;
    }
    0
}
