//! MMFF94 bond/angle/torsion/out-of-plane type classification.

use super::params::MMFFParams;
use crate::ff::mmff::tables::{mmff_angle, mmff_def, mmff_oop};

/// Typify an MMFF bond: 0=normal, 1=delocalized/aromatic.
pub(crate) fn typify_bond(t1: u32, t2: u32, bond_order: f64, params: &MMFFParams) -> u32 {
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

/// Typify an MMFF angle from the two MMFF bond classes forming the angle.
///
/// Returns 0..8 based on MMFF angle type rules.
pub(crate) fn typify_angle(bt_ij: u32, bt_jk: u32) -> u32 {
    match (bt_ij, bt_jk) {
        (0, 0) => 0,
        (1, 0) | (0, 1) => 1,
        (1, 1) => 2,
        _ => 0,
    }
}

/// Typify an MMFF dihedral from the three MMFF bond classes in the dihedral.
///
/// Returns 0..5 based on MMFF torsion type rules.
pub(crate) fn typify_dihedral(bt_ij: u32, bt_jk: u32, bt_kl: u32) -> u32 {
    // Only the central bond matters primarily
    if bt_jk == 1 {
        return 1;
    }
    if bt_ij == 1 && bt_kl == 1 {
        return 2;
    }
    0
}

/// Resolve the canonical MMFF94 angle parameter key for angle `i-j-k` (centre
/// `j`) of angle-type `angle_type`. Mirrors the RDKit-validated
/// `MMFFAngleCollection` lookup: the two end types are equivalence-degraded and
/// sorted (centre fixed) until a parameter row exists, and that row's
/// `"{at}_{i}_{j}_{k}"` key — exactly how the `mmff_angle` style is keyed — is
/// returned so the kernel resolves params by exact match. Falls back to the
/// level-0 sorted key when no row exists (MMFF would apply empirical rules
/// there; the table-keyed path surfaces that as an explicit unknown-type error).
pub(crate) fn resolve_angle_label(angle_type: u32, ti: u32, tj: u32, tk: u32) -> String {
    let (at, j) = (angle_type as u8, tj as u8);
    let (i, k) = (ti as u8, tk as u8);
    for level in 0..4 {
        let (mut ci, mut ck) = (eq_level(i, level), eq_level(k, level));
        if ci > ck {
            std::mem::swap(&mut ci, &mut ck);
        }
        if mmff_angle(at, ci, j, ck).is_some() {
            return format!("{at}_{ci}_{j}_{ck}");
        }
    }
    let (lo, hi) = if i <= k { (i, k) } else { (k, i) };
    format!("{at}_{lo}_{j}_{hi}")
}

/// MMFF equivalence-level degrade of an atom type (RDKit `MMFFDef`): level 0 is
/// the type itself; higher levels fall back to broader classes, terminating at
/// the type-0 wildcard. Mirrors the energy path's `eq_level`.
fn eq_level(atom_type: u8, level: usize) -> u8 {
    mmff_def(atom_type)
        .map(|d| d.eq_level[level])
        .unwrap_or(atom_type)
}

/// Resolve the canonical MMFF94 out-of-plane parameter key for a trigonal centre
/// of type `center` whose three peripheral atoms have types `periph`.
///
/// Mirrors the RDKit-validated `MMFFOopCollection` lookup: the three peripheral
/// types are degraded through up to four equivalence levels and sorted ascending
/// (the centre stays fixed in the second position) until a parameter row exists.
/// Returns that row's `"{i}_{j}_{k}_{l}"` key — exactly how the `mmff_oop`
/// improper style is keyed, so the kernel resolves `koop` by exact match — or
/// `None` when MMFF defines no out-of-plane term for the centre.
pub(crate) fn resolve_oop_label(center: u32, periph: [u32; 3]) -> Option<String> {
    let j = center as u8;
    let p = [periph[0] as u8, periph[1] as u8, periph[2] as u8];
    for level in 0..4 {
        let mut ikl = [
            eq_level(p[0], level),
            eq_level(p[1], level),
            eq_level(p[2], level),
        ];
        ikl.sort_unstable();
        if mmff_oop(ikl[0], j, ikl[1], ikl[2]).is_some() {
            return Some(format!("{}_{}_{}_{}", ikl[0], j, ikl[1], ikl[2]));
        }
    }
    None
}
