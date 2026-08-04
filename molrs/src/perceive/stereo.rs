//! Stereochemistry support for molecular graphs.
//!
//! Provides:
//! * [`TetrahedralStereo`] — CW / CCW / Unspecified chirality at a tetrahedral centre.
//! * [`BondStereo`] — E / Z / Either / None for double-bond stereochemistry.
//! * [`chiral_volume`] — signed scalar triple product from 3-D coordinates.
//! * [`find_chiral_centers`] — atoms with 4 distinct neighbours.
//! * [`assign_stereo_from_3d`] — infer tetrahedral chirality from coordinates.
//! * [`assign_bond_stereo_from_3d`] — infer E/Z from coordinates.
//!
//! # Storage convention
//! Stereochemistry labels may be persisted in atom/bond properties:
//! * atom `"stereo"` → `"CW"` | `"CCW"` | `"unspecified"`
//! * bond `"stereo"` → `"E"` | `"Z"` | `"either"` | `"none"`
//!
//! Two of those strings are *sentinels* rather than descriptors: `"unspecified"`
//! ([`TetrahedralStereo::Unspecified`]) means "this atom is not a stereocentre",
//! and `"none"` ([`BondStereo::None`]) means "this bond is not a stereo bond".
//! They record the **absence** of a stereochemical fact, not a perceived one.
//! Accordingly, the builder [`Perceive::find_stereo`](crate::perceive::Perceive::find_stereo)
//! writes **only** the real descriptors — `"CW"` / `"CCW"` on atoms and `"E"` /
//! `"Z"` / `"either"` on bonds — and omits the sentinels entirely, so a `"stereo"`
//! prop is present exactly where stereochemistry was perceived and absent
//! everywhere else. The full four-way vocabulary above still applies to graphs
//! written by other producers (a file reader, or a caller persisting the maps
//! returned by [`assign_stereo_from_3d`] / [`assign_bond_stereo_from_3d`], both of
//! which are total and do carry the sentinel variants).
//!
//! # Chiral-volume sign convention
//! Positive volume → CCW (S configuration when substituents are in CIP order).
//! Negative volume → CW  (R configuration).

use std::collections::HashMap;

use crate::store::keys;
use crate::system::atomistic::{AtomId, Atomistic, BondId};
use crate::system::bond::BondType;

// ---------------------------------------------------------------------------
// Public enums
// ---------------------------------------------------------------------------

/// Tetrahedral stereochemistry at an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TetrahedralStereo {
    /// Clockwise (R) — from the lowest-priority substituent's viewpoint.
    CW,
    /// Counter-clockwise (S).
    CCW,
    /// No stereo information available or applicable.
    Unspecified,
}

/// Double-bond stereochemistry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BondStereo {
    /// E (trans) — high-priority groups on opposite sides.
    E,
    /// Z (cis) — high-priority groups on the same side.
    Z,
    /// Double bond with unspecified stereo.
    Either,
    /// Not a stereo bond (no double bond or not applicable).
    None,
}

// ---------------------------------------------------------------------------
// Chiral volume
// ---------------------------------------------------------------------------

/// Compute the signed scalar triple product (chiral volume) at `center`.
///
/// `neighbor_order` should list the four substituents in the order that
/// determines the sign convention you want to test (typically CIP priority
/// order, lowest last).
///
/// * Positive return → CCW arrangement of n1→n2→n3 when viewed from n4.
/// * Negative return → CW arrangement.
/// * Zero → the four atoms are coplanar (degenerate).
///
/// Returns `0.0` if any atom lacks `x`/`y`/`z` coordinates.
pub fn chiral_volume(mol: &Atomistic, center: AtomId, neighbor_order: &[AtomId; 4]) -> f64 {
    let pos = |id: AtomId| -> Option<[f64; 3]> {
        let a = mol.get_atom(id).ok()?;
        Some([a.get_f64("x")?, a.get_f64("y")?, a.get_f64("z")?])
    };

    let c = match pos(center) {
        Some(p) => p,
        None => return 0.0,
    };
    let p: Vec<[f64; 3]> = neighbor_order.iter().filter_map(|&id| pos(id)).collect();
    if p.len() < 4 {
        return 0.0;
    }

    // Vectors from center to each neighbour
    let v: Vec<[f64; 3]> = p.iter().map(|q| sub(*q, c)).collect();

    // Scalar triple product of v1, v2, v3  (v4 is the "viewing" direction)
    // volume = v[0] · (v[1] × v[2])
    let cross = cross3(v[1], v[2]);
    dot3(v[0], cross)
}

// ---------------------------------------------------------------------------
// Chiral centre detection
// ---------------------------------------------------------------------------

/// Return the atom IDs that are potential tetrahedral stereocentres:
/// atoms with exactly 4 distinct neighbour atom IDs.
///
/// Note: this is a *topological* screen only.  Two neighbours may be
/// constitutionally identical.  CIP rank comparison is outside the scope
/// of this module.
pub fn find_chiral_centers(mol: &Atomistic) -> Vec<AtomId> {
    let mut centers = Vec::new();
    for (id, _atom) in mol.atoms() {
        let nbrs: Vec<AtomId> = mol.neighbors(id).collect();
        if nbrs.len() == 4 {
            // Check all four are distinct
            let mut unique = nbrs.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() == 4 {
                centers.push(id);
            }
        }
    }
    centers
}

// ---------------------------------------------------------------------------
// Stereo assignment from 3D coordinates
// ---------------------------------------------------------------------------

/// Infer tetrahedral chirality for every potential stereocentre from 3-D
/// coordinates.
///
/// The sign of the chiral volume is computed using the neighbours in the
/// order they are returned by `mol.neighbors()`.  This gives a
/// *geometry-based* label (not CIP-ranked), but is stable for a given
/// molecule and useful for detecting whether two conformers have the same
/// chirality.
///
/// Returns a map `AtomId → TetrahedralStereo`.  Atoms without 3-D coordinates
/// receive `Unspecified`.
pub fn assign_stereo_from_3d(mol: &Atomistic) -> HashMap<AtomId, TetrahedralStereo> {
    let mut result = HashMap::new();
    for center in find_chiral_centers(mol) {
        let nbrs: Vec<AtomId> = mol.neighbors(center).collect();
        if nbrs.len() < 4 {
            result.insert(center, TetrahedralStereo::Unspecified);
            continue;
        }
        let arr = [nbrs[0], nbrs[1], nbrs[2], nbrs[3]];
        let vol = chiral_volume(mol, center, &arr);
        let stereo = if vol > 1e-9 {
            TetrahedralStereo::CCW
        } else if vol < -1e-9 {
            TetrahedralStereo::CW
        } else {
            TetrahedralStereo::Unspecified
        };
        result.insert(center, stereo);
    }
    result
}

/// Infer E/Z stereochemistry for every double bond from 3-D coordinates.
///
/// A bond A=B is considered to have E/Z stereo if:
/// * It is a double bond (`bond_type == Double`, or no class property and
///   degree rules suggest double).
/// * Both A and B have at least one other neighbour (substituents exist).
///
/// The dihedral angle φ between the highest-atomic-number substituent on A
/// and the highest-atomic-number substituent on B determines the label:
/// * |cos φ| < 0 (φ > 90°) → Z (same side, cis).
/// * |cos φ| > 0 (φ < 90°) → E (opposite sides, trans).
///
/// Returns a map `BondId → BondStereo`.
pub fn assign_bond_stereo_from_3d(mol: &Atomistic) -> HashMap<BondId, BondStereo> {
    let mut result = HashMap::new();

    for (bid, bond) in mol.bonds() {
        // A genuine double bond. An aromatic ring bond has no E/Z even when its
        // Kekulé phase is double — the ring is planar and the phase arbitrary.
        if BondType::from_prop(bond.props.get(keys::BOND_TYPE)) != BondType::Double {
            result.insert(bid, BondStereo::None);
            continue;
        }

        let (a, b) = (bond.nodes[0], bond.nodes[1]);

        // Substituents on A (excluding B) and on B (excluding A)
        let subs_a: Vec<AtomId> = mol.neighbors(a).filter(|&x| x != b).collect();
        let subs_b: Vec<AtomId> = mol.neighbors(b).filter(|&x| x != a).collect();

        if subs_a.is_empty() || subs_b.is_empty() {
            result.insert(bid, BondStereo::None);
            continue;
        }

        // Pick the substituent with the highest atomic number as the
        // representative. This is a simplified priority rule — one CIP sphere,
        // not the full hierarchical digraph.
        //
        // The tie-break is load-bearing, not a detail: when both substituents
        // are the same element the key ties, and `max_by_key` keeps the *last*
        // maximum — so the label became a function of bond insertion order
        // rather than of the geometry. 3-methyl-2-pentene came out E or Z
        // depending only on which of the two C3 bonds was added first. Ranking
        // by (Z, lowest index) makes the answer a property of the molecule
        // again; it is still approximate where CIP would look further out, but
        // it is at least the *same* approximation every time.
        let z_of = |s: AtomId| -> u8 {
            mol.get_atom(s)
                .ok()
                .and_then(|a| {
                    a.get_str("element")
                        .and_then(molrs::Element::by_symbol)
                        .map(|e| e.z())
                })
                .unwrap_or(0)
        };
        // Ranked by (Z, then the atom's own position in the molecule). The
        // neighbour list's order is itself insertion-dependent, so breaking the
        // tie on a position *within that list* would not fix anything.
        let order_of = |s: AtomId| -> usize {
            mol.atoms()
                .position(|(id, _)| id == s)
                .unwrap_or(usize::MAX)
        };
        let pick = |atom_id: AtomId, subs: &[AtomId]| -> AtomId {
            subs.iter()
                .copied()
                .max_by_key(|&s| (z_of(s), std::cmp::Reverse(order_of(s))))
                .unwrap_or(atom_id)
        };

        let sa = pick(a, &subs_a);
        let sb = pick(b, &subs_b);

        // Get 3-D positions
        let pos = |id: AtomId| -> Option<[f64; 3]> {
            let atom = mol.get_atom(id).ok()?;
            Some([atom.get_f64("x")?, atom.get_f64("y")?, atom.get_f64("z")?])
        };

        let (pa, pb, psa, psb) = match (pos(a), pos(b), pos(sa), pos(sb)) {
            (Some(pa), Some(pb), Some(psa), Some(psb)) => (pa, pb, psa, psb),
            _ => {
                result.insert(bid, BondStereo::Either);
                continue;
            }
        };

        // Vectors from double-bond axis ends to substituents
        let va = sub(psa, pa); // A → sub_a
        let vb = sub(psb, pb); // B → sub_b
        let ab = sub(pb, pa); // A → B (bond axis)

        // Project va and vb onto the plane perpendicular to ab
        let va_perp = sub(va, scale(ab, dot3(va, ab) / dot3(ab, ab)));
        let vb_perp = sub(vb, scale(ab, dot3(vb, ab) / dot3(ab, ab)));

        let cos_angle = dot3(va_perp, vb_perp) / (vec_len(va_perp) * vec_len(vb_perp) + 1e-15);

        // cos > 0 → same side → Z; cos < 0 → opposite sides → E
        let stereo = if cos_angle > 1e-9 {
            BondStereo::Z
        } else if cos_angle < -1e-9 {
            BondStereo::E
        } else {
            BondStereo::Either
        };
        result.insert(bid, stereo);
    }

    result
}

// ---------------------------------------------------------------------------
// 3-D math helpers
// ---------------------------------------------------------------------------

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn vec_len(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::molgraph::Atom;

    fn atom_xyz(sym: &str, x: f64, y: f64, z: f64) -> Atom {
        Atom::xyz(sym, x, y, z)
    }

    fn add_double_bond(mol: &mut Atomistic, a: AtomId, b: AtomId) {
        if let Ok(bid) = mol.add_bond(a, b) {
            let _ = mol.set_bond_type(bid, BondType::Double);
        }
    }

    // --- Chiral volume tests ---

    #[test]
    fn test_chiral_volume_sign() {
        // Four atoms at known positions forming a right-handed tetrahedron.
        // center at origin, neighbours along +x, +y, +z, and -x-y-z.
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let n1 = g.add_atom(atom_xyz("H", 1.0, 0.0, 0.0));
        let n2 = g.add_atom(atom_xyz("H", 0.0, 1.0, 0.0));
        let n3 = g.add_atom(atom_xyz("H", 0.0, 0.0, 1.0));
        let n4 = g.add_atom(atom_xyz("H", -1.0, -1.0, -1.0));
        for &n in &[n1, n2, n3, n4] {
            g.add_bond(c, n).expect("add bond");
        }
        let vol = chiral_volume(&g, c, &[n1, n2, n3, n4]);
        assert!(vol > 0.0, "expected positive chiral volume, got {}", vol);
    }

    #[test]
    fn test_chiral_volume_opposite_sign() {
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let n1 = g.add_atom(atom_xyz("H", 1.0, 0.0, 0.0));
        let n2 = g.add_atom(atom_xyz("H", 0.0, 1.0, 0.0));
        let n3 = g.add_atom(atom_xyz("H", 0.0, 0.0, 1.0));
        let n4 = g.add_atom(atom_xyz("H", -1.0, -1.0, -1.0));
        for &n in &[n1, n2, n3, n4] {
            g.add_bond(c, n).expect("add bond");
        }
        // Swapping two neighbours flips the sign
        let vol_swapped = chiral_volume(&g, c, &[n2, n1, n3, n4]);
        assert!(vol_swapped < 0.0);
    }

    // --- find_chiral_centers ---

    #[test]
    fn test_no_chiral_centers_in_ethane() {
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let c2 = g.add_atom(atom_xyz("C", 1.5, 0.0, 0.0));
        g.add_bond(c1, c2).expect("add bond");
        assert!(find_chiral_centers(&g).is_empty());
    }

    #[test]
    fn test_4_neighbor_atom_detected_as_center() {
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        for i in 0..4_usize {
            let h = g.add_atom(atom_xyz("H", i as f64, 0.0, 0.0));
            g.add_bond(c, h).expect("add bond");
        }
        // 4 neighbours, all with distinct IDs → detected
        let centers = find_chiral_centers(&g);
        assert_eq!(centers.len(), 1);
        assert_eq!(centers[0], c);
    }

    // --- assign_stereo_from_3d ---

    #[test]
    fn test_assign_stereo_returns_entry_for_center() {
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let n1 = g.add_atom(atom_xyz("F", 1.0, 0.0, 0.0));
        let n2 = g.add_atom(atom_xyz("Cl", 0.0, 1.0, 0.0));
        let n3 = g.add_atom(atom_xyz("Br", 0.0, 0.0, 1.0));
        let n4 = g.add_atom(atom_xyz("H", -1.0, -1.0, -1.0));
        for &n in &[n1, n2, n3, n4] {
            g.add_bond(c, n).expect("add bond");
        }
        let stereo = assign_stereo_from_3d(&g);
        assert!(stereo.contains_key(&c));
        assert_ne!(stereo[&c], TetrahedralStereo::Unspecified);
    }

    // --- assign_bond_stereo_from_3d ---

    #[test]
    fn test_cis_2_butene_is_z() {
        // cis-2-butene: both methyl groups on same side.
        //   CH3      CH3
        //      \    /
        //       C=C
        // Place C1 at origin, C2 at (1.34, 0, 0).
        // Sub on C1 in +y direction, sub on C2 also in +y direction → Z.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let c2 = g.add_atom(atom_xyz("C", 1.34, 0.0, 0.0));
        let sub1 = g.add_atom(atom_xyz("C", -0.5, 1.0, 0.0)); // +y side
        let sub2 = g.add_atom(atom_xyz("C", 1.84, 1.0, 0.0)); // +y side
        add_double_bond(&mut g, c1, c2);
        g.add_bond(c1, sub1).expect("add bond");
        g.add_bond(c2, sub2).expect("add bond");

        let stereo = assign_bond_stereo_from_3d(&g);
        let double_bid = stereo
            .iter()
            .find(|&(_, v)| *v == BondStereo::Z || *v == BondStereo::E)
            .map(|(&k, _)| k);
        assert!(double_bid.is_some(), "no E/Z bond found");
        assert_eq!(stereo[&double_bid.unwrap()], BondStereo::Z);
    }

    #[test]
    fn test_trans_2_butene_is_e() {
        // trans: sub on C1 in +y, sub on C2 in -y direction.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let c2 = g.add_atom(atom_xyz("C", 1.34, 0.0, 0.0));
        let sub1 = g.add_atom(atom_xyz("C", -0.5, 1.0, 0.0)); // +y side
        let sub2 = g.add_atom(atom_xyz("C", 1.84, -1.0, 0.0)); // -y side
        add_double_bond(&mut g, c1, c2);
        g.add_bond(c1, sub1).expect("add bond");
        g.add_bond(c2, sub2).expect("add bond");

        let stereo = assign_bond_stereo_from_3d(&g);
        let double_bid = stereo
            .iter()
            .find(|&(_, v)| *v == BondStereo::E || *v == BondStereo::Z)
            .map(|(&k, _)| k);
        assert!(double_bid.is_some(), "no E/Z bond found");
        assert_eq!(stereo[&double_bid.unwrap()], BondStereo::E);
    }

    #[test]
    fn test_single_bond_has_no_stereo() {
        let mut g = Atomistic::new();
        let a = g.add_atom(atom_xyz("C", 0.0, 0.0, 0.0));
        let b = g.add_atom(atom_xyz("C", 1.5, 0.0, 0.0));
        g.add_bond(a, b).expect("add bond"); // default single bond
        let stereo = assign_bond_stereo_from_3d(&g);
        let bid = g.bonds().next().unwrap().0;
        assert_eq!(stereo[&bid], BondStereo::None);
    }
    /// The E/Z label must be a function of the geometry, not of bond order.
    ///
    /// 3-methyl-2-pentene: both substituents on C3 are carbon, so the
    /// atomic-number key ties. `max_by_key` keeps the last maximum, so swapping
    /// the two C3 bonds used to flip the label between E and Z on identical
    /// coordinates.
    #[test]
    fn the_ez_label_does_not_depend_on_bond_insertion_order() {
        let build = |methyl_first: bool| {
            let mut mol = Atomistic::new();
            let c1 = mol.add_atom(Atom::xyz("C", -2.5, 0.6, 0.0));
            let c2 = mol.add_atom(Atom::xyz("C", -1.2, 0.0, 0.0));
            let c3 = mol.add_atom(Atom::xyz("C", 0.0, 0.6, 0.0));
            let me = mol.add_atom(Atom::xyz("C", 0.1, 2.1, 0.0));
            let et = mol.add_atom(Atom::xyz("C", 1.3, -0.1, 0.0));
            mol.add_bond(c1, c2).unwrap();
            let d = mol.add_bond(c2, c3).unwrap();
            mol.set_bond_type(d, crate::system::bond::BondType::Double)
                .unwrap();
            if methyl_first {
                mol.add_bond(c3, me).unwrap();
                mol.add_bond(c3, et).unwrap();
            } else {
                mol.add_bond(c3, et).unwrap();
                mol.add_bond(c3, me).unwrap();
            }
            let stereo = assign_bond_stereo_from_3d(&mol);
            stereo.get(&d).copied()
        };

        assert_eq!(
            build(true),
            build(false),
            "the same molecule with the same coordinates got two different labels"
        );
    }
}
