//! Hydrogen addition for molecular graphs.
//!
//! [`add_hydrogens`] computes the number of implicit hydrogens each heavy atom
//! requires (based on its element's default valences and the sum of its current
//! bond orders) and returns a **new** [`Atomistic`] with explicit H atoms added.
//!
//! [`remove_hydrogens`] does the inverse: it returns a new [`Atomistic`] with
//! all terminal explicit hydrogen atoms removed.
//!
//! # Immutability
//! The original `MolGraph` is never mutated; a clone is returned.
//!
//! # Bond-order convention
//! Bond order is read from the bond's `"order"` property as an `f64`.
//! If the property is absent the bond is assumed to be a single bond (1.0).
//! Aromatic bonds should be stored as 1.5.
//!
//! # Formal-charge correction
//! A formal charge is folded into the element identity, not into the bond
//! demand: the valence list of `Z − formal_charge` is used. This is RDKit's
//! `getEffectiveAtomicNum` rule and gets the group-13/14 cation case right
//! (e.g. `[CH3+]` → C(Z=6) − (+1) = B(Z=5), valence 3 → 3 H, rather than the
//! naive `bond_order_sum − formal_charge` which over-counts to 5 H). For the
//! late atoms N/O/F the two formulations happen to agree, but for early atoms
//! (B, C, Si, …) they diverge, which is exactly the bug this rule fixes.

use crate::system::atomistic::{AtomId, Atomistic, BondId};
use crate::system::molgraph::Atom;
use molrs::Element;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return a new [`Atomistic`] with explicit hydrogen atoms added to every
/// heavy atom that has unfilled valence.
///
/// Hydrogen atoms already present (symbol == "H") are not modified.
///
/// When a heavy atom has `x`/`y`/`z` components, each new H is placed at a
/// standard X–H length along a tetrahedral valence-completing direction
/// (initial geometry only — force fields may refine). When the heavy lacks
/// coordinates, H is added without `x`/`y`/`z` (topology-only path).
///
/// Starts from [`Clone`] of `mol` (handles preserved on the skeleton); only
/// new H atoms and bonds are appended, so parent angles/dihedrals remain.
pub fn add_hydrogens(mol: &Atomistic) -> Atomistic {
    let mut new_mol = mol.clone();

    // Collect (atom_id, n_implicit_h) for all heavy atoms up front so that
    // we don't hold a borrow while mutating.
    let additions: Vec<(AtomId, u32)> = new_mol
        .atoms()
        .filter_map(|(id, atom)| {
            let sym = atom.get_str("element")?;
            if sym.eq_ignore_ascii_case("H") {
                return None; // skip existing hydrogens
            }
            let n = implicit_h_count(&new_mol, id)?;
            if n == 0 { None } else { Some((id, n)) }
        })
        .collect();

    for (heavy_id, n) in additions {
        let heavy = new_mol.get_atom(heavy_id).ok();
        let place = heavy.as_ref().and_then(|a| {
            Some((
                a.get_f64("x")?,
                a.get_f64("y")?,
                a.get_f64("z")?,
                a.get_str("element").unwrap_or("C"),
            ))
        });

        let positions: Vec<[f64; 3]> = if let Some((hx, hy, hz, elem)) = place {
            let p = [hx, hy, hz];
            let mut existing: Vec<[f64; 3]> = Vec::new();
            for (nb, _) in new_mol.neighbor_bonds(heavy_id) {
                if let Ok(na) = new_mol.get_atom(nb)
                    && let (Some(x), Some(y), Some(z)) =
                        (na.get_f64("x"), na.get_f64("y"), na.get_f64("z"))
                {
                    existing.push(unit([x - p[0], y - p[1], z - p[2]]));
                }
            }
            let dirs = cap_directions(&existing, n as usize);
            let len = cap_length(elem);
            dirs.into_iter()
                .map(|d| [p[0] + len * d[0], p[1] + len * d[1], p[2] + len * d[2]])
                .collect()
        } else {
            vec![[0.0, 0.0, 0.0]; n as usize] // placeholder; coords omitted below
        };

        let place_coords = place.is_some();
        for pos in positions.iter().take(n as usize) {
            let mut h = Atom::new();
            h.set("element", "H");
            h.set("mass", 1.008_f64);
            if place_coords {
                h.set("x", pos[0]);
                h.set("y", pos[1]);
                h.set("z", pos[2]);
            }
            let h_id = new_mol.add_atom(h);
            if let Ok(bid) = new_mol.add_bond(heavy_id, h_id) {
                let _ = new_mol.set_bond_prop(bid, "order", 1.0_f64);
            }
        }
    }

    new_mol
}

// ---------------------------------------------------------------------------
// Initial X–H geometry (port of molpy.core.capping; values are starting
// guesses for downstream minimization, not equilibrium force-field lengths).
// ---------------------------------------------------------------------------

/// Cap X–H bond length (Å) keyed by heavy element; default 1.0 Å.
fn cap_length(element: &str) -> f64 {
    match element {
        "C" | "c" => 1.09,
        "N" | "n" => 1.01,
        "O" | "o" => 0.96,
        "S" | "s" => 1.34,
        _ => 1.0,
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-9 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

fn orthogonal(v: [f64; 3]) -> [f64; 3] {
    let seed = if v[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    unit([
        v[1] * seed[2] - v[2] * seed[1],
        v[2] * seed[0] - v[0] * seed[2],
        v[0] * seed[1] - v[1] * seed[0],
    ])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// `k` unit directions completing ~sp3 (tetrahedral) coordination.
fn cap_directions(existing: &[[f64; 3]], k: usize) -> Vec<[f64; 3]> {
    let n = existing.len();
    let mut caps: Vec<[f64; 3]> = if n >= 3 {
        let s = [
            existing[0][0] + existing[1][0] + existing[2][0],
            existing[0][1] + existing[1][1] + existing[2][1],
            existing[0][2] + existing[1][2] + existing[2][2],
        ];
        vec![unit([-s[0], -s[1], -s[2]])]
    } else if n == 2 {
        let u1 = existing[0];
        let u2 = existing[1];
        let bisector = unit([-(u1[0] + u2[0]), -(u1[1] + u2[1]), -(u1[2] + u2[2])]);
        let cr = cross(u1, u2);
        let normal = if norm(cr) < 1e-6 {
            orthogonal(u1)
        } else {
            unit(cr)
        };
        let half = 54.75_f64.to_radians();
        let (c, s) = (half.cos(), half.sin());
        vec![
            unit([
                c * bisector[0] + s * normal[0],
                c * bisector[1] + s * normal[1],
                c * bisector[2] + s * normal[2],
            ]),
            unit([
                c * bisector[0] - s * normal[0],
                c * bisector[1] - s * normal[1],
                c * bisector[2] - s * normal[2],
            ]),
        ]
    } else if n == 1 {
        let u = existing[0];
        let e1 = orthogonal(u);
        let e2 = unit(cross(u, e1));
        let theta = 109.47_f64.to_radians();
        let (ct, st) = (theta.cos(), theta.sin());
        [0.0_f64, 120.0, 240.0]
            .into_iter()
            .map(|phi_deg| {
                let phi = phi_deg.to_radians();
                let (cp, sp) = (phi.cos(), phi.sin());
                unit([
                    ct * u[0] + st * (cp * e1[0] + sp * e2[0]),
                    ct * u[1] + st * (cp * e1[1] + sp * e2[1]),
                    ct * u[2] + st * (cp * e1[2] + sp * e2[2]),
                ])
            })
            .collect()
    } else {
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
        .into_iter()
        .map(unit)
        .collect()
    };
    caps.truncate(k);
    // If fewer directions than k (shouldn't for tetrahedral cases), pad with +x-ish
    while caps.len() < k {
        caps.push([1.0, 0.0, 0.0]);
    }
    caps
}

/// Return a new [`Atomistic`] with all terminal explicit hydrogen atoms removed.
///
/// Only hydrogen atoms with exactly one neighbor (degree == 1) are removed,
/// which is the standard cheminformatics convention for "non-bridging" H.
/// Incident bonds, angles, and dihedrals are cascade-deleted by
/// [`Atomistic::remove_atom`].
///
/// The original `MolGraph` is never mutated; a clone is returned.
pub fn remove_hydrogens(mol: &Atomistic) -> Atomistic {
    let mut new_mol = mol.clone();
    let h_ids: Vec<AtomId> = new_mol
        .atoms()
        .filter_map(|(id, atom)| {
            let sym = atom.get_str("element")?;
            if !sym.eq_ignore_ascii_case("H") {
                return None;
            }
            if new_mol.neighbors(id).count() == 1 {
                Some(id)
            } else {
                None
            }
        })
        .collect();
    for h_id in h_ids {
        let _ = new_mol.remove_atom(h_id);
    }
    new_mol
}

// ---------------------------------------------------------------------------
// Implicit-H calculation
// ---------------------------------------------------------------------------

/// Compute the number of hydrogens to add to `atom_id`.
///
/// Returns `None` if the atom has no recognisable element symbol or if its
/// element has no defined default valences (e.g. noble gases).
pub fn implicit_h_count(mol: &Atomistic, atom_id: AtomId) -> Option<u32> {
    let atom = mol.get_atom(atom_id).ok()?;
    let sym = atom.get_str("element")?;
    let element = Element::by_symbol(sym)?;

    // RDKit charged-atom valence rule (`getEffectiveAtomicNum` +
    // `calculateImplicitValence` in `Code/GraphMol/Atom.cpp`):
    //
    //   1. Z_eff = Z − formal_charge  (cation → element one place earlier;
    //      anion → one place later). The valence list is taken from Z_eff,
    //      NOT from the bare element with a charge-adjusted demand.
    //   2. demand = sum of incident bond orders (no charge term here).
    //   3. target = smallest Z_eff valence ≥ demand.
    //   4. implicit_h = target − demand.
    //
    // This is what makes early atoms (B, C, Si, …) and late atoms (N, O, F)
    // behave asymmetrically under charge:
    //   [CH3+]  Z 6−(+1)=5 (B), valences [3], demand 0 → 3 H
    //   [CH3-]  Z 6−(−1)=7 (N), valences [3,5], demand 0 → 3 H
    //   [NH4+]  Z 7−(+1)=6 (C), valences [4], demand 0 → 4 H
    //   [BH4-]  Z 5−(−1)=6 (C), valences [4], demand 0 → 4 H
    //   [OH-]   Z 8−(−1)=9 (F), valences [1], demand 0 → 1 H
    //   [NH2-]  Z 7−(−1)=8 (O), valences [2], demand 0 → 2 H
    // `formal_charge` is stored as an i32-typed column, so read it through the
    // coercing `as_f64` (matching `bond_order_sum`'s order read); the strict
    // `get_f64` only matches `PropValue::F64` and would silently miss the Int
    // variant, treating every charged atom as neutral (e.g. protonating [N-]).
    let formal_charge = atom
        .get("formal_charge")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .round() as i32;

    // Fold the charge into the element identity, then read that element's
    // valence list. An out-of-range shift (or an element with no valence
    // model) means we add no hydrogens.
    let effective = element.effective_atomic_number(formal_charge)?;
    let valences = effective.default_valences();
    if valences.is_empty() {
        return None; // noble gas / effective element with no valence model
    }

    // Sum of bond orders connected to this atom (the explicit valence).
    let demand: f64 = bond_order_sum(mol, atom_id);

    // Select the smallest allowed valence ≥ the (un-charge-adjusted) demand.
    let target = valences
        .iter()
        .copied()
        .find(|&v| v as f64 >= demand - 1e-6);

    let target = target?; // if demand exceeds all valences, add nothing
    let n = target as f64 - demand;
    if n <= 0.5 {
        Some(0)
    } else {
        Some(n.round() as u32)
    }
}

/// Sum of bond orders for all bonds incident to `atom_id`.
fn bond_order_sum(mol: &Atomistic, atom_id: AtomId) -> f64 {
    // Build a local bond-id list from bond iteration (adjacency is private).
    bond_ids_for(mol, atom_id)
        .into_iter()
        .filter_map(|bid| mol.get_bond(bid).ok())
        .map(|bond| {
            // Accept order stored as either F64 (2.0) or Int (2); absent → single.
            bond.props
                .get("order")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0)
        })
        .sum()
}

/// Collect all `BondId`s incident to `atom_id` by scanning `mol.bonds()`.
///
/// O(E) — acceptable for the sizes of typical drug molecules.  If a
/// `neighbors_with_bonds` API is added to `MolGraph` in the future this can
/// be replaced with an O(degree) call.
fn bond_ids_for(mol: &Atomistic, atom_id: AtomId) -> Vec<BondId> {
    mol.bonds()
        .filter_map(|(bid, bond)| {
            if bond.nodes[0] == atom_id || bond.nodes[1] == atom_id {
                Some(bid)
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::molgraph::PropValue;

    fn atom(sym: &str) -> Atom {
        let mut a = Atom::new();
        a.set("element", sym);
        a
    }

    fn bond_with_order(mol: &mut Atomistic, a: AtomId, b: AtomId, order: f64) {
        if let Ok(bid) = mol.add_bond(a, b) {
            let _ = mol.set_bond_prop(bid, "order", PropValue::F64(order));
        }
    }

    #[test]
    fn test_methane_skeleton() {
        // Isolated C — should get 4 H.
        let mut g = Atomistic::new();
        let c = g.add_atom(atom("C"));
        let result = add_hydrogens(&g);
        // original unchanged
        assert_eq!(g.n_atoms(), 1);
        // result has C + 4H
        assert_eq!(result.n_atoms(), 5);
        assert_eq!(result.n_bonds(), 4);
        let n_h = result
            .atoms()
            .filter(|(_, a)| a.get_str("element") == Some("H"))
            .count();
        assert_eq!(n_h, 4);
        let _ = c; // suppress unused warning
    }

    #[test]
    fn test_ethane_c_c() {
        // C-C single bond: each C needs 3 H.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 1.0);
        let result = add_hydrogens(&g);
        assert_eq!(result.n_atoms(), 8); // 2C + 6H
    }

    #[test]
    fn test_ethylene_c_double_c() {
        // C=C double bond: each C needs 2 H.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        let result = add_hydrogens(&g);
        assert_eq!(result.n_atoms(), 6); // 2C + 4H
    }

    #[test]
    fn test_benzene_aromatic() {
        // 6-membered ring with bond order 1.5: each C should get 1 H.
        let mut g = Atomistic::new();
        let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
        }
        let result = add_hydrogens(&g);
        assert_eq!(result.n_atoms(), 12); // 6C + 6H
    }

    #[test]
    fn test_benzene_kekule() {
        // Kekule benzene: alternating single/double bonds.
        // Each C has bond_order_sum = 1+2 = 3, needs 1 H. Total = 6 H.
        let mut g = Atomistic::new();
        let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
        let orders = [2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], orders[i]);
        }
        let result = add_hydrogens(&g);
        let n_h = result
            .atoms()
            .filter(|(_, a)| a.get_str("element") == Some("H"))
            .count();
        assert_eq!(n_h, 6, "Kekule benzene should get 6 H, got {}", n_h);
        assert_eq!(result.n_atoms(), 12); // 6C + 6H
    }

    #[test]
    fn test_ethylene_round_trip_frame() {
        // C=C → to_frame → from_frame → add_hydrogens should give 4H not 6H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        let frame = g.to_frame();
        let g2 = Atomistic::from_frame(&frame).unwrap();
        let result = add_hydrogens(&g2);
        assert_eq!(result.n_atoms(), 6, "C=C round-trip should give 2C + 4H");
    }

    #[test]
    fn test_acetylene_round_trip_frame() {
        // C#C → to_frame → from_frame → add_hydrogens should give 2H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 3.0);
        let frame = g.to_frame();
        let g2 = Atomistic::from_frame(&frame).unwrap();
        let result = add_hydrogens(&g2);
        assert_eq!(result.n_atoms(), 4, "C#C round-trip should give 2C + 2H");
    }

    #[test]
    fn test_water() {
        // Isolated O → 2 H
        let mut g = Atomistic::new();
        let _o = g.add_atom(atom("O"));
        let result = add_hydrogens(&g);
        assert_eq!(result.n_atoms(), 3);
    }

    #[test]
    fn test_ammonia_like() {
        // N with 1 bond → 2 H  (valence 3)
        let mut g = Atomistic::new();
        let n = g.add_atom(atom("N"));
        let c = g.add_atom(atom("C"));
        bond_with_order(&mut g, n, c, 1.0);
        let result = add_hydrogens(&g);
        // N gets 2H, C gets 3H, total = 2C + 2H(on N) + 3H(on C) = 2+5 = 7
        assert_eq!(result.n_atoms(), 7);
    }

    #[test]
    fn test_nh4_plus() {
        // NH4+: formal_charge=1 on N → needs 4 H
        let mut g = Atomistic::new();
        let mut n_atom = Atom::new();
        n_atom.set("element", "N");
        n_atom.set("formal_charge", 1.0_f64);
        let n = g.add_atom(n_atom);
        let count = implicit_h_count(&g, n).unwrap();
        assert_eq!(count, 4);
    }

    /// Build a single charged heavy atom (no heavy neighbours) and check its
    /// implicit-H count against RDKit's `GetTotalNumHs()`.
    fn charged_atom_h(sym: &str, fc: f64) -> u32 {
        let mut g = Atomistic::new();
        let mut a = Atom::new();
        a.set("element", sym);
        a.set("formal_charge", fc);
        let id = g.add_atom(a);
        implicit_h_count(&g, id).unwrap_or(0)
    }

    #[test]
    fn test_rdkit_charged_valence_parity() {
        // Expected hydrogen counts baked in from RDKit 2026.03.2:
        //   for smi in [...]: Chem.MolFromSmiles(smi); atom.GetTotalNumHs()
        //
        // Charged single-heavy-atom species (the cases the old
        // bond_order_sum - formal_charge rule got wrong for group-13/14):
        assert_eq!(charged_atom_h("C", 1.0), 3, "[CH3+] -> 3 H");
        assert_eq!(charged_atom_h("C", -1.0), 3, "[CH3-] -> 3 H");
        assert_eq!(charged_atom_h("B", -1.0), 4, "[BH4-] -> 4 H");
        assert_eq!(charged_atom_h("N", 1.0), 4, "[NH4+] -> 4 H");
        assert_eq!(charged_atom_h("O", -1.0), 1, "[OH-] -> 1 H");
        assert_eq!(charged_atom_h("N", -1.0), 2, "[NH2-] -> 2 H");

        // Neutral references (unchanged by the fix):
        assert_eq!(charged_atom_h("C", 0.0), 4, "methane C -> 4 H");
        assert_eq!(charged_atom_h("O", 0.0), 2, "water O -> 2 H");
        assert_eq!(charged_atom_h("N", 0.0), 3, "ammonia N -> 3 H");
    }

    /// Like `charged_atom_h` but stores `formal_charge` as the canonical
    /// **integer** column (`PropValue::Int`) — what the parsers and the i32-typed
    /// graph schema actually emit. Guards the regression where `implicit_h_count`
    /// read the charge via the strict `get_f64` (F64-only) and silently treated
    /// every charged atom as neutral — e.g. protonating the sulfonimide [N-] in
    /// TFSI/ANI and breaking antechamber's charge balance.
    fn charged_atom_h_int(sym: &str, fc: i32) -> u32 {
        let mut g = Atomistic::new();
        let mut a = Atom::new();
        a.set("element", sym);
        a.set("formal_charge", fc); // i32 == type alias `I` → PropValue::Int
        let id = g.add_atom(a);
        implicit_h_count(&g, id).unwrap_or(0)
    }

    #[test]
    fn test_int_formal_charge_parity() {
        // Same expectations as `test_rdkit_charged_valence_parity`, but the
        // charge is an Int prop (the real on-graph representation), not f64.
        assert_eq!(charged_atom_h_int("N", -1), 2, "[NH2-] (int fc) -> 2 H");
        assert_eq!(charged_atom_h_int("N", 1), 4, "[NH4+] (int fc) -> 4 H");
        assert_eq!(charged_atom_h_int("O", -1), 1, "[OH-] (int fc) -> 1 H");
        assert_eq!(charged_atom_h_int("C", 1), 3, "[CH3+] (int fc) -> 3 H");
        assert_eq!(charged_atom_h_int("N", 0), 3, "neutral N -> 3 H");
    }

    #[test]
    fn test_sulfonimide_anion_not_protonated() {
        // The exact TFSI/ANI failure: a deprotonated sulfonimide N (two single
        // bonds, int formal_charge -1) must add NO hydrogen. Before the fix the
        // Int charge was missed → N read as neutral (valence 3, demand 2) → 1 H.
        let mut g = Atomistic::new();
        let mut n_atom = Atom::new();
        n_atom.set("element", "N");
        n_atom.set("formal_charge", -1_i32);
        let n = g.add_atom(n_atom);
        let s1 = g.add_atom(atom("S"));
        let s2 = g.add_atom(atom("S"));
        bond_with_order(&mut g, n, s1, 1.0);
        bond_with_order(&mut g, n, s2, 1.0);
        assert_eq!(h_at(&g, n), 0, "sulfonimide [N-] with two bonds -> 0 H");
    }

    /// Helper: implicit-H on `atom_id` of a built graph.
    fn h_at(g: &Atomistic, id: AtomId) -> u32 {
        implicit_h_count(g, id).unwrap_or(0)
    }

    #[test]
    fn test_rdkit_multi_atom_parity() {
        // ethane CC: each C has bos 1 -> 3 H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 1.0);
        assert_eq!(h_at(&g, c1), 3, "ethane C -> 3 H");
        assert_eq!(h_at(&g, c2), 3, "ethane C -> 3 H");

        // ethylene C=C: each C has bos 2 -> 2 H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        assert_eq!(h_at(&g, c1), 2, "ethylene C -> 2 H");

        // benzene (aromatic, bos 1.5+1.5=3): each C -> 1 H
        let mut g = Atomistic::new();
        let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
        }
        assert_eq!(h_at(&g, ids[0]), 1, "benzene C -> 1 H");

        // acetate CC(=O)[O-]: methyl C -> 3, carbonyl C -> 0,
        // carbonyl O (=O) -> 0, [O-] (single bond, fc -1) -> 0
        let mut g = Atomistic::new();
        let c_me = g.add_atom(atom("C"));
        let c_carb = g.add_atom(atom("C"));
        let o_dbl = g.add_atom(atom("O"));
        let mut o_minus = Atom::new();
        o_minus.set("element", "O");
        o_minus.set("formal_charge", -1.0_f64);
        let o_minus = g.add_atom(o_minus);
        bond_with_order(&mut g, c_me, c_carb, 1.0);
        bond_with_order(&mut g, c_carb, o_dbl, 2.0);
        bond_with_order(&mut g, c_carb, o_minus, 1.0);
        assert_eq!(h_at(&g, c_me), 3, "acetate methyl C -> 3 H");
        assert_eq!(h_at(&g, c_carb), 0, "acetate carbonyl C -> 0 H");
        assert_eq!(h_at(&g, o_dbl), 0, "acetate =O -> 0 H");
        assert_eq!(h_at(&g, o_minus), 0, "acetate [O-] -> 0 H");
    }

    #[test]
    fn test_no_double_h_on_existing_hydrogen() {
        // Existing H atoms should not get more H added.
        let mut g = Atomistic::new();
        let c = g.add_atom(atom("C"));
        let h = g.add_atom(atom("H"));
        bond_with_order(&mut g, c, h, 1.0);
        let result = add_hydrogens(&g);
        // C had 1 bond, needs 3 more H; H should remain unchanged
        let n_h = result
            .atoms()
            .filter(|(_, a)| a.get_str("element") == Some("H"))
            .count();
        assert_eq!(n_h, 4); // 1 original + 3 new
    }

    // ── remove_hydrogens tests ──────────────────────────────────────────────

    #[test]
    fn test_remove_hydrogens_methane() {
        // C + 4H → remove → 1 atom (C only), 0 bonds
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let with_h = add_hydrogens(&g);
        assert_eq!(with_h.n_atoms(), 5);
        let stripped = remove_hydrogens(&with_h);
        assert_eq!(stripped.n_atoms(), 1);
        assert_eq!(stripped.n_bonds(), 0);
    }

    #[test]
    fn test_remove_hydrogens_ethane() {
        // 2C + 6H → remove → 2 atoms, 1 bond (C-C preserved)
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 1.0);
        let with_h = add_hydrogens(&g);
        assert_eq!(with_h.n_atoms(), 8);
        let stripped = remove_hydrogens(&with_h);
        assert_eq!(stripped.n_atoms(), 2);
        assert_eq!(stripped.n_bonds(), 1);
    }

    #[test]
    fn test_remove_hydrogens_immutable() {
        // Original graph must remain unchanged after remove_hydrogens
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let with_h = add_hydrogens(&g);
        let before = with_h.n_atoms();
        let _stripped = remove_hydrogens(&with_h);
        assert_eq!(with_h.n_atoms(), before);
    }

    #[test]
    fn test_remove_hydrogens_no_h_present() {
        // C=C without any H → unchanged
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom("C"));
        let c2 = g.add_atom(atom("C"));
        bond_with_order(&mut g, c1, c2, 2.0);
        let stripped = remove_hydrogens(&g);
        assert_eq!(stripped.n_atoms(), 2);
        assert_eq!(stripped.n_bonds(), 1);
    }

    #[test]
    fn test_remove_hydrogens_cascades_angles() {
        // Build C with H and an angle involving H, then remove H → angle gone
        let mut g = Atomistic::new();
        let c = g.add_atom(atom("C"));
        let h1 = g.add_atom(atom("H"));
        let h2 = g.add_atom(atom("H"));
        bond_with_order(&mut g, c, h1, 1.0);
        bond_with_order(&mut g, c, h2, 1.0);
        g.add_angle(h1, c, h2).expect("add angle");
        assert_eq!(g.n_angles(), 1);
        let stripped = remove_hydrogens(&g);
        assert_eq!(stripped.n_atoms(), 1);
        assert_eq!(stripped.n_bonds(), 0);
        assert_eq!(stripped.n_angles(), 0);
    }

    #[test]
    fn test_add_hydrogens_places_coords_when_heavy_has_xyz() {
        let mut g = Atomistic::new();
        let c = g.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let result = add_hydrogens(&g);
        assert_eq!(result.n_atoms(), 5);
        let mut n_h = 0;
        for (id, a) in result.atoms() {
            if a.get_str("element") != Some("H") {
                continue;
            }
            n_h += 1;
            let x = a.get_f64("x").expect("H must have x");
            let y = a.get_f64("y").expect("H must have y");
            let z = a.get_f64("z").expect("H must have z");
            let dist = (x * x + y * y + z * z).sqrt();
            assert!(
                (dist - 1.09).abs() < 0.02,
                "C–H distance {dist} for H {id:?}"
            );
        }
        assert_eq!(n_h, 4);
        let _ = c;
    }

    #[test]
    fn test_add_hydrogens_no_xyz_when_heavy_lacks_coords() {
        let mut g = Atomistic::new();
        g.add_atom(atom("C"));
        let result = add_hydrogens(&g);
        for (_, a) in result.atoms() {
            if a.get_str("element") == Some("H") {
                assert!(a.get_f64("x").is_none());
            }
        }
    }

    #[test]
    fn test_add_hydrogens_preserves_parent_angles() {
        let mut g = Atomistic::new();
        let c1 = g.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let c2 = g.add_atom_xyz("C", 1.5, 0.0, 0.0);
        let c3 = g.add_atom_xyz("C", 3.0, 0.0, 0.0);
        bond_with_order(&mut g, c1, c2, 1.0);
        bond_with_order(&mut g, c2, c3, 1.0);
        g.generate_topology(true, false, false).unwrap();
        let n_ang = g.n_angles();
        assert!(n_ang > 0);
        let result = add_hydrogens(&g);
        assert!(result.n_angles() >= n_ang);
    }
}
