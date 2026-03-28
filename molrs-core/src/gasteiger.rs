//! Gasteiger-Marsili (1980) iterative electronegativity equalization.
//!
//! Direct Rust port of RDKit's `GasteigerCharges.cpp`.
//!
//! # Algorithm overview
//! Each atom type has empirical parameters (a, b, c) so that its
//! electronegativity as a function of partial charge is:
//!
//! χ(q) = a + b·q + c·q²
//!
//! Charges are iteratively redistributed between bonded atom pairs until
//! convergence (fixed number of iterations, damping by 0.5 each step).
//! Implicit hydrogens are tracked via a per-heavy-atom total H-charge.

use std::collections::HashMap;

use super::molgraph::{AtomId, MolGraph};
use crate::element::Element;
use crate::hydrogens::implicit_h_count;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Hardness parameter for implicit hydrogen (from RDKit GasteigerCharges.cpp).
const IONXH: f64 = 20.02;

/// Initial damping factor.
const DAMP: f64 = 0.5;

/// Per-iteration damping scale.
const DAMP_SCALE: f64 = 0.5;

/// Near-zero threshold.
const EPS: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Parameter table
// ---------------------------------------------------------------------------

/// Return Gasteiger parameters [a, b, c] for a given element symbol and
/// hybridization mode string.
///
/// Directly transcribed from `GasteigerParams.cpp`
/// (`defaultParamData` + `additionalParamData`).
/// Unknown combinations fall back to [0.0, 0.0, 0.0].
fn gasteiger_params(elem: &str, mode: &str) -> [f64; 3] {
    match (elem, mode) {
        // Hydrogen
        ("H", "*") => [7.17, 6.24, -0.56],
        // Carbon
        ("C", "sp3") => [7.98, 9.18, 1.88],
        ("C", "sp2") => [8.79, 9.18, 1.88],
        ("C", "sp") => [10.39, 9.18, 1.88],
        // Nitrogen
        ("N", "sp3") => [11.54, 10.82, 1.36],
        ("N", "sp2") => [12.87, 11.15, 0.85],
        ("N", "sp") => [15.68, 11.70, -0.27],
        // Oxygen
        ("O", "sp3") => [14.18, 12.92, 1.39],
        ("O", "sp2") => [17.07, 13.79, 0.47],
        // Halogens (mode-independent)
        ("F", "*") => [14.66, 13.85, 2.31],
        ("Cl", "*") => [11.00, 9.69, 1.35],
        ("Br", "*") => [10.08, 8.47, 1.16],
        ("I", "*") => [9.90, 7.96, 0.96],
        // Phosphorus
        ("P", "sp3") => [8.90, 8.24, 0.65],
        // Sulfur
        ("S", "sp3") => [10.14, 9.13, 1.38],
        ("S", "so") => [12.00, 10.81, 1.20],
        ("S", "so2") => [14.00, 12.00, 1.20],
        // Additional params
        ("Li", "*") => [3.00, 1.79, -0.15],
        ("Na", "*") => [2.84, 1.77, -0.08],
        ("K", "*") => [2.42, 1.64, -0.03],
        ("Mg", "sp3") => [5.54, 5.36, 0.61],
        ("Al", "sp3") => [5.47, 6.09, 1.19],
        ("Si", "sp3") => [6.99, 7.99, 1.87],
        ("B", "sp2") => [5.42, 6.68, 1.80],
        // Unknown / fallback
        _ => [0.0, 0.0, 0.0],
    }
}

// ---------------------------------------------------------------------------
// Hybridization mode
// ---------------------------------------------------------------------------

/// Determine the Gasteiger hybridization mode string for an atom.
///
/// Corresponds to `switch (hybridization)` in `GasteigerCharges.cpp`, but
/// inferred from bond orders rather than a stored hybridization flag.
fn hybridization_mode(mol: &MolGraph, id: AtomId) -> &'static str {
    let atom = match mol.get_atom(id) {
        Ok(a) => a,
        Err(_) => return "*",
    };
    let sym = atom.get_str("element").unwrap_or("");

    // Hydrogen is always "*"
    if sym.eq_ignore_ascii_case("H") {
        return "*";
    }

    // Mode-independent elements (halogens, alkali metals)
    if matches!(sym, "F" | "Cl" | "Br" | "I" | "Li" | "Na" | "K") {
        return "*";
    }

    let bond_orders: Vec<f64> = mol.neighbor_bonds(id).map(|(_, o)| o).collect();
    let max_order = bond_orders.iter().cloned().fold(0.0f64, f64::max);

    // Triple bond → sp
    if max_order >= 3.0 - EPS {
        return "sp";
    }

    // Double or aromatic bond (order ≥ 1.5) → sp2
    if max_order >= 1.5 - EPS {
        return "sp2";
    }

    // Sulfur: count oxygen neighbors to distinguish SO / SO2
    if sym == "S" {
        let n_oxygen = mol
            .neighbors(id)
            .filter(|&nbr| {
                mol.get_atom(nbr)
                    .ok()
                    .and_then(|a| a.get_str("element"))
                    .map(|s| s == "O")
                    .unwrap_or(false)
            })
            .count();
        return match n_oxygen {
            n if n >= 2 => "so2",
            1 => "so",
            _ => "sp3",
        };
    }

    "sp3"
}

// ---------------------------------------------------------------------------
// Bond conjugation
// ---------------------------------------------------------------------------

/// Return `true` if the bond between `a` and `b` is conjugated.
///
/// Mirrors RDKit's `setConjugation()` logic:
/// - Double / triple / aromatic bonds are intrinsically conjugated.
/// - Single bonds are conjugated when *both* endpoints are sp or sp2.
fn is_bond_conjugated(mol: &MolGraph, a: AtomId, b: AtomId) -> bool {
    let order = mol
        .neighbor_bonds(a)
        .find(|&(nbr, _)| nbr == b)
        .map(|(_, o)| o)
        .unwrap_or(1.0);

    if order >= 1.5 - EPS {
        return true; // double / triple / aromatic
    }

    let mode_a = hybridization_mode(mol, a);
    let mode_b = hybridization_mode(mol, b);
    matches!(mode_a, "sp" | "sp2") && matches!(mode_b, "sp" | "sp2")
}

// ---------------------------------------------------------------------------
// Charge redistribution in conjugated systems
// ---------------------------------------------------------------------------

/// Distribute formal charges across conjugated systems before iterating.
///
/// Corresponds to `redistributeCharges` in `GasteigerCharges.cpp`.
///
/// For each atom with a non-zero formal charge and `charges[i] ≈ 0`, the
/// function performs a 2-hop search through conjugated bonds to find atoms of
/// the same element, then divides the formal charge uniformly among them.
fn split_charge_conjugated(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    charges: &mut [f64],
    atom_idx: &HashMap<AtomId, usize>,
) {
    let n = atom_ids.len();
    for i in 0..n {
        let id_i = atom_ids[i];
        let atom_i = match mol.get_atom(id_i) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let fc = atom_i.get_f64("formal_charge").unwrap_or(0.0);
        if fc.abs() < EPS {
            continue; // no formal charge
        }
        if charges[i].abs() > EPS {
            continue; // already processed
        }

        let sym_i = atom_i.get_str("element").unwrap_or("");
        let elem_i = Element::by_symbol(sym_i);

        // 2-hop conjugated search for atoms of the same element
        let mut markers: Vec<usize> = vec![i];
        for (j_id, _) in mol.neighbor_bonds(id_i) {
            if !is_bond_conjugated(mol, id_i, j_id) {
                continue;
            }
            for (k_id, _) in mol.neighbor_bonds(j_id) {
                if k_id == id_i {
                    continue;
                }
                if !is_bond_conjugated(mol, j_id, k_id) {
                    continue;
                }
                let sym_k = mol
                    .get_atom(k_id)
                    .ok()
                    .and_then(|a| a.get_str("element"))
                    .unwrap_or("");
                let elem_k = Element::by_symbol(sym_k);
                if elem_i.is_some()
                    && elem_i == elem_k
                    && let Some(&ki) = atom_idx.get(&k_id)
                    && !markers.contains(&ki)
                {
                    markers.push(ki);
                }
            }
        }

        let chg = fc / markers.len() as f64;
        for idx in &markers {
            charges[*idx] = chg;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computed Gasteiger partial charges for one heavy atom.
#[derive(Debug, Clone, PartialEq)]
pub struct GasteigerCharges {
    /// Partial charge on the heavy atom itself.
    pub charge: f64,
    /// Total partial charge distributed across this atom's implicit hydrogens.
    pub h_charge: f64,
}

/// Compute Gasteiger-Marsili (1980) partial charges for every heavy atom in
/// `mol`.
///
/// # Parameters
/// - `mol`:  Molecular graph. Atoms need a `"element"` prop. Bond orders are
///   read from the `"order"` prop (default 1.0, aromatic = 1.5).
///   Formal charges go in `"formal_charge"` (default 0.0).
/// - `n_iter`: Equalization iterations. RDKit default is 12.
///
/// # Returns
/// One `(AtomId, GasteigerCharges)` entry per heavy atom (explicit H skipped).
pub fn compute_gasteiger_charges(mol: &MolGraph, n_iter: usize) -> Vec<(AtomId, GasteigerCharges)> {
    // Collect heavy-atom IDs (skip explicit H)
    let atom_ids: Vec<AtomId> = mol
        .atoms()
        .filter(|(_, a)| {
            a.get_str("element")
                .map(|s| !s.eq_ignore_ascii_case("H"))
                .unwrap_or(true)
        })
        .map(|(id, _)| id)
        .collect();

    let n = atom_ids.len();
    if n == 0 {
        return Vec::new();
    }

    // O(1) AtomId → index lookup
    let atom_idx: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Per-atom Gasteiger parameters
    let atm_ps: Vec<[f64; 3]> = atom_ids
        .iter()
        .map(|&id| {
            let sym = mol
                .get_atom(id)
                .ok()
                .and_then(|a| a.get_str("element"))
                .unwrap_or("");
            let mode = hybridization_mode(mol, id);
            gasteiger_params(sym, mode)
        })
        .collect();

    // ion_x[i] = p[0] + p[1] + p[2]  (hardness denominator)
    let ion_x: Vec<f64> = atm_ps.iter().map(|p| p[0] + p[1] + p[2]).collect();

    // H parameters (shared across all implicit H)
    let h_params = gasteiger_params("H", "*");

    // Start charges at zero; split_charge will set them from formal_charge
    let mut charges: Vec<f64> = vec![0.0; n];
    let mut h_chrg: Vec<f64> = vec![0.0; n];

    // Redistribute formal charges across conjugated systems
    split_charge_conjugated(mol, &atom_ids, &mut charges, &atom_idx);

    // Cache implicit H counts
    let nh: Vec<usize> = atom_ids
        .iter()
        .map(|&id| implicit_h_count(mol, id).unwrap_or(0) as usize)
        .collect();

    // Iterative equalization
    let mut damp = DAMP;
    for _itx in 0..n_iter {
        // Electronegativity at current charge
        let energ: Vec<f64> = (0..n)
            .map(|i| {
                let q = charges[i];
                atm_ps[i][0] + q * (atm_ps[i][1] + atm_ps[i][2] * q)
            })
            .collect();

        // Accumulate charge deltas (Jacobi: energies fixed for whole step)
        let mut delta: Vec<f64> = vec![0.0; n];
        let mut dh: Vec<f64> = vec![0.0; n];

        for i in 0..n {
            let id_i = atom_ids[i];
            let mut dq = 0.0;

            // Heavy-atom neighbors
            for (j_id, _) in mol.neighbor_bonds(id_i) {
                if let Some(&j) = atom_idx.get(&j_id) {
                    let dx = energ[j] - energ[i];
                    let sgn = if dx < 0.0 { 0.0 } else { 1.0 };
                    let denom = sgn * (ion_x[i] - ion_x[j]) + ion_x[j];
                    if denom.abs() > EPS {
                        dq += dx / denom;
                    }
                }
            }

            // Implicit H contribution
            let ni_hs = nh[i];
            if ni_hs > 0 {
                let q_hs = h_chrg[i] / ni_hs as f64;
                let enr_h = h_params[0] + q_hs * (h_params[1] + h_params[2] * q_hs);
                let dx = enr_h - energ[i];
                let sgn = if dx < 0.0 { 0.0 } else { 1.0 };
                let denom = sgn * (ion_x[i] - IONXH) + IONXH;
                if denom.abs() > EPS {
                    let dq_h = dx / denom;
                    dq += ni_hs as f64 * dq_h;
                    dh[i] = -(ni_hs as f64) * dq_h * damp;
                }
            }

            delta[i] = damp * dq;
        }

        // Apply updates
        for i in 0..n {
            charges[i] += delta[i];
            h_chrg[i] += dh[i];
        }

        damp *= DAMP_SCALE;
    }

    atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            (
                id,
                GasteigerCharges {
                    charge: charges[i],
                    h_charge: h_chrg[i],
                },
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molgraph::{Atom, MolGraph, PropValue};

    fn atom(sym: &str) -> Atom {
        let mut a = Atom::new();
        a.set("element", sym);
        a
    }

    fn atom_fc(sym: &str, fc: f64) -> Atom {
        let mut a = atom(sym);
        a.set("formal_charge", fc);
        a
    }

    fn bond_order(mol: &mut MolGraph, a: AtomId, b: AtomId, order: f64) {
        if let Ok(bid) = mol.add_bond(a, b)
            && let Ok(bond) = mol.get_bond_mut(bid)
        {
            bond.props
                .insert("order".to_string(), PropValue::F64(order));
        }
    }

    /// Look up the charge for a given atom ID in the result vec.
    fn charge_of(results: &[(AtomId, GasteigerCharges)], id: AtomId) -> GasteigerCharges {
        results
            .iter()
            .find(|(aid, _)| *aid == id)
            .map(|(_, c)| c.clone())
            .expect("atom not found in results")
    }

    // -----------------------------------------------------------------------
    // Water (H2O) — isolated O with 2 implicit H
    // -----------------------------------------------------------------------
    #[test]
    fn test_water() {
        let mut mol = MolGraph::new();
        let o = mol.add_atom(atom("O"));
        // No explicit H — they are implicit.

        let results = compute_gasteiger_charges(&mol, 12);
        assert_eq!(results.len(), 1);
        let gc = charge_of(&results, o);

        // O should be negative (pulls charge from H)
        assert!(
            gc.charge < -0.35,
            "O charge = {} (expected ≈ -0.401)",
            gc.charge
        );
        assert!(
            gc.charge > -0.45,
            "O charge = {} (expected ≈ -0.401)",
            gc.charge
        );

        // Total H charge should be positive (conservation: sum ≈ 0)
        assert!(
            gc.h_charge > 0.35,
            "H total charge = {} (expected ≈ +0.401)",
            gc.h_charge
        );
        // Overall charge conservation: heavy + H ≈ 0 (no formal charge)
        let total = gc.charge + gc.h_charge;
        assert!(
            total.abs() < 0.01,
            "charge not conserved: {} + {} = {}",
            gc.charge,
            gc.h_charge,
            total
        );
    }

    // -----------------------------------------------------------------------
    // Ethanol (C-C-O skeleton, sp3) — O charge ≈ -0.390
    // -----------------------------------------------------------------------
    #[test]
    fn test_ethanol_oxygen() {
        // CH3-CH2-OH: just the O connected to a C with a single bond
        let mut mol = MolGraph::new();
        let c1 = mol.add_atom(atom("C")); // methyl C
        let c2 = mol.add_atom(atom("C")); // alpha C
        let o = mol.add_atom(atom("O"));
        bond_order(&mut mol, c1, c2, 1.0);
        bond_order(&mut mol, c2, o, 1.0);

        let results = compute_gasteiger_charges(&mol, 12);
        let gc_o = charge_of(&results, o);

        assert!(
            gc_o.charge < -0.33,
            "O charge = {} (expected ≈ -0.390)",
            gc_o.charge
        );
        assert!(
            gc_o.charge > -0.45,
            "O charge = {} (expected ≈ -0.390)",
            gc_o.charge
        );
    }

    // -----------------------------------------------------------------------
    // Benzene (aromatic C6) — each C ≈ -0.048
    // -----------------------------------------------------------------------
    #[test]
    fn test_benzene() {
        let mut mol = MolGraph::new();
        let ids: Vec<AtomId> = (0..6).map(|_| mol.add_atom(atom("C"))).collect();
        for i in 0..6 {
            bond_order(&mut mol, ids[i], ids[(i + 1) % 6], 1.5);
        }

        let results = compute_gasteiger_charges(&mol, 12);
        assert_eq!(results.len(), 6);

        for &id in &ids {
            let gc = charge_of(&results, id);
            assert!(
                gc.charge.abs() < 0.10,
                "benzene C charge = {} (expected ≈ -0.048)",
                gc.charge
            );
            // All C should be roughly equal (symmetric molecule)
        }

        // All six carbons should have nearly identical charges (symmetry)
        let charges: Vec<f64> = ids
            .iter()
            .map(|&id| charge_of(&results, id).charge)
            .collect();
        let mean = charges.iter().sum::<f64>() / 6.0;
        for &c in &charges {
            assert!(
                (c - mean).abs() < 1e-6,
                "benzene charges not symmetric: {:?}",
                charges
            );
        }
    }

    // -----------------------------------------------------------------------
    // NH4+ — formal_charge=+1 on N, split_charge distributes to N only
    // -----------------------------------------------------------------------
    #[test]
    fn test_nh4_plus() {
        let mut mol = MolGraph::new();
        let n = mol.add_atom(atom_fc("N", 1.0));
        // 4 implicit H from normal valence (N valence=3) + formal_charge=+1
        // → implicit_h_count returns 4

        let results = compute_gasteiger_charges(&mol, 12);
        assert_eq!(results.len(), 1);
        let gc = charge_of(&results, n);

        // N starts with +1 formal charge; after equalization with 4 implicit H
        // the positive charge distributes, so N ends up less positive.
        // Total charge must be +1.0 (conservation).
        let total = gc.charge + gc.h_charge;
        assert!(
            (total - 1.0).abs() < 0.02,
            "NH4+ charge not conserved: charge={} h_charge={} total={}",
            gc.charge,
            gc.h_charge,
            total
        );
        // Charges should be finite (no NaN/inf)
        assert!(
            gc.charge.is_finite(),
            "N charge is not finite: {}",
            gc.charge
        );
        assert!(
            gc.h_charge.is_finite(),
            "N h_charge is not finite: {}",
            gc.h_charge
        );
    }

    // -----------------------------------------------------------------------
    // Benzamidine (PhC(=NH)NH2) — amidine N atoms are conjugated → both ≈ -0.36
    // Test: neutral molecule, both N get similar charges via iteration
    // -----------------------------------------------------------------------
    #[test]
    fn test_benzamidine_amidine_nitrogen_symmetry() {
        // Build the amidine group: C(=N1)N2 with aromatic bond order for C
        // This is the C(=NH)NH2 part; we ignore the phenyl ring for simplicity.
        let mut mol = MolGraph::new();
        let c = mol.add_atom(atom("C")); // amidine carbon (sp2)
        let n1 = mol.add_atom(atom("N")); // =N (imine N, sp2)
        let n2 = mol.add_atom(atom("N")); // -N (amine N, sp2 due to conjugation)
        // C=N1 double bond, C-N2 single bond but C is sp2 → conjugated
        bond_order(&mut mol, c, n1, 2.0);
        bond_order(&mut mol, c, n2, 1.0);

        let results = compute_gasteiger_charges(&mol, 12);
        let gc_n1 = charge_of(&results, n1);
        let gc_n2 = charge_of(&results, n2);

        // Both N atoms should be negative (more electronegative than C)
        assert!(
            gc_n1.charge < 0.0,
            "N1 should be negative, got {}",
            gc_n1.charge
        );
        assert!(
            gc_n2.charge < 0.0,
            "N2 should be negative, got {}",
            gc_n2.charge
        );

        // In a protonated amidine (formal_charge=+1 on N1), split_charge
        // distributes +0.5 to each N. Test the neutral case has similar magnitudes.
        let diff = (gc_n1.charge - gc_n2.charge).abs();
        assert!(
            diff < 0.15,
            "amidine N charges differ too much: N1={} N2={} diff={}",
            gc_n1.charge,
            gc_n2.charge,
            diff
        );
    }

    // -----------------------------------------------------------------------
    // Protonated amidine — split_charge_conjugated distributes +1 across 2 N
    // -----------------------------------------------------------------------
    #[test]
    fn test_amidine_protonated_split_charge() {
        // C(=N1H)N2H2 with formal_charge=+1 on N1
        let mut mol = MolGraph::new();
        let c = mol.add_atom(atom("C"));
        let n1 = mol.add_atom(atom_fc("N", 1.0)); // protonated imine N
        let n2 = mol.add_atom(atom("N"));
        bond_order(&mut mol, c, n1, 2.0); // C=N1 (sp2)
        bond_order(&mut mol, c, n2, 1.0); // C-N2

        let results = compute_gasteiger_charges(&mol, 12);
        let gc_n1 = charge_of(&results, n1);
        let gc_n2 = charge_of(&results, n2);

        // Total charge must be conserved (+1 formal charge on N1)
        let gc_c = charge_of(&results, c);
        let total = gc_c.charge
            + gc_c.h_charge
            + gc_n1.charge
            + gc_n1.h_charge
            + gc_n2.charge
            + gc_n2.h_charge;
        assert!(
            (total - 1.0).abs() < 0.05,
            "protonated amidine charge not conserved: total = {}",
            total
        );
    }
}
