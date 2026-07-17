//! UFF atom-type parameters and bond-rest-length formula.
//!
//! Ported from RDKit (BSD-3, Copyright (C) 2004-2025 Greg Landrum and other
//! RDKit contributors):
//!   * `$RDBASE/Code/ForceField/UFF/Params.cpp`     — the parameter table,
//!   * `$RDBASE/Code/ForceField/UFF/BondStretch.cpp` — `calcBondRestLength`,
//!   * `$RDBASE/Code/GraphMol/ForceFieldHelpers/UFF/AtomTyper.cpp` —
//!     `getAtomLabel`.
//!
//! RDKit derives the 1-2 bond bound from the UFF rest length, which itself
//! depends on the per-atom-type valence-bond radius `r1` and GMP
//! electronegativity `Xi`. We embed only the rows needed by the organic
//! subset this port targets (H, C, N, O, F, S, Cl, Br, P, I); unsupported
//! types fall back to a crude vdW estimate, matching RDKit's behaviour when
//! `getAtomTypes` cannot type an atom.

use super::mol_features::{Hybridization, PerceivedAtom};

/// `lambda` scaling factor for the Pauling bond-order correction (UFF).
const LAMBDA: f64 = 0.1332;

/// Per-atom-type UFF parameters relevant to bond rest length.
#[derive(Clone, Copy, Debug)]
struct UffParams {
    /// Valence-bond radius `r1` (Å).
    r1: f64,
    /// GMP electronegativity `Xi`.
    xi: f64,
}

/// Look up UFF parameters by RDKit atom-type label (e.g. `"C_3"`, `"C_R"`).
///
/// Returns `None` for labels outside the embedded organic subset.
fn params_for_label(label: &str) -> Option<UffParams> {
    // Values transcribed verbatim from RDKit `defaultParamData` (Params.cpp).
    let (r1, xi) = match label {
        "H_" => (0.354, 4.528),
        "C_3" => (0.757, 5.343),
        "C_R" => (0.729, 5.343),
        "C_2" => (0.732, 5.343),
        "C_1" => (0.706, 5.343),
        "N_3" => (0.700, 6.899),
        "N_R" => (0.699, 6.899),
        "N_2" => (0.685, 6.899),
        "N_1" => (0.656, 6.899),
        "O_3" => (0.658, 8.741),
        "O_R" => (0.680, 8.741),
        "O_2" => (0.634, 8.741),
        "O_1" => (0.639, 8.741),
        "F_" => (0.668, 10.874),
        "S_3+2" => (1.064, 6.928),
        "S_3+4" => (1.049, 6.928),
        "S_3+6" => (1.027, 6.928),
        "S_R" => (1.077, 6.928),
        "S_2" => (0.854, 6.928),
        "Cl" => (1.044, 8.564),
        "Br" => (1.192, 7.790),
        "P_3+3" => (1.101, 5.463),
        "P_3+5" => (1.056, 5.463),
        "I_" => (1.382, 6.822),
        _ => return None,
    };
    Some(UffParams { r1, xi })
}

/// Build the RDKit UFF atom-type label for a perceived atom.
///
/// Port of `Tools::getAtomLabel` restricted to main-group organic elements.
/// The hybridization → suffix map mirrors RDKit: `SP→1`, `SP2→2` (or `R`
/// when aromatic/conjugated for C/N/O/S), `SP3→3`; halogens and H take no
/// hybridization suffix. Sulfur additionally gets a `+2/+4/+6` charge flag
/// from its valence.
pub fn atom_label(atom: &PerceivedAtom) -> Option<String> {
    let sym = atom.element.symbol();
    // Halogens / hydrogen: no hybridization suffix.
    match sym {
        "H" => return Some("H_".to_string()),
        "F" => return Some("F_".to_string()),
        "Cl" => return Some("Cl".to_string()),
        "Br" => return Some("Br".to_string()),
        "I" => return Some("I_".to_string()),
        _ => {}
    }

    let mut key = sym.to_string();
    if key.len() == 1 {
        key.push('_');
    }

    let conj_r = matches!(sym, "C" | "N" | "O" | "S");
    match atom.hybridization {
        Hybridization::Sp => key.push('1'),
        Hybridization::Sp2 => {
            if (atom.aromatic || atom.conjugated) && conj_r {
                key.push('R');
            } else {
                key.push('2');
            }
        }
        Hybridization::Sp3 => key.push('3'),
        Hybridization::Other => key.push('3'),
    }

    // Sulfur charge flag from valence (RDKit addAtomChargeFlags, S branch).
    if sym == "S" && atom.hybridization != Hybridization::Sp2 {
        let v = atom.total_valence.round() as i64;
        let flag = match v {
            2 => "+2",
            4 => "+4",
            6 => "+6",
            _ => "+6",
        };
        key.push_str(flag);
    }

    Some(key)
}

/// UFF bond rest length (RDKit `calcBondRestLength`):
/// `r0 = ri + rj + rBO - rEN` with the Pauling and O'Keefe-Breese corrections.
fn rest_length(p1: UffParams, p2: UffParams, bond_order: f64) -> f64 {
    let (ri, rj) = (p1.r1, p2.r1);
    let r_bo = -LAMBDA * (ri + rj) * bond_order.ln();
    let (xi, xj) = (p1.xi, p2.xi);
    let dx = xi.sqrt() - xj.sqrt();
    let r_en = ri * rj * dx * dx / (xi * ri + xj * rj);
    ri + rj + r_bo - r_en
}

/// Resolve the bond order RDKit uses for the rest-length formula.
///
/// Aromatic bonds use 1.5 and amide C-N bonds use 1.41 (UFF `amideBondOrder`),
/// otherwise the integral order from the graph. Matches
/// `Bond::getBondTypeAsDouble` plus the amide special case applied in MMFF/UFF
/// typing pipelines feeding `set12Bounds`.
pub fn effective_bond_order(order: f64, aromatic: bool, amide: bool) -> f64 {
    if amide {
        1.41
    } else if aromatic {
        1.5
    } else {
        order
    }
}

/// 1-2 bound estimate `(rest_length, found_params)`.
///
/// When both atoms are typed, returns the UFF rest length. Otherwise returns
/// the crude `(vw1+vw2)/2` fallback flagged `found = false`, matching
/// RDKit's untyped branch in `set12Bounds`.
pub fn bond_rest_length(a: &PerceivedAtom, b: &PerceivedAtom, effective_order: f64) -> (f64, bool) {
    let pa = atom_label(a).and_then(|l| params_for_label(&l));
    let pb = atom_label(b).and_then(|l| params_for_label(&l));
    match (pa, pb) {
        (Some(pa), Some(pb)) if effective_order > 0.0 => {
            (rest_length(pa, pb, effective_order), true)
        }
        _ => {
            let bl = 0.5 * (rvdw(a.element.z()) + rvdw(b.element.z()));
            (bl, false)
        }
    }
}

/// Van der Waals radius (Å) as used by RDKit's `PeriodicTable::getRvdw`.
///
/// These differ from [`molrs::Element::vdw_radius`] (Bondi-style)
/// — RDKit ships its own table in `atomic_data`, and `setLowerBoundVDW` /
/// `set15Bounds` depend on the exact values, so we transcribe RDKit's.
pub fn rvdw(z: u8) -> f64 {
    match z {
        1 => 1.2,
        5 => 1.8,
        6 => 1.7,
        7 => 1.6,
        8 => 1.55,
        9 => 1.5,
        14 => 2.1,
        15 => 1.95,
        16 => 1.8,
        17 => 1.8,
        35 => 1.9,
        53 => 2.1,
        // RDKit default for unlisted elements is 2.0.
        _ => 2.0,
    }
}
