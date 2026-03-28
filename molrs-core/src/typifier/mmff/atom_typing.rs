//! MMFF94 atom type assignment.
//!
//! Computes the 8-tuple MMFF classification properties for each atom, then
//! matches them against the MMFF property table to assign integer type IDs.

use std::collections::HashMap;

use crate::element::Element;
use crate::hydrogens::implicit_h_count;
use crate::molgraph::{AtomId, MolGraph};
use crate::rings::RingInfo;

use super::params::{MMFFParams, PropKey};

/// Computed MMFF properties for one atom.
#[derive(Debug, Clone)]
struct AtomMmffProps {
    atno: u32,
    crd: u32,
    val: u32,
    pilp: u32,
    mltb: u32,
    arom: u32,
    linh: u32,
    sbmb: u32,
}

impl AtomMmffProps {
    fn as_key(&self) -> PropKey {
        (
            self.atno, self.crd, self.val, self.pilp, self.mltb, self.arom, self.linh, self.sbmb,
        )
    }
}

/// Compute the MMFF classification properties for an atom.
fn compute_mmff_props(
    mol: &MolGraph,
    atom_id: AtomId,
    ring_info: &RingInfo,
) -> Option<AtomMmffProps> {
    let atom = mol.get_atom(atom_id).ok()?;
    let sym = atom.get_str("element")?;
    let element = Element::by_symbol(sym)?;
    let atno = element as u32;

    // Coordination number = number of bonded neighbors
    let bonds: Vec<(AtomId, f64)> = mol.neighbor_bonds(atom_id).collect();
    let crd = bonds.len() as u32;

    // Sum of bond orders → effective valence
    let bond_order_sum: f64 = bonds.iter().map(|(_, o)| *o).sum();
    // Add implicit H
    let n_impl_h = implicit_h_count(mol, atom_id).unwrap_or(0) as u32;
    let val = (bond_order_sum.round() as u32) + n_impl_h;

    // Multiple bond type
    let max_order = bonds.iter().map(|(_, o)| *o).fold(0.0_f64, f64::max);
    let mltb = if max_order >= 2.5 {
        3 // triple
    } else if max_order >= 1.75 {
        2 // double
    } else if max_order >= 1.25 {
        // Could be delocalized/aromatic
        if ring_info.is_atom_in_ring(atom_id) {
            2 // aromatic counts as mltb=2 for ring atoms with 1.5 bonds
        } else {
            1 // delocalized
        }
    } else {
        // Check if any neighbor has double/triple bond (sbmb case)
        let has_adjacent_multi = bonds
            .iter()
            .any(|(nbr, _)| mol.neighbor_bonds(*nbr).any(|(_, o)| o >= 1.75));
        if has_adjacent_multi && crd > 1 {
            1 // adjacent to multiple bond
        } else {
            0
        }
    };

    // Aromaticity: atom is in a ring with aromatic (1.5 order) bonds
    let arom = if ring_info.is_atom_in_ring(atom_id)
        && bonds.iter().any(|(_, o)| (*o - 1.5).abs() < 0.1)
    {
        1
    } else {
        0
    };

    // Pi lone pair: N, O, S with lone pairs available for pi interaction
    let pilp = match atno {
        7 => {
            // N: has lone pair if sp3 or if it's in an aromatic ring (pyrrole-like)
            if (crd <= 3 && mltb == 0 && arom == 0) || (arom == 1 && crd == 3) {
                1
            } else {
                0
            }
        }
        8 => {
            // O: always has lone pair unless it's O+ or triple-bonded
            if crd <= 2 && max_order < 2.5 { 1 } else { 0 }
        }
        9 | 17 | 35 | 53 => 1, // halogens
        16 => {
            // S: has lone pair in sp3 or aromatic
            if crd <= 2 { 1 } else { 0 }
        }
        15 => {
            // P: has lone pair if crd <= 3
            if crd <= 3 { 1 } else { 0 }
        }
        _ => 0,
    };

    // Linear: sp hybridized with triple bond or cumulated doubles
    let linh = if max_order >= 2.5 && crd <= 2 { 1 } else { 0 };

    // Single bond / multiple bond: atom has both single and multiple bonds
    let has_single = bonds.iter().any(|(_, o)| *o < 1.25);
    let has_multi = bonds.iter().any(|(_, o)| *o >= 1.25);
    let sbmb = if has_single && has_multi { 1 } else { 0 };

    Some(AtomMmffProps {
        atno,
        crd,
        val,
        pilp,
        mltb,
        arom,
        linh,
        sbmb,
    })
}

/// Assign MMFF94 atom types to all atoms in the molecular graph.
pub(crate) fn assign_atom_types(
    mol: &MolGraph,
    ring_info: &RingInfo,
    params: &MMFFParams,
) -> HashMap<AtomId, u32> {
    let mut types = HashMap::new();

    for (atom_id, _) in mol.atoms() {
        let mmff_type = match compute_mmff_props(mol, atom_id, ring_info) {
            Some(props) => match_atom_type(&props, atom_id, mol, ring_info, params),
            None => 0, // unknown atom
        };
        types.insert(atom_id, mmff_type);
    }

    types
}

/// Match computed properties against the MMFF property table to find the type.
fn match_atom_type(
    props: &AtomMmffProps,
    atom_id: AtomId,
    mol: &MolGraph,
    ring_info: &RingInfo,
    params: &MMFFParams,
) -> u32 {
    let key = props.as_key();
    let candidates = params.find_types(key);

    match candidates.len() {
        0 => {
            // Try relaxing sbmb
            let relaxed_key = (key.0, key.1, key.2, key.3, key.4, key.5, key.6, 0);
            let relaxed = params.find_types(relaxed_key);
            if !relaxed.is_empty() {
                return disambiguate(relaxed, atom_id, mol, ring_info, params);
            }
            // Try relaxing pilp
            let relaxed_key = (key.0, key.1, key.2, 0, key.4, key.5, key.6, key.7);
            let relaxed = params.find_types(relaxed_key);
            if !relaxed.is_empty() {
                return disambiguate(relaxed, atom_id, mol, ring_info, params);
            }
            // Fallback: generic type for this element
            fallback_type(props.atno)
        }
        1 => candidates[0],
        _ => disambiguate(candidates, atom_id, mol, ring_info, params),
    }
}

/// When multiple types match, use ring size and neighbor context to disambiguate.
fn disambiguate(
    candidates: &[u32],
    atom_id: AtomId,
    _mol: &MolGraph,
    ring_info: &RingInfo,
    _params: &MMFFParams,
) -> u32 {
    let smallest_ring = ring_info.smallest_ring_containing_atom(atom_id);

    // Ring-size disambiguation for carbon
    for &tid in candidates {
        match tid {
            // CR4R (20): sp3 C in 4-membered ring
            20 => {
                if smallest_ring == Some(4) {
                    return 20;
                }
            }
            // CR3R (22): sp3 C in 3-membered ring
            22 => {
                if smallest_ring == Some(3) {
                    return 22;
                }
            }
            // CE4R (30): sp2 C=C in 4-membered ring
            30 => {
                if smallest_ring == Some(4) {
                    return 30;
                }
            }
            _ => {}
        }
    }

    // Default: pick the first (lowest type_id) candidate
    *candidates.iter().min().unwrap_or(&0)
}

/// Fallback type for elements with no property-table match.
fn fallback_type(atno: u32) -> u32 {
    match atno {
        6 => 1,   // C → CR
        1 => 5,   // H → HC
        8 => 6,   // O → OR
        7 => 8,   // N → NR
        9 => 11,  // F
        17 => 12, // Cl
        35 => 13, // Br
        53 => 14, // I
        16 => 15, // S
        14 => 19, // Si
        15 => 26, // P
        _ => 0,
    }
}
