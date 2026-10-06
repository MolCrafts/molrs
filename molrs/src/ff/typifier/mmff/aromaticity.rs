//! MMFF94 aromaticity perception.
//!
//! Faithful port of `RDKit::MolOps::setMMFFAromaticity`
//! (`Code/GraphMol/Aromaticity.cpp`, BSD-3, RDKit contributors).
//!
//! The algorithm walks every SSSR ring and counts pi electrons
//! (2 per ring double bond, +1 per exocyclic double bond to an aromatic
//! neighbour, +2 for a 5-ring carrying N/O/divalent-S with no exocyclic
//! double bond and an odd ring size). A ring whose pi count satisfies the
//! 4n+2 Hückel rule and contains only sp2 carbon / nitrogen is flagged
//! aromatic; its ring bonds are promoted to the aromatic bond order. The
//! outer `while` loop iterates until no further atoms are marked, so fused
//! systems converge.
//!
//! Output is a new [`Topo`] with `is_aromatic`, `ring_aromatic`, and the
//! aromatic ring-bond orders filled in (immutable update — the input is
//! left untouched).

// The ring loop indexes several parallel per-ring arrays by ring index,
// mirroring the C++ `for (i = 0; i < atomRings.size(); ++i)` structure.
#![allow(clippy::needless_range_loop)]

use super::topo::{BondOrder, Topo};
use molrs::perceive::Hybridization;

/// Run MMFF aromaticity perception, returning an updated snapshot.
pub(crate) fn set_mmff_aromaticity(input: &Topo) -> Topo {
    let mut topo = input.clone();
    let n_rings = topo.ring_idx.len();
    if n_rings == 0 {
        return topo;
    }

    let n_atoms = topo.n_atoms();
    let mut arom_bit = vec![false; n_atoms];
    let mut arom_ring = vec![false; n_rings];

    let mut arom_rings_all_set = false;
    let mut n_arom_set: i64 = 0;
    let mut old_n_arom_set: i64 = -1;

    while !arom_rings_all_set && n_arom_set > old_n_arom_set {
        for ri in 0..n_rings {
            let ring = topo.ring_idx[ri].clone();
            let rsize = ring.len();
            let mut pi_e: u32 = 0;
            let mut move_to_next_ring = false;
            let mut is_nos_in_ring = false;
            let mut exo_double_bond = false;

            for j in 0..rsize {
                if move_to_next_ring {
                    break;
                }
                let a = ring[j];
                let atno = topo.atno[a];
                if atno == 7 || atno == 8 || (atno == 16 && topo.degree(a) == 2) {
                    is_nos_in_ring = true;
                }
                let next = if j == rsize - 1 { ring[0] } else { ring[j + 1] };
                if topo.bond_order(a, next) == Some(BondOrder::Double) {
                    pi_e += 2;
                } else {
                    // carbon, or nitrogen with total bond order == 4
                    let tbo = topo.total_bond_order(a);
                    if atno != 6 && !(atno == 7 && tbo == 4) {
                        continue;
                    }
                    for (p, &nbr) in topo.nbrs[a].iter().enumerate() {
                        // exocyclic only
                        if ring.contains(&nbr) {
                            continue;
                        }
                        let bo = topo.nbr_order[a][p];
                        if bo == BondOrder::Single {
                            continue;
                        }
                        // neighbor in a ring whose aromaticity is not yet set:
                        // defer this whole ring.
                        if topo.rings.is_atom_in_ring(topo.id(nbr)) && !arom_bit[nbr] {
                            move_to_next_ring = true;
                            break;
                        }
                        if bo == BondOrder::Double {
                            if arom_bit[nbr] {
                                pi_e += 1;
                            } else {
                                exo_double_bond = true;
                            }
                        }
                    }
                }
            }

            if move_to_next_ring {
                continue;
            }

            // mark perceived; reject non-sp2 C/N rings
            let mut can_be_aromatic = true;
            for &a in &ring {
                arom_bit[a] = true;
                let atno = topo.atno[a];
                if (atno == 6 || atno == 7) && topo.hybridization[a] != Hybridization::Sp2 {
                    can_be_aromatic = false;
                }
            }
            if !can_be_aromatic {
                continue;
            }

            if is_nos_in_ring && !exo_double_bond && (rsize % 2 == 1) {
                pi_e += 2;
            }

            if pi_e > 2 && (pi_e - 2).is_multiple_of(4) {
                arom_ring[ri] = true;
                for &a in &ring {
                    topo.is_aromatic[a] = true;
                }
            }
        }

        old_n_arom_set = n_arom_set;
        n_arom_set = 0;
        arom_rings_all_set = true;
        for ring in &topo.ring_idx {
            for &a in ring {
                if arom_bit[a] {
                    n_arom_set += 1;
                } else {
                    arom_rings_all_set = false;
                }
            }
        }
    }

    // Promote aromatic ring bonds to the aromatic bond order (so later passes
    // that test for Bond::AROMATIC behave like RDKit after setMMFFAromaticity).
    topo.ring_aromatic = arom_ring.clone();
    let arom_ring_bonds: Vec<(usize, usize)> = topo
        .ring_idx
        .iter()
        .enumerate()
        .filter(|(ri, _)| arom_ring[*ri])
        .flat_map(|(_, ring)| {
            let rsize = ring.len();
            (0..rsize).map(move |j| {
                let a = ring[j];
                let b = if j == rsize - 1 { ring[0] } else { ring[j + 1] };
                (a, b)
            })
        })
        .collect();
    for (a, b) in arom_ring_bonds {
        set_bond_aromatic(&mut topo, a, b);
    }

    topo
}

fn set_bond_aromatic(topo: &mut Topo, a: usize, b: usize) {
    if let Some(p) = topo.nbrs[a].iter().position(|&k| k == b) {
        topo.nbr_order[a][p] = BondOrder::Aromatic;
    }
    if let Some(p) = topo.nbrs[b].iter().position(|&k| k == a) {
        topo.nbr_order[b][p] = BondOrder::Aromatic;
    }
}
