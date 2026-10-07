//! Lightweight chemical perception for distance-geometry typing.
//!
//! `molrs::core::MolGraph` stores only connectivity and a numeric bond
//! `"order"`; it carries neither hybridization nor aromaticity. RDKit's
//! bounds-matrix builder, however, keys almost every decision off
//! `Atom::getHybridization()` / `getIsAromatic()` and `Bond::getIsConjugated`.
//!
//! This module gathers the perception RDKit would have computed: aromaticity,
//! and RDKit's own hybridization and conjugation from
//! [`molrs::perceive::perceive_hybridizations`] / [`molrs::perceive::perceive_conjugated_atoms`]. It is
//! intentionally scoped to the organic main group (the molecules this port is
//! validated against); it is **not** a general aromaticity model and will not
//! reproduce RDKit on exotic ring systems (documented in `mod.rs`).

use std::collections::HashMap;

use molrs::core::Atomistic;
use molrs::core::Element;
use molrs::core::NodeId;
use molrs::perceive::{Hybridization, perceive_conjugated_atoms, perceive_hybridizations};
use molrs::perceive::{RingInfo, perceive_rings};

/// Per-atom perceived properties consumed by the bounds builder.
#[derive(Clone, Debug)]
pub struct PerceivedAtom {
    pub element: Element,
    pub hybridization: Hybridization,
    pub aromatic: bool,
    pub conjugated: bool,
    pub degree: usize,
    /// Heavy + H σ-bond count plus π contributions, used for S charge flags.
    pub total_valence: f64,
}

/// DgFeatures view of a molecule: index-aligned atoms, neighbour lists, bond
/// orders, ring info, and aromatic bond flags.
pub struct DgFeatures {
    pub atom_ids: Vec<NodeId>,
    pub atoms: Vec<PerceivedAtom>,
    /// `adj[i]` = sorted neighbour indices of atom `i`.
    pub adj: Vec<Vec<usize>>,
    /// `order[(min,max)]` = graph bond order between the two atoms.
    pub order: HashMap<(usize, usize), f64>,
    /// Aromatic flag per atom-pair bond.
    pub aromatic_bond: HashMap<(usize, usize), bool>,
    pub rings: RingInfo,
    /// Ring atom-index sets (each ring as a `Vec<usize>` in ring order).
    pub ring_idx: Vec<Vec<usize>>,
}

impl DgFeatures {
    /// Bond order between atom indices `i` and `j`, or `0.0` if not bonded.
    pub fn bond_order(&self, i: usize, j: usize) -> f64 {
        let key = if i < j { (i, j) } else { (j, i) };
        self.order.get(&key).copied().unwrap_or(0.0)
    }

    /// Whether the bond `i-j` is aromatic.
    pub fn is_aromatic_bond(&self, i: usize, j: usize) -> bool {
        let key = if i < j { (i, j) } else { (j, i) };
        self.aromatic_bond.get(&key).copied().unwrap_or(false)
    }
}

fn element_of(mol: &Atomistic, id: NodeId) -> Element {
    mol.get_atom(id)
        .ok()
        .and_then(|a| a.get_str("element").and_then(Element::by_symbol))
        .unwrap_or(Element::C)
}

/// Perceive hybridization, aromaticity and conjugation for `mol`.
pub fn perceive_dg_features(mol: &Atomistic) -> DgFeatures {
    let atom_ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
    let id_to_idx: HashMap<NodeId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let n = atom_ids.len();

    let mut adj = vec![Vec::new(); n];
    let mut order: HashMap<(usize, usize), f64> = HashMap::new();
    for (i, &aid) in atom_ids.iter().enumerate() {
        let mut nbrs: Vec<usize> = Vec::new();
        for (nid, bid) in mol.neighbor_bonds(aid) {
            if let Some(&j) = id_to_idx.get(&nid) {
                nbrs.push(j);
                let key = if i < j { (i, j) } else { (j, i) };
                // Distance geometry wants a bond-length proxy, so an aromatic
                // bond is the partial order its geometry actually has.
                let ord = if mol.bond_type(bid).is_aromatic() {
                    1.5
                } else {
                    mol.bond_number(bid).count().max(1) as f64
                };
                order.insert(key, ord);
            }
        }
        nbrs.sort_unstable();
        nbrs.dedup();
        adj[i] = nbrs;
    }

    let rings = perceive_rings(mol);
    let ring_idx: Vec<Vec<usize>> = rings
        .rings()
        .iter()
        .map(|r| {
            r.iter()
                .filter_map(|aid| id_to_idx.get(aid).copied())
                .collect()
        })
        .collect();

    // Aromaticity: delegate to the shared RDKit-aligned model in `perceive`
    // (`molrs::perceive::mark_aromaticity`, a port of
    // `setAromaticity(AROMATICITY_RDKIT)`) instead of re-deriving it here. It
    // annotates a *clone* of the graph with an `is_aromatic = 1` flag per
    // aromatic atom; we read those flags back, index-aligned.
    //
    // Mutating a clone keeps `perceive` non-destructive on the caller's graph
    // (the embed pipeline relies on its input being untouched), and the clone
    // preserves atom-insertion order, so `probe.atoms()` enumerates in the same
    // sequence as `atom_ids` — we can zip them index-by-index.
    let mut aromatic_atom = vec![false; n];
    {
        let mut probe = mol.clone();
        molrs::perceive::mark_aromaticity(&mut probe);
        for (i, (_, atom)) in probe.atoms().enumerate().take(n) {
            if atom.get_int("is_aromatic") == Some(1) {
                aromatic_atom[i] = true;
            }
        }
    }

    // Hybridization and conjugation are RDKit's (`perceive`), which is what
    // its bounds builder keys on.
    let hybridization = perceive_hybridizations(mol);
    let conjugated = perceive_conjugated_atoms(mol);
    let atoms: Vec<PerceivedAtom> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &aid)| PerceivedAtom {
            element: element_of(mol, aid),
            hybridization: hybridization[i],
            aromatic: aromatic_atom[i],
            conjugated: conjugated[i],
            degree: adj[i].len(),
            total_valence: adj[i].iter().map(|&j| bond_order_of(&order, i, j)).sum(),
        })
        .collect();

    // Aromatic bond flags.
    let mut aromatic_bond: HashMap<(usize, usize), bool> = HashMap::new();
    for i in 0..n {
        for &j in &adj[i] {
            if j <= i {
                continue;
            }
            let arom = aromatic_atom[i]
                && aromatic_atom[j]
                && rings.is_atom_in_ring(atom_ids[i])
                && rings.is_atom_in_ring(atom_ids[j])
                && bond_shares_ring(&ring_idx, i, j);
            aromatic_bond.insert((i, j), arom);
        }
    }

    DgFeatures {
        atom_ids,
        atoms,
        adj,
        order,
        aromatic_bond,
        rings,
        ring_idx,
    }
}

/// The graph bond order between atoms `i` and `j` (1 when unrecorded).
fn bond_order_of(order: &HashMap<(usize, usize), f64>, i: usize, j: usize) -> f64 {
    order.get(&(i.min(j), i.max(j))).copied().unwrap_or(1.0)
}

/// Whether atoms `i` and `j` are adjacent within some common ring.
fn bond_shares_ring(ring_idx: &[Vec<usize>], i: usize, j: usize) -> bool {
    for ring in ring_idx {
        let rsize = ring.len();
        for w in 0..rsize {
            let a = ring[w];
            let b = ring[(w + 1) % rsize];
            if (a == i && b == j) || (a == j && b == i) {
                return true;
            }
        }
    }
    false
}
