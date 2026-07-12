//! Lightweight chemical perception for distance-geometry typing.
//!
//! `molrs::MolGraph` stores only connectivity and a numeric bond
//! `"order"`; it carries neither hybridization nor aromaticity. RDKit's
//! bounds-matrix builder, however, keys almost every decision off
//! `Atom::getHybridization()` / `getIsAromatic()` and `Bond::getIsConjugated`.
//!
//! This module reconstructs the minimal perception RDKit would have computed:
//! hybridization from the per-atom maximum bond order and degree, aromaticity
//! from simple-ring analysis, and conjugation from adjacent π systems. It is
//! intentionally scoped to the organic main group (the molecules this port is
//! validated against); it is **not** a general aromaticity model and will not
//! reproduce RDKit on exotic ring systems (documented in `mod.rs`).

use std::collections::HashMap;

use molrs::chem::rings::{RingInfo, find_rings};
use molrs::system::atomistic::{AtomId, Atomistic};
use molrs::system::element::Element;

/// Coarse hybridization label (subset of RDKit's `Atom::HybridizationType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hybridization {
    /// Linear (one σ skeleton neighbour pattern, triple/cumulene).
    Sp,
    /// Trigonal planar (one double bond or aromatic).
    Sp2,
    /// Tetrahedral (all single bonds).
    Sp3,
    /// Anything else (hypervalent, isolated atoms).
    Other,
}

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

/// Perceived view of a molecule: index-aligned atoms, neighbour lists, bond
/// orders, ring info, and aromatic/amide bond flags.
pub struct Perceived {
    pub atom_ids: Vec<AtomId>,
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

impl Perceived {
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

fn element_of(mol: &Atomistic, id: AtomId) -> Element {
    mol.get_atom(id)
        .ok()
        .and_then(|a| a.get_str("element").and_then(Element::by_symbol))
        .unwrap_or(Element::C)
}

/// Perceive hybridization, aromaticity and conjugation for `mol`.
pub fn perceive(mol: &Atomistic) -> Perceived {
    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let id_to_idx: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let n = atom_ids.len();
    let _ = &id_to_idx;

    let mut adj = vec![Vec::new(); n];
    let mut order: HashMap<(usize, usize), f64> = HashMap::new();
    for (i, &aid) in atom_ids.iter().enumerate() {
        let mut nbrs: Vec<usize> = Vec::new();
        for (nid, ord) in mol.neighbor_bonds(aid) {
            if let Some(&j) = id_to_idx.get(&nid) {
                nbrs.push(j);
                let key = if i < j { (i, j) } else { (j, i) };
                order.insert(key, ord);
            }
        }
        nbrs.sort_unstable();
        nbrs.dedup();
        adj[i] = nbrs;
    }

    let rings = find_rings(mol);
    let ring_idx: Vec<Vec<usize>> = rings
        .rings()
        .iter()
        .map(|r| {
            r.iter()
                .filter_map(|aid| id_to_idx.get(aid).copied())
                .collect()
        })
        .collect();

    // Aromaticity: delegate to the shared RDKit-aligned model in molrs-core
    // (`molrs::perceive_aromaticity`, a port of
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
        molrs::perceive_aromaticity(&mut probe);
        for (i, (_, atom)) in probe.atoms().enumerate().take(n) {
            if atom.get_int("is_aromatic") == Some(1) {
                aromatic_atom[i] = true;
            }
        }
    }

    let mut atoms: Vec<PerceivedAtom> = Vec::with_capacity(n);
    for (i, &aid) in atom_ids.iter().enumerate() {
        let element = element_of(mol, aid);
        let degree = adj[i].len();
        let mut max_order = 0.0_f64;
        let mut valence = 0.0_f64;
        for &j in &adj[i] {
            let o = {
                let key = if i < j { (i, j) } else { (j, i) };
                order.get(&key).copied().unwrap_or(1.0)
            };
            max_order = max_order.max(o);
            valence += o;
        }
        let hybridization = if aromatic_atom[i] {
            Hybridization::Sp2
        } else if max_order >= 2.5 {
            Hybridization::Sp
        } else if (max_order - 2.0).abs() < 0.25 {
            // exactly one double bond → sp2; two cumulated doubles → sp
            let n_double = adj[i]
                .iter()
                .filter(|&&j| {
                    let key = if i < j { (i, j) } else { (j, i) };
                    order.get(&key).copied().unwrap_or(0.0) >= 1.75
                })
                .count();
            if n_double >= 2 {
                Hybridization::Sp
            } else {
                Hybridization::Sp2
            }
        } else if max_order >= 1.5 {
            Hybridization::Sp2
        } else if degree == 0 {
            Hybridization::Other
        } else {
            Hybridization::Sp3
        };

        atoms.push(PerceivedAtom {
            element,
            hybridization,
            aromatic: aromatic_atom[i],
            conjugated: false,
            degree,
            total_valence: valence,
        });
    }

    // Lone-pair conjugation: RDKit perceives the hydroxyl/ester oxygen and
    // the amide nitrogen adjacent to a carbonyl as SP2 (their lone pair
    // conjugates into the C=O π system). Upgrade an O (degree ≤ 2) or amide
    // N bonded to a carbonyl-type carbon (a C bearing a double bond to O/N)
    // to SP2. This matches RDKit's hybridization for esters/acids/amides and
    // is what selects the UFF `O_R` / `N_R` rest lengths and sp2 angles.
    let is_carbonyl_c = |i: usize| -> bool {
        if element_of(mol, atom_ids[i]).symbol() != "C" {
            return false;
        }
        adj[i].iter().any(|&j| {
            let key = if i < j { (i, j) } else { (j, i) };
            let o = order.get(&key).copied().unwrap_or(0.0);
            let s = element_of(mol, atom_ids[j]).symbol();
            o >= 1.75 && (s == "O" || s == "N")
        })
    };
    let mut upgrade_sp2 = vec![false; n];
    for i in 0..n {
        let sym = element_of(mol, atom_ids[i]).symbol();
        if atoms[i].hybridization != Hybridization::Sp3 {
            continue;
        }
        let eligible = (sym == "O" && atoms[i].degree <= 2) || (sym == "N" && atoms[i].degree == 3);
        if eligible && adj[i].iter().any(|&j| is_carbonyl_c(j)) {
            upgrade_sp2[i] = true;
        }
    }
    for i in 0..n {
        if upgrade_sp2[i] {
            atoms[i].hybridization = Hybridization::Sp2;
        }
    }

    // Conjugation: a bond is conjugated when it joins two sp2/sp/aromatic
    // atoms (RDKit's `markConjBonds` essence). An atom is conjugated if any of
    // its bonds is conjugated. This is what flips carbonyl/ester C and O to
    // the UFF `*_R` types.
    let mut conj_atom = vec![false; n];
    for i in 0..n {
        for &j in &adj[i] {
            if j <= i {
                continue;
            }
            let hi = atoms[i].hybridization;
            let hj = atoms[j].hybridization;
            let pi_i = matches!(hi, Hybridization::Sp | Hybridization::Sp2);
            let pi_j = matches!(hj, Hybridization::Sp | Hybridization::Sp2);
            if pi_i && pi_j {
                conj_atom[i] = true;
                conj_atom[j] = true;
            }
        }
    }
    for i in 0..n {
        atoms[i].conjugated = conj_atom[i] || atoms[i].aromatic;
    }

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

    Perceived {
        atom_ids,
        atoms,
        adj,
        order,
        aromatic_bond,
        rings,
        ring_idx,
    }
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
