//! Hybridization and conjugation, as RDKit perceives them.
//!
//! A port of RDKit `Code/GraphMol/ConjugHybrid.cpp` (`setConjugation`,
//! `setHybridization`) and of `countAtomElec` from `Aromaticity.cpp` (BSD-3,
//! RDKit contributors). Hybridization is counted, not guessed: an atom's
//! σ-bonds plus its lone pairs (`numBondsPlusLonePairs`) fix its orbital count,
//! and a four-orbital atom that carries a conjugated bond — the pyrrole
//! nitrogen, an amide nitrogen, an ester oxygen — drops to sp² because its lone
//! pair joins the π system.
//!
//! This is the one hybridization model in molrs. MMFF aromaticity, UFF atom
//! labels and the ETKDG bounds builder all read it.
//!
//! Hydrogens the graph implies but does not draw count
//! ([`implicit_h_count`](super::hydrogens::implicit_h_count)), so the answer
//! does not depend on whether hydrogens were made explicit. Bond orders are the
//! localized (Kekulé) numbers; a bond whose class is aromatic is also
//! conjugated, as in RDKit.

use molrs::system::Atomistic;
use molrs::system::BondNumber;
use molrs::system::Element;
use molrs::system::NodeId;
use molrs::system::PropValue;

/// An atom's hybridization — RDKit's `Atom::HybridizationType`, less its
/// `UNSPECIFIED` / `SP2D` (which `setHybridization` never assigns).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hybridization {
    /// One orbital or none: hydrogen, a bare ion.
    S,
    /// Linear: two σ-bonds plus lone pairs.
    Sp,
    /// Trigonal: three, or four with a conjugated bond.
    Sp2,
    /// Tetrahedral.
    Sp3,
    /// Five orbitals (trigonal bipyramidal).
    Sp3d,
    /// Six orbitals (octahedral).
    Sp3d2,
    /// No orbital count applies: a dummy atom, or more than six.
    Other,
}

/// The hybridization of every atom of `mol`, in its atom order.
pub fn hybridizations(mol: &Atomistic) -> Vec<Hybridization> {
    let g = Snapshot::new(mol);
    (0..g.atno.len()).map(|i| g.hybridization(i)).collect()
}

/// Whether each atom of `mol`, in its atom order, carries a conjugated bond
/// (RDKit `MolOps::atomHasConjugatedBond` after `setConjugation`).
pub fn conjugated_atoms(mol: &Atomistic) -> Vec<bool> {
    let g = Snapshot::new(mol);
    (0..g.atno.len())
        .map(|i| g.has_conjugated_bond(i))
        .collect()
}

/// The per-atom facts RDKit's two passes read, flattened once.
struct Snapshot {
    atno: Vec<u8>,
    formal_charge: Vec<i32>,
    /// Neighbour indices, with each bond's localized order and aromatic class.
    nbrs: Vec<Vec<(usize, u32, bool)>>,
    implicit_h: Vec<u32>,
}

impl Snapshot {
    fn new(mol: &Atomistic) -> Self {
        let ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
        let index: std::collections::HashMap<NodeId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut atno = Vec::with_capacity(ids.len());
        let mut formal_charge = Vec::with_capacity(ids.len());
        let mut nbrs = Vec::with_capacity(ids.len());
        let mut implicit_h = Vec::with_capacity(ids.len());
        for &id in &ids {
            let atom = mol.get_atom(id).ok();
            atno.push(
                atom.as_ref()
                    .and_then(|a| a.get_str("element"))
                    .and_then(Element::by_symbol)
                    .map_or(0, |e| e.z()),
            );
            formal_charge.push(match atom.as_ref().and_then(|a| a.get("formal_charge")) {
                Some(PropValue::F64(v)) => v.round() as i32,
                Some(PropValue::Int(v)) => *v,
                _ => 0,
            });
            nbrs.push(
                mol.neighbor_bonds(id)
                    .filter_map(|(other, bid)| {
                        let order = match mol.bond_number(bid) {
                            BondNumber::Triple | BondNumber::Quadruple => 3,
                            BondNumber::Double => 2,
                            BondNumber::Single | BondNumber::Unknown => 1,
                        };
                        Some((*index.get(&other)?, order, mol.bond_type(bid).is_aromatic()))
                    })
                    .collect(),
            );
            implicit_h.push(super::hydrogens::implicit_h_count(mol, id).unwrap_or(0));
        }
        Self {
            atno,
            formal_charge,
            nbrs,
            implicit_h,
        }
    }

    /// RDKit `getDegree()`: neighbours drawn in the graph.
    fn degree(&self, i: usize) -> usize {
        self.nbrs[i].len()
    }

    /// RDKit `getTotalDegree()`: drawn neighbours plus implied hydrogens.
    fn total_degree(&self, i: usize) -> usize {
        self.degree(i) + self.implicit_h[i] as usize
    }

    /// RDKit `getValence(EXPLICIT)`: the localized bond orders drawn.
    fn explicit_valence(&self, i: usize) -> i32 {
        self.nbrs[i].iter().map(|&(_, o, _)| o as i32).sum()
    }

    /// RDKit `getTotalValence()`.
    fn total_valence(&self, i: usize) -> i32 {
        self.explicit_valence(i) + self.implicit_h[i] as i32
    }

    /// RDKit `countAtomElec`: π electrons the atom can donate, `-1` if none.
    fn count_atom_elec(&self, i: usize) -> i32 {
        let atno = self.atno[i];
        let dv = default_valence(atno);
        if dv <= 1 {
            return -1;
        }
        let degree = self.total_degree(i) as i32;
        if degree > 3 {
            return -1;
        }
        let Some(nouter) = nouter_elecs(atno) else {
            return -1;
        };
        let nlp = (nouter - dv - self.formal_charge[i]).max(0);
        let mut res = (dv - degree) + nlp;
        if res > 1 && self.explicit_valence(i) - self.degree(i) as i32 > 1 {
            res = 1;
        }
        res
    }

    /// RDKit `isAtomConjugCand`.
    fn is_conjugation_candidate(&self, i: usize) -> bool {
        let atno = self.atno[i];
        let minv = min_valence(atno);
        if self.formal_charge[i] == 0 && minv >= 0 && self.total_valence(i) > minv {
            return false;
        }
        let Some(nouter) = nouter_elecs(atno) else {
            return false;
        };
        (atno <= 10 || (nouter != 5 && nouter != 6) || (nouter == 6 && self.total_degree(i) < 2))
            && self.count_atom_elec(i) > 0
    }

    /// Does RDKit `markConjAtomBonds(at)` mark the bond `at`–`other`?
    ///
    /// `at` must be a candidate with two or three substituents; a pair of its
    /// bonds is marked when the first reaches a candidate with valence
    /// contribution ≥ 1.5 and the second reaches a candidate of at most three
    /// substituents. Both bonds of such a pair are marked.
    fn marks_conjugated(&self, at: usize, other: usize) -> bool {
        if !self.is_conjugation_candidate(at) || !(2..=3).contains(&self.total_degree(at)) {
            return false;
        }
        let contrib = |&(_, order, aromatic): &(usize, u32, bool)| {
            if aromatic { 1.5 } else { f64::from(order) }
        };
        for (p1, b1) in self.nbrs[at].iter().enumerate() {
            if contrib(b1) < 1.5 || !self.is_conjugation_candidate(b1.0) {
                continue;
            }
            for (p2, b2) in self.nbrs[at].iter().enumerate() {
                if p1 == p2 || self.total_degree(b2.0) > 3 || !self.is_conjugation_candidate(b2.0) {
                    continue;
                }
                if b1.0 == other || b2.0 == other {
                    return true;
                }
            }
        }
        false
    }

    /// Whether atom `i` carries a conjugated bond: an aromatic one, or one
    /// `markConjAtomBonds` marks from either end.
    fn has_conjugated_bond(&self, i: usize) -> bool {
        self.nbrs[i].iter().any(|&(j, _, aromatic)| {
            aromatic || self.marks_conjugated(i, j) || self.marks_conjugated(j, i)
        })
    }

    /// RDKit `numBondsPlusLonePairs` (no radicals: the octet and sub-octet
    /// branches agree).
    fn bonds_plus_lone_pairs(&self, i: usize) -> i32 {
        let deg = self.total_degree(i) as i32;
        if self.atno[i] <= 1 {
            return deg;
        }
        let Some(nouter) = nouter_elecs(self.atno[i]) else {
            return deg;
        };
        let free = nouter - (self.total_valence(i) + self.formal_charge[i]);
        deg + free / 2
    }

    /// RDKit `setHybridization` for one atom.
    fn hybridization(&self, i: usize) -> Hybridization {
        if self.atno[i] == 0 {
            return Hybridization::Other;
        }
        let norbs = if self.atno[i] < 89 {
            self.bonds_plus_lone_pairs(i)
        } else {
            self.total_degree(i) as i32
        };
        match norbs {
            0 | 1 => Hybridization::S,
            2 => Hybridization::Sp,
            3 => Hybridization::Sp2,
            4 if self.total_degree(i) > 3 || !self.has_conjugated_bond(i) => Hybridization::Sp3,
            4 => Hybridization::Sp2,
            5 => Hybridization::Sp3d,
            6 => Hybridization::Sp3d2,
            _ => Hybridization::Other,
        }
    }
}

/// Outer-shell electrons (RDKit `getNouterElecs`), main group through period 5.
fn nouter_elecs(atno: u8) -> Option<i32> {
    Some(match atno {
        1 | 3 | 11 | 19 | 37 | 55 => 1,
        2 | 4 | 12 | 20 | 38 | 56 => 2,
        5 | 13 | 31 | 49 => 3,
        6 | 14 | 32 | 50 => 4,
        7 | 15 | 33 | 51 => 5,
        8 | 16 | 34 | 52 => 6,
        9 | 17 | 35 | 53 => 7,
        10 | 18 | 36 | 54 => 8,
        _ => return None,
    })
}

/// The first standard valence (RDKit `getValenceList().front()`), `-1` when
/// the element has none.
fn min_valence(atno: u8) -> i32 {
    match atno {
        1 | 3 | 11 | 19 | 9 | 17 | 35 | 53 => 1,
        4 | 12 | 20 | 8 | 16 | 34 => 2,
        5 | 13 | 7 | 15 | 33 => 3,
        6 | 14 | 32 => 4,
        _ => -1,
    }
}

/// RDKit `getDefaultValence`; for these elements the first standard valence.
fn default_valence(atno: u8) -> i32 {
    min_valence(atno)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::system::BondType;

    fn by_element(mol: &Atomistic, sym: &str) -> Vec<Hybridization> {
        let hyb = hybridizations(mol);
        mol.atoms()
            .zip(hyb)
            .filter(|((_, a), _)| a.get_str("element") == Some(sym))
            .map(|(_, h)| h)
            .collect()
    }

    /// Ethane, ethene, ethyne: the textbook three, with hydrogens drawn.
    #[test]
    fn carbon_follows_its_bond_orders() {
        for (order, want) in [
            (BondType::Single, Hybridization::Sp3),
            (BondType::Double, Hybridization::Sp2),
            (BondType::Triple, Hybridization::Sp),
        ] {
            let mut mol = Atomistic::new();
            let c1 = mol.add_atom_bare("C");
            let c2 = mol.add_atom_bare("C");
            let b = mol.add_bond(c1, c2).unwrap();
            mol.set_bond_type(b, order).unwrap();
            assert_eq!(by_element(&mol, "C"), vec![want, want], "{order:?}");
        }
    }

    /// N-methylformamide's nitrogen has four orbitals but its lone pair is
    /// conjugated with the carbonyl, so it is sp²; the methyl carbon is not.
    #[test]
    fn an_amide_nitrogen_is_sp2_and_conjugated() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let o = mol.add_atom_bare("O");
        let n = mol.add_atom_bare("N");
        let me = mol.add_atom_bare("C");
        let co = mol.add_bond(c, o).unwrap();
        mol.set_bond_type(co, BondType::Double).unwrap();
        mol.add_bond(c, n).unwrap();
        mol.add_bond(n, me).unwrap();
        let hyb = hybridizations(&mol);
        let conj = conjugated_atoms(&mol);
        assert_eq!(hyb, vec![Sp2, Sp2, Sp2, Sp3]);
        assert_eq!(conj, vec![true, true, true, false]);
    }

    /// One molecule as RDKit read it: atoms `(element, formal charge)`,
    /// bonds `(i, j, Kekulé order, aromatic)`, and RDKit's answer for each
    /// heavy atom, in atom order.
    struct Fixture {
        name: &'static str,
        atoms: &'static [(&'static str, i32)],
        bonds: &'static [(usize, usize, u8, bool)],
        hybridization: &'static [Hybridization],
        conjugated: &'static [bool],
    }

    use Hybridization::{S, Sp, Sp2, Sp3, Sp3d2};

    // RDKit 2026.03.6: `Chem.AddHs(Chem.MolFromSmiles(smiles))`, Kekulized.
    #[rustfmt::skip]
    const FIXTURES: &[Fixture] = &[
        Fixture {
            name: "acetanilide",
            atoms: &[("C", 0), ("C", 0), ("O", 0), ("N", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, false), (1, 3, 1, false), (3, 4, 1, false), (4, 5, 2, true), (5, 6, 1, true), (6, 7, 2, true), (7, 8, 1, true), (8, 9, 2, true), (9, 4, 1, true), (0, 10, 1, false), (0, 11, 1, false), (0, 12, 1, false), (3, 13, 1, false), (5, 14, 1, false), (6, 15, 1, false), (7, 16, 1, false), (8, 17, 1, false), (9, 18, 1, false)],
            hybridization: &[Sp3, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[false, true, true, true, true, true, true, true, true, true],
        },
        Fixture {
            name: "methyl acetate",
            atoms: &[("C", 0), ("C", 0), ("O", 0), ("O", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, false), (1, 3, 1, false), (3, 4, 1, false), (0, 5, 1, false), (0, 6, 1, false), (0, 7, 1, false), (4, 8, 1, false), (4, 9, 1, false), (4, 10, 1, false)],
            hybridization: &[Sp3, Sp2, Sp2, Sp2, Sp3],
            conjugated: &[false, true, true, true, false],
        },
        Fixture {
            name: "pyrrole",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("N", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, true), (1, 2, 2, true), (2, 3, 1, true), (3, 4, 1, true), (4, 0, 2, true), (0, 5, 1, false), (1, 6, 1, false), (2, 7, 1, false), (3, 8, 1, false), (4, 9, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true],
        },
        Fixture {
            name: "pyridine",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("N", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 2, true), (1, 2, 1, true), (2, 3, 2, true), (3, 4, 1, true), (4, 5, 2, true), (5, 0, 1, true), (0, 6, 1, false), (1, 7, 1, false), (2, 8, 1, false), (4, 9, 1, false), (5, 10, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true, true],
        },
        Fixture {
            name: "furan",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("O", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, true), (1, 2, 2, true), (2, 3, 1, true), (3, 4, 1, true), (4, 0, 2, true), (0, 5, 1, false), (1, 6, 1, false), (2, 7, 1, false), (4, 8, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true],
        },
        Fixture {
            name: "thiophene",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("S", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, true), (1, 2, 2, true), (2, 3, 1, true), (3, 4, 1, true), (4, 0, 2, true), (0, 5, 1, false), (1, 6, 1, false), (2, 7, 1, false), (4, 8, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true],
        },
        Fixture {
            name: "dimethyl sulfoxide",
            atoms: &[("C", 0), ("S", 0), ("O", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, false), (1, 3, 1, false), (0, 4, 1, false), (0, 5, 1, false), (0, 6, 1, false), (3, 7, 1, false), (3, 8, 1, false), (3, 9, 1, false)],
            hybridization: &[Sp3, Sp3, Sp2, Sp3],
            conjugated: &[false, false, false, false],
        },
        Fixture {
            name: "acetonitrile",
            atoms: &[("C", 0), ("C", 0), ("N", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 3, false), (0, 3, 1, false), (0, 4, 1, false), (0, 5, 1, false)],
            hybridization: &[Sp3, Sp, Sp],
            conjugated: &[false, false, false],
        },
        Fixture {
            name: "allene",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 2, false), (1, 2, 2, false), (0, 3, 1, false), (0, 4, 1, false), (2, 5, 1, false), (2, 6, 1, false)],
            hybridization: &[Sp2, Sp, Sp2],
            conjugated: &[true, true, true],
        },
        Fixture {
            name: "nitrobenzene",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("N", 1), ("O", 0), ("O", -1), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, true), (1, 2, 2, true), (2, 3, 1, true), (3, 4, 2, true), (4, 5, 1, true), (5, 6, 1, false), (6, 7, 2, false), (6, 8, 1, false), (5, 0, 2, true), (0, 9, 1, false), (1, 10, 1, false), (2, 11, 1, false), (3, 12, 1, false), (4, 13, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true, true, true, true, true],
        },
        Fixture {
            name: "methanesulfonamide",
            atoms: &[("C", 0), ("S", 0), ("O", 0), ("O", 0), ("N", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, false), (1, 3, 2, false), (1, 4, 1, false), (0, 5, 1, false), (0, 6, 1, false), (0, 7, 1, false), (4, 8, 1, false), (4, 9, 1, false)],
            hybridization: &[Sp3, Sp3, Sp2, Sp2, Sp3],
            conjugated: &[false, false, false, false, false],
        },
        Fixture {
            name: "trimethylphosphine",
            atoms: &[("C", 0), ("P", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 1, false), (1, 3, 1, false), (0, 4, 1, false), (0, 5, 1, false), (0, 6, 1, false), (2, 7, 1, false), (2, 8, 1, false), (2, 9, 1, false), (3, 10, 1, false), (3, 11, 1, false), (3, 12, 1, false)],
            hybridization: &[Sp3, Sp3, Sp3, Sp3],
            conjugated: &[false, false, false, false],
        },
        Fixture {
            name: "phosphoric acid",
            atoms: &[("O", 0), ("P", 0), ("O", 0), ("O", 0), ("O", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, false), (1, 3, 1, false), (1, 4, 1, false), (0, 5, 1, false), (3, 6, 1, false), (4, 7, 1, false)],
            hybridization: &[Sp3, Sp3, Sp2, Sp3, Sp3],
            conjugated: &[false, false, false, false, false],
        },
        Fixture {
            name: "acetate",
            atoms: &[("C", 0), ("C", 0), ("O", 0), ("O", -1), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, false), (1, 3, 1, false), (0, 4, 1, false), (0, 5, 1, false), (0, 6, 1, false)],
            hybridization: &[Sp3, Sp2, Sp2, Sp2],
            conjugated: &[false, true, true, true],
        },
        Fixture {
            name: "methylammonium",
            atoms: &[("C", 0), ("N", 1), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (0, 2, 1, false), (0, 3, 1, false), (0, 4, 1, false), (1, 5, 1, false), (1, 6, 1, false), (1, 7, 1, false)],
            hybridization: &[Sp3, Sp3],
            conjugated: &[false, false],
        },
        Fixture {
            name: "sulfur hexafluoride",
            atoms: &[("F", 0), ("S", 0), ("F", 0), ("F", 0), ("F", 0), ("F", 0), ("F", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 1, false), (1, 3, 1, false), (1, 4, 1, false), (1, 5, 1, false), (1, 6, 1, false)],
            hybridization: &[Sp3, Sp3d2, Sp3, Sp3, Sp3, Sp3, Sp3],
            conjugated: &[false, false, false, false, false, false, false],
        },
        Fixture {
            name: "butadiene",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 2, false), (1, 2, 1, false), (2, 3, 2, false), (0, 4, 1, false), (0, 5, 1, false), (1, 6, 1, false), (2, 7, 1, false), (3, 8, 1, false), (3, 9, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true],
        },
        Fixture {
            name: "ethanol",
            atoms: &[("C", 0), ("C", 0), ("O", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 1, false), (0, 3, 1, false), (0, 4, 1, false), (0, 5, 1, false), (1, 6, 1, false), (1, 7, 1, false), (2, 8, 1, false)],
            hybridization: &[Sp3, Sp3, Sp3],
            conjugated: &[false, false, false],
        },
        Fixture {
            name: "imidazole",
            atoms: &[("C", 0), ("C", 0), ("N", 0), ("C", 0), ("N", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 2, true), (1, 2, 1, true), (2, 3, 2, true), (3, 4, 1, true), (4, 0, 1, true), (0, 5, 1, false), (1, 6, 1, false), (3, 7, 1, false), (4, 8, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true],
        },
        Fixture {
            name: "acrolein",
            atoms: &[("C", 0), ("C", 0), ("C", 0), ("O", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 2, false), (1, 2, 1, false), (2, 3, 2, false), (0, 4, 1, false), (0, 5, 1, false), (1, 6, 1, false), (2, 7, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true],
        },
        Fixture {
            name: "aniline",
            atoms: &[("N", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, true), (2, 3, 1, true), (3, 4, 2, true), (4, 5, 1, true), (5, 6, 2, true), (6, 1, 1, true), (0, 7, 1, false), (0, 8, 1, false), (2, 9, 1, false), (3, 10, 1, false), (4, 11, 1, false), (5, 12, 1, false), (6, 13, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true, true, true],
        },
        Fixture {
            name: "vinyl ether",
            atoms: &[("C", 0), ("C", 0), ("O", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 2, false), (1, 2, 1, false), (2, 3, 1, false), (0, 4, 1, false), (0, 5, 1, false), (1, 6, 1, false), (3, 7, 1, false), (3, 8, 1, false), (3, 9, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp3],
            conjugated: &[true, true, true, false],
        },
        Fixture {
            name: "urea",
            atoms: &[("N", 0), ("C", 0), ("N", 0), ("O", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 1, false), (1, 3, 2, false), (0, 4, 1, false), (0, 5, 1, false), (2, 6, 1, false), (2, 7, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true],
        },
        Fixture {
            name: "guanidinium",
            atoms: &[("N", 0), ("C", 0), ("N", 0), ("N", 1), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 1, false), (1, 3, 2, false), (0, 4, 1, false), (0, 5, 1, false), (2, 6, 1, false), (2, 7, 1, false), (3, 8, 1, false), (3, 9, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true],
        },
        Fixture {
            name: "phenol",
            atoms: &[("O", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 2, true), (2, 3, 1, true), (3, 4, 2, true), (4, 5, 1, true), (5, 6, 2, true), (6, 1, 1, true), (0, 7, 1, false), (2, 8, 1, false), (3, 9, 1, false), (4, 10, 1, false), (5, 11, 1, false), (6, 12, 1, false)],
            hybridization: &[Sp2, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true, true, true],
        },
        Fixture {
            name: "benzonitrile",
            atoms: &[("N", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("C", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 3, false), (1, 2, 1, false), (2, 3, 2, true), (3, 4, 1, true), (4, 5, 2, true), (5, 6, 1, true), (6, 7, 2, true), (7, 2, 1, true), (3, 8, 1, false), (4, 9, 1, false), (5, 10, 1, false), (6, 11, 1, false), (7, 12, 1, false)],
            hybridization: &[Sp, Sp, Sp2, Sp2, Sp2, Sp2, Sp2, Sp2],
            conjugated: &[true, true, true, true, true, true, true, true],
        },
        Fixture {
            name: "thioacetamide",
            atoms: &[("C", 0), ("C", 0), ("N", 0), ("S", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0), ("H", 0)],
            bonds: &[(0, 1, 1, false), (1, 2, 1, false), (1, 3, 2, false), (0, 4, 1, false), (0, 5, 1, false), (0, 6, 1, false), (2, 7, 1, false), (2, 8, 1, false)],
            hybridization: &[Sp3, Sp2, Sp2, Sp2],
            conjugated: &[false, true, true, true],
        },
    ];

    /// Build a fixture, with its hydrogens drawn or left implicit.
    fn build(f: &Fixture, with_hydrogens: bool) -> Atomistic {
        let mut mol = Atomistic::new();
        let mut ids = Vec::new();
        for &(sym, charge) in f.atoms {
            if !with_hydrogens && sym == "H" {
                ids.push(None);
                continue;
            }
            let id = mol.add_atom_bare(sym);
            if charge != 0 {
                mol.set_atom(id, "formal_charge", charge).unwrap();
            }
            ids.push(Some(id));
        }
        for &(i, j, order, aromatic) in f.bonds {
            let (Some(a), Some(b)) = (ids[i], ids[j]) else {
                continue;
            };
            let bid = mol.add_bond(a, b).unwrap();
            let number = match order {
                1 => BondNumber::Single,
                2 => BondNumber::Double,
                _ => BondNumber::Triple,
            };
            let class = if aromatic {
                BondType::Aromatic
            } else {
                match order {
                    1 => BondType::Single,
                    2 => BondType::Double,
                    _ => BondType::Triple,
                }
            };
            mol.set_bond_class(bid, class, number).unwrap();
        }
        mol
    }

    /// Every heavy atom's hybridization and conjugation is RDKit's — with the
    /// hydrogens drawn, and with them left for the valence model to imply.
    #[test]
    fn heavy_atoms_hybridize_and_conjugate_as_rdkit_says() {
        for f in FIXTURES {
            for with_hydrogens in [true, false] {
                let mol = build(f, with_hydrogens);
                let heavy: Vec<usize> = mol
                    .atoms()
                    .enumerate()
                    .filter(|(_, (_, a))| a.get_str("element") != Some("H"))
                    .map(|(i, _)| i)
                    .collect();
                let hyb = hybridizations(&mol);
                let conj = conjugated_atoms(&mol);
                let got_h: Vec<_> = heavy.iter().map(|&i| hyb[i]).collect();
                let got_c: Vec<_> = heavy.iter().map(|&i| conj[i]).collect();
                assert_eq!(
                    got_h, f.hybridization,
                    "{} (H drawn: {with_hydrogens})",
                    f.name
                );
                assert_eq!(
                    got_c, f.conjugated,
                    "{} (H drawn: {with_hydrogens})",
                    f.name
                );
            }
        }
    }

    /// A lone ion and dihydrogen: no orbitals to count past one.
    #[test]
    fn a_bare_ion_and_dihydrogen_are_s() {
        let mut mol = Atomistic::new();
        let na = mol.add_atom_bare("Na");
        mol.set_atom(na, "formal_charge", 1).unwrap();
        let h1 = mol.add_atom_bare("H");
        let h2 = mol.add_atom_bare("H");
        mol.add_bond(h1, h2).unwrap();
        assert_eq!(hybridizations(&mol), vec![S, S, S]);
    }
}
