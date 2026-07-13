//! The molecule as the ATD rule language sees it.
//!
//! Every constraint an [`AtdRule`](crate::ff::params::AtdRule) can state is a
//! question about counts — atomic number, degree, attached hydrogens, ring
//! membership, how many single / double / aromatic / delocalized bonds an atom
//! carries. [`MolFacts`] answers all of them from **one** pass over the graph,
//! so a table with 200 rules costs one traversal, not 200.
//!
//! The facts are table-independent: `sb`/`db`/`ab`/`DL` mean the same thing in
//! `ATOMTYPE_BCC.DEF` and in `ATOMTYPE_GAS.DEF`. That is what lets one engine
//! walk every table.

use std::collections::{HashMap, HashSet};

use molrs::perceive::bond_type::BCC_BOND_TYPE;
use molrs::store::keys;
use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic, Bond, BondId, Element, find_rings};

use crate::ff::params::AtomProp;

/// Pre-computed answers to every question an ATD rule can ask about an atom.
///
/// All vectors are indexed by the atom's position in `mol.atoms()` order;
/// [`MolFacts::index_of`] maps an [`AtomId`] onto that position.
#[derive(Debug, Clone)]
pub(super) struct MolFacts {
    /// Atom id -> row index into every vector below.
    pub(super) index: HashMap<AtomId, usize>,
    /// Atomic number.
    pub(super) atomic_number: Vec<u8>,
    /// Number of bonded neighbours.
    pub(super) degree: Vec<usize>,
    /// Number of bonded hydrogens.
    pub(super) hydrogen_count: Vec<usize>,
    /// Whether the atom is electron-withdrawing (`EW`).
    pub(super) ewd: Vec<bool>,
    /// Residue name, or `*` when the graph carries none.
    pub(super) residue: Vec<String>,
    /// Ring / aromaticity / bond-order counts.
    pub(super) props: Vec<AtomPropertyFacts>,
    /// Neighbours as `(atom, antechamber bond type, bond)`.
    pub(super) neighbors: Vec<Vec<(AtomId, i32, BondId)>>,
}

impl MolFacts {
    /// Derive the facts of `mol`, whose bonds must already carry perceived
    /// antechamber bond types under [`BCC_BOND_TYPE`] (see
    /// [`Perceive::find_bond_types`](molrs::perceive::Perceive::find_bond_types)).
    pub(super) fn new(mol: &Atomistic) -> Result<Self, String> {
        let atom_ids: Vec<_> = mol.atoms().map(|(aid, _)| aid).collect();
        let index: HashMap<AtomId, usize> = atom_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, aid)| (aid, i))
            .collect();
        let mut atomic_number = Vec::with_capacity(atom_ids.len());
        let mut residue = Vec::with_capacity(atom_ids.len());
        for aid in &atom_ids {
            let atom = mol.get_atom(*aid).map_err(|e| e.to_string())?;
            let symbol = atom.get_str(keys::ELEMENT).ok_or_else(|| {
                format!("ATD atom typing requires `{}` for {aid:?}", keys::ELEMENT)
            })?;
            let element = Element::by_symbol(symbol)
                .ok_or_else(|| format!("unsupported element symbol `{symbol}` for {aid:?}"))?;
            atomic_number.push(element.z());
            residue.push(atom.get_str(keys::RES_NAME).unwrap_or("*").to_owned());
        }

        let mut neighbors = vec![Vec::new(); atom_ids.len()];
        for (bid, bond) in mol.bonds() {
            let bond_type = antechamber_bond_type(&bond)?;
            let a = bond.nodes[0];
            let b = bond.nodes[1];
            let ia = index[&a];
            let ib = index[&b];
            neighbors[ia].push((b, bond_type, bid));
            neighbors[ib].push((a, bond_type, bid));
        }
        let degree: Vec<usize> = neighbors.iter().map(Vec::len).collect();
        let mut hydrogen_count = vec![0; atom_ids.len()];
        for (i, nbs) in neighbors.iter().enumerate() {
            hydrogen_count[i] = nbs
                .iter()
                .filter(|(nb, _, _)| atomic_number[index[nb]] == 1)
                .count();
        }
        let ewd: Vec<bool> = atomic_number
            .iter()
            .map(|z| matches!(*z, 7 | 8 | 9 | 16 | 17 | 35 | 53))
            .collect();

        let ring_info = find_rings(mol);
        let mut props = vec![AtomPropertyFacts::default(); atom_ids.len()];
        for ring in ring_info.rings() {
            let size = ring.len();
            let class = classify_ring(ring, &index, &atomic_number, &degree, &neighbors)?;
            for aid in ring {
                let p = &mut props[index[aid]];
                p.rg[0] += 1;
                if size < p.rg.len() {
                    p.rg[size] += 1;
                }
                match class {
                    RingClass::Ar1 => p.ar1 += 1,
                    RingClass::Ar2 => p.ar2 += 1,
                    RingClass::Ar3 => p.ar3 += 1,
                    RingClass::Ar4 => p.ar4 += 1,
                    RingClass::Ar5 => p.ar5 += 1,
                }
            }
        }
        for p in &mut props {
            p.nr = usize::from(p.rg[0] == 0);
        }
        for (i, nbs) in neighbors.iter().enumerate() {
            for (_, bond_type, _) in nbs {
                props[i].add_bond_type(*bond_type);
            }
        }

        Ok(Self {
            index,
            atomic_number,
            degree,
            hydrogen_count,
            ewd,
            residue,
            props,
            neighbors,
        })
    }

    /// The row index of `aid`.
    pub(super) fn index_of(&self, aid: AtomId) -> Result<usize, String> {
        self.index
            .get(&aid)
            .copied()
            .ok_or_else(|| format!("unknown atom id {aid:?}"))
    }

    /// Electron-withdrawing atoms around the atom `aid` hangs off.
    ///
    /// The `.DEF` column counts the EW neighbours of the *attachment point*, not
    /// of the candidate itself — that is how a hydrogen learns about the
    /// substituents of the carbon it sits on.
    pub(super) fn ewd_count_around_attachment(&self, aid: AtomId) -> Option<usize> {
        let i = self.index_of(aid).ok()?;
        let attached = self.neighbors[i].first()?.0;
        let j = self.index_of(attached).ok()?;
        Some(
            self.neighbors[j]
                .iter()
                .filter(|(nb, _, _)| self.ewd[self.index[nb]])
                .count(),
        )
    }
}

/// Ring, aromaticity and bond-order counts of a single atom.
///
/// The lowercase source tokens (`sb`, `db`, `tb`) count aromatic and delocalized
/// bonds too; the uppercase ones (`SB`, `DB`, `TB`) are strict. Both are counted
/// here, and [`AtomPropertyFacts::count`] hands out whichever the rule asked for.
#[derive(Debug, Clone, Default)]
pub(super) struct AtomPropertyFacts {
    /// `rg[0]` = rings of any size; `rg[n]` = rings of size `n`.
    rg: [usize; 12],
    /// `NR` — in no ring.
    nr: usize,
    /// `AR1` — rings of this atom that are pure aromatic (benzene, pyridine).
    ar1: usize,
    /// `AR2` — planar rings of this atom with two continuous single bonds and at
    /// least two double bonds (imidazole, thiophene, pyrrole).
    ar2: usize,
    /// `AR3` — planar rings of this atom whose double bonds are formed between
    /// ring atoms and non-ring atoms (a quinone, a pyridone).
    ar3: usize,
    /// `AR4` — rings of this atom that are none of AR1, AR2, AR3 or AR5.
    ar4: usize,
    /// `AR5` — pure aliphatic rings of this atom, made of sp3 carbon.
    ar5: usize,
    /// `sb` — single bonds, aromatic and delocalized ones included.
    sb: usize,
    /// `SB` — single bonds, strictly.
    sb_strict: usize,
    /// `db` — double bonds, aromatic and delocalized ones included.
    db: usize,
    /// `DB` — double bonds, strictly.
    db_strict: usize,
    /// `tb` — triple bonds, aromatic ones included.
    tb: usize,
    /// `TB` — triple bonds, strictly.
    tb_strict: usize,
    /// `AB` — aromatic bonds.
    ab: usize,
    /// `DL` — delocalized bonds.
    dl: usize,
}

impl AtomPropertyFacts {
    /// Fold one incident bond into the counts.
    fn add_bond_type(&mut self, bond_type: i32) {
        match bond_type {
            1 => {
                self.sb += 1;
                self.sb_strict += 1;
            }
            2 => {
                self.db += 1;
                self.db_strict += 1;
            }
            3 => {
                self.tb += 1;
                self.tb_strict += 1;
            }
            7 => {
                self.ab += 1;
                self.sb += 1;
            }
            8 => {
                self.ab += 1;
                self.db += 1;
            }
            9 => {
                self.sb += 1;
                self.sb_strict += 1;
                self.dl += 1;
            }
            10 => {
                self.ab += 1;
            }
            _ => {}
        }
    }

    /// How many times this atom satisfies `prop`.
    pub(super) fn count(&self, prop: AtomProp) -> usize {
        match prop {
            AtomProp::Rg => self.rg[0],
            AtomProp::Rg3 => self.rg[3],
            AtomProp::Rg4 => self.rg[4],
            AtomProp::Rg5 => self.rg[5],
            AtomProp::Rg6 => self.rg[6],
            AtomProp::Rg7 => self.rg[7],
            AtomProp::Rg8 => self.rg[8],
            AtomProp::Rg9 => self.rg[9],
            AtomProp::Rg10 => self.rg[10],
            AtomProp::Nr => self.nr,
            AtomProp::Ar1 => self.ar1,
            AtomProp::Ar2 => self.ar2,
            AtomProp::Ar3 => self.ar3,
            AtomProp::Ar4 => self.ar4,
            AtomProp::Ar5 => self.ar5,
            AtomProp::SbStrict => self.sb_strict,
            AtomProp::SbAny => self.sb,
            AtomProp::DbStrict => self.db_strict,
            AtomProp::DbAny => self.db,
            AtomProp::TbStrict => self.tb_strict,
            AtomProp::TbAny => self.tb,
            AtomProp::Ab => self.ab,
            AtomProp::Dl => self.dl,
        }
    }
}

/// The class of a whole ring, as `ATOMTYPE_*.DEF`'s own header defines it.
///
/// The header (`AR1` … `AR5`, verbatim) is the specification:
///
/// * **AR1** — "Pure aromatic atom (such as benzene and pyridine)".
/// * **AR2** — "Atom in a planar ring, usually the ring has two continuous
///   single bonds and at least two double bonds".
/// * **AR3** — "Atom in a planar ring, which has one or several double bonds
///   formed between non-ring atoms and the ring atoms".
/// * **AR4** — "Atom other than AR1, AR2, AR3 and AR5".
/// * **AR5** — "Pure aliphatic atom in a ring, which is made of sp3 carbon".
///
/// The property is a fact about the **ring**, not about the atom: every atom of
/// the ring gets the ring's class, and an atom in two rings is counted once per
/// ring (`[2AR1]` is a rule a table may legitimately write).
///
/// The AR1/AR2 boundary is what separates benzene from imidazole, and it is the
/// only place the five-membered heteroaromatics are visible: `ATOMTYPE_GFF.DEF`
/// spells `ca` as `[AR1]` and `cc` as `[sb,db,AR2]`, so calling a pyrrole-type
/// ring AR1 types thiophene's carbons `ca` where antechamber says `cc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RingClass {
    Ar1,
    Ar2,
    Ar3,
    Ar4,
    Ar5,
}

/// Which class `ring` belongs to.
///
/// `ring` is a closed path (consecutive atoms are bonded, and the last is bonded
/// back to the first), which is what [`find_rings`] returns.
fn classify_ring(
    ring: &[AtomId],
    index: &HashMap<AtomId, usize>,
    atomic_number: &[u8],
    degree: &[usize],
    neighbors: &[Vec<(AtomId, i32, BondId)>],
) -> Result<RingClass, String> {
    let n = ring.len();
    let members: HashSet<AtomId> = ring.iter().copied().collect();

    // The bond joining ring[k] to ring[k + 1], wrapping at the end.
    let mut ring_bonds = Vec::with_capacity(n);
    for k in 0..n {
        let a = ring[k];
        let b = ring[(k + 1) % n];
        let ia = *index
            .get(&a)
            .ok_or_else(|| format!("ring atom {a:?} is not in the molecule"))?;
        let bond_type = neighbors[ia]
            .iter()
            .find(|(nb, _, _)| *nb == b)
            .map(|(_, bond_type, _)| *bond_type)
            .ok_or_else(|| format!("ring path is not closed: {a:?} is not bonded to {b:?}"))?;
        ring_bonds.push(bond_type);
    }

    // AR1 — a *pure* aromatic ring: every bond aromatic, and every atom carrying
    // one of the ring's double bonds. Benzene and pyridine alternate perfectly;
    // imidazole's pyrrole-type N sits between two aromatic *single* bonds, which
    // is precisely the "two continuous single bonds" AR2 names. An odd-membered
    // ring can never alternate, so a 5-ring is never AR1 — as intended.
    let aromatic_ring = ring_bonds.iter().all(|t| is_aromatic_bond_type(*t));
    let mut carries_ring_double = vec![false; n];
    for (k, bond_type) in ring_bonds.iter().enumerate() {
        if is_double_bond_type(*bond_type) {
            carries_ring_double[k] = true;
            carries_ring_double[(k + 1) % n] = true;
        }
    }
    if aromatic_ring && carries_ring_double.iter().all(|carries| *carries) {
        return Ok(RingClass::Ar1);
    }

    let planar = ring
        .iter()
        .all(|aid| is_planar_ring_atom(*aid, index, atomic_number, neighbors));
    if planar {
        // AR3 before AR2: the header separates them by whether the ring's double
        // bonds point *out* of the ring (a quinone, a pyridone), and a ring that
        // has exocyclic double bonds usually has continuous single bonds too. The
        // seven tables mirror every AR2 rule with an identical AR3 one, so this
        // order is not observable through them — but the header's wording is, and
        // it is what a future table would be written against.
        let exocyclic_double = ring.iter().any(|aid| {
            let i = index[aid];
            neighbors[i]
                .iter()
                .any(|(nb, bond_type, _)| !members.contains(nb) && is_double_bond_type(*bond_type))
        });
        return Ok(if exocyclic_double {
            RingClass::Ar3
        } else {
            RingClass::Ar2
        });
    }

    // AR5 — "pure aliphatic ... made of sp3 carbon": cyclohexane, cyclopropane.
    let aliphatic = ring.iter().all(|aid| {
        let i = index[aid];
        atomic_number[i] == 6 && degree[i] == 4
    });
    Ok(if aliphatic {
        RingClass::Ar5
    } else {
        RingClass::Ar4
    })
}

/// Is this ring atom planar — i.e. does it keep the ring flat?
///
/// Either it is sp2 itself (it carries an aromatic or double bond, whether
/// inside the ring or hanging off it), or it is a heteroatom donating a lone
/// pair into a neighbouring sp2 centre — the `N` of pyrrole, the `O` of furan,
/// the `S` of thiophene, none of which carry a double bond of their own.
///
/// An sp3 carbon is neither, which is what keeps ethylene carbonate's ring
/// (`-O-CH2-CH2-O-`) out of AR2 / AR3 despite its exocyclic `C=O`.
fn is_planar_ring_atom(
    aid: AtomId,
    index: &HashMap<AtomId, usize>,
    atomic_number: &[u8],
    neighbors: &[Vec<(AtomId, i32, BondId)>],
) -> bool {
    let i = index[&aid];
    if is_sp2(i, neighbors) {
        return true;
    }
    let lone_pair_donor = matches!(atomic_number[i], 7 | 8 | 15 | 16);
    lone_pair_donor
        && neighbors[i]
            .iter()
            .any(|(nb, _, _)| is_sp2(index[nb], neighbors))
}

/// Does the atom at row `i` carry any bond that pins it into a plane?
fn is_sp2(i: usize, neighbors: &[Vec<(AtomId, i32, BondId)>]) -> bool {
    neighbors[i].iter().any(|(_, bond_type, _)| {
        is_double_bond_type(*bond_type) || is_aromatic_bond_type(*bond_type)
    })
}

/// Antechamber's aromatic bond types: 7 (aromatic single), 8 (aromatic double),
/// 10 (the unresolved aromatic precursor).
fn is_aromatic_bond_type(bond_type: i32) -> bool {
    matches!(bond_type, 7 | 8 | 10)
}

/// A bond that makes both endpoints sp2: a plain double (2) or an aromatic
/// double (8). Aromatic singles (7) and delocalized bonds (9) are not doubles —
/// that distinction is exactly what the AR1/AR2 boundary rests on.
fn is_double_bond_type(bond_type: i32) -> bool {
    matches!(bond_type, 2 | 8)
}

/// The antechamber bond type a bond carries: 1 single, 2 double, 3 triple,
/// 7/8/10 aromatic, 9 delocalized.
///
/// Read from [`BCC_BOND_TYPE`] — the key
/// [`Perceive::find_bond_types`](molrs::perceive::Perceive::find_bond_types) writes
/// it to — and **never** from the bond's `type`, which is the caller's and holds
/// their force-field bond-type *name*.
///
/// Shared with the BCC corrector, whose correction rows are keyed on the same
/// integer. A bond without one is an error, never a guessed single bond.
pub(crate) fn antechamber_bond_type(bond: &Bond) -> Result<i32, String> {
    match bond.props.get(BCC_BOND_TYPE) {
        Some(PropValue::Int(v)) => Ok(*v),
        Some(PropValue::F64(v)) if (*v - v.round()).abs() < 1.0e-6 => Ok(v.round() as i32),
        Some(PropValue::F64(v)) => Err(format!("`{BCC_BOND_TYPE}` must be integral, got {v}")),
        Some(PropValue::Str(s)) => s
            .parse::<i32>()
            .map_err(|e| format!("`{BCC_BOND_TYPE}` must be an integer string: {e}")),
        Some(PropValue::Bool(_)) => Err(format!("`{BCC_BOND_TYPE}` must be numeric, not bool")),
        None => Err(format!(
            "BCC correction requires a perceived `{BCC_BOND_TYPE}` on every bond"
        )),
    }
}
