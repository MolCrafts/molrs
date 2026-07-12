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

use std::collections::HashMap;

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
    /// Derive the facts of `mol`, whose bonds must already carry antechamber
    /// bond `type`s (see
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
            for aid in ring {
                let p = &mut props[index[aid]];
                p.rg[0] += 1;
                if size < p.rg.len() {
                    p.rg[size] += 1;
                }
            }
        }
        for (i, aid) in atom_ids.iter().copied().enumerate() {
            props[i].nr = usize::from(props[i].rg[0] == 0);
            if is_aromatic_atom(mol, aid)? {
                props[i].ar1 = 1;
            } else if props[i].rg[0] > 0 {
                props[i].ar5 = 1;
            }
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
    /// `AR1` — pure aromatic ring.
    ar1: usize,
    /// `AR2` — planar ring fused to a pure aromatic ring.
    ar2: usize,
    /// `AR3` — planar ring, otherwise unclassified.
    ar3: usize,
    /// `AR4` — ring with sp3 atoms.
    ar4: usize,
    /// `AR5` — non-planar ring.
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

/// Is `aid` aromatic — by its own flag, or by any bond it carries?
fn is_aromatic_atom(mol: &Atomistic, aid: AtomId) -> Result<bool, String> {
    let atom = mol.get_atom(aid).map_err(|e| e.to_string())?;
    if prop_truthy(atom.get("is_aromatic")) {
        return Ok(true);
    }
    for (bid, _) in mol.incident_bond_ids(aid) {
        let bond = mol.get_bond(bid).map_err(|e| e.to_string())?;
        if is_aromatic_bond(&bond) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Is this bond aromatic — by flag, by order 1.5, or by antechamber type 7/8/10?
fn is_aromatic_bond(bond: &Bond) -> bool {
    prop_truthy(bond.props.get("is_aromatic"))
        || bond
            .props
            .get(keys::ORDER)
            .and_then(PropValue::as_f64)
            .is_some_and(|order| (order - 1.5).abs() < 1.0e-6)
        || bond
            .props
            .get(keys::TYPE)
            .and_then(|v| match v {
                PropValue::Int(v) => Some(*v),
                PropValue::F64(v) if (*v - v.round()).abs() < 1.0e-6 => Some(v.round() as i32),
                PropValue::Str(s) => s.parse::<i32>().ok(),
                PropValue::F64(_) | PropValue::Bool(_) => None,
            })
            .is_some_and(|t| matches!(t, 7 | 8 | 10))
}

/// Read a truthy graph property, whatever type it was stored as.
fn prop_truthy(prop: Option<&PropValue>) -> bool {
    match prop {
        Some(PropValue::Bool(v)) => *v,
        Some(PropValue::Int(v)) => *v != 0,
        Some(PropValue::F64(v)) => *v != 0.0,
        Some(PropValue::Str(v)) => matches!(v.as_str(), "1" | "true" | "True" | "TRUE" | "ar"),
        None => false,
    }
}

/// The antechamber bond type a bond carries: 1 single, 2 double, 3 triple,
/// 7/8/10 aromatic, 9 delocalized.
///
/// Shared with the BCC corrector, whose correction rows are keyed on the same
/// integer. A bond without one is an error, never a guessed single bond.
pub(crate) fn antechamber_bond_type(bond: &Bond) -> Result<i32, String> {
    match bond.props.get(keys::TYPE) {
        Some(PropValue::Int(v)) => Ok(*v),
        Some(PropValue::F64(v)) if (*v - v.round()).abs() < 1.0e-6 => Ok(v.round() as i32),
        Some(PropValue::F64(v)) => Err(format!("BCC bond `type` must be integral, got {v}")),
        Some(PropValue::Str(s)) => s
            .parse::<i32>()
            .map_err(|e| format!("BCC bond `type` must be an integer string: {e}")),
        Some(PropValue::Bool(_)) => Err("BCC bond `type` must be numeric, not bool".to_owned()),
        None => Err("BCC correction requires BCC bond `type`".to_owned()),
    }
}
