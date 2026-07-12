//! AM1-BCC charge typifiers.
//!
//! The public shape is intentionally parallel to `MMFFTypifier` and
//! `OPLSAATypifier`: a struct owns the model/backend state, `typify()` returns
//! a labeled `Atomistic`, and `charge` is written as a canonical atom property.
//!
//! The implementation is a native port surface: external programs are reference
//! material only, not runtime dependencies. Missing AM1 backends, BCC atom
//! types, bond types, or correction rows return errors.

use molrs::perceive::Perceive;
use molrs::store::keys;
use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic, Bond, BondId, Element, find_rings};
use std::collections::HashMap;

use super::Typifier;
use crate::ff::params::{
    AtdRule, AtdTable, AtomPattern, AtomProp, BccAlias, BccCorrectionRow, EnvBond, EnvBondType,
    PatternAtom, PropExpr, PropRelation, PropUnit,
};

/// AM1 base-charge result supplied by an AM1 backend.
#[derive(Debug, Clone, PartialEq)]
pub struct AM1ChargeResult {
    pub charges: Vec<f64>,
    pub total_charge: Option<f64>,
    pub heat_of_formation_kcal_mol: Option<f64>,
    pub reference: String,
}

impl AM1ChargeResult {
    pub fn new(charges: Vec<f64>) -> Self {
        Self {
            charges,
            total_charge: None,
            heat_of_formation_kcal_mol: None,
            reference: String::new(),
        }
    }
}

/// AM1-BCC correction-family selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BccParameterSet {
    /// Original AM1-BCC corrections (`BCCPARM.DAT`, `-c bcc`).
    Bcc,
    /// ABCG2 corrections (`BCCPARM_ABCG2.DAT`, `-c abcg2`).
    Abcg2,
}

impl BccParameterSet {
    /// The set's bond charge corrections, as compile-time table data.
    fn corrections(self) -> &'static [BccCorrectionRow] {
        match self {
            Self::Bcc => crate::ff::params::BCC_CORRECTIONS,
            Self::Abcg2 => crate::ff::params::ABCG2_CORRECTIONS,
        }
    }

    /// The set's `CORR` alias rows.
    fn aliases(self) -> &'static [BccAlias] {
        match self {
            Self::Bcc => crate::ff::params::BCC_ALIASES,
            Self::Abcg2 => crate::ff::params::ABCG2_ALIASES,
        }
    }

    /// The set's atom-type definition rules.
    fn atd_table(self) -> AtdTable {
        match self {
            Self::Bcc => crate::ff::params::ATOMTYPE_BCC,
            Self::Abcg2 => crate::ff::params::ATOMTYPE_ABCG2,
        }
    }
}

/// Backend trait for obtaining AM1 base charges.
pub trait AM1ChargeBackend {
    fn compute_am1_charges(&self, mol: &Atomistic) -> Result<AM1ChargeResult, String>;
}

/// Guarded default backend used until Atomiverse's AM1 solver is linked in.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableAM1Backend;

impl AM1ChargeBackend for UnavailableAM1Backend {
    fn compute_am1_charges(&self, _mol: &Atomistic) -> Result<AM1ChargeResult, String> {
        Err("AM1-BCC requires an AM1ChargeBackend backed by Atomiverse AM1; no backend is configured".to_owned())
    }
}

/// AM1 base-charge typifier without bond charge corrections.
#[derive(Debug, Clone)]
pub struct AM1ChargeTypifier<B = UnavailableAM1Backend> {
    backend: B,
    total_charge: Option<f64>,
}

impl Default for AM1ChargeTypifier<UnavailableAM1Backend> {
    fn default() -> Self {
        Self::new(UnavailableAM1Backend)
    }
}

impl<B> AM1ChargeTypifier<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            total_charge: None,
        }
    }

    pub fn with_total_charge(mut self, total_charge: f64) -> Self {
        self.total_charge = Some(total_charge);
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B> Typifier for AM1ChargeTypifier<B>
where
    B: AM1ChargeBackend,
{
    type Mol = Atomistic;

    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let am1 = self.backend.compute_am1_charges(mol)?;
        if am1.charges.len() != mol.n_atoms() {
            return Err(format!(
                "AM1 backend returned {} charges for {} atoms",
                am1.charges.len(),
                mol.n_atoms()
            ));
        }

        let mut out = mol.clone();
        let atom_ids: Vec<AtomId> = out.atoms().map(|(id, _)| id).collect();
        for (i, aid) in atom_ids.iter().copied().enumerate() {
            out.set_atom(aid, keys::CHARGE, am1.charges[i])
                .map_err(|e| e.to_string())?;
        }
        if let Some(target) = self.total_charge {
            normalize_total_charge(&mut out, target)?;
        }
        Ok(out)
    }
}

/// Graph-based BCC atom/bond typifier.
#[derive(Debug, Clone)]
pub struct BCCAtomTypifier {
    model: BccParameterSet,
}

impl Default for BCCAtomTypifier {
    fn default() -> Self {
        Self::bcc()
    }
}

impl BCCAtomTypifier {
    pub fn parameter_set(model: BccParameterSet) -> Self {
        Self { model }
    }

    pub fn bcc() -> Self {
        Self::parameter_set(BccParameterSet::Bcc)
    }

    pub fn abcg2() -> Self {
        Self::parameter_set(BccParameterSet::Abcg2)
    }

    /// Perceive BCC bond types, then label every atom from `ATOMTYPE_BCC.DEF`.
    ///
    /// The bond types are always **perceived** (via
    /// [`Perceive::find_bond_types`](molrs::perceive::Perceive::find_bond_types)),
    /// never read off the input: the atom-type rules count `sb`/`db`/`ab`/`DL`
    /// bonds, so they need the delocalized (9) and aromatic (7/8) types that a bond
    /// *order* cannot express — and an input `type` may be the unresolved aromatic
    /// precursor (10), which must be resolved, not trusted. To apply corrections
    /// with types of your own, drive [`BCCCorrector`] directly.
    pub fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let table = self.model.atd_table();
        let mut out = Perceive::new().find_bond_types(mol);

        let facts = BCCMolFacts::new(&out)?;
        let atom_ids: Vec<AtomId> = out.atoms().map(|(aid, _)| aid).collect();
        for aid in atom_ids {
            let atom_type = assign_atom_type(&table, aid, &facts)
                .ok_or_else(|| format!("missing BCC atom type for {aid:?}"))?;
            out.set_atom(aid, keys::TYPE, atom_type)
                .map_err(|e| e.to_string())?;
        }
        Ok(out)
    }
}

/// Assign the first [`AtdRule`] of `table` that matches `aid`.
///
/// Rule order is the `.DEF` file's order and is load-bearing: the table is
/// written most-specific-first, so the first match wins.
fn assign_atom_type(table: &AtdTable, aid: AtomId, facts: &BCCMolFacts) -> Option<&'static str> {
    table
        .rules
        .iter()
        .find(|rule| rule_matches(rule, aid, facts))
        .map(|rule| rule.atom_type)
}

/// Test one pre-parsed `ATD` rule against an atom.
fn rule_matches(rule: &AtdRule, aid: AtomId, facts: &BCCMolFacts) -> bool {
    let Ok(i) = facts.index_of(aid) else {
        return false;
    };
    if rule.residue != "*" && rule.residue != facts.residue[i].as_str() {
        return false;
    }
    if rule
        .atomic_number
        .is_some_and(|z| z != facts.atomic_number[i])
    {
        return false;
    }
    if rule.degree.is_some_and(|degree| degree != facts.degree[i]) {
        return false;
    }
    if rule
        .hydrogen_count
        .is_some_and(|h_count| h_count != facts.hydrogen_count[i])
    {
        return false;
    }
    if rule.ewd_count.is_some_and(|n| {
        facts
            .ewd_count_around_attachment(aid)
            .is_none_or(|actual| actual != n)
    }) {
        return false;
    }
    if let Some(prop) = rule.atom_property
        && !facts.atom_property_matches(aid, None, &prop)
    {
        return false;
    }
    if let Some(env) = rule.environment
        && !facts.environment_matches(aid, env, rule.environment_bonds)
    {
        return false;
    }
    true
}

#[derive(Debug, Clone)]
struct BCCMolFacts {
    index: HashMap<AtomId, usize>,
    atomic_number: Vec<u8>,
    degree: Vec<usize>,
    hydrogen_count: Vec<usize>,
    ewd: Vec<bool>,
    residue: Vec<String>,
    props: Vec<AtomPropertyFacts>,
    neighbors: Vec<Vec<(AtomId, i32, BondId)>>,
}

impl BCCMolFacts {
    fn new(mol: &Atomistic) -> Result<Self, String> {
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
                format!("BCC atom typing requires `{}` for {aid:?}", keys::ELEMENT)
            })?;
            let element = Element::by_symbol(symbol)
                .ok_or_else(|| format!("unsupported element symbol `{symbol}` for {aid:?}"))?;
            atomic_number.push(element.z());
            residue.push(atom.get_str(keys::RES_NAME).unwrap_or("*").to_owned());
        }

        let mut neighbors = vec![Vec::new(); atom_ids.len()];
        for (bid, bond) in mol.bonds() {
            let bond_type = bcc_bond_type(&bond)?;
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

    fn index_of(&self, aid: AtomId) -> Result<usize, String> {
        self.index
            .get(&aid)
            .copied()
            .ok_or_else(|| format!("unknown atom id {aid:?}"))
    }

    fn ewd_count_around_attachment(&self, aid: AtomId) -> Option<usize> {
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

    /// Walk a pre-parsed `[...]` expression: an AND of ORs.
    fn atom_property_matches(&self, aid: AtomId, prev: Option<AtomId>, expr: &PropExpr) -> bool {
        expr.constraints.iter().all(|constraint| {
            constraint
                .units
                .iter()
                .any(|unit| self.atom_property_unit_matches(aid, prev, unit))
        })
    }

    fn atom_property_unit_matches(
        &self,
        aid: AtomId,
        prev: Option<AtomId>,
        unit: &PropUnit,
    ) -> bool {
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        let count = self.props[i].count(unit.prop);
        let count_matches = unit.count.map_or(count > 0, |n| count == n);
        if !count_matches {
            return false;
        }
        match unit.relation {
            Some(PropRelation::BondedToPrev) => {
                prev.is_some_and(|prev| self.bond_to_prev_matches(aid, prev, unit.prop))
            }
            Some(PropRelation::NotBondedToPrev) => {
                prev.is_some_and(|prev| !self.bond_to_prev_matches(aid, prev, unit.prop))
            }
            None => true,
        }
    }

    fn bond_to_prev_matches(&self, aid: AtomId, prev: AtomId, prop: AtomProp) -> bool {
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        self.neighbors[i]
            .iter()
            .find(|(nb, _, _)| *nb == prev)
            .is_some_and(|(_, bond_type, _)| bond_type_matches_property(*bond_type, prop))
    }

    fn environment_matches(
        &self,
        aid: AtomId,
        patterns: &'static [AtomPattern],
        env_bonds: Option<&'static [EnvBond]>,
    ) -> bool {
        let mut labels = HashMap::new();
        self.match_pattern_list(aid, None, patterns, &mut labels)
            && env_bonds.is_none_or(|bonds| self.environment_bonds_match(bonds, &labels))
    }

    fn environment_bonds_match(
        &self,
        bonds: &[EnvBond],
        labels: &HashMap<&'static str, AtomId>,
    ) -> bool {
        for constraint in bonds {
            let Some(&a) = labels.get(constraint.a) else {
                return false;
            };
            let Some(&b) = labels.get(constraint.b) else {
                return false;
            };
            let Ok(i) = self.index_of(a) else {
                return false;
            };
            let Some((_, bond_type, _)) = self.neighbors[i].iter().find(|(nb, _, _)| *nb == b)
            else {
                return false;
            };
            if !environment_bond_type_matches(*bond_type, constraint.bond) {
                return false;
            }
        }
        true
    }

    fn match_pattern_list(
        &self,
        parent: AtomId,
        prev: Option<AtomId>,
        patterns: &'static [AtomPattern],
        labels: &mut HashMap<&'static str, AtomId>,
    ) -> bool {
        let mut used = Vec::new();
        self.match_pattern_list_rec(parent, prev, patterns, 0, labels, &mut used)
    }

    fn match_pattern_list_rec(
        &self,
        parent: AtomId,
        prev: Option<AtomId>,
        patterns: &'static [AtomPattern],
        pos: usize,
        labels: &mut HashMap<&'static str, AtomId>,
        used: &mut Vec<AtomId>,
    ) -> bool {
        if pos == patterns.len() {
            return true;
        }
        let Ok(parent_i) = self.index_of(parent) else {
            return false;
        };
        for (candidate, _, _) in &self.neighbors[parent_i] {
            if Some(*candidate) == prev || used.contains(candidate) {
                continue;
            }
            let mut labels_next = labels.clone();
            if self.atom_pattern_matches(*candidate, parent, &patterns[pos], &mut labels_next) {
                used.push(*candidate);
                if self.match_pattern_list_rec(
                    parent,
                    prev,
                    patterns,
                    pos + 1,
                    &mut labels_next,
                    used,
                ) {
                    *labels = labels_next;
                    return true;
                }
                used.pop();
            }
        }
        false
    }

    fn atom_pattern_matches(
        &self,
        aid: AtomId,
        prev: AtomId,
        pattern: &'static AtomPattern,
        labels: &mut HashMap<&'static str, AtomId>,
    ) -> bool {
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        if !self.pattern_atom_matches(i, pattern.atom) {
            return false;
        }
        if pattern
            .degree
            .is_some_and(|degree| degree != self.degree[i])
        {
            return false;
        }
        if let Some(prop) = pattern.property
            && !self.atom_property_matches(aid, Some(prev), &prop)
        {
            return false;
        }
        if let Some(label) = pattern.label
            && labels.insert(label, aid).is_some_and(|old| old != aid)
        {
            return false;
        }
        self.match_pattern_list(aid, Some(prev), pattern.children, labels)
    }

    /// Match a pattern atom whose name the generator already resolved.
    ///
    /// `EW` / `WILDATOM` / element resolution happened at table-generation time,
    /// so there is no name table to consult here.
    fn pattern_atom_matches(&self, atom_index: usize, atom: PatternAtom) -> bool {
        match atom {
            PatternAtom::ElectronWithdrawing => self.ewd[atom_index],
            PatternAtom::Wild(specs) => specs.iter().any(|spec| {
                spec.z == self.atomic_number[atom_index]
                    && spec
                        .degree
                        .is_none_or(|degree| degree == self.degree[atom_index])
            }),
            PatternAtom::Element(z) => z == self.atomic_number[atom_index],
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AtomPropertyFacts {
    rg: [usize; 12],
    nr: usize,
    ar1: usize,
    ar2: usize,
    ar3: usize,
    ar4: usize,
    ar5: usize,
    sb: usize,
    sb_strict: usize,
    db: usize,
    db_strict: usize,
    tb: usize,
    tb_strict: usize,
    ab: usize,
    dl: usize,
}

impl AtomPropertyFacts {
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

    fn count(&self, prop: AtomProp) -> usize {
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

/// Does a BCC bond type satisfy a bond-valued [`AtomProp`]?
///
/// Ring and aromaticity properties (`RG*`, `AR*`, `NR`) are atom-valued, not
/// bond-valued, so they never constrain a bond.
fn bond_type_matches_property(bond_type: i32, prop: AtomProp) -> bool {
    match prop {
        AtomProp::SbStrict => matches!(bond_type, 1 | 9),
        AtomProp::SbAny => matches!(bond_type, 1 | 7 | 9 | 10),
        AtomProp::DbStrict => matches!(bond_type, 2 | 9),
        AtomProp::DbAny => matches!(bond_type, 2 | 8 | 9 | 10),
        AtomProp::TbStrict | AtomProp::TbAny => bond_type == 3,
        AtomProp::Dl => bond_type == 9,
        AtomProp::Ab => matches!(bond_type, 7 | 8 | 10),
        AtomProp::Rg
        | AtomProp::Rg3
        | AtomProp::Rg4
        | AtomProp::Rg5
        | AtomProp::Rg6
        | AtomProp::Rg7
        | AtomProp::Rg8
        | AtomProp::Rg9
        | AtomProp::Rg10
        | AtomProp::Nr
        | AtomProp::Ar1
        | AtomProp::Ar2
        | AtomProp::Ar3
        | AtomProp::Ar4
        | AtomProp::Ar5 => false,
    }
}

/// Does a BCC bond type satisfy an `a:b:TYPE` environment-bond constraint?
fn environment_bond_type_matches(bond_type: i32, bond: EnvBondType) -> bool {
    match bond {
        EnvBondType::Any => true,
        EnvBondType::Single => bond_type_matches_property(bond_type, AtomProp::SbAny),
        EnvBondType::Double => bond_type_matches_property(bond_type, AtomProp::DbAny),
        EnvBondType::Triple => bond_type_matches_property(bond_type, AtomProp::TbAny),
        EnvBondType::Aromatic => bond_type_matches_property(bond_type, AtomProp::Ab),
    }
}

/// Oriented BCC correction table.
///
/// A row `(left, right, bond_type, delta)` means a bond typed
/// `left|right|bond_type` adds `+delta` to the left atom and `-delta` to the
/// right atom. A reversed bond applies the same magnitude with reversed sign.
#[derive(Debug, Clone, Default)]
pub struct BCCCorrectionTable {
    corrections: HashMap<(String, String, i32), f64>,
    aliases: HashMap<String, String>,
}

impl BCCCorrectionTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the table for a parameter set from its compile-time rows.
    ///
    /// Fallible only to keep the call shape of the former text-parsing
    /// constructor; the tables are typed Rust data, so this cannot fail.
    pub fn parameter_set(model: BccParameterSet) -> Result<Self, String> {
        let mut table = Self::new();
        for row in model.corrections() {
            table = table.insert_bond_type(row.left, row.right, row.bond_type, row.delta);
        }
        for alias in model.aliases() {
            table = table.alias(alias.atom_type, alias.reference);
        }
        Ok(table)
    }

    pub fn bcc() -> Result<Self, String> {
        Self::parameter_set(BccParameterSet::Bcc)
    }

    pub fn abcg2() -> Result<Self, String> {
        Self::parameter_set(BccParameterSet::Abcg2)
    }

    pub fn insert(self, left: impl Into<String>, right: impl Into<String>, delta: f64) -> Self {
        self.insert_bond_type(left, right, 1, delta)
    }

    pub fn insert_bond_type(
        mut self,
        left: impl Into<String>,
        right: impl Into<String>,
        bond_type: i32,
        delta: f64,
    ) -> Self {
        self.corrections
            .insert((left.into(), right.into(), bond_type), delta);
        self
    }

    pub fn alias(mut self, atom_type: impl Into<String>, reference: impl Into<String>) -> Self {
        self.aliases.insert(atom_type.into(), reference.into());
        self
    }

    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    pub fn alias_len(&self) -> usize {
        self.aliases.len()
    }

    pub fn correction(&self, left: &str, right: &str, bond_type: i32) -> Option<f64> {
        if let Some(v) = self.direct_correction(left, right, bond_type) {
            return Some(v);
        }

        let left_alias = self.aliases.get(left).map(String::as_str);
        let right_alias = self.aliases.get(right).map(String::as_str);

        if let Some(alias) = left_alias
            && let Some(v) = self.direct_correction(alias, right, bond_type)
        {
            return Some(v);
        }
        if let Some(alias) = right_alias
            && let Some(v) = self.direct_correction(left, alias, bond_type)
        {
            return Some(v);
        }
        if let (Some(left_alias), Some(right_alias)) = (left_alias, right_alias)
            && let Some(v) = self.direct_correction(left_alias, right_alias, bond_type)
        {
            return Some(v);
        }
        None
    }

    fn direct_correction(&self, left: &str, right: &str, bond_type: i32) -> Option<f64> {
        self.corrections
            .get(&(left.to_owned(), right.to_owned(), bond_type))
            .copied()
            .or_else(|| {
                self.corrections
                    .get(&(right.to_owned(), left.to_owned(), bond_type))
                    .map(|v| -*v)
            })
    }
}

#[derive(Debug, Clone)]
pub struct BCCCorrector {
    table: BCCCorrectionTable,
}

impl BCCCorrector {
    pub fn new(table: BCCCorrectionTable) -> Self {
        Self { table }
    }

    pub fn apply(&self, mol: &mut Atomistic) -> Result<(), String> {
        let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
        let mut index = HashMap::new();
        for (i, aid) in atom_ids.iter().copied().enumerate() {
            index.insert(aid, i);
        }

        let mut delta = vec![0.0; atom_ids.len()];
        let bond_rows: Vec<_> = mol.bonds().collect();
        for (bid, bond) in bond_rows {
            let a = bond.nodes[0];
            let b = bond.nodes[1];
            let bond_type = bcc_bond_type(&bond)?;
            let a_label = mol
                .get_atom(a)
                .map_err(|e| e.to_string())?
                .get_str(keys::TYPE)
                .ok_or_else(|| "BCCCorrector requires atom `type` labels".to_owned())?
                .to_owned();
            let b_label = mol
                .get_atom(b)
                .map_err(|e| e.to_string())?
                .get_str(keys::TYPE)
                .ok_or_else(|| "BCCCorrector requires atom `type` labels".to_owned())?
                .to_owned();
            let Some(dq) = self.table.correction(&a_label, &b_label, bond_type) else {
                return Err(format!(
                    "missing BCC correction for bond {:?}: {a_label}|{b_label}|{bond_type}",
                    bid
                ));
            };
            delta[index[&a]] += dq;
            delta[index[&b]] -= dq;
        }

        for (i, aid) in atom_ids.iter().copied().enumerate() {
            let q0 = mol
                .get_atom(aid)
                .map_err(|e| e.to_string())?
                .get_f64(keys::CHARGE)
                .ok_or_else(|| "BCCCorrector requires AM1 base `charge`".to_owned())?;
            mol.set_atom(aid, keys::CHARGE, q0 + delta[i])
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

impl Default for BCCCorrector {
    fn default() -> Self {
        Self::new(BCCCorrectionTable::new())
    }
}

/// AM1-BCC typifier: AM1 backend + BCC atom typifier + correction table.
#[derive(Debug, Clone)]
pub struct AM1BCCTypifier<B = UnavailableAM1Backend> {
    backend: B,
    atom_typifier: BCCAtomTypifier,
    corrector: BCCCorrector,
    total_charge: Option<f64>,
}

impl Default for AM1BCCTypifier<UnavailableAM1Backend> {
    fn default() -> Self {
        Self::new(UnavailableAM1Backend)
    }
}

impl<B> AM1BCCTypifier<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            atom_typifier: BCCAtomTypifier::bcc(),
            corrector: BCCCorrector::default(),
            total_charge: None,
        }
    }

    pub fn bcc(backend: B) -> Result<Self, String> {
        Ok(Self::new(backend)
            .with_atom_typifier(BCCAtomTypifier::bcc())
            .with_correction_table(BCCCorrectionTable::bcc()?))
    }

    pub fn abcg2(backend: B) -> Result<Self, String> {
        Ok(Self::new(backend)
            .with_atom_typifier(BCCAtomTypifier::abcg2())
            .with_correction_table(BCCCorrectionTable::abcg2()?))
    }

    pub fn with_atom_typifier(mut self, atom_typifier: BCCAtomTypifier) -> Self {
        self.atom_typifier = atom_typifier;
        self
    }

    pub fn with_correction_table(mut self, table: BCCCorrectionTable) -> Self {
        self.corrector = BCCCorrector::new(table);
        self
    }

    pub fn with_corrector(mut self, corrector: BCCCorrector) -> Self {
        self.corrector = corrector;
        self
    }

    pub fn with_total_charge(mut self, total_charge: f64) -> Self {
        self.total_charge = Some(total_charge);
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B> AM1BCCTypifier<B>
where
    B: AM1ChargeBackend,
{
    fn typify_am1bcc(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let am1 = self.backend.compute_am1_charges(mol)?;
        if am1.charges.len() != mol.n_atoms() {
            return Err(format!(
                "AM1 backend returned {} charges for {} atoms",
                am1.charges.len(),
                mol.n_atoms()
            ));
        }

        let mut out = self.atom_typifier.typify(mol)?;
        let atom_ids: Vec<AtomId> = out.atoms().map(|(id, _)| id).collect();
        for (i, aid) in atom_ids.iter().copied().enumerate() {
            out.set_atom(aid, keys::CHARGE, am1.charges[i])
                .map_err(|e| e.to_string())?;
        }

        self.corrector.apply(&mut out)?;
        if let Some(target) = self.total_charge.or(am1.total_charge) {
            normalize_total_charge(&mut out, target)?;
        }
        Ok(out)
    }
}

impl<B> Typifier for AM1BCCTypifier<B>
where
    B: AM1ChargeBackend,
{
    type Mol = Atomistic;

    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        self.typify_am1bcc(mol)
    }
}

fn normalize_total_charge(mol: &mut Atomistic, target: f64) -> Result<(), String> {
    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    if atom_ids.is_empty() {
        return Ok(());
    }
    let mut sum = 0.0;
    for aid in &atom_ids {
        sum += mol
            .get_atom(*aid)
            .map_err(|e| e.to_string())?
            .get_f64(keys::CHARGE)
            .ok_or_else(|| "cannot normalize missing charge".to_owned())?;
    }
    let shift = (target - sum) / atom_ids.len() as f64;
    for aid in atom_ids {
        let q = mol
            .get_atom(aid)
            .map_err(|e| e.to_string())?
            .get_f64(keys::CHARGE)
            .ok_or_else(|| "cannot normalize missing charge".to_owned())?;
        mol.set_atom(aid, keys::CHARGE, q + shift)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

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

fn prop_truthy(prop: Option<&PropValue>) -> bool {
    match prop {
        Some(PropValue::Bool(v)) => *v,
        Some(PropValue::Int(v)) => *v != 0,
        Some(PropValue::F64(v)) => *v != 0.0,
        Some(PropValue::Str(v)) => matches!(v.as_str(), "1" | "true" | "True" | "TRUE" | "ar"),
        None => false,
    }
}

fn bcc_bond_type(bond: &Bond) -> Result<i32, String> {
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
