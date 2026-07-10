//! AM1-BCC charge typifiers.
//!
//! The public shape is intentionally parallel to `MMFFTypifier` and
//! `OPLSAATypifier`: a struct owns the model/backend state, `typify()` returns
//! a labeled `Atomistic`, and `charge` is written as a canonical atom property.
//!
//! The implementation is a native port surface: external programs are reference
//! material only, not runtime dependencies. Missing AM1 backends, BCC atom
//! types, bond types, or correction rows return errors.

use molrs::store::keys;
use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic, Bond, BondId, Element, find_rings};
use std::collections::HashMap;

use super::Typifier;

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
    fn embedded_table(self) -> &'static str {
        match self {
            Self::Bcc => molrs::data::ANTECHAMBER_BCCPARM_DAT,
            Self::Abcg2 => molrs::data::ANTECHAMBER_BCCPARM_ABCG2_DAT,
        }
    }

    fn embedded_atom_type_def(self) -> &'static str {
        match self {
            Self::Bcc => molrs::data::ANTECHAMBER_ATOMTYPE_BCC_DEF,
            Self::Abcg2 => molrs::data::ANTECHAMBER_ATOMTYPE_ABCG2_DEF,
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

    pub fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let rules = BCCAtomTypeRules::parse_str(self.model.embedded_atom_type_def())?;
        let mut out = mol.clone();

        let bond_ids: Vec<_> = out.bonds().map(|(bid, _)| bid).collect();
        for bid in bond_ids {
            let bond = out.get_bond(bid).map_err(|e| e.to_string())?;
            if bond.props.contains_key(keys::TYPE) {
                bcc_bond_type(&bond).map_err(|e| format!("{e} for BCC bond type on {bid:?}"))?;
                continue;
            }
            let bond_type = infer_bcc_bond_type(&bond)
                .map_err(|e| format!("{e} while assigning BCC bond type on {bid:?}"))?;
            out.set_bond_prop(bid, keys::TYPE, bond_type)
                .map_err(|e| e.to_string())?;
        }

        let facts = BCCMolFacts::new(&out)?;
        let atom_ids: Vec<AtomId> = out.atoms().map(|(aid, _)| aid).collect();
        for aid in atom_ids {
            let atom_type = rules
                .assign_type(aid, &facts)
                .ok_or_else(|| format!("missing BCC atom type for {aid:?}"))?;
            out.set_atom(aid, keys::TYPE, atom_type)
                .map_err(|e| e.to_string())?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct BCCAtomTypeRules {
    wildatoms: HashMap<String, Vec<WildAtomSpec>>,
    rules: Vec<BCCAtomRule>,
}

impl BCCAtomTypeRules {
    fn parse_str(input: &str) -> Result<Self, String> {
        let mut wildatoms: HashMap<String, Vec<WildAtomSpec>> = HashMap::new();
        let mut rules = Vec::new();
        for (lineno, raw) in input.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with("//")
                || line.starts_with('-')
                || line.starts_with('=')
            {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }
            if fields[0] == "WILDATOM" {
                if fields.len() < 3 {
                    return Err(format!("invalid WILDATOM row at line {}", lineno + 1));
                }
                let specs = fields[2..]
                    .iter()
                    .map(|s| WildAtomSpec::parse(s))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("{e} at line {}", lineno + 1))?;
                wildatoms.insert(fields[1].to_owned(), specs);
                continue;
            }
            if fields[0] != "ATD" {
                continue;
            }
            let Some(rule) = BCCAtomRule::parse(&fields, lineno + 1)? else {
                continue;
            };
            rules.push(rule);
        }
        Ok(Self { wildatoms, rules })
    }

    fn assign_type(&self, aid: AtomId, facts: &BCCMolFacts) -> Option<&str> {
        self.rules
            .iter()
            .find(|rule| rule.matches(aid, facts, self))
            .map(|rule| rule.atom_type.as_str())
    }
}

#[derive(Debug, Clone)]
struct WildAtomSpec {
    z: u8,
    degree: Option<usize>,
}

impl WildAtomSpec {
    fn parse(input: &str) -> Result<Self, String> {
        let split = input
            .char_indices()
            .find_map(|(i, c)| c.is_ascii_digit().then_some(i))
            .unwrap_or(input.len());
        let symbol = &input[..split];
        let degree = if split < input.len() {
            Some(
                input[split..]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid WILDATOM degree `{input}`: {e}"))?,
            )
        } else {
            None
        };
        let element = Element::by_symbol(symbol)
            .ok_or_else(|| format!("invalid WILDATOM element symbol `{symbol}`"))?;
        Ok(Self {
            z: element.z(),
            degree,
        })
    }
}

#[derive(Debug, Clone)]
struct BCCAtomRule {
    atom_type: String,
    residue: String,
    atomic_number: Option<u8>,
    degree: Option<usize>,
    hydrogen_count: Option<usize>,
    ewd_count: Option<usize>,
    atom_property: Option<String>,
    environment: Option<String>,
    environment_bonds: Option<String>,
}

impl BCCAtomRule {
    fn parse(fields: &[&str], lineno: usize) -> Result<Option<Self>, String> {
        if fields.len() < 3 {
            return Err(format!("invalid ATD row at line {lineno}"));
        }
        let atom_type = fields[1];
        if atom_type == "DU" {
            return Ok(None);
        }
        let residue = fields.get(2).copied().unwrap_or("&");
        if residue == "&" {
            return Ok(None);
        }
        let atomic_number = parse_optional_u8(fields.get(3).copied(), lineno, "atomic number")?;
        let degree = parse_optional_usize(fields.get(4).copied(), lineno, "degree")?;
        let hydrogen_count =
            parse_optional_usize(fields.get(5).copied(), lineno, "hydrogen count")?;
        let ewd_count = parse_optional_usize(fields.get(6).copied(), lineno, "EW count")?;
        let atom_property = parse_optional_str(fields.get(7).copied());
        let environment = parse_optional_str(fields.get(8).copied());
        let environment_bonds = parse_optional_str(fields.get(9).copied());
        Ok(Some(Self {
            atom_type: atom_type.to_owned(),
            residue: residue.to_owned(),
            atomic_number,
            degree,
            hydrogen_count,
            ewd_count,
            atom_property,
            environment,
            environment_bonds,
        }))
    }

    fn matches(&self, aid: AtomId, facts: &BCCMolFacts, rules: &BCCAtomTypeRules) -> bool {
        let Ok(i) = facts.index_of(aid) else {
            return false;
        };
        if self.residue != "*" && self.residue.as_str() != facts.residue[i].as_str() {
            return false;
        }
        if self
            .atomic_number
            .is_some_and(|z| z != facts.atomic_number[i])
        {
            return false;
        }
        if self.degree.is_some_and(|degree| degree != facts.degree[i]) {
            return false;
        }
        if self
            .hydrogen_count
            .is_some_and(|h_count| h_count != facts.hydrogen_count[i])
        {
            return false;
        }
        if self.ewd_count.is_some_and(|n| {
            facts
                .ewd_count_around_attachment(aid)
                .is_none_or(|actual| actual != n)
        }) {
            return false;
        }
        if let Some(prop) = &self.atom_property
            && !facts.atom_property_matches(aid, None, prop)
        {
            return false;
        }
        if let Some(env) = &self.environment
            && !facts.environment_matches(aid, env, self.environment_bonds.as_deref(), rules)
        {
            return false;
        }
        true
    }
}

fn parse_optional_str(field: Option<&str>) -> Option<String> {
    match field {
        Some("*" | "&") | None => None,
        Some(v) => Some(v.to_owned()),
    }
}

fn parse_optional_usize(
    field: Option<&str>,
    lineno: usize,
    label: &str,
) -> Result<Option<usize>, String> {
    match field {
        Some("*" | "&") | None => Ok(None),
        Some(v) => v
            .parse::<usize>()
            .map(Some)
            .map_err(|e| format!("invalid ATD {label} `{v}` at line {lineno}: {e}")),
    }
}

fn parse_optional_u8(
    field: Option<&str>,
    lineno: usize,
    label: &str,
) -> Result<Option<u8>, String> {
    match field {
        Some("*" | "&") | None => Ok(None),
        Some(v) => v
            .parse::<u8>()
            .map(Some)
            .map_err(|e| format!("invalid ATD {label} `{v}` at line {lineno}: {e}")),
    }
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

    fn atom_property_matches(&self, aid: AtomId, prev: Option<AtomId>, expr: &str) -> bool {
        let Some(inner) = expr.strip_prefix('[').and_then(|s| s.strip_suffix(']')) else {
            return false;
        };
        inner.split(',').all(|constraint| {
            constraint
                .split('.')
                .any(|unit| self.atom_property_unit_matches(aid, prev, unit))
        })
    }

    fn atom_property_unit_matches(&self, aid: AtomId, prev: Option<AtomId>, unit: &str) -> bool {
        let digit_end = unit
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_digit())
            .map(|(i, c)| i + c.len_utf8())
            .last()
            .unwrap_or(0);
        let required_count = if digit_end == 0 {
            None
        } else {
            unit[..digit_end].parse::<usize>().ok()
        };
        let mut prop = &unit[digit_end..];
        let relation = if let Some(base) = prop.strip_suffix("''") {
            prop = base;
            Some(false)
        } else if let Some(base) = prop.strip_suffix('\'') {
            prop = base;
            Some(true)
        } else {
            None
        };
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        let count = self.props[i].count(prop);
        let count_matches = required_count.map_or(count > 0, |n| count == n);
        if !count_matches {
            return false;
        }
        match relation {
            Some(true) => prev.is_some_and(|prev| self.bond_to_prev_matches(aid, prev, prop)),
            Some(false) => prev.is_some_and(|prev| !self.bond_to_prev_matches(aid, prev, prop)),
            None => true,
        }
    }

    fn bond_to_prev_matches(&self, aid: AtomId, prev: AtomId, prop: &str) -> bool {
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
        env: &str,
        env_bonds: Option<&str>,
        rules: &BCCAtomTypeRules,
    ) -> bool {
        let Ok(patterns) = parse_environment(env) else {
            return false;
        };
        let mut labels = HashMap::new();
        self.match_pattern_list(aid, None, &patterns, rules, &mut labels)
            && env_bonds.is_none_or(|spec| self.environment_bonds_match(spec, &labels))
    }

    fn environment_bonds_match(&self, spec: &str, labels: &HashMap<String, AtomId>) -> bool {
        for raw in spec.split(',') {
            let parts: Vec<_> = raw.split(':').collect();
            if parts.len() != 3 {
                return false;
            }
            let Some(&a) = labels.get(parts[0]) else {
                return false;
            };
            let Some(&b) = labels.get(parts[1]) else {
                return false;
            };
            let Ok(i) = self.index_of(a) else {
                return false;
            };
            let Some((_, bond_type, _)) = self.neighbors[i].iter().find(|(nb, _, _)| *nb == b)
            else {
                return false;
            };
            if !environment_bond_type_matches(*bond_type, parts[2]) {
                return false;
            }
        }
        true
    }

    fn match_pattern_list(
        &self,
        parent: AtomId,
        prev: Option<AtomId>,
        patterns: &[AtomPattern],
        rules: &BCCAtomTypeRules,
        labels: &mut HashMap<String, AtomId>,
    ) -> bool {
        let mut used = Vec::new();
        self.match_pattern_list_rec(parent, prev, patterns, 0, rules, labels, &mut used)
    }

    #[allow(clippy::too_many_arguments)] // private recursive matcher (same precedent as smarts/matcher.rs)
    fn match_pattern_list_rec(
        &self,
        parent: AtomId,
        prev: Option<AtomId>,
        patterns: &[AtomPattern],
        pos: usize,
        rules: &BCCAtomTypeRules,
        labels: &mut HashMap<String, AtomId>,
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
            if self.atom_pattern_matches(
                *candidate,
                parent,
                &patterns[pos],
                rules,
                &mut labels_next,
            ) {
                used.push(*candidate);
                if self.match_pattern_list_rec(
                    parent,
                    prev,
                    patterns,
                    pos + 1,
                    rules,
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
        pattern: &AtomPattern,
        rules: &BCCAtomTypeRules,
        labels: &mut HashMap<String, AtomId>,
    ) -> bool {
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        if !self.pattern_atom_name_matches(i, &pattern.name, rules) {
            return false;
        }
        if pattern
            .degree
            .is_some_and(|degree| degree != self.degree[i])
        {
            return false;
        }
        if let Some(prop) = &pattern.property
            && !self.atom_property_matches(aid, Some(prev), prop)
        {
            return false;
        }
        if let Some(label) = &pattern.label
            && labels
                .insert(label.clone(), aid)
                .is_some_and(|old| old != aid)
        {
            return false;
        }
        self.match_pattern_list(aid, Some(prev), &pattern.children, rules, labels)
    }

    fn pattern_atom_name_matches(
        &self,
        atom_index: usize,
        name: &str,
        rules: &BCCAtomTypeRules,
    ) -> bool {
        if name == "EW" {
            return self.ewd[atom_index];
        }
        if let Some(specs) = rules.wildatoms.get(name) {
            return specs.iter().any(|spec| {
                spec.z == self.atomic_number[atom_index]
                    && spec
                        .degree
                        .is_none_or(|degree| degree == self.degree[atom_index])
            });
        }
        Element::by_symbol(name)
            .is_some_and(|element| element.z() == self.atomic_number[atom_index])
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

    fn count(&self, prop: &str) -> usize {
        match prop {
            "RG" => self.rg[0],
            "RG3" => self.rg[3],
            "RG4" => self.rg[4],
            "RG5" => self.rg[5],
            "RG6" => self.rg[6],
            "RG7" => self.rg[7],
            "RG8" => self.rg[8],
            "RG9" => self.rg[9],
            "RG10" => self.rg[10],
            "NR" => self.nr,
            "AR1" => self.ar1,
            "AR2" => self.ar2,
            "AR3" => self.ar3,
            "AR4" => self.ar4,
            "AR5" => self.ar5,
            "SB" => self.sb_strict,
            "sb" => self.sb,
            "DB" => self.db_strict,
            "db" => self.db,
            "TB" => self.tb_strict,
            "tb" => self.tb,
            "AB" => self.ab,
            "DL" => self.dl,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
struct AtomPattern {
    name: String,
    degree: Option<usize>,
    property: Option<String>,
    label: Option<String>,
    children: Vec<AtomPattern>,
}

fn parse_environment(input: &str) -> Result<Vec<AtomPattern>, String> {
    let inner = input
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| format!("invalid BCC environment `{input}`"))?;
    parse_pattern_list(inner)
}

fn parse_pattern_list(input: &str) -> Result<Vec<AtomPattern>, String> {
    let mut out = Vec::new();
    for part in split_top_level(input, ',') {
        if part.is_empty() {
            continue;
        }
        let (pattern, pos) = parse_atom_pattern(part, 0)?;
        if pos != part.len() {
            return Err(format!("unparsed BCC environment fragment `{part}`"));
        }
        out.push(pattern);
    }
    Ok(out)
}

fn parse_atom_pattern(input: &str, mut pos: usize) -> Result<(AtomPattern, usize), String> {
    let name_start = pos;
    while pos < input.len() && input.as_bytes()[pos].is_ascii_alphabetic() {
        pos += 1;
    }
    if pos == name_start {
        return Err(format!("expected atom name in `{input}`"));
    }
    let name = input[name_start..pos].to_owned();
    let degree_start = pos;
    while pos < input.len() && input.as_bytes()[pos].is_ascii_digit() {
        pos += 1;
    }
    let degree = if pos > degree_start {
        Some(
            input[degree_start..pos]
                .parse::<usize>()
                .map_err(|e| format!("invalid atom-pattern degree in `{input}`: {e}"))?,
        )
    } else {
        None
    };
    let mut property = None;
    let mut label = None;
    let mut children = Vec::new();
    loop {
        if pos >= input.len() {
            break;
        }
        let ch = input.as_bytes()[pos] as char;
        match ch {
            '[' => {
                let end = find_matching(input, pos, '[', ']')?;
                property = Some(input[pos..=end].to_owned());
                pos = end + 1;
            }
            '<' => {
                let end = find_matching(input, pos, '<', '>')?;
                label = Some(input[pos + 1..end].to_owned());
                pos = end + 1;
            }
            '(' => {
                let end = find_matching(input, pos, '(', ')')?;
                children = parse_pattern_list(&input[pos + 1..end])?;
                pos = end + 1;
            }
            _ => break,
        }
    }
    Ok((
        AtomPattern {
            name,
            degree,
            property,
            label,
            children,
        },
        pos,
    ))
}

fn split_top_level(input: &str, sep: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut depth = 0_i32;
    for (i, ch) in input.char_indices() {
        match ch {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            _ if ch == sep && depth == 0 => {
                out.push(input[start..i].trim());
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }
    out.push(input[start..].trim());
    out
}

fn find_matching(input: &str, start: usize, open: char, close: char) -> Result<usize, String> {
    let mut depth = 0_i32;
    for (offset, ch) in input[start..].char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Ok(start + offset);
            }
        }
    }
    Err(format!("unmatched `{open}` in `{input}`"))
}

fn bond_type_matches_property(bond_type: i32, prop: &str) -> bool {
    match prop {
        "SB" => matches!(bond_type, 1 | 9),
        "sb" => matches!(bond_type, 1 | 7 | 9 | 10),
        "DB" => matches!(bond_type, 2 | 9),
        "db" => matches!(bond_type, 2 | 8 | 9 | 10),
        "TB" | "tb" => bond_type == 3,
        "DL" => bond_type == 9,
        "AB" => matches!(bond_type, 7 | 8 | 10),
        _ => false,
    }
}

fn environment_bond_type_matches(bond_type: i32, prop: &str) -> bool {
    match prop {
        "any" | "ANY" | "Any" => true,
        "SB" | "sb" => bond_type_matches_property(bond_type, "sb"),
        "DB" | "db" => bond_type_matches_property(bond_type, "db"),
        "TB" | "tb" => bond_type_matches_property(bond_type, "tb"),
        "AB" | "ab" => bond_type_matches_property(bond_type, "AB"),
        _ => false,
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

    /// Parse a `BCCPARM*.DAT` table.
    ///
    /// `CORR` rows become explicit aliases; ordinary rows become corrections.
    pub fn parse_str(input: &str) -> Result<Self, String> {
        let mut table = Self::new();
        for (lineno, raw) in input.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields[0].eq_ignore_ascii_case("CORR") {
                if fields.len() < 3 {
                    return Err(format!(
                        "invalid BCC CORR row at line {}: {raw}",
                        lineno + 1
                    ));
                }
                table = table.alias(fields[1], fields[2]);
                continue;
            }
            if fields.len() < 5 {
                return Err(format!(
                    "invalid BCC parameter row at line {}: {raw}",
                    lineno + 1
                ));
            }
            let bond_type = fields[3]
                .parse::<i32>()
                .map_err(|e| format!("invalid BCC bond type at line {}: {e}", lineno + 1))?;
            let delta = fields[4]
                .parse::<f64>()
                .map_err(|e| format!("invalid BCC delta at line {}: {e}", lineno + 1))?;
            table = table.insert_bond_type(fields[1], fields[2], bond_type, delta);
        }
        Ok(table)
    }

    pub fn parameter_set(model: BccParameterSet) -> Result<Self, String> {
        Self::parse_str(model.embedded_table())
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

fn infer_bcc_bond_type(bond: &Bond) -> Result<i32, String> {
    if is_aromatic_bond(bond) {
        return Ok(10);
    }
    let order = bond
        .props
        .get(keys::ORDER)
        .and_then(PropValue::as_f64)
        .ok_or_else(|| "BCC bond typification requires numeric bond `order`".to_owned())?;
    if (order - 1.0).abs() < 1.0e-6 {
        Ok(1)
    } else if (order - 2.0).abs() < 1.0e-6 {
        Ok(2)
    } else if (order - 3.0).abs() < 1.0e-6 {
        Ok(3)
    } else {
        Err(format!(
            "cannot infer BCC bond type from bond order {order}"
        ))
    }
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
